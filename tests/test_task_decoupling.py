"""CPU tests for task-region counterfactual feature extraction."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from resilient.task_decoupling.audit import (
    binary_mask_metrics,
    effective_rank,
    ridge_probe,
)
from resilient.task_decoupling.config import validate_task_decoupling_config
from resilient.task_decoupling.counterfactual import build_local_blur_counterfactual
from resilient.task_decoupling.entities import load_entity_specification
from resilient.task_decoupling.gate4 import evaluate_gate4, summarize_gate4_rows
from resilient.task_decoupling.pipeline import TaskRegionDisentangler
from resilient.task_decoupling.prompts import boxes_from_binary_masks
from resilient.task_decoupling.providers.file_service import FileServiceMaskProvider
from resilient.task_decoupling.providers.oracle import ArrayOracleMaskProvider
from resilient.task_decoupling.types import BoxPromptBatch, EntitySpec, MaskBatch


class _FakeFastWAM:
    device = torch.device("cpu")
    torch_dtype = torch.float32
    vae = type("FakeVAE", (), {"model": torch.nn.Identity()})()

    @staticmethod
    def _encode_video_latents(video: torch.Tensor, *, tiled: bool) -> torch.Tensor:
        if tiled:
            raise AssertionError("Audit encoding must use the non-tiled baseline path.")
        # Preserve a three-slice temporal latent contract without depending on the real VAE.
        return torch.stack(
            [
                video[:, :, :3].mean(dim=2),
                video[:, :, 3:6].mean(dim=2),
                video[:, :, 6:].mean(dim=2),
            ],
            dim=2,
        )


class CounterfactualTests(unittest.TestCase):
    def test_gate3_configuration_rejects_silent_algorithm_changes(self) -> None:
        section = OmegaConf.load("configs/task_decoupling/sam3p1_local_blur.yaml")
        validate_task_decoupling_config(section)
        section.counterfactual.edge_feather_px = 2
        with self.assertRaisesRegex(ValueError, "feather"):
            validate_task_decoupling_config(section)

    def test_training_override_requires_explicit_gate_acknowledgement(self) -> None:
        section = OmegaConf.load(
            "configs/task_decoupling/samhq_oracle_box_local_blur.yaml"
        )
        section.mode = "training_gate4_override"
        section.reward = {
            "type": "task_direction",
            "gate4_override": True,
            "invalid_mask_policy": "error",
        }
        validate_task_decoupling_config(section)
        section.reward.gate4_override = False
        with self.assertRaisesRegex(ValueError, "acknowledgement"):
            validate_task_decoupling_config(section)

    def test_entity_table_is_bound_to_the_rollout_task(self) -> None:
        table = Path("configs/task_entities/libero_10_task7.yaml")
        entities = load_entity_specification(table, expected_suite="libero_10", expected_task_id=7)
        self.assertEqual(
            [item.entity_id for item in entities],
            ["alphabet_soup", "cream_cheese", "basket"],
        )
        with self.assertRaisesRegex(ValueError, "task"):
            load_entity_specification(table, expected_suite="libero_10", expected_task_id=6)

    def test_counterfactual_changes_only_selected_pixels(self) -> None:
        video = np.zeros((9, 12, 12, 3), dtype=np.uint8)
        video[:, 4:8, 4:8] = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        masks = np.zeros((1, 9, 12, 12), dtype=bool)
        masks[:, :, 4:8, 4:8] = True

        result = build_local_blur_counterfactual(
            {"image": video},
            masks,
            camera_names=("image",),
            sigma_short_edge_ratio=0.1,
        )["image"]

        self.assertTrue(np.array_equal(result[~masks[0]], video[~masks[0]]))
        self.assertFalse(np.array_equal(result[masks[0]], video[masks[0]]))

    def test_pipeline_excludes_first_latent_slice(self) -> None:
        generator = np.random.default_rng(7)
        videos = {
            "image": generator.integers(0, 256, (9, 12, 12, 3), dtype=np.uint8),
            "wrist_image": generator.integers(0, 256, (9, 12, 12, 3), dtype=np.uint8),
        }
        masks = np.zeros((2, 1, 9, 12, 12), dtype=bool)
        masks[:, :, :, 3:9, 3:9] = True
        entities = (EntitySpec("object", "manipulated", "object"),)
        pipeline = TaskRegionDisentangler(
            mask_provider=ArrayOracleMaskProvider(masks, ("image", "wrist_image")),
            entities=entities,
            camera_order=("image", "wrist_image"),
            concat_mode="horizontal",
            minimum_component_area_px=0,
            fill_hole_area_px=0,
            dilation_radius_px=0,
            sigma_short_edge_ratio=0.1,
        )

        result = pipeline.extract(videos, _FakeFastWAM())

        self.assertEqual(result.delta_latent.shape[2], 3)
        self.assertEqual(result.future_delta.shape[2], 2)
        torch.testing.assert_close(result.future_delta, result.delta_latent[:, :, 1:])
        self.assertTrue(result.valid)

    def test_oracle_masks_become_clipped_per_frame_box_prompts(self) -> None:
        masks = np.zeros((1, 1, 2, 10, 12), dtype=bool)
        masks[0, 0, 0, 0:2, 1:4] = True
        prompts = boxes_from_binary_masks(
            masks,
            camera_names=("image",),
            entity_ids=("item",),
            source="test_oracle",
            padding_px=2,
        )
        np.testing.assert_array_equal(prompts.boxes_xyxy[0, 0, 0], [0, 0, 6, 4])
        self.assertTrue(prompts.valid[0, 0, 0])
        self.assertFalse(prompts.valid[0, 0, 1])


class FileServiceTests(unittest.TestCase):
    def test_filesystem_provider_validates_and_caches_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory)
            provider = FileServiceMaskProvider(
                queue,
                model_identity="test-model",
                timeout_seconds=2.0,
                poll_seconds=0.01,
            )
            videos = {"image": np.zeros((3, 8, 8, 3), dtype=np.uint8)}
            entities = (EntitySpec("item", "manipulated", "item"),)

            def respond() -> None:
                while not list((queue / "requests").glob("*.json")):
                    time.sleep(0.005)
                request_path = next((queue / "requests").glob("*.json"))
                token = request_path.stem
                masks = np.ones((1, 1, 3, 8, 8), dtype=bool)
                response_npz = queue / "responses" / f"{token}.npz"
                np.savez_compressed(
                    response_npz,
                    packed_masks=np.packbits(masks.reshape(-1)),
                    mask_shape=np.asarray(masks.shape, dtype=np.int64),
                )
                (queue / "responses" / f"{token}.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "status": "ok",
                            "model_identity": "test-model",
                            "output_probability_threshold": 0.5,
                            "prompt_mode": "text_video",
                        }
                    ),
                    encoding="utf-8",
                )

            worker = threading.Thread(target=respond)
            worker.start()
            first = provider.segment(videos, entities)
            worker.join()
            second = provider.segment(videos, entities)

            self.assertTrue(first.masks.all())
            self.assertTrue(np.array_equal(first.masks, second.masks))
            self.assertEqual(len(list((queue / "cache").glob("*.npz"))), 1)

    def test_box_service_serializes_structured_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = Path(directory)
            provider = FileServiceMaskProvider(
                queue,
                model_identity="box-model",
                timeout_seconds=2.0,
                poll_seconds=0.01,
                prompt_mode="box_per_frame",
            )
            videos = {"image": np.zeros((2, 8, 8, 3), dtype=np.uint8)}
            entities = (EntitySpec("item", "manipulated", "item"),)
            prompts = BoxPromptBatch(
                camera_names=("image",),
                entity_ids=("item",),
                boxes_xyxy=np.asarray([[[[1, 1, 5, 5], [2, 2, 6, 6]]]], dtype=np.float32),
                valid=np.ones((1, 1, 2), dtype=bool),
                source="test_boxes",
            )

            def respond() -> None:
                while not list((queue / "requests").glob("*.json")):
                    time.sleep(0.005)
                request_path = next((queue / "requests").glob("*.json"))
                request = json.loads(request_path.read_text(encoding="utf-8"))
                with np.load(queue / "requests" / request["payload_file"]) as payload:
                    np.testing.assert_array_equal(payload["prompt_boxes_xyxy"], prompts.boxes_xyxy)
                token = request_path.stem
                masks = np.ones((1, 1, 2, 8, 8), dtype=bool)
                np.savez_compressed(
                    queue / "responses" / f"{token}.npz",
                    packed_masks=np.packbits(masks.reshape(-1)),
                    mask_shape=np.asarray(masks.shape, dtype=np.int64),
                )
                (queue / "responses" / f"{token}.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "status": "ok",
                            "model_identity": "box-model",
                            "output_probability_threshold": 0.5,
                            "prompt_mode": "box_per_frame",
                        }
                    ),
                    encoding="utf-8",
                )

            worker = threading.Thread(target=respond)
            worker.start()
            result = provider.segment(videos, entities, prompts)
            worker.join()
            self.assertTrue(result.masks.all())


class AuditMetricTests(unittest.TestCase):
    def test_gate4_launcher_owns_service_lifecycle(self) -> None:
        source = Path("scripts/resilient/run_task_reward_audit.sh").read_text(encoding="utf-8")
        self.assertIn("trap cleanup EXIT", source)
        self.assertIn('stop_process_group "${SAM_PID}"', source)
        self.assertIn('stop_process_group "${AUDIT_PID}"', source)
        self.assertIn(
            "accelerator.end_training",
            Path("scripts/resilient/audit_task_reward.py").read_text(encoding="utf-8"),
        )

    def test_segmentation_and_probe_metrics(self) -> None:
        truth = np.zeros((1, 2, 4, 4), dtype=bool)
        truth[..., 1:3, 1:3] = True
        metrics = binary_mask_metrics(truth, truth)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)

        x_train = np.arange(16, dtype=np.float64).reshape(8, 2)
        y_train = x_train[:, 0] * 2.0
        probe = ridge_probe(
            x_train,
            y_train,
            x_train,
            y_train,
            regularization=1.0e-6,
        )
        self.assertGreater(probe.r2, 0.999)
        rank = effective_rank(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
        self.assertGreaterEqual(rank, 1.0)
        self.assertLessEqual(rank, 2.0)

    def test_mask_contract_rejects_non_boolean_arrays(self) -> None:
        with self.assertRaises(TypeError):
            MaskBatch(("image",), ("item",), np.zeros((1, 1, 2, 4, 4), dtype=np.uint8))

    def test_gate4_uses_only_complete_valid_groups(self) -> None:
        rows = []
        for state_index in (35, 36):
            for candidate_index in range(4):
                progress = float(candidate_index)
                rows.append(
                    {
                        "group_id": f"state_{state_index}",
                        "group_size": 4,
                        "state_index": state_index,
                        "mask_valid": not (state_index == 36 and candidate_index == 0),
                        "terminal_before_anchor": False,
                        "task_progress": progress,
                        "joint1_motion": float((candidate_index + 1) % 4),
                        "reward_full": -progress,
                        "reward_counterfactual": 0.0,
                        "reward_margin": progress,
                        "reward_task_direction": progress,
                    }
                )
        summary = summarize_gate4_rows(
            rows,
            state_indices=[35, 36],
            minimum_reward_std=1.0e-4,
            progress_tolerance=1.0e-4,
        )
        self.assertEqual(summary["requested_group_count"], 2)
        self.assertEqual(summary["valid_group_count"], 1)
        self.assertEqual(summary["rewards"]["reward_task_direction"]["pairwise_accuracy"], 1.0)
        decision = evaluate_gate4(
            {
                "valid_group_fraction": 1.0,
                "rewards": {
                    "reward_task_direction": {
                        "pairwise_accuracy": 0.8,
                        "active_group_fraction": 1.0,
                        "motion_partial_correlation": 0.1,
                    },
                    "reward_full": {
                        "pairwise_accuracy": 0.6,
                        "motion_partial_correlation": 0.4,
                    },
                },
            },
            {
                "minimum_valid_group_fraction": 0.5,
                "minimum_pairwise_accuracy": 0.6,
                "minimum_pairwise_accuracy_improvement_over_full": 0.05,
                "minimum_active_group_fraction": 0.8,
                "maximum_abs_motion_partial_correlation": 1.0,
            },
        )
        self.assertTrue(decision["passed"])


if __name__ == "__main__":
    unittest.main()
