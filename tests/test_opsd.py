"""CPU tests for Fast-WAM-aligned OPSD-Flow components."""

from __future__ import annotations

import inspect
import json
import unittest

import torch
import torch.nn as nn

from fastwam.models.wan22.fastwam import FastWAM
from resilient.faults.paired_observation import PairedObservation
from resilient.opsd.adapters import (
    FastWAMLoraConfig,
    adapter_state_dict,
    discover_fastwam_lora_targets,
    inject_fastwam_aligned_lora,
    load_fastwam_lora_adapter,
    lora_disabled,
)
from resilient.opsd.losses import opsd_flow_loss
from resilient.opsd.runtime import build_rollout_schedule
from resilient.opsd.teacher_inputs import TeacherInputContext, build_teacher_input_provider
from resilient.opsd.trainer import OPSDFlowTrainer, _scalar_to_float
from resilient.opsd.types import ActionConditioning, StudentTrajectory
from resilient.state_banks import (
    assert_disjoint_manifests,
    load_task_states,
    state_fingerprint,
)


class _ToyFastWAM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_expert = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        self.action_expert = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        self.proprio_encoder = nn.Linear(2, 4)
        self.vae = nn.Linear(4, 4)
        self.mot = nn.Module()
        self.mot.video_expert = self.video_expert
        self.mot.action_expert = self.action_expert

    def forward(self, video, action, proprio):
        return self.video_expert(video) + self.proprio_encoder(proprio), self.action_expert(action)


class OPSDLossTests(unittest.TestCase):
    def test_state_bank_rejects_leakage_and_verifies_tensor(self) -> None:
        import tempfile
        from pathlib import Path

        training = {
            "schema_version": 1,
            "states": [{"state_sha256": "seen", "generation_seed": 1}],
        }
        validation = {
            "schema_version": 1,
            "states": [{"state_sha256": "unseen", "generation_seed": 2}],
        }
        assert_disjoint_manifests(validation, training)
        with self.assertRaisesRegex(ValueError, "State leakage"):
            assert_disjoint_manifests(training, training)

        state = torch.tensor([1.0, 2.0], dtype=torch.float64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(torch.stack([state]), root / "task.pt")
            manifest = {
                "schema_version": 1,
                "states": [
                    {
                        "suite": "suite",
                        "task_id": 0,
                        "state_index": 0,
                        "state_sha256": state_fingerprint(state),
                        "generation_seed": 2,
                        "file": "task.pt",
                    }
                ],
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = load_task_states(
                path, suite="suite", task_id=0, expected_count=1
            )
            self.assertTrue(torch.equal(loaded[0], state))

    def test_rollout_schedule_is_shuffled_balanced_and_reproducible(self) -> None:
        kwargs = {
            "suites": ["suite_a", "suite_b"],
            "tasks_per_suite": 2,
            "rollouts_per_task": 3,
            "epoch": 0,
            "schedule_seed": 42,
            "environment_seed": 42,
            "inference_seed": 42,
        }
        first = build_rollout_schedule(**kwargs)
        second = build_rollout_schedule(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        ordered = [
            (suite, task_id, rollout_id)
            for suite in kwargs["suites"]
            for task_id in range(2)
            for rollout_id in range(3)
        ]
        self.assertNotEqual(
            [(item.suite, item.task_id, item.rollout_id) for item in first], ordered
        )
        self.assertEqual(len({item.environment_seed for item in first}), 12)
        self.assertEqual(len({item.inference_seed for item in first}), 12)
        for suite in kwargs["suites"]:
            for task_id in range(2):
                selected = [
                    item for item in first if item.suite == suite and item.task_id == task_id
                ]
                self.assertEqual({item.initial_state_index for item in selected}, {0, 1, 2})

    def test_scalar_metric_accepts_tensor_and_deepspeed_float(self) -> None:
        self.assertEqual(_scalar_to_float(torch.tensor(1.25, requires_grad=True)), 1.25)
        self.assertEqual(_scalar_to_float(2.5), 2.5)

    def test_checkpoint_retention_keeps_newest_completed_states(self) -> None:
        import tempfile
        from pathlib import Path

        class _Accelerator:
            is_main_process = True

            @staticmethod
            def wait_for_everyone() -> None:
                return None

        trainer = OPSDFlowTrainer.__new__(OPSDFlowTrainer)
        trainer.accelerator = _Accelerator()
        with tempfile.TemporaryDirectory() as directory:
            state_root = Path(directory) / "checkpoints" / "state"
            for step in (4, 8, 12):
                (state_root / f"step_{step:08d}").mkdir(parents=True)
            trainer.prune_checkpoints(Path(directory), keep=2)
            self.assertEqual(
                sorted(path.name for path in state_root.iterdir()),
                ["step_00000008", "step_00000012"],
            )

    def test_weighted_student_trajectory_loss_and_teacher_detach(self) -> None:
        student = torch.zeros((1, 2, 1, 1), requires_grad=True)
        teacher = torch.tensor([[[[1.0]], [[2.0]]]], requires_grad=True)
        loss, metrics = opsd_flow_loss(student, teacher, torch.tensor([1.0, 3.0]))
        self.assertAlmostEqual(float(loss), 3.25)
        self.assertAlmostEqual(float(metrics["loss_raw"]), 2.5)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_pointwise_clipping_caps_each_coordinate(self) -> None:
        student = torch.zeros((1, 1, 1, 2))
        teacher = torch.tensor([[[[1.0, 2.0]]]])
        loss, metrics = opsd_flow_loss(student, teacher, torch.tensor([1.0]), pointwise_clip=0.05)
        self.assertAlmostEqual(float(loss), 0.05)
        self.assertAlmostEqual(float(metrics["clip_fraction"]), 1.0)


class FastWAMLoraTests(unittest.TestCase):
    def test_targets_match_fastwam_trainable_scope_and_exclude_vae(self) -> None:
        model = _ToyFastWAM()
        config = FastWAMLoraConfig(rank=3, alpha=6)
        targets, ranks = discover_fastwam_lora_targets(model, config)
        self.assertTrue(any(name.startswith("video_expert") for name in targets))
        self.assertTrue(any(name.startswith("action_expert") for name in targets))
        self.assertTrue(any(name.startswith("proprio_encoder") for name in targets))
        self.assertFalse(any(name.startswith("vae") for name in targets))
        self.assertTrue(all(rank <= 3 for rank in ranks.values()))

    def test_injection_freezes_base_and_teacher_can_disable_lora(self) -> None:
        model = _ToyFastWAM()
        audit = inject_fastwam_aligned_lora(model, FastWAMLoraConfig(rank=2, alpha=4))
        self.assertGreater(audit["trainable_parameters"], 0)
        self.assertTrue(
            all(
                ".lora_" in name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
        )
        for name, parameter in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                nn.init.constant_(parameter, 0.25)
        inputs = (torch.ones(1, 4), torch.ones(1, 4), torch.ones(1, 2))
        student_output = model(*inputs)
        with lora_disabled(model):
            teacher_output = model(*inputs)
        self.assertFalse(torch.allclose(student_output[0], teacher_output[0]))
        self.assertFalse(torch.allclose(student_output[1], teacher_output[1]))
        self.assertTrue(all(not name.startswith("mot.") for name in adapter_state_dict(model)))

    def test_saved_adapter_round_trip(self) -> None:
        import tempfile
        from pathlib import Path

        config = FastWAMLoraConfig(rank=2, alpha=4)
        source = _ToyFastWAM()
        inject_fastwam_aligned_lora(source, config)
        for name, parameter in source.named_parameters():
            if ".lora_" in name:
                nn.init.constant_(parameter, 0.125)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "adapter.pt"
            torch.save(adapter_state_dict(source), checkpoint)
            restored = _ToyFastWAM()
            load_fastwam_lora_adapter(restored, config=config, checkpoint=checkpoint)
            expected = adapter_state_dict(source)
            actual = adapter_state_dict(restored)
            self.assertEqual(set(expected), set(actual))
            self.assertTrue(all(torch.equal(expected[key], actual[key]) for key in expected))


class TeacherAndTraceTests(unittest.TestCase):
    def test_clean_visual_provider_replaces_only_image(self) -> None:
        student_image = torch.zeros(1, 3, 2, 2)
        clean_image = torch.ones(1, 3, 2, 2)
        proprio = torch.tensor([[0.2, 0.4]])
        student = ActionConditioning(input_image=student_image, prompt="task", proprio=proprio)
        trajectory = StudentTrajectory.from_inference_result(
            {
                "action": torch.zeros(1, 2),
                "denoising_trace": [
                    {
                        "latent": torch.zeros(1, 1, 2),
                        "timestep": torch.tensor([1.0]),
                        "delta_sigma": torch.tensor([-1.0]),
                    }
                ],
            },
            expected_steps=1,
        )
        paired = PairedObservation("fault", "clean", {})
        provider = build_teacher_input_provider(
            {"type": "clean_visual"}, image_encoder=lambda observation: clean_image
        )
        teacher = provider.build(TeacherInputContext(paired, student, trajectory, extras={}))
        self.assertIs(teacher.input_image, clean_image)
        self.assertEqual(teacher.prompt, student.prompt)
        self.assertIs(teacher.proprio, proprio)
        self.assertIs(teacher.context, student.context)

    def test_fastwam_trace_switch_is_default_off(self) -> None:
        parameter = inspect.signature(FastWAM.infer_action).parameters["return_denoising_trace"]
        self.assertIs(parameter.default, False)


if __name__ == "__main__":
    unittest.main()
