"""CPU tests for compact Stage-I data and anchored Stage-II scheduling."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from resilient.outcome_fpo.runtime import build_outcome_rollout_schedule
from resilient.two_stage_opsd.dataset import FaultRolloutWindowDataset


class FaultRolloutDatasetTests(unittest.TestCase):
    def test_compact_episode_yields_native_fastwam_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = "put both objects in the basket"
            prompt = DEFAULT_PROMPT.format(task=task)
            text_dir = root / "text"
            text_dir.mkdir()
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            torch.save(
                {
                    "context": torch.ones(4, 6),
                    "mask": torch.tensor([True, True, False, False]),
                },
                text_dir / f"{digest}.t5_len4.wan22ti2v5b.pt",
            )
            episode_dir = root / "episodes"
            episode_dir.mkdir()
            torch.save(
                {
                    "image": torch.arange(33, dtype=torch.uint8)
                    .view(33, 1, 1, 1)
                    .expand(-1, 3, 16, 32)
                    .contiguous(),
                    "action": torch.zeros(32, 7),
                    "proprio": torch.zeros(33, 8),
                },
                episode_dir / "episode.pt",
            )
            manifest = {
                "schema_version": 1,
                "frame_steps": list(range(0, 33, 4)),
                "task_description": task,
                "windows": [{"episode": "episodes/episode.pt", "start": 0}],
            }
            manifest_path = root / "dataset_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            sample = FaultRolloutWindowDataset(
                manifest_path,
                text_embedding_cache_dir=text_dir,
                context_len=4,
            )[0]

            self.assertEqual(sample["video"].shape, (3, 9, 16, 32))
            self.assertEqual(sample["action"].shape, (32, 7))
            self.assertEqual(sample["proprio"].shape, (32, 8))
            torch.testing.assert_close(
                sample["video"][0, :, 0, 0],
                torch.tensor(
                    [2.0 * value / 255.0 - 1.0 for value in range(0, 33, 4)]
                ),
            )
            self.assertTrue(torch.equal(sample["context_mask"], torch.ones(4, dtype=torch.bool)))
            self.assertTrue(torch.equal(sample["context"][2:], torch.zeros(2, 6)))


class AnchoredScheduleTests(unittest.TestCase):
    def test_schedule_covers_every_state_anchor_pair_once(self) -> None:
        schedule = build_outcome_rollout_schedule(
            suites=["libero_10"],
            task_ids=[7],
            rollouts_per_task=3,
            anchor_steps=[0, 80, 160, 240],
            epoch=0,
            schedule_seed=42,
            environment_seed=42,
            inference_seed=42,
        )
        self.assertEqual(len(schedule), 12)
        self.assertEqual(sorted(item.sample_id for item in schedule), list(range(12)))
        self.assertEqual(
            sorted((item.initial_state_index, item.anchor_step) for item in schedule),
            [(state, anchor) for state in range(3) for anchor in (0, 80, 160, 240)],
        )
        self.assertEqual(
            schedule,
            build_outcome_rollout_schedule(
                suites=["libero_10"],
                task_ids=[7],
                rollouts_per_task=3,
                anchor_steps=[0, 80, 160, 240],
                epoch=0,
                schedule_seed=42,
                environment_seed=42,
                inference_seed=42,
            ),
        )


if __name__ == "__main__":
    unittest.main()
