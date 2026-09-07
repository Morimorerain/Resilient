"""Regression tests for the LIBERO evaluation task lifecycle."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf


class _FakeEnvironment:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeFaultPipeline:
    @staticmethod
    def metadata() -> dict:
        return {"enabled": False, "faults": []}


class LiberoEvaluationLifecycleTests(unittest.TestCase):
    def test_task_description_is_available_when_results_are_initialized(self) -> None:
        from experiments.libero import eval_libero_single

        cfg = OmegaConf.create(
            {
                "seed": 42,
                "EVALUATION": {
                    "num_trials": 1,
                    "task_id": 0,
                    "save_rollout_videos": False,
                    "visualize_future_video": False,
                    "fault_detection": {"enabled": False},
                    "fault": {"pipeline": {"enabled": False}},
                },
            }
        )
        env = _FakeEnvironment()
        pipeline = _FakeFaultPipeline()

        with tempfile.TemporaryDirectory() as temp_dir:
            video_dir = Path(temp_dir) / "videos"
            predicted_video_dir = Path(temp_dir) / "predicted_videos"
            with (
                patch.object(
                    eval_libero_single,
                    "build_fault_pipeline",
                    return_value=pipeline,
                ),
                patch.object(
                    eval_libero_single,
                    "get_libero_env",
                    return_value=(env, "test task description"),
                ),
                patch.object(
                    eval_libero_single,
                    "run_single_episode",
                    return_value=(False, [], [], None, None, None),
                ),
            ):
                results = eval_libero_single.run_single_task(
                    task=object(),
                    initial_states=[object()],
                    model=None,
                    processor=None,
                    cfg=cfg,
                    video_dir=video_dir,
                    predicted_video_dir=predicted_video_dir,
                    action_horizon=10,
                    input_w=224,
                    input_h=224,
                    model_device="cpu",
                )

        self.assertEqual(results["task_description"], "test task description")
        self.assertEqual(results["failure_episodes"], [0])
        self.assertTrue(env.closed)


if __name__ == "__main__":
    unittest.main()
