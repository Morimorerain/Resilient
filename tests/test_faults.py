"""Tests for reusable, severity-aware fault injection."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path
from types import ModuleType

from resilient.faults import FaultPipeline, FaultRuntime, build_fault_pipeline
from resilient.faults.visual.camera_pose import (
    local_xyz_offset_quaternion,
    rotate_vector,
)


def _load_evaluation_script() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "resilient"
        / "evaluate_fault.py"
    )
    spec = importlib.util.spec_from_file_location("evaluate_fault", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeModel:
    def __init__(self) -> None:
        self.cam_pos = [[1.0, 2.0, 3.0]]
        self.cam_quat = [[1.0, 0.0, 0.0, 0.0]]

    @staticmethod
    def camera_name2id(name: str) -> int:
        if name != "agentview":
            raise ValueError(name)
        return 0


class _FakeSimulation:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.forward_calls = 0

    def forward(self) -> None:
        self.forward_calls += 1


class _FakeEnvironment:
    def __init__(self) -> None:
        self.sim = _FakeSimulation()


def _camera_config(severity: float = 30.0) -> dict:
    return {
        "pipeline": {
            "id": "camera_test",
            "seed": 7,
            "faults": [
                {
                    "id": "camera",
                    "family": "visual.camera_pose",
                    "target": "agentview",
                    "operation": {
                        "type": "local_axis_rotation",
                        "axis": "z",
                        "direction": "positive",
                    },
                    "severity": {
                        "name": "rotation_angle",
                        "value": severity,
                        "unit": "degree",
                    },
                    "parameters": {"position_offset": [0.1, -0.2, 0.3]},
                }
            ],
        }
    }


class _ExtensionFault(FaultRuntime):
    family = "test.extension"
    fault_id = "extension"
    scopes = frozenset({"observation", "action"})

    def transform_observation(self, observation, env, context):
        del env, context
        observation["faulted"] = True
        return observation

    def transform_action(self, action, env, context):
        del env, context
        return [value + 1 for value in action]

    def metadata(self):
        return {"id": self.fault_id, "family": self.family}

    def slug(self):
        return "extension"


class FaultPipelineTests(unittest.TestCase):
    def test_local_x_rotation_uses_wxyz_and_degrees(self) -> None:
        quaternion = local_xyz_offset_quaternion((90.0, 0.0, 0.0))
        self.assertAlmostEqual(quaternion[0], math.sqrt(0.5))
        self.assertAlmostEqual(quaternion[1], math.sqrt(0.5))
        self.assertAlmostEqual(quaternion[2], 0.0)
        self.assertAlmostEqual(quaternion[3], 0.0)

    def test_local_position_offset_is_rotated_into_parent_frame(self) -> None:
        quarter_turn_about_z = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
        rotated = rotate_vector(quarter_turn_about_z, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(rotated[0], 0.0)
        self.assertAlmostEqual(rotated[1], 1.0)
        self.assertAlmostEqual(rotated[2], 0.0)

    def test_repeated_reset_does_not_accumulate_and_suspend_restores(self) -> None:
        env = _FakeEnvironment()
        pipeline = build_fault_pipeline(_camera_config())
        context = pipeline.context()
        pipeline.on_reset(env, context)
        first_position = list(env.sim.model.cam_pos[0])
        first_quaternion = list(env.sim.model.cam_quat[0])
        pipeline.on_reset(env, context)
        self.assertEqual(list(env.sim.model.cam_pos[0]), first_position)
        self.assertEqual(list(env.sim.model.cam_quat[0]), first_quaternion)
        with pipeline.suspend(env, context=context):
            self.assertEqual(list(env.sim.model.cam_pos[0]), [1.0, 2.0, 3.0])
            self.assertEqual(list(env.sim.model.cam_quat[0]), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(list(env.sim.model.cam_pos[0]), first_position)
        pipeline.detach(env)
        self.assertEqual(list(env.sim.model.cam_pos[0]), [1.0, 2.0, 3.0])

    def test_severities_share_family_but_have_distinct_slugs(self) -> None:
        pipelines = [build_fault_pipeline(_camera_config(value)) for value in (10, 20, 30)]
        self.assertEqual(
            {pipeline.metadata()["faults"][0]["family"] for pipeline in pipelines},
            {"visual.camera_pose"},
        )
        self.assertEqual(len({pipeline.slug() for pipeline in pipelines}), 3)

    def test_pipeline_accepts_future_observation_and_action_faults(self) -> None:
        pipeline = FaultPipeline("extension", 5, [_ExtensionFault()])
        observation = {"clean": True}
        transformed = pipeline.transform_observation(observation, object(), pipeline.context())
        self.assertEqual(transformed, {"clean": True, "faulted": True})
        self.assertEqual(observation, {"clean": True})
        self.assertEqual(
            pipeline.transform_action([0, 2], object(), pipeline.context()), [1, 3]
        )

    def test_disabled_pipeline_is_a_no_op(self) -> None:
        pipeline = build_fault_pipeline({"pipeline": {"enabled": False}})
        self.assertFalse(pipeline.enabled)
        self.assertEqual(pipeline.transform_action([1], object(), pipeline.context()), [1])


class FaultEvaluationScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = _load_evaluation_script()

    def test_gpu_parser(self) -> None:
        self.assertEqual(self.script.parse_gpu_ids("0,2,4,6"), (0, 2, 4, 6))

    def test_standard_run_name_contains_family_target_and_severity(self) -> None:
        name = self.script.build_run_name(_camera_config(20), Path("example.pt"))
        self.assertEqual(
            name,
            "visual-camera_pose-agentview-rotation_angle-p20p0degree__model-example",
        )

    def test_markdown_summary_uses_weighted_total(self) -> None:
        summary = {
            "suite_stats": {
                "libero_spatial": {"total_successes": 9, "total_trials": 10},
                "libero_object": {"total_successes": 5, "total_trials": 5},
            }
        }
        table = self.script.build_markdown_summary(
            summary, ("libero_spatial", "libero_object")
        )
        self.assertIn("| Spatial | 9/10, 90.0% |", table)
        self.assertIn("| Object | 5/5, 100.0% |", table)
        self.assertIn("| **Total** | **14/15, 93.33%** |", table)


if __name__ == "__main__":
    unittest.main()
