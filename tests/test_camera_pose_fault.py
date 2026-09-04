"""Tests for deterministic LIBERO camera-pose fault injection."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path
from types import ModuleType

from resilient.camera_pose_fault import (
    CameraPoseFaultApplier,
    CameraPoseFaultSpec,
    local_xyz_offset_quaternion,
    rotate_vector,
)


def _load_evaluation_script() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "resilient"
        / "evaluate_camera_pose_fault.py"
    )
    spec = importlib.util.spec_from_file_location("evaluate_camera_pose_fault", script_path)
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


class CameraPoseFaultTests(unittest.TestCase):
    def test_local_x_rotation_uses_wxyz_and_degrees(self) -> None:
        quaternion = local_xyz_offset_quaternion((90.0, 0.0, 0.0))

        self.assertAlmostEqual(quaternion[0], math.sqrt(0.5))
        self.assertAlmostEqual(quaternion[1], math.sqrt(0.5))
        self.assertAlmostEqual(quaternion[2], 0.0)
        self.assertAlmostEqual(quaternion[3], 0.0)

    def test_repeated_application_does_not_accumulate_offsets(self) -> None:
        env = _FakeEnvironment()
        spec = CameraPoseFaultSpec(
            enabled=True,
            camera="agentview",
            position_offset=(0.1, -0.2, 0.3),
            rotation_offset_deg=(0.0, 0.0, 90.0),
        )
        applier = CameraPoseFaultApplier(spec)

        first = applier.apply(env)
        second = applier.apply(env)

        self.assertEqual(first["effective_position"], [1.1, 1.8, 3.3])
        self.assertEqual(second["effective_position"], [1.1, 1.8, 3.3])
        self.assertEqual(env.sim.forward_calls, 2)

    def test_local_position_offset_is_rotated_into_parent_frame(self) -> None:
        quarter_turn_about_z = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))

        rotated = rotate_vector(quarter_turn_about_z, (1.0, 0.0, 0.0))

        self.assertAlmostEqual(rotated[0], 0.0)
        self.assertAlmostEqual(rotated[1], 1.0)
        self.assertAlmostEqual(rotated[2], 0.0)

    def test_disabled_spec_preserves_default_behavior(self) -> None:
        spec = CameraPoseFaultSpec.from_config(None)

        self.assertFalse(spec.enabled)
        self.assertEqual(spec.position_offset, (0.0, 0.0, 0.0))
        self.assertEqual(spec.rotation_offset_deg, (0.0, 0.0, 0.0))


class CameraPoseFaultScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = _load_evaluation_script()

    def test_gpu_parser(self) -> None:
        self.assertEqual(self.script.parse_gpu_ids("0,2,4,6"), (0, 2, 4, 6))

    def test_standard_run_name_contains_camera_and_offsets(self) -> None:
        name = self.script.build_run_name(
            "robot0_eye_in_hand",
            (0.0, -0.01, 0.025),
            (10.0, 0.0, -2.5),
            Path("example.pt"),
        )

        self.assertEqual(
            name,
            "robot0_eye_in_hand-pos_xp0p000_ym0p010_zp0p025m-"
            "rot_xp10p0_yp0p0_zm2p5deg-model_example",
        )

    def test_markdown_summary_uses_weighted_total(self) -> None:
        summary = {
            "suite_stats": {
                "libero_spatial": {"total_successes": 9, "total_trials": 10},
                "libero_object": {"total_successes": 5, "total_trials": 5},
            }
        }

        table = self.script.build_markdown_summary(
            summary,
            ("libero_spatial", "libero_object"),
        )

        self.assertIn("| Spatial | 9/10, 90.0% |", table)
        self.assertIn("| Object | 5/5, 100.0% |", table)
        self.assertIn("| **Total** | **14/15, 93.33%** |", table)


if __name__ == "__main__":
    unittest.main()
