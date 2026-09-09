"""Tests for reusable, severity-aware fault injection."""

from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np

from resilient.faults import (
    FaultPipeline,
    FaultRuntime,
    build_fault_pipeline,
    install_fault_pipeline,
)
from resilient.faults.structure import available_joint_motion_laws
from resilient.faults.visual.camera_pose import (
    local_xyz_offset_quaternion,
    rotate_vector,
)
from resilient.visualization import compose_comparison_frame


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


class _FakeJointModel:
    _indices = {"robot0_joint1": 0, "robot0_joint3": 2}

    def get_joint_qpos_addr(self, name: str) -> int:
        return self._indices[name]

    def get_joint_qvel_addr(self, name: str) -> int:
        return self._indices[name]


class _FakeJointData:
    def __init__(self) -> None:
        self.qpos = np.zeros(3, dtype=np.float64)
        self.qvel = np.zeros(3, dtype=np.float64)


class _FakeJointSimulation:
    def __init__(self) -> None:
        self.model = _FakeJointModel()
        self.data = _FakeJointData()
        self.forward_calls = 0
        self.step_displacement = np.asarray([0.8, 0.4, -0.2], dtype=np.float64)
        self.step_velocity = np.asarray([1.2, 0.7, -0.6], dtype=np.float64)

    def step(self) -> None:
        self.data.qpos += self.step_displacement
        self.data.qvel[:] = self.step_velocity

    def forward(self) -> None:
        self.forward_calls += 1


class _FakeJointEnvironment:
    def __init__(self) -> None:
        self.sim = _FakeJointSimulation()

    def reset(self) -> dict:
        self.sim.data.qpos[:] = 0.0
        self.sim.data.qvel[:] = 0.0
        return self._get_observations()

    def set_init_state(self, state) -> dict:
        self.sim.data.qpos[:] = state
        self.sim.data.qvel[:] = 0.0
        return self._get_observations()

    def step(self, action):
        del action
        self.sim.step()
        return self._get_observations(), 0.0, False, {}

    def _get_observations(self, force_update: bool = False) -> dict:
        del force_update
        return {"qpos": self.sim.data.qpos.copy()}

    def check_success(self) -> bool:
        return False

    def close(self) -> None:
        pass


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


def _joint_motion_config(targets=None, retention: float = 0.5) -> dict:
    return {
        "pipeline": {
            "id": "joint_motion_test",
            "seed": 11,
            "faults": [
                {
                    "id": "joint_motion",
                    "family": "structure.joint_motion",
                    "targets": targets or ["robot0_joint1"],
                    "operation": {"type": "proportional"},
                    "severity": {
                        "name": "motion_retention",
                        "value": retention,
                        "unit": "ratio",
                    },
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

    def test_joint_motion_fault_scales_selected_joint_at_environment_step(self) -> None:
        raw_env = _FakeJointEnvironment()
        pipeline = build_fault_pipeline(_joint_motion_config())
        env = install_fault_pipeline(raw_env, pipeline, initial_episode_index=3)
        env.reset()
        env.step(np.zeros(7, dtype=np.float64))

        self.assertTrue(pipeline.requires_post_step_observation_refresh)
        np.testing.assert_allclose(env.sim.data.qpos, [0.4, 0.4, -0.2])
        np.testing.assert_allclose(env.sim.data.qvel, [0.6, 0.7, -0.6])
        self.assertEqual(env.sim.forward_calls, 1)
        self.assertEqual(
            pipeline.metadata()["faults"][0]["injection_layer"],
            "faulted_environment_dynamics",
        )
        env.close()

    def test_fault_runtime_state_can_be_copied_to_a_shadow_environment(self) -> None:
        source = install_fault_pipeline(
            _FakeJointEnvironment(),
            build_fault_pipeline(_joint_motion_config()),
            initial_episode_index=7,
        )
        shadow = install_fault_pipeline(
            _FakeJointEnvironment(),
            build_fault_pipeline(_joint_motion_config()),
            initial_episode_index=7,
        )
        source.reset()
        source.step(np.zeros(7, dtype=np.float64))
        state = source.fault_runtime_state_dict()
        shadow.reset()
        shadow.set_init_state(source.sim.data.qpos.copy())
        observation = shadow.load_fault_runtime_state_dict(state)
        self.assertEqual(state["episode_index"], 7)
        self.assertEqual(state["step_index"], 1)
        np.testing.assert_allclose(observation["qpos"], source.sim.data.qpos)
        self.assertEqual(shadow.fault_runtime_state_dict(), state)
        source.close()
        shadow.close()

    def test_joint_motion_fault_supports_multiple_targets(self) -> None:
        raw_env = _FakeJointEnvironment()
        pipeline = build_fault_pipeline(
            _joint_motion_config(["robot0_joint1", "robot0_joint3"], retention=0.25)
        )
        env = install_fault_pipeline(raw_env, pipeline)
        env.reset()
        env.sim.step_displacement[:] = [0.8, 0.4, -0.4]
        env.sim.step_velocity[:] = [1.2, 0.7, -0.8]
        env.step(np.zeros(7, dtype=np.float64))

        np.testing.assert_allclose(env.sim.data.qpos, [0.2, 0.4, -0.1])
        np.testing.assert_allclose(env.sim.data.qvel, [0.3, 0.7, -0.2])
        self.assertEqual(available_joint_motion_laws(), ("proportional",))
        env.close()

    def test_faulted_environment_owns_joint_fault_lifecycle(self) -> None:
        raw_env = _FakeJointEnvironment()
        pipeline = build_fault_pipeline(_joint_motion_config())
        env = install_fault_pipeline(raw_env, pipeline)

        env.reset()
        initial_observation = env.set_init_state(np.zeros(3, dtype=np.float64))
        observation, _, _, _ = env.step(np.zeros(7, dtype=np.float64))

        np.testing.assert_allclose(initial_observation["qpos"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(observation["qpos"], [0.4, 0.4, -0.2])
        self.assertEqual(env.fault_metadata["activation"], "environment_startup")
        env.close()

    def test_simulator_level_faults_compose_in_one_environment(self) -> None:
        config = _joint_motion_config(["robot0_joint1"], retention=0.5)
        second = _joint_motion_config(["robot0_joint3"], retention=0.5)["pipeline"][
            "faults"
        ][0]
        second["id"] = "joint3_motion"
        config["pipeline"]["faults"].append(second)
        raw_env = _FakeJointEnvironment()
        env = install_fault_pipeline(raw_env, build_fault_pipeline(config))

        env.reset()
        observation, _, _, _ = env.step(np.zeros(7, dtype=np.float64))

        np.testing.assert_allclose(observation["qpos"], [0.4, 0.4, -0.1])
        env.close()

    def test_joint_motion_fault_rejects_invalid_retention(self) -> None:
        with self.assertRaisesRegex(ValueError, "retention ratio"):
            build_fault_pipeline(_joint_motion_config(retention=1.5))

    def test_fault_comparison_frame_preserves_aligned_panel_geometry(self) -> None:
        clean = np.zeros((8, 12, 3), dtype=np.uint8)
        fault = np.full((8, 12, 3), 255, dtype=np.uint8)
        frame = compose_comparison_frame(
            clean,
            fault,
            title="test",
            clean_label="clean",
            fault_label="fault",
            clean_telemetry={"q": 0.0},
            fault_telemetry={"q": 1.0},
            timestamp_seconds=0.0,
            panel_scale=2,
        )
        self.assertEqual(frame.shape, (92, 48, 3))
        self.assertEqual(frame.dtype, np.uint8)


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
