"""Unit coverage for RoboTwin's simulator-owned Fault adapters."""

from __future__ import annotations

import copy
import math
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from resilient.robotwin.faults import install_fault_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakePose:
    def __init__(self, p=None, q=None):
        self.p = np.asarray([0.0, 0.0, 0.0] if p is None else p, dtype=np.float64)
        self.q = np.asarray([1.0, 0.0, 0.0, 0.0] if q is None else q, dtype=np.float64)


class FakePoseOwner:
    def __init__(self, pose=None):
        self._pose = FakePose() if pose is None else pose

    def get_pose(self):
        return FakePose(self._pose.p.copy(), self._pose.q.copy())

    def set_pose(self, pose):
        self._pose = FakePose(pose.p, pose.q)


class FakeCamera:
    def __init__(self):
        self.entity = FakePoseOwner()


class FakeCameras:
    def __init__(self):
        self.left_camera = FakeCamera()
        self.right_camera = FakeCamera()
        self.static_camera_name = ["head_camera"]
        self.static_camera_list = [FakeCamera()]

    def update_wrist_camera(self, left_pose, right_pose):
        self.left_camera.entity.set_pose(left_pose)
        self.right_camera.entity.set_pose(right_pose)


class FakeJoint:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


class FakeEntity:
    def __init__(self, joints):
        self._joints = list(joints)
        self._qpos = np.zeros(len(joints), dtype=np.float64)
        self._qvel = np.zeros(len(joints), dtype=np.float64)

    def get_active_joints(self):
        return list(self._joints)

    def get_qpos(self):
        return self._qpos.copy()

    def get_qvel(self):
        return self._qvel.copy()

    def set_qpos(self, values):
        self._qpos = np.asarray(values, dtype=np.float64).copy()

    def set_qvel(self, values):
        self._qvel = np.asarray(values, dtype=np.float64).copy()


class FakeRobot:
    def __init__(self):
        self.left_arm_joints = [FakeJoint("fl_joint1"), FakeJoint("fl_joint2")]
        self.right_arm_joints = [FakeJoint("fr_joint1"), FakeJoint("fr_joint2")]
        self.left_entity = FakeEntity(self.left_arm_joints)
        self.right_entity = FakeEntity(self.right_arm_joints)
        self.left_camera = FakePoseOwner(FakePose([-0.1, 0.0, 0.0]))
        self.right_camera = FakePoseOwner(FakePose([0.1, 0.0, 0.0]))
        self.commands = []

    def set_arm_joints(self, target_position, target_velocity, arm_tag):
        position = np.asarray(target_position, dtype=np.float64)
        velocity = np.asarray(target_velocity, dtype=np.float64)
        entity = self.left_entity if arm_tag == "left" else self.right_entity
        entity.set_qpos(position)
        entity.set_qvel(velocity)
        self.commands.append((arm_tag, position.copy(), velocity.copy()))


class FakeTask:
    def __init__(self):
        self.robot = FakeRobot()
        self.cameras = FakeCameras()
        self.now_obs = None
        self.render_updates = 0

    def _update_render(self):
        self.render_updates += 1
        self.cameras.update_wrist_camera(
            self.robot.left_camera.get_pose(),
            self.robot.right_camera.get_pose(),
        )

    def get_obs(self):
        payload = {
            "observation": {
                name: {"rgb": np.full((8, 8, 3), 100, dtype=np.uint8)}
                for name in ("head_camera", "left_camera", "right_camera")
            }
        }
        self.now_obs = copy.deepcopy(payload)
        return payload


def pipeline(faults, *, enabled=True):
    return {
        "pipeline": {
            "id": "test_robotwin",
            "enabled": enabled,
            "seed": 7,
            "faults": faults,
        }
    }


class RoboTwinFaultTests(unittest.TestCase):
    def test_disabled_pipeline_preserves_environment_methods(self):
        task = FakeTask()
        get_obs = task.get_obs
        set_arm_joints = task.robot.set_arm_joints
        controller = install_fault_pipeline(task, pipeline([], enabled=False))
        self.assertIsNone(controller)
        self.assertEqual(task.get_obs, get_obs)
        self.assertEqual(task.robot.set_arm_joints, set_arm_joints)

    def test_optical_fault_transforms_nested_robotwin_observations(self):
        task = FakeTask()
        config = pipeline(
            [
                {
                    "id": "dark_left",
                    "family": "visual.illumination",
                    "targets": ["left_camera"],
                    "operation": {"type": "affine_color_response"},
                    "severity": {"name": "intensity_gain", "value": 0.2, "unit": "ratio"},
                    "parameters": {
                        "color_matrix": [[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
                        "color_bias": [0, 0, 0],
                    },
                }
            ]
        )
        controller = install_fault_pipeline(task, config)
        self.assertIsNotNone(controller)
        observation = task.get_obs()
        self.assertTrue(np.all(observation["observation"]["left_camera"]["rgb"] == 20))
        self.assertTrue(np.all(observation["observation"]["head_camera"]["rgb"] == 100))
        self.assertEqual(task.now_obs["observation"]["left_camera"]["rgb"].mean(), 20)
        controller.detach()
        self.assertTrue(np.all(task.get_obs()["observation"]["left_camera"]["rgb"] == 100))

    def test_camera_extrinsics_are_applied_and_restored(self):
        task = FakeTask()
        config = pipeline(
            [
                {
                    "id": "head_shift",
                    "family": "visual.camera_pose",
                    "target": "head_camera",
                    "operation": {"type": "pose_offset"},
                    "severity": {"name": "translation_norm", "value": 0.1, "unit": "metre"},
                    "parameters": {
                        "position_offset": [0.1, 0.0, 0.0],
                        "rotation_offset_deg": [0.0, 0.0, 0.0],
                    },
                },
                {
                    "id": "left_roll",
                    "family": "visual.camera_pose",
                    "target": "left_camera",
                    "operation": {"type": "local_axis_rotation", "axis": "z"},
                    "severity": {"name": "rotation_angle", "value": 90.0, "unit": "degree"},
                    "parameters": {"position_offset": [0.0, 0.0, 0.0]},
                },
            ]
        )
        controller = install_fault_pipeline(task, config)
        head_pose = task.cameras.static_camera_list[0].entity.get_pose()
        left_pose = task.cameras.left_camera.entity.get_pose()
        np.testing.assert_allclose(head_pose.p, [0.1, 0.0, 0.0], atol=1e-8)
        self.assertAlmostEqual(abs(left_pose.q[3]), math.sqrt(0.5), places=6)
        self.assertEqual(
            controller.metadata()["faults"][0]["injection_layer"],
            "sapien_camera_extrinsics",
        )
        controller.detach()
        np.testing.assert_allclose(
            task.cameras.static_camera_list[0].entity.get_pose().p,
            [0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(task.cameras.left_camera.entity.get_pose().q, [1, 0, 0, 0])

    def test_left_motion_retention_modifies_sapien_drive_target(self):
        task = FakeTask()
        config = pipeline(
            [
                {
                    "id": "left_retention",
                    "family": "structure.joint_motion",
                    "targets": ["left_arm_joint1"],
                    "operation": {"type": "proportional"},
                    "severity": {"name": "motion_retention", "value": 0.2, "unit": "ratio"},
                }
            ]
        )
        controller = install_fault_pipeline(task, config)
        task.robot.set_arm_joints([1.0, 0.5], [2.0, 3.0], "left")
        np.testing.assert_allclose(task.robot.left_entity.get_qpos(), [0.2, 0.5])
        np.testing.assert_allclose(task.robot.left_entity.get_qvel(), [0.4, 3.0])
        task.robot.set_arm_joints([1.0, 0.5], [2.0, 3.0], "right")
        np.testing.assert_allclose(task.robot.right_entity.get_qpos(), [1.0, 0.5])
        self.assertEqual(
            controller.metadata()["faults"][0]["injection_layer"],
            "sapien_articulation_drive_target",
        )

    def test_two_arm_faults_are_isolated_on_one_dual_arm_articulation(self):
        task = FakeTask()
        robot = task.robot
        combined_joints = [*robot.left_arm_joints, *robot.right_arm_joints]
        combined_entity = FakeEntity(combined_joints)
        robot.left_entity = combined_entity
        robot.right_entity = combined_entity

        def set_dual_arm_joints(target_position, target_velocity, arm_tag):
            position = combined_entity.get_qpos()
            velocity = combined_entity.get_qvel()
            offset = 0 if arm_tag == "left" else len(robot.left_arm_joints)
            indices = np.arange(offset, offset + len(target_position))
            position[indices] = target_position
            velocity[indices] = target_velocity
            combined_entity.set_qpos(position)
            combined_entity.set_qvel(velocity)

        robot.set_arm_joints = set_dual_arm_joints
        config = pipeline(
            [
                {
                    "id": "left_retention",
                    "family": "structure.joint_motion",
                    "targets": ["left_arm_joint1"],
                    "operation": {"type": "proportional"},
                    "severity": {"name": "motion_retention", "value": 0.5, "unit": "ratio"},
                },
                {
                    "id": "right_retention",
                    "family": "structure.joint_motion",
                    "targets": ["right_arm_joint1"],
                    "operation": {"type": "proportional"},
                    "severity": {"name": "motion_retention", "value": 0.25, "unit": "ratio"},
                },
            ]
        )
        install_fault_pipeline(task, config)
        robot.set_arm_joints([1.0, 0.2], [0.0, 0.0], "left")
        robot.set_arm_joints([1.0, -0.2], [0.0, 0.0], "right")
        np.testing.assert_allclose(combined_entity.get_qpos(), [0.5, 0.2, 0.25, -0.2])

    def test_one_stateful_fault_cannot_share_history_across_arms(self):
        task = FakeTask()
        config = pipeline(
            [
                {
                    "id": "invalid_shared_backlash",
                    "family": "structure.joint_backlash",
                    "targets": ["left_arm_joint1", "right_arm_joint1"],
                    "operation": {"type": "reversal_dead_travel"},
                    "severity": {"name": "backlash_gap", "value": 5.0, "unit": "degree"},
                    "parameters": {"gap_deg": 5.0},
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "spans both arms"):
            install_fault_pipeline(task, config)

    def test_right_alias_and_runtime_state_round_trip(self):
        task = FakeTask()
        config = pipeline(
            [
                {
                    "id": "right_backlash",
                    "family": "structure.joint_backlash",
                    "targets": ["right_arm_joint1"],
                    "operation": {"type": "reversal_dead_travel"},
                    "severity": {"name": "backlash_gap", "value": 5.0, "unit": "degree"},
                    "parameters": {"gap_deg": 5.0},
                }
            ]
        )
        controller = install_fault_pipeline(task, config)
        task.robot.set_arm_joints([0.2, 0.0], [0.0, 0.0], "right")
        task.robot.set_arm_joints([-0.2, 0.0], [0.0, 0.0], "right")
        state = controller.state_dict()
        self.assertEqual(state["arm_step_indices"]["right"], 2)
        controller.load_state_dict(state)
        self.assertEqual(controller.state_dict()["arm_step_indices"]["right"], 2)
        self.assertEqual(
            controller.last_joint_diagnostics[-1]["targets"],
            ["right_arm_joint1"],
        )
        self.assertIn("right_backlash", state["backlash_command_states"])
        np.testing.assert_allclose(
            state["backlash_command_states"]["right_backlash"]["nominal"],
            [-0.2],
        )

    def test_backlash_consumes_incremental_reverse_drive_targets(self):
        task = FakeTask()
        config = pipeline(
            [
                {
                    "id": "left_backlash",
                    "family": "structure.joint_backlash",
                    "targets": ["left_arm_joint1"],
                    "operation": {"type": "reversal_dead_travel"},
                    "severity": {"name": "backlash_gap", "value": 6.0, "unit": "degree"},
                    "parameters": {"gap_deg": 6.0},
                }
            ]
        )
        install_fault_pipeline(task, config)
        task.robot.set_arm_joints([0.2, 0.0], [0.0, 0.0], "left")
        task.robot.set_arm_joints([0.1, 0.0], [0.0, 0.0], "left")
        first_reverse_target = task.robot.commands[-1][1][0]
        self.assertAlmostEqual(first_reverse_target, 0.2, places=7)
        task.robot.set_arm_joints([0.0, 0.0], [0.0, 0.0], "left")
        self.assertLess(task.robot.commands[-1][1][0], first_reverse_target)

    def test_all_robotwin_catalog_configs_construct(self):
        config_root = PROJECT_ROOT / "configs" / "fault" / "robotwin" / "catalog"
        paths = sorted(config_root.glob("*/*.yaml"))
        self.assertEqual(len(paths), 10)
        for path in paths:
            task = FakeTask()
            config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
            if "structure" in path.parts:
                config = copy.deepcopy(config)
                config["pipeline"]["faults"][0]["targets"] = ["left_arm_joint1"]
            controller = install_fault_pipeline(task, config)
            self.assertIsNotNone(controller, path)
            controller.detach()

    def test_hydra_default_keeps_robotwin_faults_disabled(self):
        config = OmegaConf.load(PROJECT_ROOT / "configs" / "sim_robotwin.yaml")
        self.assertIsNone(config.EVALUATION.fault_config)

    def test_upstream_evaluator_forwards_fault_runtime_arguments(self):
        source = (
            PROJECT_ROOT / "third_party" / "RoboTwin" / "script" / "eval_policy.py"
        ).read_text(encoding="utf-8")
        self.assertIn('fault_config = usr_args.get("fault_config")', source)
        self.assertIn('args["fault_config"] = str(fault_config)', source)
        self.assertIn('args["sim_cfg_path"] = str(sim_cfg_path)', source)
        self.assertIn("install_faults()", source)


if __name__ == "__main__":
    unittest.main()
