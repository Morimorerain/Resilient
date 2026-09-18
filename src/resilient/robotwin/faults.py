"""RoboTwin simulator adapters for the shared Resilient Fault catalog."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from resilient.faults import build_fault_pipeline
from resilient.faults.base import FaultContext
from resilient.faults.pipeline import FaultPipeline
from resilient.faults.structure.joint_backlash import JointBacklashFaultRuntime
from resilient.faults.structure.joint_motion import JointMotionFaultRuntime
from resilient.faults.structure.joint_position_bias import JointPositionBiasFaultRuntime
from resilient.faults.structure.joint_state import JointStateFaultRuntime
from resilient.faults.structure.periodic_joint_freeze import (
    PeriodicJointFreezeFaultRuntime,
)
from resilient.faults.visual.camera_pose import (
    CameraPoseFaultRuntime,
    local_xyz_offset_quaternion,
    multiply_quaternions,
    rotate_vector,
)
from resilient.faults.visual.image_sensor import ImageSensorFaultRuntime

ROBOTWIN_CAMERAS = ("head_camera", "left_camera", "right_camera")
ROBOTWIN_ARMS = ("left", "right")


def load_fault_config(path: str | Path) -> dict[str, Any]:
    """Load one resolved Fault YAML without retaining machine-specific paths."""
    payload = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("RoboTwin Fault configuration must resolve to a mapping.")
    return payload


def _joint_name(joint: Any) -> str:
    getter = getattr(joint, "get_name", None)
    if callable(getter):
        return str(getter())
    name = getattr(joint, "name", None)
    if name is None:
        raise AttributeError("RoboTwin articulation joint does not expose a name.")
    return str(name)


def _pose_components(pose: Any) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(pose.p, dtype=np.float64)
    quaternion = np.asarray(pose.q, dtype=np.float64)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("SAPIEN camera pose must contain xyz and wxyz quaternion values.")
    return position.copy(), quaternion.copy()


def _new_pose(reference: Any, position: np.ndarray, quaternion: np.ndarray) -> Any:
    pose_type = type(reference)
    try:
        return pose_type(position.tolist(), quaternion.tolist())
    except TypeError:
        return pose_type(p=position.tolist(), q=quaternion.tolist())


def _offset_pose(pose: Any, fault: CameraPoseFaultRuntime) -> Any:
    position, quaternion = _pose_components(pose)
    world_offset = np.asarray(rotate_vector(quaternion, fault.position_offset))
    offset_quaternion = local_xyz_offset_quaternion(fault.rotation_offset_deg)
    effective_quaternion = np.asarray(
        multiply_quaternions(quaternion, offset_quaternion), dtype=np.float64
    )
    return _new_pose(pose, position + world_offset, effective_quaternion)


@dataclass(frozen=True)
class ResolvedJointTarget:
    """One canonical Fault target resolved to a RoboTwin arm joint."""

    configured_name: str
    arm: str
    arm_index: int
    simulator_name: str


class RoboTwinFaultController:
    """Install shared Faults at RoboTwin's SAPIEN sensor and actuator boundaries."""

    def __init__(self, task_environment: Any, pipeline: FaultPipeline) -> None:
        if not pipeline.enabled:
            raise ValueError("RoboTwinFaultController requires an enabled Fault pipeline.")
        self.task_environment = task_environment
        self.pipeline = pipeline
        self._original_get_obs: Any | None = None
        self._original_set_arm_joints: Any | None = None
        self._original_update_wrist_camera: Any | None = None
        self._camera_original_poses: dict[str, tuple[Any, np.ndarray, np.ndarray]] = {}
        self._joint_targets: dict[str, tuple[ResolvedJointTarget, ...]] = {}
        self._motion_command_states: dict[str, dict[str, np.ndarray]] = {}
        self._backlash_command_states: dict[str, dict[str, np.ndarray]] = {}
        self._periodic_observation_states: dict[str, np.ndarray] = {}
        self._arm_step_indices = {arm: 0 for arm in ROBOTWIN_ARMS}
        self.last_joint_diagnostics: list[dict[str, Any]] = []
        self._attached = False

    def _context(self, arm: str | None = None) -> FaultContext:
        step_index = 0 if arm is None else self._arm_step_indices[arm]
        return self.pipeline.context(episode_index=0, step_index=step_index)

    def _arm_joints(self, arm: str) -> Sequence[Any]:
        if arm not in ROBOTWIN_ARMS:
            raise ValueError(f"RoboTwin arm must be one of {ROBOTWIN_ARMS}, got {arm!r}.")
        return tuple(getattr(self.task_environment.robot, f"{arm}_arm_joints"))

    def _resolve_joint_target(self, configured_name: str) -> ResolvedJointTarget:
        aliases: dict[str, ResolvedJointTarget] = {}
        exact: dict[str, list[ResolvedJointTarget]] = {}
        for arm in ROBOTWIN_ARMS:
            for index, joint in enumerate(self._arm_joints(arm)):
                simulator_name = _joint_name(joint)
                target = ResolvedJointTarget(
                    configured_name=configured_name,
                    arm=arm,
                    arm_index=index,
                    simulator_name=simulator_name,
                )
                aliases[f"{arm}_arm_joint{index + 1}"] = target
                exact.setdefault(simulator_name, []).append(target)
        if configured_name in aliases:
            return aliases[configured_name]
        matches = exact.get(configured_name, [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Joint name {configured_name!r} exists on both arms; use "
                "left_arm_jointN or right_arm_jointN."
            )
        known = sorted(aliases)
        raise ValueError(
            f"Unknown RoboTwin joint target {configured_name!r}; canonical targets: {known}."
        )

    def _prepare_joint_faults(self) -> None:
        for fault in self.pipeline.faults:
            if not isinstance(fault, JointMotionFaultRuntime | JointStateFaultRuntime):
                continue
            targets = tuple(self._resolve_joint_target(name) for name in fault.joint_names)
            arms = {target.arm for target in targets}
            if len(arms) != 1:
                raise ValueError(
                    f"RoboTwin Fault {fault.fault_id!r} spans both arms. Define one Fault "
                    "entry per arm so stateful actuator histories remain independent."
                )
            fault.reset_runtime() if isinstance(fault, JointStateFaultRuntime) else None
            self._joint_targets[fault.fault_id] = targets

    def _arm_real_qpos(self, arm: str) -> np.ndarray:
        robot = self.task_environment.robot
        entity = getattr(robot, f"{arm}_entity")
        active_joints = list(entity.get_active_joints())
        qpos = np.asarray(entity.get_qpos(), dtype=np.float64)
        indices = [active_joints.index(joint) for joint in self._arm_joints(arm)]
        return qpos[np.asarray(indices, dtype=np.int64)].copy()

    def _transform_arm_command(
        self,
        target_position: Any,
        target_velocity: Any,
        arm: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(target_position, dtype=np.float64).copy()
        velocity = np.asarray(target_velocity, dtype=np.float64).copy()
        expected = len(self._arm_joints(arm))
        if position.shape != (expected,) or velocity.shape != (expected,):
            raise ValueError(
                f"RoboTwin {arm} arm command must be {expected}D, got "
                f"{position.shape} and {velocity.shape}."
            )
        current = self._arm_real_qpos(arm)
        context = self._context(arm)
        for fault in self.pipeline.faults:
            targets = self._joint_targets.get(fault.fault_id)
            if not targets or targets[0].arm != arm:
                continue
            indices = np.asarray([target.arm_index for target in targets], dtype=np.int64)
            before = current[indices]
            candidate = position[indices]
            candidate_velocity = velocity[indices]
            if isinstance(fault, JointMotionFaultRuntime):
                # RoboTwin sends an absolute drive target at every control update. Scale each
                # new nominal command increment exactly once; repeatedly sending an unchanged
                # target must not let a degraded actuator asymptotically catch up.
                command_state = self._motion_command_states.get(fault.fault_id)
                if command_state is None:
                    nominal_before = before
                    applied_before = before
                else:
                    nominal_before = command_state["nominal"]
                    applied_before = command_state["applied"]
                nominal_increment = candidate - nominal_before
                applied_increment, applied_velocity = fault.law.apply(
                    nominal_increment,
                    candidate_velocity,
                    context=context,
                )
                applied = applied_before + np.asarray(applied_increment, dtype=np.float64)
                self._motion_command_states[fault.fault_id] = {
                    "nominal": candidate.copy(),
                    "applied": np.asarray(applied, dtype=np.float64).copy(),
                }
                diagnostics = {
                    "nominal_increment_deg": np.rad2deg(nominal_increment).tolist(),
                    "applied_increment_deg": np.rad2deg(applied_increment).tolist(),
                    "nominal_target_error_deg": np.rad2deg(candidate - before).tolist(),
                    "applied_target_error_deg": np.rad2deg(applied - before).tolist(),
                }
            elif isinstance(fault, JointPositionBiasFaultRuntime):
                # A fixed encoder-zero error offsets every physical actuator target. Applying
                # it once to simulator qpos (LIBERO) and on every drive target (RoboTwin) are
                # the corresponding non-cumulative implementations at each backend boundary.
                applied = candidate + fault.bias_rad
                applied_velocity = candidate_velocity
                diagnostics = {"bias_deg": np.rad2deg(fault.bias_rad).tolist()}
            elif isinstance(fault, JointBacklashFaultRuntime):
                # RoboTwin writes absolute articulation drive targets. Backlash must consume
                # successive command increments after a reversal, rather than repeatedly using
                # the full target-to-physical-state error, which would collapse the dead travel.
                command_state = self._backlash_command_states.get(fault.fault_id)
                if command_state is None:
                    applied, applied_velocity, diagnostics = fault.transform_realized_state(
                        before,
                        candidate,
                        candidate_velocity,
                        context=context,
                    )
                    diagnostics["command_history_initialized"] = True
                else:
                    nominal_increment = candidate - command_state["nominal"]
                    applied_before = command_state["applied"]
                    applied_candidate = applied_before + nominal_increment
                    applied, applied_velocity, diagnostics = fault.transform_realized_state(
                        applied_before,
                        applied_candidate,
                        candidate_velocity,
                        context=context,
                    )
                self._backlash_command_states[fault.fault_id] = {
                    "nominal": candidate.copy(),
                    "applied": np.asarray(applied, dtype=np.float64).copy(),
                }
            elif isinstance(fault, PeriodicJointFreezeFaultRuntime):
                previous_actual = self._periodic_observation_states.get(fault.fault_id)
                if previous_actual is None:
                    applied = candidate
                    applied_velocity = candidate_velocity
                    diagnostics = {
                        "triggered": [False] * len(indices),
                        "remaining_hold_steps": [0] * len(indices),
                        "hold_position_deg": np.rad2deg(before).tolist(),
                        "last_trigger_index": [None] * len(indices),
                        "trigger_source": "realized_joint_motion_initialized",
                    }
                else:
                    applied, applied_velocity, diagnostics = (
                        fault.transform_drive_target_from_observed_motion(
                            previous_actual,
                            before,
                            candidate,
                            candidate_velocity,
                            context=context,
                        )
                    )
                self._periodic_observation_states[fault.fault_id] = before.copy()
            elif isinstance(fault, JointStateFaultRuntime):
                applied, applied_velocity, diagnostics = fault.transform_realized_state(
                    before,
                    candidate,
                    candidate_velocity,
                    context=context,
                )
            else:  # pragma: no cover - guarded by _prepare_joint_faults
                continue
            applied = np.asarray(applied, dtype=np.float64)
            applied_velocity = np.asarray(applied_velocity, dtype=np.float64)
            if (
                applied.shape != candidate.shape
                or applied_velocity.shape != candidate_velocity.shape
                or not np.all(np.isfinite(applied))
                or not np.all(np.isfinite(applied_velocity))
            ):
                raise ValueError(f"Fault {fault.fault_id!r} produced an invalid arm command.")
            position[indices] = applied
            velocity[indices] = applied_velocity
            self.last_joint_diagnostics.append(
                {
                    "fault_id": fault.fault_id,
                    "family": fault.family,
                    "arm": arm,
                    "targets": [target.configured_name for target in targets],
                    "simulator_joints": [target.simulator_name for target in targets],
                    "command_index": self._arm_step_indices[arm],
                    **diagnostics,
                }
            )
        self._arm_step_indices[arm] += 1
        return position, velocity

    def _camera_object(self, camera_name: str) -> Any:
        cameras = self.task_environment.cameras
        if camera_name in {"left_camera", "right_camera"}:
            return getattr(cameras, camera_name)
        for name, camera in zip(
            cameras.static_camera_name, cameras.static_camera_list, strict=True
        ):
            if name == camera_name:
                return camera
        raise ValueError(f"RoboTwin camera {camera_name!r} is not active in this environment.")

    def _camera_pose_faults(self) -> dict[str, list[CameraPoseFaultRuntime]]:
        result = {camera: [] for camera in ROBOTWIN_CAMERAS}
        for fault in self.pipeline.faults:
            if isinstance(fault, CameraPoseFaultRuntime):
                if fault.camera not in result:
                    raise ValueError(
                        f"Camera-pose target {fault.camera!r} is not a RoboTwin camera."
                    )
                result[fault.camera].append(fault)
        return {name: faults for name, faults in result.items() if faults}

    @staticmethod
    def _apply_pose_faults(pose: Any, faults: Sequence[CameraPoseFaultRuntime]) -> Any:
        transformed = pose
        for fault in faults:
            transformed = _offset_pose(transformed, fault)
        return transformed

    def _install_camera_pose_faults(self) -> None:
        pose_faults = self._camera_pose_faults()
        if not pose_faults:
            return
        for camera_name, faults in pose_faults.items():
            camera = self._camera_object(camera_name)
            pose = camera.entity.get_pose()
            position, quaternion = _pose_components(pose)
            self._camera_original_poses[camera_name] = (pose, position, quaternion)
            if camera_name == "head_camera":
                camera.entity.set_pose(self._apply_pose_faults(pose, faults))

        wrist_faults = {
            name: faults
            for name, faults in pose_faults.items()
            if name in {"left_camera", "right_camera"}
        }
        if wrist_faults:
            cameras = self.task_environment.cameras
            self._original_update_wrist_camera = cameras.update_wrist_camera
            original = self._original_update_wrist_camera

            def update_wrist_camera(left_pose: Any, right_pose: Any) -> None:
                if "left_camera" in wrist_faults:
                    left_pose = self._apply_pose_faults(left_pose, wrist_faults["left_camera"])
                if "right_camera" in wrist_faults:
                    right_pose = self._apply_pose_faults(right_pose, wrist_faults["right_camera"])
                original(left_pose, right_pose)

            cameras.update_wrist_camera = update_wrist_camera
        self.task_environment._update_render()

    def _install_observation_faults(self) -> None:
        if not any(isinstance(fault, ImageSensorFaultRuntime) for fault in self.pipeline.faults):
            return
        self._original_get_obs = self.task_environment.get_obs
        original = self._original_get_obs

        def get_obs() -> Any:
            raw = original()
            transformed = self.pipeline.transform_observation(
                raw,
                self.task_environment,
                self._context(),
            )
            self.task_environment.now_obs = copy.deepcopy(transformed)
            return transformed

        self.task_environment.get_obs = get_obs

    def _install_actuator_faults(self) -> None:
        if not self._joint_targets:
            return
        robot = self.task_environment.robot
        self._original_set_arm_joints = robot.set_arm_joints
        original = self._original_set_arm_joints

        def set_arm_joints(
            target_position: Any,
            target_velocity: Any,
            arm_tag: str,
        ) -> Any:
            position, velocity = self._transform_arm_command(
                target_position,
                target_velocity,
                str(arm_tag),
            )
            return original(position, velocity, arm_tag)

        robot.set_arm_joints = set_arm_joints

    def attach(self) -> RoboTwinFaultController:
        """Install all configured adapters after one RoboTwin scene is created."""
        if self._attached:
            return self
        self._prepare_joint_faults()
        supported = (
            CameraPoseFaultRuntime,
            ImageSensorFaultRuntime,
            JointMotionFaultRuntime,
            JointStateFaultRuntime,
        )
        unsupported = [
            fault.family for fault in self.pipeline.faults if not isinstance(fault, supported)
        ]
        if unsupported:
            raise TypeError(f"Fault families lack a RoboTwin adapter: {unsupported}.")
        self._attached = True
        try:
            self._install_camera_pose_faults()
            self._install_observation_faults()
            self._install_actuator_faults()
            setattr(self.task_environment, "_resilient_fault_controller", self)
        except Exception:
            self.detach()
            raise
        return self

    def metadata(self) -> dict[str, Any]:
        """Return portable metadata with the concrete RoboTwin injection layer."""
        payload = copy.deepcopy(self.pipeline.metadata())
        payload["simulator"] = "robotwin_sapien"
        for fault in payload["faults"]:
            family = str(fault["family"])
            if family == "visual.camera_pose":
                fault["injection_layer"] = "sapien_camera_extrinsics"
            elif family.startswith("visual."):
                fault["injection_layer"] = "robotwin_environment_camera_sensor"
            elif family.startswith("structure."):
                fault["injection_layer"] = "sapien_articulation_drive_target"
        return payload

    def state_dict(self) -> dict[str, Any]:
        """Return stateful fault history for deterministic continuation."""
        return {
            "schema_version": 1,
            "pipeline": copy.deepcopy(self.pipeline.state_dict()),
            "arm_step_indices": dict(self._arm_step_indices),
            "motion_command_states": {
                fault_id: {name: value.tolist() for name, value in command_state.items()}
                for fault_id, command_state in self._motion_command_states.items()
            },
            "backlash_command_states": {
                fault_id: {name: value.tolist() for name, value in command_state.items()}
                for fault_id, command_state in self._backlash_command_states.items()
            },
            "periodic_observation_states": {
                fault_id: value.tolist()
                for fault_id, value in self._periodic_observation_states.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported RoboTwin Fault runtime-state schema.")
        self.pipeline.load_state_dict(copy.deepcopy(dict(state["pipeline"])))
        indices = {arm: int(state["arm_step_indices"][arm]) for arm in ROBOTWIN_ARMS}
        if min(indices.values()) < 0:
            raise ValueError("RoboTwin Fault command indices must be non-negative.")
        self._arm_step_indices = indices
        self._motion_command_states = self._restore_command_states(
            state.get("motion_command_states", {}),
            runtime_type=JointMotionFaultRuntime,
            state_label="motion",
        )
        self._backlash_command_states = self._restore_command_states(
            state.get("backlash_command_states", {}),
            runtime_type=JointBacklashFaultRuntime,
            state_label="backlash",
        )
        periodic_runtimes = {
            fault.fault_id
            for fault in self.pipeline.faults
            if isinstance(fault, PeriodicJointFreezeFaultRuntime)
        }
        restored_periodic: dict[str, np.ndarray] = {}
        for fault_id, values in state.get("periodic_observation_states", {}).items():
            if fault_id not in periodic_runtimes or fault_id not in self._joint_targets:
                raise ValueError(f"Unknown RoboTwin periodic state: {fault_id!r}.")
            expected = len(self._joint_targets[fault_id])
            actual = np.asarray(values, dtype=np.float64)
            if actual.shape != (expected,) or not np.all(np.isfinite(actual)):
                raise ValueError(f"RoboTwin periodic state mismatch: {fault_id!r}.")
            restored_periodic[str(fault_id)] = actual.copy()
        self._periodic_observation_states = restored_periodic

    def _restore_command_states(
        self,
        payload: Mapping[str, Any],
        *,
        runtime_type: type[JointMotionFaultRuntime] | type[JointBacklashFaultRuntime],
        state_label: str,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Validate target-history state used by absolute-drive-target adapters."""
        restored: dict[str, dict[str, np.ndarray]] = {}
        runtimes = {
            fault.fault_id: fault
            for fault in self.pipeline.faults
            if isinstance(fault, runtime_type)
        }
        for fault_id, command_state in payload.items():
            if fault_id not in self._joint_targets or fault_id not in runtimes:
                raise ValueError(f"Unknown RoboTwin {state_label} state: {fault_id!r}.")
            expected = len(self._joint_targets[fault_id])
            nominal = np.asarray(command_state["nominal"], dtype=np.float64)
            applied = np.asarray(command_state["applied"], dtype=np.float64)
            if (
                nominal.shape != (expected,)
                or applied.shape != (expected,)
                or not np.all(np.isfinite(nominal))
                or not np.all(np.isfinite(applied))
            ):
                raise ValueError(
                    f"RoboTwin {state_label} state shape/value mismatch: {fault_id!r}."
                )
            restored[str(fault_id)] = {
                "nominal": nominal.copy(),
                "applied": applied.copy(),
            }
        return restored

    def detach(self) -> None:
        """Restore unmodified RoboTwin environment methods and camera poses."""
        if not self._attached:
            return
        if self._original_get_obs is not None:
            self.task_environment.get_obs = self._original_get_obs
        if self._original_set_arm_joints is not None:
            self.task_environment.robot.set_arm_joints = self._original_set_arm_joints
        cameras = self.task_environment.cameras
        if self._original_update_wrist_camera is not None:
            cameras.update_wrist_camera = self._original_update_wrist_camera
        for camera_name, (reference, position, quaternion) in self._camera_original_poses.items():
            camera = self._camera_object(camera_name)
            camera.entity.set_pose(_new_pose(reference, position, quaternion))
        if self._original_update_wrist_camera is not None:
            self._original_update_wrist_camera(
                self.task_environment.robot.left_camera.get_pose(),
                self.task_environment.robot.right_camera.get_pose(),
            )
        if getattr(self.task_environment, "_resilient_fault_controller", None) is self:
            delattr(self.task_environment, "_resilient_fault_controller")
        self._attached = False


def install_fault_pipeline(
    task_environment: Any,
    config: Mapping[str, Any] | str | Path | None,
) -> RoboTwinFaultController | None:
    """Install a configured pipeline after RoboTwin ``setup_demo`` completes."""
    if config is None or (isinstance(config, str) and not config.strip()):
        return None
    mapping = load_fault_config(config) if isinstance(config, str | Path) else dict(config)
    pipeline = build_fault_pipeline(mapping)
    if not pipeline.enabled:
        return None
    controller = RoboTwinFaultController(task_environment, pipeline)
    return controller.attach()


def detach_fault_pipeline(task_environment: Any) -> None:
    """Detach an installed pipeline if present; otherwise preserve upstream behavior."""
    controller = getattr(task_environment, "_resilient_fault_controller", None)
    if controller is not None:
        controller.detach()


def metadata_json(controller: RoboTwinFaultController) -> str:
    """Serialize resolved metadata for manifests and human-readable logs."""
    return json.dumps(controller.metadata(), sort_keys=True, ensure_ascii=False)
