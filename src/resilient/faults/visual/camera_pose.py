"""Severity-aware, reversible LIBERO camera-pose fault."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from ..base import FaultContext, FaultRuntime
from ..types import SeveritySpec

SUPPORTED_CAMERAS = ("agentview", "robot0_eye_in_hand")
SUPPORTED_AXES = ("x", "y", "z")


def _as_triplet(values: Sequence[float], name: str) -> tuple[float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain exactly three finite values.")
    return result


def _normalize_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    quaternion = tuple(float(value) for value in values)
    if len(quaternion) != 4:
        raise ValueError("Quaternion must contain four values.")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-12:
        raise ValueError("Quaternion must have non-zero length.")
    return tuple(value / norm for value in quaternion)


def multiply_quaternions(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float]:
    """Multiply MuJoCo quaternions in ``wxyz`` order."""
    lw, lx, ly, lz = _normalize_quaternion(left)
    rw, rx, ry, rz = _normalize_quaternion(right)
    return _normalize_quaternion(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def rotate_vector(
    quaternion: Sequence[float], vector: Sequence[float]
) -> tuple[float, float, float]:
    """Rotate a three-vector by a ``wxyz`` quaternion."""
    w, x, y, z = _normalize_quaternion(quaternion)
    vx, vy, vz = _as_triplet(vector, "vector")
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def local_xyz_offset_quaternion(
    rotation_offset_deg: Sequence[float],
) -> tuple[float, float, float, float]:
    """Return an X-then-Y-then-Z local rotation in ``wxyz`` order."""
    rotation = _as_triplet(rotation_offset_deg, "rotation_offset_deg")
    offset = (1.0, 0.0, 0.0, 0.0)
    for axis, angle_deg in enumerate(rotation):
        half_angle = math.radians(angle_deg) / 2.0
        values = [math.cos(half_angle), 0.0, 0.0, 0.0]
        values[axis + 1] = math.sin(half_angle)
        offset = multiply_quaternions(offset, values)
    return offset


def _encode_number(value: float, precision: int = 1) -> str:
    normalized = 0.0 if abs(value) < 0.5 * 10 ** (-precision) else value
    sign = "p" if normalized >= 0 else "m"
    return sign + f"{abs(normalized):.{precision}f}".replace(".", "p")


class CameraPoseFaultRuntime(FaultRuntime):
    """Apply one camera-local pose offset and restore it without accumulation."""

    family = "visual.camera_pose"
    scopes = frozenset({"visual", "observation", "environment"})

    def __init__(
        self,
        *,
        fault_id: str,
        camera: str,
        position_offset: Sequence[float],
        rotation_offset_deg: Sequence[float],
        severity: SeveritySpec,
        operation: Mapping[str, Any],
    ):
        if camera not in SUPPORTED_CAMERAS:
            raise ValueError(f"Unsupported camera {camera!r}; expected one of {SUPPORTED_CAMERAS}.")
        self.fault_id = str(fault_id)
        self.camera = camera
        self.position_offset = _as_triplet(position_offset, "position_offset")
        self.rotation_offset_deg = _as_triplet(rotation_offset_deg, "rotation_offset_deg")
        self.severity = severity
        self.operation = dict(operation)
        self._model: Any | None = None
        self._camera_id: int | None = None
        self._original_position: tuple[float, float, float] | None = None
        self._original_quaternion: tuple[float, float, float, float] | None = None
        self._active = False
        self._last_metadata: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> CameraPoseFaultRuntime:
        operation = dict(config.get("operation", {}))
        severity = SeveritySpec.from_config(dict(config["severity"]))
        operation_type = str(operation.get("type", "pose_offset"))
        parameters = dict(config.get("parameters", {}))
        position_offset = parameters.get("position_offset", (0.0, 0.0, 0.0))
        if operation_type == "local_axis_rotation":
            if severity.unit not in {"degree", "degrees", "deg"}:
                raise ValueError("local_axis_rotation severity unit must be degree.")
            axis = str(operation.get("axis", "")).lower()
            if axis not in SUPPORTED_AXES:
                raise ValueError(f"Rotation axis must be one of {SUPPORTED_AXES}.")
            direction = str(operation.get("direction", "positive")).lower()
            if direction not in {"positive", "negative"}:
                raise ValueError("Rotation direction must be positive or negative.")
            angle = severity.value if direction == "positive" else -severity.value
            rotation = [0.0, 0.0, 0.0]
            rotation[SUPPORTED_AXES.index(axis)] = angle
            rotation_offset_deg = rotation
        elif operation_type == "pose_offset":
            rotation_offset_deg = parameters.get("rotation_offset_deg", (0.0, 0.0, 0.0))
        else:
            raise ValueError(f"Unsupported camera-pose operation: {operation_type}")
        return cls(
            fault_id=str(config.get("id", "camera_pose")),
            camera=str(config.get("target", parameters.get("camera", "agentview"))),
            position_offset=position_offset,
            rotation_offset_deg=rotation_offset_deg,
            severity=severity,
            operation=operation,
        )

    @staticmethod
    def _get_sim(env: Any) -> Any:
        if hasattr(env, "sim"):
            return env.sim
        if hasattr(env, "env") and hasattr(env.env, "sim"):
            return env.env.sim
        raise AttributeError("Environment does not expose a MuJoCo simulation object.")

    def _capture_original(self, env: Any) -> Any:
        sim = self._get_sim(env)
        model = sim.model
        camera_id = int(model.camera_name2id(self.camera))
        if self._model is not model:
            self._model = model
            self._camera_id = camera_id
            self._original_position = tuple(float(value) for value in model.cam_pos[camera_id])
            self._original_quaternion = _normalize_quaternion(model.cam_quat[camera_id])
            self._active = False
        return sim

    def _apply(self, env: Any) -> None:
        sim = self._capture_original(env)
        if (
            self._camera_id is None
            or self._original_position is None
            or self._original_quaternion is None
        ):
            raise RuntimeError("Camera pose cache is incomplete.")
        parent_offset = rotate_vector(self._original_quaternion, self.position_offset)
        effective_position = tuple(
            original + offset
            for original, offset in zip(self._original_position, parent_offset, strict=True)
        )
        effective_quaternion = multiply_quaternions(
            self._original_quaternion,
            local_xyz_offset_quaternion(self.rotation_offset_deg),
        )
        sim.model.cam_pos[self._camera_id] = effective_position
        sim.model.cam_quat[self._camera_id] = effective_quaternion
        sim.forward()
        self._active = True
        self._last_metadata = {
            **self.metadata(),
            "original_position": list(self._original_position),
            "original_quaternion_wxyz": list(self._original_quaternion),
            "position_offset_in_parent_frame": list(parent_offset),
            "effective_position": list(effective_position),
            "effective_quaternion_wxyz": list(effective_quaternion),
        }

    def _restore(self, env: Any) -> None:
        if not self._active:
            return
        sim = self._capture_original(env)
        if (
            self._camera_id is None
            or self._original_position is None
            or self._original_quaternion is None
        ):
            raise RuntimeError("Camera pose cache is incomplete.")
        sim.model.cam_pos[self._camera_id] = self._original_position
        sim.model.cam_quat[self._camera_id] = self._original_quaternion
        sim.forward()
        self._active = False

    def on_reset(self, env: Any, context: FaultContext) -> None:
        del context
        self._apply(env)

    @contextmanager
    def suspend(
        self,
        env: Any,
        scopes: frozenset[str],
        context: FaultContext,
    ) -> Iterator[None]:
        del scopes, context
        was_active = self._active
        if was_active:
            self._restore(env)
        try:
            yield
        finally:
            if was_active:
                self._apply(env)

    def state_dict(self) -> dict[str, Any]:
        return {"active": self._active}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = bool(state.get("active", False))
        if expected != self._active:
            raise RuntimeError("Restore the environment before loading camera fault runtime state.")

    def metadata(self) -> dict[str, Any]:
        payload = {
            "id": self.fault_id,
            "family": self.family,
            "target": self.camera,
            "operation": self.operation,
            "severity": self.severity.to_dict(),
            "parameters": {
                "position_offset": list(self.position_offset),
                "position_frame": "camera_local",
                "rotation_offset_deg": list(self.rotation_offset_deg),
                "rotation_frame": "camera_local",
                "rotation_order": "xyz",
            },
        }
        return payload

    @property
    def last_metadata(self) -> dict[str, Any] | None:
        return self._last_metadata

    def slug(self) -> str:
        safe_target = re.sub(r"[^A-Za-z0-9_-]+", "-", self.camera).strip("-")
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", self.severity.name).strip("-")
        safe_unit = re.sub(r"[^A-Za-z0-9_-]+", "-", self.severity.unit).strip("-")
        severity = f"{_encode_number(self.severity.value)}{safe_unit}"
        return f"visual-camera_pose-{safe_target}-{safe_name}-{severity}"

    def detach(self, env: Any) -> None:
        self._restore(env)
        self._model = None
        self._camera_id = None
        self._original_position = None
        self._original_quaternion = None
