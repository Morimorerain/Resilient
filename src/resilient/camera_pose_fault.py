"""Camera-pose fault injection for LIBERO evaluation environments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SUPPORTED_CAMERAS = ("agentview", "robot0_eye_in_hand")


def _as_triplet(values: Sequence[float], name: str) -> tuple[float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three values, got {len(result)}.")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _normalize_quaternion(quaternion: Sequence[float]) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in quaternion)
    if len(values) != 4:
        raise ValueError(f"Quaternion must contain four values, got {len(values)}.")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ValueError("Quaternion must have non-zero length.")
    return tuple(value / norm for value in values)


def multiply_quaternions(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float]:
    """Multiply two quaternions in MuJoCo ``wxyz`` order."""
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
    """Return the local X-then-Y-then-Z rotation offset in ``wxyz`` order."""
    rotation = _as_triplet(rotation_offset_deg, "rotation_offset_deg")
    axis_quaternions = []
    for axis, angle_deg in enumerate(rotation):
        half_angle = math.radians(angle_deg) / 2.0
        values = [math.cos(half_angle), 0.0, 0.0, 0.0]
        values[axis + 1] = math.sin(half_angle)
        axis_quaternions.append(tuple(values))

    offset = (1.0, 0.0, 0.0, 0.0)
    for axis_quaternion in axis_quaternions:
        offset = multiply_quaternions(offset, axis_quaternion)
    return offset


@dataclass(frozen=True)
class CameraPoseFaultSpec:
    """Configuration for one deterministic camera-pose fault."""

    enabled: bool = False
    camera: str = "agentview"
    position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_offset_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.camera not in SUPPORTED_CAMERAS:
            raise ValueError(
                f"Unsupported camera {self.camera!r}; expected one of {SUPPORTED_CAMERAS}."
            )
        object.__setattr__(
            self,
            "position_offset",
            _as_triplet(self.position_offset, "position_offset"),
        )
        object.__setattr__(
            self,
            "rotation_offset_deg",
            _as_triplet(self.rotation_offset_deg, "rotation_offset_deg"),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> CameraPoseFaultSpec:
        """Build a validated specification from a Hydra-compatible mapping."""
        if config is None:
            return cls()
        return cls(
            enabled=bool(config.get("enabled", False)),
            camera=str(config.get("camera", "agentview")),
            position_offset=tuple(config.get("position_offset", (0.0, 0.0, 0.0))),
            rotation_offset_deg=tuple(
                config.get("rotation_offset_deg", (0.0, 0.0, 0.0))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "enabled": self.enabled,
            "camera": self.camera,
            "position_offset": list(self.position_offset),
            "position_frame": "camera_local",
            "rotation_offset_deg": list(self.rotation_offset_deg),
            "rotation_frame": "camera_local",
            "rotation_order": "xyz",
        }


class CameraPoseFaultApplier:
    """Apply a fault without accumulating offsets across environment resets."""

    def __init__(self, spec: CameraPoseFaultSpec):
        if not spec.enabled:
            raise ValueError("CameraPoseFaultApplier requires an enabled specification.")
        self.spec = spec
        self.last_metadata: dict[str, Any] | None = None
        self._model: Any | None = None
        self._original_position: tuple[float, float, float] | None = None
        self._original_quaternion: tuple[float, float, float, float] | None = None

    @staticmethod
    def _get_sim(env: Any) -> Any:
        if hasattr(env, "sim"):
            return env.sim
        if hasattr(env, "env") and hasattr(env.env, "sim"):
            return env.env.sim
        raise AttributeError("LIBERO environment does not expose a MuJoCo simulation object.")

    def apply(self, env: Any) -> dict[str, Any]:
        """Apply the configured offset and return original/effective pose metadata."""
        sim = self._get_sim(env)
        model = sim.model
        camera_id = int(model.camera_name2id(self.spec.camera))

        if self._model is not model:
            original_position = tuple(float(value) for value in model.cam_pos[camera_id])
            original_quaternion = _normalize_quaternion(model.cam_quat[camera_id])
            self._model = model
            self._original_position = original_position
            self._original_quaternion = original_quaternion
        else:
            if self._original_position is None or self._original_quaternion is None:
                raise RuntimeError("Original camera pose cache is incomplete.")
            original_position = self._original_position
            original_quaternion = self._original_quaternion

        parent_position_offset = rotate_vector(
            original_quaternion,
            self.spec.position_offset,
        )
        effective_position = tuple(
            original + offset
            for original, offset in zip(
                original_position,
                parent_position_offset,
                strict=True,
            )
        )
        offset_quaternion = local_xyz_offset_quaternion(self.spec.rotation_offset_deg)
        effective_quaternion = multiply_quaternions(
            original_quaternion,
            offset_quaternion,
        )

        model.cam_pos[camera_id] = effective_position
        model.cam_quat[camera_id] = effective_quaternion
        sim.forward()

        self.last_metadata = {
            **self.spec.to_dict(),
            "original_position": list(original_position),
            "original_quaternion_wxyz": list(original_quaternion),
            "position_offset_in_parent_frame": list(parent_position_offset),
            "effective_position": list(effective_position),
            "effective_quaternion_wxyz": list(effective_quaternion),
        }
        return self.last_metadata
