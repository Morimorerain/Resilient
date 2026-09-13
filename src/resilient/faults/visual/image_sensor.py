"""Simulator-owned camera sensor faults applied before observations reach a policy."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from ..base import FaultContext, FaultRuntime
from ..types import SeveritySpec

CAMERA_OBSERVATION_KEYS = {
    "agentview": "agentview_image",
    "robot0_eye_in_hand": "robot0_eye_in_hand_image",
}


def _as_targets(config: Mapping[str, Any]) -> tuple[str, ...]:
    raw_targets = config.get("targets", config.get("target"))
    if isinstance(raw_targets, str):
        targets = (raw_targets,)
    elif isinstance(raw_targets, Sequence):
        targets = tuple(str(target) for target in raw_targets)
    else:
        raise TypeError("Image-sensor fault requires target or targets.")
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("Image-sensor targets must be a non-empty unique list.")
    unknown = sorted(set(targets) - set(CAMERA_OBSERVATION_KEYS))
    if unknown:
        raise ValueError(f"Unsupported camera sensor targets: {unknown}.")
    return targets


def _finite_vector(values: Sequence[float], *, size: int, name: str) -> np.ndarray:
    result = np.asarray(tuple(float(value) for value in values), dtype=np.float32)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain exactly {size} finite values.")
    return result


def _encode_number(value: float) -> str:
    sign = "p" if value >= 0 else "m"
    return sign + f"{abs(value):.4f}".rstrip("0").rstrip(".").replace(".", "p")


class ImageSensorFaultRuntime(FaultRuntime):
    """Base class for deterministic optical faults at the environment sensor boundary."""

    scopes = frozenset({"visual", "observation", "sensor"})

    def __init__(
        self,
        *,
        fault_id: str,
        targets: Sequence[str],
        severity: SeveritySpec,
        operation: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> None:
        self.fault_id = str(fault_id)
        self.targets = tuple(targets)
        self.severity = severity
        self.operation = dict(operation)
        self.parameters = dict(parameters)
        self._suspend_depth = 0

    @classmethod
    def _common_config(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "fault_id": str(config.get("id", cls.family.rsplit(".", 1)[-1])),
            "targets": _as_targets(config),
            "severity": SeveritySpec.from_config(dict(config["severity"])),
            "operation": dict(config.get("operation", {})),
            "parameters": dict(config.get("parameters", {})),
        }

    def transform_observation(
        self,
        observation: Any,
        env: Any,
        context: FaultContext,
    ) -> Any:
        del env, context
        if self._suspend_depth:
            return observation
        if not isinstance(observation, Mapping):
            raise TypeError(f"{self.family} expects a mapping observation.")
        for camera in self.targets:
            key = CAMERA_OBSERVATION_KEYS[camera]
            if key not in observation:
                raise KeyError(f"Observation does not contain camera key {key!r}.")
            observation[key] = self.apply_image(np.asarray(observation[key]))
        return observation

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """Transform one RGB uint8 sensor frame."""
        raise NotImplementedError

    @contextmanager
    def suspend(
        self,
        env: Any,
        scopes: frozenset[str],
        context: FaultContext,
    ) -> Iterator[None]:
        del env, context
        if not scopes.intersection(self.scopes):
            yield
            return
        self._suspend_depth += 1
        try:
            yield
        finally:
            self._suspend_depth -= 1

    def state_dict(self) -> dict[str, Any]:
        return {"suspend_depth": self._suspend_depth}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("suspend_depth", 0)) != 0:
            raise ValueError("Cannot restore a suspended image-sensor fault.")
        self._suspend_depth = 0

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.fault_id,
            "family": self.family,
            "targets": list(self.targets),
            "operation": self.operation,
            "severity": self.severity.to_dict(),
            "parameters": self.parameters,
            "injection_layer": "faulted_environment_camera_sensor",
        }

    def slug(self) -> str:
        family = re.sub(r"[^A-Za-z0-9_-]+", "-", self.family).strip("-")
        targets = "+".join(self.targets)
        unit = re.sub(r"[^A-Za-z0-9_-]+", "-", self.severity.unit).strip("-")
        magnitude = _encode_number(self.severity.value)
        return f"{family}-{targets}-{self.severity.name}-{magnitude}{unit}"


class DefocusBlurFaultRuntime(ImageSensorFaultRuntime):
    """Apply deterministic Gaussian defocus blur to selected camera sensors."""

    family = "visual.defocus_blur"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.severity.unit not in {"pixel", "pixels", "px"}:
            raise ValueError("Defocus blur severity unit must be pixel.")
        if self.severity.value < 0:
            raise ValueError("Gaussian blur sigma must be non-negative.")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> DefocusBlurFaultRuntime:
        return cls(**cls._common_config(config))

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Camera image must have shape [H, W, 3], got {image.shape}.")
        uint8 = np.clip(image, 0, 255).astype(np.uint8, copy=False)
        blurred = Image.fromarray(uint8).filter(ImageFilter.GaussianBlur(self.severity.value))
        return np.ascontiguousarray(np.asarray(blurred))


class LocalOcclusionFaultRuntime(ImageSensorFaultRuntime):
    """Overlay a fixed lens occlusion rectangle on selected camera sensors."""

    family = "visual.local_occlusion"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.severity.unit not in {"ratio", "fraction"}:
            raise ValueError("Local occlusion severity unit must be ratio or fraction.")
        if not 0.0 <= self.severity.value <= 1.0:
            raise ValueError("Occlusion ratio must be in [0, 1].")
        rectangle = self.parameters.get("rectangle", (0.25, 0.25, 0.5, 0.5))
        self.rectangle = _finite_vector(rectangle, size=4, name="rectangle")
        if np.any(self.rectangle < 0.0) or np.any(self.rectangle > 1.0):
            raise ValueError("Normalized occlusion rectangle values must be in [0, 1].")
        x, y, width, height = self.rectangle
        if x + width > 1.0 or y + height > 1.0 or width <= 0.0 or height <= 0.0:
            raise ValueError("Occlusion rectangle must fit inside the normalized image.")
        self.color = _finite_vector(
            self.parameters.get("color_rgb", (24, 24, 24)), size=3, name="color_rgb"
        )
        if np.any(self.color < 0.0) or np.any(self.color > 255.0):
            raise ValueError("Occlusion color must be in [0, 255].")
        configured_area = float(width * height)
        if not math.isclose(configured_area, self.severity.value, rel_tol=0.05, abs_tol=1e-6):
            raise ValueError(
                "Occlusion severity must equal the normalized rectangle area within 5%."
            )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> LocalOcclusionFaultRuntime:
        return cls(**cls._common_config(config))

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Camera image must have shape [H, W, 3], got {image.shape}.")
        result = np.asarray(image).copy()
        height, width = result.shape[:2]
        x, y, rect_width, rect_height = self.rectangle
        left = min(width - 1, int(round(float(x) * width)))
        top = min(height - 1, int(round(float(y) * height)))
        right = max(left + 1, min(width, int(round(float(x + rect_width) * width))))
        bottom = max(top + 1, min(height, int(round(float(y + rect_height) * height))))
        result[top:bottom, left:right] = self.color.astype(result.dtype)
        return np.ascontiguousarray(result)


class IlluminationFaultRuntime(ImageSensorFaultRuntime):
    """Apply a fixed camera color-response matrix and bias."""

    family = "visual.illumination"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.severity.unit not in {"ratio", "fraction"}:
            raise ValueError("Illumination severity unit must be ratio or fraction.")
        if self.severity.value < 0:
            raise ValueError("Illumination gain must be non-negative.")
        default_matrix = np.eye(3, dtype=np.float32) * self.severity.value
        matrix = np.asarray(self.parameters.get("color_matrix", default_matrix), dtype=np.float32)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("color_matrix must be a finite 3 x 3 matrix.")
        self.color_matrix = matrix
        self.color_bias = _finite_vector(
            self.parameters.get("color_bias", (0.0, 0.0, 0.0)),
            size=3,
            name="color_bias",
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> IlluminationFaultRuntime:
        return cls(**cls._common_config(config))

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Camera image must have shape [H, W, 3], got {image.shape}.")
        transformed = np.asarray(image, dtype=np.float32) @ self.color_matrix.T
        transformed += self.color_bias
        return np.ascontiguousarray(np.clip(transformed, 0.0, 255.0).astype(np.uint8))
