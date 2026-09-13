"""Reduced joint-range fault at the simulator dynamics boundary."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..base import FaultContext
from .joint_state import JointStateFaultRuntime, numeric_parameter


class JointRangeLimitFaultRuntime(JointStateFaultRuntime):
    """Clamp realized joint positions to configured reduced absolute bounds."""

    family = "structure.joint_range_limit"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.severity.unit not in {"degree", "degrees", "deg"}:
            raise ValueError("Joint-range severity unit must be degree.")
        lower_deg = numeric_parameter(self.parameters, "lower_deg", len(self.joint_names))
        upper_deg = numeric_parameter(self.parameters, "upper_deg", len(self.joint_names))
        if np.any(lower_deg >= upper_deg):
            raise ValueError("Every reduced joint lower bound must be below its upper bound.")
        widths = upper_deg - lower_deg
        if not np.allclose(widths, self.severity.value, rtol=0.01, atol=1e-6):
            raise ValueError("Joint-range severity must equal every configured range width.")
        self.lower_rad = np.deg2rad(lower_deg)
        self.upper_rad = np.deg2rad(upper_deg)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> JointRangeLimitFaultRuntime:
        return cls(**cls.common_config(config))

    def transform_realized_state(
        self,
        before_qpos: np.ndarray,
        candidate_qpos: np.ndarray,
        candidate_qvel: np.ndarray,
        *,
        context: FaultContext,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        del before_qpos, context
        applied = np.clip(candidate_qpos, self.lower_rad, self.upper_rad)
        clipped = ~np.isclose(applied, candidate_qpos, atol=1e-12)
        velocity = candidate_qvel.copy()
        velocity[clipped] = 0.0
        return applied, velocity, {"clipped": clipped.tolist()}

    def metadata(self) -> dict[str, Any]:
        payload = super().metadata()
        payload["parameters"] = {
            "lower_deg": np.rad2deg(self.lower_rad).tolist(),
            "upper_deg": np.rad2deg(self.upper_rad).tolist(),
        }
        payload["semantics"] = "absolute_realized_joint_position_clip"
        return payload

    def slug(self) -> str:
        targets = "+".join(re.sub(r"[^A-Za-z0-9_-]+", "-", x) for x in self.joint_names)
        return f"structure-joint_range_limit-{targets}-{self.severity.value:g}deg"
