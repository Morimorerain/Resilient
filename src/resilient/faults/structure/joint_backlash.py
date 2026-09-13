"""Direction-reversal backlash fault at the simulator dynamics boundary."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..base import FaultContext
from .joint_state import JointStateFaultRuntime, numeric_parameter


class JointBacklashFaultRuntime(JointStateFaultRuntime):
    """Consume commanded travel in a dead band after each direction reversal."""

    family = "structure.joint_backlash"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.severity.unit not in {"degree", "degrees", "deg"}:
            raise ValueError("Joint-backlash severity unit must be degree.")
        gap_deg = numeric_parameter(
            self.parameters,
            "gap_deg",
            len(self.joint_names),
            default=self.severity.value,
        )
        if np.any(gap_deg < 0.0):
            raise ValueError("Joint-backlash gaps must be non-negative.")
        self.gap_rad = np.deg2rad(gap_deg)
        self.direction_epsilon_rad = float(self.parameters.get("direction_epsilon_rad", 1e-7))
        if self.direction_epsilon_rad < 0.0:
            raise ValueError("direction_epsilon_rad must be non-negative.")
        self._last_direction = np.zeros(len(self.joint_names), dtype=np.int8)
        self._remaining_gap = np.zeros(len(self.joint_names), dtype=np.float64)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> JointBacklashFaultRuntime:
        return cls(**cls.common_config(config))

    def reset_runtime(self) -> None:
        self._last_direction.fill(0)
        self._remaining_gap.fill(0.0)

    def transform_realized_state(
        self,
        before_qpos: np.ndarray,
        candidate_qpos: np.ndarray,
        candidate_qvel: np.ndarray,
        *,
        context: FaultContext,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        del context
        displacement = candidate_qpos - before_qpos
        applied = displacement.copy()
        reversals: list[bool] = []
        for index, delta in enumerate(displacement):
            direction = int(np.sign(delta)) if abs(delta) > self.direction_epsilon_rad else 0
            reversed_direction = bool(
                direction
                and self._last_direction[index]
                and direction != self._last_direction[index]
            )
            if reversed_direction:
                self._remaining_gap[index] = self.gap_rad[index]
            reversals.append(reversed_direction)
            if direction:
                consumed = min(abs(delta), self._remaining_gap[index])
                self._remaining_gap[index] -= consumed
                applied[index] = direction * (abs(delta) - consumed)
                self._last_direction[index] = direction
        velocity = candidate_qvel.copy()
        velocity[np.isclose(applied, 0.0, atol=self.direction_epsilon_rad)] = 0.0
        return (
            before_qpos + applied,
            velocity,
            {
                "direction_reversal": reversals,
                "remaining_gap_deg": np.rad2deg(self._remaining_gap).tolist(),
                "candidate_displacement_deg": np.rad2deg(displacement).tolist(),
                "applied_displacement_deg": np.rad2deg(applied).tolist(),
            },
        )

    def mutable_state_dict(self) -> dict[str, Any]:
        return {
            "last_direction": self._last_direction.tolist(),
            "remaining_gap_rad": self._remaining_gap.tolist(),
        }

    def restore_mutable_state(self, state: Mapping[str, Any]) -> None:
        direction = np.asarray(state.get("last_direction", self._last_direction), dtype=np.int8)
        remaining = np.asarray(
            state.get("remaining_gap_rad", self._remaining_gap), dtype=np.float64
        )
        if (
            direction.shape != self._last_direction.shape
            or remaining.shape != self._remaining_gap.shape
        ):
            raise ValueError("Joint-backlash checkpoint shape does not match configured targets.")
        self._last_direction[:] = direction
        self._remaining_gap[:] = remaining

    def metadata(self) -> dict[str, Any]:
        payload = super().metadata()
        payload["parameters"] = {
            "gap_deg": np.rad2deg(self.gap_rad).tolist(),
            "direction_epsilon_rad": self.direction_epsilon_rad,
        }
        payload["semantics"] = "direction_reversal_dead_travel"
        return payload

    def slug(self) -> str:
        targets = "+".join(re.sub(r"[^A-Za-z0-9_-]+", "-", x) for x in self.joint_names)
        return f"structure-joint_backlash-{targets}-{self.severity.value:g}deg"
