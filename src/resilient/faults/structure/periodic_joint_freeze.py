"""Angular-position-triggered periodic joint sticking fault."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..base import FaultContext
from .joint_state import JointStateFaultRuntime, numeric_parameter


class PeriodicJointFreezeFaultRuntime(JointStateFaultRuntime):
    """Hold joints when motion crosses periodic defective gear angles."""

    family = "structure.periodic_joint_freeze"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.severity.unit not in {"step", "steps", "control_step", "control_steps"}:
            raise ValueError("Periodic joint-freeze severity unit must be control_steps.")
        self.hold_steps = int(round(self.severity.value))
        if self.hold_steps <= 0 or not math.isclose(self.hold_steps, self.severity.value):
            raise ValueError("Periodic joint-freeze hold_steps must be a positive integer.")
        period_deg = numeric_parameter(self.parameters, "period_deg", len(self.joint_names))
        phase_deg = numeric_parameter(
            self.parameters, "phase_deg", len(self.joint_names), default=0.0
        )
        if np.any(period_deg <= 0.0):
            raise ValueError("period_deg must be positive.")
        self.period_rad = np.deg2rad(period_deg)
        self.phase_rad = np.deg2rad(phase_deg)
        self.release_fraction = float(self.parameters.get("release_fraction", 0.25))
        if not 0.0 < self.release_fraction < 0.5:
            raise ValueError("release_fraction must be in (0, 0.5).")
        self._remaining_hold = np.zeros(len(self.joint_names), dtype=np.int64)
        self._last_trigger_index: list[int | None] = [None] * len(self.joint_names)
        self._last_trigger_direction = np.zeros(len(self.joint_names), dtype=np.int8)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> PeriodicJointFreezeFaultRuntime:
        return cls(**cls.common_config(config))

    def reset_runtime(self) -> None:
        self._remaining_hold.fill(0)
        self._last_trigger_index = [None] * len(self.joint_names)
        self._last_trigger_direction.fill(0)

    @staticmethod
    def _crossed_trigger(
        start: float,
        end: float,
        period: float,
        phase: float,
        last_index: int | None,
    ) -> int | None:
        if math.isclose(start, end, abs_tol=1e-12):
            return None
        low, high = sorted((start, end))
        first = math.ceil((low - phase) / period)
        last = math.floor((high - phase) / period)
        candidates = range(first, last + 1) if end > start else range(last, first - 1, -1)
        for trigger_index in candidates:
            trigger = phase + trigger_index * period
            crossed = start < trigger <= end if end > start else end <= trigger < start
            if crossed and trigger_index != last_index:
                return trigger_index
        return None

    def transform_realized_state(
        self,
        before_qpos: np.ndarray,
        candidate_qpos: np.ndarray,
        candidate_qvel: np.ndarray,
        *,
        context: FaultContext,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        del context
        applied = candidate_qpos.copy()
        velocity = candidate_qvel.copy()
        triggered = [False] * len(self.joint_names)
        for index, (start, end) in enumerate(zip(before_qpos, candidate_qpos, strict=True)):
            last_index = self._last_trigger_index[index]
            if last_index is not None:
                last_angle = self.phase_rad[index] + last_index * self.period_rad[index]
                release_distance = self.release_fraction * self.period_rad[index]
                direction = self._last_trigger_direction[index]
                passed_forward = direction > 0 and start >= last_angle + release_distance
                passed_reverse = direction < 0 and start <= last_angle - release_distance
                if passed_forward or passed_reverse:
                    self._last_trigger_index[index] = None
                    self._last_trigger_direction[index] = 0
            if self._remaining_hold[index] > 0:
                applied[index] = start
                velocity[index] = 0.0
                self._remaining_hold[index] -= 1
                continue
            trigger_index = self._crossed_trigger(
                float(start),
                float(end),
                float(self.period_rad[index]),
                float(self.phase_rad[index]),
                self._last_trigger_index[index],
            )
            if trigger_index is not None:
                applied[index] = start
                velocity[index] = 0.0
                self._remaining_hold[index] = self.hold_steps - 1
                self._last_trigger_index[index] = trigger_index
                self._last_trigger_direction[index] = int(np.sign(end - start))
                triggered[index] = True
        return (
            applied,
            velocity,
            {
                "triggered": triggered,
                "remaining_hold_steps": self._remaining_hold.tolist(),
                "last_trigger_index": self._last_trigger_index,
            },
        )

    def mutable_state_dict(self) -> dict[str, Any]:
        return {
            "remaining_hold_steps": self._remaining_hold.tolist(),
            "last_trigger_index": self._last_trigger_index,
            "last_trigger_direction": self._last_trigger_direction.tolist(),
        }

    def restore_mutable_state(self, state: Mapping[str, Any]) -> None:
        remaining = np.asarray(
            state.get("remaining_hold_steps", self._remaining_hold), dtype=np.int64
        )
        trigger_indices = list(state.get("last_trigger_index", self._last_trigger_index))
        trigger_direction = np.asarray(
            state.get("last_trigger_direction", self._last_trigger_direction),
            dtype=np.int8,
        )
        if remaining.shape != self._remaining_hold.shape or len(trigger_indices) != len(
            self._last_trigger_index
        ) or trigger_direction.shape != self._last_trigger_direction.shape:
            raise ValueError("Periodic-freeze checkpoint shape does not match configured targets.")
        self._remaining_hold[:] = remaining
        self._last_trigger_index = [
            None if value is None else int(value) for value in trigger_indices
        ]
        self._last_trigger_direction[:] = trigger_direction

    def metadata(self) -> dict[str, Any]:
        payload = super().metadata()
        payload["parameters"] = {
            "period_deg": np.rad2deg(self.period_rad).tolist(),
            "phase_deg": np.rad2deg(self.phase_rad).tolist(),
            "hold_steps": self.hold_steps,
            "release_fraction": self.release_fraction,
        }
        payload["semantics"] = "angular_position_triggered_periodic_gear_stick"
        return payload

    def slug(self) -> str:
        targets = "+".join(re.sub(r"[^A-Za-z0-9_-]+", "-", x) for x in self.joint_names)
        period = "+".join(f"{x:g}" for x in np.rad2deg(self.period_rad))
        return f"structure-periodic_joint_freeze-{targets}-{period}deg-{self.hold_steps}steps"
