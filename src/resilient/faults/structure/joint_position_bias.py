"""Fixed joint zero-offset fault at the simulator dynamics boundary."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..base import FaultContext
from .joint_state import JointStateFaultRuntime, numeric_parameter


class JointPositionBiasFaultRuntime(JointStateFaultRuntime):
    """Introduce one non-cumulative physical joint-position offset per loaded state."""

    family = "structure.joint_position_bias"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.severity.unit not in {"degree", "degrees", "deg"}:
            raise ValueError("Joint-position bias severity unit must be degree.")
        configured = numeric_parameter(
            self.parameters,
            "bias_deg",
            len(self.joint_names),
            default=self.severity.value,
        )
        self.bias_rad = np.deg2rad(configured)
        self._initialized = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> JointPositionBiasFaultRuntime:
        return cls(**cls.common_config(config))

    def reset_runtime(self) -> None:
        self._initialized = False

    def transform_realized_state(
        self,
        before_qpos: np.ndarray,
        candidate_qpos: np.ndarray,
        candidate_qvel: np.ndarray,
        *,
        context: FaultContext,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        del before_qpos, context
        applied_qpos = candidate_qpos.copy()
        introduced = not self._initialized
        if introduced:
            applied_qpos += self.bias_rad
            self._initialized = True
        return (
            applied_qpos,
            candidate_qvel,
            {
                "bias_deg": np.rad2deg(self.bias_rad).tolist(),
                "bias_introduced_this_step": introduced,
            },
        )

    def mutable_state_dict(self) -> dict[str, Any]:
        return {"initialized": self._initialized}

    def restore_mutable_state(self, state: Mapping[str, Any]) -> None:
        self._initialized = bool(state.get("initialized", False))

    def metadata(self) -> dict[str, Any]:
        payload = super().metadata()
        payload["parameters"] = {"bias_deg": np.rad2deg(self.bias_rad).tolist()}
        payload["semantics"] = "fixed_non_cumulative_joint_zero_offset_per_loaded_state"
        return payload

    def slug(self) -> str:
        targets = "+".join(re.sub(r"[^A-Za-z0-9_-]+", "-", x) for x in self.joint_names)
        values = "+".join(
            f"{x:g}".replace("-", "m").replace(".", "p")
            for x in np.rad2deg(self.bias_rad)
        )
        return f"structure-joint_position_bias-{targets}-{values}deg"
