"""Shared value objects for fault injection and transition recording."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SeveritySpec:
    """A physical, ordered fault magnitude defined by one fault plugin."""

    name: str
    value: float
    unit: str
    level: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Severity name must not be empty.")
        if not self.unit.strip():
            raise ValueError("Severity unit must not be empty.")
        if not math.isfinite(float(self.value)):
            raise ValueError("Severity value must be finite.")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SeveritySpec:
        if not isinstance(config, dict):
            raise TypeError("Fault severity must be a mapping.")
        return cls(
            name=str(config["name"]),
            value=float(config["value"]),
            unit=str(config["unit"]),
            level=None if config.get("level") is None else str(config["level"]),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
        }
        if self.level is not None:
            payload["level"] = self.level
        return payload


@dataclass
class FaultTransition:
    """One environment transition with nominal and faulted values kept separate."""

    raw_observation: Any
    student_observation: Any
    policy_action: Any | None = None
    executed_action: Any | None = None
    next_raw_observation: Any | None = None
    next_student_observation: Any | None = None
    reward: float | None = None
    done: bool | None = None
    info: dict[str, Any] = field(default_factory=dict)
    fault_metadata: list[dict[str, Any]] = field(default_factory=list)
