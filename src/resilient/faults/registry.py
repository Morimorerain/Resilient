"""Registry-based construction for fault plugins."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import FaultRuntime

FaultFactory = Callable[[Mapping[str, Any]], FaultRuntime]
_REGISTRY: dict[str, FaultFactory] = {}


def register_fault(family: str, factory: FaultFactory) -> None:
    """Register one fault family and reject ambiguous duplicate implementations."""
    normalized = family.strip()
    if not normalized:
        raise ValueError("Fault family must not be empty.")
    if normalized in _REGISTRY:
        raise ValueError(f"Fault family is already registered: {normalized}")
    _REGISTRY[normalized] = factory


def registered_faults() -> tuple[str, ...]:
    """Return fault families in deterministic order."""
    return tuple(sorted(_REGISTRY))


def _register_builtins() -> None:
    from .visual.camera_pose import CameraPoseFaultRuntime

    if CameraPoseFaultRuntime.family not in _REGISTRY:
        register_fault(CameraPoseFaultRuntime.family, CameraPoseFaultRuntime.from_config)


def build_fault(config: Mapping[str, Any]) -> FaultRuntime:
    """Instantiate one enabled fault from a resolved configuration mapping."""
    _register_builtins()
    family = str(config.get("family", "")).strip()
    if family not in _REGISTRY:
        known = ", ".join(registered_faults()) or "<none>"
        raise ValueError(f"Unknown fault family {family!r}; registered families: {known}.")
    return _REGISTRY[family](config)


def build_fault_pipeline(config: Mapping[str, Any] | None):
    """Build a pipeline while keeping an absent/disabled configuration a no-op."""
    from .pipeline import FaultPipeline

    if config is None:
        return FaultPipeline(pipeline_id="none", seed=0, faults=[])
    pipeline_cfg = config.get("pipeline", config)
    if not isinstance(pipeline_cfg, Mapping):
        raise TypeError("Fault pipeline configuration must be a mapping.")
    enabled = bool(pipeline_cfg.get("enabled", True))
    raw_faults = pipeline_cfg.get("faults", []) if enabled else []
    if raw_faults is None:
        raw_faults = []
    faults = [build_fault(item) for item in raw_faults if bool(item.get("enabled", True))]
    return FaultPipeline(
        pipeline_id=str(pipeline_cfg.get("id", "none" if not faults else "fault_pipeline")),
        seed=int(pipeline_cfg.get("seed", 0)),
        faults=faults,
    )
