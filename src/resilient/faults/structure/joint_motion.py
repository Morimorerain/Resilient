"""Joint-motion degradation with pluggable displacement-retention laws."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..base import FaultContext, FaultRuntime
from ..types import FaultTransition, SeveritySpec


class JointMotionLaw(Protocol):
    """Transform nominal per-environment-step joint displacement and velocity."""

    name: str

    def apply(
        self,
        displacement: np.ndarray,
        velocity: np.ndarray,
        *,
        context: FaultContext,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def metadata(self) -> dict[str, Any]: ...


JointMotionLawFactory = Callable[[Mapping[str, Any], SeveritySpec], JointMotionLaw]
_LAW_REGISTRY: dict[str, JointMotionLawFactory] = {}


def register_joint_motion_law(name: str, factory: JointMotionLawFactory) -> None:
    """Register a deterministic degradation law for future non-linear extensions."""
    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("Joint-motion law name must not be empty.")
    if normalized in _LAW_REGISTRY:
        raise ValueError(f"Joint-motion law already registered: {normalized}")
    _LAW_REGISTRY[normalized] = factory


def available_joint_motion_laws() -> tuple[str, ...]:
    """Return registered law names in deterministic order."""
    _register_builtin_laws()
    return tuple(sorted(_LAW_REGISTRY))


@dataclass(frozen=True)
class ProportionalMotionLaw:
    """Retain a fixed proportion of nominal displacement and velocity."""

    retention_ratio: float
    name: str = "proportional"

    def apply(
        self,
        displacement: np.ndarray,
        velocity: np.ndarray,
        *,
        context: FaultContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        del context
        return displacement * self.retention_ratio, velocity * self.retention_ratio

    def metadata(self) -> dict[str, Any]:
        return {"type": self.name, "retention_ratio": self.retention_ratio}


def _build_proportional_law(
    operation: Mapping[str, Any], severity: SeveritySpec
) -> ProportionalMotionLaw:
    del operation
    if severity.unit not in {"ratio", "fraction"}:
        raise ValueError("Proportional joint-motion severity unit must be ratio or fraction.")
    ratio = float(severity.value)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("Joint-motion retention ratio must be in [0, 1].")
    return ProportionalMotionLaw(retention_ratio=ratio)


def _register_builtin_laws() -> None:
    if "proportional" not in _LAW_REGISTRY:
        register_joint_motion_law("proportional", _build_proportional_law)


def _get_sim(env: Any) -> Any:
    current = env
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        sim = getattr(current, "sim", None)
        if sim is not None:
            return sim
        current = getattr(current, "env", None)
        if current is None:
            break
    raise AttributeError("Environment does not expose a MuJoCo simulation object.")


def _as_joint_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    raw_targets = config.get("targets", config.get("target"))
    if isinstance(raw_targets, str):
        targets = (raw_targets,)
    elif isinstance(raw_targets, Sequence):
        targets = tuple(str(target) for target in raw_targets)
    else:
        raise TypeError("Joint-motion fault requires target or targets.")
    if not targets or any(not target.strip() for target in targets):
        raise ValueError("Joint-motion targets must contain at least one non-empty joint name.")
    if len(set(targets)) != len(targets):
        raise ValueError(f"Joint-motion targets contain duplicates: {targets}")
    return targets


class JointMotionFaultRuntime(FaultRuntime):
    """Reduce selected joint motion at the environment dynamics boundary."""

    family = "structure.joint_motion"
    scopes = frozenset({"structure", "dynamics", "joint"})
    requires_post_step_observation_refresh = True

    def __init__(
        self,
        *,
        fault_id: str,
        joint_names: Sequence[str],
        severity: SeveritySpec,
        operation: Mapping[str, Any],
        law: JointMotionLaw,
    ) -> None:
        self.fault_id = str(fault_id)
        self.joint_names = tuple(joint_names)
        self.severity = severity
        self.operation = dict(operation)
        self.law = law
        self._sim: Any | None = None
        self._qpos_indices: np.ndarray | None = None
        self._qvel_indices: np.ndarray | None = None
        self._before_qpos: np.ndarray | None = None
        self._pending_context: tuple[int, int] | None = None
        self._suspend_depth = 0

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> JointMotionFaultRuntime:
        operation = dict(config.get("operation", {}))
        law_name = str(operation.get("type", "proportional")).strip().lower()
        severity = SeveritySpec.from_config(dict(config["severity"]))
        _register_builtin_laws()
        if law_name not in _LAW_REGISTRY:
            known = ", ".join(available_joint_motion_laws())
            raise ValueError(f"Unknown joint-motion law {law_name!r}; available laws: {known}.")
        law = _LAW_REGISTRY[law_name](operation, severity)
        return cls(
            fault_id=str(config.get("id", "joint_motion")),
            joint_names=_as_joint_names(config),
            severity=severity,
            operation=operation,
            law=law,
        )

    def _resolve_joint_indices(self, sim: Any) -> None:
        qpos_indices = []
        qvel_indices = []
        for joint_name in self.joint_names:
            try:
                qpos_address = sim.model.get_joint_qpos_addr(joint_name)
                qvel_address = sim.model.get_joint_qvel_addr(joint_name)
            except Exception as error:
                raise ValueError(f"Unknown or unsupported joint target: {joint_name}") from error
            if not isinstance(qpos_address, int | np.integer) or not isinstance(
                qvel_address, int | np.integer
            ):
                raise ValueError(
                    f"Joint {joint_name!r} is not a scalar one-DoF joint: "
                    f"qpos={qpos_address}, qvel={qvel_address}."
                )
            qpos_indices.append(int(qpos_address))
            qvel_indices.append(int(qvel_address))
        self._qpos_indices = np.asarray(qpos_indices, dtype=np.int64)
        self._qvel_indices = np.asarray(qvel_indices, dtype=np.int64)

    def _ensure_simulator(self, env: Any) -> None:
        sim = _get_sim(env)
        if self._sim is sim:
            return
        self._sim = sim
        self._resolve_joint_indices(sim)

    def attach(self, env: Any, context: FaultContext) -> None:
        del context
        self._ensure_simulator(env)

    def on_reset(self, env: Any, context: FaultContext) -> None:
        del context
        self._ensure_simulator(env)
        self._before_qpos = None
        self._pending_context = None

    def on_state_loaded(self, env: Any, context: FaultContext) -> None:
        del context
        self._ensure_simulator(env)
        self._before_qpos = None
        self._pending_context = None

    def before_step(self, env: Any, action: Any, context: FaultContext) -> None:
        del action
        if self._suspend_depth:
            return
        self._ensure_simulator(env)
        if self._pending_context is not None:
            raise RuntimeError(
                "Joint-motion fault received a new step before completing the prior step."
            )
        if self._sim is None or self._qpos_indices is None:
            raise RuntimeError("Joint-motion fault is not installed on a simulator.")
        self._before_qpos = np.asarray(
            self._sim.data.qpos[self._qpos_indices], dtype=np.float64
        ).copy()
        self._pending_context = (context.episode_index, context.step_index)

    def after_step(
        self,
        env: Any,
        transition: FaultTransition,
        context: FaultContext,
    ) -> None:
        if self._suspend_depth:
            return
        expected = (context.episode_index, context.step_index)
        if self._pending_context != expected or self._before_qpos is None:
            raise RuntimeError(
                "Joint-motion fault post-step hook has no matching environment snapshot."
            )
        self._ensure_simulator(env)
        if self._sim is None or self._qpos_indices is None or self._qvel_indices is None:
            raise RuntimeError("Joint-motion fault is not installed on a simulator.")

        nominal_qpos = np.asarray(
            self._sim.data.qpos[self._qpos_indices], dtype=np.float64
        ).copy()
        nominal_qvel = np.asarray(
            self._sim.data.qvel[self._qvel_indices], dtype=np.float64
        ).copy()
        nominal_displacement = nominal_qpos - self._before_qpos
        degraded_displacement, degraded_velocity = self.law.apply(
            nominal_displacement,
            nominal_qvel,
            context=context,
        )
        degraded_displacement = np.asarray(degraded_displacement, dtype=np.float64)
        degraded_velocity = np.asarray(degraded_velocity, dtype=np.float64)
        if degraded_displacement.shape != nominal_displacement.shape:
            raise ValueError("Joint-motion law changed the displacement shape.")
        if degraded_velocity.shape != nominal_qvel.shape:
            raise ValueError("Joint-motion law changed the velocity shape.")
        if not np.all(np.isfinite(degraded_displacement)) or not np.all(
            np.isfinite(degraded_velocity)
        ):
            raise ValueError("Joint-motion law produced non-finite state values.")

        self._sim.data.qpos[self._qpos_indices] = self._before_qpos + degraded_displacement
        self._sim.data.qvel[self._qvel_indices] = degraded_velocity
        self._sim.forward()
        transition.info.setdefault("joint_motion_faults", []).append(
            {
                "id": self.fault_id,
                "joints": list(self.joint_names),
                "nominal_displacement_rad": nominal_displacement.tolist(),
                "applied_displacement_rad": degraded_displacement.tolist(),
            }
        )
        self._before_qpos = None
        self._pending_context = None

    @contextmanager
    def suspend(
        self,
        env: Any,
        scopes: frozenset[str],
        context: FaultContext,
    ):
        del env, context
        if not scopes.intersection(self.scopes):
            yield
            return
        if self._pending_context is not None:
            raise RuntimeError("Cannot suspend joint-motion degradation during an active step.")
        self._suspend_depth += 1
        try:
            yield
        finally:
            self._suspend_depth -= 1

    def state_dict(self) -> dict[str, Any]:
        if self._pending_context is not None:
            raise RuntimeError("Cannot checkpoint joint-motion degradation during an active step.")
        return {"suspend_depth": self._suspend_depth}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("suspend_depth", 0)) != 0:
            raise ValueError("Cannot restore a suspended joint-motion fault.")
        self._suspend_depth = 0
        self._before_qpos = None
        self._pending_context = None

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.fault_id,
            "family": self.family,
            "targets": list(self.joint_names),
            "operation": self.law.metadata(),
            "severity": self.severity.to_dict(),
            "semantics": "per_environment_dynamics_step_joint_displacement_retention",
            "injection_layer": "faulted_environment_dynamics",
        }

    def slug(self) -> str:
        targets = "+".join(
            re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-") for name in self.joint_names
        )
        ratio = f"{self.severity.value:.4f}".rstrip("0").rstrip(".").replace(".", "p")
        return f"structure-joint_motion-{targets}-envstep-retention-p{ratio}"

    def detach(self, env: Any) -> None:
        del env
        self._sim = None
        self._qpos_indices = None
        self._qvel_indices = None
        self._before_qpos = None
        self._pending_context = None
        self._suspend_depth = 0
