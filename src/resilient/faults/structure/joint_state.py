"""Shared simulator-step machinery for stateful scalar-joint faults."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from ..base import FaultContext, FaultRuntime
from ..types import FaultTransition, SeveritySpec


def get_simulator(env: Any) -> Any:
    """Find the MuJoCo simulation through transparent environment wrappers."""
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


def joint_names_from_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Read and validate one or more scalar joint names."""
    raw_targets = config.get("targets", config.get("target"))
    if isinstance(raw_targets, str):
        targets = (raw_targets,)
    elif isinstance(raw_targets, Sequence):
        targets = tuple(str(target) for target in raw_targets)
    else:
        raise TypeError("Joint fault requires target or targets.")
    if not targets or any(not target.strip() for target in targets):
        raise ValueError("Joint targets must contain at least one non-empty name.")
    if len(set(targets)) != len(targets):
        raise ValueError(f"Joint targets contain duplicates: {targets}")
    return targets


def numeric_parameter(
    parameters: Mapping[str, Any],
    name: str,
    count: int,
    *,
    default: float | None = None,
) -> np.ndarray:
    """Expand a scalar or validate a target-aligned numeric parameter."""
    if name not in parameters:
        if default is None:
            raise ValueError(f"Missing required parameter: {name}")
        raw: Any = default
    else:
        raw = parameters[name]
    if isinstance(raw, int | float):
        values = np.full(count, float(raw), dtype=np.float64)
    else:
        values = np.asarray(tuple(float(value) for value in raw), dtype=np.float64)
    if values.shape != (count,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite scalar or contain one value per target.")
    return values


class JointStateFaultRuntime(FaultRuntime):
    """Base for faults that modify realized joint state after each environment step."""

    scopes = frozenset({"structure", "dynamics", "joint"})
    requires_post_step_observation_refresh = True

    def __init__(
        self,
        *,
        fault_id: str,
        joint_names: Sequence[str],
        severity: SeveritySpec,
        operation: Mapping[str, Any],
        parameters: Mapping[str, Any],
    ) -> None:
        self.fault_id = str(fault_id)
        self.joint_names = tuple(joint_names)
        self.severity = severity
        self.operation = dict(operation)
        self.parameters = dict(parameters)
        self._sim: Any | None = None
        self._qpos_indices: np.ndarray | None = None
        self._qvel_indices: np.ndarray | None = None
        self._before_qpos: np.ndarray | None = None
        self._pending_context: tuple[int, int] | None = None
        self._suspend_depth = 0

    @classmethod
    def common_config(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "fault_id": str(config.get("id", cls.family.rsplit(".", 1)[-1])),
            "joint_names": joint_names_from_config(config),
            "severity": SeveritySpec.from_config(dict(config["severity"])),
            "operation": dict(config.get("operation", {})),
            "parameters": dict(config.get("parameters", {})),
        }

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
                raise ValueError(f"Joint {joint_name!r} is not a scalar one-DoF joint.")
            qpos_indices.append(int(qpos_address))
            qvel_indices.append(int(qvel_address))
        self._qpos_indices = np.asarray(qpos_indices, dtype=np.int64)
        self._qvel_indices = np.asarray(qvel_indices, dtype=np.int64)

    def _ensure_simulator(self, env: Any) -> None:
        sim = get_simulator(env)
        if self._sim is sim:
            return
        self._sim = sim
        self._resolve_joint_indices(sim)

    def attach(self, env: Any, context: FaultContext) -> None:
        del context
        self._ensure_simulator(env)

    def _clear_step_state(self) -> None:
        self._before_qpos = None
        self._pending_context = None

    def reset_runtime(self) -> None:
        """Reset subclass state at an episode or simulator-state boundary."""

    def on_reset(self, env: Any, context: FaultContext) -> None:
        del context
        self._ensure_simulator(env)
        self._clear_step_state()
        self.reset_runtime()

    def on_state_loaded(self, env: Any, context: FaultContext) -> None:
        del context
        self._ensure_simulator(env)
        self._clear_step_state()
        self.reset_runtime()

    def before_step(self, env: Any, action: Any, context: FaultContext) -> None:
        del action
        if self._suspend_depth:
            return
        self._ensure_simulator(env)
        if self._pending_context is not None:
            raise RuntimeError(f"{self.family} received a new step before the prior step ended.")
        if self._sim is None or self._qpos_indices is None:
            raise RuntimeError(f"{self.family} is not installed on a simulator.")
        self._before_qpos = np.asarray(
            self._sim.data.qpos[self._qpos_indices], dtype=np.float64
        ).copy()
        self._pending_context = (context.episode_index, context.step_index)

    def transform_realized_state(
        self,
        before_qpos: np.ndarray,
        candidate_qpos: np.ndarray,
        candidate_qvel: np.ndarray,
        *,
        context: FaultContext,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return faulted qpos/qvel and transition diagnostics."""
        raise NotImplementedError

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
            raise RuntimeError(f"{self.family} has no matching pre-step simulator snapshot.")
        self._ensure_simulator(env)
        if self._sim is None or self._qpos_indices is None or self._qvel_indices is None:
            raise RuntimeError(f"{self.family} is not installed on a simulator.")
        candidate_qpos = np.asarray(
            self._sim.data.qpos[self._qpos_indices], dtype=np.float64
        ).copy()
        candidate_qvel = np.asarray(
            self._sim.data.qvel[self._qvel_indices], dtype=np.float64
        ).copy()
        applied_qpos, applied_qvel, diagnostics = self.transform_realized_state(
            self._before_qpos,
            candidate_qpos,
            candidate_qvel,
            context=context,
        )
        applied_qpos = np.asarray(applied_qpos, dtype=np.float64)
        applied_qvel = np.asarray(applied_qvel, dtype=np.float64)
        if applied_qpos.shape != candidate_qpos.shape or applied_qvel.shape != candidate_qvel.shape:
            raise ValueError(f"{self.family} changed the selected-joint state shape.")
        if not np.all(np.isfinite(applied_qpos)) or not np.all(np.isfinite(applied_qvel)):
            raise ValueError(f"{self.family} produced non-finite joint state.")
        self._sim.data.qpos[self._qpos_indices] = applied_qpos
        self._sim.data.qvel[self._qvel_indices] = applied_qvel
        self._sim.forward()
        transition.info.setdefault("joint_state_faults", []).append(
            {
                "id": self.fault_id,
                "family": self.family,
                "joints": list(self.joint_names),
                "candidate_qpos_rad": candidate_qpos.tolist(),
                "applied_qpos_rad": applied_qpos.tolist(),
                **diagnostics,
            }
        )
        self._clear_step_state()

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
        if self._pending_context is not None:
            raise RuntimeError(f"Cannot suspend {self.family} during an active step.")
        self._suspend_depth += 1
        try:
            yield
        finally:
            self._suspend_depth -= 1

    def mutable_state_dict(self) -> dict[str, Any]:
        """Return subclass state for exact shadow-environment reproduction."""
        return {}

    def restore_mutable_state(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ValueError(f"{self.family} does not define mutable runtime state.")

    def state_dict(self) -> dict[str, Any]:
        if self._pending_context is not None:
            raise RuntimeError(f"Cannot checkpoint {self.family} during an active step.")
        return {"suspend_depth": self._suspend_depth, "runtime": self.mutable_state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("suspend_depth", 0)) != 0:
            raise ValueError(f"Cannot restore a suspended {self.family} fault.")
        self._suspend_depth = 0
        self._clear_step_state()
        self.restore_mutable_state(dict(state.get("runtime", {})))

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.fault_id,
            "family": self.family,
            "targets": list(self.joint_names),
            "operation": self.operation,
            "severity": self.severity.to_dict(),
            "parameters": self.parameters,
            "injection_layer": "faulted_environment_dynamics",
        }

    def detach(self, env: Any) -> None:
        del env
        self._sim = None
        self._qpos_indices = None
        self._qvel_indices = None
        self._clear_step_state()
        self._suspend_depth = 0
        self.reset_runtime()
