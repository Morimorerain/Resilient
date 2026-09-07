"""Ordered composition of heterogeneous robot faults."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any

from .base import FaultContext, FaultRuntime
from .types import FaultTransition


class FaultPipeline:
    """Apply environment, observation, action, and dynamics faults in declared order."""

    def __init__(self, pipeline_id: str, seed: int, faults: Sequence[FaultRuntime]):
        self.pipeline_id = str(pipeline_id)
        self.seed = int(seed)
        self.faults = tuple(faults)
        ids = [fault.fault_id for fault in self.faults]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Fault IDs must be unique within a pipeline: {ids}")
        self._attached_environment_id: int | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.faults)

    @property
    def requires_post_step_observation_refresh(self) -> bool:
        """Return whether a fault mutates simulator state after ``env.step``."""
        return any(fault.requires_post_step_observation_refresh for fault in self.faults)

    def context(self, episode_index: int = 0, step_index: int = 0) -> FaultContext:
        return FaultContext(self.seed, int(episode_index), int(step_index))

    def attach(self, env: Any, context: FaultContext) -> None:
        env_id = id(env)
        if self._attached_environment_id == env_id:
            return
        if self._attached_environment_id is not None:
            raise RuntimeError("A FaultPipeline instance cannot be shared across environments.")
        for fault in self.faults:
            fault.attach(env, context)
        self._attached_environment_id = env_id

    def on_reset(self, env: Any, context: FaultContext) -> None:
        self.attach(env, context)
        for fault in self.faults:
            fault.on_reset(env, context)

    def on_state_loaded(self, env: Any, context: FaultContext) -> None:
        for fault in self.faults:
            fault.on_state_loaded(env, context)

    def transform_observation(self, observation: Any, env: Any, context: FaultContext) -> Any:
        transformed = copy.deepcopy(observation)
        for fault in self.faults:
            transformed = fault.transform_observation(transformed, env, context)
        return transformed

    def transform_action(self, action: Any, env: Any, context: FaultContext) -> Any:
        transformed = copy.deepcopy(action)
        for fault in self.faults:
            transformed = fault.transform_action(transformed, env, context)
        return transformed

    def before_step(self, env: Any, action: Any, context: FaultContext) -> None:
        for fault in self.faults:
            fault.before_step(env, action, context)

    def after_step(self, env: Any, transition: FaultTransition, context: FaultContext) -> None:
        for fault in reversed(self.faults):
            fault.after_step(env, transition, context)

    @contextmanager
    def suspend(
        self,
        env: Any,
        scopes: Sequence[str] = ("visual", "observation"),
        context: FaultContext | None = None,
    ) -> Iterator[None]:
        selected = frozenset(str(scope) for scope in scopes)
        active_context = context or self.context()
        with ExitStack() as stack:
            for fault in reversed(self.faults):
                if selected.intersection(fault.scopes):
                    stack.enter_context(fault.suspend(env, selected, active_context))
            yield

    def state_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "seed": self.seed,
            "faults": {fault.fault_id: fault.state_dict() for fault in self.faults},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("pipeline_id") != self.pipeline_id or int(state.get("seed")) != self.seed:
            raise ValueError("Fault pipeline checkpoint does not match the active pipeline.")
        fault_states = state.get("faults", {})
        if set(fault_states) != {fault.fault_id for fault in self.faults}:
            raise ValueError("Fault pipeline checkpoint contains a different fault set.")
        for fault in self.faults:
            fault.load_state_dict(fault_states[fault.fault_id])

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "pipeline_id": self.pipeline_id,
            "seed": self.seed,
            "enabled": self.enabled,
            "activation": "environment_startup",
            "faults": [fault.metadata() for fault in self.faults],
        }

    def slug(self) -> str:
        if not self.faults:
            return "none"
        return "__".join(fault.slug() for fault in self.faults)

    def detach(self, env: Any) -> None:
        for fault in reversed(self.faults):
            fault.detach(env)
        self._attached_environment_id = None
