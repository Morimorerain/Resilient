"""Fault runtime lifecycle shared by visual, action, and dynamics faults."""

from __future__ import annotations

from abc import ABC
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FaultContext:
    """Deterministic runtime coordinates supplied to every fault hook."""

    seed: int
    episode_index: int = 0
    step_index: int = 0


class FaultRuntime(ABC):
    """Per-environment stateful fault instance with optional lifecycle hooks."""

    family: str
    fault_id: str
    scopes: frozenset[str] = frozenset()

    def attach(self, env: Any, context: FaultContext) -> None:
        """Attach to an environment without changing baseline state."""

    def on_reset(self, env: Any, context: FaultContext) -> None:
        """Apply reset-sensitive environment mutations."""

    def transform_observation(
        self,
        observation: Any,
        env: Any,
        context: FaultContext,
    ) -> Any:
        """Return the observation exposed to the policy."""
        return observation

    def transform_action(self, action: Any, env: Any, context: FaultContext) -> Any:
        """Return the action executed by the environment."""
        return action

    def before_step(self, env: Any, action: Any, context: FaultContext) -> None:
        """Run immediately before the environment step."""

    def after_step(
        self,
        env: Any,
        transition: Any,
        context: FaultContext,
    ) -> None:
        """Run immediately after the environment step."""

    def suspend(
        self,
        env: Any,
        scopes: frozenset[str],
        context: FaultContext,
    ) -> AbstractContextManager[None]:
        """Temporarily disable selected capabilities without advancing the environment."""
        del env, scopes, context
        return nullcontext()

    def state_dict(self) -> dict[str, Any]:
        """Return deterministic mutable state for checkpointing."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore deterministic mutable state from a checkpoint."""
        if state:
            raise ValueError(f"{type(self).__name__} does not define restorable state.")

    def metadata(self) -> dict[str, Any]:
        """Return a portable description of this resolved fault instance."""
        raise NotImplementedError

    def slug(self) -> str:
        """Return a deterministic filesystem-safe instance identifier."""
        raise NotImplementedError

    def detach(self, env: Any) -> None:
        """Restore environment-owned state before disposal."""
