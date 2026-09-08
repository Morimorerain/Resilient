"""Transparent environment ownership for simulator-level fault pipelines."""

from __future__ import annotations

import copy
from typing import Any

from .pipeline import FaultPipeline
from .types import FaultTransition


def capture_current_observation(env: Any) -> Any:
    """Capture an observation directly from a wrapped simulator environment."""
    current = env
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        method = getattr(current, "_get_observations", None)
        if callable(method):
            try:
                return method(force_update=True)
            except TypeError:
                return method()
        current = getattr(current, "env", None)
        if current is None:
            break
    raise AttributeError("Environment does not expose _get_observations().")


class FaultedEnvironment:
    """Expose the normal environment API while owning all fault lifecycle hooks."""

    def __init__(
        self,
        environment: Any,
        pipeline: FaultPipeline,
        *,
        initial_episode_index: int = 0,
    ) -> None:
        if not pipeline.enabled:
            raise ValueError("FaultedEnvironment requires an enabled fault pipeline.")
        self._environment = environment
        self.fault_pipeline = pipeline
        self._next_episode_index = int(initial_episode_index)
        self._episode_index = int(initial_episode_index)
        self._step_index = 0
        self._raw_observation: Any | None = None
        self._observation: Any | None = None
        self._last_transition: FaultTransition | None = None
        self._closed = False
        self.fault_pipeline.attach(self._environment, self._context())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    def _context(self):
        return self.fault_pipeline.context(self._episode_index, self._step_index)

    @property
    def raw_environment(self) -> Any:
        """Return the underlying environment for diagnostics, not policy control."""
        return self._environment

    @property
    def fault_metadata(self) -> dict[str, Any]:
        return self.fault_pipeline.metadata()

    @property
    def raw_observation(self) -> Any:
        return self._raw_observation

    @property
    def last_transition(self) -> FaultTransition | None:
        return self._last_transition

    def fault_runtime_state_dict(self) -> dict[str, Any]:
        """Return portable Fault coordinates for a same-state shadow environment."""
        return {
            "schema_version": 1,
            "episode_index": self._episode_index,
            "next_episode_index": self._next_episode_index,
            "step_index": self._step_index,
            "pipeline": copy.deepcopy(self.fault_pipeline.state_dict()),
        }

    def load_fault_runtime_state_dict(self, state: dict[str, Any]) -> Any:
        """Synchronize Fault time/state without loading another simulator state."""
        if int(state.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported FaultedEnvironment runtime-state schema.")
        episode_index = int(state["episode_index"])
        next_episode_index = int(state["next_episode_index"])
        step_index = int(state["step_index"])
        if min(episode_index, next_episode_index, step_index) < 0:
            raise ValueError("Fault runtime coordinates must be non-negative.")
        self.fault_pipeline.load_state_dict(copy.deepcopy(state["pipeline"]))
        self._episode_index = episode_index
        self._next_episode_index = next_episode_index
        self._step_index = step_index
        self._last_transition = None
        return self._publish_observation(capture_current_observation(self._environment))

    def _publish_observation(self, raw_observation: Any) -> Any:
        self._raw_observation = raw_observation
        self._observation = self.fault_pipeline.transform_observation(
            raw_observation,
            self._environment,
            self._context(),
        )
        return self._observation

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Reset the simulator and reinstall reset-sensitive faults before returning."""
        self._episode_index = self._next_episode_index
        self._next_episode_index += 1
        self._step_index = 0
        self._last_transition = None
        self._environment.reset(*args, **kwargs)
        self.fault_pipeline.on_reset(self._environment, self._context())
        return self._publish_observation(capture_current_observation(self._environment))

    def set_init_state(self, initial_state: Any) -> Any:
        """Load one simulator state while preserving the installed fault pipeline."""
        raw_observation = self._environment.set_init_state(initial_state)
        self.fault_pipeline.on_state_loaded(self._environment, self._context())
        return self._publish_observation(raw_observation)

    def regenerate_obs_from_state(self, simulator_state: Any) -> Any:
        """Mirror LIBERO's state-loading helper without exposing unfaulted observations."""
        raw_observation = self._environment.regenerate_obs_from_state(simulator_state)
        self.fault_pipeline.on_state_loaded(self._environment, self._context())
        return self._publish_observation(raw_observation)

    def step(self, policy_action: Any):
        """Execute one standard environment step with the installed fault composition."""
        context = self._context()
        executed_action = self.fault_pipeline.transform_action(
            policy_action,
            self._environment,
            context,
        )
        self.fault_pipeline.before_step(self._environment, executed_action, context)
        next_raw, reward, done, info = self._environment.step(executed_action)
        next_observation = self.fault_pipeline.transform_observation(
            next_raw,
            self._environment,
            context,
        )
        transition = FaultTransition(
            raw_observation=self._raw_observation,
            student_observation=self._observation,
            policy_action=policy_action,
            executed_action=executed_action,
            next_raw_observation=next_raw,
            next_student_observation=next_observation,
            reward=float(reward),
            done=bool(done),
            info={} if info is None else dict(info),
            fault_metadata=self.fault_pipeline.metadata()["faults"],
        )
        self.fault_pipeline.after_step(self._environment, transition, context)
        if self.fault_pipeline.requires_post_step_observation_refresh:
            next_raw = capture_current_observation(self._environment)
            next_observation = self.fault_pipeline.transform_observation(
                next_raw,
                self._environment,
                context,
            )
            transition.next_raw_observation = next_raw
            transition.next_student_observation = next_observation
            transition.done = bool(self._environment.check_success())

        self._raw_observation = next_raw
        self._observation = next_observation
        self._last_transition = transition
        self._step_index += 1
        return next_observation, reward, bool(transition.done), transition.info

    def close(self) -> None:
        """Detach faults before releasing simulator-owned resources."""
        if self._closed:
            return
        self.fault_pipeline.detach(self._environment)
        close = getattr(self._environment, "close", None)
        if callable(close):
            close()
        self._closed = True


def install_fault_pipeline(
    environment: Any,
    pipeline: FaultPipeline,
    *,
    initial_episode_index: int = 0,
) -> Any:
    """Return a transparent fault-owning environment or the unchanged baseline."""
    if not pipeline.enabled:
        return environment
    return FaultedEnvironment(
        environment,
        pipeline,
        initial_episode_index=initial_episode_index,
    )
