"""Policy-independent collection of counterfactual action-chunk outcomes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FutureRollout:
    """Frames realized by one action chunk in a same-state Fault environment."""

    observations: tuple[Any, ...]
    frame_steps: tuple[int, ...]
    success: bool


def validate_frame_steps(frame_steps: Sequence[int], action_horizon: int) -> tuple[int, ...]:
    """Validate an increasing frame schedule that spans the complete action chunk."""
    steps = tuple(int(value) for value in frame_steps)
    if not steps or steps[0] != 0 or steps[-1] != int(action_horizon):
        raise ValueError("Frame steps must start at 0 and end at action_horizon.")
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError("Frame steps must be strictly increasing.")
    return steps


def restore_shadow_state(
    shadow_env: Any,
    *,
    simulator_state: Any,
    fault_runtime_state: dict[str, Any],
) -> Any:
    """Restore simulator and Fault coordinates before a candidate rollout."""
    shadow_env.reset()
    shadow_env.set_init_state(simulator_state)
    loader = getattr(shadow_env, "load_fault_runtime_state_dict", None)
    if not callable(loader):
        raise TypeError("Shadow environment does not expose Fault runtime restoration.")
    return loader(fault_runtime_state)


def collect_action_chunk_future(
    env: Any,
    initial_observation: Any,
    action_chunk: Sequence[Any],
    *,
    frame_steps: Sequence[int],
) -> FutureRollout:
    """Execute all actions even after success so every candidate has nine aligned frames."""
    steps = validate_frame_steps(frame_steps, len(action_chunk))
    observations = [initial_observation]
    selected = set(steps[1:])
    success = False
    for step_index, action in enumerate(action_chunk, start=1):
        observation, _, done, _ = env.step(action)
        success = success or bool(done)
        if step_index in selected:
            observations.append(observation)
    if len(observations) != len(steps):
        raise RuntimeError("The candidate rollout did not produce every requested frame.")
    return FutureRollout(tuple(observations), steps, success)


def stack_observation_video(
    observations: Sequence[Any],
    image_encoder: Callable[[Any], torch.Tensor],
) -> torch.Tensor:
    """Apply the standard Fast-WAM image path and return [1,C,T,H,W]."""
    frames: list[torch.Tensor] = []
    for observation in observations:
        image = image_encoder(observation)
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("image_encoder must return [1,C,H,W].")
        frames.append(image[0].detach().cpu())
    if not frames:
        raise ValueError("At least one observation is required.")
    return torch.stack(frames, dim=1).unsqueeze(0)
