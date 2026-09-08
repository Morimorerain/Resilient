"""Realized-future collection that is independent of policy implementations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FutureRollout:
    """Observations sampled while a complete Student action chunk is executed."""

    observations: tuple[Any, ...]
    frame_steps: tuple[int, ...]
    steps_taken: int
    success: bool


def validate_frame_steps(frame_steps: Sequence[int], action_horizon: int) -> tuple[int, ...]:
    """Return a strict, increasing frame schedule spanning the action horizon."""
    steps = tuple(int(value) for value in frame_steps)
    if not steps or steps[0] != 0:
        raise ValueError("future frame steps must begin at zero.")
    if steps[-1] != int(action_horizon):
        raise ValueError("future frame steps must end at action_horizon.")
    if any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError("future frame steps must be strictly increasing.")
    return steps


def collect_action_chunk_future(
    env: Any,
    initial_observation: Any,
    action_chunk: Sequence[Any],
    *,
    frame_steps: Sequence[int],
) -> FutureRollout:
    """Execute a Student chunk and sample observations at configured step boundaries."""
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
        raise RuntimeError(
            f"Collected {len(observations)} frames for {len(steps)} requested boundaries."
        )
    return FutureRollout(
        observations=tuple(observations),
        frame_steps=steps,
        steps_taken=len(action_chunk),
        success=success,
    )


def stack_observation_video(
    observations: Sequence[Any],
    image_encoder: Callable[[Any], torch.Tensor],
) -> torch.Tensor:
    """Encode observations with the evaluation image path and stack [B,C,T,H,W]."""
    frames = []
    for observation in observations:
        image = image_encoder(observation)
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(
                "image_encoder must return one frame shaped [1,C,H,W], "
                f"got {tuple(image.shape)}"
            )
        frames.append(image[0])
    if not frames:
        raise ValueError("At least one observation is required.")
    return torch.stack(frames, dim=1).unsqueeze(0)
