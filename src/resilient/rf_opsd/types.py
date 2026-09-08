"""Typed values for RF-OPSD rollout and training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RealizedFutureWindow:
    """One Student action chunk realized under the configured Fault dynamics."""

    video: torch.Tensor
    action: torch.Tensor
    frame_steps: tuple[int, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.video.ndim != 5:
            raise ValueError(f"video must be [B,C,T,H,W], got {tuple(self.video.shape)}")
        if self.action.ndim != 3:
            raise ValueError(f"action must be [B,T,D], got {tuple(self.action.shape)}")
        if self.video.shape[2] != len(self.frame_steps):
            raise ValueError("Video frame count does not match frame_steps.")
        if not self.frame_steps or self.frame_steps[0] != 0:
            raise ValueError("frame_steps must begin at zero.")


@dataclass(frozen=True)
class VideoFlowBatch:
    """Inputs used to score one realized-future latent without action learning."""

    target_latents: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    proprio: torch.Tensor | None = None

