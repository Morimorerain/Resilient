"""Typed rollout values for outcome-guided FPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ActionConditioning:
    """Fast-WAM inputs that are available to the deployed Student policy."""

    input_image: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    proprio: torch.Tensor | None = None

    def to(self, device: torch.device | str, dtype: torch.dtype) -> ActionConditioning:
        """Move floating inputs while preserving the boolean attention mask."""
        return ActionConditioning(
            input_image=self.input_image.to(device=device, dtype=dtype),
            context=self.context.to(device=device, dtype=dtype),
            context_mask=self.context_mask.to(device=device, dtype=torch.bool),
            proprio=(
                None if self.proprio is None else self.proprio.to(device=device, dtype=dtype)
            ),
        )

    def cpu(self) -> ActionConditioning:
        """Detach rollout conditioning for the short-lived on-policy buffer."""
        return ActionConditioning(
            input_image=self.input_image.detach().cpu(),
            context=self.context.detach().cpu(),
            context_mask=self.context_mask.detach().cpu(),
            proprio=None if self.proprio is None else self.proprio.detach().cpu(),
        )


@dataclass(frozen=True)
class CandidateSample:
    """One sampled action and the frozen FPO statistics from its rollout policy."""

    normalized_action: torch.Tensor
    reward: float
    mc_noise: torch.Tensor
    mc_timestep: torch.Tensor
    old_cfm_loss: torch.Tensor
    inference_seed: int
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.normalized_action.ndim != 2:
            raise ValueError("normalized_action must have shape [T,D].")
        if self.mc_noise.ndim != 3:
            raise ValueError("mc_noise must have shape [N,T,D].")
        if self.mc_noise.shape[1:] != self.normalized_action.shape:
            raise ValueError("MC noise and action shapes do not match.")
        if self.mc_timestep.shape != (self.mc_noise.shape[0],):
            raise ValueError("mc_timestep must have one value per MC noise sample.")
        if self.old_cfm_loss.shape != (self.mc_noise.shape[0],):
            raise ValueError("old_cfm_loss must have one value per MC sample.")


@dataclass(frozen=True)
class OutcomeGroup:
    """Candidates generated from one identical current simulator state."""

    conditioning: ActionConditioning
    candidates: tuple[CandidateSample, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if len(self.candidates) < 2:
            raise ValueError("Outcome-guided FPO requires at least two same-state candidates.")

    @property
    def rewards(self) -> torch.Tensor:
        return torch.tensor([item.reward for item in self.candidates], dtype=torch.float32)
