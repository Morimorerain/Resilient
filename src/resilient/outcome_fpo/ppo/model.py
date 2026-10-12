"""Training-only value head and the existing Fast-WAM FPO action scorer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from resilient.outcome_fpo.policy import FastWAMFPOScoringModule
from resilient.outcome_fpo.types import ActionConditioning


class ConditionValueHead(nn.Module):
    """Estimate future task-direction return from information available at decision time."""

    def __init__(self, *, context_dim: int, proprio_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proprio_dim = int(proprio_dim)
        self.network = nn.Sequential(
            nn.Linear(int(context_dim) + 3 * 8 * 8 + self.proprio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, conditioning: ActionConditioning) -> torch.Tensor:
        image = conditioning.input_image.float()
        if image.shape[:2] != (1, 3):
            raise ValueError("Value head expects one RGB Fast-WAM image.")
        image_features = F.adaptive_avg_pool2d(image, (8, 8)).flatten(1)
        context = conditioning.context.float()
        mask = conditioning.context_mask.float().unsqueeze(-1)
        context_features = (context * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        if conditioning.proprio is None:
            if self.proprio_dim:
                raise ValueError("Configured value head requires proprio input.")
            features = torch.cat((image_features, context_features), dim=1)
        else:
            if conditioning.proprio.shape != (1, self.proprio_dim):
                raise ValueError("Value head proprio shape does not match Fast-WAM.")
            features = torch.cat(
                (image_features, context_features, conditioning.proprio.float()), dim=1
            )
        return self.network(features.to(dtype=self.network[0].weight.dtype)).flatten()


class ActorCriticFPOScoringModule(FastWAMFPOScoringModule):
    """Keep actor replay and critic in one distributed module for synchronized updates."""

    def __init__(self, fastwam: nn.Module, *, hidden_dim: int) -> None:
        super().__init__(fastwam)
        self.value_head = ConditionValueHead(
            context_dim=int(fastwam.text_dim),
            proprio_dim=int(fastwam.proprio_dim or 0),
            hidden_dim=int(hidden_dim),
        )

    def forward(
        self,
        conditioning: ActionConditioning,
        action: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfm_loss = super().forward(conditioning, action, noise, timestep)
        value = self.value_head(
            conditioning.to(device=self.fastwam.device, dtype=self.fastwam.torch_dtype)
        )
        return cfm_loss, value

    @torch.no_grad()
    def rollout_value(self, conditioning: ActionConditioning) -> float:
        value = self.value_head(
            conditioning.to(device=self.fastwam.device, dtype=self.fastwam.torch_dtype)
        )
        return float(value.item())
