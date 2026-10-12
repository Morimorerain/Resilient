"""Causal, chunk-level returns and generalized advantage estimates."""

from __future__ import annotations

import torch


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    bootstrap_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE over action chunks, not over individual simulator steps."""
    if rewards.ndim != 1 or values.shape != rewards.shape or dones.shape != rewards.shape:
        raise ValueError("rewards, values, and dones must be equal-length vectors.")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in [0, 1].")
    advantages = torch.empty_like(rewards, dtype=torch.float32)
    next_value = torch.as_tensor(bootstrap_value, dtype=torch.float32)
    next_advantage = torch.zeros((), dtype=torch.float32)
    for index in range(len(rewards) - 1, -1, -1):
        continuation = 1.0 - dones[index].float()
        delta = rewards[index] + gamma * next_value * continuation - values[index]
        next_advantage = delta + gamma * gae_lambda * continuation * next_advantage
        advantages[index] = next_advantage
        next_value = values[index]
    return advantages, advantages + values.float()
