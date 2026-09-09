"""State-matched control variates for outcome-guided FPO."""

from __future__ import annotations

import torch


def leave_one_out_advantages(
    rewards: torch.Tensor,
    *,
    eps: float = 1e-6,
    clip: float | None = 5.0,
    minimum_std: float = 1e-4,
) -> tuple[torch.Tensor, bool]:
    """Return leave-one-out standardized advantages and whether to update.

    The baseline for candidate ``i`` excludes its own reward. This keeps the
    same-state baseline independent of that candidate's sampled action.
    """
    values = rewards.detach().float().flatten()
    if values.numel() < 2:
        raise ValueError("At least two rewards are required for leave-one-out advantages.")
    if not torch.isfinite(values).all():
        raise ValueError("Rewards must be finite.")
    std = values.std(unbiased=False)
    if float(std) < float(minimum_std):
        return torch.zeros_like(values), False
    baseline = (values.sum() - values) / float(values.numel() - 1)
    advantages = (values - baseline) / (std + float(eps))
    if clip is not None:
        if clip <= 0:
            raise ValueError("Advantage clip must be positive when enabled.")
        advantages = advantages.clamp(min=-float(clip), max=float(clip))
    return advantages, True
