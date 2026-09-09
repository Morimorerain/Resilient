"""FPO ratio and clipped surrogate objective.

The ratio follows McAllister et al., Flow Matching Policy Gradients (ICLR
2026). The implementation is an independent PyTorch adaptation of the
Apache-2.0 Playground reference implementation at akanazawa/fpo.
"""

from __future__ import annotations

import torch


def compute_fpo_ratio(
    old_cfm_loss: torch.Tensor,
    current_cfm_loss: torch.Tensor,
    *,
    log_ratio_clip: float | None = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average MC losses before exponentiating the old-minus-current loss."""
    if old_cfm_loss.shape != current_cfm_loss.shape:
        raise ValueError("Old and current CFM losses must have identical shapes.")
    if old_cfm_loss.ndim < 1:
        raise ValueError("CFM losses must include an MC-sample dimension.")
    log_ratio = old_cfm_loss.detach().float().mean(dim=-1) - current_cfm_loss.float().mean(
        dim=-1
    )
    if log_ratio_clip is not None:
        if log_ratio_clip <= 0:
            raise ValueError("log_ratio_clip must be positive when enabled.")
        log_ratio = log_ratio.clamp(-float(log_ratio_clip), float(log_ratio_clip))
    return torch.exp(log_ratio), log_ratio


def fpo_clipped_objective(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    *,
    clipping_epsilon: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the minimization form of the PPO-clipped FPO objective."""
    if not 0.0 < clipping_epsilon < 1.0:
        raise ValueError("clipping_epsilon must be in (0, 1).")
    ratio, advantage = torch.broadcast_tensors(ratio.float(), advantage.detach().float())
    surrogate = ratio * advantage
    clipped_ratio = ratio.clamp(1.0 - clipping_epsilon, 1.0 + clipping_epsilon)
    clipped_surrogate = clipped_ratio * advantage
    selected = torch.minimum(surrogate, clipped_surrogate)
    loss = -selected.mean()
    metrics = {
        "policy_loss": loss.detach(),
        "ratio_mean": ratio.detach().mean(),
        "ratio_min": ratio.detach().min(),
        "ratio_max": ratio.detach().max(),
        "clip_fraction": (ratio.detach().sub(1.0).abs() > clipping_epsilon)
        .float()
        .mean(),
    }
    return loss, metrics
