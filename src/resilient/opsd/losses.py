"""Continuous trajectory-scoring loss used by OPSD-Flow."""

from __future__ import annotations

import torch


def pointwise_flow_step_loss(
    student_flow: torch.Tensor,
    teacher_flow: torch.Tensor,
    *,
    action_valid_mask: torch.Tensor | None = None,
    pointwise_clip: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return one unweighted ``[B,H,D]`` trajectory-position loss."""
    if student_flow.shape != teacher_flow.shape or student_flow.ndim != 3:
        raise ValueError("One flow step must share shape [B,H,D].")
    batch_size, horizon, action_dim = student_flow.shape
    raw = (student_flow.float() - teacher_flow.detach().float()).square()
    if action_valid_mask is None:
        valid = torch.ones((batch_size, horizon), device=raw.device, dtype=raw.dtype)
    else:
        if action_valid_mask.shape != (batch_size, horizon):
            raise ValueError("action_valid_mask must have shape [B,H].")
        valid = action_valid_mask.to(device=raw.device, dtype=raw.dtype)
    if pointwise_clip is not None and pointwise_clip <= 0:
        raise ValueError("pointwise_clip must be positive when enabled.")
    clipped = raw.clamp(max=float(pointwise_clip)) if pointwise_clip is not None else raw
    expanded_valid = valid[:, :, None]
    denominator = (valid.sum(dim=1) * float(action_dim)).clamp(min=1.0)
    per_sample = (clipped * expanded_valid).sum(dim=(1, 2)) / denominator
    selected = expanded_valid.expand_as(raw).bool()
    clip_fraction = (
        (raw[selected] > float(pointwise_clip)).float().mean()
        if pointwise_clip is not None
        else torch.zeros((), device=raw.device, dtype=raw.dtype)
    )
    return per_sample.mean(), {
        "raw": (raw * expanded_valid).sum() / denominator.sum(),
        "clip_fraction": clip_fraction,
    }


def opsd_flow_loss(
    student_flow: torch.Tensor,
    teacher_flow: torch.Tensor,
    delta_sigma: torch.Tensor,
    *,
    action_valid_mask: torch.Tensor | None = None,
    pointwise_clip: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match vector fields on Student latents with normalized sigma quadrature.

    ``student_flow`` and ``teacher_flow`` are ``[B,K,H,D]``. The Teacher is
    detached here as a second safety barrier in addition to its no-grad forward.
    """
    if student_flow.shape != teacher_flow.shape or student_flow.ndim != 4:
        raise ValueError(
            "student_flow and teacher_flow must share shape [B,K,H,D], got "
            f"{tuple(student_flow.shape)} and {tuple(teacher_flow.shape)}."
        )
    batch_size, num_steps, horizon, action_dim = student_flow.shape
    if delta_sigma.ndim == 1:
        if delta_sigma.shape[0] != num_steps:
            raise ValueError("delta_sigma length must match the trajectory step count.")
        delta_sigma = delta_sigma.unsqueeze(0).expand(batch_size, -1)
    if delta_sigma.shape != (batch_size, num_steps):
        raise ValueError(
            f"delta_sigma must be [K] or [B,K], got {tuple(delta_sigma.shape)}."
        )
    if pointwise_clip is not None and pointwise_clip <= 0:
        raise ValueError("pointwise_clip must be positive when enabled.")

    raw_error = (student_flow.float() - teacher_flow.detach().float()).square()
    if action_valid_mask is None:
        valid = torch.ones(
            (batch_size, horizon), device=raw_error.device, dtype=raw_error.dtype
        )
    else:
        if action_valid_mask.shape != (batch_size, horizon):
            raise ValueError(
                f"action_valid_mask must be [B,H], got {tuple(action_valid_mask.shape)}."
            )
        valid = action_valid_mask.to(device=raw_error.device, dtype=raw_error.dtype)

    clipped_error = (
        raw_error.clamp(max=float(pointwise_clip))
        if pointwise_clip is not None
        else raw_error
    )
    expanded_valid = valid[:, None, :, None]
    valid_coordinates = valid.sum(dim=1).clamp(min=1.0) * float(action_dim)
    step_loss = (clipped_error * expanded_valid).sum(dim=(2, 3)) / valid_coordinates[:, None]

    sigma_weight = delta_sigma.abs().to(device=raw_error.device, dtype=raw_error.dtype)
    sigma_weight = sigma_weight / sigma_weight.sum(dim=1, keepdim=True).clamp(min=1e-12)
    per_sample = (step_loss * sigma_weight).sum(dim=1)
    loss = per_sample.mean()

    if pointwise_clip is None:
        clip_fraction = torch.zeros((), device=raw_error.device, dtype=raw_error.dtype)
    else:
        selected = expanded_valid.expand_as(raw_error).bool()
        clip_fraction = (raw_error[selected] > float(pointwise_clip)).float().mean()
    metrics = {
        "loss_raw": (raw_error * expanded_valid).sum()
        / (valid.sum() * float(num_steps * action_dim)).clamp(min=1.0),
        "loss_clipped": loss.detach(),
        "clip_fraction": clip_fraction.detach(),
    }
    return loss, metrics
