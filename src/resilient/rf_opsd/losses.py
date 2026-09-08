"""Future-only RF-OPSD vector-field losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def future_flow_loss(
    student_flow: torch.Tensor,
    teacher_flow: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    mse_weight: float = 1.0,
    smooth_l1_weight: float = 0.0,
    cosine_weight: float = 0.0,
    exclude_initial_latent: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare Student and frozen Teacher flow targets outside the fixed first latent."""
    if student_flow.shape != teacher_flow.shape or student_flow.ndim != 5:
        raise ValueError(
            "student_flow and teacher_flow must share [B,C,T,H,W], got "
            f"{tuple(student_flow.shape)} and {tuple(teacher_flow.shape)}"
        )
    if min(mse_weight, smooth_l1_weight, cosine_weight) < 0:
        raise ValueError("RF-OPSD loss weights must be non-negative.")
    if mse_weight + smooth_l1_weight + cosine_weight <= 0:
        raise ValueError("At least one RF-OPSD loss weight must be positive.")
    student = student_flow[:, :, 1:] if exclude_initial_latent else student_flow
    teacher = teacher_flow.detach()
    teacher = teacher[:, :, 1:] if exclude_initial_latent else teacher
    if student.shape[2] == 0:
        raise ValueError("RF-OPSD loss has no future latent after excluding the initial slice.")

    reduce_dims = tuple(range(1, student.ndim))
    mse = (student.float() - teacher.float()).square().mean(dim=reduce_dims)
    smooth_l1 = F.smooth_l1_loss(
        student.float(), teacher.float(), reduction="none"
    ).mean(dim=reduce_dims)
    cosine_distance = 1.0 - F.cosine_similarity(
        student.float().flatten(1), teacher.float().flatten(1), dim=1, eps=1e-8
    )
    per_sample = (
        float(mse_weight) * mse
        + float(smooth_l1_weight) * smooth_l1
        + float(cosine_weight) * cosine_distance
    )
    if sample_weight is None:
        weight = torch.ones_like(per_sample)
    else:
        weight = sample_weight.to(device=per_sample.device, dtype=per_sample.dtype).flatten()
        if weight.shape != per_sample.shape:
            raise ValueError(
                "sample_weight must have shape "
                f"{tuple(per_sample.shape)}, got {tuple(weight.shape)}"
            )
    loss = (weight * per_sample).mean()
    return loss, {
        "loss": loss.detach(),
        "mse": mse.mean().detach(),
        "smooth_l1": smooth_l1.mean().detach(),
        "cosine_distance": cosine_distance.mean().detach(),
    }
