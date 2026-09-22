"""Frozen Fast-WAM VAE task-region residuals."""

from __future__ import annotations

import torch


@torch.no_grad()
def encode_task_residual(
    model,
    full_video: torch.Tensor,
    counterfactual_video: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode full/counterfactual videos in one batch and return future-only delta."""
    if full_video.shape != counterfactual_video.shape or full_video.ndim != 5:
        raise ValueError("Videos must share [B,C,T,H,W] shape.")
    vae_model = model.vae.model
    if any(parameter.requires_grad for parameter in vae_model.parameters()):
        raise RuntimeError("Task residuals require a completely frozen Fast-WAM VAE.")
    combined = torch.cat([full_video, counterfactual_video], dim=0).to(
        device=model.device,
        dtype=model.torch_dtype,
    )
    latent = model._encode_video_latents(combined, tiled=False).detach().float().cpu()
    batch = full_video.shape[0]
    full_latent, counterfactual_latent = latent[:batch], latent[batch:]
    delta = full_latent - counterfactual_latent
    if delta.shape[2] < 2:
        raise ValueError("Encoded video has no future latent slice.")
    return full_latent, counterfactual_latent, delta, delta[:, :, 1:]
