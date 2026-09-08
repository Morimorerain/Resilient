"""Frozen Fast-WAM VAE used as a realized-future representation Teacher."""

from __future__ import annotations

import torch


class FrozenFastWAMVAETeacher:
    """Encode realized videos in Fast-WAM's native scaled latent space."""

    def __init__(self, fastwam) -> None:
        self.fastwam = fastwam
        self.fastwam.vae.requires_grad_(False)
        self.fastwam.vae.eval()

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Return a detached target latent without changing the video content."""
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
        latent = self.fastwam._encode_video_latents(
            video.to(device=self.fastwam.device, dtype=self.fastwam.torch_dtype),
            tiled=False,
        )
        if latent.requires_grad:
            raise RuntimeError("Frozen VAE Teacher unexpectedly produced a grad-enabled target.")
        return latent.detach()

    def audit(self) -> dict[str, int | bool]:
        """Describe and verify the frozen Teacher parameters."""
        parameters = list(self.fastwam.vae.parameters())
        trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        if trainable:
            raise RuntimeError("Frozen VAE Teacher contains trainable parameters.")
        return {
            "parameter_count": sum(parameter.numel() for parameter in parameters),
            "trainable_parameter_count": trainable,
            "training": bool(self.fastwam.vae.training),
        }

