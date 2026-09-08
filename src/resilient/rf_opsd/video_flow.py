"""Differentiable Fast-WAM video-only flow scoring for RF-OPSD."""

from __future__ import annotations

import torch
import torch.nn as nn

from .losses import future_flow_loss
from .types import VideoFlowBatch


class RFVideoFlowModule(nn.Module):
    """Train the Video DiT while leaving the action expert outside the graph."""

    def __init__(
        self,
        fastwam: nn.Module,
        *,
        mse_weight: float,
        smooth_l1_weight: float,
        cosine_weight: float,
        exclude_initial_latent: bool,
    ) -> None:
        super().__init__()
        self.fastwam = fastwam
        self.mse_weight = float(mse_weight)
        self.smooth_l1_weight = float(smooth_l1_weight)
        self.cosine_weight = float(cosine_weight)
        self.exclude_initial_latent = bool(exclude_initial_latent)

    def forward(self, batch: VideoFlowBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model = self.fastwam
        target_latents = batch.target_latents.detach().to(
            device=model.device, dtype=model.torch_dtype
        )
        context = batch.context.to(device=model.device, dtype=model.torch_dtype)
        context_mask = batch.context_mask.to(device=model.device, dtype=torch.bool)
        if batch.proprio is not None:
            context, context_mask = model._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=batch.proprio.to(device=model.device, dtype=model.torch_dtype),
            )

        batch_size = target_latents.shape[0]
        noise = torch.randn_like(target_latents)
        timestep = model.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=model.device,
            dtype=target_latents.dtype,
        )
        noisy_latents = model.train_video_scheduler.add_noise(
            target_latents, noise, timestep
        )
        noisy_latents[:, :, :1] = target_latents[:, :, :1]
        teacher_flow = model.train_video_scheduler.training_target(
            target_latents, noise, timestep
        ).detach()
        student_flow = model.video_expert(
            x=noisy_latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
        )
        sample_weight = model.train_video_scheduler.training_weight(timestep)
        return future_flow_loss(
            student_flow,
            teacher_flow,
            sample_weight=sample_weight,
            mse_weight=self.mse_weight,
            smooth_l1_weight=self.smooth_l1_weight,
            cosine_weight=self.cosine_weight,
            exclude_initial_latent=self.exclude_initial_latent,
        )
