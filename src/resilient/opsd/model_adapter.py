"""Fast-WAM action vector-field scoring without running the scheduler."""

from __future__ import annotations

import torch
import torch.nn as nn

from .types import ActionConditioning, StudentDenoisingStep


class FastWAMActionFlowScorer:
    """Evaluate one Student latent under arbitrary visual conditioning.

    This class deliberately calls no scheduler step and therefore cannot create
    a Teacher trajectory. It reuses Fast-WAM's action denoiser exactly.
    """

    def __init__(self, model):
        self.model = model

    def _context(self, conditioning: ActionConditioning) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.model
        if conditioning.prompt is not None:
            context, context_mask = model.encode_prompt(conditioning.prompt)
        else:
            if conditioning.context is None or conditioning.context_mask is None:
                raise ValueError("Precomputed context and mask are incomplete.")
            context = conditioning.context
            context_mask = conditioning.context_mask
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context = context.to(device=model.device, dtype=model.torch_dtype)
            context_mask = context_mask.to(device=model.device, dtype=torch.bool)
        if conditioning.proprio is not None:
            proprio = conditioning.proprio
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            context, context_mask = model._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=model.device, dtype=model.torch_dtype),
            )
        return context, context_mask

    def score(
        self,
        conditioning: ActionConditioning,
        step: StudentDenoisingStep,
    ) -> torch.Tensor:
        """Return the vector field at a fixed Student latent and timestep."""
        model = self.model
        latent = step.latent.to(device=model.device, dtype=model.torch_dtype)
        image = conditioning.input_image
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(device=model.device, dtype=model.torch_dtype)
        if image.shape[0] not in {1, latent.shape[0]}:
            raise ValueError("Conditioning image batch must be one or match the latent batch.")
        if image.shape[0] > 1 and not torch.equal(image, image[:1].expand_as(image)):
            raise ValueError(
                "Batched image scoring currently requires repeated copies of one frame."
            )
        first_frame_latents = model._encode_input_image_latents_tensor(image[:1])
        if latent.shape[0] > 1:
            first_frame_latents = first_frame_latents.expand(latent.shape[0], -1, -1, -1, -1)
        context, context_mask = self._context(conditioning)
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            device=model.device,
            dtype=first_frame_latents.dtype,
        )
        (
            video_tokens,
            _t_video,
            video_t_mod,
            video_context,
            video_context_mask,
            video_freqs,
            _f_video,
            _h_video,
            _w_video,
            tokens_per_frame,
        ) = model.video_expert.prepare(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)
            ),
        )
        video_seq_len = int(video_tokens.shape[1])
        timestep = step.timestep.to(device=model.device, dtype=model.torch_dtype)
        attention_mask = model._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=int(latent.shape[1]),
            video_tokens_per_frame=tokens_per_frame,
            device=video_tokens.device,
        )
        video_cache_k, video_cache_v = model.mot.prefill_video_cache_tensor(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context=video_context,
            video_context_mask=video_context_mask,
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        return model._denoise_action_with_video_cache(
            latents_action=latent,
            timestep_action=timestep,
            context=context,
            context_mask=context_mask,
            video_cache_k=video_cache_k,
            video_cache_v=video_cache_v,
            action_attention_mask=attention_mask[video_seq_len:, :],
        )


class OPSDScoringModule(nn.Module):
    """Expose flow scoring through ``forward`` so DDP synchronizes LoRA gradients."""

    def __init__(self, fastwam: nn.Module):
        super().__init__()
        self.fastwam = fastwam

    def forward(
        self,
        conditioning: ActionConditioning,
        step: StudentDenoisingStep,
    ) -> torch.Tensor:
        return FastWAMActionFlowScorer(self.fastwam).score(conditioning, step)
