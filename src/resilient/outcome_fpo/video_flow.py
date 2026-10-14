"""Native Fast-WAM video-flow supervision for realized Fault rollouts."""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from .types import ActionConditioning


def realized_video_flow_loss(
    model,
    conditioning: ActionConditioning,
    real_latent: torch.Tensor,
    *,
    seed: int,
    checkpoint_blocks: bool,
) -> torch.Tensor:
    """Score the real nine-frame video without adding an action imitation loss."""
    if real_latent.ndim != 5 or real_latent.shape[0] != 1 or real_latent.shape[2] != 3:
        raise ValueError("Realized video latent must have shape [1,C,3,H,W].")
    if real_latent.requires_grad:
        raise ValueError("Frozen-VAE video targets must not require gradients.")
    if not bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)):
        raise ValueError("This auxiliary loss requires Fast-WAM's fused first-frame latent.")
    if bool(getattr(model.video_expert, "action_conditioned", False)):
        raise ValueError("This scorer is for the released action-unconditioned Video DiT.")

    latent = real_latent.to(device=model.device, dtype=model.torch_dtype).detach()
    context = conditioning.context.to(device=model.device, dtype=model.torch_dtype)
    context_mask = conditioning.context_mask.to(device=model.device, dtype=torch.bool)
    if conditioning.proprio is not None:
        context, context_mask = model._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=conditioning.proprio.to(device=model.device, dtype=model.torch_dtype),
        )

    # A CPU generator makes each transition's perturbation reproducible without
    # changing the policy or simulator RNG streams.
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32).to(
        device=model.device, dtype=model.torch_dtype
    )
    uniform = torch.rand((1,), generator=generator, dtype=torch.float32).to(device=model.device)
    scheduler = model.train_video_scheduler
    timestep = (scheduler._phi(uniform, scheduler.shift) * float(scheduler.num_train_timesteps)).to(
        dtype=latent.dtype
    )
    noisy = scheduler.add_noise(latent, noise, timestep)
    target = scheduler.training_target(latent, noise, timestep)
    noisy[:, :, :1] = latent[:, :, :1]

    expert = model.video_expert
    if checkpoint_blocks:
        (
            tokens,
            t,
            t_mod,
            embedded_context,
            embedded_mask,
            freqs,
            frames,
            height,
            width,
            tokens_per_frame,
        ) = expert.prepare(
            x=noisy,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )
        attention_mask = (
            None
            if expert.video_attention_mask_mode == "bidirectional"
            else expert.build_video_to_video_mask(
                video_seq_len=int(tokens.shape[1]),
                video_tokens_per_frame=int(tokens_per_frame),
                device=tokens.device,
            )
        )
        for block in expert.blocks:

            def run_block(value: torch.Tensor, current_block=block) -> torch.Tensor:
                return current_block(
                    value,
                    embedded_context,
                    t_mod,
                    freqs,
                    context_mask=embedded_mask,
                    self_attn_mask=attention_mask,
                )

            tokens = checkpoint(run_block, tokens, use_reentrant=False)
        predicted = expert.post(tokens, t, frames, height, width)
    else:
        predicted = expert(
            x=noisy,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=True,
        )

    per_sample = model._compute_video_loss_per_sample(
        pred_video=predicted[:, :, 1:],
        target_video=target[:, :, 1:],
        image_is_pad=None,
        include_initial_video_step=False,
    )
    weight = scheduler.training_weight(timestep).to(
        device=per_sample.device, dtype=per_sample.dtype
    )
    return (per_sample * weight).mean()
