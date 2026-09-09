"""Frozen nominal-outcome Teacher and representation-only reward."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from resilient.opsd.adapters import lora_disabled

from .types import ActionConditioning


@dataclass(frozen=True)
class OutcomeTarget:
    """Nominal hidden target and shared perturbation used to compare candidates."""

    hidden: torch.Tensor
    noise: torch.Tensor
    timestep: torch.Tensor
    tokens_per_frame: int


class FrozenNominalOutcomeTeacher:
    """Use LoRA-disabled Fast-WAM video features as a non-differentiable evaluator."""

    def __init__(self, model, *, layer_index: int, reward_timestep: float) -> None:
        self.model = model
        self.layer_index = int(layer_index)
        self.reward_timestep = float(reward_timestep)
        if not 0 <= self.layer_index < len(model.video_expert.blocks):
            raise ValueError("Outcome layer index is outside the Video DiT.")
        max_timestep = float(model.train_video_scheduler.num_train_timesteps)
        if not 0.0 < self.reward_timestep < max_timestep:
            raise ValueError("reward_timestep must be strictly inside the training interval.")

    def _context(
        self,
        conditioning: ActionConditioning,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.model
        context = conditioning.context.to(device=model.device, dtype=model.torch_dtype)
        context_mask = conditioning.context_mask.to(device=model.device, dtype=torch.bool)
        if conditioning.proprio is not None:
            context, context_mask = model._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=conditioning.proprio.to(device=model.device, dtype=model.torch_dtype),
            )
        return context, context_mask

    def _video_hidden(
        self,
        latent: torch.Tensor,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        expert = self.model.video_expert
        (
            tokens,
            _t,
            t_mod,
            embedded_context,
            embedded_context_mask,
            freqs,
            _f,
            _h,
            _w,
            tokens_per_frame,
        ) = expert.prepare(
            x=latent,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=bool(
                getattr(expert, "fuse_vae_embedding_in_latents", False)
            ),
        )
        attention_mask = expert.build_video_to_video_mask(
            video_seq_len=int(tokens.shape[1]),
            video_tokens_per_frame=int(tokens_per_frame),
            device=tokens.device,
        )
        for index, block in enumerate(expert.blocks):
            tokens = block(
                tokens,
                embedded_context,
                t_mod,
                freqs,
                context_mask=embedded_context_mask,
                self_attn_mask=attention_mask,
            )
            if index == self.layer_index:
                return tokens.detach().float(), int(tokens_per_frame)
        raise AssertionError("Validated outcome layer was not reached.")

    def _predict_nominal_latent(
        self,
        conditioning: ActionConditioning,
        *,
        num_video_frames: int,
        num_inference_steps: int,
        sigma_shift: float | None,
        seed: int,
        rand_device: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self.model
        image = conditioning.input_image.to(device=model.device, dtype=model.torch_dtype)
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Outcome Teacher expects one image shaped [1,C,H,W].")
        _, _, height, width = image.shape
        latent_t = (int(num_video_frames) - 1) // model.vae.temporal_downsample_factor + 1
        latent_h = height // model.vae.upsampling_factor
        latent_w = width // model.vae.upsampling_factor
        generator = torch.Generator(device=rand_device).manual_seed(int(seed))
        latent = torch.randn(
            (1, model.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=model.device, dtype=model.torch_dtype)
        first_frame = model._encode_input_image_latents_tensor(input_image=image, tiled=False)
        latent[:, :, :1] = first_frame
        context, context_mask = self._context(conditioning)
        timesteps, deltas = model.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=int(num_inference_steps),
            device=model.device,
            dtype=latent.dtype,
            shift_override=sigma_shift,
        )
        for step_timestep, step_delta in zip(timesteps, deltas, strict=True):
            flow = model.video_expert(
                x=latent,
                timestep=step_timestep.unsqueeze(0),
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=bool(
                    getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)
                ),
            )
            latent = model.infer_video_scheduler.step(flow, step_delta, latent)
            latent[:, :, :1] = first_frame
        return latent.detach(), context, context_mask

    def _perturbed_hidden(
        self,
        clean_latent: torch.Tensor,
        *,
        noise: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        noisy = self.model.train_video_scheduler.add_noise(clean_latent, noise, timestep)
        noisy[:, :, :1] = clean_latent[:, :, :1]
        return self._video_hidden(
            noisy,
            context=context,
            context_mask=context_mask,
            timestep=timestep,
        )

    @torch.no_grad()
    def build_target(
        self,
        conditioning: ActionConditioning,
        *,
        num_video_frames: int,
        num_inference_steps: int,
        sigma_shift: float | None,
        prediction_seed: int,
        reward_noise_seed: int,
        rand_device: str,
    ) -> OutcomeTarget:
        """Predict a nominal future and extract its fixed-layer future representation."""
        model = self.model
        with lora_disabled(model):
            model.eval()
            latent, context, context_mask = self._predict_nominal_latent(
                conditioning,
                num_video_frames=num_video_frames,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=prediction_seed,
                rand_device=rand_device,
            )
            generator = torch.Generator(device=rand_device).manual_seed(int(reward_noise_seed))
            noise = torch.randn(
                latent.shape,
                generator=generator,
                device=rand_device,
                dtype=torch.float32,
            ).to(device=model.device, dtype=model.torch_dtype)
            timestep = torch.full(
                (latent.shape[0],),
                self.reward_timestep,
                device=model.device,
                dtype=model.torch_dtype,
            )
            hidden, tokens_per_frame = self._perturbed_hidden(
                latent,
                noise=noise,
                timestep=timestep,
                context=context,
                context_mask=context_mask,
            )
        return OutcomeTarget(hidden, noise.detach(), timestep.detach(), tokens_per_frame)

    @torch.no_grad()
    def encode_realized(
        self,
        video: torch.Tensor,
        conditioning: ActionConditioning,
        target: OutcomeTarget,
    ) -> torch.Tensor:
        """Extract candidate features with the target's identical noise and timestep."""
        model = self.model
        with lora_disabled(model):
            model.eval()
            latent = model._encode_video_latents(
                video.to(device=model.device, dtype=model.torch_dtype),
                tiled=False,
            )
            if latent.shape != target.noise.shape:
                raise ValueError(
                    f"Realized latent shape {tuple(latent.shape)} does not match target "
                    f"shape {tuple(target.noise.shape)}."
                )
            context, context_mask = self._context(conditioning)
            hidden, tokens_per_frame = self._perturbed_hidden(
                latent,
                noise=target.noise,
                timestep=target.timestep,
                context=context,
                context_mask=context_mask,
            )
        if tokens_per_frame != target.tokens_per_frame:
            raise RuntimeError("Target and realized video token layouts differ.")
        return hidden


def temporal_delta_features(hidden: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
    """Remove current-frame content and return normalized future change tokens."""
    if hidden.ndim != 3 or tokens_per_frame <= 0:
        raise ValueError("Expected hidden [B,S,D] and positive tokens_per_frame.")
    if hidden.shape[1] % tokens_per_frame != 0 or hidden.shape[1] <= tokens_per_frame:
        raise ValueError("Hidden tokens do not contain aligned current and future frames.")
    normalized = F.layer_norm(hidden.float(), (hidden.shape[-1],))
    batch, _, width = normalized.shape
    frames = normalized.reshape(batch, -1, tokens_per_frame, width)
    return (frames[:, 1:] - frames[:, :1]).reshape(batch, -1, width)


def outcome_cosine_reward(
    realized_hidden: torch.Tensor,
    target: OutcomeTarget,
) -> torch.Tensor:
    """Return only the mean future-token cosine similarity requested by the method."""
    target_features = temporal_delta_features(target.hidden, target.tokens_per_frame)
    realized_features = temporal_delta_features(realized_hidden, target.tokens_per_frame)
    if target_features.shape != realized_features.shape:
        raise ValueError("Target and realized outcome representations differ in shape.")
    return F.cosine_similarity(target_features, realized_features, dim=-1).mean()
