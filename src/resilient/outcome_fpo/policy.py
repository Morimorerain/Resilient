"""Fast-WAM action sampling and conditional flow-matching scores for FPO."""

from __future__ import annotations

import torch
import torch.nn as nn

from resilient.opsd.model_adapter import FastWAMActionFlowScorer
from resilient.opsd.types import ActionConditioning as OPSDActionConditioning
from resilient.opsd.types import StudentDenoisingStep

from .types import ActionConditioning


def _expand_conditioning(
    conditioning: ActionConditioning,
    batch_size: int,
) -> OPSDActionConditioning:
    """Broadcast one state condition across FPO Monte Carlo samples."""

    def expand_first(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] != 1:
            raise ValueError("Stored FPO conditioning must have batch size one.")
        return tensor.expand(batch_size, *tensor.shape[1:])

    return OPSDActionConditioning(
        input_image=expand_first(conditioning.input_image),
        context=expand_first(conditioning.context),
        context_mask=expand_first(conditioning.context_mask),
        proprio=(
            None if conditioning.proprio is None else expand_first(conditioning.proprio)
        ),
    )


def sample_action_chunk(
    model,
    conditioning: ActionConditioning,
    *,
    action_horizon: int,
    num_inference_steps: int,
    sigma_shift: float | None,
    seed: int,
    rand_device: str,
    compile_action_infer: bool,
) -> torch.Tensor:
    """Sample one final normalized action chunk with Fast-WAM's native solver."""
    result = model.infer_action(
        prompt=None,
        input_image=conditioning.input_image,
        action_horizon=int(action_horizon),
        proprio=conditioning.proprio,
        context=conditioning.context,
        context_mask=conditioning.context_mask,
        num_inference_steps=int(num_inference_steps),
        sigma_shift=sigma_shift,
        seed=int(seed),
        rand_device=str(rand_device),
        compile_action_infer=bool(compile_action_infer),
        return_denoising_trace=False,
    )
    action = result["action"].detach().cpu().float()
    if action.shape != (int(action_horizon), model.action_expert.action_dim):
        raise RuntimeError(f"Unexpected Fast-WAM action shape: {tuple(action.shape)}")
    return action


def sample_cfm_pairs(
    action: torch.Tensor,
    *,
    num_samples: int,
    num_train_timesteps: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw reproducible continuous-time noise pairs independently of inference noise."""
    if action.ndim != 2:
        raise ValueError("action must have shape [T,D].")
    if num_samples <= 0 or num_train_timesteps <= 0:
        raise ValueError("FPO MC sample count and training timesteps must be positive.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(
        (int(num_samples), *action.shape),
        generator=generator,
        dtype=torch.float32,
    )
    # Fast-WAM's released action scheduler uses shift=1, so this exactly matches
    # sample_training_t while allowing a dedicated deterministic generator.
    timestep = torch.rand(
        (int(num_samples),),
        generator=generator,
        dtype=torch.float32,
    ) * float(num_train_timesteps)
    return noise, timestep


class FastWAMFPOScoringModule(nn.Module):
    """Evaluate per-MC epsilon-MSE through the native Fast-WAM action field."""

    def __init__(self, fastwam: nn.Module):
        super().__init__()
        self.fastwam = fastwam

    def forward(
        self,
        conditioning: ActionConditioning,
        action: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        model = self.fastwam
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.shape[0] != 1:
            raise ValueError("FPO scores one sampled action chunk at a time.")
        if noise.ndim != 3 or timestep.shape != (noise.shape[0],):
            raise ValueError("Expected noise [N,T,D] and timestep [N].")
        if tuple(noise.shape[1:]) != tuple(action.shape[1:]):
            raise ValueError("Action and MC noise shapes do not match.")

        device = model.device
        dtype = model.torch_dtype
        count = int(noise.shape[0])
        clean_action = action.to(device=device, dtype=dtype).expand(count, -1, -1)
        noise_device = noise.to(device=device, dtype=dtype)
        timestep_device = timestep.to(device=device, dtype=dtype)
        noisy_action = model.train_action_scheduler.add_noise(
            clean_action,
            noise_device,
            timestep_device,
        )
        score_conditioning = _expand_conditioning(
            conditioning.to(device=device, dtype=dtype),
            count,
        )
        step = StudentDenoisingStep(
            latent=noisy_action,
            timestep=timestep_device,
            delta_sigma=torch.zeros((), device=device, dtype=dtype),
        )
        velocity = FastWAMActionFlowScorer(model).score(score_conditioning, step)
        train_steps = float(model.train_action_scheduler.num_train_timesteps)
        sigma = (timestep_device.float() / train_steps).view(count, 1, 1)
        epsilon_prediction = noisy_action.float() + (1.0 - sigma) * velocity.float()
        return (epsilon_prediction - noise_device.float()).square().mean(dim=(1, 2))
