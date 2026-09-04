"""Student-only Fast-WAM action generation for on-policy OPSD trajectories."""

from __future__ import annotations

from typing import Any

import torch

from .types import ActionConditioning, StudentTrajectory


def generate_student_trajectory(
    model,
    conditioning: ActionConditioning,
    *,
    action_horizon: int,
    num_inference_steps: int,
    sigma_shift: float | None,
    seed: int | None,
    rand_device: str,
    compile_action_infer: bool,
) -> StudentTrajectory:
    """Run the original Fast-WAM sampler once and preserve its exact latent path."""
    result = model.infer_action(
        prompt=conditioning.prompt,
        input_image=conditioning.input_image,
        action_horizon=int(action_horizon),
        proprio=conditioning.proprio,
        context=conditioning.context,
        context_mask=conditioning.context_mask,
        num_inference_steps=int(num_inference_steps),
        sigma_shift=sigma_shift,
        seed=seed,
        rand_device=rand_device,
        compile_action_infer=bool(compile_action_infer),
        return_denoising_trace=True,
    )
    return StudentTrajectory.from_inference_result(
        result,
        expected_steps=num_inference_steps,
        metadata={
            "action_horizon": int(action_horizon),
            "num_inference_steps": int(num_inference_steps),
            "sigma_shift": sigma_shift,
            "seed": seed,
            "rand_device": rand_device,
        },
    )


def validate_trace_against_scheduler(model, trajectory: StudentTrajectory) -> None:
    """Reject a trace that diverges from Fast-WAM's configured inference schedule."""
    expected_t, expected_delta = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=len(trajectory.steps),
        device=model.device,
        dtype=model.torch_dtype,
        shift_override=trajectory.metadata.get("sigma_shift"),
    )
    for index, step in enumerate(trajectory.steps):
        if not torch.allclose(
            step.timestep.flatten().to(expected_t), expected_t[index : index + 1]
        ):
            raise ValueError(f"Student trace timestep mismatch at index {index}.")
        if not torch.allclose(
            step.delta_sigma.flatten().to(expected_delta),
            expected_delta[index : index + 1],
        ):
            raise ValueError(f"Student trace delta mismatch at index {index}.")


def trajectory_to_checkpoint(trajectory: StudentTrajectory) -> dict[str, Any]:
    """Move a Student trajectory to CPU for ignored rollout/checkpoint storage."""
    return {
        "steps": [
            {
                "latent": step.latent.detach().cpu(),
                "timestep": step.timestep.detach().cpu(),
                "delta_sigma": step.delta_sigma.detach().cpu(),
            }
            for step in trajectory.steps
        ],
        "final_action": trajectory.final_action.detach().cpu(),
        "metadata": dict(trajectory.metadata),
    }
