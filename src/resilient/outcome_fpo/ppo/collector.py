"""Sequential on-policy E5 episodes with full 32-action, nine-frame outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from libero.libero import benchmark, get_libero_path
from omegaconf import DictConfig, OmegaConf

from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env
from resilient.faults import build_fault_pipeline
from resilient.opsd.runtime import RolloutDescriptor, _derive_seed
from resilient.state_banks import state_fingerprint
from resilient.task_decoupling.libero_oracle import (
    install_robosuite_numpy2_segmentation_compatibility,
)
from resilient.task_decoupling.reward import task_outcome_rewards

from ..collector import collect_action_chunk_future
from ..outcome import FrozenNominalOutcomeTeacher
from ..policy import FastWAMFPOScoringModule, sample_action_chunk, sample_cfm_pairs
from ..runtime import _denormalize_candidate, _execute_warmup, _make_conditioning
from ..task_features import build_task_disentangler, extract_task_features
from ..types import ActionConditioning, CandidateSample


@dataclass(frozen=True)
class PPOTransition:
    """One current-state decision followed by an actual Fault-environment chunk."""

    conditioning: ActionConditioning
    candidate: CandidateSample
    old_value: float
    reward: float
    done: bool


@dataclass(frozen=True)
class PPOEpisode:
    """One unbranched Student trajectory and its bootstrap value."""

    transitions: tuple[PPOTransition, ...]
    bootstrap_value: float
    metadata: dict[str, Any]


def collect_ppo_episode(
    *,
    cfg: DictConfig,
    descriptor: RolloutDescriptor,
    schedule_epoch: int,
    processor,
    model,
    value_model,
    video_height: int,
    video_width: int,
    training_records: dict[tuple[str, int, int], dict[str, Any]],
) -> PPOEpisode:
    """Run the live Student for up to 16 chunks, without a shadow simulator."""
    model.eval()
    section = cfg.outcome_fpo
    ppo = section.ppo
    suite = benchmark.get_benchmark_dict()[descriptor.suite]()
    task = suite.get_task(descriptor.task_id)
    states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    state = torch.load(states_path, map_location="cpu", weights_only=False)[
        descriptor.initial_state_index
    ]
    key = (descriptor.suite, descriptor.task_id, descriptor.initial_state_index)
    expected = training_records.get(key)
    if expected is None or state_fingerprint(state) != expected["state_sha256"]:
        raise ValueError(f"Official training state does not match its manifest: {key}")
    fault_cfg = OmegaConf.to_container(cfg.fault, resolve=True)
    install_robosuite_numpy2_segmentation_compatibility()
    episode_index = schedule_epoch * 50 + descriptor.sample_id
    env, task_description = get_libero_env(
        task,
        LIBERO_ENV_RESOLUTION,
        descriptor.environment_seed,
        fault_pipeline=build_fault_pipeline(fault_cfg),
        fault_episode_index=episode_index,
        segmentation=True,
    )
    try:
        env.reset()
        observation = env.set_init_state(state)
        warmed = _execute_warmup(env, int(section.rollout.num_steps_wait))
        if warmed is not None:
            observation = warmed
        task_disentangler = build_task_disentangler(cfg, project_root=Path.cwd())
        if task_disentangler is None:
            raise ValueError("Actor-critic E5 study requires the task-direction reward extractor.")
        teacher = FrozenNominalOutcomeTeacher(
            model,
            layer_index=int(section.outcome.layer_index),
            reward_timestep=float(section.outcome.reward_timestep),
        )
        old_scorer = FastWAMFPOScoringModule(model)
        transitions: list[PPOTransition] = []
        for chunk_index in range(int(ppo.max_chunks_per_episode)):
            conditioning = _make_conditioning(
                observation,
                task_description=task_description,
                cfg=cfg,
                processor=processor,
                model=model,
                video_height=video_height,
                video_width=video_width,
            )
            with torch.no_grad():
                old_value = value_model.rollout_value(conditioning)
                target = teacher.build_target(
                    conditioning,
                    num_video_frames=int(section.outcome.num_video_frames),
                    num_inference_steps=int(section.outcome.num_inference_steps),
                    sigma_shift=None,
                    prediction_seed=_derive_seed(
                        descriptor.inference_seed, descriptor.sample_id, chunk_index, "teacher"
                    ),
                    reward_noise_seed=_derive_seed(
                        descriptor.inference_seed, descriptor.sample_id, chunk_index, "noise"
                    ),
                    rand_device=str(section.rollout.rand_device),
                )
                inference_seed = _derive_seed(
                    descriptor.inference_seed, descriptor.sample_id, chunk_index, "action"
                )
                normalized_action = sample_action_chunk(
                    model,
                    conditioning,
                    action_horizon=int(section.rollout.action_horizon),
                    num_inference_steps=int(section.rollout.num_inference_steps),
                    sigma_shift=None,
                    seed=inference_seed,
                    rand_device=str(section.rollout.rand_device),
                    compile_action_infer=bool(section.rollout.compile_action_infer),
                )
            action = _denormalize_candidate(
                normalized_action,
                processor,
                binarize_gripper=bool(section.rollout.binarize_gripper),
            )
            future = collect_action_chunk_future(
                env,
                observation,
                action,
                frame_steps=section.rollout.future_frame_steps,
            )
            residual, _ = extract_task_features(
                task_disentangler,
                observations=future.observations,
                processor=processor,
                model=model,
                env=env,
            )
            if not residual.valid:
                raise RuntimeError("Task-direction reward received an invalid task mask.")
            full_hidden = teacher.encode_realized_latent(residual.full_latent, conditioning, target)
            counterfactual_hidden = teacher.encode_realized_latent(
                residual.counterfactual_latent, conditioning, target
            )
            reward = float(
                task_outcome_rewards(full_hidden, counterfactual_hidden, target).task_direction
            )
            cfm_seed = _derive_seed(
                int(section.fpo.mc_seed), descriptor.sample_id, chunk_index, "cfm"
            )
            mc_noise, mc_timestep = sample_cfm_pairs(
                normalized_action,
                num_samples=int(section.fpo.num_mc_samples),
                num_train_timesteps=model.train_action_scheduler.num_train_timesteps,
                seed=cfm_seed,
            )
            with torch.no_grad():
                old_cfm_loss = (
                    old_scorer(conditioning, normalized_action, mc_noise, mc_timestep)
                    .detach()
                    .cpu()
                )
            transitions.append(
                PPOTransition(
                    conditioning=conditioning.cpu(),
                    candidate=CandidateSample(
                        normalized_action=normalized_action,
                        reward=reward,
                        mc_noise=mc_noise,
                        mc_timestep=mc_timestep,
                        old_cfm_loss=old_cfm_loss,
                        inference_seed=inference_seed,
                        metadata={"chunk_index": chunk_index, "cfm_seed": cfm_seed},
                    ),
                    old_value=old_value,
                    reward=reward,
                    done=bool(future.success),
                )
            )
            observation = future.observations[-1]
            if future.success:
                break
        bootstrap = 0.0
        if transitions and not transitions[-1].done:
            final_conditioning = _make_conditioning(
                observation,
                task_description=task_description,
                cfg=cfg,
                processor=processor,
                model=model,
                video_height=video_height,
                video_width=video_width,
            )
            with torch.no_grad():
                bootstrap = value_model.rollout_value(final_conditioning)
        return PPOEpisode(
            transitions=tuple(transitions),
            bootstrap_value=bootstrap,
            metadata={
                "suite": descriptor.suite,
                "task_id": descriptor.task_id,
                "state_index": descriptor.initial_state_index,
                "sample_id": descriptor.sample_id,
                "environment_seed": descriptor.environment_seed,
                "inference_seed": descriptor.inference_seed,
                "faults": env.fault_metadata["faults"],
                "success": bool(transitions and transitions[-1].done),
                "control_steps": 32 * len(transitions),
                "future_source": "main_on_policy_full_action_chunk",
            },
        )
    finally:
        env.close()
