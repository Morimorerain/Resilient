"""Reproducible multi-GPU LIBERO runtime for outcome-guided FPO."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from libero.libero import benchmark, get_libero_path
from omegaconf import DictConfig, OmegaConf

from experiments.libero.eval_libero_single import (
    _denormalize_action,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _obs_to_model_input,
)
from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    invert_gripper_action,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from resilient.faults import build_fault_pipeline
from resilient.opsd.adapters import (
    FastWAMLoraConfig,
    inject_fastwam_aligned_lora,
    load_adapter_state_dict,
)
from resilient.opsd.runtime import RolloutDescriptor, _derive_seed, build_rollout_schedule
from resilient.state_banks import (
    assert_disjoint_manifests,
    load_manifest,
    state_fingerprint,
)

from .collector import (
    collect_action_chunk_future,
    restore_shadow_state,
    stack_observation_video,
)
from .outcome import FrozenNominalOutcomeTeacher, outcome_cosine_reward
from .policy import FastWAMFPOScoringModule, sample_action_chunk, sample_cfm_pairs
from .trainer import OutcomeFPOTrainer
from .types import ActionConditioning, CandidateSample, OutcomeGroup


def resolved_config_hash(cfg: DictConfig) -> str:
    """Hash all algorithmic settings while excluding runtime locations."""
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved outcome-FPO configuration must be a mapping.")
    payload["output_dir"] = "<runtime-output-dir>"
    payload["resume"] = None
    if isinstance(payload.get("outcome_fpo"), dict):
        payload["outcome_fpo"]["validate_only"] = False
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve_project_path(value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _resolve_resume(output_dir: Path, resume: str | None) -> Path | None:
    if resume in {None, "", "null"}:
        return None
    if resume != "auto":
        return _resolve_project_path(str(resume))
    checkpoint_root = output_dir / "checkpoints"
    candidates = [
        *(checkpoint_root / "state").glob("step_*"),
        *(checkpoint_root / "epochs").glob("epoch_*_step_*"),
    ]
    completed: list[tuple[tuple[int, int, int], Path]] = []
    for candidate in candidates:
        state_path = candidate / "trainer_state.json"
        if not state_path.is_file():
            continue
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        completed.append(
            (
                (
                    int(payload.get("group_index", -1)),
                    int(payload.get("epoch", -1)),
                    int(payload.get("global_step", -1)),
                ),
                candidate,
            )
        )
    return max(completed, key=lambda item: item[0])[1] if completed else None


def validate_outcome_fpo_config(
    cfg: DictConfig,
    *,
    world_size: int | None = None,
) -> dict[str, Any]:
    """Validate FPO, Fast-WAM, Fault, and train/validation split invariants."""
    section = cfg.outcome_fpo
    requested_gpus = int(section.distributed.num_processes)
    if requested_gpus not in {4, 8}:
        raise ValueError("Outcome-FPO supports exactly 4 or 8 GPUs.")
    if world_size is not None and world_size != requested_gpus:
        raise ValueError("Accelerate world size does not match outcome_fpo.distributed.")
    if int(section.num_epochs) <= 0:
        raise ValueError("outcome_fpo.num_epochs must be positive.")
    if int(section.rollout.rollouts_per_task) != 50:
        raise ValueError("The initial embodied-Fault study fixes 50 states per task.")
    if int(section.rollout.action_horizon) != int(cfg.data.train.num_frames) - 1:
        raise ValueError("Action horizon must equal data.train.num_frames - 1.")
    expected_video_frames = (
        int(section.rollout.action_horizon) // int(cfg.data.train.action_video_freq_ratio) + 1
    )
    if int(section.outcome.num_video_frames) != expected_video_frames:
        raise ValueError("Outcome video length does not match the action/video frequency ratio.")
    frame_steps = [int(value) for value in section.rollout.future_frame_steps]
    action_video_ratio = int(cfg.data.train.action_video_freq_ratio)
    if frame_steps != list(
        range(0, int(section.rollout.action_horizon) + 1, action_video_ratio)
    ):
        raise ValueError("Outcome frame steps must follow Fast-WAM's action/video alignment.")
    if int(section.rollout.num_inference_steps) != int(cfg.eval_num_inference_steps):
        raise ValueError("Student action sampling must keep Fast-WAM's configured step count.")
    if int(section.outcome.num_inference_steps) != int(cfg.eval_num_inference_steps):
        raise ValueError("Teacher video prediction must keep Fast-WAM's configured step count.")
    if section.rollout.sigma_shift is not None:
        raise ValueError("Released Fast-WAM action inference must use its native shift=1 setting.")
    if not bool(cfg.model.skip_dit_load_from_pretrain):
        raise ValueError(
            "Training must load both experts only from the pinned Fast-WAM checkpoint."
        )
    if cfg.model.action_dit_pretrained_path is not None:
        raise ValueError("Redundant ActionDiT pretraining must remain disabled.")
    if not bool(cfg.model.load_text_encoder):
        raise ValueError("LIBERO prompts require the Fast-WAM text encoder.")
    group_size = int(section.fpo.group_size)
    if group_size < 2:
        raise ValueError("FPO same-state group_size must be at least two.")
    if int(section.fpo.num_mc_samples) <= 0:
        raise ValueError("FPO num_mc_samples must be positive.")
    if str(section.fpo.loss_type) != "epsilon_mse":
        raise ValueError("The first implementation supports the official epsilon_mse mode.")
    if not bool(section.fpo.average_losses_before_exp):
        raise ValueError("The stable FPO path averages MC losses before exponentiation.")
    if not 0.0 < float(section.fpo.clipping_epsilon) < 1.0:
        raise ValueError("FPO clipping_epsilon must be in (0,1).")
    global_groups = int(section.fpo.groups_per_global_update)
    if global_groups <= 0 or global_groups % requested_gpus != 0:
        raise ValueError("groups_per_global_update must divide evenly across all ranks.")
    total_groups = len(section.rollout.suites) * 10 * int(section.rollout.rollouts_per_task)
    if total_groups % global_groups != 0:
        raise ValueError("The global rollout schedule must divide into complete FPO updates.")

    fault_cfg = OmegaConf.to_container(cfg.fault, resolve=True)
    fault_metadata = build_fault_pipeline(fault_cfg).metadata()
    faults = fault_metadata["faults"]
    if len(faults) != 1 or faults[0]["family"] != "structure.joint_motion":
        raise ValueError("The first outcome-FPO configuration requires one joint-motion Fault.")
    if faults[0]["targets"] != ["robot0_joint1"]:
        raise ValueError("The first outcome-FPO configuration targets robot0_joint1.")
    severity = faults[0]["severity"]
    if severity["name"] != "motion_retention" or float(severity["value"]) != 0.5:
        raise ValueError("The first outcome-FPO configuration requires retention=0.5.")

    training_path = _resolve_project_path(str(section.split.training_reference_manifest))
    validation_path = _resolve_project_path(str(section.split.validation_manifest))
    if not training_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(
            "Training/validation state manifests are required; generate the documented state bank."
        )
    training_manifest = load_manifest(training_path)
    validation_manifest = load_manifest(validation_path)
    assert_disjoint_manifests(validation_manifest, training_manifest)
    adapter = FastWAMLoraConfig.from_config(OmegaConf.to_container(cfg.adapter, resolve=True))
    return {
        "config_sha256": resolved_config_hash(cfg),
        "method": "outcome_guided_flow_policy_optimization",
        "fpo_upstream": "akanazawa/fpo@418c2554f7cd22d52e14c07d951280929d73bf2f",
        "fault": fault_metadata,
        "adapter": {
            "rank": adapter.rank,
            "alpha": adapter.alpha,
            "dropout": adapter.dropout,
        },
        "num_processes": requested_gpus,
        "groups_per_rank_per_update": global_groups // requested_gpus,
        "groups_per_epoch": total_groups,
        "candidates_per_epoch": total_groups * group_size,
        "train_state_count": len(training_manifest.get("states", [])),
        "validation_state_count": len(validation_manifest.get("states", [])),
        "split_overlap_checked": True,
    }


def _training_state_records(manifest_path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    records = {
        (str(item["suite"]), int(item["task_id"]), int(item["state_index"])): item
        for item in manifest.get("states", [])
    }
    if len(records) != len(manifest.get("states", [])):
        raise ValueError("Training reference manifest contains duplicate state identities.")
    return records


def _execute_warmup(env: Any, steps: int) -> Any:
    obs = None
    for _ in range(int(steps)):
        obs, _, done, _ = env.step(get_libero_dummy_action())
        if done:
            raise RuntimeError("LIBERO task terminated during warm-up.")
    return obs


def _denormalize_candidate(
    normalized_action: torch.Tensor,
    processor: FastWAMProcessor,
    *,
    binarize_gripper: bool,
) -> np.ndarray:
    action = _denormalize_action(normalized_action, processor)[0]
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if binarize_gripper:
        action[..., -1] = np.sign(action[..., -1])
    return action


def _make_conditioning(
    obs: Any,
    *,
    task_description: str,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    model,
    video_height: int,
    video_width: int,
) -> ActionConditioning:
    image, proprio, _ = _obs_to_model_input(
        obs=obs,
        cfg=cfg,
        processor=processor,
        width=video_width,
        height=video_height,
        device=str(model.device),
        dtype=model.torch_dtype,
    )
    prompt = DEFAULT_PROMPT.format(task=task_description)
    with torch.no_grad():
        context, context_mask = model.encode_prompt(prompt)
    return ActionConditioning(image, context, context_mask, proprio)


def _collect_group(
    *,
    cfg: DictConfig,
    descriptor: RolloutDescriptor,
    schedule_epoch: int,
    global_groups_per_epoch: int,
    processor: FastWAMProcessor,
    model,
    video_height: int,
    video_width: int,
    training_records: dict[tuple[str, int, int], dict[str, Any]],
) -> OutcomeGroup:
    section = cfg.outcome_fpo
    suite = benchmark.get_benchmark_dict()[descriptor.suite]()
    task = suite.get_task(descriptor.task_id)
    states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    initial_states = torch.load(states_path, map_location="cpu", weights_only=False)
    state = initial_states[descriptor.initial_state_index]
    record_key = (descriptor.suite, descriptor.task_id, descriptor.initial_state_index)
    expected_record = training_records.get(record_key)
    if expected_record is None or state_fingerprint(state) != expected_record["state_sha256"]:
        raise ValueError(f"Official training state does not match its manifest: {record_key}")

    episode_index = schedule_epoch * global_groups_per_epoch + descriptor.sample_id
    fault_cfg = OmegaConf.to_container(cfg.fault, resolve=True)
    main_env, task_description = get_libero_env(
        task,
        LIBERO_ENV_RESOLUTION,
        descriptor.environment_seed,
        fault_pipeline=build_fault_pipeline(fault_cfg),
        fault_episode_index=episode_index,
    )
    shadow_env, _ = get_libero_env(
        task,
        LIBERO_ENV_RESOLUTION,
        descriptor.environment_seed,
        fault_pipeline=build_fault_pipeline(fault_cfg),
        fault_episode_index=episode_index,
    )
    try:
        main_env.reset()
        obs = main_env.set_init_state(state)
        warmed_obs = _execute_warmup(main_env, int(section.rollout.num_steps_wait))
        if warmed_obs is not None:
            obs = warmed_obs
        conditioning = _make_conditioning(
            obs,
            task_description=task_description,
            cfg=cfg,
            processor=processor,
            model=model,
            video_height=video_height,
            video_width=video_width,
        )
        simulator_state = main_env.get_sim_state()
        fault_runtime_state = main_env.fault_runtime_state_dict()
        teacher = FrozenNominalOutcomeTeacher(
            model,
            layer_index=int(section.outcome.layer_index),
            reward_timestep=float(section.outcome.reward_timestep),
        )
        target = teacher.build_target(
            conditioning,
            num_video_frames=int(section.outcome.num_video_frames),
            num_inference_steps=int(section.outcome.num_inference_steps),
            sigma_shift=(
                None
                if section.outcome.sigma_shift is None
                else float(section.outcome.sigma_shift)
            ),
            prediction_seed=_derive_seed(
                int(section.rollout.inference_seed), descriptor.sample_id, "teacher-video"
            ),
            reward_noise_seed=_derive_seed(
                int(section.rollout.inference_seed), descriptor.sample_id, "reward-noise"
            ),
            rand_device=str(section.rollout.rand_device),
        )

        def encode_observation(item: Any) -> torch.Tensor:
            image, _, _ = _obs_to_model_input(
                obs=item,
                cfg=cfg,
                processor=processor,
                width=video_width,
                height=video_height,
                device="cpu",
                dtype=torch.float32,
            )
            return image

        candidates: list[CandidateSample] = []
        old_scorer = FastWAMFPOScoringModule(model)
        for candidate_index in range(int(section.fpo.group_size)):
            inference_seed = _derive_seed(
                descriptor.inference_seed, descriptor.sample_id, candidate_index, "candidate"
            )
            with torch.no_grad():
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
            shadow_obs = restore_shadow_state(
                shadow_env,
                simulator_state=simulator_state,
                fault_runtime_state=fault_runtime_state,
            )
            future = collect_action_chunk_future(
                shadow_env,
                shadow_obs,
                action,
                frame_steps=section.rollout.future_frame_steps,
            )
            video = stack_observation_video(future.observations, encode_observation)
            realized_hidden = teacher.encode_realized(video, conditioning, target)
            reward = float(outcome_cosine_reward(realized_hidden, target))
            cfm_seed = _derive_seed(
                int(section.fpo.mc_seed), descriptor.sample_id, candidate_index, "cfm"
            )
            mc_noise, mc_timestep = sample_cfm_pairs(
                normalized_action,
                num_samples=int(section.fpo.num_mc_samples),
                num_train_timesteps=model.train_action_scheduler.num_train_timesteps,
                seed=cfm_seed,
            )
            with torch.no_grad():
                old_loss = old_scorer(
                    conditioning,
                    normalized_action,
                    mc_noise,
                    mc_timestep,
                ).detach().cpu()
            candidates.append(
                CandidateSample(
                    normalized_action=normalized_action,
                    reward=reward,
                    mc_noise=mc_noise,
                    mc_timestep=mc_timestep,
                    old_cfm_loss=old_loss,
                    inference_seed=inference_seed,
                    metadata={
                        "candidate_index": candidate_index,
                        "cfm_seed": cfm_seed,
                        "counterfactual_success": future.success,
                    },
                )
            )
        return OutcomeGroup(
            conditioning=conditioning.cpu(),
            candidates=tuple(candidates),
            metadata={
                "suite": descriptor.suite,
                "task_id": descriptor.task_id,
                "rollout_id": descriptor.rollout_id,
                "initial_state_index": descriptor.initial_state_index,
                "sample_id": descriptor.sample_id,
                "environment_seed": descriptor.environment_seed,
                "episode_index": episode_index,
                "faults": main_env.fault_metadata["faults"],
                "future_source": "same_state_fault_counterfactual_action_chunk",
            },
        )
    finally:
        main_env.close()
        shadow_env.close()


def run_outcome_fpo_training(cfg: DictConfig) -> None:
    """Train a Fault-specialized Recovery LoRA with exactly 4 or 8 ranks."""
    accelerator = Accelerator(mixed_precision=str(cfg.mixed_precision))
    provenance = validate_outcome_fpo_config(cfg, world_size=accelerator.num_processes)
    output_dir = _resolve_project_path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
        (output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )

    set_global_seed(int(cfg.seed) + accelerator.process_index, get_worker_init_fn=False)
    model_dtype = _mixed_precision_to_model_dtype(str(cfg.mixed_precision))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(accelerator.device))
    _load_model_checkpoint(model, str(_resolve_project_path(str(cfg.ckpt))))
    adapter_cfg = FastWAMLoraConfig.from_config(OmegaConf.to_container(cfg.adapter, resolve=True))
    adapter_audit = inject_fastwam_aligned_lora(model, adapter_cfg)
    initial_adapter = cfg.adapter.get("initial_checkpoint")
    if initial_adapter not in {None, "", "null"}:
        initial_state = torch.load(
            _resolve_project_path(str(initial_adapter)), map_location="cpu", weights_only=True
        )
        load_adapter_state_dict(model, initial_state)
    if accelerator.is_main_process:
        (output_dir / "adapter_audit.json").write_text(
            json.dumps(adapter_audit, indent=2) + "\n", encoding="utf-8"
        )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.learning_rate),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=float(cfg.weight_decay),
    )
    scoring_model = FastWAMFPOScoringModule(model)
    scoring_model, optimizer = accelerator.prepare(scoring_model, optimizer)
    section = cfg.outcome_fpo
    trainer = OutcomeFPOTrainer(
        scoring_model=scoring_model,
        optimizer=optimizer,
        accelerator=accelerator,
        update_epochs=int(section.fpo.update_epochs),
        clipping_epsilon=float(section.fpo.clipping_epsilon),
        log_ratio_clip=float(section.fpo.log_ratio_clip),
        advantage_clip=float(section.fpo.advantage_clip),
        advantage_eps=float(section.fpo.advantage_eps),
        minimum_reward_std=float(section.fpo.minimum_reward_std),
        max_grad_norm=float(cfg.max_grad_norm),
    )
    resume_path = _resolve_resume(output_dir, None if cfg.resume is None else str(cfg.resume))
    if resume_path is not None:
        trainer.load_checkpoint(resume_path, config_hash=provenance["config_sha256"])

    stats = load_dataset_stats_from_json(
        str(_resolve_project_path(str(section.dataset_stats_path)))
    )
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    video_height, video_width = (int(value) for value in cfg.data.train.video_size)
    suites = [str(value) for value in section.rollout.suites]
    rollouts_per_task = int(section.rollout.rollouts_per_task)
    global_groups_per_epoch = len(suites) * 10 * rollouts_per_task
    local_groups_per_epoch = global_groups_per_epoch // accelerator.num_processes
    local_collection_size = int(section.fpo.groups_per_global_update) // accelerator.num_processes
    completed_before_epoch = trainer.epoch * local_groups_per_epoch
    completed_in_epoch = trainer.group_index - completed_before_epoch
    if not 0 <= completed_in_epoch <= local_groups_per_epoch:
        raise ValueError("Checkpoint group index is inconsistent with the configured world size.")
    training_manifest_path = _resolve_project_path(
        str(section.split.training_reference_manifest)
    )
    training_records = _training_state_records(training_manifest_path)
    rollout_log = output_dir / f"rollouts_rank_{accelerator.process_index:02d}.jsonl"
    update_log = output_dir / f"updates_rank_{accelerator.process_index:02d}.jsonl"

    for epoch_index in range(trainer.epoch, int(section.num_epochs)):
        schedule_epoch = epoch_index + int(section.epoch_offset)
        schedule = build_rollout_schedule(
            suites=suites,
            tasks_per_suite=10,
            rollouts_per_task=rollouts_per_task,
            epoch=schedule_epoch,
            schedule_seed=int(section.rollout.schedule_seed),
            environment_seed=int(section.rollout.environment_seed),
            inference_seed=int(section.rollout.inference_seed),
        )[accelerator.process_index :: accelerator.num_processes]
        offset = completed_in_epoch if epoch_index == trainer.epoch else 0
        if offset % local_collection_size != 0:
            raise ValueError("Resume checkpoint is not at an on-policy collection boundary.")
        for start in range(offset, len(schedule), local_collection_size):
            descriptors = schedule[start : start + local_collection_size]
            groups = [
                _collect_group(
                    cfg=cfg,
                    descriptor=descriptor,
                    schedule_epoch=schedule_epoch,
                    global_groups_per_epoch=global_groups_per_epoch,
                    processor=processor,
                    model=trainer.fastwam,
                    video_height=video_height,
                    video_width=video_width,
                    training_records=training_records,
                )
                for descriptor in descriptors
            ]
            with rollout_log.open("a", encoding="utf-8") as stream:
                for group in groups:
                    stream.write(
                        json.dumps(
                            {
                                **group.metadata,
                                "rewards": [item.reward for item in group.candidates],
                                "candidate_seeds": [
                                    item.inference_seed for item in group.candidates
                                ],
                                "counterfactual_success": [
                                    item.metadata["counterfactual_success"]
                                    for item in group.candidates
                                ],
                            }
                        )
                        + "\n"
                    )
            metrics = trainer.train_groups(groups)
            with update_log.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "epoch": schedule_epoch + 1,
                            "global_step": trainer.global_step,
                            "local_group_index": trainer.group_index,
                            "sample_ids": [item.sample_id for item in descriptors],
                            **metrics,
                        }
                    )
                    + "\n"
                )
            del groups
            save_every = int(cfg.save_every)
            before_final_collection = start + local_collection_size < len(schedule)
            if (
                save_every > 0
                and trainer.global_step % save_every == 0
                and before_final_collection
            ):
                trainer.save_checkpoint(
                    output_dir,
                    config_hash=provenance["config_sha256"],
                    fault_state={},
                )
                trainer.prune_checkpoints(output_dir, keep=int(cfg.max_checkpoints))

        trainer.epoch = epoch_index + 1
        trainer.save_checkpoint(
            output_dir,
            config_hash=provenance["config_sha256"],
            fault_state={},
            epoch_number=schedule_epoch + 1,
        )
        trainer.prune_checkpoints(output_dir, keep=int(cfg.max_checkpoints))
        logging.info(
            "Completed outcome-FPO epoch %d/%d at optimizer step %d.",
            trainer.epoch,
            int(section.num_epochs),
            trainer.global_step,
        )
        completed_in_epoch = 0
