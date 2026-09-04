"""Hydra runtime for reproducible multi-GPU LIBERO OPSD-Flow training."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
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
from resilient.faults import FaultTransition, build_fault_pipeline
from resilient.faults.paired_observation import capture_paired_observation

from .adapters import FastWAMLoraConfig, inject_fastwam_aligned_lora, load_adapter_state_dict
from .model_adapter import OPSDScoringModule
from .rollout import generate_student_trajectory, validate_trace_against_scheduler
from .teacher_inputs import TeacherInputContext, build_teacher_input_provider
from .trainer import OPSDFlowTrainer
from .types import ActionConditioning


@dataclass(frozen=True)
class RolloutDescriptor:
    """Identity of one reproducible, independently reset OPSD training sample."""

    suite: str
    task_id: int
    rollout_id: int
    initial_state_index: int
    environment_seed: int
    inference_seed: int
    sample_id: int


def _derive_seed(base_seed: int, *identity: object) -> int:
    """Derive a stable non-negative seed without depending on Python's hash salt."""
    encoded = ":".join([str(base_seed), *(str(value) for value in identity)]).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**31)


def build_rollout_schedule(
    *,
    suites: list[str],
    tasks_per_suite: int,
    rollouts_per_task: int,
    epoch: int,
    schedule_seed: int,
    environment_seed: int,
    inference_seed: int,
) -> list[RolloutDescriptor]:
    """Build and deterministically shuffle the global schedule for one epoch."""
    descriptors: list[RolloutDescriptor] = []
    for suite_index, suite_name in enumerate(suites):
        for task_id in range(tasks_per_suite):
            for rollout_id in range(rollouts_per_task):
                sample_id = (
                    (suite_index * tasks_per_suite + task_id) * rollouts_per_task + rollout_id
                )
                identity = (epoch, suite_name, task_id, rollout_id)
                descriptors.append(
                    RolloutDescriptor(
                        suite=suite_name,
                        task_id=task_id,
                        rollout_id=rollout_id,
                        initial_state_index=rollout_id,
                        environment_seed=_derive_seed(environment_seed, *identity, "environment"),
                        inference_seed=_derive_seed(inference_seed, *identity, "inference"),
                        sample_id=sample_id,
                    )
                )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(schedule_seed) + int(epoch))
    permutation = torch.randperm(len(descriptors), generator=generator).tolist()
    return [descriptors[index] for index in permutation]


def resolved_config_hash(cfg: DictConfig) -> str:
    """Hash the fully resolved configuration for strict checkpoint matching."""
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved OPSD configuration must be a mapping.")
    # Runtime location/control values may change during resume without changing the run.
    payload["output_dir"] = "<runtime-output-dir>"
    payload["resume"] = None
    if isinstance(payload.get("opsd"), dict):
        payload["opsd"]["validate_only"] = False
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_opsd_config(cfg: DictConfig, *, world_size: int | None = None) -> dict[str, Any]:
    """Validate algorithm invariants without loading the model or using a GPU."""
    opsd = cfg.opsd
    if int(opsd.num_epochs) <= 0:
        raise ValueError("opsd.num_epochs must be positive.")
    if str(cfg.lr_scheduler_type) != "cosine":
        raise ValueError("The initial OPSD configuration supports lr_scheduler_type=cosine.")
    if not 0.0 <= float(cfg.lr_warmup_ratio) < 1.0:
        raise ValueError("lr_warmup_ratio must be in [0, 1).")
    if int(cfg.max_checkpoints) <= 0:
        raise ValueError("max_checkpoints must be positive.")
    requested_gpus = int(opsd.distributed.num_processes)
    if requested_gpus not in {4, 8}:
        raise ValueError("OPSD supports the documented 4-GPU or 8-GPU launch modes.")
    if world_size is not None and world_size != requested_gpus:
        raise ValueError(
            f"Accelerate world size {world_size} does not match configured {requested_gpus}."
        )
    if int(opsd.rollout.num_inference_steps) != int(cfg.eval_num_inference_steps):
        raise ValueError(
            "OPSD rollout steps must interpolate from Fast-WAM eval_num_inference_steps."
        )
    if int(opsd.rollout.action_horizon) != int(cfg.data.train.num_frames) - 1:
        raise ValueError("OPSD action horizon must match Fast-WAM data.train.num_frames - 1.")
    if not bool(cfg.model.skip_dit_load_from_pretrain):
        raise ValueError("OPSD must skip redundant video-DiT pretraining before checkpoint load.")
    if cfg.model.action_dit_pretrained_path is not None:
        raise ValueError("OPSD must skip redundant ActionDiT pretraining before checkpoint load.")
    if not bool(cfg.model.load_text_encoder):
        raise ValueError("On-policy OPSD prompts require the Fast-WAM text encoder.")
    local_rollouts = (
        len(opsd.rollout.suites) * 10 * int(opsd.rollout.rollouts_per_task) // requested_gpus
    )
    total_rollouts = len(opsd.rollout.suites) * 10 * int(opsd.rollout.rollouts_per_task)
    if total_rollouts % requested_gpus != 0:
        raise ValueError("The global rollout schedule must divide evenly across ranks.")
    accumulation = int(cfg.gradient_accumulation_steps)
    if accumulation <= 0 or local_rollouts % accumulation != 0:
        raise ValueError(
            "Per-rank rollout count must be divisible by gradient_accumulation_steps "
            "so no final accumulated gradient is discarded."
        )
    fault_cfg = OmegaConf.to_container(cfg.fault, resolve=True)
    pipeline = build_fault_pipeline(fault_cfg)
    adapter_cfg = FastWAMLoraConfig.from_config(OmegaConf.to_container(cfg.adapter, resolve=True))
    if str(cfg.teacher_input.type) != "clean_visual":
        raise ValueError("The camera-pose smoke configuration requires clean_visual Teacher input.")
    return {
        "config_sha256": resolved_config_hash(cfg),
        "fault": pipeline.metadata(),
        "adapter": {
            "rank": adapter_cfg.rank,
            "alpha": adapter_cfg.alpha,
            "dropout": adapter_cfg.dropout,
        },
        "num_processes": requested_gpus,
        "num_inference_steps": int(opsd.rollout.num_inference_steps),
        "action_horizon": int(opsd.rollout.action_horizon),
        "gradient_accumulation_steps": accumulation,
        "lr_scheduler_type": str(cfg.lr_scheduler_type),
        "lr_warmup_ratio": float(cfg.lr_warmup_ratio),
        "num_epochs": int(opsd.num_epochs),
        "epoch_offset": int(opsd.epoch_offset),
        "rollouts_per_task": int(opsd.rollout.rollouts_per_task),
        "schedule_seed": int(opsd.rollout.schedule_seed),
        "schedule_policy": "global_deterministic_shuffle_then_rank_stride",
        "max_checkpoints": int(cfg.max_checkpoints),
    }


def _capture_current_observation(env: Any) -> Any:
    current = env
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        method = getattr(current, "_get_observations", None)
        if callable(method):
            try:
                return method(force_update=True)
            except TypeError:
                return method()
        next_env = getattr(current, "env", None)
        if next_env is None:
            break
        current = next_env
    raise AttributeError("LIBERO environment does not expose _get_observations().")


def _execute_action_chunk(
    *,
    env: Any,
    pipeline,
    raw_observation: Any,
    student_observation: Any,
    action_chunk,
    episode_index: int,
    start_step: int,
) -> tuple[Any, Any, bool, int]:
    raw_obs = raw_observation
    obs = student_observation
    done = False
    steps_taken = 0
    for offset, policy_action in enumerate(action_chunk):
        context = pipeline.context(episode_index, start_step + offset)
        executed_action = pipeline.transform_action(policy_action, env, context)
        pipeline.before_step(env, executed_action, context)
        next_raw, reward, done, info = env.step(executed_action)
        next_obs = pipeline.transform_observation(next_raw, env, context)
        transition = FaultTransition(
            raw_observation=raw_obs,
            student_observation=obs,
            policy_action=policy_action,
            executed_action=executed_action,
            next_raw_observation=next_raw,
            next_student_observation=next_obs,
            reward=float(reward),
            done=bool(done),
            info={} if info is None else dict(info),
            fault_metadata=pipeline.metadata()["faults"],
        )
        pipeline.after_step(env, transition, context)
        raw_obs, obs = next_raw, next_obs
        steps_taken += 1
        if done:
            break
    return raw_obs, obs, bool(done), steps_taken


def _resolve_resume(output_dir: Path, resume: str | None) -> Path | None:
    if resume in {None, "", "null"}:
        return None
    if resume != "auto":
        path = Path(str(resume)).expanduser()
        return path if path.is_absolute() else (Path.cwd() / path).resolve()
    candidates = sorted((output_dir / "checkpoints" / "state").glob("step_*"))
    return candidates[-1] if candidates else None


def run_opsd_training(cfg: DictConfig) -> None:
    """Run fixed-count on-policy LIBERO fragments on exactly 4 or 8 ranks."""
    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        mixed_precision=str(cfg.mixed_precision),
    )
    summary = validate_opsd_config(cfg, world_size=accelerator.num_processes)
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
        (output_dir / "provenance.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    process_seed = int(cfg.seed) + accelerator.process_index
    set_global_seed(process_seed, get_worker_init_fn=False)
    model_dtype = _mixed_precision_to_model_dtype(str(cfg.mixed_precision))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(accelerator.device))
    _load_model_checkpoint(model, str(Path(str(cfg.ckpt)).expanduser().resolve()))
    adapter_cfg = FastWAMLoraConfig.from_config(OmegaConf.to_container(cfg.adapter, resolve=True))
    adapter_audit = inject_fastwam_aligned_lora(model, adapter_cfg)
    initial_adapter = cfg.adapter.get("initial_checkpoint")
    if initial_adapter not in {None, "", "null"}:
        initial_adapter_path = Path(str(initial_adapter)).expanduser().resolve()
        if not initial_adapter_path.is_file():
            raise FileNotFoundError(f"Initial LoRA checkpoint not found: {initial_adapter_path}")
        initial_state = torch.load(initial_adapter_path, map_location="cpu", weights_only=True)
        if not isinstance(initial_state, dict):
            raise TypeError("Initial LoRA checkpoint must contain a state-dict mapping.")
        load_adapter_state_dict(model, initial_state)
    if accelerator.is_main_process:
        (output_dir / "adapter_audit.json").write_text(
            json.dumps(adapter_audit, indent=2) + "\n", encoding="utf-8"
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        betas=(0.9, 0.95),
    )
    local_trajectories = (
        len(cfg.opsd.rollout.suites)
        * 10
        * int(cfg.opsd.rollout.rollouts_per_task)
        // accelerator.num_processes
    )
    total_optimizer_steps = (
        local_trajectories * int(cfg.opsd.num_epochs) // int(cfg.gradient_accumulation_steps)
    )
    warmup_steps = int(total_optimizer_steps * float(cfg.lr_warmup_ratio))

    def learning_rate_multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_optimizer_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    scoring_model = OPSDScoringModule(model)
    scoring_model, optimizer, scheduler = accelerator.prepare(scoring_model, optimizer, scheduler)
    trainer = OPSDFlowTrainer(
        model=scoring_model,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        pointwise_clip=(
            float(cfg.opsd.loss.pointwise_clip)
            if bool(cfg.opsd.loss.pointwise_clip_enabled)
            else None
        ),
        max_grad_norm=float(cfg.max_grad_norm),
    )
    config_hash = summary["config_sha256"]
    resume_path = _resolve_resume(output_dir, str(cfg.resume) if cfg.resume is not None else None)
    if resume_path is not None:
        trainer.load_checkpoint(resume_path, config_hash=config_hash)
        if trainer.epoch >= int(cfg.opsd.num_epochs):
            logging.info("Checkpoint already completed the configured OPSD epoch.")
            accelerator.wait_for_everyone()
            return

    dataset_stats = load_dataset_stats_from_json(
        str(Path(str(cfg.opsd.dataset_stats_path)).expanduser().resolve())
    )
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    video_h, video_w = (int(value) for value in cfg.data.train.video_size)
    suites = [str(value) for value in cfg.opsd.rollout.suites]
    rollouts_per_task = int(cfg.opsd.rollout.rollouts_per_task)
    global_rollouts_per_epoch = len(suites) * 10 * rollouts_per_task
    local_rollouts_per_epoch = global_rollouts_per_epoch // accelerator.num_processes
    completed_before_epoch = trainer.epoch * local_rollouts_per_epoch
    completed_in_resume_epoch = trainer.rollout_index - completed_before_epoch
    if not 0 <= completed_in_resume_epoch <= local_rollouts_per_epoch:
        raise ValueError(
            "Resume checkpoint epoch and rollout_index are inconsistent with this world size."
        )
    resume_rollout_offset = completed_in_resume_epoch
    resume_epoch = trainer.epoch
    metrics_path = output_dir / f"metrics_rank_{accelerator.process_index:02d}.jsonl"

    for epoch_index in range(resume_epoch, int(cfg.opsd.num_epochs)):
        schedule_epoch = epoch_index + int(cfg.opsd.epoch_offset)
        global_schedule = build_rollout_schedule(
            suites=suites,
            tasks_per_suite=10,
            rollouts_per_task=rollouts_per_task,
            epoch=schedule_epoch,
            schedule_seed=int(cfg.opsd.rollout.schedule_seed),
            environment_seed=int(cfg.opsd.rollout.environment_seed),
            inference_seed=int(cfg.opsd.rollout.inference_seed),
        )
        local_schedule = global_schedule[
            accelerator.process_index :: accelerator.num_processes
        ]
        rollout_offset = resume_rollout_offset if epoch_index == resume_epoch else 0
        for local_rollout_index, descriptor in enumerate(local_schedule):
            if local_rollout_index < rollout_offset:
                continue
            suite_name, task_id = descriptor.suite, descriptor.task_id
            suite = benchmark.get_benchmark_dict()[suite_name]()
            task = suite.get_task(task_id)
            states_path = (
                Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            )
            initial_states = torch.load(states_path, weights_only=False)
            if len(initial_states) < rollouts_per_task:
                raise ValueError(
                    f"{suite_name} task {task_id} has {len(initial_states)} initial states, "
                    f"but {rollouts_per_task} distinct states were requested."
                )
            env, task_description = get_libero_env(
                task, LIBERO_ENV_RESOLUTION, descriptor.environment_seed
            )
            fault_cfg = OmegaConf.to_container(cfg.fault, resolve=True)
            pipeline = build_fault_pipeline(fault_cfg)
            episode_index = schedule_epoch * global_rollouts_per_epoch + descriptor.sample_id
            context = pipeline.context(episode_index, 0)
            try:
                env.reset()
                pipeline.on_reset(env, context)
                raw_obs = env.set_init_state(initial_states[descriptor.initial_state_index])
                obs = pipeline.transform_observation(raw_obs, env, context)
                env_step = 0
                for _ in range(int(cfg.opsd.rollout.num_steps_wait)):
                    raw_obs, obs, done, taken = _execute_action_chunk(
                        env=env,
                        pipeline=pipeline,
                        raw_observation=raw_obs,
                        student_observation=obs,
                        action_chunk=[get_libero_dummy_action()],
                        episode_index=episode_index,
                        start_step=env_step,
                    )
                    env_step += taken
                    if done:
                        raise RuntimeError("LIBERO task terminated during the warm-up period.")

                paired = capture_paired_observation(
                    env=env,
                    pipeline=pipeline,
                    capture_raw_observation=lambda: _capture_current_observation(env),
                    context=pipeline.context(episode_index, env_step),
                )
                student_image, student_proprio, _ = _obs_to_model_input(
                    obs=paired.student,
                    cfg=cfg,
                    processor=processor,
                    width=video_w,
                    height=video_h,
                    device=str(accelerator.device),
                    dtype=model_dtype,
                )
                prompt = DEFAULT_PROMPT.format(task=task_description)
                student_conditioning = ActionConditioning(
                    input_image=student_image,
                    prompt=prompt,
                    proprio=student_proprio,
                )
                base_model = trainer.fastwam
                trajectory = generate_student_trajectory(
                    base_model,
                    student_conditioning,
                    action_horizon=int(cfg.opsd.rollout.action_horizon),
                    num_inference_steps=int(cfg.opsd.rollout.num_inference_steps),
                    sigma_shift=float(cfg.opsd.rollout.sigma_shift),
                    seed=descriptor.inference_seed,
                    rand_device=str(cfg.opsd.rollout.rand_device),
                    compile_action_infer=bool(cfg.opsd.rollout.compile_action_infer),
                )
                validate_trace_against_scheduler(base_model, trajectory)

                def encode_privileged(privileged_obs: object) -> torch.Tensor:
                    image, _, _ = _obs_to_model_input(
                        obs=privileged_obs,
                        cfg=cfg,
                        processor=processor,
                        width=video_w,
                        height=video_h,
                        device=str(accelerator.device),
                        dtype=model_dtype,
                    )
                    return image

                provider = build_teacher_input_provider(
                    OmegaConf.to_container(cfg.teacher_input, resolve=True),
                    image_encoder=encode_privileged,
                )
                teacher_conditioning = provider.build(
                    TeacherInputContext(
                        paired_observation=paired,
                        student_conditioning=student_conditioning,
                        student_trajectory=trajectory,
                        extras={"suite": suite_name, "task_id": task_id},
                    )
                )
                metrics = trainer.train_trajectory(
                    student_conditioning=student_conditioning,
                    teacher_conditioning=teacher_conditioning,
                    trajectory=trajectory,
                )
                record = {
                    "epoch": schedule_epoch + 1,
                    "global_step": trainer.global_step,
                    "rollout_index": trainer.rollout_index,
                    "suite": suite_name,
                    "task_id": task_id,
                    "rollout_in_task": descriptor.rollout_id,
                    "initial_state_index": descriptor.initial_state_index,
                    "environment_seed": descriptor.environment_seed,
                    "inference_seed": descriptor.inference_seed,
                    "sample_id": descriptor.sample_id,
                    "faults": pipeline.metadata()["faults"],
                    **metrics,
                }
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")

                normalized_action = trajectory.final_action
                action = _denormalize_action(normalized_action, processor)[0]
                action[..., -1] = action[..., -1] * 2 - 1
                action = invert_gripper_action(action)
                if bool(cfg.opsd.rollout.binarize_gripper):
                    action[..., -1] = np.sign(action[..., -1])
                raw_obs, obs, done, taken = _execute_action_chunk(
                    env=env,
                    pipeline=pipeline,
                    raw_observation=raw_obs,
                    student_observation=obs,
                    action_chunk=action[: int(cfg.opsd.rollout.replan_steps)],
                    episode_index=episode_index,
                    start_step=env_step,
                )
                env_step += taken
            finally:
                pipeline.detach(env)
                close_fn = getattr(env, "close", None)
                if close_fn is not None:
                    close_fn()

            accelerator.wait_for_everyone()
            save_every = int(cfg.save_every)
            before_final_rollout = local_rollout_index + 1 < len(local_schedule)
            if (
                save_every > 0
                and trainer.global_step > 0
                and trainer.global_step % save_every == 0
                and before_final_rollout
            ):
                trainer.save_checkpoint(output_dir, config_hash=config_hash, fault_state={})
                trainer.prune_checkpoints(output_dir, keep=int(cfg.max_checkpoints))

        trainer.epoch = epoch_index + 1
        trainer.save_checkpoint(output_dir, config_hash=config_hash, fault_state={})
        trainer.prune_checkpoints(output_dir, keep=int(cfg.max_checkpoints))
        logging.info(
            "Completed OPSD epoch %d/%d at global step %d.",
            trainer.epoch,
            int(cfg.opsd.num_epochs),
            trainer.global_step,
        )
        resume_rollout_offset = 0
