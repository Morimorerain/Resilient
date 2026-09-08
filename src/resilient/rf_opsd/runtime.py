"""Hydra runtime for multi-GPU RF-OPSD training on LIBERO."""

from __future__ import annotations

import hashlib
import json
import logging
import math
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
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from resilient.faults import build_fault_pipeline
from resilient.opsd.runtime import build_rollout_schedule

from .adapters import FastWAMWorldLoraConfig, inject_fastwam_world_lora, lora_disabled
from .future import collect_action_chunk_future, stack_observation_video, validate_frame_steps
from .teacher import FrozenFastWAMVAETeacher
from .trainer import RFOPSDTrainer
from .types import VideoFlowBatch
from .video_flow import RFVideoFlowModule


def resolved_config_hash(cfg: DictConfig) -> str:
    """Hash algorithmic settings while allowing only runtime resume location to change."""
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved RF-OPSD configuration must be a mapping.")
    payload["output_dir"] = "<runtime-output-dir>"
    payload["resume"] = None
    if isinstance(payload.get("rf_opsd"), dict):
        payload["rf_opsd"]["validate_only"] = False
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve_resume(output_dir: Path, resume: str | None) -> Path | None:
    if resume in {None, "", "null"}:
        return None
    if resume != "auto":
        path = Path(str(resume)).expanduser()
        return path if path.is_absolute() else (Path.cwd() / path).resolve()
    candidates = sorted((output_dir / "checkpoints" / "state").glob("step_*"))
    return candidates[-1] if candidates else None


def validate_rf_opsd_config(
    cfg: DictConfig, *, world_size: int | None = None
) -> dict[str, Any]:
    """Validate reproducibility and Fast-WAM alignment without loading CUDA weights."""
    settings = cfg.rf_opsd
    if not bool(settings.enabled):
        raise ValueError("Dedicated RF-OPSD training requires rf_opsd.enabled=true.")
    requested_gpus = int(settings.distributed.num_processes)
    if requested_gpus not in {4, 8}:
        raise ValueError("RF-OPSD supports the documented 4-GPU or 8-GPU modes.")
    if world_size is not None and int(world_size) != requested_gpus:
        raise ValueError("Accelerate world size does not match RF-OPSD configuration.")
    if int(settings.num_epochs) <= 0:
        raise ValueError("rf_opsd.num_epochs must be positive.")
    rollout = settings.rollout
    action_horizon = int(rollout.action_horizon)
    if action_horizon != int(cfg.data.train.num_frames) - 1:
        raise ValueError("RF-OPSD action horizon must match Fast-WAM training.")
    if int(rollout.num_inference_steps) != int(cfg.eval_num_inference_steps):
        raise ValueError("RF-OPSD must use Fast-WAM's configured inference step count.")
    frame_steps = validate_frame_steps(rollout.future_frame_steps, action_horizon)
    if frame_steps != tuple(range(0, action_horizon + 1, int(rollout.frame_stride))):
        raise ValueError("RF-OPSD future frames must follow the configured uniform stride.")
    expected_frames = (
        int(cfg.data.train.num_frames) - 1
    ) // int(cfg.data.train.action_video_freq_ratio) + 1
    if len(frame_steps) != expected_frames:
        raise ValueError("RF-OPSD future frame count must match Fast-WAM video training.")
    if str(settings.future_source.type) != "shadow_action_chunk":
        raise ValueError("The initial RF-OPSD implementation supports shadow_action_chunk.")
    if bool(cfg.model.video_dit_config.action_conditioned):
        raise ValueError("Released RF-OPSD v1 expects action_conditioned=false.")
    if (
        not bool(cfg.model.skip_dit_load_from_pretrain)
        or cfg.model.action_dit_pretrained_path is not None
    ):
        raise ValueError("RF-OPSD must load the released checkpoint without redundant DiT weights.")
    if not bool(cfg.model.load_text_encoder):
        raise ValueError("RF-OPSD on-policy prompts require the text encoder.")
    tasks_per_suite = int(rollout.tasks_per_suite)
    if not 1 <= tasks_per_suite <= 10:
        raise ValueError("RF-OPSD tasks_per_suite must be in [1, 10].")
    total_fragments = (
        len(rollout.suites) * tasks_per_suite * int(rollout.rollouts_per_task)
    )
    if total_fragments % requested_gpus:
        raise ValueError("The global RF-OPSD schedule must divide evenly across ranks.")
    local_fragments = total_fragments // requested_gpus
    accumulation = int(cfg.gradient_accumulation_steps)
    if accumulation <= 0 or local_fragments % accumulation:
        raise ValueError("Per-rank fragments must divide gradient accumulation exactly.")
    fault_cfg = OmegaConf.to_container(cfg.fault, resolve=True)
    pipeline = build_fault_pipeline(fault_cfg)
    faults = pipeline.metadata()["faults"]
    if len(faults) != 1 or faults[0]["family"] != "structure.joint_motion":
        raise ValueError("RF-OPSD v1 is scoped to one structure.joint_motion Fault.")
    adapter_cfg = FastWAMWorldLoraConfig.from_config(
        OmegaConf.to_container(cfg.rf_world_adapter, resolve=True)
    )
    return {
        "schema_version": 1,
        "config_sha256": resolved_config_hash(cfg),
        "method": "RF-OPSD world-representation distillation",
        "teacher": "frozen_fastwam_vae",
        "future_semantics": "realized_counterfactual_student_action_chunk",
        "fault": pipeline.metadata(),
        "adapter": {
            "rank": adapter_cfg.rank,
            "alpha": adapter_cfg.alpha,
            "dropout": adapter_cfg.dropout,
            "target_roots": ["video_expert"],
        },
        "num_processes": requested_gpus,
        "num_epochs": int(settings.num_epochs),
        "rollouts_per_task": int(rollout.rollouts_per_task),
        "tasks_per_suite": tasks_per_suite,
        "total_fragments_per_epoch": total_fragments,
        "frame_steps": list(frame_steps),
        "action_horizon": action_horizon,
        "replan_steps": int(rollout.replan_steps),
        "num_inference_steps": int(rollout.num_inference_steps),
    }


def _prepare_shadow(shadow_env: Any, main_env: Any) -> Any:
    shadow_env.reset()
    observation = shadow_env.set_init_state(main_env.get_sim_state())
    if hasattr(main_env, "fault_runtime_state_dict") and hasattr(
        shadow_env, "load_fault_runtime_state_dict"
    ):
        observation = shadow_env.load_fault_runtime_state_dict(
            main_env.fault_runtime_state_dict()
        )
    return observation


def _execute_prefix(env: Any, action_chunk: np.ndarray, count: int) -> tuple[bool, int]:
    success = False
    taken = 0
    for action in action_chunk[: int(count)]:
        _, _, done, _ = env.step(action)
        success = success or bool(done)
        taken += 1
        if done:
            break
    return success, taken


def run_rf_opsd_training(cfg: DictConfig) -> None:
    """Train a video-only world adapter from Student-realized Fault futures."""
    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        mixed_precision=str(cfg.mixed_precision),
    )
    summary = validate_rf_opsd_config(cfg, world_size=accelerator.num_processes)
    output_dir = Path(str(cfg.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
        (output_dir / "provenance.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    set_global_seed(int(cfg.seed) + accelerator.process_index, get_worker_init_fn=False)
    model_dtype = _mixed_precision_to_model_dtype(str(cfg.mixed_precision))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(accelerator.device))
    _load_model_checkpoint(model, str(Path(str(cfg.ckpt)).expanduser().resolve()))
    adapter_cfg = FastWAMWorldLoraConfig.from_config(
        OmegaConf.to_container(cfg.rf_world_adapter, resolve=True)
    )
    adapter_audit = inject_fastwam_world_lora(model, adapter_cfg)
    teacher = FrozenFastWAMVAETeacher(model)
    if accelerator.is_main_process:
        (output_dir / "adapter_audit.json").write_text(
            json.dumps(adapter_audit, indent=2) + "\n", encoding="utf-8"
        )
        (output_dir / "teacher_audit.json").write_text(
            json.dumps(teacher.audit(), indent=2) + "\n", encoding="utf-8"
        )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
        betas=(0.9, 0.95),
    )
    rollout = cfg.rf_opsd.rollout
    tasks_per_suite = int(rollout.tasks_per_suite)
    global_fragments = (
        len(rollout.suites) * tasks_per_suite * int(rollout.rollouts_per_task)
    )
    local_fragments = global_fragments // accelerator.num_processes
    total_steps = local_fragments * int(cfg.rf_opsd.num_epochs) // int(
        cfg.gradient_accumulation_steps
    )
    warmup_steps = int(total_steps * float(cfg.lr_warmup_ratio))

    def lr_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scoring_model = RFVideoFlowModule(
        model,
        mse_weight=float(cfg.rf_opsd.loss.mse_weight),
        smooth_l1_weight=float(cfg.rf_opsd.loss.smooth_l1_weight),
        cosine_weight=float(cfg.rf_opsd.loss.cosine_weight),
        exclude_initial_latent=bool(cfg.rf_opsd.loss.exclude_initial_latent),
    )
    scoring_model, optimizer, scheduler = accelerator.prepare(
        scoring_model, optimizer, scheduler
    )
    trainer = RFOPSDTrainer(
        model=scoring_model,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        max_grad_norm=float(cfg.max_grad_norm),
    )
    teacher = FrozenFastWAMVAETeacher(trainer.fastwam)
    config_hash = summary["config_sha256"]
    resume_path = _resolve_resume(output_dir, cfg.resume)
    if resume_path is not None:
        trainer.load_checkpoint(resume_path, config_hash=config_hash)

    stats = load_dataset_stats_from_json(
        str(Path(str(cfg.rf_opsd.dataset_stats_path)).expanduser().resolve())
    )
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    video_h, video_w = (int(value) for value in cfg.data.train.video_size)
    suites = [str(value) for value in rollout.suites]
    rollouts_per_task = int(rollout.rollouts_per_task)
    local_per_epoch = global_fragments // accelerator.num_processes
    completed_in_epoch = trainer.fragment_index - trainer.epoch * local_per_epoch
    if not 0 <= completed_in_epoch <= local_per_epoch:
        raise ValueError("RF-OPSD resume fragment offset is inconsistent.")
    metrics_path = output_dir / f"metrics_rank_{accelerator.process_index:02d}.jsonl"

    for epoch_index in range(trainer.epoch, int(cfg.rf_opsd.num_epochs)):
        schedule_epoch = epoch_index + int(cfg.rf_opsd.epoch_offset)
        schedule = build_rollout_schedule(
            suites=suites,
            tasks_per_suite=tasks_per_suite,
            rollouts_per_task=rollouts_per_task,
            epoch=schedule_epoch,
            schedule_seed=int(rollout.schedule_seed),
            environment_seed=int(rollout.environment_seed),
            inference_seed=int(rollout.inference_seed),
        )[accelerator.process_index :: accelerator.num_processes]
        offset = completed_in_epoch if epoch_index == trainer.epoch else 0
        for local_index, descriptor in enumerate(schedule):
            if local_index < offset:
                continue
            suite = benchmark.get_benchmark_dict()[descriptor.suite]()
            task = suite.get_task(descriptor.task_id)
            states_path = (
                Path(get_libero_path("init_states"))
                / task.problem_folder
                / task.init_states_file
            )
            initial_states = torch.load(states_path, weights_only=False)
            fault_config = OmegaConf.to_container(cfg.fault, resolve=True)
            episode_index = schedule_epoch * global_fragments + descriptor.sample_id
            main_env, task_description = get_libero_env(
                task,
                LIBERO_ENV_RESOLUTION,
                descriptor.environment_seed,
                fault_pipeline=build_fault_pipeline(fault_config),
                fault_episode_index=episode_index,
            )
            shadow_env, _ = get_libero_env(
                task,
                LIBERO_ENV_RESOLUTION,
                descriptor.environment_seed,
                fault_pipeline=build_fault_pipeline(fault_config),
                fault_episode_index=episode_index,
            )
            try:
                main_env.reset()
                observation = main_env.set_init_state(
                    initial_states[descriptor.initial_state_index]
                )
                for _ in range(int(rollout.num_steps_wait)):
                    observation, _, done, _ = main_env.step(get_libero_dummy_action())
                    if done:
                        raise RuntimeError("LIBERO task terminated during RF-OPSD warm-up.")
                image, proprio, _ = _obs_to_model_input(
                    observation,
                    cfg=cfg,
                    processor=processor,
                    width=video_w,
                    height=video_h,
                    device=str(accelerator.device),
                    dtype=model_dtype,
                )
                prompt = DEFAULT_PROMPT.format(task=task_description)
                context, context_mask = trainer.fastwam.encode_prompt(prompt)
                with torch.no_grad(), lora_disabled(trainer.fastwam):
                    inference = trainer.fastwam.infer_action(
                        prompt=None,
                        input_image=image,
                        action_horizon=int(rollout.action_horizon),
                        proprio=proprio,
                        context=context,
                        context_mask=context_mask,
                        num_inference_steps=int(rollout.num_inference_steps),
                        sigma_shift=float(rollout.sigma_shift),
                        seed=descriptor.inference_seed,
                        rand_device=str(rollout.rand_device),
                        compile_action_infer=bool(rollout.compile_action_infer),
                    )
                normalized_action = inference["action"]
                action = _denormalize_action(normalized_action, processor)[0]
                action[..., -1] = action[..., -1] * 2 - 1
                action = invert_gripper_action(action)
                if bool(rollout.binarize_gripper):
                    action[..., -1] = np.sign(action[..., -1])

                shadow_observation = _prepare_shadow(shadow_env, main_env)
                future = collect_action_chunk_future(
                    shadow_env,
                    shadow_observation,
                    action,
                    frame_steps=rollout.future_frame_steps,
                )

                def encode_image(item: Any) -> torch.Tensor:
                    return _obs_to_model_input(
                        item,
                        cfg=cfg,
                        processor=processor,
                        width=video_w,
                        height=video_h,
                        device=str(accelerator.device),
                        dtype=model_dtype,
                    )[0]

                video = stack_observation_video(future.observations, encode_image)
                target_latents = teacher.encode(video)
                metrics = trainer.train_fragment(
                    VideoFlowBatch(
                        target_latents=target_latents,
                        context=context,
                        context_mask=context_mask,
                        proprio=proprio,
                    )
                )
                main_success, main_steps = _execute_prefix(
                    main_env, action, int(rollout.replan_steps)
                )
                record = {
                    "epoch": schedule_epoch + 1,
                    "global_step": trainer.global_step,
                    "fragment_index": trainer.fragment_index,
                    "suite": descriptor.suite,
                    "task_id": descriptor.task_id,
                    "rollout_in_task": descriptor.rollout_id,
                    "initial_state_index": descriptor.initial_state_index,
                    "environment_seed": descriptor.environment_seed,
                    "inference_seed": descriptor.inference_seed,
                    "sample_id": descriptor.sample_id,
                    "future_source": "realized_counterfactual_student_action_chunk",
                    "future_success": future.success,
                    "main_prefix_success": main_success,
                    "main_prefix_steps": main_steps,
                    **metrics,
                }
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
            finally:
                main_env.close()
                shadow_env.close()

            accelerator.wait_for_everyone()
            if (
                int(cfg.save_every) > 0
                and trainer.global_step > 0
                and trainer.global_step % int(cfg.save_every) == 0
                and local_index + 1 < len(schedule)
            ):
                trainer.save_checkpoint(output_dir, config_hash=config_hash)
                trainer.prune_checkpoints(output_dir, keep=int(cfg.max_checkpoints))

        trainer.epoch = epoch_index + 1
        trainer.save_checkpoint(output_dir, config_hash=config_hash)
        trainer.prune_checkpoints(output_dir, keep=int(cfg.max_checkpoints))
        logging.info(
            "Completed RF-OPSD epoch %d/%d at step %d.",
            trainer.epoch,
            int(cfg.rf_opsd.num_epochs),
            trainer.global_step,
        )
        completed_in_epoch = 0
