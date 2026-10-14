"""Distributed single-trajectory actor-critic FPO for LIBERO embodied Faults."""

from __future__ import annotations

import json
import logging
from typing import Any

import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from experiments.libero.eval_libero_single import (
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from resilient.faults import build_fault_pipeline
from resilient.faults.validation import validate_embodied_fault_contract
from resilient.opsd.adapters import (
    FastWAMLoraConfig,
    inject_fastwam_aligned_lora,
    load_adapter_state_dict,
)
from resilient.state_banks import assert_disjoint_manifests, load_manifest

from ..runtime import (
    _checkpoint_retention,
    _resolve_project_path,
    _resolve_resume,
    _training_state_records,
    build_outcome_rollout_schedule,
    resolved_config_hash,
)
from .collector import collect_ppo_episode
from .model import ActorCriticFPOScoringModule
from .trainer import ActorCriticFPOTrainer


def validate_actor_critic_fpo_config(
    cfg: DictConfig, *, world_size: int | None = None
) -> dict[str, Any]:
    """Enforce a fixed-scale, split-safe E5 PPO-style comparison."""
    section = cfg.outcome_fpo
    ppo = section.ppo
    gpu_count = int(section.distributed.num_processes)
    if gpu_count not in {4, 8} or (world_size is not None and world_size != gpu_count):
        raise ValueError("Actor-critic FPO requires exactly 4 or 8 matching GPU processes.")
    if str(section.algorithm) != "actor_critic_fpo":
        raise ValueError("Expected outcome_fpo.algorithm=actor_critic_fpo.")
    if int(section.num_epochs) <= 0:
        raise ValueError("num_epochs must be positive.")
    if int(section.rollout.rollouts_per_task) != 50 or int(ppo.episodes_per_epoch) != 48:
        raise ValueError("The E5 comparison uses 50 bank states and 48 rotated episodes per epoch.")
    if int(ppo.episodes_per_epoch) % gpu_count:
        raise ValueError("PPO episode count must divide evenly across training GPUs.")
    if len(section.rollout.suites) != 1 or list(section.rollout.task_ids) != [7]:
        raise ValueError("The initial PPO experiment is restricted to LIBERO-10 task 7.")
    if list(section.rollout.anchor_steps) != [0]:
        raise ValueError("PPO runs one causal main trajectory from each selected initial state.")
    if int(section.rollout.action_horizon) != 32 or int(section.rollout.replan_steps) != 32:
        raise ValueError("The first PPO comparison executes complete 32-action chunks.")
    if list(section.rollout.future_frame_steps) != list(range(0, 33, 4)):
        raise ValueError("PPO task-direction reward requires frames at steps 0,4,...,32.")
    if int(section.outcome.num_video_frames) != 9:
        raise ValueError("PPO task-direction reward requires nine video frames.")
    if int(section.rollout.num_inference_steps) != int(cfg.eval_num_inference_steps):
        raise ValueError("Action sampling must preserve Fast-WAM's inference steps.")
    if int(section.fpo.num_mc_samples) != 8 or str(section.fpo.loss_type) != "epsilon_mse":
        raise ValueError("The PPO comparison reuses the eight-sample epsilon-MSE FPO scorer.")
    if int(ppo.max_chunks_per_episode) != 16 or int(ppo.minibatch_size) != 4:
        raise ValueError("First E5 PPO comparison fixes 16 chunks and minibatches of four.")
    if cfg.adapter.get("initial_checkpoint") not in {None, "", "null"}:
        raise ValueError("The direct PPO comparison must start from Fast-WAM without Stage I.")
    if int(ppo.value_hidden_dim) <= 0 or float(ppo.value_learning_rate) <= 0:
        raise ValueError("PPO value head dimension and learning rate must be positive.")
    if float(ppo.reward_scale) <= 0:
        raise ValueError("PPO reward scale must be positive.")
    if not (0 <= float(ppo.gamma) <= 1 and 0 <= float(ppo.gae_lambda) <= 1):
        raise ValueError("PPO gamma and GAE lambda are outside their valid ranges.")
    task = cfg.task_decoupling
    if not bool(task.enabled) or str(task.reward.type) != "task_direction":
        raise ValueError("This experiment uses only task-direction reward.")
    if str(task.mask_provider.type) != "samhq_file_service":
        raise ValueError("The E5 comparison reuses the HQ-SAM oracle-box provider.")
    training_path = _resolve_project_path(str(section.split.training_reference_manifest))
    validation_path = _resolve_project_path(str(section.split.validation_manifest))
    training = load_manifest(training_path)
    validation = load_manifest(validation_path)
    assert_disjoint_manifests(validation, training)
    suite = str(section.rollout.suites[0])
    for name, manifest in (("training", training), ("validation", validation)):
        selected = [
            row
            for row in manifest["states"]
            if str(row["suite"]) == suite and int(row["task_id"]) == 7
        ]
        if len(selected) != 50:
            raise ValueError(f"{name} state bank must contain 50 LIBERO-10 task-7 states.")
    fault = build_fault_pipeline(OmegaConf.to_container(cfg.fault, resolve=True)).metadata()
    expected = cfg.get("expected_embodied_fault")
    validate_embodied_fault_contract(
        fault,
        None if expected is None else OmegaConf.to_container(expected, resolve=True),
    )
    adapter = FastWAMLoraConfig.from_config(OmegaConf.to_container(cfg.adapter, resolve=True))
    return {
        "config_sha256": resolved_config_hash(cfg),
        "method": "single_trajectory_ppo_style_fpo",
        "ratio_is_approximate": True,
        "reward": "task_direction_only_samhq_oracle_box_simulation",
        "future_source": "main_on_policy_full_action_chunk",
        "train_states": 50,
        "validation_states": 50,
        "episodes_per_epoch": int(ppo.episodes_per_epoch),
        "maximum_chunks_per_epoch": int(ppo.episodes_per_epoch) * 16,
        "maximum_fault_control_steps_per_epoch": int(ppo.episodes_per_epoch) * 16 * 32,
        "num_processes": gpu_count,
        "adapter": {"rank": adapter.rank, "alpha": adapter.alpha},
        "fault": fault,
    }


def run_actor_critic_fpo_training(cfg: DictConfig) -> None:
    """Train the direct-from-base E5 recovery adapter without touching GRPO state."""
    accelerator = Accelerator(mixed_precision=str(cfg.mixed_precision))
    provenance = validate_actor_critic_fpo_config(cfg, world_size=accelerator.num_processes)
    output_dir = _resolve_project_path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = _resolve_resume(output_dir, None if cfg.resume is None else str(cfg.resume))
    if resume is not None:
        stored = json.loads((resume / "trainer_state.json").read_text(encoding="utf-8"))
        if stored["config_sha256"] != provenance["config_sha256"]:
            raise ValueError("PPO checkpoint requires the exact original resolved configuration.")
    if accelerator.is_main_process:
        OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
        (output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    set_global_seed(int(cfg.seed) + accelerator.process_index, get_worker_init_fn=False)
    model = instantiate(
        cfg.model,
        model_dtype=_mixed_precision_to_model_dtype(str(cfg.mixed_precision)),
        device=str(accelerator.device),
    )
    _load_model_checkpoint(model, str(_resolve_project_path(str(cfg.ckpt))))
    adapter_cfg = FastWAMLoraConfig.from_config(OmegaConf.to_container(cfg.adapter, resolve=True))
    audit = inject_fastwam_aligned_lora(model, adapter_cfg)
    initial_adapter = cfg.adapter.get("initial_checkpoint")
    if initial_adapter not in {None, "", "null"}:
        state = torch.load(
            _resolve_project_path(str(initial_adapter)), map_location="cpu", weights_only=True
        )
        load_adapter_state_dict(model, state)
    if accelerator.is_main_process:
        (output_dir / "adapter_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
    wrapped = ActorCriticFPOScoringModule(
        model, hidden_dim=int(cfg.outcome_fpo.ppo.value_hidden_dim)
    )
    actor_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": actor_parameters, "lr": float(cfg.learning_rate)},
            {
                "params": list(wrapped.value_head.parameters()),
                "lr": float(cfg.outcome_fpo.ppo.value_learning_rate),
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=float(cfg.weight_decay),
    )
    wrapped, optimizer = accelerator.prepare(wrapped, optimizer)
    ppo = cfg.outcome_fpo.ppo
    trainer = ActorCriticFPOTrainer(
        scoring_model=wrapped,
        optimizer=optimizer,
        accelerator=accelerator,
        update_epochs=int(cfg.outcome_fpo.fpo.update_epochs),
        clipping_epsilon=float(cfg.outcome_fpo.fpo.clipping_epsilon),
        log_ratio_clip=float(cfg.outcome_fpo.fpo.log_ratio_clip),
        advantage_clip=float(cfg.outcome_fpo.fpo.advantage_clip),
        advantage_eps=float(cfg.outcome_fpo.fpo.advantage_eps),
        max_grad_norm=float(cfg.max_grad_norm),
        max_chunks_per_episode=int(ppo.max_chunks_per_episode),
        minibatch_size=int(ppo.minibatch_size),
        gamma=float(ppo.gamma),
        gae_lambda=float(ppo.gae_lambda),
        reward_center=float(ppo.reward_center),
        reward_scale=float(ppo.reward_scale),
        value_coefficient=float(ppo.value_coefficient),
        value_clip_range=float(ppo.value_clip_range),
    )
    if resume is not None:
        trainer.load_checkpoint(resume, config_hash=provenance["config_sha256"])
    stats = load_dataset_stats_from_json(
        str(_resolve_project_path(str(cfg.outcome_fpo.dataset_stats_path)))
    )
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    video_height, video_width = (int(value) for value in cfg.data.train.video_size)
    records = _training_state_records(
        _resolve_project_path(str(cfg.outcome_fpo.split.training_reference_manifest))
    )
    local_count = int(ppo.episodes_per_epoch) // accelerator.num_processes
    offset = trainer.group_index - trainer.epoch * local_count
    if not 0 <= offset <= local_count:
        raise ValueError("PPO checkpoint episode index does not match this world size.")
    rollout_log = output_dir / f"ppo_rollouts_rank_{accelerator.process_index:02d}.jsonl"
    update_log = output_dir / f"ppo_updates_rank_{accelerator.process_index:02d}.jsonl"
    retention = _checkpoint_retention(cfg)
    for epoch_index in range(trainer.epoch, int(cfg.outcome_fpo.num_epochs)):
        schedule_epoch = epoch_index + int(cfg.outcome_fpo.epoch_offset)
        schedule = build_outcome_rollout_schedule(
            suites=[str(value) for value in cfg.outcome_fpo.rollout.suites],
            task_ids=[7],
            rollouts_per_task=50,
            anchor_steps=[0],
            epoch=schedule_epoch,
            schedule_seed=int(cfg.outcome_fpo.rollout.schedule_seed),
            environment_seed=int(cfg.outcome_fpo.rollout.environment_seed),
            inference_seed=int(cfg.outcome_fpo.rollout.inference_seed),
        )[: int(ppo.episodes_per_epoch)][accelerator.process_index :: accelerator.num_processes]
        for descriptor in schedule[offset if epoch_index == trainer.epoch else 0 :]:
            episode = collect_ppo_episode(
                cfg=cfg,
                descriptor=descriptor,
                schedule_epoch=schedule_epoch,
                processor=processor,
                model=trainer.fastwam,
                value_model=accelerator.unwrap_model(wrapped),
                video_height=video_height,
                video_width=video_width,
                training_records=records,
            )
            with rollout_log.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "epoch": schedule_epoch + 1,
                            **episode.metadata,
                            "rewards": [item.reward for item in episode.transitions],
                            "old_values": [item.old_value for item in episode.transitions],
                        }
                    )
                    + "\n"
                )
            metrics = trainer.train_episode(episode)
            with update_log.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "epoch": schedule_epoch + 1,
                            "global_step": trainer.global_step,
                            "local_episode_index": trainer.group_index,
                            "sample_id": descriptor.sample_id,
                            **metrics,
                        }
                    )
                    + "\n"
                )
            del episode
        trainer.epoch = epoch_index + 1
        checkpoint = trainer.save_checkpoint(
            output_dir,
            config_hash=provenance["config_sha256"],
            fault_state={},
            epoch_number=schedule_epoch + 1,
        )
        if accelerator.is_main_process:
            OmegaConf.save(cfg, checkpoint / "resolved_config.yaml", resolve=True)
        accelerator.wait_for_everyone()
        trainer.prune_checkpoints(
            output_dir,
            keep=retention.rolling_keep_last,
            keep_epochs=retention.epoch_keep_last,
            preserve_epochs=retention.preserve_epochs,
            keep_epoch_full_states=retention.epoch_full_state_keep_last,
            preserve_full_state_epochs=retention.preserve_full_state_epochs,
        )
        logging.info("Completed actor-critic FPO epoch %d.", trainer.epoch)
        offset = 0
    accelerator.wait_for_everyone()
    accelerator.end_training()
