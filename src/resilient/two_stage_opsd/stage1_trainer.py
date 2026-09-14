"""Native Fast-WAM video/action joint LoRA adaptation on Fault rollouts."""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from experiments.libero.eval_libero_single import (
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
)
from fastwam.utils.pytorch_utils import set_global_seed
from resilient.opsd.adapters import (
    FastWAMLoraConfig,
    adapter_state_dict,
    inject_fastwam_aligned_lora,
    load_adapter_state_dict,
)

from .common import resolve_project_path, resolved_config_hash, validate_fault_and_split
from .dataset import FaultRolloutWindowDataset


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted((output_dir / "checkpoints").glob("epoch_*"))
    return checkpoints[-1] if checkpoints else None


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and payload["cuda"]:
        torch.cuda.set_rng_state_all(payload["cuda"])


def _save_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    optimizer,
    scheduler,
    output_dir: Path,
    epoch: int,
    global_step: int,
    config_hash: str,
    keep_last: int,
) -> Path:
    checkpoint = output_dir / "checkpoints" / f"epoch_{epoch:03d}"
    if accelerator.is_main_process:
        checkpoint.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    torch.save(
        optimizer.state_dict(),
        checkpoint / f"optimizer_rank_{accelerator.process_index:02d}.pt",
    )
    torch.save(
        {"scheduler": scheduler.state_dict(), "rng": _rng_state()},
        checkpoint / f"runtime_rank_{accelerator.process_index:02d}.pt",
    )
    if accelerator.is_main_process:
        torch.save(
            adapter_state_dict(accelerator.unwrap_model(model)),
            checkpoint / "joint_adapter.pt",
        )
        _write_json(
            checkpoint / "trainer_state.json",
            {
                "schema_version": 1,
                "config_sha256": config_hash,
                "epoch": epoch,
                "global_step": global_step,
                "checkpoint_boundary": "completed_epoch",
                "world_size": accelerator.num_processes,
            },
        )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoints = sorted((output_dir / "checkpoints").glob("epoch_*"))
        for old in checkpoints[:-int(keep_last)]:
            shutil.rmtree(old)
    accelerator.wait_for_everyone()
    return checkpoint


def _load_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    optimizer,
    scheduler,
    checkpoint: Path,
    config_hash: str,
) -> tuple[int, int]:
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if state["config_sha256"] != config_hash:
        raise ValueError("Stage-I checkpoint configuration hash mismatch.")
    if int(state["world_size"]) != accelerator.num_processes:
        raise ValueError("Stage-I resume requires the same distributed world size.")
    adapter = torch.load(checkpoint / "joint_adapter.pt", map_location="cpu", weights_only=True)
    load_adapter_state_dict(accelerator.unwrap_model(model), adapter)
    optimizer.load_state_dict(
        torch.load(
            checkpoint / f"optimizer_rank_{accelerator.process_index:02d}.pt",
            map_location="cpu",
            weights_only=False,
        )
    )
    runtime = torch.load(
        checkpoint / f"runtime_rank_{accelerator.process_index:02d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    scheduler.load_state_dict(runtime["scheduler"])
    _restore_rng_state(runtime["rng"])
    return int(state["epoch"]), int(state["global_step"])


def run_stage1_training(cfg: DictConfig) -> Path:
    """Train one joint LoRA through Fast-WAM's unmodified native training loss."""
    accumulation = int(cfg.gradient_accumulation_steps)
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation,
        mixed_precision=str(cfg.mixed_precision),
        step_scheduler_with_optimizer=False,
    )
    section = cfg.two_stage_opsd.stage1.training
    expected_world_size = int(cfg.two_stage_opsd.distributed.num_processes)
    if accelerator.num_processes != expected_world_size:
        raise ValueError("Accelerate world size does not match two_stage_opsd.distributed.")
    validated = validate_fault_and_split(cfg)
    if int(cfg.batch_size) * accelerator.num_processes * accumulation != int(
        section.global_batch_size
    ):
        raise ValueError(
            "Local batch, world size, and accumulation do not match global_batch_size."
        )
    if str(cfg.lr_scheduler_type) != "cosine" or int(cfg.num_epochs) != 10:
        raise ValueError("Stage I intentionally matches Fast-WAM's cosine/10-epoch recipe.")
    if bool(cfg.model.video_dit_config.action_conditioned):
        raise ValueError("Stage I must keep released action_conditioned=false.")
    if float(cfg.model.loss.lambda_video) != 1.0 or float(cfg.model.loss.lambda_action) != 1.0:
        raise ValueError("Stage I keeps equal native video/action loss weights.")

    output_dir = resolve_project_path(cfg.output_dir) / "stage1"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = resolved_config_hash(cfg)
    manifest_path = (
        resolve_project_path(cfg.two_stage_opsd.stage1.dataset_dir)
        / "dataset_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != config_hash:
        raise ValueError("Collected Stage-I dataset configuration hash mismatch.")
    expected_windows = int(section.expected_windows)
    if int(manifest["window_count"]) != expected_windows:
        raise ValueError(
            f"Stage-I manifest has {manifest['window_count']} windows, expected {expected_windows}."
        )
    dataset = FaultRolloutWindowDataset(
        manifest_path,
        text_embedding_cache_dir=resolve_project_path(cfg.data.train.text_embedding_cache_dir),
        context_len=int(cfg.data.train.context_len),
        cache_episodes=int(section.cache_episodes),
    )
    set_global_seed(int(cfg.seed) + accelerator.process_index, get_worker_init_fn=False)
    loader_generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        generator=loader_generator,
        drop_last=False,
    )

    dtype = _mixed_precision_to_model_dtype(str(cfg.mixed_precision))
    model = instantiate(cfg.model, model_dtype=dtype, device=str(accelerator.device))
    _load_model_checkpoint(model, str(resolve_project_path(cfg.ckpt)))
    adapter_cfg = FastWAMLoraConfig.from_config(OmegaConf.to_container(cfg.adapter, resolve=True))
    adapter_audit = inject_fastwam_aligned_lora(model, adapter_cfg)
    model.eval()
    model.video_expert.train()
    model.action_expert.train()
    if model.proprio_encoder is not None:
        model.proprio_encoder.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.learning_rate),
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=float(cfg.weight_decay),
    )
    micro_steps_per_epoch = math.ceil(
        len(dataset) / (int(cfg.batch_size) * accelerator.num_processes)
    )
    steps_per_epoch = math.ceil(micro_steps_per_epoch / accumulation)
    total_steps = steps_per_epoch * int(cfg.num_epochs)
    warmup_steps = int(total_steps * float(section.warmup_ratio))
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - warmup_steps, 1),
        eta_min=float(cfg.learning_rate) * 0.01,
    )
    if warmup_steps > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=1.0 / max(warmup_steps, 1),
                    end_factor=1.0,
                    total_iters=warmup_steps,
                ),
                cosine,
            ],
            milestones=[warmup_steps],
        )
    else:
        scheduler = cosine
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    if accelerator.is_main_process:
        OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
        _write_json(
            output_dir / "provenance.json",
            {
                "schema_version": 1,
                "config_sha256": config_hash,
                "method": "fault_rollout_native_video_action_joint_lora",
                "native_training_loss": "FastWAM.training_loss",
                "action_conditioned_video_dit": False,
                "task": OmegaConf.to_container(cfg.two_stage_opsd.task, resolve=True),
                "fault": validated["fault"],
                "dataset_manifest": str(cfg.two_stage_opsd.stage1.dataset_dir)
                + "/dataset_manifest.json",
                "adapter": adapter_audit,
                "optimizer_steps_per_epoch": steps_per_epoch,
                "optimizer_steps_total": total_steps,
            },
        )

    resume_value = cfg.resume
    checkpoint = None
    if resume_value not in {None, "", "null"}:
        checkpoint = (
            _latest_checkpoint(output_dir)
            if str(resume_value) == "auto"
            else resolve_project_path(str(resume_value))
        )
    start_epoch = 0
    global_step = 0
    if checkpoint is not None:
        start_epoch, global_step = _load_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint=checkpoint,
            config_hash=config_hash,
        )

    metrics_path = output_dir / f"metrics_rank_{accelerator.process_index:02d}.jsonl"
    run_started = time.monotonic()
    for epoch_index in range(start_epoch, int(cfg.num_epochs)):
        running_total = 0.0
        running_video = 0.0
        running_action = 0.0
        micro_count = 0
        optimizer.zero_grad(set_to_none=True)
        for sample in loader:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss, loss_dict = model(sample)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), float(cfg.max_grad_norm)
                    )
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                    global_step += 1
                optimizer.zero_grad(set_to_none=True)
            running_total += float(loss.detach())
            running_video += float(loss_dict["loss_video"])
            running_action += float(loss_dict["loss_action"])
            micro_count += 1
            if accelerator.sync_gradients and global_step % int(cfg.log_every) == 0:
                payload = {
                    "epoch": epoch_index + 1,
                    "global_step": global_step,
                    "loss": running_total / micro_count,
                    "loss_video": running_video / micro_count,
                    "loss_action": running_action / micro_count,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm),
                    "elapsed_seconds": time.monotonic() - run_started,
                }
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(payload) + "\n")
        checkpoint = _save_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            epoch=epoch_index + 1,
            global_step=global_step,
            config_hash=config_hash,
            keep_last=int(section.keep_last_checkpoints),
        )
    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint / "joint_adapter.pt", final_dir / "joint_adapter.pt")
        _write_json(
            final_dir / "completion.json",
            {
                "schema_version": 1,
                "completed_epochs": int(cfg.num_epochs),
                "global_step": global_step,
                "source_checkpoint": checkpoint.relative_to(output_dir).as_posix(),
            },
        )
    accelerator.wait_for_everyone()
    return output_dir / "final" / "joint_adapter.pt"
