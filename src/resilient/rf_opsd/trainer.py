"""Distributed RF-OPSD world-adapter optimization and resume support."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from resilient.opsd.adapters import load_adapter_state_dict

from .adapters import world_adapter_state_dict
from .types import VideoFlowBatch


def _as_float(value: torch.Tensor | float) -> float:
    return float(value.detach().float().item()) if isinstance(value, torch.Tensor) else float(value)


class RFOPSDTrainer:
    """Optimize one realized-future fragment at a time with bounded memory."""

    def __init__(
        self,
        *,
        model,
        optimizer: torch.optim.Optimizer,
        scheduler,
        accelerator,
        max_grad_norm: float,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.max_grad_norm = float(max_grad_norm)
        self.global_step = 0
        self.epoch = 0
        self.fragment_index = 0

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @property
    def fastwam(self):
        return self.unwrapped_model.fastwam

    def train_fragment(self, batch: VideoFlowBatch) -> dict[str, float]:
        self.model.train()
        with self.accelerator.accumulate(self.model):
            with self.accelerator.autocast():
                loss, metrics = self.model(batch)
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
            else:
                grad_norm = torch.zeros((), device=self.accelerator.device)
        self.fragment_index += 1
        return {
            "loss": _as_float(metrics["loss"]),
            "flow_mse": _as_float(metrics["mse"]),
            "flow_smooth_l1": _as_float(metrics["smooth_l1"]),
            "flow_cosine_distance": _as_float(metrics["cosine_distance"]),
            "grad_norm": _as_float(grad_norm),
        }

    def save_checkpoint(self, output_dir: Path, *, config_hash: str) -> Path:
        state_dir = output_dir / "checkpoints" / "state" / f"step_{self.global_step:08d}"
        if self.accelerator.is_main_process:
            state_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(str(state_dir / "accelerate"))
        if self.accelerator.is_main_process:
            torch.save(
                world_adapter_state_dict(self.fastwam),
                state_dir / "rf_world_adapter.pt",
            )
            payload = {
                "schema_version": 1,
                "config_hash": config_hash,
                "global_step": self.global_step,
                "epoch": self.epoch,
                "fragment_index": self.fragment_index,
            }
            (state_dir / "trainer_state.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        self.accelerator.wait_for_everyone()
        return state_dir

    def load_checkpoint(self, state_dir: Path, *, config_hash: str) -> None:
        payload = json.loads((state_dir / "trainer_state.json").read_text(encoding="utf-8"))
        if payload.get("config_hash") != config_hash:
            raise ValueError("RF-OPSD checkpoint configuration hash does not match this run.")
        adapter = torch.load(
            state_dir / "rf_world_adapter.pt", map_location="cpu", weights_only=True
        )
        load_adapter_state_dict(self.fastwam, adapter)
        self.accelerator.load_state(str(state_dir / "accelerate"))
        self.global_step = int(payload["global_step"])
        self.epoch = int(payload["epoch"])
        self.fragment_index = int(payload["fragment_index"])

    def prune_checkpoints(self, output_dir: Path, *, keep: int) -> None:
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            checkpoints = sorted((output_dir / "checkpoints" / "state").glob("step_*"))
            for stale in checkpoints[:-int(keep)]:
                shutil.rmtree(stale)
        self.accelerator.wait_for_everyone()
