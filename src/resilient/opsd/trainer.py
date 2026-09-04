"""Memory-bounded Student-trajectory scoring for OPSD-Flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .adapters import adapter_state_dict, load_adapter_state_dict, lora_disabled
from .losses import pointwise_flow_step_loss
from .model_adapter import FastWAMActionFlowScorer
from .types import ActionConditioning, StudentTrajectory


def _scalar_to_float(value: torch.Tensor | float) -> float:
    """Convert scalar metrics returned by native PyTorch or DeepSpeed."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().item())
    return float(value)


class OPSDFlowTrainer:
    """Score a detached Student path while never constructing a Teacher path."""

    def __init__(
        self,
        *,
        model,
        optimizer: torch.optim.Optimizer,
        scheduler,
        accelerator,
        pointwise_clip: float | None,
        max_grad_norm: float,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.pointwise_clip = pointwise_clip
        self.max_grad_norm = float(max_grad_norm)
        self.global_step = 0
        self.epoch = 0
        self.rollout_index = 0

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @property
    def fastwam(self):
        return self.unwrapped_model.fastwam

    def train_trajectory(
        self,
        *,
        student_conditioning: ActionConditioning,
        teacher_conditioning: ActionConditioning,
        trajectory: StudentTrajectory,
        action_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Backpropagate the weighted sum one latent at a time to bound memory."""
        if not trajectory.steps:
            raise ValueError("Student trajectory must contain at least one denoising step.")
        model = self.fastwam
        teacher_scorer = FastWAMActionFlowScorer(model)
        delta = torch.stack(
            [step.delta_sigma.float().abs().reshape(()) for step in trajectory.steps]
        )
        weights = delta / delta.sum().clamp(min=1e-12)
        weighted_loss = 0.0
        weighted_raw = 0.0
        weighted_clip_fraction = 0.0

        model.train()
        with self.accelerator.accumulate(self.model):
            for index, (step, weight) in enumerate(zip(trajectory.steps, weights, strict=True)):
                del index
                with torch.no_grad(), lora_disabled(model):
                    model.eval()
                    teacher_flow = teacher_scorer.score(teacher_conditioning, step)
                model.train()
                with self.accelerator.autocast():
                    student_flow = self.model(student_conditioning, step)
                    step_loss, metrics = pointwise_flow_step_loss(
                        student_flow,
                        teacher_flow,
                        action_valid_mask=action_valid_mask,
                        pointwise_clip=self.pointwise_clip,
                    )
                    loss = weight.to(step_loss.device) * step_loss
                self.accelerator.backward(loss)
                weighted_loss += float(loss.detach().item())
                weighted_raw += float(weight.item()) * float(metrics["raw"].detach().item())
                weighted_clip_fraction += float(weight.item()) * float(
                    metrics["clip_fraction"].detach().item()
                )

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
        self.rollout_index += 1
        return {
            "loss": weighted_loss,
            "loss_raw": weighted_raw,
            "clip_fraction": weighted_clip_fraction,
            "grad_norm": _scalar_to_float(grad_norm),
        }

    def save_checkpoint(
        self,
        output_dir: Path,
        *,
        config_hash: str,
        fault_state: dict[str, Any],
    ) -> Path:
        """Save adapter plus Accelerate state at a completed trajectory boundary."""
        state_dir = output_dir / "checkpoints" / "state" / f"step_{self.global_step:08d}"
        if self.accelerator.is_main_process:
            state_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(str(state_dir / "accelerate"))
        if self.accelerator.is_main_process:
            torch.save(adapter_state_dict(self.fastwam), state_dir / "adapter.pt")
            payload = {
                "schema_version": 1,
                "config_hash": config_hash,
                "global_step": self.global_step,
                "epoch": self.epoch,
                "rollout_index": self.rollout_index,
                "fault_state": fault_state,
            }
            (state_dir / "trainer_state.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        self.accelerator.wait_for_everyone()
        return state_dir

    def load_checkpoint(self, state_dir: Path, *, config_hash: str) -> dict[str, Any]:
        """Restore optimizer/RNG and reject configuration drift."""
        payload = json.loads((state_dir / "trainer_state.json").read_text(encoding="utf-8"))
        if payload.get("config_hash") != config_hash:
            raise ValueError("Checkpoint configuration hash does not match this run.")
        adapter = torch.load(state_dir / "adapter.pt", map_location="cpu", weights_only=True)
        load_adapter_state_dict(self.fastwam, adapter)
        self.accelerator.load_state(str(state_dir / "accelerate"))
        self.global_step = int(payload["global_step"])
        self.epoch = int(payload["epoch"])
        self.rollout_index = int(payload["rollout_index"])
        return dict(payload.get("fault_state", {}))
