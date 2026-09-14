"""Memory-bounded FPO updates and complete recovery checkpoints."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from resilient.opsd.adapters import adapter_state_dict, load_adapter_state_dict

from .advantage import leave_one_out_advantages
from .objective import compute_fpo_ratio, fpo_clipped_objective
from .types import OutcomeGroup


def _as_float(value: torch.Tensor | float) -> float:
    return float(value.detach().float().item()) if isinstance(value, torch.Tensor) else float(value)


class OutcomeFPOTrainer:
    """Apply FPO to final Fast-WAM chunks without differentiating through sampling."""

    def __init__(
        self,
        *,
        scoring_model,
        optimizer: torch.optim.Optimizer,
        accelerator,
        update_epochs: int,
        clipping_epsilon: float,
        log_ratio_clip: float | None,
        advantage_clip: float | None,
        advantage_eps: float,
        minimum_reward_std: float,
        max_grad_norm: float,
    ) -> None:
        if update_epochs <= 0:
            raise ValueError("FPO update_epochs must be positive.")
        self.scoring_model = scoring_model
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.update_epochs = int(update_epochs)
        self.clipping_epsilon = float(clipping_epsilon)
        self.log_ratio_clip = log_ratio_clip
        self.advantage_clip = advantage_clip
        self.advantage_eps = float(advantage_eps)
        self.minimum_reward_std = float(minimum_reward_std)
        self.max_grad_norm = float(max_grad_norm)
        self.global_step = 0
        self.epoch = 0
        self.group_index = 0

    @property
    def fastwam(self):
        return self.accelerator.unwrap_model(self.scoring_model).fastwam

    def train_groups(self, groups: list[OutcomeGroup]) -> dict[str, float]:
        """Reuse one on-policy collection for a configured number of FPO epochs."""
        if not groups:
            raise ValueError("At least one outcome group is required.")
        advantages: list[torch.Tensor] = []
        active_groups = 0
        for group in groups:
            group_advantage, active = leave_one_out_advantages(
                group.rewards,
                eps=self.advantage_eps,
                clip=self.advantage_clip,
                minimum_std=self.minimum_reward_std,
            )
            advantages.append(group_advantage)
            active_groups += int(active)

        candidate_count = sum(len(group.candidates) for group in groups)
        metric_sums = {
            "policy_loss": 0.0,
            "ratio_mean": 0.0,
            "ratio_min": float("inf"),
            "ratio_max": float("-inf"),
            "clip_fraction": 0.0,
            "current_cfm_loss": 0.0,
        }
        self.scoring_model.train()
        for _ in range(self.update_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            for group, group_advantage in zip(groups, advantages, strict=True):
                for candidate, advantage in zip(
                    group.candidates, group_advantage, strict=True
                ):
                    with self.accelerator.autocast():
                        current_cfm_loss = self.scoring_model(
                            group.conditioning,
                            candidate.normalized_action,
                            candidate.mc_noise,
                            candidate.mc_timestep,
                        )
                        ratio, _ = compute_fpo_ratio(
                            candidate.old_cfm_loss.to(current_cfm_loss.device),
                            current_cfm_loss,
                            log_ratio_clip=self.log_ratio_clip,
                        )
                        loss, metrics = fpo_clipped_objective(
                            ratio,
                            advantage.to(ratio.device),
                            clipping_epsilon=self.clipping_epsilon,
                        )
                        scaled_loss = loss / float(candidate_count)
                    self.accelerator.backward(scaled_loss)
                    metric_sums["policy_loss"] += _as_float(metrics["policy_loss"])
                    metric_sums["ratio_mean"] += _as_float(metrics["ratio_mean"])
                    metric_sums["ratio_min"] = min(
                        metric_sums["ratio_min"], _as_float(metrics["ratio_min"])
                    )
                    metric_sums["ratio_max"] = max(
                        metric_sums["ratio_max"], _as_float(metrics["ratio_max"])
                    )
                    metric_sums["clip_fraction"] += _as_float(metrics["clip_fraction"])
                    metric_sums["current_cfm_loss"] += _as_float(
                        current_cfm_loss.detach().float().mean()
                    )

            grad_norm = self.accelerator.clip_grad_norm_(
                self.scoring_model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        denominator = float(candidate_count * self.update_epochs)
        rewards = torch.cat([group.rewards for group in groups])
        self.group_index += len(groups)
        return {
            "policy_loss": metric_sums["policy_loss"] / denominator,
            "current_cfm_loss": metric_sums["current_cfm_loss"] / denominator,
            "ratio_mean": metric_sums["ratio_mean"] / denominator,
            "ratio_min": metric_sums["ratio_min"],
            "ratio_max": metric_sums["ratio_max"],
            "clip_fraction": metric_sums["clip_fraction"] / denominator,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std(unbiased=False)),
            "active_group_fraction": active_groups / float(len(groups)),
            "grad_norm": _as_float(grad_norm),
        }

    def save_checkpoint(
        self,
        output_dir: Path,
        *,
        config_hash: str,
        fault_state: dict[str, Any],
        epoch_number: int | None = None,
    ) -> Path:
        """Save a rolling state or a permanent epoch-boundary state."""
        if epoch_number is None:
            state_dir = output_dir / "checkpoints" / "state" / f"step_{self.global_step:08d}"
            checkpoint_boundary = "completed_on_policy_collection"
        else:
            if epoch_number <= 0:
                raise ValueError("Epoch checkpoint number must be positive.")
            state_dir = (
                output_dir
                / "checkpoints"
                / "epochs"
                / f"epoch_{epoch_number:04d}_step_{self.global_step:08d}"
            )
            checkpoint_boundary = "completed_epoch"
        if self.accelerator.is_main_process:
            state_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(str(state_dir / "accelerate"))
        if self.accelerator.is_main_process:
            torch.save(adapter_state_dict(self.fastwam), state_dir / "recovery_adapter.pt")
            payload = {
                "schema_version": 1,
                "config_sha256": config_hash,
                "global_step": self.global_step,
                "epoch": self.epoch,
                "group_index": self.group_index,
                "fault_state": fault_state,
                "checkpoint_boundary": checkpoint_boundary,
                "schedule_epoch": epoch_number,
            }
            (state_dir / "trainer_state.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
        self.accelerator.wait_for_everyone()
        return state_dir

    def load_checkpoint(self, state_dir: Path, *, config_hash: str) -> None:
        """Restore optimizer/RNG state and strictly verify the resolved config."""
        payload = json.loads((state_dir / "trainer_state.json").read_text(encoding="utf-8"))
        if payload.get("config_sha256") != config_hash:
            raise ValueError("Outcome-FPO checkpoint configuration hash mismatch.")
        self.accelerator.load_state(str(state_dir / "accelerate"))
        adapter = torch.load(
            state_dir / "recovery_adapter.pt", map_location="cpu", weights_only=True
        )
        load_adapter_state_dict(self.fastwam, adapter)
        self.global_step = int(payload["global_step"])
        self.epoch = int(payload["epoch"])
        self.group_index = int(payload["group_index"])

    def prune_checkpoints(
        self,
        output_dir: Path,
        *,
        keep: int,
        keep_epochs: int | None = None,
        preserve_epochs: Sequence[int] = (),
    ) -> None:
        """Prune rolling states and optionally retain selected epoch checkpoints."""
        if keep <= 0:
            raise ValueError("Checkpoint retention must be positive.")
        if keep_epochs is not None and int(keep_epochs) <= 0:
            raise ValueError("Epoch checkpoint retention must be positive when enabled.")
        preserved = {int(epoch) for epoch in preserve_epochs}
        if any(epoch <= 0 for epoch in preserved):
            raise ValueError("Preserved epoch checkpoint numbers must be positive.")
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            root = output_dir / "checkpoints" / "state"
            checkpoints = sorted(root.glob("step_*")) if root.exists() else []
            for path in checkpoints[:-keep]:
                shutil.rmtree(path)
            if keep_epochs is not None:
                epoch_root = output_dir / "checkpoints" / "epochs"
                epoch_checkpoints = (
                    sorted(epoch_root.glob("epoch_*_step_*"))
                    if epoch_root.exists()
                    else []
                )
                newest = set(epoch_checkpoints[-int(keep_epochs) :])
                for path in epoch_checkpoints:
                    epoch_number = int(path.name.split("_", maxsplit=2)[1])
                    if path not in newest and epoch_number not in preserved:
                        shutil.rmtree(path)
        self.accelerator.wait_for_everyone()
