"""PPO-style actor-critic updates using the established FPO flow surrogate."""

from __future__ import annotations

from pathlib import Path

import torch

from ..objective import compute_fpo_ratio, fpo_clipped_objective
from ..trainer import OutcomeFPOTrainer
from .advantage import generalized_advantage_estimate
from .collector import PPOEpisode


class ActorCriticFPOTrainer(OutcomeFPOTrainer):
    """Train one on-policy episode before collecting another Student trajectory."""

    def __init__(
        self,
        *,
        max_chunks_per_episode: int,
        minibatch_size: int,
        gamma: float,
        gae_lambda: float,
        reward_center: float,
        reward_scale: float,
        value_coefficient: float,
        value_clip_range: float,
        **kwargs,
    ) -> None:
        super().__init__(minimum_reward_std=0.0, **kwargs)
        self.max_chunks_per_episode = int(max_chunks_per_episode)
        self.minibatch_size = int(minibatch_size)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.reward_center = float(reward_center)
        self.reward_scale = float(reward_scale)
        self.value_coefficient = float(value_coefficient)
        self.value_clip_range = float(value_clip_range)
        if self.max_chunks_per_episode % self.minibatch_size:
            raise ValueError("max_chunks_per_episode must divide by minibatch_size.")
        if self.reward_scale <= 0 or self.value_coefficient < 0 or self.value_clip_range <= 0:
            raise ValueError("Invalid reward scale, value coefficient, or value clip range.")

    def train_episode(self, episode: PPOEpisode) -> dict[str, float]:
        """Pad only optimizer work, never simulator experience, across uneven terminal episodes."""
        steps = episode.transitions
        if not steps or len(steps) > self.max_chunks_per_episode:
            raise ValueError("The on-policy episode has an invalid transition count.")
        rewards = torch.tensor(
            [(item.reward - self.reward_center) / self.reward_scale for item in steps],
            dtype=torch.float32,
        )
        values = torch.tensor([item.old_value for item in steps], dtype=torch.float32)
        dones = torch.tensor([item.done for item in steps], dtype=torch.bool)
        advantages, returns = generalized_advantage_estimate(
            rewards,
            values,
            dones,
            bootstrap_value=episode.bootstrap_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        if self.advantage_clip is not None:
            advantages = advantages.clamp(-self.advantage_clip, self.advantage_clip)
        active_count = len(steps)
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "ratio_mean": 0.0,
            "clip_fraction": 0.0,
            "approx_kl": 0.0,
            "grad_norm": 0.0,
        }
        self.scoring_model.train()
        # Every rank executes the same number of forwards/optimizer steps under DDP.
        # Padded indices repeat a real input but contribute exactly zero loss.
        for _ in range(self.update_epochs):
            order = torch.randperm(active_count).tolist()
            order.extend([order[-1]] * (self.max_chunks_per_episode - active_count))
            for start in range(0, self.max_chunks_per_episode, self.minibatch_size):
                self.optimizer.zero_grad(set_to_none=True)
                for slot in range(start, start + self.minibatch_size):
                    index = order[slot]
                    item = steps[index]
                    candidate = item.candidate
                    active = float(slot < active_count)
                    with self.accelerator.autocast():
                        current_cfm_loss, current_value = self.scoring_model(
                            item.conditioning,
                            candidate.normalized_action,
                            candidate.mc_noise,
                            candidate.mc_timestep,
                        )
                        ratio, log_ratio = compute_fpo_ratio(
                            candidate.old_cfm_loss.to(current_cfm_loss.device),
                            current_cfm_loss,
                            log_ratio_clip=self.log_ratio_clip,
                        )
                        policy_loss, policy_metrics = fpo_clipped_objective(
                            ratio,
                            advantages[index].to(ratio.device),
                            clipping_epsilon=self.clipping_epsilon,
                        )
                        old_value = values[index].to(current_value.device)
                        target = returns[index].to(current_value.device)
                        clipped_value = old_value + (current_value - old_value).clamp(
                            -self.value_clip_range, self.value_clip_range
                        )
                        value_loss = (
                            0.5
                            * torch.maximum(
                                (current_value - target).square(),
                                (clipped_value - target).square(),
                            ).mean()
                        )
                        loss = (
                            active
                            * (policy_loss + self.value_coefficient * value_loss)
                            / float(self.minibatch_size)
                        )
                    self.accelerator.backward(loss)
                    if active:
                        metrics["policy_loss"] += float(policy_loss.detach().item())
                        metrics["value_loss"] += float(value_loss.detach().item())
                        metrics["ratio_mean"] += float(policy_metrics["ratio_mean"].item())
                        metrics["clip_fraction"] += float(policy_metrics["clip_fraction"].item())
                        metrics["approx_kl"] += float(
                            ((ratio.detach() - 1.0) - log_ratio.detach()).mean().item()
                        )
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.scoring_model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                metrics["grad_norm"] += float(grad_norm)
        self.group_index += 1
        denominator = float(active_count * self.update_epochs)
        update_count = self.update_epochs * self.max_chunks_per_episode // self.minibatch_size
        return {
            **{
                name: value / (float(update_count) if name == "grad_norm" else denominator)
                for name, value in metrics.items()
            },
            "reward_mean": float(rewards.mean().item()),
            "reward_raw_mean": sum(item.reward for item in steps) / active_count,
            "reward_raw_std": float(
                torch.tensor([item.reward for item in steps]).std(unbiased=False).item()
            ),
            "advantage_mean": float(advantages.mean().item()),
            "return_mean": float(returns.mean().item()),
            "value_old_mean": float(values.mean().item()),
            "chunks": active_count,
            "optimizer_steps": update_count,
        }

    def save_checkpoint(self, output_dir: Path, **kwargs) -> Path:
        """Store the value head beside the adapter for audit and strict resume."""
        directory = super().save_checkpoint(output_dir, **kwargs)
        if self.accelerator.is_main_process:
            value_head = self.accelerator.unwrap_model(self.scoring_model).value_head
            torch.save(value_head.state_dict(), directory / "value_head.pt")
        self.accelerator.wait_for_everyone()
        return directory
