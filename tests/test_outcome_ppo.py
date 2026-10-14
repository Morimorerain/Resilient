"""CPU regression checks for the optional actor-critic FPO branch."""

from __future__ import annotations

import unittest
from contextlib import nullcontext

import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir

from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from resilient.outcome_fpo.ppo.advantage import generalized_advantage_estimate
from resilient.outcome_fpo.ppo.collector import PPOEpisode, PPOTransition
from resilient.outcome_fpo.ppo.model import ConditionValueHead
from resilient.outcome_fpo.ppo.runtime import validate_actor_critic_fpo_config
from resilient.outcome_fpo.ppo.trainer import ActorCriticFPOTrainer
from resilient.outcome_fpo.types import ActionConditioning, CandidateSample
from resilient.outcome_fpo.video_flow import realized_video_flow_loss


class GAEChecks(unittest.TestCase):
    def test_zero_gamma_uses_only_current_chunk_outcome(self) -> None:
        advantages, returns = generalized_advantage_estimate(
            torch.tensor([1.0, -2.0]),
            torch.tensor([0.25, 0.5]),
            torch.tensor([False, True]),
            bootstrap_value=100.0,
            gamma=0.0,
            gae_lambda=0.95,
        )
        self.assertTrue(torch.allclose(returns, torch.tensor([1.0, -2.0])))
        self.assertTrue(torch.allclose(advantages, torch.tensor([0.75, -2.5])))

    def test_terminal_transition_does_not_bootstrap(self) -> None:
        advantages, returns = generalized_advantage_estimate(
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.2, 0.3]),
            torch.tensor([False, True]),
            bootstrap_value=100.0,
            gamma=0.9,
            gae_lambda=1.0,
        )
        self.assertTrue(torch.allclose(returns, torch.tensor([2.8, 2.0])))
        self.assertTrue(torch.allclose(advantages, torch.tensor([2.6, 1.7])))

    def test_truncation_uses_critic_bootstrap(self) -> None:
        _, returns = generalized_advantage_estimate(
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            torch.tensor([False]),
            bootstrap_value=2.0,
            gamma=0.5,
            gae_lambda=0.95,
        )
        self.assertAlmostEqual(float(returns[0]), 2.0)


class ValueHeadChecks(unittest.TestCase):
    def test_current_input_only_and_masked_prompt(self) -> None:
        head = ConditionValueHead(context_dim=4, proprio_dim=2, hidden_dim=8)
        base = ActionConditioning(
            input_image=torch.zeros(1, 3, 16, 16),
            context=torch.tensor([[[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]]]),
            context_mask=torch.tensor([[True, False]]),
            proprio=torch.ones(1, 2),
        )
        shifted = ActionConditioning(
            base.input_image,
            torch.tensor([[[1.0, 2.0, 3.0, 4.0], [100.0] * 4]]),
            base.context_mask,
            base.proprio,
        )
        self.assertTrue(torch.equal(head(base), head(shifted)))
        head(base).sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in head.parameters()))

    def test_value_head_accepts_bfloat16_weights(self) -> None:
        head = ConditionValueHead(context_dim=4, proprio_dim=2, hidden_dim=8).to(
            dtype=torch.bfloat16
        )
        conditioning = ActionConditioning(
            input_image=torch.zeros(1, 3, 16, 16, dtype=torch.bfloat16),
            context=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
            context_mask=torch.ones(1, 2, dtype=torch.bool),
            proprio=torch.zeros(1, 2, dtype=torch.bfloat16),
        )
        self.assertEqual(head(conditioning).dtype, torch.bfloat16)


class ConfigChecks(unittest.TestCase):
    def test_existing_group_relative_mode_is_unchanged(self) -> None:
        from pathlib import Path

        config_root = Path(__file__).resolve().parents[1] / "configs"
        with initialize_config_dir(version_base="1.3", config_dir=str(config_root)):
            original = compose(
                config_name="two_stage_opsd/fastwam_libero10_task7_e5_direct_stage2_task_direction_10ep"
            )
            ppo = compose(
                config_name="two_stage_opsd/fastwam_libero10_task7_e5_direct_stage2_task_direction_ppo_10ep"
            )
        self.assertEqual(
            original.outcome_fpo.get("algorithm", "group_relative_fpo"),
            "group_relative_fpo",
        )
        self.assertEqual(ppo.outcome_fpo.algorithm, "actor_critic_fpo")
        self.assertIsNone(ppo.adapter.initial_checkpoint)
        self.assertEqual(ppo.outcome_fpo.rollout.replan_steps, 32)
        provenance = validate_actor_critic_fpo_config(ppo)
        self.assertEqual(provenance["maximum_chunks_per_epoch"], 768)
        self.assertEqual(provenance["validation_states"], 50)

    def test_immediate_ppo_calibration_preserves_comparison_contract(self) -> None:
        from pathlib import Path

        config_root = Path(__file__).resolve().parents[1] / "configs"
        with initialize_config_dir(version_base="1.3", config_dir=str(config_root)):
            cfg = compose(
                config_name="two_stage_opsd/fastwam_libero10_task7_e5_direct_stage2_task_direction_ppo_immediate_10ep"
            )
        self.assertEqual(cfg.outcome_fpo.ppo.gamma, 0.0)
        self.assertEqual(cfg.two_stage_opsd.distributed.num_processes, 4)
        self.assertIsNone(cfg.adapter.initial_checkpoint)
        self.assertEqual(validate_actor_critic_fpo_config(cfg)["validation_states"], 50)

    def test_video_auxiliary_is_opt_in(self) -> None:
        from pathlib import Path

        config_root = Path(__file__).resolve().parents[1] / "configs"
        with initialize_config_dir(version_base="1.3", config_dir=str(config_root)):
            baseline = compose(
                config_name="two_stage_opsd/fastwam_libero10_task7_e5_direct_stage2_task_direction_ppo_immediate_10ep"
            )
            auxiliary = compose(
                config_name="two_stage_opsd/fastwam_libero10_task7_e5_direct_stage2_task_direction_ppo_immediate_video_10ep"
            )
        self.assertFalse(bool(baseline.outcome_fpo.ppo.get("video_aux", {}).get("enabled", False)))
        self.assertTrue(auxiliary.outcome_fpo.ppo.video_aux.enabled)
        self.assertEqual(auxiliary.outcome_fpo.ppo.video_aux.weight, 0.001)
        self.assertEqual(validate_actor_critic_fpo_config(auxiliary)["video_aux"]["weight"], 0.001)


class VideoFlowChecks(unittest.TestCase):
    def test_checkpointed_and_plain_video_loss_match_and_train_video_only(self) -> None:
        class TinyBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = nn.Parameter(torch.tensor(0.1))

            def forward(self, x, _context, _t_mod, _freqs, **_kwargs):
                return x * (1 + self.scale)

        class TinyExpert(nn.Module):
            fuse_vae_embedding_in_latents = True
            action_conditioned = False
            video_attention_mask_mode = "bidirectional"

            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([TinyBlock()])

            def prepare(self, *, x, timestep, context, context_mask, **_kwargs):
                return x, timestep, timestep, context, context_mask, None, 3, 2, 2, 4

            def post(self, tokens, _t, _frames, _height, _width):
                return tokens

            def forward(self, *, x, timestep, context, context_mask, **kwargs):
                tokens, t, t_mod, ctx, mask, freqs, frames, height, width, _ = self.prepare(
                    x=x, timestep=timestep, context=context, context_mask=context_mask, **kwargs
                )
                for block in self.blocks:
                    tokens = block(tokens, ctx, t_mod, freqs, context_mask=mask)
                return self.post(tokens, t, frames, height, width)

        class TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.video_expert = TinyExpert()
                self.train_video_scheduler = WanContinuousFlowMatchScheduler(1000, 5.0)
                self.device = torch.device("cpu")
                self.torch_dtype = torch.float32

            @staticmethod
            def _compute_video_loss_per_sample(pred_video, target_video, **_kwargs):
                return (pred_video.float() - target_video.float()).square().mean(dim=(1, 2, 3, 4))

        model = TinyModel()
        conditioning = ActionConditioning(
            torch.zeros(1, 3, 4, 4),
            torch.zeros(1, 2, 4),
            torch.ones(1, 2, dtype=torch.bool),
        )
        latent = torch.randn(1, 2, 3, 2, 2)
        plain = realized_video_flow_loss(
            model, conditioning, latent, seed=42, checkpoint_blocks=False
        )
        checked = realized_video_flow_loss(
            model, conditioning, latent, seed=42, checkpoint_blocks=True
        )
        self.assertTrue(torch.allclose(plain, checked))
        checked.backward()
        self.assertIsNotNone(model.video_expert.blocks[0].scale.grad)
        self.assertNotEqual(float(model.video_expert.blocks[0].scale.grad), 0.0)
        self.assertIsNone(latent.grad)


class TrainerChecks(unittest.TestCase):
    def test_actor_and_critic_update_on_one_terminal_episode(self) -> None:
        class TinyScorer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.actor = nn.Parameter(torch.zeros(()))
                self.critic = nn.Parameter(torch.zeros(()))

            def forward(self, _condition, _action, noise, _timestep):
                return self.actor.expand(noise.shape[0]), self.critic.reshape(1)

        class CPUAccelerator:
            @staticmethod
            def autocast():
                return nullcontext()

            @staticmethod
            def backward(loss):
                loss.backward()

            @staticmethod
            def clip_grad_norm_(parameters, max_norm):
                return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm))

        scorer = TinyScorer()
        trainer = ActorCriticFPOTrainer(
            scoring_model=scorer,
            optimizer=torch.optim.SGD(scorer.parameters(), lr=0.01),
            accelerator=CPUAccelerator(),
            update_epochs=2,
            clipping_epsilon=0.05,
            log_ratio_clip=3.0,
            advantage_clip=5.0,
            advantage_eps=1e-6,
            max_grad_norm=1.0,
            max_chunks_per_episode=16,
            minibatch_size=4,
            gamma=0.99,
            gae_lambda=0.95,
            reward_center=0.14,
            reward_scale=0.04,
            value_coefficient=0.25,
            value_clip_range=0.5,
        )
        candidate = CandidateSample(
            normalized_action=torch.zeros(32, 7),
            reward=0.18,
            mc_noise=torch.zeros(8, 32, 7),
            mc_timestep=torch.zeros(8),
            old_cfm_loss=torch.zeros(8),
            inference_seed=1,
            metadata={},
        )
        conditioning = ActionConditioning(
            torch.zeros(1, 3, 16, 16),
            torch.zeros(1, 2, 4),
            torch.ones(1, 2, dtype=torch.bool),
        )
        episode = PPOEpisode(
            transitions=(PPOTransition(conditioning, candidate, 0.0, 0.18, True),),
            bootstrap_value=0.0,
            metadata={},
        )
        result = trainer.train_episode(episode)
        self.assertEqual(result["chunks"], 1)
        self.assertEqual(result["optimizer_steps"], 8)
        self.assertLess(float(scorer.actor), 0.0)
        self.assertGreater(float(scorer.critic), 0.0)

    def test_video_auxiliary_updates_only_when_enabled(self) -> None:
        class TinyScorer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.actor = nn.Parameter(torch.zeros(()))
                self.critic = nn.Parameter(torch.zeros(()))
                self.video = nn.Parameter(torch.zeros(()))

            def forward(
                self,
                _condition,
                _action=None,
                noise=None,
                _timestep=None,
                *,
                objective="actor_critic",
                **_kwargs,
            ):
                if objective == "video_flow":
                    return (self.video - 1).square()
                return self.actor.expand(noise.shape[0]), self.critic.reshape(1)

        class CPUAccelerator:
            @staticmethod
            def autocast():
                return nullcontext()

            @staticmethod
            def backward(loss):
                loss.backward()

            @staticmethod
            def clip_grad_norm_(parameters, max_norm):
                return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm))

        scorer = TinyScorer()
        trainer = ActorCriticFPOTrainer(
            scoring_model=scorer,
            optimizer=torch.optim.SGD(scorer.parameters(), lr=0.01),
            accelerator=CPUAccelerator(),
            update_epochs=2,
            clipping_epsilon=0.05,
            log_ratio_clip=3.0,
            advantage_clip=5.0,
            advantage_eps=1e-6,
            max_grad_norm=1.0,
            max_chunks_per_episode=16,
            minibatch_size=4,
            gamma=0.0,
            gae_lambda=0.95,
            reward_center=0.14,
            reward_scale=0.04,
            value_coefficient=0.25,
            value_clip_range=0.5,
            video_aux_weight=0.02,
            video_aux_update_epochs=1,
        )
        candidate = CandidateSample(
            normalized_action=torch.zeros(32, 7),
            reward=0.18,
            mc_noise=torch.zeros(8, 32, 7),
            mc_timestep=torch.zeros(8),
            old_cfm_loss=torch.zeros(8),
            inference_seed=1,
            metadata={},
        )
        conditioning = ActionConditioning(
            torch.zeros(1, 3, 16, 16),
            torch.zeros(1, 2, 4),
            torch.ones(1, 2, dtype=torch.bool),
        )
        episode = PPOEpisode(
            transitions=(
                PPOTransition(
                    conditioning,
                    candidate,
                    0.0,
                    0.18,
                    True,
                    torch.zeros(1, 2, 3, 2, 2),
                    42,
                ),
            ),
            bootstrap_value=0.0,
            metadata={},
        )
        result = trainer.train_episode(episode)
        self.assertGreater(float(scorer.video), 0.0)
        self.assertGreater(result["video_loss"], 0.0)
        self.assertEqual(result["video_aux_weight"], 0.02)


if __name__ == "__main__":
    unittest.main()
