"""CPU regression checks for the optional actor-critic FPO branch."""

from __future__ import annotations

import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.nn as nn
from accelerate.utils import DistributedType
from hydra import compose, initialize_config_dir

from resilient.outcome_fpo.ppo.advantage import generalized_advantage_estimate
from resilient.outcome_fpo.ppo.collector import PPOEpisode, PPOTransition
from resilient.outcome_fpo.ppo.gradient_control import clip_actor_critic_gradients
from resilient.outcome_fpo.ppo.model import ConditionValueHead
from resilient.outcome_fpo.ppo.runtime import validate_actor_critic_fpo_config
from resilient.outcome_fpo.ppo.trainer import ActorCriticFPOTrainer
from resilient.outcome_fpo.types import ActionConditioning, CandidateSample


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
        self.assertEqual(cfg.outcome_fpo.ppo.gradient_control, "separate")
        self.assertEqual(cfg.two_stage_opsd.distributed.num_processes, 4)
        self.assertIsNone(cfg.adapter.initial_checkpoint)
        self.assertEqual(validate_actor_critic_fpo_config(cfg)["validation_states"], 50)


class GradientControlChecks(unittest.TestCase):
    def test_non_deepspeed_groups_are_clipped_independently(self) -> None:
        actor = nn.Parameter(torch.zeros(2))
        critic = nn.Parameter(torch.zeros(2))
        actor.grad = torch.tensor([3.0, 4.0])
        critic.grad = torch.tensor([0.0, 6.0])
        optimizer = torch.optim.SGD([{"params": [actor]}, {"params": [critic]}], lr=0.1)
        metrics = clip_actor_critic_gradients(
            accelerator=SimpleNamespace(distributed_type=DistributedType.NO),
            optimizer=optimizer,
            actor_max_norm=1.0,
            critic_max_norm=2.0,
        )
        self.assertAlmostEqual(metrics["actor_grad_norm"], 5.0)
        self.assertAlmostEqual(metrics["critic_grad_norm"], 6.0)
        self.assertAlmostEqual(float(actor.grad.norm()), 1.0, places=5)
        self.assertAlmostEqual(float(critic.grad.norm()), 2.0, places=5)

    def test_zero2_partitioned_groups_are_clipped_independently(self) -> None:
        actor_gradient = torch.tensor([3.0, 4.0])
        critic_gradient = torch.tensor([0.0, 6.0])

        class FakeZero2:
            partition_gradients = True
            cpu_offload = False
            loss_scale = 1.0
            bit16_groups = [[], []]
            optimizer = SimpleNamespace(param_groups=[{}, {}])
            averaged_gradients = {0: [actor_gradient], 1: [critic_gradient, torch.zeros(2)]}
            params_in_partition = {0: [object()], 1: [object()]}
            partition_size = {0: 2, 1: 4}

            @staticmethod
            def get_grad_norm_direct(gradients, parameters):
                return torch.linalg.vector_norm(
                    torch.cat([gradient for gradient, _ in zip(gradients, parameters)])
                )

        engine = SimpleNamespace(optimizer=FakeZero2(), gradient_clipping=lambda: 0.0)
        accelerator = SimpleNamespace(
            distributed_type=DistributedType.DEEPSPEED,
            deepspeed_engine_wrapped=SimpleNamespace(engine=engine),
        )
        metrics = clip_actor_critic_gradients(
            accelerator=accelerator,
            optimizer=None,
            actor_max_norm=1.0,
            critic_max_norm=2.0,
        )
        self.assertEqual(metrics["actor_grad_clipped"], 1.0)
        self.assertEqual(metrics["critic_grad_clipped"], 1.0)
        self.assertAlmostEqual(float(actor_gradient.norm()), 1.0, places=5)
        self.assertAlmostEqual(float(critic_gradient.norm()), 2.0, places=5)


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


if __name__ == "__main__":
    unittest.main()
