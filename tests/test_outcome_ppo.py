"""CPU regression checks for the optional actor-critic FPO branch."""

from __future__ import annotations

import unittest

import torch
from hydra import compose, initialize_config_dir

from resilient.outcome_fpo.ppo.advantage import generalized_advantage_estimate
from resilient.outcome_fpo.ppo.model import ConditionValueHead
from resilient.outcome_fpo.ppo.runtime import validate_actor_critic_fpo_config
from resilient.outcome_fpo.types import ActionConditioning


class GAEChecks(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
