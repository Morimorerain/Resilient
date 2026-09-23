"""CPU tests for outcome-guided Flow Policy Optimization."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from resilient.opsd.model_adapter import FastWAMActionFlowScorer
from resilient.opsd.types import ActionConditioning as OPSDActionConditioning
from resilient.opsd.types import StudentDenoisingStep
from resilient.outcome_fpo.advantage import leave_one_out_advantages
from resilient.outcome_fpo.collector import (
    collect_action_chunk_future,
    validate_frame_steps,
)
from resilient.outcome_fpo.objective import compute_fpo_ratio, fpo_clipped_objective
from resilient.outcome_fpo.outcome import OutcomeTarget, outcome_cosine_reward
from resilient.outcome_fpo.policy import sample_action_chunk, sample_cfm_pairs
from resilient.outcome_fpo.runtime import (
    _assert_resume_config_compatible,
    _resolve_resume,
    resolved_config_hash,
)
from resilient.outcome_fpo.trainer import OutcomeFPOTrainer
from resilient.outcome_fpo.types import ActionConditioning, CandidateSample, OutcomeGroup
from resilient.task_decoupling.reward import (
    group_ranking_metrics,
    task_outcome_rewards,
    within_group_motion_partial_correlation,
)


class FPOObjectiveTests(unittest.TestCase):
    def test_lower_current_cfm_loss_increases_ratio(self) -> None:
        old = torch.tensor([[1.0, 1.0]])
        current = torch.tensor([[0.5, 0.5]])
        ratio, log_ratio = compute_fpo_ratio(old, current)
        self.assertAlmostEqual(float(log_ratio), 0.5)
        self.assertGreater(float(ratio), 1.0)

    def test_positive_advantage_gradient_reduces_current_loss(self) -> None:
        current = torch.tensor([[1.0, 1.0]], requires_grad=True)
        ratio, _ = compute_fpo_ratio(torch.ones_like(current), current)
        loss, _ = fpo_clipped_objective(
            ratio,
            torch.tensor([1.0]),
            clipping_epsilon=0.05,
        )
        loss.backward()
        self.assertTrue(torch.all(current.grad > 0))

    def test_ratio_averages_mc_losses_before_exponentiation(self) -> None:
        old = torch.tensor([[0.0, 2.0]])
        current = torch.tensor([[0.0, 0.0]])
        ratio, _ = compute_fpo_ratio(old, current)
        self.assertAlmostEqual(float(ratio), torch.exp(torch.tensor(1.0)).item())

    def test_leave_one_out_advantage_is_zero_centered_and_skips_ties(self) -> None:
        advantage, active = leave_one_out_advantages(torch.tensor([1.0, 2.0, 4.0, 5.0]))
        self.assertTrue(active)
        self.assertAlmostEqual(float(advantage.mean()), 0.0, places=6)
        tied, tied_active = leave_one_out_advantages(torch.ones(4))
        self.assertFalse(tied_active)
        self.assertTrue(torch.equal(tied, torch.zeros(4)))


class OutcomeRewardTests(unittest.TestCase):
    def test_temporal_delta_reward_ignores_shared_static_content(self) -> None:
        target_hidden = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
        shifted = target_hidden + 10.0
        target = OutcomeTarget(
            hidden=target_hidden,
            noise=torch.empty(1),
            timestep=torch.empty(1),
            tokens_per_frame=1,
        )
        reward = outcome_cosine_reward(shifted, target)
        self.assertTrue(torch.isfinite(reward))
        self.assertGreater(float(reward), 0.99)

    def test_task_reward_decomposition_uses_counterfactual_difference(self) -> None:
        target_hidden = torch.tensor([[[0.0, 0.0], [1.0, -1.0], [2.0, -2.0]]])
        full_hidden = target_hidden.clone()
        counterfactual_hidden = torch.zeros_like(target_hidden)
        target = OutcomeTarget(
            hidden=target_hidden,
            noise=torch.empty(1),
            timestep=torch.empty(1),
            tokens_per_frame=1,
        )
        rewards = task_outcome_rewards(full_hidden, counterfactual_hidden, target)
        self.assertGreater(float(rewards.full), 0.99)
        self.assertGreater(float(rewards.margin), 0.99)
        self.assertGreater(float(rewards.task_direction), 0.99)

    def test_gate4_metrics_are_group_local(self) -> None:
        group = np.asarray([0, 0, 0, 1, 1, 1])
        progress = np.asarray([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
        reward = np.asarray([0.0, 0.8, 2.1, 10.0, 11.2, 11.9])
        metrics = group_ranking_metrics(reward, progress, group)
        self.assertEqual(metrics["pair_count"], 6)
        self.assertAlmostEqual(float(metrics["pairwise_accuracy"]), 1.0)
        motion = np.asarray([2.0, 0.5, 0.2, 2.0, 1.5, 0.0])
        correlation = within_group_motion_partial_correlation(reward, progress, motion, group)
        self.assertTrue(np.isfinite(correlation))

    def test_cfm_pair_sampling_is_reproducible_and_independent(self) -> None:
        action = torch.zeros((32, 7))
        first = sample_cfm_pairs(action, num_samples=8, num_train_timesteps=1000, seed=12)
        second = sample_cfm_pairs(action, num_samples=8, num_train_timesteps=1000, seed=12)
        third = sample_cfm_pairs(action, num_samples=8, num_train_timesteps=1000, seed=13)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertFalse(torch.equal(first[0], third[0]))
        self.assertTrue(torch.all((first[1] >= 0) & (first[1] < 1000)))


class _CounterfactualEnvironment:
    def __init__(self) -> None:
        self.step_index = 0

    def step(self, action):
        self.step_index += 1
        return {"step": self.step_index, "action": action}, 0.0, self.step_index == 2, {}


class CollectionTests(unittest.TestCase):
    def test_future_collection_keeps_alignment_after_success(self) -> None:
        frame_steps = validate_frame_steps([0, 2, 4], 4)
        rollout = collect_action_chunk_future(
            _CounterfactualEnvironment(),
            {"step": 0},
            [0, 1, 2, 3],
            frame_steps=frame_steps,
        )
        self.assertEqual([item["step"] for item in rollout.observations], [0, 2, 4])
        self.assertTrue(rollout.success)

    def test_types_reject_misaligned_monte_carlo_tensors(self) -> None:
        with self.assertRaisesRegex(ValueError, "MC noise"):
            CandidateSample(
                normalized_action=torch.zeros(32, 7),
                reward=0.0,
                mc_noise=torch.zeros(8, 31, 7),
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
        with self.assertRaisesRegex(ValueError, "at least two"):
            OutcomeGroup(conditioning, (), {})

    def test_native_sampler_does_not_request_a_denoising_trace(self) -> None:
        source = inspect.getsource(sample_action_chunk)
        self.assertIn("return_denoising_trace=False", source)
        self.assertNotIn("sde", source.lower())


class CheckpointTests(unittest.TestCase):
    def test_retention_keeps_preserved_and_latest_epoch(self) -> None:
        class _Accelerator:
            is_main_process = True

            @staticmethod
            def wait_for_everyone() -> None:
                return None

        trainer = OutcomeFPOTrainer.__new__(OutcomeFPOTrainer)
        trainer.accelerator = _Accelerator()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            state_root = output_dir / "checkpoints" / "state"
            epoch_root = output_dir / "checkpoints" / "epochs"
            for step in (100, 200, 300, 400):
                (state_root / f"step_{step:08d}").mkdir(parents=True)
            for epoch_index in range(1, 5):
                (epoch_root / f"epoch_{epoch_index:04d}_step_{epoch_index * 50:08d}").mkdir(
                    parents=True
                )

            trainer.prune_checkpoints(
                output_dir,
                keep=3,
                keep_epochs=1,
                preserve_epochs=[1],
            )

            self.assertEqual(
                sorted(path.name for path in state_root.iterdir()),
                ["step_00000200", "step_00000300", "step_00000400"],
            )
            self.assertEqual(
                sorted(path.name for path in epoch_root.iterdir()),
                ["epoch_0001_step_00000050", "epoch_0004_step_00000200"],
            )

    def test_retention_keeps_all_adapters_and_only_latest_full_state(self) -> None:
        class _Accelerator:
            is_main_process = True

            @staticmethod
            def wait_for_everyone() -> None:
                return None

        trainer = OutcomeFPOTrainer.__new__(OutcomeFPOTrainer)
        trainer.accelerator = _Accelerator()
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            epoch_root = output_dir / "checkpoints" / "epochs"
            for epoch_index in range(1, 5):
                checkpoint = epoch_root / f"epoch_{epoch_index:04d}_step_{epoch_index * 50:08d}"
                (checkpoint / "accelerate").mkdir(parents=True)
                (checkpoint / "accelerate/model.pt").write_bytes(b"state")
                (checkpoint / "recovery_adapter.pt").write_bytes(b"adapter")
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps({"epoch": epoch_index}), encoding="utf-8"
                )

            trainer.prune_checkpoints(
                output_dir,
                keep=1,
                keep_epochs=None,
                keep_epoch_full_states=1,
            )

            checkpoints = sorted(epoch_root.iterdir())
            self.assertEqual(len(checkpoints), 4)
            self.assertTrue(all((path / "recovery_adapter.pt").is_file() for path in checkpoints))
            self.assertEqual(
                [path.name for path in checkpoints if (path / "accelerate").is_dir()],
                ["epoch_0004_step_00000200"],
            )
            for path in checkpoints[:-1]:
                marker = json.loads((path / "resume_state_pruned.json").read_text(encoding="utf-8"))
                self.assertFalse(marker["strict_resume_available"])

    def test_explicit_resume_rejects_adapter_only_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "epoch_0001_step_00000050"
            checkpoint.mkdir()
            with self.assertRaisesRegex(ValueError, "no strict-resume state"):
                _resolve_resume(Path(directory), str(checkpoint))

    def test_extended_horizon_is_resume_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            checkpoint = output_dir / "checkpoints/epochs/epoch_0001_step_00000050"
            checkpoint.mkdir(parents=True)
            previous = OmegaConf.create(
                {
                    "output_dir": str(output_dir),
                    "resume": None,
                    "learning_rate": 1.0e-6,
                    "max_checkpoints": 3,
                    "outcome_fpo": {"validate_only": False, "num_epochs": 1},
                }
            )
            OmegaConf.save(previous, output_dir / "resolved_config.yaml", resolve=True)
            stored_hash = resolved_config_hash(previous)
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"config_sha256": stored_hash, "epoch": 1, "global_step": 50}),
                encoding="utf-8",
            )
            current = OmegaConf.create(
                {
                    "output_dir": str(output_dir),
                    "resume": "auto",
                    "learning_rate": 1.0e-6,
                    "max_checkpoints": 3,
                    "checkpoint_retention": {
                        "rolling_keep_last": 3,
                        "epoch_keep_last": 1,
                        "preserve_epochs": [1],
                    },
                    "outcome_fpo": {"validate_only": False, "num_epochs": 7},
                }
            )

            checkpoint_hash, config_path = _assert_resume_config_compatible(
                current,
                checkpoint_dir=checkpoint,
                output_dir=output_dir,
            )

            self.assertEqual(checkpoint_hash, stored_hash)
            self.assertEqual(config_path, output_dir / "resolved_config.yaml")
            current.learning_rate = 2.0e-6
            with self.assertRaisesRegex(ValueError, "algorithmic setting"):
                _assert_resume_config_compatible(
                    current,
                    checkpoint_dir=checkpoint,
                    output_dir=output_dir,
                )

    def test_auto_resume_considers_epoch_and_rolling_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            rolling = output_dir / "checkpoints" / "state" / "step_00000400"
            epoch = output_dir / "checkpoints" / "epochs" / "epoch_0001_step_00000500"
            for path, group_index, epoch_index, step in (
                (rolling, 1600, 0, 400),
                (epoch, 2000, 1, 500),
            ):
                path.mkdir(parents=True)
                (path / "accelerate").mkdir()
                (path / "trainer_state.json").write_text(
                    json.dumps(
                        {
                            "group_index": group_index,
                            "epoch": epoch_index,
                            "global_step": step,
                        }
                    ),
                    encoding="utf-8",
                )

            self.assertEqual(_resolve_resume(output_dir, "auto"), epoch)

            (epoch / "accelerate").rmdir()
            self.assertEqual(_resolve_resume(output_dir, "auto"), rolling)


class _ToyVideoExpert(nn.Module):
    fuse_vae_embedding_in_latents = False

    def prepare(self, *, x, context, context_mask, **_kwargs):
        batch = x.shape[0]
        tokens = torch.zeros(batch, 2, 4)
        t_mod = torch.zeros(batch, 1, 4)
        freqs = torch.zeros(2, 1)
        return (
            tokens,
            None,
            t_mod,
            context,
            context_mask,
            freqs,
            1,
            1,
            1,
            2,
        )


class _ToyMoT:
    @staticmethod
    def prefill_video_cache_tensor(**kwargs):
        batch = kwargs["video_tokens"].shape[0]
        return [torch.zeros(batch, 1)], [torch.zeros(batch, 1)]


class _ToyScoringFastWAM:
    device = torch.device("cpu")
    torch_dtype = torch.float32

    def __init__(self) -> None:
        self.video_expert = _ToyVideoExpert()
        self.mot = _ToyMoT()
        self.encoded_batch_sizes: list[int] = []

    def _encode_input_image_latents_tensor(self, image):
        self.encoded_batch_sizes.append(int(image.shape[0]))
        return torch.zeros(1, 1, 1, 1, 1)

    @staticmethod
    def _build_mot_attention_mask(*, video_seq_len, action_seq_len, **_kwargs):
        return torch.zeros(video_seq_len + action_seq_len, video_seq_len + action_seq_len)

    @staticmethod
    def _denoise_action_with_video_cache(*, latents_action, **_kwargs):
        return latents_action * 2.0


class BatchedFlowScorerTests(unittest.TestCase):
    def test_mc_batch_encodes_one_repeated_image_then_expands_latent(self) -> None:
        model = _ToyScoringFastWAM()
        image = torch.ones(1, 3, 16, 16).expand(8, -1, -1, -1)
        conditioning = OPSDActionConditioning(
            input_image=image,
            context=torch.zeros(8, 2, 4),
            context_mask=torch.ones(8, 2, dtype=torch.bool),
        )
        step = StudentDenoisingStep(
            latent=torch.ones(8, 32, 7),
            timestep=torch.ones(8),
            delta_sigma=torch.tensor(0.0),
        )
        result = FastWAMActionFlowScorer(model).score(conditioning, step)
        self.assertEqual(result.shape, (8, 32, 7))
        self.assertEqual(model.encoded_batch_sizes, [1])

    def test_mc_batch_rejects_different_images(self) -> None:
        model = _ToyScoringFastWAM()
        image = torch.zeros(2, 3, 16, 16)
        image[1] = 1.0
        conditioning = OPSDActionConditioning(
            input_image=image,
            context=torch.zeros(2, 2, 4),
            context_mask=torch.ones(2, 2, dtype=torch.bool),
        )
        step = StudentDenoisingStep(
            latent=torch.ones(2, 32, 7),
            timestep=torch.ones(2),
            delta_sigma=torch.tensor(0.0),
        )
        with self.assertRaisesRegex(ValueError, "repeated copies"):
            FastWAMActionFlowScorer(model).score(conditioning, step)


if __name__ == "__main__":
    unittest.main()
