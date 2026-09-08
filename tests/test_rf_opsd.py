"""CPU tests for RF-OPSD future collection, losses, and world adapters."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from resilient.rf_opsd.adapters import (
    FastWAMWorldLoraConfig,
    discover_fastwam_world_lora_targets,
    inject_fastwam_world_lora,
    world_adapter_state_dict,
)
from resilient.rf_opsd.future import collect_action_chunk_future, stack_observation_video
from resilient.rf_opsd.losses import future_flow_loss


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_expert = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4))
        self.action_expert = nn.Linear(4, 2)
        self.proprio_encoder = nn.Linear(2, 4)
        self.vae = nn.Linear(4, 4)


class _CounterEnvironment:
    def __init__(self) -> None:
        self.value = 0

    def step(self, action):
        self.value += int(action)
        return {"value": self.value}, 0.0, self.value >= 5, {}


class RFOPSDTests(unittest.TestCase):
    def test_world_adapter_targets_only_video_expert(self) -> None:
        model = _ToyModel()
        config = FastWAMWorldLoraConfig(rank=2, alpha=4)
        targets, ranks = discover_fastwam_world_lora_targets(model, config)
        self.assertTrue(targets)
        self.assertTrue(all(name.startswith("video_expert.") for name in targets))
        self.assertTrue(all(value <= 2 for value in ranks.values()))
        audit = inject_fastwam_world_lora(model, config)
        self.assertGreater(audit["trainable_parameters"], 0)
        self.assertTrue(
            all(
                name.startswith("video_expert.") and ".lora_" in name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
        )
        self.assertTrue(world_adapter_state_dict(model))

    def test_future_loss_excludes_initial_latent_and_detaches_teacher(self) -> None:
        student = torch.zeros((1, 1, 3, 1, 1), requires_grad=True)
        teacher = torch.tensor([[[[[100.0]], [[1.0]], [[2.0]]]]], requires_grad=True)
        loss, metrics = future_flow_loss(student, teacher, cosine_weight=0.0)
        self.assertAlmostEqual(float(loss), 2.5)
        self.assertAlmostEqual(float(metrics["mse"]), 2.5)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertEqual(float(student.grad[:, :, :1].abs().sum()), 0.0)
        self.assertIsNone(teacher.grad)

    def test_cosine_distance_has_positive_loss_sign(self) -> None:
        teacher = torch.tensor([[[[[0.0]], [[1.0]]], [[[0.0]], [[0.0]]]]])
        aligned = teacher.clone().requires_grad_(True)
        opposed = (-teacher).clone().requires_grad_(True)
        aligned_loss, _ = future_flow_loss(
            aligned, teacher, mse_weight=0.0, cosine_weight=1.0
        )
        opposed_loss, _ = future_flow_loss(
            opposed, teacher, mse_weight=0.0, cosine_weight=1.0
        )
        self.assertLess(float(aligned_loss), float(opposed_loss))

    def test_collects_exact_frame_boundaries_even_after_success(self) -> None:
        env = _CounterEnvironment()
        rollout = collect_action_chunk_future(
            env,
            {"value": 0},
            [1] * 8,
            frame_steps=[0, 2, 4, 6, 8],
        )
        self.assertEqual([item["value"] for item in rollout.observations], [0, 2, 4, 6, 8])
        self.assertTrue(rollout.success)
        video = stack_observation_video(
            rollout.observations,
            lambda item: torch.full((1, 3, 2, 2), float(item["value"])),
        )
        self.assertEqual(tuple(video.shape), (1, 3, 5, 2, 2))


if __name__ == "__main__":
    unittest.main()
