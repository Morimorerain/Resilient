"""CPU tests for Fast-WAM-aligned OPSD-Flow components."""

from __future__ import annotations

import inspect
import unittest

import torch
import torch.nn as nn

from fastwam.models.wan22.fastwam import FastWAM
from resilient.faults.paired_observation import PairedObservation
from resilient.opsd.adapters import (
    FastWAMLoraConfig,
    adapter_state_dict,
    discover_fastwam_lora_targets,
    inject_fastwam_aligned_lora,
    load_fastwam_lora_adapter,
    lora_disabled,
)
from resilient.opsd.losses import opsd_flow_loss
from resilient.opsd.teacher_inputs import TeacherInputContext, build_teacher_input_provider
from resilient.opsd.trainer import _scalar_to_float
from resilient.opsd.types import ActionConditioning, StudentTrajectory


class _ToyFastWAM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_expert = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        self.action_expert = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        self.proprio_encoder = nn.Linear(2, 4)
        self.vae = nn.Linear(4, 4)
        self.mot = nn.Module()
        self.mot.video_expert = self.video_expert
        self.mot.action_expert = self.action_expert

    def forward(self, video, action, proprio):
        return self.video_expert(video) + self.proprio_encoder(proprio), self.action_expert(
            action
        )


class OPSDLossTests(unittest.TestCase):
    def test_scalar_metric_accepts_tensor_and_deepspeed_float(self) -> None:
        self.assertEqual(_scalar_to_float(torch.tensor(1.25, requires_grad=True)), 1.25)
        self.assertEqual(_scalar_to_float(2.5), 2.5)

    def test_weighted_student_trajectory_loss_and_teacher_detach(self) -> None:
        student = torch.zeros((1, 2, 1, 1), requires_grad=True)
        teacher = torch.tensor([[[[1.0]], [[2.0]]]], requires_grad=True)
        loss, metrics = opsd_flow_loss(student, teacher, torch.tensor([1.0, 3.0]))
        self.assertAlmostEqual(float(loss), 3.25)
        self.assertAlmostEqual(float(metrics["loss_raw"]), 2.5)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)

    def test_pointwise_clipping_caps_each_coordinate(self) -> None:
        student = torch.zeros((1, 1, 1, 2))
        teacher = torch.tensor([[[[1.0, 2.0]]]])
        loss, metrics = opsd_flow_loss(
            student, teacher, torch.tensor([1.0]), pointwise_clip=0.05
        )
        self.assertAlmostEqual(float(loss), 0.05)
        self.assertAlmostEqual(float(metrics["clip_fraction"]), 1.0)


class FastWAMLoraTests(unittest.TestCase):
    def test_targets_match_fastwam_trainable_scope_and_exclude_vae(self) -> None:
        model = _ToyFastWAM()
        config = FastWAMLoraConfig(rank=3, alpha=6)
        targets, ranks = discover_fastwam_lora_targets(model, config)
        self.assertTrue(any(name.startswith("video_expert") for name in targets))
        self.assertTrue(any(name.startswith("action_expert") for name in targets))
        self.assertTrue(any(name.startswith("proprio_encoder") for name in targets))
        self.assertFalse(any(name.startswith("vae") for name in targets))
        self.assertTrue(all(rank <= 3 for rank in ranks.values()))

    def test_injection_freezes_base_and_teacher_can_disable_lora(self) -> None:
        model = _ToyFastWAM()
        audit = inject_fastwam_aligned_lora(
            model, FastWAMLoraConfig(rank=2, alpha=4)
        )
        self.assertGreater(audit["trainable_parameters"], 0)
        self.assertTrue(
            all(
                ".lora_" in name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
        )
        for name, parameter in model.named_parameters():
            if ".lora_B." in name:
                nn.init.constant_(parameter, 0.25)
        inputs = (torch.ones(1, 4), torch.ones(1, 4), torch.ones(1, 2))
        student_output = model(*inputs)
        with lora_disabled(model):
            teacher_output = model(*inputs)
        self.assertFalse(torch.allclose(student_output[0], teacher_output[0]))
        self.assertFalse(torch.allclose(student_output[1], teacher_output[1]))
        self.assertTrue(
            all(not name.startswith("mot.") for name in adapter_state_dict(model))
        )

    def test_saved_adapter_round_trip(self) -> None:
        import tempfile
        from pathlib import Path

        config = FastWAMLoraConfig(rank=2, alpha=4)
        source = _ToyFastWAM()
        inject_fastwam_aligned_lora(source, config)
        for name, parameter in source.named_parameters():
            if ".lora_" in name:
                nn.init.constant_(parameter, 0.125)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "adapter.pt"
            torch.save(adapter_state_dict(source), checkpoint)
            restored = _ToyFastWAM()
            load_fastwam_lora_adapter(restored, config=config, checkpoint=checkpoint)
            expected = adapter_state_dict(source)
            actual = adapter_state_dict(restored)
            self.assertEqual(set(expected), set(actual))
            self.assertTrue(all(torch.equal(expected[key], actual[key]) for key in expected))


class TeacherAndTraceTests(unittest.TestCase):
    def test_clean_visual_provider_replaces_only_image(self) -> None:
        student_image = torch.zeros(1, 3, 2, 2)
        clean_image = torch.ones(1, 3, 2, 2)
        proprio = torch.tensor([[0.2, 0.4]])
        student = ActionConditioning(
            input_image=student_image, prompt="task", proprio=proprio
        )
        trajectory = StudentTrajectory.from_inference_result(
            {
                "action": torch.zeros(1, 2),
                "denoising_trace": [
                    {
                        "latent": torch.zeros(1, 1, 2),
                        "timestep": torch.tensor([1.0]),
                        "delta_sigma": torch.tensor([-1.0]),
                    }
                ],
            },
            expected_steps=1,
        )
        paired = PairedObservation("fault", "clean", {})
        provider = build_teacher_input_provider(
            {"type": "clean_visual"}, image_encoder=lambda observation: clean_image
        )
        teacher = provider.build(
            TeacherInputContext(paired, student, trajectory, extras={})
        )
        self.assertIs(teacher.input_image, clean_image)
        self.assertEqual(teacher.prompt, student.prompt)
        self.assertIs(teacher.proprio, proprio)
        self.assertIs(teacher.context, student.context)

    def test_fastwam_trace_switch_is_default_off(self) -> None:
        parameter = inspect.signature(FastWAM.infer_action).parameters[
            "return_denoising_trace"
        ]
        self.assertIs(parameter.default, False)


if __name__ == "__main__":
    unittest.main()
