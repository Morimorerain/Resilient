"""Teacher input that replaces only faulted images with clean images."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..types import ActionConditioning
from .base import TeacherInputContext, TeacherInputProvider


class CleanVisualTeacherInput(TeacherInputProvider):
    """Preserve prompt/proprio while encoding the privileged clean observation."""

    def __init__(self, image_encoder: Callable[[object], torch.Tensor]):
        self.image_encoder = image_encoder

    def build(self, context: TeacherInputContext) -> ActionConditioning:
        student = context.student_conditioning
        return ActionConditioning(
            input_image=self.image_encoder(context.paired_observation.privileged),
            prompt=student.prompt,
            proprio=student.proprio,
            context=student.context,
            context_mask=student.context_mask,
        )
