"""Teacher-input providers kept independent from fault implementations."""

from .base import TeacherInputContext, TeacherInputProvider
from .clean_visual import CleanVisualTeacherInput
from .registry import build_teacher_input_provider, register_teacher_input_provider

__all__ = [
    "CleanVisualTeacherInput",
    "TeacherInputContext",
    "TeacherInputProvider",
    "build_teacher_input_provider",
    "register_teacher_input_provider",
]
