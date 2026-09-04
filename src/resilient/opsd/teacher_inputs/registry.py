"""Configuration registry for privileged Teacher inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch

from .base import TeacherInputProvider
from .clean_visual import CleanVisualTeacherInput

TeacherInputFactory = Callable[
    [Mapping[str, object], Callable[[object], torch.Tensor]], TeacherInputProvider
]
_PROVIDERS: dict[str, TeacherInputFactory] = {}


def register_teacher_input_provider(
    provider_type: str,
    factory: TeacherInputFactory,
) -> None:
    """Register a provider without coupling new privileged modalities to Trainer."""
    normalized = provider_type.strip()
    if not normalized:
        raise ValueError("Teacher-input provider type must not be empty.")
    if normalized in _PROVIDERS:
        raise ValueError(f"Teacher-input provider is already registered: {normalized}")
    _PROVIDERS[normalized] = factory


def _register_builtins() -> None:
    if "clean_visual" not in _PROVIDERS:
        register_teacher_input_provider(
            "clean_visual",
            lambda config, image_encoder: CleanVisualTeacherInput(image_encoder=image_encoder),
        )


def build_teacher_input_provider(
    config: Mapping[str, object],
    *,
    image_encoder: Callable[[object], torch.Tensor],
) -> TeacherInputProvider:
    _register_builtins()
    provider_type = str(config.get("type", ""))
    if provider_type not in _PROVIDERS:
        known = ", ".join(sorted(_PROVIDERS))
        raise ValueError(
            f"Unknown Teacher-input provider {provider_type!r}; registered: {known}."
        )
    return _PROVIDERS[provider_type](config, image_encoder)
