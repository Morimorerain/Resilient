"""Video-only LoRA adapters for RF-OPSD world learning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from resilient.opsd.adapters import (
    adapter_state_dict,
    load_adapter_state_dict,
    lora_disabled,
)


@dataclass(frozen=True)
class FastWAMWorldLoraConfig:
    """LoRA settings restricted to the Fast-WAM Video DiT expert."""

    rank: int = 64
    alpha: int = 128
    dropout: float = 0.0
    cap_rank_to_dimension: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> FastWAMWorldLoraConfig:
        return cls(
            rank=int(config.get("rank", 64)),
            alpha=int(config.get("alpha", 128)),
            dropout=float(config.get("dropout", 0.0)),
            cap_rank_to_dimension=bool(config.get("cap_rank_to_dimension", True)),
        )

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("World LoRA rank and alpha must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("World LoRA dropout must be in [0, 1).")


def discover_fastwam_world_lora_targets(
    model: nn.Module,
    config: FastWAMWorldLoraConfig,
) -> tuple[list[str], dict[str, int]]:
    """Find every Linear layer owned by the Video DiT, excluding all action modules."""
    if not isinstance(getattr(model, "video_expert", None), nn.Module):
        raise ValueError("Fast-WAM model is missing video_expert.")
    targets: list[str] = []
    rank_pattern: dict[str, int] = {}
    for name, module in model.named_modules():
        if not name.startswith("video_expert.") or not isinstance(module, nn.Linear):
            continue
        targets.append(name)
        if config.cap_rank_to_dimension:
            rank_pattern[name] = min(config.rank, module.in_features, module.out_features)
    if not targets:
        raise RuntimeError("World LoRA target discovery found no Video DiT Linear layers.")
    return sorted(targets), rank_pattern


def inject_fastwam_world_lora(
    model: nn.Module,
    config: FastWAMWorldLoraConfig,
) -> dict[str, Any]:
    """Freeze Fast-WAM and inject LoRA only into the world/video expert."""
    from peft import LoraConfig, inject_adapter_in_model

    targets, rank_pattern = discover_fastwam_world_lora_targets(model, config)
    model.requires_grad_(False)
    peft_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=targets,
        rank_pattern=rank_pattern,
        bias="none",
    )
    inject_adapter_in_model(peft_config, model)
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    unexpected = [
        name
        for name, _ in trainable
        if ".lora_" not in name or not name.startswith("video_expert.")
    ]
    if unexpected:
        raise RuntimeError(f"Non-world LoRA parameters remained trainable: {unexpected[:10]}")
    return {
        "config": asdict(config),
        "target_modules": targets,
        "target_count": len(targets),
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def world_adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return the strict video-only adapter state."""
    state = adapter_state_dict(model)
    unexpected = [name for name in state if not name.startswith("video_expert.")]
    if unexpected:
        raise RuntimeError(f"RF-OPSD checkpoint contains non-video adapter keys: {unexpected[:10]}")
    return state


def load_fastwam_world_lora(
    model: nn.Module,
    *,
    config: FastWAMWorldLoraConfig,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Inject the configured world adapter and strictly load a checkpoint."""
    audit = inject_fastwam_world_lora(model, config)
    state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("RF-OPSD world adapter checkpoint must be a mapping.")
    load_adapter_state_dict(model, state)
    return audit


__all__ = [
    "FastWAMWorldLoraConfig",
    "discover_fastwam_world_lora_targets",
    "inject_fastwam_world_lora",
    "load_fastwam_world_lora",
    "lora_disabled",
    "world_adapter_state_dict",
]
