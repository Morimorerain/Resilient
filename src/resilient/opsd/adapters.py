"""Fast-WAM-aligned LoRA injection and fixed-base Teacher contexts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FastWAMLoraConfig:
    """LoRA settings covering the same high-level scope as Fast-WAM training."""

    rank: int = 64
    alpha: int = 128
    dropout: float = 0.0
    cap_rank_to_dimension: bool = True
    require_proprio_encoder: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> FastWAMLoraConfig:
        return cls(
            rank=int(config.get("rank", 64)),
            alpha=int(config.get("alpha", 128)),
            dropout=float(config.get("dropout", 0.0)),
            cap_rank_to_dimension=bool(config.get("cap_rank_to_dimension", True)),
            require_proprio_encoder=bool(config.get("require_proprio_encoder", True)),
        )

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1).")


def discover_fastwam_lora_targets(
    model: nn.Module,
    config: FastWAMLoraConfig,
) -> tuple[list[str], dict[str, int]]:
    """Select every Linear layer in Fast-WAM's original trainable module scope."""
    required_roots = ("video_expert", "action_expert")
    for root in required_roots:
        if not isinstance(getattr(model, root, None), nn.Module):
            raise ValueError(f"Fast-WAM model is missing required module: {root}")
    if config.require_proprio_encoder and not isinstance(
        getattr(model, "proprio_encoder", None), nn.Module
    ):
        raise ValueError("Fast-WAM model is missing the required proprio_encoder.")

    roots = (*required_roots, "proprio_encoder")
    targets: list[str] = []
    rank_pattern: dict[str, int] = {}
    root_counts = {root: 0 for root in roots}
    for name, module in model.named_modules():
        root = name.split(".", 1)[0]
        if root not in roots or not isinstance(module, nn.Linear):
            continue
        targets.append(name)
        root_counts[root] += 1
        if config.cap_rank_to_dimension:
            rank_pattern[name] = min(config.rank, module.in_features, module.out_features)
    if root_counts["video_expert"] == 0 or root_counts["action_expert"] == 0:
        raise RuntimeError(f"LoRA target discovery missed a Fast-WAM expert: {root_counts}")
    if config.require_proprio_encoder and root_counts["proprio_encoder"] == 0:
        raise RuntimeError("LoRA target discovery missed proprio_encoder.")
    return sorted(targets), rank_pattern


def inject_fastwam_aligned_lora(
    model: nn.Module,
    config: FastWAMLoraConfig,
) -> dict[str, Any]:
    """Freeze base weights and inject reversible LoRA into the aligned scope."""
    from peft import LoraConfig, inject_adapter_in_model

    targets, rank_pattern = discover_fastwam_lora_targets(model, config)
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
    unexpected = [name for name, _ in trainable if ".lora_" not in name]
    if unexpected:
        raise RuntimeError(f"Non-LoRA parameters remained trainable: {unexpected[:10]}")
    counts = {
        "video_expert": sum("video_expert" in name for name, _ in trainable),
        "action_expert": sum("action_expert" in name for name, _ in trainable),
        "proprio_encoder": sum("proprio_encoder" in name for name, _ in trainable),
    }
    if counts["video_expert"] == 0 or counts["action_expert"] == 0:
        raise RuntimeError(f"Injected LoRA does not cover both experts: {counts}")
    trainable_parameters = sum(parameter.numel() for _, parameter in trainable)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "config": asdict(config),
        "target_modules": targets,
        "target_count": len(targets),
        "trainable_tensor_counts": counts,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / max(total_parameters, 1),
    }


@contextmanager
def lora_disabled(model: nn.Module) -> Iterator[None]:
    """Disable every injected adapter so the fixed Teacher equals the base checkpoint."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    layers = [module for module in model.modules() if isinstance(module, BaseTunerLayer)]
    previous = [bool(layer.disable_adapters) for layer in layers]
    for layer in layers:
        layer.enable_adapters(False)
    try:
        yield
    finally:
        for layer, was_disabled in zip(layers, previous, strict=True):
            layer.enable_adapters(not was_disabled)


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return unique LoRA parameters without shared-module alias duplication."""
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters(remove_duplicate=True)
        if ".lora_" in name
    }


def load_adapter_state_dict(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Restore a strict LoRA-only checkpoint into an already injected model."""
    expected = set(adapter_state_dict(model))
    provided = set(state)
    if expected != provided:
        raise ValueError(
            "Adapter checkpoint keys do not match the injected model: "
            f"missing={sorted(expected - provided)[:10]}, "
            f"unexpected={sorted(provided - expected)[:10]}."
        )
    model.load_state_dict(dict(state), strict=False)
