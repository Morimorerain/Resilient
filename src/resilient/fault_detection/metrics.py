"""Online metrics between predicted and observed Fast-WAM video latents."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

PairMetric = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PairMetricDefinition:
    """One scalar metric evaluated on an aligned latent pair."""

    name: str
    label: str
    function: PairMetric


_PAIR_METRICS: dict[str, PairMetricDefinition] = {}


def register_pair_metric(name: str, label: str) -> Callable[[PairMetric], PairMetric]:
    """Register a clip-level metric without changing the rollout collector."""

    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Metric name must not be empty.")

    def decorator(function: PairMetric) -> PairMetric:
        if normalized in _PAIR_METRICS:
            raise ValueError(f"Metric already registered: {normalized}")
        _PAIR_METRICS[normalized] = PairMetricDefinition(normalized, label, function)
        return function

    return decorator


def _flatten(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().reshape(-1)


@register_pair_metric("lpe", "Latent Prediction Error")
def latent_prediction_error(
    predicted: torch.Tensor, real: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """Mean squared latent prediction error."""
    del predicted, real
    return residual.float().square().mean()


@register_pair_metric("lcd", "Latent Cosine Distance")
def latent_cosine_distance(
    predicted: torch.Tensor, real: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """One minus cosine similarity between flattened latent tensors."""
    del residual
    similarity = F.cosine_similarity(_flatten(predicted), _flatten(real), dim=0, eps=1e-12)
    return 1.0 - similarity


@register_pair_metric("rm", "Residual Magnitude")
def residual_magnitude(
    predicted: torch.Tensor, real: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """Mean absolute residual magnitude."""
    del predicted, real
    return residual.float().abs().mean()


def available_pair_metrics() -> tuple[str, ...]:
    """Return registered clip-level metric names in stable order."""
    return tuple(_PAIR_METRICS)


def pair_metric_label(name: str) -> str:
    """Return the display label associated with a registered pair metric."""
    normalized = name.strip().lower()
    if normalized not in _PAIR_METRICS:
        raise KeyError(f"Unknown pair metric: {normalized}")
    return _PAIR_METRICS[normalized].label


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(_flatten(left), _flatten(right), dim=0, eps=1e-12).item())


class EpisodeLatentAccumulator:
    """Aggregate aligned clips without retaining every high-dimensional residual."""

    def __init__(self, pair_metrics: Iterable[str] = ("lpe", "lcd", "rm")) -> None:
        names = tuple(dict.fromkeys(name.strip().lower() for name in pair_metrics))
        unknown = sorted(set(names).difference(_PAIR_METRICS))
        if unknown:
            raise ValueError(
                f"Unknown pair metrics {unknown}; available metrics: {available_pair_metrics()}."
            )
        self.metric_names = names
        self._values = {name: [] for name in names}
        self._residual_sum: torch.Tensor | None = None
        self._previous_residual: torch.Tensor | None = None
        self._temporal_cosines: list[float] = []
        self._clip_count = 0

    def update(self, predicted: torch.Tensor, real: torch.Tensor) -> None:
        """Consume one shape-aligned pair after excluding the conditioning latent."""
        if predicted.shape != real.shape:
            raise ValueError(
                f"Predicted and real latent shapes differ: {predicted.shape} versus {real.shape}."
            )
        if predicted.numel() == 0:
            raise ValueError("Latent pair must contain at least one predicted element.")
        predicted_f32 = predicted.detach().float()
        real_f32 = real.detach().float()
        residual = real_f32 - predicted_f32
        for name in self.metric_names:
            value = _PAIR_METRICS[name].function(predicted_f32, real_f32, residual)
            self._values[name].append(float(value.item()))

        residual_cpu = residual.reshape(-1).cpu()
        if self._residual_sum is None:
            self._residual_sum = residual_cpu.clone()
        else:
            if residual_cpu.shape != self._residual_sum.shape:
                raise ValueError("Residual shape changed within one episode.")
            self._residual_sum.add_(residual_cpu)
        if self._previous_residual is not None:
            self._temporal_cosines.append(_cosine(self._previous_residual, residual_cpu))
        self._previous_residual = residual_cpu
        self._clip_count += 1

    def finalize(self) -> tuple[dict[str, float | int | None], np.ndarray | None]:
        """Return episode means, RTC, and the mean residual vector."""
        metrics: dict[str, float | int | None] = {"num_aligned_clips": self._clip_count}
        for name, values in self._values.items():
            metrics[name] = float(np.mean(values)) if values else None
        metrics["rtc"] = float(np.mean(self._temporal_cosines)) if self._temporal_cosines else None
        prototype = None
        if self._residual_sum is not None and self._clip_count > 0:
            prototype = self._residual_sum.div(float(self._clip_count)).numpy()
        return metrics, prototype
