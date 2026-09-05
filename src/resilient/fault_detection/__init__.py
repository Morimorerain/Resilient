"""Extensible latent-space metrics for fault-severity studies."""

from .metrics import EpisodeLatentAccumulator, available_pair_metrics, pair_metric_label
from .study import analyze_severity_study

__all__ = [
    "EpisodeLatentAccumulator",
    "analyze_severity_study",
    "available_pair_metrics",
    "pair_metric_label",
]
