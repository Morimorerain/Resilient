"""Cross-severity aggregation, uncertainty estimation, and plotting."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import available_pair_metrics, pair_metric_label


def _finite(values: Sequence[float | None]) -> np.ndarray:
    return np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )


def _summary(values: Sequence[float | None]) -> dict[str, float | int | None]:
    array = _finite(values)
    if array.size == 0:
        return {"count": 0, "mean": None, "variance": None, "std": None, "sem": None}
    variance = float(np.var(array, ddof=1)) if array.size > 1 else 0.0
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "variance": variance,
        "std": float(math.sqrt(variance)),
        "sem": float(math.sqrt(variance / array.size)),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute tie-aware Spearman correlation without adding SciPy."""
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("Spearman inputs must be equal-length vectors with at least two values.")
    ranked_x = _average_ranks(x)
    ranked_y = _average_ranks(y)
    if np.std(ranked_x) == 0.0 or np.std(ranked_y) == 0.0:
        return float("nan")
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute ROC-AUC using average ranks, including score ties."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("ROC labels and scores must be equal-length vectors.")
    positives = y == 1
    negatives = y == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC requires both clean and fault samples.")
    ranks = _average_ranks(s)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _bootstrap_auc(
    clean: np.ndarray,
    fault: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    values = []
    for _ in range(samples):
        sampled_clean = clean[rng.integers(0, clean.size, size=clean.size)]
        sampled_fault = fault[rng.integers(0, fault.size, size=fault.size)]
        labels = np.concatenate(
            [
                np.zeros(sampled_clean.size, dtype=np.int64),
                np.ones(sampled_fault.size, dtype=np.int64),
            ]
        )
        values.append(roc_auc(labels, np.concatenate([sampled_clean, sampled_fault])))
    return values


def _load_run(run: Mapping[str, Any]) -> dict[str, Any]:
    result_path = Path(str(run["result_path"]))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    residual_path = Path(str(run["residual_path"]))
    with np.load(residual_path) as payload:
        prototypes = payload["prototypes"].astype(np.float32, copy=False)
        episode_indices = payload["episode_indices"].astype(np.int64, copy=False)
    if prototypes.ndim != 2 or prototypes.shape[0] != episode_indices.size:
        raise ValueError(f"Malformed residual prototype file: {residual_path}")
    if prototypes.shape[0] == 0:
        raise ValueError(f"No complete latent windows were recorded in {residual_path}.")
    if len(set(episode_indices.tolist())) != episode_indices.size:
        raise ValueError(f"Duplicate episode indices in {residual_path}.")
    by_episode = {
        int(episode["episode_index"]): dict(episode)
        for episode in result["fault_detection"]["episodes"]
    }
    prototype_by_episode = {
        int(index): prototype for index, prototype in zip(episode_indices, prototypes, strict=True)
    }
    return {
        "severity": float(run["severity"]),
        "result": result,
        "episodes": by_episode,
        "prototypes": prototype_by_episode,
    }


def _plot_errorbar(
    path: Path,
    severities: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    ylabel: str,
    error_bar: str,
    severity_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = np.asarray([row["mean"] for row in rows], dtype=np.float64)
    errors = np.asarray([row[error_bar] for row in rows], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.errorbar(severities, means, yerr=errors, marker="o", capsize=4, linewidth=1.5)
    axis.set_xlabel(severity_label)
    axis.set_ylabel(ylabel)
    axis.set_title(f"{ylabel} vs Fault Severity (error: {error_bar.upper()})")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_matrix(
    path: Path, severities: np.ndarray, matrix: np.ndarray, severity_label: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.2, 5.4))
    image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    labels = [f"{value:g}" for value in severities]
    axis.set_xticks(range(len(labels)), labels=labels)
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel(severity_label)
    axis.set_ylabel(severity_label)
    axis.set_title("Residual Cosine Similarity Matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Cosine similarity")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_episode_matrix(
    path: Path, matrix: np.ndarray, episode_indices: Sequence[int], severity: float
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_xlabel("Episode index")
    axis.set_ylabel("Episode index")
    axis.set_title(f"Episode Residual Similarity (severity={severity:g})")
    if len(episode_indices) <= 12:
        labels = [str(index) for index in episode_indices]
        axis.set_xticks(range(len(labels)), labels=labels)
        axis.set_yticks(range(len(labels)), labels=labels)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
    figure.colorbar(image, ax=axis, label="Cosine similarity")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def analyze_severity_study(
    runs: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    metrics: Sequence[str],
    clean_severity: float,
    error_bar: str = "std",
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
    severity_name: str = "severity",
    severity_unit: str = "",
) -> dict[str, Any]:
    """Aggregate episode metrics and write every requested study-level plot."""
    if error_bar not in {"std", "sem"}:
        raise ValueError("error_bar must be either 'std' or 'sem'.")
    loaded = sorted((_load_run(run) for run in runs), key=lambda item: item["severity"])
    severities = np.asarray([item["severity"] for item in loaded], dtype=np.float64)
    if len(set(severities.tolist())) != len(severities):
        raise ValueError("Fault severities must be unique.")
    clean_matches = np.flatnonzero(np.isclose(severities, clean_severity, atol=1e-12, rtol=0.0))
    needs_clean = bool({"fault_score", "roc_auc"}.intersection(metrics))
    if needs_clean and clean_matches.size != 1:
        raise ValueError("FaultScore and ROC-AUC require exactly one clean severity run.")
    reference_index = int(clean_matches[0]) if clean_matches.size else 0
    reference = loaded[reference_index]
    output_dir.mkdir(parents=True, exist_ok=True)
    severity_label = severity_name + (f" ({severity_unit})" if severity_unit else "")

    pair_metric_names = set(available_pair_metrics())
    episode_metric_names = [name for name in metrics if name in pair_metric_names or name == "rtc"]
    episode_values: dict[str, list[list[float | None]]] = {
        name: [
            [episode.get(name) for _, episode in sorted(run["episodes"].items())] for run in loaded
        ]
        for name in episode_metric_names
    }

    common_reference_episodes = set(reference["prototypes"])
    rcs_values: list[list[float]] = []
    for run in loaded:
        indices = sorted(common_reference_episodes.intersection(run["prototypes"]))
        rcs_values.append(
            [_cosine(run["prototypes"][index], reference["prototypes"][index]) for index in indices]
        )
    if "rcs" in metrics:
        episode_values["rcs"] = rcs_values

    if needs_clean:
        clean_rm = _finite(
            [episode.get("rm") for _, episode in sorted(reference["episodes"].items())]
        )
        if clean_rm.size < 2:
            raise ValueError("FaultScore requires at least two valid clean episode RM values.")
        clean_mean = float(np.mean(clean_rm))
        clean_std = float(np.std(clean_rm, ddof=1)) if clean_rm.size > 1 else 0.0
        if clean_std <= 1e-12:
            raise ValueError("Clean RM standard deviation is zero; FaultScore is undefined.")
        fault_scores = [
            [
                None
                if episode.get("rm") is None
                else (float(episode["rm"]) - clean_mean) / clean_std
                for _, episode in sorted(run["episodes"].items())
            ]
            for run in loaded
        ]
        if "fault_score" in metrics:
            episode_values["fault_score"] = fault_scores
    else:
        clean_mean = None
        clean_std = None

    labels = {
        "lpe": "Latent Prediction Error (LPE)",
        "lcd": "Latent Cosine Distance (LCD)",
        "rm": "Residual Magnitude (RM)",
        "rcs": "Residual Cosine Similarity (RCS)",
        "rtc": "Residual Temporal Consistency (RTC)",
        "fault_score": "Fault Detection Score",
    }
    labels.update({name: pair_metric_label(name) for name in pair_metric_names})
    summaries: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for metric, values_by_severity in episode_values.items():
        summaries[metric] = []
        for severity, values in zip(severities, values_by_severity, strict=True):
            summary = {"severity": float(severity), **_summary(values)}
            if summary["count"] == 0:
                raise ValueError(
                    f"Metric {metric} has no complete episodes at severity {severity:g}."
                )
            summaries[metric].append(summary)
            rows.append({"metric": metric, **summary})
        _plot_errorbar(
            output_dir / f"{metric}_severity_errorbar.png",
            severities,
            summaries[metric],
            metric=metric,
            ylabel=labels.get(metric, metric.upper()),
            error_bar=error_bar,
            severity_label=severity_label,
        )

    matrix = np.empty((len(loaded), len(loaded)), dtype=np.float64)
    for left_index, left in enumerate(loaded):
        for right_index, right in enumerate(loaded):
            indices = sorted(set(left["prototypes"]).intersection(right["prototypes"]))
            similarities = [
                _cosine(left["prototypes"][index], right["prototypes"][index]) for index in indices
            ]
            if not similarities:
                raise ValueError(
                    "Residual similarity requires at least one shared episode for "
                    f"severities {left['severity']:g} and {right['severity']:g}."
                )
            matrix[left_index, right_index] = float(np.mean(similarities))
    if "residual_matrix" in metrics:
        np.savetxt(
            output_dir / "residual_similarity_matrix.csv",
            matrix,
            delimiter=",",
            header=",".join(f"{value:g}" for value in severities),
            comments="",
        )
        _plot_matrix(
            output_dir / "residual_similarity_matrix.png", severities, matrix, severity_label
        )
        for run in loaded:
            episode_indices = sorted(run["prototypes"])
            episode_matrix = np.empty(
                (len(episode_indices), len(episode_indices)), dtype=np.float64
            )
            for row, left_episode in enumerate(episode_indices):
                for column, right_episode in enumerate(episode_indices):
                    episode_matrix[row, column] = _cosine(
                        run["prototypes"][left_episode],
                        run["prototypes"][right_episode],
                    )
            severity_slug = (
                "p" if run["severity"] >= 0 else "m"
            ) + f"{abs(run['severity']):g}".replace(".", "p")
            np.savetxt(
                output_dir / f"residual_similarity_episodes_severity-{severity_slug}.csv",
                episode_matrix,
                delimiter=",",
                header=",".join(str(index) for index in episode_indices),
                comments="",
            )
            _plot_episode_matrix(
                output_dir / f"residual_similarity_episodes_severity-{severity_slug}.png",
                episode_matrix,
                episode_indices,
                run["severity"],
            )

    rng = np.random.default_rng(bootstrap_seed)
    study_metrics: dict[str, Any] = {}
    if "spearman_rho" in metrics:
        rm_by_severity = [
            _finite([episode.get("rm") for episode in run["episodes"].values()]) for run in loaded
        ]
        severity_samples = np.concatenate(
            [np.full(values.size, severity) for severity, values in zip(severities, rm_by_severity)]
        )
        rm_samples = np.concatenate(rm_by_severity)
        observed = spearman_correlation(severity_samples, rm_samples)
        if not math.isfinite(observed):
            raise ValueError("Spearman rho is undefined because RM has zero rank variance.")
        bootstrap_values = []
        for _ in range(bootstrap_samples):
            sampled_rm = []
            sampled_severity = []
            for severity, values in zip(severities, rm_by_severity, strict=True):
                sample = values[rng.integers(0, values.size, size=values.size)]
                sampled_rm.append(sample)
                sampled_severity.append(np.full(sample.size, severity))
            bootstrap_values.append(
                spearman_correlation(np.concatenate(sampled_severity), np.concatenate(sampled_rm))
            )
        bootstrap_summary = _summary(bootstrap_values)
        study_metrics["spearman_rho"] = {
            "observed": observed,
            "bootstrap": bootstrap_summary,
        }
        rows.append(
            {
                "metric": "spearman_rho",
                "severity": "all",
                "count": bootstrap_summary["count"],
                "mean": observed,
                "variance": bootstrap_summary["variance"],
                "std": bootstrap_summary["std"],
                "sem": bootstrap_summary["sem"],
            }
        )
        _plot_errorbar(
            output_dir / "spearman_rho_errorbar.png",
            np.asarray([0.0]),
            [{"mean": observed, error_bar: bootstrap_summary[error_bar]}],
            metric="spearman_rho",
            ylabel="Spearman rho (severity vs RM)",
            error_bar=error_bar,
            severity_label="Study",
        )

    if "roc_auc" in metrics:
        clean_rm = _finite([episode.get("rm") for episode in reference["episodes"].values()])
        auc_rows = []
        for severity, run in zip(severities, loaded, strict=True):
            fault_rm = _finite([episode.get("rm") for episode in run["episodes"].values()])
            if np.isclose(severity, clean_severity, atol=1e-12, rtol=0.0):
                observed_auc = 0.5
                bootstrap_values = [0.5] * bootstrap_samples
            else:
                observed_auc = roc_auc(
                    np.concatenate(
                        [
                            np.zeros(clean_rm.size, dtype=np.int64),
                            np.ones(fault_rm.size, dtype=np.int64),
                        ]
                    ),
                    np.concatenate([clean_rm, fault_rm]),
                )
                bootstrap_values = _bootstrap_auc(
                    clean_rm, fault_rm, samples=bootstrap_samples, rng=rng
                )
            auc_rows.append(
                {
                    "severity": float(severity),
                    "observed": observed_auc,
                    **_summary(bootstrap_values),
                }
            )
            rows.append(
                {
                    "metric": "roc_auc",
                    "severity": float(severity),
                    "count": len(bootstrap_values),
                    "mean": observed_auc,
                    "variance": auc_rows[-1]["variance"],
                    "std": auc_rows[-1]["std"],
                    "sem": auc_rows[-1]["sem"],
                }
            )
        study_metrics["roc_auc"] = auc_rows
        _plot_errorbar(
            output_dir / "roc_auc_severity_errorbar.png",
            severities,
            [{"mean": row["observed"], "std": row["std"], "sem": row["sem"]} for row in auc_rows],
            metric="roc_auc",
            ylabel="ROC-AUC (clean vs severity)",
            error_bar=error_bar,
            severity_label=severity_label,
        )

    summary_payload = {
        "schema_version": 1,
        "severity": {"name": severity_name, "unit": severity_unit, "values": severities.tolist()},
        "error_bar": error_bar,
        "episode_metrics": summaries,
        "study_metrics": study_metrics,
        "clean_reference": {
            "severity": clean_severity,
            "rm_mean": clean_mean,
            "rm_std": clean_std,
        },
        "residual_similarity_matrix": matrix.tolist(),
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary_payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["metric", "severity", "count", "mean", "variance", "std", "sem"]
        )
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "# Fault severity metrics",
        "",
        f"Error bars: `{error_bar}`. Variance: episode sample variance (ddof=1).",
        "",
        "| Metric | Severity | Count | Mean | Variance | Std | SEM |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def display(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.8g}"

    for row in rows:
        severity_display = (
            f"{row['severity']:g}"
            if isinstance(row["severity"], int | float)
            else str(row["severity"])
        )
        markdown.append(
            f"| {row['metric']} | {severity_display} | {row['count']} | "
            f"{display(row['mean'])} | {display(row['variance'])} | "
            f"{display(row['std'])} | {display(row['sem'])} |"
        )
    if "spearman_rho" in study_metrics:
        markdown.extend(
            [
                "",
                "Spearman rho (severity vs RM): "
                f"**{study_metrics['spearman_rho']['observed']:.6f}**.",
            ]
        )
    (output_dir / "metrics_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary_payload
