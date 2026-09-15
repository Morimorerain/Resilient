"""Pinned FastWAM-on-RoboTwin paper protocol and task-universe checks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PaperProtocol:
    """Published FastWAM RoboTwin evaluation constants."""

    task_count: int = 50
    episodes_per_condition_per_task: int = 100
    clean_success_percent: float = 91.88
    randomized_success_percent: float = 91.78
    displayed_average_percent: float = 91.8
    instruction_type: str = "unseen"
    action_horizon: int = 32
    replan_steps: int = 24
    num_inference_steps: int = 10
    sigma_shift: float = 5.0
    text_cfg_scale: float = 1.0

    @property
    def total_episodes(self) -> int:
        """Return the two-condition paper evaluation size."""
        return self.task_count * self.episodes_per_condition_per_task * 2


@dataclass(frozen=True)
class PaperTaskReference:
    """One published per-task result in percentage points."""

    task_name: str
    display_name: str
    clean_percent: float
    randomized_percent: float


PAPER_PROTOCOL = PaperProtocol()


def load_task_limits(path: Path) -> dict[str, int]:
    """Load and validate RoboTwin's ordered task-to-step-limit mapping."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Expected a non-empty task mapping in {path}")
    result: dict[str, int] = {}
    for raw_name, raw_limit in payload.items():
        name = str(raw_name)
        limit = int(raw_limit)
        if not name or limit <= 0:
            raise ValueError(f"Invalid RoboTwin task entry: {raw_name!r}={raw_limit!r}")
        if name in result:
            raise ValueError(f"Duplicate RoboTwin task: {name}")
        result[name] = limit
    return result


def load_paper_references(path: Path) -> list[PaperTaskReference]:
    """Load the committed Appendix Table 3 FastWAM reference values."""
    references: list[PaperTaskReference] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {"task_name", "display_name", "clean_percent", "randomized_percent"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"Unexpected paper-reference columns in {path}: {reader.fieldnames}")
        for row in reader:
            references.append(
                PaperTaskReference(
                    task_name=row["task_name"],
                    display_name=row["display_name"],
                    clean_percent=float(row["clean_percent"]),
                    randomized_percent=float(row["randomized_percent"]),
                )
            )
    names = [item.task_name for item in references]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate task in paper reference: {path}")
    return references


def validate_task_universe(
    task_limits: dict[str, int],
    references: list[PaperTaskReference],
    protocol: PaperProtocol = PAPER_PROTOCOL,
) -> None:
    """Require the simulator task universe to equal the paper's 50 tasks."""
    limit_names = set(task_limits)
    reference_names = {item.task_name for item in references}
    if len(task_limits) != protocol.task_count:
        raise ValueError(
            f"Expected {protocol.task_count} RoboTwin tasks, found {len(task_limits)}"
        )
    if len(references) != protocol.task_count:
        raise ValueError(
            f"Expected {protocol.task_count} paper rows, found {len(references)}"
        )
    if limit_names != reference_names:
        raise ValueError(
            "RoboTwin task list and paper reference differ: "
            f"missing_reference={sorted(limit_names - reference_names)}, "
            f"missing_simulator={sorted(reference_names - limit_names)}"
        )

    clean_mean = sum(item.clean_percent for item in references) / len(references)
    randomized_mean = sum(item.randomized_percent for item in references) / len(references)
    if abs(clean_mean - protocol.clean_success_percent) > 1e-9:
        raise ValueError(f"Paper clean mean mismatch: {clean_mean}")
    if abs(randomized_mean - protocol.randomized_success_percent) > 1e-9:
        raise ValueError(f"Paper randomized mean mismatch: {randomized_mean}")
