"""Regression checks for repository-level runtime hygiene."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_hydra_job_logging_is_stdout_only() -> None:
    """Keep Hydra's duplicate file handler out of the repository root."""
    config_text = (REPOSITORY_ROOT / "configs" / "train.yaml").read_text(
        encoding="utf-8"
    )

    assert "override hydra/job_logging: stdout" in config_text
