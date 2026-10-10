"""Regression checks for repository-level runtime hygiene."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_hydra_job_logging_is_stdout_only() -> None:
    """Keep Hydra's duplicate file handler out of the repository root."""
    config_text = (REPOSITORY_ROOT / "configs" / "train.yaml").read_text(
        encoding="utf-8"
    )

    assert "override hydra/job_logging: stdout" in config_text


def test_task_direction_launcher_can_select_direct_stage_two() -> None:
    """Keep the two-stage default while exposing the direct FPO entrypoint."""
    script = (REPOSITORY_ROOT / "scripts/resilient/run_task_direction_outcome_fpo.sh").read_text(
        encoding="utf-8"
    )
    assert 'TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-train_two_stage_stage2.py}"' in script
    assert '"scripts/resilient/${TRAIN_ENTRYPOINT}"' in script


def test_e5_direct_stage_two_configs_do_not_load_stage_one() -> None:
    """The two reward arms must independently start from the released base."""
    for reward in ("full", "task_direction"):
        config = (
            REPOSITORY_ROOT
            / "configs/two_stage_opsd"
            / f"fastwam_libero10_task7_e5_direct_stage2_{reward}_10ep.yaml"
        ).read_text(encoding="utf-8")
        assert "initial_adapter: null" in config
        assert "initial_checkpoint: null" in config
        assert "num_epochs: 10" in config
