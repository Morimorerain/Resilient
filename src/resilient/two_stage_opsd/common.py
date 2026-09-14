"""Shared validation and provenance helpers for two-stage adaptation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from resilient.faults import build_fault_pipeline
from resilient.state_banks import assert_disjoint_manifests, load_manifest


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a user path relative to the current project checkout."""
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a local artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_config_hash(cfg: DictConfig) -> str:
    """Hash algorithmic settings while excluding runtime-only locations."""
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved two-stage configuration must be a mapping.")
    payload["output_dir"] = "<runtime-output-dir>"
    payload["resume"] = None
    section = payload.get("two_stage_opsd")
    if isinstance(section, dict):
        section["mode"] = "<runtime-mode>"
        stage1 = section.get("stage1")
        if isinstance(stage1, dict):
            stage1["dataset_dir"] = "<generated-dataset-dir>"
        stage2 = section.get("stage2")
        if isinstance(stage2, dict):
            stage2["initial_adapter"] = "<stage1-adapter>"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def collection_config_hash(cfg: DictConfig) -> str:
    """Hash only settings that can change the generated Stage-I dataset."""
    section = cfg.two_stage_opsd
    payload = {
        "schema_version": 1,
        "checkpoint": str(cfg.ckpt),
        "seed": int(cfg.seed),
        "fault": OmegaConf.to_container(cfg.fault, resolve=True),
        "task": OmegaConf.to_container(section.task, resolve=True),
        "split": OmegaConf.to_container(section.split, resolve=True),
        "dataset_stats_path": str(section.dataset_stats_path),
        "collection": OmegaConf.to_container(
            section.stage1.collection, resolve=True
        ),
        "data": {
            "num_frames": int(cfg.data.train.num_frames),
            "action_video_freq_ratio": int(cfg.data.train.action_video_freq_ratio),
            "video_size": [int(value) for value in cfg.data.train.video_size],
            "concat_multi_camera": str(cfg.data.train.concat_multi_camera),
            "processor": OmegaConf.to_container(cfg.data.train.processor, resolve=True),
        },
        "model": {
            "action_conditioned": bool(cfg.model.video_dit_config.action_conditioned),
            "action_train_shift": float(cfg.model.action_scheduler.train_shift),
            "video_train_shift": float(cfg.model.video_scheduler.train_shift),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_fault_and_split(cfg: DictConfig) -> dict[str, Any]:
    """Fail closed on the agreed single task, Fault, and disjoint state split."""
    section = cfg.two_stage_opsd
    if str(section.task.suite) != "libero_10" or int(section.task.task_id) != 7:
        raise ValueError("The first two-stage experiment is fixed to libero_10 task 7.")

    fault_cfg = OmegaConf.to_container(cfg.fault, resolve=True)
    fault_metadata = build_fault_pipeline(fault_cfg).metadata()
    faults = fault_metadata["faults"]
    if len(faults) != 1 or faults[0]["family"] != "structure.joint_motion":
        raise ValueError("The first experiment requires one simulator joint-motion Fault.")
    if faults[0]["targets"] != ["robot0_joint1"]:
        raise ValueError("The first experiment must target robot0_joint1.")
    severity = faults[0]["severity"]
    if severity["name"] != "motion_retention" or float(severity["value"]) != 0.5:
        raise ValueError("The first experiment requires joint1 motion retention=0.5.")

    training_path = resolve_project_path(section.split.training_reference_manifest)
    validation_path = resolve_project_path(section.split.validation_manifest)
    if not training_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("The documented train/validation state manifests are required.")
    training = load_manifest(training_path)
    validation = load_manifest(validation_path)
    assert_disjoint_manifests(validation, training)
    suite = str(section.task.suite)
    task_id = int(section.task.task_id)
    training_records = [
        item
        for item in training.get("states", [])
        if str(item["suite"]) == suite and int(item["task_id"]) == task_id
    ]
    validation_records = [
        item
        for item in validation.get("states", [])
        if str(item["suite"]) == suite and int(item["task_id"]) == task_id
    ]
    expected = int(section.split.states_per_split)
    if len(training_records) != expected or len(validation_records) != expected:
        raise ValueError(
            f"Expected {expected} train and validation states for {suite}/task_{task_id:02d}, "
            f"got {len(training_records)} and {len(validation_records)}."
        )
    return {
        "fault": fault_metadata,
        "training_manifest": training_path,
        "validation_manifest": validation_path,
        "training_records": training_records,
        "validation_records": validation_records,
    }
