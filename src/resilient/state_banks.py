"""Reproducible LIBERO simulator-state banks and leakage checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCHEMA_VERSION = 1


def state_fingerprint(state: Any) -> str:
    """Hash a simulator state in a dtype-independent canonical representation."""
    array = np.ascontiguousarray(np.asarray(state, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported state-bank manifest schema: {path}")
    return manifest


def manifest_state_hashes(manifest: dict[str, Any]) -> set[str]:
    return {
        str(record["state_sha256"])
        for record in manifest.get("states", [])
    }


def manifest_generation_seeds(manifest: dict[str, Any]) -> set[int]:
    return {
        int(record["generation_seed"])
        for record in manifest.get("states", [])
        if record.get("generation_seed") is not None
    }


def assert_disjoint_manifests(
    validation_manifest: dict[str, Any], training_manifest: dict[str, Any]
) -> None:
    """Reject validation states or generation seeds exposed during training."""
    state_overlap = manifest_state_hashes(validation_manifest) & manifest_state_hashes(
        training_manifest
    )
    seed_overlap = manifest_generation_seeds(validation_manifest) & manifest_generation_seeds(
        training_manifest
    )
    if state_overlap:
        raise ValueError(f"State leakage detected: {len(state_overlap)} fingerprints overlap.")
    if seed_overlap:
        raise ValueError(f"Seed leakage detected: {len(seed_overlap)} generation seeds overlap.")


def load_task_states(
    manifest_path: Path,
    *,
    suite: str,
    task_id: int,
    expected_count: int,
) -> list[torch.Tensor]:
    """Load one task and verify every stored state against its manifest fingerprint."""
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    records = [
        record
        for record in manifest.get("states", [])
        if record["suite"] == suite and int(record["task_id"]) == int(task_id)
    ]
    if len(records) != expected_count:
        raise ValueError(
            f"State bank has {len(records)} states for {suite} task {task_id}; "
            f"expected {expected_count}."
        )
    relative_path = records[0]["file"]
    if any(record["file"] != relative_path for record in records):
        raise ValueError("A task's state-bank records must reference one tensor file.")
    state_path = (manifest_path.parent / relative_path).resolve()
    states = torch.load(state_path, map_location="cpu", weights_only=False)
    if len(states) != len(records):
        raise ValueError(f"State tensor count does not match manifest: {state_path}")
    ordered = sorted(records, key=lambda record: int(record["state_index"]))
    for index, (record, state) in enumerate(zip(ordered, states, strict=True)):
        if int(record["state_index"]) != index:
            raise ValueError(f"Non-contiguous state indices in {state_path}")
        if state_fingerprint(state) != record["state_sha256"]:
            raise ValueError(f"State fingerprint mismatch in {state_path} at index {index}.")
    return list(states)
