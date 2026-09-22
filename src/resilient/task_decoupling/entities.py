"""Portable task-to-entity specifications without runtime language models."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from .types import EntitySpec


def load_entity_specification(
    path: str | Path,
    *,
    expected_suite: str | None = None,
    expected_task_id: int | None = None,
) -> tuple[EntitySpec, ...]:
    """Load a fixed entity vocabulary from a YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Task entity configuration does not exist: {config_path}")
    payload = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("entities"), list):
        raise ValueError("Task entity configuration must contain an entities list.")
    if expected_suite is not None and str(payload.get("suite")) != expected_suite:
        raise ValueError("Task entity suite does not match the active rollout suite.")
    if expected_task_id is not None and int(payload.get("task_id", -1)) != expected_task_id:
        raise ValueError("Task entity id does not match the active rollout task.")
    entities = tuple(
        EntitySpec(
            entity_id=str(item["id"]),
            role=str(item["role"]),
            prompt=str(item["prompt"]),
            oracle_instances=tuple(str(value) for value in item.get("oracle_instances", [])),
        )
        for item in payload["entities"]
    )
    if not entities or len({item.entity_id for item in entities}) != len(entities):
        raise ValueError("Entity ids must form a non-empty unique set.")
    return entities
