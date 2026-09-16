"""Prepare machine-local RoboTwin simulator assets."""

from __future__ import annotations

from pathlib import Path


def materialize_embodiment_configs(robotwin_root: Path) -> list[Path]:
    """Render RoboTwin's ``*_tmp.yml`` planner templates for this checkout.

    The official asset archive intentionally ships CuRobo configurations as
    templates because CuRobo requires absolute URDF and collision-sphere paths.
    Rendered files live beside ignored simulator assets and must be regenerated
    after moving the repository to another machine.
    """
    robotwin_root = robotwin_root.resolve()
    embodiment_root = robotwin_root / "assets" / "embodiments"
    if not embodiment_root.is_dir():
        raise FileNotFoundError(f"RoboTwin embodiment assets not found: {embodiment_root}")

    templates = sorted(embodiment_root.rglob("*_tmp.yml"))
    if not templates:
        raise FileNotFoundError(f"No RoboTwin planner templates found under {embodiment_root}")

    rendered_paths: list[Path] = []
    replacement = str(robotwin_root)
    for template in templates:
        target = template.with_name(template.name.replace("_tmp.yml", ".yml"))
        content = template.read_text(encoding="utf-8")
        rendered = content.replace("${ASSETS_PATH}", replacement).replace(
            "$ASSETS_PATH", replacement
        )
        if "${ASSETS_PATH}" in rendered or "$ASSETS_PATH" in rendered:
            raise ValueError(f"Unresolved ASSETS_PATH placeholder in {template}")
        target.write_text(rendered, encoding="utf-8")
        rendered_paths.append(target)
    return rendered_paths


def validate_materialized_planner_config(path: Path, robotwin_root: Path) -> str | None:
    """Return an error for a missing, unresolved, or checkout-stale config."""
    if not path.is_file():
        return "missing generated planner config"
    content = path.read_text(encoding="utf-8")
    if "${ASSETS_PATH}" in content or "$ASSETS_PATH" in content:
        return "contains unresolved ASSETS_PATH placeholder"
    if str(robotwin_root.resolve()) not in content:
        return "was generated for a different RoboTwin checkout"
    return None
