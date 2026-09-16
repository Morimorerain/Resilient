"""Read-only preflight checks for the pinned RoboTwin baseline."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from ctypes.util import find_library
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .assets import validate_materialized_planner_config
from .protocol import load_paper_references, load_task_limits, validate_task_universe


@dataclass(frozen=True)
class CheckResult:
    """One machine-readable preflight result."""

    name: str
    passed: bool
    detail: str


def sha256(path: Path) -> str:
    """Hash one file without retaining it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_entry(manifest: dict[str, Any], relative_path: str) -> dict[str, Any] | None:
    direct = next(
        (item for item in manifest.get("assets", []) if item["path"] == relative_path),
        None,
    )
    if direct is not None:
        return direct
    for archive in manifest.get("robotwin_simulator_assets", {}).get("archives", []):
        if relative_path == f"downloads/robotwin/{archive['filename']}":
            return archive
    return None


def _check_file(
    project_root: Path,
    manifest: dict[str, Any],
    relative_path: str,
    *,
    verify_hashes: bool,
) -> CheckResult:
    path = project_root / relative_path
    entry = _asset_entry(manifest, relative_path)
    if entry is None:
        return CheckResult(relative_path, False, "missing manifest entry")
    if not path.is_file():
        return CheckResult(relative_path, False, "missing file")
    if path.stat().st_size != int(entry["size"]):
        return CheckResult(
            relative_path,
            False,
            f"size {path.stat().st_size} != {entry['size']}",
        )
    if verify_hashes:
        actual = sha256(path)
        if actual != entry["sha256"]:
            return CheckResult(relative_path, False, f"sha256 {actual} != {entry['sha256']}")
    return CheckResult(relative_path, True, "present" if not verify_hashes else "hash verified")


def _check_package(
    distribution_name: str,
    *,
    expected_version: str | None = None,
) -> CheckResult:
    try:
        version = importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return CheckResult(f"package:{distribution_name}", False, "not installed")
    if expected_version is not None and version != expected_version:
        return CheckResult(
            f"package:{distribution_name}",
            False,
            f"version {version} != required {expected_version}",
        )
    return CheckResult(f"package:{distribution_name}", True, version)


def run_preflight(
    project_root: Path,
    *,
    verify_hashes: bool = False,
    require_runtime: bool = True,
) -> dict[str, Any]:
    """Check task protocol, external files, simulator assets, and runtime packages."""
    project_root = project_root.resolve()
    robotwin_root = project_root / "third_party" / "RoboTwin"
    manifest_path = project_root / "manifests" / "assets.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results: list[CheckResult] = []

    try:
        task_limits = load_task_limits(robotwin_root / "task_config" / "_eval_step_limit.yml")
        references = load_paper_references(
            project_root / "reproduce" / "fastwam_robotwin" / "paper_reference.csv"
        )
        validate_task_universe(task_limits, references)
    except Exception as exc:
        results.append(CheckResult("paper_task_universe", False, repr(exc)))
    else:
        results.append(CheckResult("paper_task_universe", True, "50 tasks; published means match"))

    required_files = [
        "checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt",
        "checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json",
        "checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/"
        "models_t5_umt5-xxl-enc-bf16.safetensors",
        "checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/"
        "Wan2.2_VAE.safetensors",
        "checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/special_tokens_map.json",
        "checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/spiece.model",
        "checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json",
        "checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json",
        "downloads/robotwin/background_texture.zip",
        "downloads/robotwin/embodiments.zip",
        "downloads/robotwin/objects.zip",
    ]
    results.extend(
        _check_file(project_root, manifest, path, verify_hashes=verify_hashes)
        for path in required_files
    )

    required_asset_paths = {
        "assets/background_texture": "directory",
        "assets/embodiments/aloha-agilex/config.yml": "file",
        "assets/embodiments/aloha-agilex/curobo_left.yml": "file",
        "assets/embodiments/aloha-agilex/curobo_right.yml": "file",
        "assets/objects/objaverse/list.json": "file",
    }
    for relative_path, path_type in required_asset_paths.items():
        path = robotwin_root / relative_path
        present = path.is_dir() if path_type == "directory" else path.is_file()
        detail = f"{path_type} present" if present else f"missing {path_type}"
        results.append(CheckResult(f"robotwin:{relative_path}", present, detail))

    for filename in ("curobo_left.yml", "curobo_right.yml"):
        path = robotwin_root / "assets" / "embodiments" / "aloha-agilex" / filename
        error = validate_materialized_planner_config(path, robotwin_root)
        results.append(
            CheckResult(
                f"robotwin:materialized:{filename}",
                error is None,
                "generated for current checkout" if error is None else error,
            )
        )

    stats_path = project_root / required_files[1]
    if stats_path.is_file():
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            if not isinstance(stats, dict) or not stats:
                raise ValueError("statistics JSON is empty")
        except Exception as exc:
            results.append(CheckResult("dataset_stats_json", False, repr(exc)))
        else:
            results.append(CheckResult("dataset_stats_json", True, f"{len(stats)} top-level keys"))

    if require_runtime:
        for package in (
            "torch",
            "sapien",
            "mplib",
            "gymnasium",
            "open3d",
            "opencv-python",
            "nvidia-curobo",
        ):
            results.append(_check_package(package))
        results.append(_check_package("warp-lang", expected_version="1.12.0"))
        for executable in ("ffmpeg",):
            resolved = shutil.which(executable)
            results.append(
                CheckResult(
                    f"executable:{executable}",
                    resolved is not None,
                    resolved or "not found",
                )
            )
        vulkan_library = find_library("vulkan")
        results.append(
            CheckResult(
                "library:vulkan",
                vulkan_library is not None,
                vulkan_library or "not found",
            )
        )
        if shutil.which("vulkaninfo"):
            process = subprocess.run(
                ["vulkaninfo", "--summary"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            detail = "summary succeeded" if process.returncode == 0 else process.stderr[-500:]
            results.append(CheckResult("vulkan_runtime", process.returncode == 0, detail))

    payload = {
        "schema_version": 1,
        "project_root": str(project_root),
        "verify_hashes": verify_hashes,
        "require_runtime": require_runtime,
        "passed": all(result.passed for result in results),
        "checks": [asdict(result) for result in results],
    }
    return payload
