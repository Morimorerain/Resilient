#!/usr/bin/env python3
"""Download and safely install pinned FastWAM/RoboTwin external assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "assets.json"
HUGGINGFACE_CACHE = PROJECT_ROOT / "AILOG" / "huggingface-cache"


def sha256(path: Path) -> str:
    """Hash a downloaded file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, entry: dict[str, Any]) -> None:
    """Reject a missing, truncated, or content-mismatched external file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    if actual_size != int(entry["size"]):
        raise ValueError(f"Size mismatch for {path}: {actual_size} != {entry['size']}")
    actual_sha256 = sha256(path)
    if actual_sha256 != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual_sha256} != {entry['sha256']}")


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe path in archive: {member.filename}")
        if member.is_dir():
            continue
        mode = member.external_attr >> 16
        if mode & 0o170000 == 0o120000:
            raise ValueError(f"Symbolic link is not allowed in archive: {member.filename}")
    return members


def _install_archive(archive_path: Path, entry: dict[str, Any], install_root: Path) -> None:
    expected_name = str(entry["expected_directory"])
    destination = install_root / expected_name
    if destination.exists():
        if destination.is_dir() and any(destination.iterdir()):
            print(f"asset already installed, keeping: {destination}")
            return
        raise FileExistsError(f"Refusing to replace existing path: {destination}")

    extract_root = PROJECT_ROOT / "AILOG" / "robotwin_extract"
    extract_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{expected_name}-", dir=extract_root))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = _safe_members(archive)
            archive.extractall(temp_dir, members=members)
        candidate = temp_dir / expected_name
        if not candidate.is_dir():
            directories = [path for path in temp_dir.iterdir() if path.is_dir()]
            files = [path for path in temp_dir.iterdir() if path.is_file()]
            if len(directories) == 1 and not files:
                candidate = directories[0]
            else:
                raise ValueError(
                    f"Archive {archive_path} does not contain expected directory {expected_name}"
                )
        install_root.mkdir(parents=True, exist_ok=True)
        candidate.replace(destination)
        print(f"installed: {destination}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _download_model_assets(manifest: dict[str, Any]) -> None:
    repository = manifest["model_repository"]
    selected = {
        "robotwin_uncond_3cam_384.pt",
        "robotwin_uncond_3cam_384_dataset_stats.json",
    }
    entries = {
        Path(item["path"]).name: item
        for item in manifest["assets"]
        if Path(item["path"]).name in selected
    }
    destination = PROJECT_ROOT / "checkpoints" / "fastwam_release"
    destination.mkdir(parents=True, exist_ok=True)
    for filename in sorted(selected):
        path = Path(
            hf_hub_download(
                repo_id=repository["id"],
                revision=repository["revision"],
                filename=filename,
                local_dir=destination,
                cache_dir=HUGGINGFACE_CACHE,
            )
        )
        verify_file(path, entries[filename])
        print(f"verified: {path.relative_to(PROJECT_ROOT)}")


def _download_simulator_assets(manifest: dict[str, Any], *, install: bool) -> None:
    repository = manifest["robotwin_simulator_assets"]
    download_dir = PROJECT_ROOT / "downloads" / "robotwin"
    download_dir.mkdir(parents=True, exist_ok=True)
    install_root = PROJECT_ROOT / repository["install_root"]
    for entry in repository["archives"]:
        archive_path = Path(
            hf_hub_download(
                repo_id=repository["repository"],
                repo_type=repository["repository_type"],
                revision=repository["revision"],
                filename=entry["filename"],
                local_dir=download_dir,
                cache_dir=HUGGINGFACE_CACHE,
            )
        )
        verify_file(archive_path, entry)
        print(f"verified: {archive_path.relative_to(PROJECT_ROOT)}")
        if install:
            _install_archive(archive_path, entry, install_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=("all", "model", "simulator"),
        default="all",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Verify simulator archives but do not extract them.",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    required_bytes = sum(
        int(entry["size"])
        for entry in manifest["robotwin_simulator_assets"]["archives"]
    )
    if args.component in {"all", "model"}:
        required_bytes += sum(
            int(item["size"])
            for item in manifest["assets"]
            if "robotwin_uncond_3cam_384" in item["path"]
        )
    free_bytes = shutil.disk_usage(PROJECT_ROOT).free
    if free_bytes < required_bytes * 2:
        raise RuntimeError(
            "Insufficient free space for downloads plus extraction safety margin: "
            f"free={free_bytes}, required_margin={required_bytes * 2}"
        )

    if args.component in {"all", "model"}:
        _download_model_assets(manifest)
    if args.component in {"all", "simulator"}:
        _download_simulator_assets(manifest, install=not args.download_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
