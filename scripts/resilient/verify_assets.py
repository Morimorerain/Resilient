"""Verify external assets against the tracked manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "manifests" / "assets.json"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> tuple[str, int, int]:
    """Hash sorted relative paths and contents for a deterministic tree digest."""
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != ".gitkeep"
    )
    total_size = 0
    for path in files:
        relative_path = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative_path)
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                total_size += len(chunk)
        digest.update(b"\0")
    return digest.hexdigest(), len(files), total_size


def main() -> int:
    """Validate every file asset with a complete manifest entry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("all", "files", "dataset"),
        default="all",
        help="Select file assets, the dataset tree, or both.",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []

    if args.scope in {"all", "files"}:
        for asset in manifest["assets"]:
            relative_path = Path(asset["path"])
            path = PROJECT_ROOT / relative_path
            if not path.is_file():
                failures.append(f"missing: {relative_path}")
                continue
            if path.stat().st_size != asset["size"]:
                failures.append(
                    f"size mismatch: {relative_path} "
                    f"({path.stat().st_size} != {asset['size']})"
                )
                continue
            expected_digest = asset.get("sha256")
            if expected_digest is None:
                failures.append(f"manifest checksum missing: {relative_path}")
                continue
            actual_digest = sha256(path)
            if actual_digest != expected_digest:
                failures.append(
                    f"checksum mismatch: {relative_path} ({actual_digest} != {expected_digest})"
                )
                continue
            print(f"verified: {relative_path}")

    if args.scope in {"all", "dataset"}:
        dataset = manifest["dataset_repository"]
        relative_path = Path(dataset["local_path"])
        path = PROJECT_ROOT / relative_path
        if not path.is_dir():
            failures.append(f"missing dataset directory: {relative_path}")
        elif any(dataset.get(key) is None for key in ("snapshot_sha256", "file_count", "size")):
            failures.append(f"dataset manifest metadata incomplete: {relative_path}")
        else:
            digest, file_count, total_size = tree_sha256(path)
            if digest != dataset["snapshot_sha256"]:
                failures.append(
                    f"dataset checksum mismatch: {relative_path} "
                    f"({digest} != {dataset['snapshot_sha256']})"
                )
            if file_count != dataset["file_count"]:
                failures.append(
                    f"dataset file-count mismatch: {relative_path} "
                    f"({file_count} != {dataset['file_count']})"
                )
            if total_size != dataset["size"]:
                failures.append(
                    f"dataset size mismatch: {relative_path} "
                    f"({total_size} != {dataset['size']})"
                )
            if not failures:
                print(f"verified dataset: {relative_path}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
