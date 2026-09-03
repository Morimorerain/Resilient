"""Verify external assets against the tracked manifest."""

from __future__ import annotations

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


def main() -> int:
    """Validate every file asset with a complete manifest entry."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []

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

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
