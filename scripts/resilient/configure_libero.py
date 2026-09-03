"""Create a project-local LIBERO path configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--libero-repo",
        type=Path,
        default=Path("third_party/LIBERO"),
        help="Path to the pinned LIBERO checkout.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("AILOG/libero"),
        help="Ignored directory in which to create config.yaml.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/upstream.json"),
        help="Manifest containing the expected LIBERO commit.",
    )
    return parser.parse_args()


def git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    repository = args.libero_repo.resolve()
    package_root = repository / "libero" / "libero"
    required = ("assets", "bddl_files", "init_files")
    missing = [name for name in required if not (package_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Invalid LIBERO checkout at {repository}; missing directories: {missing}"
        )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_commit = manifest["libero"]["commit"]
    actual_commit = git_commit(repository)
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"LIBERO commit mismatch: expected {expected_commit}, found {actual_commit}"
        )

    config_dir = args.config_dir.resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(package_root),
        "bddl_files": str(package_root / "bddl_files"),
        "init_states": str(package_root / "init_files"),
        "datasets": str(repository / "datasets"),
        "assets": str(package_root / "assets"),
    }
    output_path = config_dir / "config.yaml"
    output_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    print(f"Wrote LIBERO configuration: {output_path}")


if __name__ == "__main__":
    main()
