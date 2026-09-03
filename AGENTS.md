# Resilient development rules

These rules apply to the entire repository.

1. Preserve reproducibility. Never commit machine-specific absolute paths. Resolve paths from the repository/config location or accept them as CLI/config parameters.
2. When Python or a dependency changes, update `pyproject.toml`, the relevant `requirements*.txt`, and the environment/reproduction sections of both `README.md` and `README.zh-CN.md` in the same change.
3. Before code consumes a dataset or model weight, document its official URL, license, destination path, expected layout, and SHA-256 in both user-facing README versions and the relevant asset README.
4. Keep datasets, weights, checkpoints, secrets, caches, logs, and generated outputs out of Git. Extend `.gitignore` before generating a new class of artifact.
5. Record AI-assisted work in `AILOG/WORKLOG.md`. Create it locally if absent. `AILOG/` must remain ignored and must never be committed.
6. Remove one-off smoke-test scripts and their outputs after validation. Convert durable checks into maintained tests under `tests/`.
7. Before handoff, run the documented tests and lint checks, inspect `git status`, and update documentation whenever commands, paths, hardware requirements, or behavior changed.
8. Use English for source code, identifiers, docstrings, code comments, configuration comments, CI comments, commit messages, and the primary `README.md`. Keep `README.zh-CN.md` as the synchronized Simplified Chinese user guide.
