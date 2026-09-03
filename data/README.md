# Data directory

No dataset is required by the initial scaffold.

- Put immutable downloaded source data in `data/raw/<dataset_name>/`.
- Put reproducibly generated data in `data/processed/<dataset_name>/`.
- Keep all actual data out of Git; `.gitkeep` only preserves the directory layout.
- Before adding data-dependent code, document the official URL, license, version/date, extraction layout, download command, and SHA-256 in this file and the root `README.md`.
- Prefer a checked-in download/preparation script over manual steps. The script must use repository-relative or user-supplied paths.

Example checksum command:

```bash
sha256sum data/raw/<dataset_name>/<archive_name>
```
