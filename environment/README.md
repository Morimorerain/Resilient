# Environment contract

The validated path uses uv `0.11.7` with a uv-managed standalone CPython. This prevents Conda base-library RPATHs from leaking into the environment.

```bash
uv python install 3.10.20
uv venv --python 3.10.20 .venv
source .venv/bin/activate

uv pip install --index-strategy unsafe-best-match -r requirements.txt
uv pip install -r requirements-libero.txt
uv pip install --no-deps -e .
```

The `unsafe-best-match` setting is required because `requirements.txt` uses PyTorch's CUDA 12.8 extra index. Every package remains exactly pinned; the setting only permits uv to resolve ordinary packages from PyPI after seeing a package name on the PyTorch index.

After the first validated resolution, reproduce the complete direct and transitive package set with:

```bash
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r environment/pip-freeze-cu128.txt
```

The lock snapshot excludes editable local packages. Install Resilient and the pinned LIBERO checkout separately as described in the root README.

`environment.yml` is an equivalent Conda bootstrap specification, but the uv path is the validated reference environment. Do not update packages interactively without updating `requirements.txt`, `requirements-libero.txt`, `pyproject.toml`, `pip-freeze-cu128.txt`, and both root README files. A hardware snapshot belongs in ignored `AILOG/` during development.
