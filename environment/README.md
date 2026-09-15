# Environment contract

The validated path uses uv `0.11.7` with a uv-managed standalone CPython. This prevents Conda base-library RPATHs from leaking into the environment. LIBERO and RoboTwin use separate environments because their pinned NumPy/OpenCV/MPLib constraints are incompatible.

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

The lock snapshot excludes editable local packages. Install Resilient and the pinned LIBERO checkout separately as described in the root README. OPSD adapter support uses `peft==0.15.2`; it is part of every primary dependency declaration and the lock snapshot.

For RoboTwin, use the reproducible helper rather than modifying the LIBERO environment:

```bash
bash scripts/resilient/create_robotwin_environment.sh .venv-robotwin
source .venv-robotwin/bin/activate
python scripts/resilient/verify_robotwin.py
```

The helper installs build tools first, then the NumPy-1.26 simulator profile, and finally builds
CuRobo v0.7.7 with build isolation disabled. It defaults to `/usr/local/cuda-12.8` because the
FastWAM wheel is `torch==2.7.1+cu128`; override this only with
`RESILIENT_CUDA_HOME=/path/to/cuda-12.8`. `pip-freeze-robotwin-cu128.txt` is the validated complete
snapshot. `environment-robotwin.yml` bootstraps the Conda packages but intentionally cannot perform
the CuRobo no-build-isolation step; run the helper's final two commands afterward.

`environment.yml` is an equivalent Conda bootstrap specification for LIBERO, but the uv path is the validated reference environment. Run `conda env create -f environment/environment.yml` from the repository root. Do not update packages interactively without updating the applicable `requirements*.txt`, `pyproject.toml`, lock snapshot, and both root README files. A hardware snapshot belongs in ignored `AILOG/` during development.
