# Environment contract

Create the base Conda environment from `environment.yml`, install the exact runtime dependencies from the repository root, and then install this repository without dependency resolution:

```bash
conda env create -f environment/environment.yml
conda activate resilient-fastwam
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Do not update packages interactively without updating `requirements.txt`, `pyproject.toml`, and both root README files. A post-install `pip freeze` and hardware snapshot belong in `AILOG/` during development; the validated package list is promoted to this directory after reproduction.
