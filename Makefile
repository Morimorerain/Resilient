PYTHON ?= python

.PHONY: install install-dev test lint verify-assets capture-environment clean

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-deps -e .

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install --no-deps -e .

test:
	PYTHONPATH=src $(PYTHON) -m pytest -m "not gpu and not libero"

lint:
	$(PYTHON) -m ruff check src/resilient tests scripts/resilient

verify-assets:
	$(PYTHON) scripts/resilient/verify_assets.py

capture-environment:
	$(PYTHON) scripts/resilient/capture_environment.py --output AILOG/environment.json

clean:
	find src tests scripts/resilient -type d -name __pycache__ -prune -exec rm -rf {} +
	find src tests scripts/resilient -type f -name '*.py[co]' -delete
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache htmlcov
	rm -f .coverage coverage.xml report.xml
