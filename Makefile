PYTHON ?= python3.10
VENV ?= .venv

.PHONY: setup test lint run clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip==24.0
	$(VENV)/bin/python -m pip install -r requirements-dev.txt

test:
	$(VENV)/bin/python -m unittest discover -s tests -v

lint:
	$(VENV)/bin/ruff check .

run:
	$(VENV)/bin/python -m resilient --version

clean:
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache htmlcov
	rm -f .coverage
