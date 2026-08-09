# mlp-study-kit Makefile
# Usage: make <target>
# Requires: Python 3.11+, pip

.PHONY: install install-dev test lint audit clean run-ex09 run-ex10 run-debugger

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt
	pip install -e . --no-deps

install-dev:
	pip install -r requirements-dev.txt
	pip install -e . --no-deps

# ──────────────────────────────────────────────────────────────────────────────
# Quality
# ──────────────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src/nn_core --cov-report=term-missing

lint:
	ruff check src/nn_core/

audit:
	pip-audit -r requirements.txt --skip-editable

# ──────────────────────────────────────────────────────────────────────────────
# Run exercises
# ──────────────────────────────────────────────────────────────────────────────

run-ex09:
	python exercises/ex09_full_backprop.py

run-ex10:
	python exercises/ex10_bias_early_stop.py

run-debugger:
	python tools/backprop_debugger.py

# ──────────────────────────────────────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info
