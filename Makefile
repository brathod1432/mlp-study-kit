# mlp-study-kit Makefile
.PHONY: install install-dev test test-cov lint typecheck audit security \
        pre-commit run-ex09 run-ex10 run-debugger notebook clean

# ── Setup ──────────────────────────────────────────────────────────────────
install:
	pip install -e . --no-deps
	pip install -r requirements.txt

install-dev:
	pip install -e . --no-deps
	pip install -r requirements-dev.txt

# ── Quality ────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short \
	  --cov=src/nn_core \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov

lint:
	ruff check src/nn_core/
	@echo "--- exercises/tools (warnings only) ---"
	ruff check exercises/ tools/ || true

typecheck:
	mypy src/nn_core/ --ignore-missing-imports

bandit:
	bandit -r src/ -c pyproject.toml

audit:
	pip-audit -r requirements.txt --skip-editable

security: bandit audit

pre-commit:
	pre-commit run --all-files

# ── Run exercises ──────────────────────────────────────────────────────────
run-ex09:
	python exercises/ex09_full_backprop.py

run-ex10:
	python exercises/ex10_bias_early_stop.py

run-debugger:
	python tools/backprop_debugger.py

notebook:
	jupyter notebook notebooks/

# ── Cleanup ────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info
	@echo "Clean done."
