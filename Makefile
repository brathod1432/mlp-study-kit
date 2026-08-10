# mlp-study-kit Makefile
# Usage: make <target>
# Detect OS for platform-specific targets
UNAME := $(shell uname 2>/dev/null || echo Windows)

.PHONY: install install-dev install-gpu test test-cov lint typecheck \
        bandit audit security pre-commit run-ex09 run-ex10 run-debugger \
        notebook clean help

# ── Help ───────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "mlp-study-kit — available targets"
	@echo ""
	@echo "  Setup:"
	@echo "    install          core deps (requirements.txt) + editable package"
	@echo "    install-dev      dev tools (requirements_dev.txt) + editable package"
	@echo "    install-gpu      GPU deps for this OS + dev install"
	@echo ""
	@echo "  Quality:"
	@echo "    test             run pytest suite"
	@echo "    test-cov         run pytest with HTML coverage report"
	@echo "    lint             ruff: strict on nn_core, informational on exercises"
	@echo "    typecheck        mypy on nn_core"
	@echo "    bandit           bandit static security scan on "
	@echo "    audit            pip-audit CVE scan on requirements.txt"
	@echo "    security         bandit + audit"
	@echo "    pre-commit       run all pre-commit hooks on every file"
	@echo ""
	@echo "  Run:"
	@echo "    run-ex09         exercises/ex09_full_backprop.py"
	@echo "    run-ex10         exercises/ex10_bias_early_stop.py"
	@echo "    run-debugger     tools/backprop_debugger.py"
	@echo "    notebook         open Jupyter in notebooks/"
	@echo ""
	@echo "  Clean:"
	@echo "    clean            remove __pycache__, build, dist, coverage"
	@echo ""

# ── Setup ──────────────────────────────────────────────────────────────────
install:
	pip install -e . --no-deps
	pip install -r requirements.txt

install-dev:
	pip install -e . --no-deps
	pip install -r requirements_dev.txt

install-gpu:
ifeq ($(UNAME),Windows)
	pip install -r requirements_gpu_windows.txt
else
	pip install -r requirements_gpu_linux.txt
endif

# ── Quality ────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short \
	  --cov=nn_core \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov

lint:
	ruff check nn_core/
	@echo "--- exercises/ tools/ (informational) ---"
	ruff check exercises/ tools/ || true

typecheck:
	mypy nn_core/ --ignore-missing-imports

bandit:
	bandit -r  -c pyproject.toml

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
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info
	@echo "Clean done."
