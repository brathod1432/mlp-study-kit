# Changelog

All notable changes to mlp-study-kit are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

## [0.2.0] - 2026-08-09

### Fixed
- **CRITICAL** 9 broken `from modules.GeneralUtils import ...` statements across
  `tools/`, `exercises/`, and `homework/hw_03/` (crashed on first run)
- **CRITICAL** Inverted early-stopping logic in `nn_core/network.py` --
  `basic_early_stop()` was stopping training when loss was improving fastest
- Deprecated `input_shape=` argument inside `Dense` layer in `ex12_keras_intro.py`
  replaced with explicit `tf.keras.Input` layer
- `tf.keras.losses.MAE` (function, not class) replaced with `'mae'` string alias in ex12
- `sys.exit()` in `nn_core/activations.py` and `nn_core/losses.py` replaced with
  `raise ValueError` (correct behaviour for library code)
- `np. random.seed(100)` whitespace typo fixed in ex07, ex08, ex09, ex10
- Duplicate `predict()` method removed from `nn_core/network.py`
- `plt.show()` blocking call replaced with `plt.show(block=False)` + `plt.pause(0.1)`;
  added `save_plot` parameter to `train()` for non-interactive use

### Added
- `pyproject.toml` -- proper `src/` layout packaging; `pip install -e .` now works
- `requirements-dev.txt` -- dev/test dependencies (pytest, pytest-cov, ruff, pip-audit)
- `requirements-gpu.txt` -- optional GPU dependencies (torch, tensorflow-datasets)
- `tests/` -- full pytest suite: `test_logger.py`, `test_activations.py`,
  `test_losses.py`, `test_network.py`
- `tests/conftest.py` -- auto-adds `src/` to `sys.path` for all tests
- `.github/workflows/ci.yml` -- CI: tests on Python 3.11 + 3.12, ruff lint, pip-audit
- `SECURITY.md` -- vulnerability reporting policy and secure-coding guidelines
- `LICENSE` -- MIT licence
- `CHANGELOG.md` -- this file
- `CONTRIBUTING.md` -- contribution guidelines
- `Makefile` -- shortcuts: `make install`, `make test`, `make lint`, `make audit`
- `run.ps1` -- Windows PowerShell equivalents of Makefile targets
- `.python-version` -- pins Python 3.11 for pyenv / mise
- `logs/.gitkeep` -- ensures `logs/` directory exists after clone
- `.env.example` -- documents environment variable conventions
- Type hints added to `train()` and `predict()` in `nn_core/network.py`

### Changed
- `requirements.txt` cleaned: removed `caffe` (not used), all packages now have
  explicit upper-bound version caps, comments added for each group
- `.gitignore` extended: `*.h5`, `*.keras`, `*.pt`, `*.pth`, `*.npy`, `*.npz`,
  `*.tfevents.*`, `runs/`, `.coverage`, `htmlcov/`, `.pytest_cache/`

## [0.1.0] - 2026-08-09

### Added
- Initial commit: organized migration of `neural_networks/` from Robotics_Guide
- `src/nn_core/` shared package: `logger.py`, `activations.py`, `losses.py`, `network.py`
- `exercises/` -- ex06 through ex12 with descriptive filenames
- `homework/hw_01`, `hw_02`, `hw_03`
- `tools/` -- backprop debugger, step calculator, gradient-check trainer
- `experiments/` -- date-stamped exploration scripts
