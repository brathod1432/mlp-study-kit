# Changelog

All notable changes to mlp-study-kit are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

## [0.3.0] - 2026-08-09

### Fixed — Critical
- **`pyproject.toml` wrong `build-backend`** — `setuptools.backends.legacy:build`
  changed to `setuptools.build_meta` (the correct, stable, documented entry-point).
  `pip install -e .` now works reliably across all setuptools versions.
- **`import matplotlib.pyplot as plt` at module top-level** in `network.py` —
  moved to a lazy import inside `_plot_history()`. `from nn_core import NeuralNetwork`
  no longer triggers matplotlib backend detection or `_tkinter.TclError` in
  headless environments.
- **`sys.exit()` in all 4 exercise files (ex07–ex10)** — replaced with
  `raise ValueError()` in both `Activation_fcn.output()` and `Loss_fcn.output()`.
  Callers now receive a proper exception they can catch.
- **`ENV = os.getcwd()` + `sys.path.append(ENV)` hacks** — removed from all
  remaining files: `experiments/20251215.py`, `20251223.py`, `20251224.py`,
  `20251224_v2.py`, `homework/hw_01/hw01_tasks.py`, `hw_02/hw02_tasks.py`,
  `hw_03/hw03_with_colab_output.py`.
- **Inline `ObjLogger` copies** — removed from all 7 affected files; replaced
  with `from nn_core.logger import ObjLogger, title_message` using a self-
  contained `_ROOT` path resolution that works with or without `pip install -e .`.

### Fixed — Security
- `SECURITY.md` — `pip-audit` command updated to include `--skip-editable`
  (matches the CI command; avoids false warning on the editable project itself).
- `pyproject.toml` — `tqdm` added to `[project.dependencies]` (was in
  `requirements.txt` but missing from package metadata).
- `[project.optional-dependencies].tensorflow` — pinned to `tensorflow<2.16`
  and `keras<3.0` to prevent accidental Keras 3 install that breaks `tf.keras`.
- `numpy` upper bound loosened to `<3.0` for core `nn_core` (TF extra still
  pins `<2.0`); removes the confusing numpy 2.x downgrade for non-TF users.

### Added
- **`NeuralNetwork.__repr__` / `__str__`** — `print(model)` now shows a
  human-readable architecture summary with layer index, units, activation,
  bias flag, and weight shape.
- **`NeuralNetwork.save_weights(net, path)`** — saves all weight matrices to
  a `.npy` file using `np.save(..., allow_pickle=True)`.
- **`NeuralNetwork.load_weights(net, path)`** — restores weights from a saved
  file with shape validation (raises `ValueError` on mismatch).
- **Matplotlib backend guard** in all exercise files (`ex06`, `ex09`, `ex10`,
  `ex12`) and in `experiments/20251223.py`, `20251224.py`, `20251224_v2.py`,
  `homework/hw_01/hw01_tasks.py`, `hw_02/hw02_tasks.py`,
  `hw_03/hw03_with_colab_output.py`. Setting `MPLBACKEND=Agg` suppresses all
  windows without changing any other code.
- **`NeuralNetwork._plot_history()`** — internal static method that handles all
  plot logic (lazy matplotlib import, `MPLBACKEND` env var, `save_plot` path,
  `plt.close()` after save).
- **`notebooks/01_nn_core_intro.ipynb`** — end-to-end Jupyter notebook: import,
  build, train with early stopping, visualise predictions, save/load weights,
  activations explorer.
- **`.pre-commit-config.yaml`** — ruff (fix mode), bandit, trailing-whitespace,
  end-of-file, check-yaml/toml, debug-statement detector, large-file guard.
- **`tools/README.md`** — guide explaining when to use each of the 5 debug tools,
  notation reference (lecture symbol → code variable), and usage examples.
- **`experiments/README.md`** — index of all 4 dated scripts with chronological
  description, key concepts per file, and relation to `nn_core/`.
- **5 new tests** in `tests/test_network.py`:
  - `TestRepr::test_repr_uninitialised`
  - `TestRepr::test_repr_shows_layers`
  - `TestRepr::test_str_equals_repr`
  - `TestSaveLoadWeights::test_save_load_roundtrip`
  - `TestSaveLoadWeights::test_load_shape_mismatch_raises`
- **`pyproject.toml` updates**: correct `build-backend`, `tqdm` in deps, Python
  3.10/3.12 classifiers, OS independent classifier, `[tool.bandit]`, `[tool.mypy]`
  sections, `[tool.ruff.lint]` moved to correct subsection.
- **CI (`ci.yml`) updates**: `MPLBACKEND: Agg` env var on test job, 3-version
  matrix (3.10 / 3.11 / 3.12), `concurrency` cancel-in-progress, `typecheck`
  job (mypy), `bandit` added to security job, lint job now covers
  `exercises/` + `tools/` (informational), coverage XML artifact upload.
- **`requirements-dev.txt` updates**: added `bandit[toml]`, `mypy`, `pre-commit`,
  `jupyter`, `ipykernel`.
- **`Makefile` updates**: `bandit`, `typecheck`, `security`, `pre-commit`,
  `notebook` targets added.
- **`notebooks/` added to `.gitignore`** output artefacts (`.png`, `.npy`,
  `.ipynb_checkpoints/`).

### Changed
- `README.md` fully overhauled:
  - CI badge, Python badge, MIT licence badge at top
  - `pip install -e .` as primary installation method (not `pip install -r`)
  - Quick-verify command: `python -c "from nn_core import NeuralNetwork; ..."`
  - `MPLBACKEND=Agg` headless section
  - `save_plot` parameter documented with example
  - `NeuralNetwork.__repr__` output shown in usage example
  - `save_weights` / `load_weights` shown in usage example
  - Full development workflow section (pytest, ruff, bandit, pre-commit)
  - Removed incorrect `sys.path.insert(0, "src")` hack from "Shared Package" section

## [0.2.0] - 2026-08-09

### Fixed
- 9 broken `from modules.GeneralUtils import ...` statements across tools/,
  exercises/, homework/hw_03/ — crashed on first run
- Inverted early-stopping logic — `basic_early_stop()` was stopping when loss
  was improving most; fixed `>` to `<`
- Deprecated `input_shape=` in `Dense` layer (ex12) → `tf.keras.Input`
- `tf.keras.losses.MAE` (function) → `'mae'` string alias (ex12)
- `sys.exit()` in `nn_core/activations.py` + `losses.py` → `raise ValueError`
- `np. random.seed(100)` whitespace typo in ex07, ex08, ex09, ex10
- `plt.show()` blocking → `plt.show(block=False)` + `save_plot` in `network.py`

### Added
- `pyproject.toml`, `requirements-dev.txt`, `requirements-gpu.txt`
- `tests/` with 49 passing tests
- `.github/workflows/ci.yml`
- `SECURITY.md`, `LICENSE` (MIT), `CHANGELOG.md`, `CONTRIBUTING.md`
- `Makefile`, `run.ps1`, `.python-version`, `logs/.gitkeep`, `.env.example`

## [0.1.0] - 2026-08-09

### Added
- Initial commit: organised migration of `neural_networks/` from Robotics_Guide
- `src/nn_core/` shared package: logger, activations, losses, network
- `exercises/` ex06–ex12, `homework/` hw01–hw03, `tools/`, `experiments/`
