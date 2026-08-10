# mlp-study-kit — Claude Project Overview

> Author: Brijesh Rathod (bgrathod00@gmail.com) | Python 3.10+ | MIT License
> Repo: https://github.com/brathod1432/mlp-study-kit

---

## What This Project Is

A progressive study kit that builds a Multi-Layer Perceptron (MLP) from scratch in
pure NumPy — from a single neuron to TensorFlow/Keras. Every equation is visible in code.

**Flat layout** — both packages live at the project root (no `src/` wrapper).

---

## Package Structure

```
nn_core/       MLP building blocks  — ActivationFn, LossFn, NeuralNetwork, ObjLogger
modules/       Helper utilities     — data_utils, plot_utils, general_utils, metrics
tests/         191 pytest tests, all passing
exercises/     ex06 → ex12 progressive lecture exercises
homework/      hw_01, hw_02, hw_03
tools/         Backpropagation debug tools
examples/      01_regression.py, 02_classification.py, 03_custom_csv_data.py
experiments/   Date-stamped exploration scripts
notebooks/     Jupyter notebooks
outputs/       Generated files (gitignored)
```

---

## Essential Commands

```bash
pip install -e .                          # install both packages (editable)
python -c "from nn_core import NeuralNetwork; print('OK')"
pytest tests/ -v                          # run 191 tests
MPLBACKEND=Agg python examples/01_regression.py
bandit -r nn_core/ modules/ -c pyproject.toml
```

---

## Key Conventions

- **No `src/` layout** — both packages at project root; `pyproject.toml` uses `where=["."]`
- **No `allow_pickle=True`** — weights use `.npz` format
- **No `sys.exit()`** in library code — raise `ValueError` or `RuntimeError`
- **`MPLBACKEND=Agg`** — always set before importing matplotlib in tests/CI
- **train() returns tuple** — `(final_loss, history_train, history_test)`
- **Git author** — only `bgrathod00@gmail.com`

---

## Detailed Theory Skills

Full theory, math, and codebase references are in `.claude/skills/`:

| Topic | File |
|-------|------|
| Activation functions (all 7 with derivatives) | `.claude/skills/activation-functions/SKILL.md` |
| Backpropagation (chain rule, delta chain, weight update) | `.claude/skills/backpropagation/SKILL.md` |
| Loss functions (MSE, BCE, metrics) | `.claude/skills/loss-functions/SKILL.md` |
| Training workflow (data prep, init, optimizers, API) | `.claude/skills/neural-network-training/SKILL.md` |
