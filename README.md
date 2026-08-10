# mlp-study-kit

[![CI](https://github.com/brathod1432/mlp-study-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/brathod1432/mlp-study-kit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A structured study kit for building a **Multi-Layer Perceptron from scratch** in pure NumPy,
progressing step-by-step from a single neuron all the way to TensorFlow/Keras.
Every component is purposely transparent — no black boxes.

**Author:** Brijesh Rathod (`bgrathod00@gmail.com`) | **Origin:** WUT Neural Networks course

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/brathod1432/mlp-study-kit.git
cd mlp-study-kit

# 2. Install (editable — makes 'import nn_core' work everywhere)
pip install -e .

# 3. Verify installation
python -c "from nn_core import NeuralNetwork, ActivationFn, LossFn; print('nn_core OK')"

# 4. Run the full-backprop exercise
python exercises/ex09_full_backprop.py

# 5. Run the exam-prep debugger
python tools/backprop_debugger.py
```

> **Headless / CI environments (no display):** set `MPLBACKEND=Agg` to suppress
> all matplotlib windows. All exercises and `network.train()` honour this env var.

---

## Repository Structure

```
mlp-study-kit/                     ← project root (flat layout — no src/ wrapper)
|
|-- nn_core/                       # MLP core package — import: from nn_core import ...
|   |-- __init__.py                # Exports NeuralNetwork, ActivationFn, LossFn, ObjLogger
|   |-- logger.py                  # ObjLogger — ANSI-coloured logger (Windows-safe)
|   |-- activations.py             # ActivationFn: linear/sigmoid/tanh/relu/leaky_relu/elu
|   |-- losses.py                  # LossFn: MSE + BCE (forward + derivative)
|   +-- network.py                 # NeuralNetwork: build, forward, backprop, train, save/load
|
|-- modules/                       # Helper package — import: from modules.data_utils import ...
|   |-- __init__.py                # Top-level re-exports for all four sub-modules
|   |-- general_utils.py           # ensure_directory, as_float_array, print_matrices …
|   |-- plot_utils.py              # plot_loss_history, plot_predictions, plot_activations …
|   |-- data_utils.py              # make_regression_data, train_test_split, load_csv, k_fold_split
|   +-- metrics.py                 # mse, rmse, mae, r2_score, accuracy, f1_score, evaluate …
|
|-- tests/                         # pytest suite (191 tests, all passing)
|   |-- conftest.py                # Adds project root to sys.path; sets MPLBACKEND=Agg
|   |-- test_activations.py
|   |-- test_losses.py
|   |-- test_network.py
|   |-- test_logger.py
|   |-- test_general_utils.py
|   |-- test_data_utils.py
|   |-- test_plot_utils.py
|   +-- test_metrics.py
|
|-- examples/                      # Runnable end-to-end scripts
|   |-- 01_regression.py           # sin(2x)+cos(x)+5 regression with weight save
|   |-- 02_classification.py       # 2D binary classifier + decision boundary
|   |-- 03_custom_csv_data.py      # Load your own CSV and train
|   +-- README.md
|
|-- exercises/                     # Progressive lecture exercises (standalone)
|   |-- ex06_neuron_basics.py      # Stage 1 — dict neuron, activation + loss functions
|   |-- ex07_forward_pass.py       # Stage 2 — OOP, forward propagation only
|   |-- ex08_derivatives.py        # Stage 3 — derivatives, backprop stubs
|   |-- ex09_full_backprop.py      # Stage 4 — complete backprop + training loop
|   |-- ex10_bias_early_stop.py    # Stage 5 — bias, train/test split, early stopping
|   +-- ex12_keras_intro.py        # Stage 6 — same task via TensorFlow/Keras
|
|-- homework/hw_01, hw_02, hw_03/  # Homework assignments
|-- tools/                         # Exam-prep backprop debuggers (see tools/README.md)
|-- experiments/                   # Date-stamped exploration scripts (see experiments/README.md)
|-- notebooks/01_nn_core_intro.ipynb  # Interactive Jupyter walkthrough
|-- outputs/                       # Generated plots/weights/CSVs (gitignored, dir tracked)
|-- logs/                          # Runtime logs (gitignored)
|
|-- pyproject.toml                 # Package config — pip install -e . discovers nn_core + modules
|-- requirements.txt               # Cross-platform base
|-- requirements_windows.txt       # + TensorFlow for Windows
|-- requirements_linux.txt         # + TensorFlow for Linux
|-- requirements_dev.txt           # Dev tools (pytest, ruff, mypy, bandit, pre-commit)
|-- requirements_gpu_windows.txt   # Windows GPU (torch CUDA/DirectML)
|-- requirements_gpu_linux.txt     # Linux GPU (torch CUDA/ROCm)
|-- .github/workflows/ci.yml       # CI: test (3.10-3.12) + lint + mypy + bandit + pip-audit
|-- .pre-commit-config.yaml        # ruff + bandit on every commit
|-- Makefile / run.ps1             # Developer shortcuts (make test, make lint, ...)
+-- AGENTS.md                      # Full knowledge base for AI agents / Devin
```

---

## Exercise Progression

| # | File | New Concept |
|---|------|------------|
| 1 | `ex06_neuron_basics.py` | Neuron as dict; manual activation + loss functions |
| 2 | `ex07_forward_pass.py` | First OOP: `Activation_fcn`, `Loss_fcn`, forward-only |
| 3 | `ex08_derivatives.py` | Analytical derivatives; `backward`/`update` as student stubs |
| 4 | `ex09_full_backprop.py` | **Complete backprop + training**; sin+cos regression |
| 5 | `ex10_bias_early_stop.py` | Bias terms, train/test split, early stopping |
| 6 | `ex12_keras_intro.py` | Same task via `tf.keras` API — 40 lines |

---

## Package Overview

Two packages live at the **project root** (flat layout) — both auto-discovered by `pip install -e .`:

| Package | Import | Purpose |
|---------|--------|---------|
| `nn_core` | `from nn_core import NeuralNetwork` | MLP building blocks — activations, losses, network, logger |
| `modules` | `from modules.data_utils import make_regression_data` | Helper utilities — data generation, visualisation, array validation |

```python
# nn_core — build and train the network
from nn_core import NeuralNetwork, ActivationFn, LossFn, ObjLogger

# modules — everything around the network
from modules.data_utils    import make_regression_data, train_test_split, normalize
from modules.plot_utils    import plot_loss_history, plot_predictions, plot_activations
from modules.general_utils import ensure_directory, as_float_array, describe_array
```

---

## Using `nn_core`

After `pip install -e .`, import directly — no path hacks needed:

```python
import numpy as np
from nn_core import NeuralNetwork, ActivationFn, LossFn, ObjLogger

# 1. Define architecture
structure = [
    {"type": "input",  "units": 1},
    {"type": "dense",  "units": 32, "activation_function": "tanh",   "bias": True},
    {"type": "dense",  "units": 1,  "activation_function": "linear", "bias": True},
]

# 2. Build + inspect
model = NeuralNetwork()
net   = model.create_network(structure)
print(model)          # human-readable architecture summary
# NeuralNetwork(
#   [0] input   units=1
#   [1] dense   units=32   act=tanh         bias=True  weights=(32, 2)
#   [2] dense   units=1    act=linear       bias=True  weights=(1, 33)
# )

# 3. Train
X = np.linspace(-3, 3, 100).reshape(-1, 1)
Y = np.sin(2 * X) + np.cos(X) + 5.0

model.train(
    net, X, Y,
    l_rate=0.01, n_epoch=500, loss_function="mse",
    verbose=1,
    save_plot="loss.png",   # omit to show interactive window
)

# 4. Predict
preds = model.predict(net, X)

# 5. Save / reload weights
model.save_weights(net, "weights.npy")
model.load_weights(net, "weights.npy")
```

---

## Using `modules`

```python
import numpy as np
from modules.data_utils    import make_regression_data, train_test_split, normalize
from modules.plot_utils    import plot_loss_history, plot_predictions, plot_activations
from modules.general_utils import ensure_directory, as_float_array

# 1. Generate data — same dataset as exercises/ex09 and ex10
X, Y = make_regression_data(n=150, noise=0.15, seed=42)

# 2. Split
X_tr, Y_tr, X_te, Y_te = train_test_split(X, Y, test_ratio=0.25)

# 3. Normalise (apply train stats to both sets)
X_tr_n, mu, sigma = normalize(X_tr)
X_te_n, _,  _     = normalize(X_te, mean=mu, std=sigma)

# 4. Train the network (using nn_core)
from nn_core import NeuralNetwork
model = NeuralNetwork()
net   = model.create_network([
    {"type": "input",  "units": 1},
    {"type": "dense",  "units": 32, "activation_function": "tanh",   "bias": True},
    {"type": "dense",  "units": 1,  "activation_function": "linear", "bias": True},
])
model.train(net, X_tr_n, Y_tr, X_te_n, Y_te, l_rate=0.02, n_epoch=500, verbose=0)

# 5. Visualise
ensure_directory("outputs")
preds = model.predict(net, X_te_n)
plot_predictions(X_te, Y_te, preds, save_path="outputs/predictions.png")
plot_loss_history([0.5, 0.3, 0.1], save_path="outputs/loss.png")
plot_activations(save_path="outputs/activations.png")
```

---

## Installation

```bash
# Python 3.10+ required

# Step 1 — install the nn_core package (editable — no PYTHONPATH hacks)
pip install -e .

# Step 2 — pick the requirements file for your OS and use case:
```

| File | OS | Content |
|------|----|---------|
| `requirements.txt` | Both | **Base** — numpy, matplotlib, scipy, sklearn, colorama, tqdm |
| `requirements_windows.txt` | Windows | Base + TensorFlow for Windows (includes VC++ note) |
| `requirements_linux.txt` | Linux / macOS | Base + TensorFlow for Linux |
| `requirements_dev.txt` | Both | Base + pytest, ruff, mypy, bandit, pre-commit, jupyter |
| `requirements_gpu_windows.txt` | Windows | Windows + PyTorch CPU/CUDA + DirectML (AMD) + tf-datasets |
| `requirements_gpu_linux.txt` | Linux | Linux + PyTorch CPU/CUDA + ROCm (AMD) + tf-datasets |

```bash
# Windows — core exercises only
pip install -r requirements_windows.txt

# Linux — core exercises only
pip install -r requirements_linux.txt

# Either OS — development (tests, lint, security, Jupyter)
pip install -r requirements_dev.txt

# Either OS — GPU support for hw_03/gpu_test.py
pip install -r requirements_gpu_windows.txt   # Windows
pip install -r requirements_gpu_linux.txt     # Linux

# Or use the pyproject.toml extras:
pip install -e ".[tensorflow]"    # TF-compatible numpy + TF 2.14
pip install -e ".[dev]"           # dev tools only
```

> **NumPy version note:** `nn_core` works with NumPy 1.x and 2.x.
> TensorFlow 2.14 requires `numpy < 2.0`, so the `_windows` / `_linux`
> requirements pin `numpy<2.0`. The base `requirements.txt` allows `<3.0`
> for users who don't need TensorFlow.

---

## Development Workflow

```bash
# Run all tests
pytest                              # or: make test  / .\run.ps1 test

# Run with coverage
pytest --cov=nn_core --cov-report=term-missing

# Lint
ruff check nn_core/             # or: make lint

# Security scan
bandit -r src/                      # or: make audit
pip-audit -r requirements.txt --skip-editable

# Install pre-commit hooks (run once after clone)
pre-commit install
```

---

## Headless / Batch Environments

Set `MPLBACKEND=Agg` to prevent any matplotlib window from opening:

```bash
# Linux / Mac
export MPLBACKEND=Agg
python exercises/ex09_full_backprop.py

# Windows PowerShell
$env:MPLBACKEND = "Agg"
python exercises/ex09_full_backprop.py
```

Or use the `save_plot` parameter to save the training plot to a file instead:

```python
model.train(net, X, Y, verbose=1, save_plot="training_history.png")
```

---

## License

This project is released under the **[MIT License](LICENSE)** — free to use,
copy, modify, and distribute for any purpose, including commercial use.

```
MIT License  |  Copyright (c) 2026 Brijesh Rathod
```

---

## Contributing

Contributions are welcome from anyone:

- **Open an issue** — bug reports, questions, suggestions →
  [github.com/brathod1432/mlp-study-kit/issues](https://github.com/brathod1432/mlp-study-kit/issues)
- **Submit a PR** — anyone can fork and open a pull request;
  [@brathod1432](https://github.com/brathod1432) reviews and merges

Full guide: [CONTRIBUTING.md](CONTRIBUTING.md) | Security: [SECURITY.md](SECURITY.md)
