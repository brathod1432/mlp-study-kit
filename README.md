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
mlp-study-kit/
|
|-- src/nn_core/                   # Importable shared package
|   |-- __init__.py
|   |-- logger.py                  # ObjLogger — single ANSI-colour logger
|   |-- activations.py             # ActivationFn: linear/sigmoid/tanh/relu/leaky_relu/elu
|   |-- losses.py                  # LossFn: MSE + BCE (forward + derivative)
|   +-- network.py                 # NeuralNetwork: forward, backprop, train, save/load
|
|-- exercises/                     # Progressive standalone lecture exercises
|   |-- ex06_neuron_basics.py      # Stage 1 — dict neuron, plain activation functions
|   |-- ex07_forward_pass.py       # Stage 2 — OOP classes, forward propagation only
|   |-- ex08_derivatives.py        # Stage 3 — activation/loss derivatives; stubs for backprop
|   |-- ex09_full_backprop.py      # Stage 4 — complete backprop + training loop
|   |-- ex10_bias_early_stop.py    # Stage 5 — bias, train/test split, early stopping
|   |-- ex12_keras_intro.py        # Stage 6 — same task via TensorFlow/Keras
|   +-- test1_logger_check.py      # Logger smoke test
|
|-- homework/
|   |-- hw_01/hw01_tasks.py        # Python fundamentals: compute_cost, PlotXY
|   |-- hw_02/hw02_tasks.py        # LeakyReLU, ELU, Glorot init, Adagrad
|   +-- hw_03/
|       |-- hw03_tasks.py          # HW3 full implementation
|       |-- hw03_with_colab_output.py  # Colab-annotated version
|       +-- gpu_test.py            # GPU / DirectML availability test
|
|-- tools/                         # Exam-prep and debugging utilities
|   |-- backprop_debugger.py       # Full one-iteration debugger (lecture notation)
|   |-- backprop_debug.py          # Compact debugger for specific exam questions
|   |-- backprop_debug_v2.py       # Generic multi-layer debug (configurable sizes)
|   |-- backprop_step_calculator.py# Step-by-step manual calculator
|   +-- mlp_trainer_with_gradient_check.py  # Training + numerical gradient check
|   +-- README.md                  # Which tool to use when
|
|-- experiments/                   # Date-stamped exploration scripts
|   |-- 20251215.py                # Early MLP experiments
|   |-- 20251223.py                # Extended training + classification
|   |-- 20251224.py                # Pre-HW2 exploration
|   |-- 20251224_v2.py             # HW2: Glorot, Adagrad, ELU, batch processing
|   +-- README.md                  # What each script covers
|
|-- notebooks/                     # Interactive Jupyter examples
|   +-- 01_nn_core_intro.ipynb     # End-to-end: build, train, save, visualise
|
|-- tests/                         # pytest suite (49 tests, all passing)
|-- logs/                          # Auto-generated at runtime — gitignored
|-- .github/workflows/ci.yml       # CI: test + lint + bandit + pip-audit
|-- .pre-commit-config.yaml        # Pre-commit: ruff + bandit on every commit
|-- pyproject.toml                 # Package config (pip install -e . ready)
|-- requirements.txt               # Runtime deps
|-- requirements-dev.txt           # Dev deps (pytest, ruff, bandit, mypy …)
|-- requirements-gpu.txt           # Optional GPU deps (torch, tf-datasets)
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

## Using the Shared Package (`nn_core`)

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
pytest --cov=src/nn_core --cov-report=term-missing

# Lint
ruff check src/nn_core/             # or: make lint

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
Security issues: see [SECURITY.md](SECURITY.md).
