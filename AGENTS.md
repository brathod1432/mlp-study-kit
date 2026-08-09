# AGENTS.md -- mlp-study-kit

> **Author:** Brijesh Rathod (bgrathod00@gmail.com)
> **Python:** 3.11+ | **Origin:** `neural_networks/` module from Robotics_Guide

---

## Repository Purpose

A clean, organized study kit for building a Multi-Layer Perceptron from scratch in NumPy.
Content is identical to the `neural_networks/` module in Robotics_Guide but reorganized to:
- Eliminate code duplication (`ObjLogger`, `Activation_fcn` defined once in `src/nn_core/`)
- Give files descriptive names (not `ex06eng.py` or `20251224_v2.py`)
- Separate concerns: shared core / exercises / homework / tools / experiments

---

## Key Rules

- **Single author:** All commits use `bgrathod00@gmail.com` -- no other contributor emails
- **No `*.log` files committed** -- `logs/` is gitignored, created at runtime
- **No `*.docx` / `*.pdf` committed** -- source code only
- **exercises/** files are kept standalone intentionally -- they show the learning progression from scratch and do NOT import from `nn_core`
- **src/nn_core/** is the canonical, import-ready package -- use this in new scripts

---

## Module Map

### `src/nn_core/` -- Shared Package

| File | Class / Function | Purpose |
|------|-----------------|---------|
| `logger.py` | `ObjLogger`, `title_message` | ANSI-colored console logger |
| `activations.py` | `ActivationFn` | All activations + derivatives (linear, sigmoid, tanh, relu, leaky_relu, elu) |
| `losses.py` | `LossFn` | MSE and BCE (forward + derivative) |
| `network.py` | `NeuralNetwork` | Full MLP: create, forward, backward, update, train with early stopping |

### `exercises/` -- Lecture Exercise Series

| File | Key Content |
|------|-------------|
| `ex06_neuron_basics.py` | Dict neuron, standalone activation & loss functions, plots |
| `ex07_forward_pass.py` | OOP classes, `Neural_network.forward_propagate()`, `predict()` |
| `ex08_derivatives.py` | Derivative flag added; `backward_propagate()` and `update_weights()` are stubs |
| `ex09_full_backprop.py` | Complete backprop; sin(2x)+cos(x) regression training |
| `ex10_bias_early_stop.py` | Bias term, `sklearn` train/test split, `basic_early_stop()`, regression + classification |
| `ex12_keras_intro.py` | `tf.keras.Sequential`, SGD, MAE loss, `model.fit()` |
| `test1_logger_check.py` | Logger smoke test (imports from `modules/` -- path must be set) |

### `homework/`

| Folder | File | Content |
|--------|------|---------|
| `hw_01/` | `hw01_tasks.py` | `compute_cost()`, `prime_range()`, `PlotXY`, `FileOperations` |
| `hw_02/` | `hw02_tasks.py` | Extended NN core: same as hw_01 tasks + NN extensions |
| `hw_03/` | `hw03_tasks.py` | HW3 merged implementation |
| `hw_03/` | `hw03_with_colab_output.py` | Same with Google Colab execution output included |
| `hw_03/` | `gpu_test.py` | GPU/CUDA availability check |

### `tools/` -- Exam & Debug Utilities

| File | Purpose |
|------|---------|
| `backprop_debugger.py` | `DebugMLP` dataclass -- one full forward→backward→update with all intermediate values printed in lecture notation (z[l], a[l], δ[l], ΔW[l]) |
| `backprop_debug.py` | Compact debug helper |
| `backprop_debug_v2.py` | Refined compact helper |
| `backprop_step_calculator.py` | `BackpropCalc` -- single-step step-by-step calculator |
| `mlp_trainer_with_gradient_check.py` | Trainer with numerical gradient verification |

### `experiments/` -- Date-Stamped Working Scripts

| File | Key Extensions Over exercises/ |
|------|-------------------------------|
| `20251215.py` | Early prototype experiments |
| `20251223.py` | Extended training experiments |
| `20251224.py` | Pre-hw2 exploration |
| `20251224_v2.py` | **Most evolved:** `NeuralNetCore` (static methods, batch-first), Glorot init, Adagrad, LeakyReLU, ELU, `Exercise10Data` class |

---

## Key Algorithms Reference

| Algorithm | Where Implemented | Notes |
|-----------|------------------|-------|
| Forward propagation | `ex07`+, `nn_core/network.py` | `W @ inp + b`, activation applied per layer |
| Backpropagation | `ex09`+, `nn_core/network.py` | Delta chain: output δ = ∂L/∂y · f'(v); hidden δ = W^T @ δ_next · f'(v) |
| SGD weight update | `ex09`+, `nn_core/network.py` | `W -= lr * outer(δ, prev_output)` |
| Early stopping | `ex10`, `nn_core/network.py` | Stop if test loss improvement < epsilon |
| Glorot Normal init | `experiments/20251224_v2.py` | σ = sqrt(2 / (n_in + n_out)) |
| Adagrad | `experiments/20251224_v2.py` | G += dW²; W -= (lr / sqrt(G + ε)) * dW |

---

## Git Workflow

- **Only author:** `bgrathod00@gmail.com` (set as local repo git config)
- Commit messages: `[MLP] <verb> <what>` e.g. `[MLP] Add ex09 full backprop`
- Branch naming: `feature/<topic>` or `fix/<topic>`
- Remote will be added when repo URL is provided

---

## Build & Run

```bash
# Install deps
pip install -r requirements.txt

# Add src/ to Python path (or install the package)
export PYTHONPATH=src   # Linux/Mac
$env:PYTHONPATH="src"  # PowerShell

# Run exercises
python exercises/ex09_full_backprop.py
python exercises/ex10_bias_early_stop.py

# Run exam debugger
python tools/backprop_debugger.py

# Run most advanced experiment
python experiments/20251224_v2.py
```
