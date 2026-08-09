# mlp-study-kit

A structured study kit for building a **Multi-Layer Perceptron from scratch** in pure NumPy,
progressing step-by-step from a single neuron all the way to TensorFlow/Keras.

**Author:** Brijesh Rathod | **Python:** 3.11+ | **Origin:** Neural Networks module, WUT

---

## Repository Structure

```
mlp-study-kit/
|
|-- src/nn_core/                 # Shared package -- canonical, deduplicated implementations
|   |-- __init__.py
|   |-- logger.py                # ObjLogger (single source of truth)
|   |-- activations.py           # ActivationFn: linear, sigmoid, tanh, relu, leaky_relu, elu
|   |-- losses.py                # LossFn: MSE, Binary Cross-Entropy (forward + derivative)
|   +-- network.py               # NeuralNetwork: full MLP with bias, backprop, early stopping
|
|-- exercises/                   # Progressive lecture exercises (standalone, from scratch)
|   |-- ex06_neuron_basics.py    # Dict-based neuron, activation & loss functions as plain fns
|   |-- ex07_forward_pass.py     # First OOP: Activation_fcn, Loss_fcn, forward propagation
|   |-- ex08_derivatives.py      # Derivatives added; backward/update as stubs (student skeleton)
|   |-- ex09_full_backprop.py    # Complete backprop + training loop; sin+cos regression
|   |-- ex10_bias_early_stop.py  # Bias terms, train/test split, early stopping
|   |-- ex12_keras_intro.py      # TensorFlow/Keras bridge -- linear regression in 40 lines
|   +-- test1_logger_check.py    # Logger integration smoke test
|
|-- homework/
|   |-- hw_01/hw01_tasks.py      # Python fundamentals: compute_cost, prime_range, PlotXY
|   |-- hw_02/hw02_tasks.py      # Extended activations (LeakyReLU, ELU), Glorot init, Adagrad
|   +-- hw_03/
|       |-- hw03_tasks.py        # HW3 merged implementation
|       |-- hw03_with_colab_output.py  # Colab-annotated version with run outputs
|       +-- gpu_test.py          # GPU availability test
|
|-- tools/                       # Exam-prep and debugging utilities
|   |-- backprop_debugger.py     # Step-by-step forward->backward->update with lecture notation
|   |-- backprop_debug.py        # Compact backprop debug helper
|   |-- backprop_debug_v2.py     # Refined debug helper v2
|   |-- backprop_step_calculator.py  # Single-step backprop calculator
|   +-- mlp_trainer_with_gradient_check.py  # Trainer with numerical gradient verification
|
|-- experiments/                 # Date-stamped working scripts (exploration / testing)
|   |-- 20251215.py
|   |-- 20251223.py
|   |-- 20251224.py
|   +-- 20251224_v2.py           # Most evolved: HW2 extensions (ELU, Adagrad, Glorot, batch)
|
|-- logs/                        # Auto-generated at runtime -- gitignored
|-- requirements.txt
+-- AGENTS.md                    # Full knowledge base for AI agents
```

---

## Exercise Progression

| # | File | New Concept Introduced |
|---|------|----------------------|
| 1 | `ex06_neuron_basics.py` | Neuron as dict, activation fns (linear/tanh/relu), MSE & BCE loss |
| 2 | `ex07_forward_pass.py` | OOP: `Activation_fcn`, `Loss_fcn`, `Neural_network`, forward-only |
| 3 | `ex08_derivatives.py` | Analytical derivatives for all activations and losses; stubs for backprop |
| 4 | `ex09_full_backprop.py` | Complete backpropagation + training loop; sin+cos regression task |
| 5 | `ex10_bias_early_stop.py` | Bias terms, train/test split (`sklearn`), early stopping criterion |
| 6 | `ex12_keras_intro.py` | TensorFlow/Keras API; same task with `model.fit()` |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full backprop exercise
python exercises/ex09_full_backprop.py

# Run the exam-prep debugger (shows every intermediate value)
python tools/backprop_debugger.py

# Run the most advanced homework (hw2 with Adagrad + Glorot init)
python experiments/20251224_v2.py
```

---

## Shared Package Usage

Once `src/` is on your path (or installed), exercises can import from `nn_core`:

```python
import sys
sys.path.insert(0, "src")

from nn_core import NeuralNetwork, ActivationFn, LossFn, ObjLogger

logger = ObjLogger("MyRun")
model  = NeuralNetwork()
```

---

## Installation

```bash
# Python 3.11+ required
pip install -r requirements.txt
```

Key packages: `numpy`, `matplotlib`, `scikit-learn`, `tensorflow`, `keras`
