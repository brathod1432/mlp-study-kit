# tools/ — Backprop Debug & Exam-Prep Utilities

Five standalone Python scripts for stepping through backpropagation manually — essential for exam preparation and understanding the algorithm at a granular level.

---

## Which Tool to Use

| Goal | Tool |
|------|------|
| Verify a specific exam question (given x, t, W, b, η) | `backprop_debugger.py` |
| Debug a fixed 2-2-2 network (no bias) in one step | `backprop_debug.py` |
| Debug any layer-size configuration in one step | `backprop_debug_v2.py` |
| Calculate one backprop step and print every intermediate value | `backprop_step_calculator.py` |
| Train a small network and confirm gradients numerically | `mlp_trainer_with_gradient_check.py` |

---

## `backprop_debugger.py` — Full One-Iteration Debugger

**Best for:** Answering exam questions of the form "given these weights, inputs, and targets, what are the updated weights after one backprop step?"

```bash
python tools/backprop_debugger.py
```

Edit the `EXAMPLE_CONFIG` dict at the bottom to match your question:

```python
EXAMPLE_CONFIG = {
    "x":         [1.0, 0.5],        # y^(0) — input sample
    "t":         [1.0],             # t — target output
    "lr":        0.5,               # η — learning rate
    "loss_type": "mse_sigmoid",     # or "cross_entropy_sigmoid"
    "W": [
        [[0.10, -0.20], [0.40, 0.30]],   # w^(1): INPUT → HIDDEN
        [[0.20, -0.50]],                  # w^(2): HIDDEN → OUTPUT
    ],
    "b": [[0.0, 0.0], [0.0]],      # biases per layer
}
```

**Output includes** (in lecture notation):
- `a[0]` = input `y^(0)`
- `z[l]` = pre-activation `v^(l)`
- `a[l]` = post-activation `y^(l)`
- `delta[l]` = `δ^(l)` — error signal
- `dW[l]` = gradient `∂E/∂w^(l)`
- `ΔW[l]` = weight update `η * dW[l]`
- Loss before and after — sanity check that loss decreased

---

## `backprop_debug.py` — Fixed 2-2-2 Network (No Bias)

**Best for:** The classic exam topology: 2 inputs → 2 hidden → 2 outputs, sigmoid activation, MSE loss, one sample.

Hardcoded for the specific question structure from the WUT Neural Networks exam:
- Inputs: `x1=4, x2=4` | Targets: `t1=2, t2=2` | `η=0.5`
- Update rule: `w := w + η * δ * input` (lecture-style sign)

Edit the weight values directly in the script and run:

```bash
python tools/backprop_debug.py
```

---

## `backprop_debug_v2.py` — Generic Multi-Layer Debugger

**Best for:** Any network topology — configure `D` (inputs), `L_hidden` (hidden layers), neuron counts, and weights as matrices.

```bash
python tools/backprop_debug_v2.py
```

All sigmoid activation throughout. Prints every forward + backward + update value.

---

## `backprop_step_calculator.py` — Step-by-Step Calculator

**Best for:** Walking through backprop one variable at a time with clean terminal output.

Supports both `mse_sigmoid` and `cross_entropy_sigmoid` loss types.

```bash
python tools/backprop_step_calculator.py
```

---

## `mlp_trainer_with_gradient_check.py` — Training + Gradient Check

**Best for:** Verifying that a backprop implementation is correct by comparing analytical gradients (from backprop) against numerical gradients (finite differences).

A mismatch between the two reveals bugs in the backprop implementation.

```bash
python tools/mlp_trainer_with_gradient_check.py
```

---

## Notation Reference (Lecture → Code Mapping)

| Lecture symbol | Code variable | Meaning |
|---------------|---------------|---------|
| `y^(0)` | `x` or `a[0]` | Input vector |
| `v^(l)` | `z[l-1]` | Pre-activation (net input) |
| `y^(l)` | `a[l]` | Post-activation output |
| `δ^(l)` | `delta[l-1]` | Error signal (delta) |
| `w^(l)` | `W[l-1]` | Weight matrix to layer l |
| `η` | `lr` | Learning rate |
| `t` | `t` | Target / desired output |

---

## Running in Headless Environments

```bash
# All tools honour MPLBACKEND to suppress windows
export MPLBACKEND=Agg
python tools/backprop_debugger.py
```
