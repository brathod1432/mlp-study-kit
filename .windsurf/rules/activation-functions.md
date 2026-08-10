# mlp-study-kit — Activation Functions Reference

## 1. Purpose of Activation Functions

A neural network without activation functions is nothing more than a linear transformation, no matter how many layers it has. Stack two linear layers and the result is still one linear layer: `W2 @ (W1 @ x + b1) + b2 = W_eff @ x + b_eff`. This is called **linear collapse** and it means the network cannot learn any non-linear relationship in the data.

Activation functions introduce non-linearity after each linear combination, giving the network the expressive power to represent arbitrarily complex functions. This is formalised by the **Universal Approximation Theorem**: a single hidden layer with a non-linear activation and enough units can approximate any continuous function on a compact domain to arbitrary precision. In practice, deeper networks with moderate width generalise better, but the theoretical foundation relies entirely on the non-linearity provided by activations.

Without non-linear activations:
- All hidden layers collapse to a single equivalent linear map.
- The network can only model linear input-output relationships.
- Adding more layers provides zero additional expressive power.

## 2. Neuron Computation

Each neuron in a dense layer performs two sequential operations:

**Step 1 — Linear combination (activation potential):**

```
v = W @ y_prev + b
```

- `W` is the weight matrix for this layer (shape: `[n_out, n_in]`)
- `y_prev` is the output vector from the previous layer (or the input `x` for layer 1)
- `b` is the bias vector (shape: `[n_out]`), when `bias=True`
- `v` is the **activation potential** — the raw pre-activation value

**Step 2 — Non-linear activation:**

```
y = f(v)
```

- `f` is the activation function chosen for this layer
- `y` is the **output** of the neuron / layer

**Layer dictionary keys used in this codebase:**

| Key | Contents |
|-----|----------|
| `layer["activation_potential"]` | The `v` vector computed in Step 1 |
| `layer["output"]` | The `y` vector computed in Step 2 |
| `layer["delta"]` | The `δ` error signal, filled during backprop |
| `layer["weights"]` | The `W` matrix for this layer |
| `layer["activation_function"]` | String name, e.g. `"relu"`, `"tanh"` |
| `layer["bias"]` | Boolean flag |

The `layer` dict is the single source of truth. `forward_propagate()` fills `activation_potential` and `output`; `backward_propagate()` reads both and writes `delta`.

## 3. Full Reference — All 7 Activations in nn_core/activations.py

### 3.1 Linear

```
f(v) = v
f'(v) = 1
```

- Range: `(-∞, +∞)`
- The identity function — passes the activation potential through unchanged.
- Derivative is always 1, so gradients flow without scaling.
- **Use for:** regression output layers where the target is unbounded. Never use in hidden layers (linear collapse).
- No saturation, no vanishing gradient issues.

### 3.2 Sigmoid (Logistic)

```
f(v)  = 1 / (1 + exp(-v))
f'(v) = y * (1 - y)       where y = f(v)
```

- Range: `(0, 1)` — strictly between 0 and 1, never reaches the bounds.
- Output can be interpreted as a probability.
- Derivative is computed from the **output** `y`, not from `v` directly — this is why `layer["output"]` must be set before calling the derivative.
- **Use for:** binary classification output layers; maps any real number to a probability.
- **Vanishing gradient warning:** for large `|v|` the sigmoid saturates (output ≈ 0 or ≈ 1), the derivative `y(1-y)` approaches 0, and gradients in earlier layers shrink exponentially. Avoid in hidden layers of deep networks.

### 3.3 Tanh (Hyperbolic Tangent)

```
f(v)  = (exp(v) - exp(-v)) / (exp(v) + exp(-v))
f'(v) = 1 - y^2            where y = f(v)
```

- Range: `(-1, 1)`
- **Zero-centred** — outputs are symmetric around 0. This property helps gradient flow compared to sigmoid, whose outputs are always positive.
- Derivative is also computed from the output `y`.
- Maximum derivative is 1.0 (at `v=0`), vs sigmoid's maximum of 0.25 — stronger gradients in the centre.
- **Use for:** hidden layers when you want a bounded, zero-centred activation. Preferable to sigmoid for hidden layers.
- Still saturates for large `|v|` — vanishing gradient persists in very deep networks or with large weights.

### 3.4 ReLU (Rectified Linear Unit)

```
f(v)  = max(0, v)
f'(v) = 1   if v >= 0
        0   if v < 0
```

- Range: `[0, +∞)`
- Computationally cheap — just a threshold operation.
- Does not saturate on the positive side, so gradients remain large for active neurons.
- **Dying ReLU problem:** if a neuron's activation potential is always negative (e.g. due to large negative bias or bad initialisation), the output is always 0 and the gradient is always 0. That neuron never updates — it is "dead". Mitigation: good weight init (He init), small learning rates, or switch to Leaky ReLU.
- **Use for:** hidden layers in most feed-forward networks; default choice for deep networks.

### 3.5 Leaky ReLU

```
f(v)  = v           if v >= 0
        alpha * v   if v < 0      (default alpha = 0.01)

f'(v) = 1           if v >= 0
        alpha       if v < 0
```

- Range: `(-∞, +∞)`
- The small slope `alpha` in the negative region prevents neurons from dying completely — a small gradient always flows.
- `alpha` is a hyperparameter; `0.01` is the conventional default. Larger values (e.g. `0.1`) allow more signal from negative activations.
- **Use for:** hidden layers when dying ReLU is a known problem; drop-in replacement for ReLU.

### 3.6 ELU (Exponential Linear Unit)

```
f(v)  = v                   if v >= 0
        alpha * (exp(v) - 1) if v < 0   (default alpha = 1.0)

f'(v) = 1                   if v >= 0
        alpha * exp(v)       if v < 0
        (equivalently: f(v) + alpha  for v < 0)
```

- Range: `(-alpha, +∞)` — negative outputs are bounded below by `-alpha`.
- **Smooth at 0:** unlike ReLU and Leaky ReLU, ELU has a smooth (differentiable) curve through the origin when `alpha=1.0`, which can improve optimisation.
- Mean activations pushed toward 0 (negative saturation at `-alpha`), reducing internal covariate shift without batch normalisation.
- Slightly more expensive than ReLU due to the `exp` call.
- **Use for:** hidden layers when you want smooth gradients and near-zero mean activations; a step up from Leaky ReLU in expressivity.

## 4. When to Use Which Activation

### Hidden Layers

| Situation | Recommended |
|-----------|-------------|
| General default, deep network | ReLU |
| Dead neurons are a problem | Leaky ReLU |
| Want smooth, zero-mean activations | ELU |
| Shallow network, stable training | Tanh |
| Legacy / interpretability needed | Sigmoid (avoid for deep) |
| Never in hidden layers | Linear |

### Output Layers

| Task | Activation | Loss |
|------|------------|------|
| Regression (unbounded target) | Linear | MSE |
| Binary classification | Sigmoid | BCE |
| Multi-class (not in this kit) | Softmax | Cross-entropy |

**Rule of thumb:** match the output activation to the output range. If targets are in `(-∞, +∞)`, use Linear. If targets are probabilities `(0, 1)`, use Sigmoid.

## 5. How to Call from Code

```python
from nn_core.activations import ActivationFn

af = ActivationFn()

# Prepare a layer dict (as network.py does internally)
import numpy as np
v_arr = np.array([1.0, -0.5, 0.3])
layer = {
    "activation_potential": v_arr,
    "output": None,
}

# Forward pass — compute output
y = af.output(layer, "relu")
layer["output"] = y          # must store so derivative can read it
print(y)                     # [1.  0.  0.3]

# Derivative pass — f'(v), uses layer["output"] internally
dy = af.output(layer, "relu", derivative=True)
print(dy)                    # [1. 0. 1.]

# Works for all activation names:
#   "linear", "sigmoid", "tanh", "relu", "leaky_relu", "elu"
```

Key design detail: the derivative for sigmoid and tanh is expressed in terms of `y` (the output), not `v`. This is why `layer["output"]` must be populated before calling `derivative=True`. The `ActivationFn.output()` method reads `layer["output"]` when computing the derivative for those functions.

## 6. Key Files

| File | Role |
|------|------|
| `nn_core/activations.py` | **Canonical implementation** — `ActivationFn` class, all 7 activations, forward + derivative |
| `exercises/ex06_neuron_basics.py` | Standalone (no nn_core imports) — activations defined as plain functions, used with a dict neuron |
| `exercises/ex07_forward_pass.py` | OOP version — `ActivationFn` first appears as a class inside the exercise |
| `exercises/ex08_derivatives.py` | Derivative flag added to the standalone version |
| `exercises/ex09_full_backprop.py` | First complete use of activations in a training loop |
| `exercises/ex10_bias_early_stop.py` | Adds bias support; uses tanh hidden + linear output for regression |
| `modules/plot_utils.py` | `plot_activations()` — plots all activation curves side by side |
| `tests/test_activations.py` | 19 unit tests covering forward and derivative for all functions |
