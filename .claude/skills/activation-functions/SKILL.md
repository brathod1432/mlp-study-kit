# Claude Skill: Activation Functions â€” Theory, Math & Implementation

## When Claude should use this skill
Invoke when working on any of the following in mlp-study-kit:
- `nn_core/activations.py` â€” canonical ActivationFn class
- `exercises/ex06_neuron_basics.py` through `ex10_bias_early_stop.py`
- `experiments/20251224_v2.py` (HW2 extensions)
- Any question about which activation to use, what a derivative is, or why a network diverges
- Adding a new activation function

---

## 1. Why Activation Functions Exist

A neural network without activation functions is just a chain of matrix multiplications:

```
y = W_L * (W_{L-1} * ... (W_1 * x)...)
  = (W_L * W_{L-1} * ... * W_1) * x
  = W_combined * x
```

No matter how many layers you stack, the result is still **linear** â€” equivalent to a
single-layer network. To approximate non-linear functions (sin, XOR, image features),
every hidden layer must apply a non-linear transformation called an **activation function**.

**Universal Approximation Theorem**: A network with one hidden layer of sufficient width
and a non-linear activation can approximate *any* continuous function on a compact domain.
In practice, deeper + narrower networks learn better representations.

---

## 2. The Neuron Computation

Given a layer with N neurons:

```
v_j = sum_i (w_ji * y_i) + b_j      # activation potential (pre-activation)
y_j = f(v_j)                          # output after activation function f
```

Where:
- `w_ji`  = weight from neuron i (previous layer) to neuron j (this layer)
- `y_i`   = output of previous layer neuron i
- `b_j`   = bias term for neuron j
- `v_j`   = activation potential (also called net input, induced local field)
- `y_j`   = output of neuron j (also called activation, signal)
- `f`     = activation function

In this codebase, the layer dict carries both:
```python
layer["activation_potential"]  # = v  (set during forward_propagate)
layer["output"]                # = y = f(v)  (set during forward_propagate)
```

---

## 3. Activation Functions â€” Complete Reference

### 3.1 Linear (Identity)

**Formula:**
```
f(v) = v
f'(v) = 1   (derivative is constant 1 everywhere)
```

**Behavior:** Output equals input. No saturation. No non-linearity.

**Numerical properties:**
- No vanishing gradient â€” gradient stays exactly 1
- But no non-linearity either, so stacking linear layers = one linear layer

**Use cases:**
- Output layer for **regression** tasks (unbounded output)
- Rarely used in hidden layers (kills expressiveness)

**In exercises:**
- ex06 output layer: `neuron_linear(neuron) = neuron['activation_potential']`
- ex07â€“ex10 hidden layers when testing basic propagation

**Code (nn_core/activations.py):**
```python
@staticmethod
def _linear(layer: dict) -> np.ndarray:
    return np.asarray(layer["activation_potential"], dtype=float)

@staticmethod
def _d_linear(layer: dict) -> np.ndarray:
    return np.ones_like(np.asarray(layer["activation_potential"], dtype=float))
```

---

### 3.2 Sigmoid (Logistic)

**Formula:**
```
f(v) = 1 / (1 + exp(-v))

f'(v) = f(v) * (1 - f(v))     # expressed using the output y=f(v), NOT v
```

**Derivation of derivative:**
```
f'(v) = exp(-v) / (1 + exp(-v))^2
      = [1/(1+exp(-v))] * [exp(-v)/(1+exp(-v))]
      = f(v) * (1 - f(v))
```

This is why the code stores `layer["output"]` and uses it in the derivative, not `v`:
```python
def _d_logistic(layer: dict) -> np.ndarray:
    y = np.asarray(layer["output"], dtype=float)
    return y * (1.0 - y)   # must have called forward first to populate output
```

**Behavior:**
```
v = -inf  -->  f(v) = 0
v =  0    -->  f(v) = 0.5
v = +inf  -->  f(v) = 1
```
- Squashes any real number into (0, 1)
- Symmetric around (0, 0.5)
- Derivative peaks at v=0: f'(0) = 0.5 * 0.5 = 0.25

**Vanishing gradient problem:**
For |v| > 4, f'(v) < 0.02. In a deep network, multiplying many such small derivatives
causes gradients to shrink exponentially toward the first layers â†’ weights near the input
barely update â†’ network trains very slowly.

**Numerical stability:**
- For very negative v: `exp(-v)` overflows. Use: `1 / (1 + exp(-v))` directly (NumPy handles this correctly with float64)
- For very positive v: `exp(-v)` underflows to 0, giving `f(v) â‰ˆ 1` correctly

**Use cases:**
- Output layer for **binary classification** (output is probability in (0,1))
- Gates in LSTM/GRU (not used in this codebase)
- Avoid in hidden layers due to vanishing gradients

**Code call:**
```python
af = ActivationFn()
layer = {"activation_potential": np.array([0.0, 2.0, -2.0]), "output": None}
y  = af.output(layer, "sigmoid")         # forward:  [0.5, 0.88, 0.12]
layer["output"] = y
dy = af.output(layer, "sigmoid", derivative=True)  # [0.25, 0.105, 0.105]
```

---

### 3.3 Tanh (Hyperbolic Tangent)

**Formula:**
```
f(v) = (exp(v) - exp(-v)) / (exp(v) + exp(-v))
     = 2 * sigmoid(2v) - 1       # relationship to sigmoid

f'(v) = 1 - f(v)^2               # expressed using output y=f(v)
```

**Derivation:**
```
f(v) = tanh(v)
f'(v) = 1 - tanh(v)^2 = 1 - f(v)^2
```

**Behavior:**
```
v = -inf  -->  f(v) = -1
v =  0    -->  f(v) =  0
v = +inf  -->  f(v) = +1
```
- Squashes any real number into (-1, 1)
- Zero-centered: output mean â‰ˆ 0 â†’ helps next layer (no bias shift)
- Derivative peaks at v=0: f'(0) = 1.0

**Tanh vs Sigmoid:**
- Tanh output range (-1, 1) vs sigmoid (0, 1)
- Tanh is zero-centered â€” preferred for hidden layers over sigmoid
- Tanh derivative is 4x sigmoid's at v=0 (less vanishing gradient)
- But still has vanishing gradient problem for |v| > 2

**Code (ex06 hand-coded version):**
```python
def neuron_tanh(neuron):
    v = neuron['activation_potential']
    return (np.exp(v) - np.exp(-v)) / (np.exp(v) + np.exp(-v))
```

**Code (nn_core/activations.py â€” numerically stable via np.tanh):**
```python
@staticmethod
def _tanh(layer: dict) -> np.ndarray:
    v = np.asarray(layer["activation_potential"], dtype=float)
    ep, em = np.exp(v), np.exp(-v)
    return (ep - em) / (ep + em)

@staticmethod
def _d_tanh(layer: dict) -> np.ndarray:
    y = np.asarray(layer["output"], dtype=float)
    return 1.0 - np.power(y, 2)
```

**Use cases:**
- Hidden layers in regression networks
- RNN/LSTM hidden state (not in this codebase)
- Any hidden layer where you want zero-centered outputs

---

### 3.4 ReLU (Rectified Linear Unit)

**Formula:**
```
f(v) = max(0, v)

f'(v) = { 1   if v >= 0
         { 0   if v < 0
```

Note: f'(0) is undefined mathematically, but set to 1 (or 0) by convention â€” doesn't matter in practice.

**Behavior:**
```
v < 0  -->  f(v) = 0  (dead zone)
v >= 0 -->  f(v) = v  (identity)
```

**Advantages over sigmoid/tanh:**
1. No vanishing gradient for positive inputs (derivative = 1)
2. Computationally cheap: just `max(0, v)`, no exp()
3. Sparse activation: ~50% of neurons output 0 in a random init â†’ efficient
4. Faster convergence in practice

**Dying ReLU problem:**
If a neuron's weights are updated such that its `v` is always negative, its gradient
is permanently 0 â†’ the neuron never recovers ("dies"). Typically 10â€“30% of ReLU
neurons die in deep networks. Solutions: Leaky ReLU, careful weight init (Glorot/He).

**Code:**
```python
@staticmethod
def _relu(layer: dict) -> np.ndarray:
    return np.maximum(0.0, np.asarray(layer["activation_potential"], dtype=float))

@staticmethod
def _d_relu(layer: dict) -> np.ndarray:
    v = np.asarray(layer["activation_potential"], dtype=float)
    return (v >= 0.0).astype(float)
```

**Use cases:**
- Default choice for **hidden layers** in deep networks
- NOT for output layer (unless regression with positive outputs only)
- Combine with He initialization (Glorot Normal with factor 2 instead of 1)

---

### 3.5 Leaky ReLU

**Formula:**
```
f(v) = { v          if v >= 0
        { alpha * v  if v < 0

f'(v) = { 1      if v >= 0
         { alpha  if v < 0

Default alpha = 0.01
```

**Why it fixes Dying ReLU:**
Negative inputs produce a small non-zero gradient (alpha instead of 0), so the
neuron can always recover. Even if `v` is very negative, the gradient is still
`alpha` (0.01), so the weights can still update.

**Behavior:**
```
v = -10  -->  f(v) = -0.1   (not dead!)
v =  0   -->  f(v) =  0
v = +10  -->  f(v) = 10
```

**Alpha parameter:**
- `alpha = 0.01` is the conventional default (Leaky ReLU)
- `alpha = 0.2` is common in generative models (DCGAN)
- Parametric ReLU (PReLU): alpha is learned via backpropagation (not implemented here)

**Code:**
```python
@staticmethod
def _leaky_relu(layer: dict, alpha: float = 0.01) -> np.ndarray:
    v = np.asarray(layer["activation_potential"], dtype=float)
    return np.where(v >= 0.0, v, alpha * v)

@staticmethod
def _d_leaky_relu(layer: dict, alpha: float = 0.01) -> np.ndarray:
    v = np.asarray(layer["activation_potential"], dtype=float)
    return np.where(v >= 0.0, 1.0, alpha).astype(float)
```

**Usage with custom alpha:**
```python
layer = {"activation_potential": v_arr, "output": None}
y  = af.output(layer, "leaky_relu", alpha=0.1)     # forward with alpha=0.1
dy = af.output(layer, "leaky_relu", derivative=True, alpha=0.1)
```

**Use cases:**
- Drop-in replacement for ReLU when dying neurons are a problem
- Image generation networks
- Added in `experiments/20251224_v2.py` (HW2 extension)

---

### 3.6 ELU (Exponential Linear Unit)

**Formula:**
```
f(v) = { v                     if v >= 0
        { alpha * (exp(v) - 1)  if v < 0

f'(v) = { 1                    if v >= 0
         { alpha * exp(v)       if v < 0
         = f(v) + alpha         for v < 0   (using output y=f(v))
```

Default alpha = 1.0

**Behavior:**
```
v = -inf  -->  f(v) = -alpha  (saturates to -alpha, not 0)
v =  0    -->  f(v) =  0      (continuous at 0)
v = +inf  -->  f(v) = +inf    (linear, no saturation)
```

**Advantages over Leaky ReLU:**
- Smoother at v=0 (f is continuous AND f' is continuous)
- Negative saturation at -alpha provides noise robustness
- Mean activations closer to 0 â†’ faster convergence
- Theoretically better gradient flow than Leaky ReLU

**Cost:** `exp(v)` is slower than the simple comparison in Leaky ReLU.

**Code:**
```python
@staticmethod
def _elu(layer: dict, alpha: float = 1.0) -> np.ndarray:
    v = np.asarray(layer["activation_potential"], dtype=float)
    return np.where(v >= 0.0, v, alpha * (np.exp(v) - 1.0))

@staticmethod
def _d_elu(layer: dict, alpha: float = 1.0) -> np.ndarray:
    v = np.asarray(layer["activation_potential"], dtype=float)
    return np.where(v >= 0.0, 1.0, alpha * np.exp(v)).astype(float)
```

**Use cases:**
- When Leaky ReLU is not sufficient (noisy data, very deep networks)
- Added in `experiments/20251224_v2.py` (HW2 extension)

---

## 4. Choosing the Right Activation

| Layer type | Recommended | Why |
|-----------|-------------|-----|
| Hidden (general) | ReLU | Fast, no vanishing gradient |
| Hidden (dying problem) | Leaky ReLU | Keeps gradients alive |
| Hidden (smooth needed) | ELU or Tanh | Continuous derivative |
| Output â€” regression | Linear | Unbounded output |
| Output â€” binary classification | Sigmoid | Output in (0,1) |
| Output â€” multi-class | Softmax* | Probabilities sum to 1 |

*Softmax is not implemented in this codebase â€” add it to `nn_core/activations.py` if needed.

---

## 5. Saturation and Vanishing Gradient Summary

| Function | Saturates? | Gradient range | Dead neurons? |
|----------|-----------|----------------|---------------|
| Linear | No | Always 1 | No |
| Sigmoid | Yes (both ends) | (0, 0.25] | No |
| Tanh | Yes (both ends) | (0, 1] | No |
| ReLU | Yes (negative) | {0, 1} | Yes (~20%) |
| Leaky ReLU | No | {alpha, 1} | No |
| ELU | Partial (neg only) | (0, inf) | No |

**Vanishing gradient** occurs when gradients become very small (< 1e-4) after
many layers â€” early layers stop learning. Sigmoid and Tanh are worst for this.
ReLU largely solves it for positive activations. Leaky/ELU fully solve it.

---

## 6. Adding a New Activation to nn_core

1. Add forward method `_myact(layer)` and derivative `_d_myact(layer)` as `@staticmethod` in `ActivationFn`
2. Register in `self._dispatch` dict in `__init__`
3. If the function has a parameter (like alpha), follow the `_leaky_relu(layer, alpha)` pattern
4. Add to the `if key in (...)` branch in `output()` if alpha-parameterized
5. Add tests in `tests/test_activations.py`
6. Export in `nn_core/__init__.py` if needed

---

## 7. Key Code Locations

| Component | File | Method |
|-----------|------|--------|
| All activations (forward + derivative) | `nn_core/activations.py` | `ActivationFn` class |
| Forward propagation using activations | `nn_core/network.py` | `forward_propagate()` |
| Backward propagation using derivatives | `nn_core/network.py` | `backward_propagate()` |
| Standalone ex06 implementations | `exercises/ex06_neuron_basics.py` | `neuron_linear`, `neuron_tanh`, `neuron_relu` |
| OOP ex07â€“ex10 implementations | `exercises/ex07_forward_pass.py` â€¦ | `Activation_fcn` class |
| HW2 Leaky ReLU + ELU | `experiments/20251224_v2.py` | `Activation_fcn` class |
| Activation visualisation | `modules/plot_utils.py` | `plot_activations()` |
| Test coverage | `tests/test_activations.py` | 19 tests |

