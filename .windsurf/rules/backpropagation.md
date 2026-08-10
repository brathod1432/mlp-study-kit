# mlp-study-kit — Backpropagation Reference

## 1. Why Backpropagation

Training a neural network means finding weights `W` that minimise a loss `E`. To use gradient descent we need `dE/dW` — the gradient of the loss with respect to every weight in every layer. A network with even a few hundred weights makes finite-difference numerical gradients prohibitively slow (one forward pass per weight).

**Backpropagation** (backprop) applies the chain rule of calculus in a single backward pass to compute `dE/dW` for all weights simultaneously. The key insight: the error at each layer can be expressed in terms of the error at the layer above it, propagated backward through the weights. This gives an O(W) algorithm instead of O(W²).

Without backprop: gradient computation is either intractable or approximate.
With backprop: exact gradients for every weight in two passes (one forward, one backward).

## 2. Notation Used in This Codebase

| Symbol | Meaning | Layer dict key |
|--------|---------|---------------|
| `y^(0)` | Network input — the data vector `x` | (input layer `"output"`) |
| `v^(l)` | Activation potential at layer `l`: `W^(l) @ y^(l-1) + b` | `layer["activation_potential"]` |
| `y^(l)` | Output of layer `l`: `f^(l)(v^(l))` | `layer["output"]` |
| `δ^(l)` | Delta — error signal at layer `l`: `dE/dv^(l)` | `layer["delta"]` |
| `W^(l)` | Weight matrix for layer `l` (shape: `[n_l, n_{l-1}]`) | `layer["weights"]` |
| `f^(l)` | Activation function for layer `l` | `layer["activation_function"]` |
| `η` | Learning rate (scalar hyperparameter) | `l_rate` argument to `train()` |
| `L` | Index of the output (last) layer | |
| `E` | Scalar loss value for one sample | |

**Index convention:** layers are numbered `l = 1, 2, …, L` where `l=1` is the first hidden layer and `l=L` is the output layer. The input is `y^(0) = x`.

## 3. Forward Pass

For each layer `l` from 1 to L:

```
v^(l) = W^(l) @ y^(l-1)          # linear combination (+ bias absorbed into W or added separately)
y^(l) = f^(l)( v^(l) )           # non-linear activation
```

After the forward pass, every `layer["activation_potential"]` and `layer["output"]` is populated. The loss is computed from `y^(L)` (the final layer output) and the target `t`:

```
E = loss( t, y^(L) )
```

In `nn_core/network.py` this is `forward_propagate(x)`. It iterates through the `net` list of layer dicts and fills both keys.

## 4. Output Layer Delta — δ^(L)

The output layer delta is the product of two terms:

```
δ^(L) = (dE / dy^(L)) * f'^(L)( v^(L) )
```

- `dE / dy^(L)` comes from the loss function derivative (e.g. `y - t` for MSE, or the BCE gradient for binary cross-entropy).
- `f'^(L)(v^(L))` is the derivative of the output activation evaluated at `v^(L)`.

**Special combined case — Sigmoid + BCE:**
When you pair a sigmoid output with binary cross-entropy loss, the combined gradient simplifies beautifully:

```
δ^(L) = y^(L) - t
```

This clean form avoids vanishing gradients that arise from the sigmoid derivative in the saturated region. It is one of the primary reasons to prefer BCE over MSE for classification.

## 5. Hidden Layer Delta — δ^(l)

For each hidden layer `l` from `L-1` down to `1`:

```
δ^(l) = ( W^(l+1).T @ δ^(l+1) ) * f'^(l)( v^(l) )
```

Breaking this down:

**Term 1 — `W^(l+1).T @ δ^(l+1)` (error back-projection):**
The weights `W^(l+1)` connect layer `l` to layer `l+1`. Transposing routes the error signal from layer `l+1` back to layer `l`, distributing it proportionally to each weight's contribution to the error. Neurons with larger weights receive larger error signals.

**Term 2 — `f'^(l)(v^(l))` (local gradient gate):**
The activation derivative acts as a gate. If `f'` is near 0 (saturated sigmoid/tanh, or dead ReLU), the gradient is attenuated or zeroed — this is the vanishing / dying gradient problem.

**Element-wise multiply (`*`):**
Both terms are vectors of the same length `n_l`. The multiplication is element-wise, not a dot product.

## 6. Weight Update

Once all deltas are computed (backward pass complete), update every weight matrix:

```
W^(l) -= η * outer( δ^(l), y^(l-1) )
```

- `outer(δ^(l), y^(l-1))` is the outer product: a matrix of shape `[n_l, n_{l-1}]`.
- Entry `[i, j]` = `δ^(l)[i] * y^(l-1)[j]` — the gradient for weight `W[i,j]`.
- This simultaneously computes the gradient for every weight in the layer.
- `η` scales the step size.

**Bias update (when `bias=True`):**

```
b^(l) -= η * δ^(l)
```

The bias gradient is just `δ^(l)` itself (because `∂v/∂b = 1`).

## 7. Gradient Clipping

Exploding gradients occur when `δ` values grow very large — usually from a high learning rate, poor initialisation, or a badly scaled dataset. Clipping caps the gradient norm before the update:

```python
norm = np.linalg.norm(delta)
if norm > grad_clip:
    delta = delta * (grad_clip / norm)
```

This rescales the delta vector so its L2 norm is exactly `grad_clip`, preserving direction but bounding magnitude. In `nn_core/network.py`, this is applied inside `_update_weights_clipped()` before the outer product.

Recommended starting value: `grad_clip=1.0`. Set `grad_clip=None` to disable (default for well-behaved problems).

## 8. Code Flow in nn_core/network.py

### `forward_propagate(inp)`

```python
def forward_propagate(self, inp):
    current = inp
    for layer in net:
        if layer["type"] == "input":
            layer["output"] = current
            continue
        v = layer["weights"] @ current   # (+ bias if present)
        layer["activation_potential"] = v
        y = af.output(layer, layer["activation_function"])
        layer["output"] = y
        current = y
    return current   # y^(L)
```

Iterates forward through `net`, filling `activation_potential` and `output` at each dense layer.

### `backward_propagate(target, output, loss_fn)`

```python
def backward_propagate(self, target, output, loss_fn):
    # Output layer delta
    loss_grad = lf.output(loss_fn, target, output, derivative=True)
    f_prime = af.output(output_layer, act_fn, derivative=True)
    output_layer["delta"] = loss_grad * f_prime

    # Hidden layer deltas (reverse order)
    for l in reversed(range(1, L)):
        next_layer = net[l + 1]
        curr_layer = net[l]
        W_next = next_layer["weights"]
        delta_next = next_layer["delta"]
        f_prime = af.output(curr_layer, curr_layer["activation_function"], derivative=True)
        curr_layer["delta"] = (W_next.T @ delta_next) * f_prime
```

### `_update_weights_clipped(eta, grad_clip)`

```python
def _update_weights_clipped(self, eta, grad_clip):
    for l in range(1, L + 1):
        layer = net[l]
        delta = layer["delta"]
        if grad_clip is not None:
            norm = np.linalg.norm(delta)
            if norm > grad_clip:
                delta = delta * (grad_clip / norm)
        prev_output = net[l - 1]["output"]
        layer["weights"] -= eta * np.outer(delta, prev_output)
        if layer["bias"]:
            layer["bias_weights"] -= eta * delta
```

### `train(net, X, Y, ...)`

The main training loop:

```python
for epoch in range(n_epoch):
    for x, t in zip(X, Y):
        y_pred = self.forward_propagate(x)
        self.backward_propagate(t, y_pred, loss_function)
        self._update_weights_clipped(l_rate, grad_clip)
    # compute epoch loss, check early stopping, log history
```

Returns `(final_loss, history_train, history_test)` — always a 3-tuple.

## 9. Diagnosing Training Problems

### NaN Loss

**Cause:** Exploding gradients — weights and deltas grow without bound until numerical overflow.

**Fix:**
1. Add `grad_clip=1.0` to `train()`.
2. Reduce learning rate (try dividing by 10).
3. Check that inputs are normalised (zero mean, unit variance).
4. Verify weight initialisation isn't too large (default `* 0.2` is conservative).

### Loss Not Decreasing

**Cause 1 — Learning rate too small:** Gradients are valid but steps are tiny. Fix: increase `l_rate` by 10× and watch the first few epochs.

**Cause 2 — Wrong output activation:** Using ReLU on the output layer clips negative targets to 0. Fix: use `"linear"` for regression, `"sigmoid"` for binary classification.

**Cause 3 — Dying ReLU in hidden layers:** Many neurons permanently output 0. Fix: switch to `"leaky_relu"` or `"elu"`, or use He weight init.

**Cause 4 — Missing normalisation:** Features at wildly different scales cause the loss landscape to be poorly conditioned. Fix: normalise inputs with `modules/data_utils.normalize()`.

### Oscillating / Diverging Loss

**Cause:** Learning rate too large — overshooting the minimum.
**Fix:** Reduce `l_rate` by half or an order of magnitude. Watch the loss curve over the first 100 epochs.

### Overfitting (train loss << test loss)

**Cause:** Model memorises training data.
**Fix:**
1. Enable early stopping: `epsilon=1e-5` in `train()`.
2. Reduce network size (fewer units or fewer layers).
3. Add more training data or data augmentation.

## 10. Debug Tools

| File | Purpose |
|------|---------|
| `tools/backprop_debugger.py` | `DebugMLP` dataclass — runs one complete forward→backward→update cycle and prints every intermediate value in lecture notation: `z[l]`, `a[l]`, `δ[l]`, `ΔW[l]` |
| `tools/backprop_debug.py` | Compact version of the debugger, minimal output |
| `tools/backprop_debug_v2.py` | Refined compact helper with cleaner formatting |
| `tools/backprop_step_calculator.py` | `BackpropCalc` — interactive step-by-step calculator for hand-checking exam problems |
| `tools/mlp_trainer_with_gradient_check.py` | Full trainer with numerical gradient verification (finite differences vs backprop) |

To inspect a single training iteration in full detail:

```bash
python tools/backprop_debugger.py
```

This prints `v^(l)`, `y^(l)`, `δ^(l)`, and `ΔW^(l)` for every layer, matching the notation used in lecture slides.

## 11. Key Files

| File | Role |
|------|------|
| `nn_core/network.py` | Canonical implementation — `forward_propagate`, `backward_propagate`, `_update_weights_clipped`, `train` |
| `tools/backprop_debugger.py` | Step-by-step debug print for any network/input combination |
| `exercises/ex08_derivatives.py` | Derivative flag added to activations; `backward_propagate` and `update_weights` are left as stubs to complete |
| `exercises/ex09_full_backprop.py` | First complete end-to-end backprop + training loop; regression on `sin(2x)+cos(x)` |
| `exercises/ex10_bias_early_stop.py` | Adds bias term and early stopping; covers both regression and classification |
