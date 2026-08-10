# Claude Skill: Backpropagation â€” Complete Theory & Implementation

## When Claude should use this skill
Invoke when working on:
- `nn_core/network.py` â€” `backward_propagate()`, `update_weights()`, `train()`
- `tools/backprop_debugger.py`, `backprop_debug.py`, `backprop_step_calculator.py`
- `exercises/ex08_derivatives.py`, `ex09_full_backprop.py`, `ex10_bias_early_stop.py`
- Debugging: loss not decreasing, NaN loss, gradients exploding
- Explaining deltas, weight updates, or the chain rule

---

## 1. The Core Problem: How to Adjust Weights?

A neural network learns by adjusting its weights `W` to minimize the loss `E`. We need
to know: *for each weight w_ji, how much does E change if we slightly increase w_ji?*

This is the **gradient**: `dE/dw_ji`

If `dE/dw_ji > 0`: increasing w_ji increases E â†’ decrease w_ji
If `dE/dw_ji < 0`: increasing w_ji decreases E â†’ increase w_ji
If `dE/dw_ji = 0`: w_ji is at a local minimum â†’ leave it alone

**Weight update rule (Gradient Descent / SGD):**
```
w_ji(new) = w_ji(old) - eta * dE/dw_ji
```
Where `eta` (Î·) is the learning rate â€” controls step size.

The problem: computing `dE/dw_ji` for each weight directly is expensive.
Backpropagation computes ALL gradients in a single backward pass using the **chain rule**.

---

## 2. The Chain Rule (Foundation of Backpropagation)

For composed functions `E = f(g(h(x)))`:
```
dE/dx = (dE/df) * (df/dg) * (dg/dh) * (dh/dx)
```

In a neural network, the computation graph is:
```
x â†’ [W1] â†’ v1 â†’ [f1] â†’ y1 â†’ [W2] â†’ v2 â†’ [f2] â†’ y2 â†’ [Loss] â†’ E
```

To compute `dE/dW1`, we must chain through every operation from E back to W1:
```
dE/dW1 = (dE/dy2) * (dy2/dv2) * (dv2/dy1) * (dy1/dv1) * (dv1/dW1)
```

Each term is one part of the chain:
- `dE/dy2`  = loss derivative (from loss function)
- `dy2/dv2` = activation derivative at layer 2 (from activation function)
- `dv2/dy1` = weights W2 (since v2 = W2 @ y1, so dv2/dy1 = W2)
- `dy1/dv1` = activation derivative at layer 1
- `dv1/dW1` = y0 (input, since v1 = W1 @ y0)

Backpropagation organizes this computation efficiently by reusing intermediate values.

---

## 3. Notation (Lecture Convention Used in This Codebase)

```
y^(0)  = input vector x (layer 0 output = input)
v^(l)  = activation potential (net input) of layer l
       = W^(l) @ y^(l-1) + b^(l)
y^(l)  = output of layer l = f^(l)(v^(l))
t      = target output vector
E      = loss(t, y^(L))   where L = output layer
eta    = learning rate Î·
delta^(l) = error signal for layer l
```

**In code (nn_core/network.py), layer dict keys:**
```python
layer["activation_potential"]  # = v^(l)
layer["output"]                # = y^(l)
layer["delta"]                 # = delta^(l)
layer["weights"]               # = W^(l)
```

**In tools/backprop_debugger.py (DebugMLP):**
```python
cache["z"][l]   # = v^(l+1) in lecture notation (z is pre-activation)
cache["a"][l]   # = y^(l)   in lecture notation (a is activation)
deltas[l]       # = delta^(l+1) in lecture notation
```

---

## 4. Forward Pass

For each layer l = 1, 2, ..., L (1=first hidden, L=output):

```
v^(l) = W^(l) @ y^(l-1)   [+ b^(l) if bias=True]
y^(l) = f^(l)(v^(l))
```

**Code (nn_core/network.py â€” forward_propagate):**
```python
def forward_propagate(self, nnetwork, inputs):
    inp = np.asarray(inputs, dtype=float).flatten()
    for i in range(1, len(nnetwork)):
        layer = nnetwork[i]
        if layer["bias"]:
            inp = np.append(inp, 1.0)          # append bias neuron
        # v^(l) = W^(l) @ y^(l-1)
        layer["activation_potential"] = np.matmul(layer["weights"], inp).flatten()
        # y^(l) = f^(l)(v^(l))
        layer["output"] = self.af.output(layer, layer["activation_function"])
        inp = layer["output"]                  # y^(l) becomes next layer's input
    return inp
```

---

## 5. Loss Computation

After forward pass, compute scalar loss:
```
E = loss(t, y^(L))     # one value per output neuron, summed
```

In practice (training loop in `train()`):
```python
err_sum += float(np.sum(self.loss.output(loss_function, y_arr, nnetwork[-1]["output"])))
```

---

## 6. Backward Pass â€” Delta Computation

The **delta** Î´^(l) for layer l is defined as:
```
delta^(l) = dE/dv^(l)   (gradient of loss w.r.t. activation potential at layer l)
```

Deltas are computed from the OUTPUT layer backwards to the FIRST hidden layer.

### Output layer delta (l = L):

```
delta^(L) = (dE/dy^(L)) * f'^(L)(v^(L))
```

Where:
- `dE/dy^(L)` is the loss function derivative
- `f'^(L)(v^(L))` is the output activation derivative
- The multiplication is element-wise

**Special case â€” sigmoid output with BCE loss:**
```
delta^(L) = y^(L) - t   (simplified â€” see loss-functions skill)
```

### Hidden layer delta (l < L):

```
delta^(l) = (W^(l+1).T @ delta^(l+1)) * f'^(l)(v^(l))
```

Where:
- `W^(l+1).T @ delta^(l+1)` propagates the error signal backward through weights
- `f'^(l)(v^(l))` "gates" the error by how much the neuron was changing at that point
- The `@` is matrix multiply; the `*` is element-wise multiply

**Why W.T?** Because in the forward pass, v^(l+1) = W^(l+1) @ y^(l). By the chain rule,
`dy^(l) / dv^(l+1)` = W^(l+1). Transposing distributes the error back proportionally
to each weight's contribution.

**Code (nn_core/network.py â€” backward_propagate):**
```python
def backward_propagate(self, loss_function, nnetwork, expected):
    N = len(nnetwork) - 1  # index of output layer
    for i in range(N, 0, -1):
        if i < N:
            # Hidden layer: propagate delta backward through weights
            w = nnetwork[i + 1]["weights"]
            if nnetwork[i + 1]["bias"]:
                w = w[:, :-1]              # strip bias column (no delta for bias neuron)
            errors = np.matmul(nnetwork[i + 1]["delta"], w)   # W^(l+1).T @ delta^(l+1)
        else:
            # Output layer: loss derivative
            errors = self.loss.output(loss_function, expected,
                                      nnetwork[-1]["output"], derivative=True)
        # Element-wise multiply by activation derivative
        nnetwork[i]["delta"] = np.multiply(
            errors,
            self.af.output(nnetwork[i], nnetwork[i]["activation_function"], derivative=True)
        )
```

---

## 7. Weight Update (SGD)

Once all deltas are computed, update each weight matrix:

```
W^(l)(new) = W^(l)(old) - eta * delta^(l) âŠ— y^(l-1)
```

Where `âŠ—` is the **outer product** â€” produces a matrix of shape (n_l, n_{l-1}):

```
outer(delta, prev_output)[j, i] = delta[j] * prev_output[i]
```

This is the gradient `dE/dw_ji = delta^(l)[j] * y^(l-1)[i]`, computed for ALL weights
simultaneously via the outer product.

**Code (nn_core/network.py â€” _update_weights_clipped):**
```python
def _update_weights_clipped(self, nnetwork, inputs, l_rate, grad_clip):
    inp = np.asarray(inputs, dtype=float).flatten()
    for i in range(1, len(nnetwork)):
        if nnetwork[i]["bias"]:
            inp = np.append(inp, 1.0)   # bias neuron has output=1 always
        delta = nnetwork[i]["delta"]
        # Optional gradient clipping (prevents exploding gradients)
        if grad_clip is not None:
            norm = float(np.linalg.norm(delta))
            if norm > grad_clip:
                delta = delta * (grad_clip / norm)
        # W = W - eta * outer(delta, prev_output)
        nnetwork[i]["weights"] -= l_rate * np.outer(delta, inp)
        inp = nnetwork[i]["output"]   # for next layer, inp = current layer's output
```

---

## 8. Complete Training Loop (One Epoch, One Sample)

```python
for x_row, y_row in zip(x_train, y_train):
    # Step 1: Forward pass â€” compute all v^(l) and y^(l)
    self.forward_propagate(nnetwork, x_row)

    # Step 2: Compute deltas â€” backward pass
    self.backward_propagate(loss_function, nnetwork, y_row)

    # Step 3: Update weights
    self._update_weights_clipped(nnetwork, x_row, l_rate, grad_clip)
```

This is **online (stochastic) gradient descent** â€” one weight update per sample.
For **batch gradient descent**, accumulate gradients over all samples first, then update once.

---

## 9. Diagnosing Training Problems

### Loss is NaN
```
Cause:    Exploding gradients (often with high learning rate)
Fix 1:    Reduce learning rate (try 10x smaller)
Fix 2:    Use gradient clipping: model.train(..., grad_clip=1.0)
Fix 3:    Check for NaN/inf in input data
Check:    train() raises RuntimeError("NaN") immediately now (nn_core/network.py)
```

### Loss not decreasing
```
Cause A:  Learning rate too small â†’ try 10x larger
Cause B:  Learning rate too large â†’ loss oscillates or increases â†’ try 10x smaller
Cause C:  Wrong activation for output layer (e.g., relu for BCE â†’ negative outputs)
Cause D:  Dying ReLU â†’ switch to Leaky ReLU
Cause E:  Not enough epochs
Cause F:  Poor weight initialisation â†’ check make_regression_data returns sane values
```

### Loss decreases then plateaus
```
Cause A:  Early stopping triggered too aggressively (reduce epsilon)
Cause B:  Network too small for the problem (add neurons/layers)
Cause C:  Learning rate too large at the plateau (use learning rate schedule)
```

### Good train loss, poor test loss (overfitting)
```
Cause:    Network too large for dataset
Fix:      More data, add early stopping (epsilon > 0), reduce network size
```

---

## 10. Gradient Flow Visualization

```
Input layer                 Hidden layer 1         Output layer
x â”€â”€W^(1)â”€â”€> v^(1) â†’ y^(1) â”€â”€W^(2)â”€â”€> v^(2) â†’ y^(2) â†’ E
              â†‘                  â†‘                  â†‘
             f'(v^(1))          f'(v^(2))          dE/dy^(2)
              |                  |                  |
         â†delta^(1)â†â”€â”€â”€â”€â”€â”€â†delta^(2)â†â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
          (W^(2).T @ delta^(2)) * f'(v^(1))

Weight gradient at W^(1): outer(delta^(1), x)
Weight gradient at W^(2): outer(delta^(2), y^(1))
```

---

## 11. Using the Debug Tools

### For exam-style one-step debugging (tools/backprop_debugger.py):
```python
# Edit EXAMPLE_CONFIG and run:
python tools/backprop_debugger.py

# Output shows (in lecture notation):
# a[0] = y^(0) = input
# z[l] = v^(l+1) = pre-activation
# a[l] = y^(l)   = post-activation
# delta[l] = Î´^(l+1)
# dW[l] = gradient of W^(l+1)
# Delta_W[l] = lr * dW[l] = actual weight change
# Loss before and after (sanity check: must decrease)
```

### For generic layer sizes (tools/backprop_debug_v2.py):
- Configure D (inputs), hidden layers, K (outputs), weights as matrices
- Runs full one-step forward + backward + update with debug logs

### For step-by-step calculation (tools/backprop_step_calculator.py):
- Prints every intermediate value labeled by lecture notation

---

## 12. Key Code Locations

| Component | File | Method |
|-----------|------|--------|
| Forward pass | `nn_core/network.py` | `forward_propagate()` |
| Backward pass (deltas) | `nn_core/network.py` | `backward_propagate()` |
| Weight update + gradient clipping | `nn_core/network.py` | `_update_weights_clipped()` |
| Complete training loop | `nn_core/network.py` | `train()` |
| Exam-prep debugger | `tools/backprop_debugger.py` | `DebugMLP.one_iteration_debug()` |
| Gradient check | `tools/mlp_trainer_with_gradient_check.py` | numerical vs analytical gradients |
| Stage 3 (stubs) | `exercises/ex08_derivatives.py` | `backward_propagate()` (empty) |
| Stage 4 (complete) | `exercises/ex09_full_backprop.py` | full implementation |
| Stage 5 (bias + early stop) | `exercises/ex10_bias_early_stop.py` | bias-aware backprop |

