# Codex Skill: Loss Functions â€” Theory, Math & Implementation

## When to apply this context
Invoke when working on:
- `nn_core/losses.py` â€” canonical LossFn class
- Choosing between MSE and BCE for a task
- Understanding why a loss is not decreasing
- Implementing a new loss function
- Interpreting `modules/metrics.py` functions

---

## 1. What is a Loss Function?

A loss function (also called cost function, objective function, error function) measures
how far the network's predictions are from the true targets. It produces a **single scalar
value** that:
1. Summarises the error across all output neurons
2. Is the function we minimize during training
3. Its gradient (w.r.t. the output) is the starting point of backpropagation

Key principle: **the loss function must be differentiable** with respect to the network's
output so we can compute gradients and update weights.

---

## 2. MSE â€” Mean Squared Error

### 2.1 Formula

For a single sample with output `y` (network output) and target `t`:
```
E = 0.5 * sum_j (t_j - y_j)^2
```

The `0.5` factor is a mathematical convenience â€” when we differentiate, the 2 from
the square cancels with the 0.5, giving a clean derivative.

For a training set of N samples, the average loss:
```
E_total = (1/N) * sum_n E_n
        = (1/N) * sum_n [ 0.5 * sum_j (t_nj - y_nj)^2 ]
```

### 2.2 Derivative w.r.t. output y_j

```
dE/dy_j = -(t_j - y_j)   =   y_j - t_j
```

The sign is positive because we're measuring loss w.r.t. the output `y`, not the target `t`.
When y > t (overestimate), gradient is positive â†’ weight update moves y down.
When y < t (underestimate), gradient is negative â†’ weight update moves y up.

### 2.3 Properties

**Convex** (for a linear output layer): has a unique global minimum.

**Sensitive to outliers**: because error is squared, a single prediction that is far
off (outlier) dominates the loss. For example, error=10 contributes 50 to MSE vs
error=1 contributing 0.5.

**Output scale matters**: MSE is not scale-invariant. If your targets are in [0,1]
vs [0,1000], the MSE values are incomparable. Always normalize targets when comparing.

**Best for:** regression tasks where outputs are continuous real values.

### 2.4 Code

```python
# nn_core/losses.py
@staticmethod
def _mse(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.5 * np.power(t - y, 2)      # element-wise, sum over outputs by caller

@staticmethod
def _d_mse(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -(t - y)    # = y - t
```

### 2.5 Usage in exercises

```python
# In training loop (ex09, ex10):
error = np.sum(loss.output("mse", y_row, nnetwork[-1]["output"]))
# Derivative fed into backprop:
errors = self.loss.output("mse", expected, nnetwork[-1]["output"], derivative=True)
```

### 2.6 Evaluation metrics (modules/metrics.py)

```python
from modules.metrics import mse, rmse, mae, r2_score

mse_val  = mse(y_true, y_pred)    # Mean Squared Error (not 0.5 * sum, but mean)
rmse_val = rmse(y_true, y_pred)   # Root MSE â€” same units as target
mae_val  = mae(y_true, y_pred)    # Mean Absolute Error â€” outlier robust
r2       = r2_score(y_true, y_pred) # 1.0 = perfect, 0 = baseline mean, <0 = worse
```

Note: `modules/metrics.mse` uses `mean((y_true - y_pred)^2)` (no 0.5), while
`nn_core/losses._mse` uses `0.5 * (t - y)^2`. The factor differs for computational
convenience â€” the 0.5 only matters for the derivative in backprop, not for evaluation.

---

## 3. Binary Cross-Entropy (BCE)

### 3.1 Information Theory Background

Cross-entropy comes from information theory. For a true distribution `p` and a predicted
distribution `q`, cross-entropy is:

```
H(p, q) = -sum_j p_j * log(q_j)
```

For binary classification (two classes: 0 and 1), with target `t âˆˆ {0, 1}` and
network output `y âˆˆ (0, 1)` (from sigmoid):

```
E = -[ t * log(y) + (1-t) * log(1-y) ]
```

When `t = 1`: `E = -log(y)` â€” loss is 0 when y=1, infinity when y=0
When `t = 0`: `E = -log(1-y)` â€” loss is 0 when y=0, infinity when y=1

### 3.2 Derivative w.r.t. output y

```
dE/dy = -( t/y - (1-t)/(1-y) )
```

For `t = 1`:  `dE/dy = -1/y`   (negative, push y toward 1)
For `t = 0`:  `dE/dy = 1/(1-y)` (positive, push y toward 0)

### 3.3 Combined with Sigmoid Output

When the output layer uses sigmoid, the combined derivative simplifies beautifully:

```
delta_output = dE/dv = (dE/dy) * (dy/dv)
             = -(t/y - (1-t)/(1-y)) * y*(1-y)
             = -(t*(1-y) - (1-t)*y)
             = -(t - ty - y + ty)
             = -(t - y)
             = y - t
```

This is the same as MSE's derivative! The simplification makes sigmoid + BCE numerically
cleaner and faster than sigmoid + MSE.

### 3.4 Why BCE for Classification, not MSE?

**MSE with sigmoid output:**
- Gradient near saturated regions (yâ‰ˆ0 or yâ‰ˆ1) is very small because sigmoid derivative â‰ˆ 0
- â†’ Very slow learning when predictions are confident but wrong

**BCE with sigmoid output:**
- Gradient = y - t regardless of saturation
- â†’ Large gradient when wrong (y=0.01, t=1 â†’ gradient = -0.99), even in saturated regions
- â†’ Much faster convergence

### 3.5 Numerical Stability

The formula `log(y)` is undefined for `y=0` and returns -inf for `yâ†’0`. Add epsilon:

```python
# This is NOT in the current implementation â€” add if needed for stability:
eps = 1e-12
y_safe = np.clip(y, eps, 1 - eps)
E = -t * np.log(y_safe) - (1-t) * np.log(1-y_safe)
```

The current implementation in `nn_core/losses.py` does NOT clip. If network outputs
exact 0 or 1 (unlikely with float64 but possible), this could produce NaN. Safe usage:
ensure the output activation is sigmoid (never exactly 0 or 1 in float64).

### 3.6 Code

```python
# nn_core/losses.py
@staticmethod
def _bce(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -t * np.log(y) - (1.0 - t) * np.log(1.0 - y)

@staticmethod
def _d_bce(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -(t / y - (1.0 - t) / (1.0 - y))
```

### 3.7 Evaluation metrics (modules/metrics.py)

```python
from modules.metrics import accuracy, precision, recall, f1_score, classification_report

acc   = accuracy(y_true, y_pred)           # fraction correct
prec  = precision(y_true, y_pred)          # TP / (TP + FP)
rec   = recall(y_true, y_pred)             # TP / (TP + FN)
f1    = f1_score(y_true, y_pred)           # harmonic mean of P and R
report = classification_report(y_true, y_pred)  # full formatted report
```

---

## 4. Loss in the Training Loop

### Position in backpropagation

```
Forward pass:
  v = W @ input + bias
  y = f(v)              â† activation function
  E = loss(t, y)        â† loss function (scalar per sample)

Backward pass:
  delta_output = dE/dy * f'(v)   â† output layer delta
  delta_hidden = (W_next.T @ delta_next) * f'(v)   â† hidden layer deltas

Weight update:
  W -= lr * outer(delta, prev_output)
```

The `dE/dy` (loss derivative) is computed in `nn_core/losses.py`.
It is the ONLY contribution of the loss function to backpropagation â€” everything else
comes from the activation derivatives.

### Code (nn_core/network.py â€” backward_propagate):
```python
# Output layer: loss derivative feeds into delta computation
errors = self.loss.output(
    loss_function,          # "mse" or "binary_cross_entropy"
    np.asarray(expected),   # t
    nnetwork[-1]["output"], # y
    derivative=True,        # dE/dy
)
nnetwork[N]["delta"] = np.multiply(
    errors,
    self.af.output(nnetwork[N], nnetwork[N]["activation_function"], derivative=True)
)
```

---

## 5. Choosing the Right Loss Function

| Task | Loss | Output activation | Why |
|------|------|------------------|-----|
| Regression (continuous values) | MSE | Linear | Unbounded output, penalizes large errors |
| Binary classification | BCE | Sigmoid | Probabilistic output, better gradients |
| Multi-class classification* | Categorical Cross-Entropy* | Softmax* | Mutually exclusive classes |

*Not implemented in this codebase. Would require adding softmax to `nn_core/activations.py`
and categorical CE to `nn_core/losses.py`.

---

## 6. Adding a New Loss Function to nn_core

1. Add `_myloss(t, y)` and `_d_myloss(t, y)` as `@staticmethod` in `LossFn`
2. Register in `self._dispatch` in `__init__`
3. Add tests in `tests/test_losses.py`
4. If needed, add evaluation metrics to `modules/metrics.py`

---

## 7. Key Code Locations

| Component | File | Method/Function |
|-----------|------|----------------|
| MSE + BCE (forward + derivative) | `nn_core/losses.py` | `LossFn._mse`, `._bce`, etc. |
| Loss used in training | `nn_core/network.py` | `backward_propagate()`, `train()` |
| Evaluation metrics | `modules/metrics.py` | `mse`, `rmse`, `mae`, `r2_score`, `accuracy`, `f1_score`, etc. |
| Tests | `tests/test_losses.py` | 8 tests |
| Test metrics | `tests/test_metrics.py` | 58 tests |

