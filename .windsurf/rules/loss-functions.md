# mlp-study-kit — Loss Functions Reference

## 1. MSE — Mean Squared Error

**Formula (forward):**

```
E = 0.5 * sum( (t - y)^2 )
```

- `t` — target (ground truth) vector
- `y` — network output vector (`y^(L)`)
- The sum is over all output units

**Derivative (backward):**

```
dE/dy = -(t - y) = y - t
```

The chain rule gives this as the gradient with respect to the output `y^(L)`, which feeds directly into the output layer delta computation.

**Why the 0.5 factor?**
It is a mathematical convenience. When you differentiate `0.5 * (t - y)^2` with respect to `y`, the power rule brings down a factor of `2` which cancels the `0.5`, leaving the clean gradient `y - t`. It has no effect on the optimisation — scaling the loss by a constant shifts the learning rate but not the gradient direction.

**Properties:**
- Differentiable everywhere — smooth loss landscape, compatible with gradient descent.
- Penalises large errors heavily (squared term), which is both a strength and a weakness.
- **Sensitive to outliers:** a single data point with a large error contributes disproportionately to the loss and gradient. Consider MAE (not implemented in `nn_core`) for outlier-robust regression.
- When output units are independent, MSE decomposes per-unit, making it easy to interpret.

**When to use:**
- Regression tasks where the target is a continuous, unbounded real value.
- Pair with a `"linear"` output activation so the network can predict any value in `(-∞, +∞)`.
- Not suitable for classification — see BCE below.

## 2. BCE — Binary Cross-Entropy

**Formula (forward):**

```
E = -( t * log(y) + (1 - t) * log(1 - y) )
```

- `t ∈ {0, 1}` — binary target
- `y ∈ (0, 1)` — network output (must be a probability; use sigmoid output)
- The log terms penalise confident wrong predictions extremely harshly (log→−∞ as the probability approaches the wrong extreme)

**Derivative (backward):**

```
dE/dy = -( t/y - (1-t)/(1-y) )
```

Simplified: `(y - t) / ( y * (1 - y) )`

**Combined Sigmoid + BCE gradient (key result):**

When the output activation is sigmoid and the loss is BCE, the output layer delta simplifies to:

```
δ^(L) = y - t
```

This is derived by substituting both the sigmoid derivative `y(1-y)` and the BCE gradient `(y-t)/(y(1-y))` into the delta formula:

```
δ^(L) = dE/dy * f'(v) = [(y-t)/(y(1-y))] * [y(1-y)] = y - t
```

The `y(1-y)` terms cancel exactly. This means the gradient at the output layer is simply `y - t`, regardless of saturation. **This is the primary reason to pair BCE with sigmoid rather than MSE with sigmoid.**

## 3. Why BCE Beats MSE for Classification

When using MSE with a sigmoid output, the output layer delta is:

```
δ^(L)_mse+sigmoid = (y - t) * y * (1 - y)
```

The `y(1-y)` term is the sigmoid derivative. When the sigmoid is saturated (output near 0 or 1), this term is near 0. Even when the network is confidently wrong — say `y ≈ 0` for `t = 1` — the gradient nearly vanishes and the weights update only slightly. Training stalls.

With BCE + sigmoid:

```
δ^(L)_bce+sigmoid = y - t
```

When the network is confidently wrong (`y ≈ 0`, `t = 1`), this gradient is `≈ -1` — a strong, consistent update signal regardless of saturation. Training does not stall.

**Summary of the asymmetry:**

| Setting | Gradient at saturated output | Learning behaviour |
|---------|-----------------------------|--------------------|
| MSE + Sigmoid | Near 0 | Slow / stalls |
| BCE + Sigmoid | `y - t` (full strength) | Consistent |

This is not a minor numerical difference — it is the fundamental reason maximum likelihood training (BCE) is theoretically correct for probabilistic outputs and MSE is not.

## 4. Choosing the Right Loss

| Task | Output Activation | Loss Function | Reasoning |
|------|------------------|---------------|-----------|
| Regression (unbounded) | Linear | MSE | Targets are real-valued; linear output covers all of `ℝ` |
| Binary classification | Sigmoid | BCE | Output is a probability; BCE gives clean gradients |
| Multi-class classification | Softmax (not in this kit) | Categorical cross-entropy | Generalises BCE to K classes |

**Do not mix:**
- MSE + Sigmoid for classification — works but converges slowly due to vanishing gradients.
- BCE + Linear for regression — BCE requires `y ∈ (0, 1)` and is undefined otherwise.

## 5. Code

```python
from nn_core.losses import LossFn

lf = LossFn()

# Forward — compute scalar loss value
loss_mse = lf.output("mse", t, y)             # E = 0.5 * sum((t-y)^2)
loss_bce = lf.output("bce", t, y)             # E = -(t*log(y) + (1-t)*log(1-y))

# Backward — compute gradient dE/dy (used as input to output layer delta)
grad_mse = lf.output("mse", t, y, derivative=True)   # y - t
grad_bce = lf.output("bce", t, y, derivative=True)   # -(t/y - (1-t)/(1-y))

# The NeuralNetwork.backward_propagate() method calls lf.output(..., derivative=True)
# internally and multiplies by the output activation derivative to get δ^(L).
```

Supported loss names: `"mse"`, `"bce"`. Pass the string to both `train()` and manual calls.

## 6. Evaluation Metrics — modules/metrics.py

The training loss (`nn_core/losses.py`) measures gradient signal during training. **Evaluation metrics** (`modules/metrics.py`) measure the quality of predictions after training and are kept entirely separate — they do not affect weight updates.

### Regression Metrics

| Function | Formula | Purpose |
|----------|---------|---------|
| `mse(y_true, y_pred)` | `mean((y_true - y_pred)^2)` | Mean squared error |
| `rmse(y_true, y_pred)` | `sqrt(mse(...))` | Root MSE — same units as target |
| `mae(y_true, y_pred)` | `mean(abs(y_true - y_pred))` | Robust to outliers |
| `r2_score(y_true, y_pred)` | `1 - SS_res/SS_tot` | Fraction of variance explained; 1.0 is perfect |
| `regression_summary(y_true, y_pred)` | All of the above | Prints a formatted table of all regression metrics |

### Classification Metrics

| Function | Description |
|----------|-------------|
| `accuracy(y_true, y_pred)` | Fraction of correct predictions |
| `precision(y_true, y_pred)` | `TP / (TP + FP)` — of predicted positives, how many are correct |
| `recall(y_true, y_pred)` | `TP / (TP + FN)` — of actual positives, how many were found |
| `f1_score(y_true, y_pred)` | Harmonic mean of precision and recall: `2 * P * R / (P + R)` |
| `confusion_matrix(y_true, y_pred)` | 2×2 matrix: `[[TN, FP], [FN, TP]]` |
| `classification_report(y_true, y_pred)` | Formatted table with accuracy, precision, recall, F1, and confusion matrix |

### Convenience Wrapper

```python
from modules.metrics import evaluate

# Regression
results = evaluate(y_true, y_pred, task="regression")

# Classification (expects binary 0/1 predictions, not raw probabilities)
y_class = (y_pred > 0.5).astype(int)
results = evaluate(y_true, y_class, task="classification")
```

`evaluate()` dispatches to the appropriate summary function based on `task` and returns a dict of metric values.

**Note on classification thresholding:** network outputs are probabilities `∈ (0, 1)`. Before passing to classification metrics, threshold at 0.5: `y_pred_binary = (y_pred >= 0.5).astype(int)`. The metrics functions expect integer class labels, not floats.

## 7. Key Files

| File | Role |
|------|------|
| `nn_core/losses.py` | `LossFn` class — MSE and BCE, forward and derivative |
| `modules/metrics.py` | All evaluation metrics — regression and classification, separate from training |
| `tests/test_losses.py` | 8 unit tests — MSE and BCE forward and derivative, edge cases |
| `tests/test_metrics.py` | 58 unit tests — full coverage of all metric functions |
