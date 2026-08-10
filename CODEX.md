# CODEX.md — OpenAI Codex / ChatGPT / GPT-4 Reference

> Project context for OpenAI-powered coding assistants working on mlp-study-kit.
> Author: Brijesh Rathod (bgrathod00@gmail.com) | MIT License | Python 3.10+

---

## What This Project Is

`mlp-study-kit` is a **teaching repository** that builds a Multi-Layer Perceptron from
scratch in pure NumPy. It is NOT a framework — every component is intentionally transparent
so a learner can see every equation implemented in code.

Structure:
- `nn_core/` — core MLP package (flat layout, importable after `pip install -e .`)
- `modules/` — helper utilities (data, plots, metrics, array validation)
- `exercises/` — 6 progressive lectures (ex06→ex12) building the network step-by-step
- `homework/` — university homework assignments (hw_01, hw_02, hw_03)
- `tools/` — backpropagation debuggers for exam preparation
- `examples/` — 3 runnable end-to-end scripts
- `tests/` — 191 pytest tests, all passing

---

## Architecture Reference

### Layer Dict Format (used throughout nn_core)

```python
layer = {
    "weights":              np.ndarray,  # shape: (n_units, n_prev [+1 if bias])
    "bias":                 bool,        # True = append 1 to input vector
    "activation_function":  str,         # name for ActivationFn dispatch
    "activation_potential": np.ndarray,  # v = W@y_prev  (filled by forward_propagate)
    "output":               np.ndarray,  # y = f(v)      (filled by forward_propagate)
    "delta":                np.ndarray,  # δ = dE/dv     (filled by backward_propagate)
}
```

### Activation Functions (nn_core/activations.py)

The `ActivationFn` class dispatches by string name. All use the same API:

```python
af    = ActivationFn()
layer = {"activation_potential": v_arr, "output": None}
y     = af.output(layer, name)                      # forward
layer["output"] = y
dy    = af.output(layer, name, derivative=True)     # derivative
```

**Mathematical definitions:**

**Linear** (`"linear"`):
```
f(v)  = v
f'(v) = 1
```

**Sigmoid** (`"sigmoid"` or `"logistic"`):
```
f(v)  = 1 / (1 + exp(-v))        range: (0, 1)
f'(v) = y * (1 - y)              uses output y, not v
```
Saturates for |v|>4 → vanishing gradient in deep networks.

**Tanh** (`"tanh"`):
```
f(v)  = (exp(v) - exp(-v)) / (exp(v) + exp(-v))    range: (-1, 1)
f'(v) = 1 - y²                                      zero-centered
```
Better than sigmoid for hidden layers (zero-centered output).

**ReLU** (`"relu"`):
```
f(v)  = max(0, v)
f'(v) = 1 if v >= 0 else 0
```
No vanishing gradient for positive v. ~20% neurons may die (v always negative).
Default choice for hidden layers.

**Leaky ReLU** (`"leaky_relu"`, alpha=0.01):
```
f(v)  = v       if v >= 0
f(v)  = alpha*v if v < 0
f'(v) = 1       if v >= 0
f'(v) = alpha   if v < 0
```
Fixes dying neuron problem. Call: `af.output(layer, "leaky_relu", alpha=0.1)`

**ELU** (`"elu"`, alpha=1.0):
```
f(v)  = v                    if v >= 0
f(v)  = alpha * (exp(v) - 1) if v < 0
f'(v) = 1                    if v >= 0
f'(v) = alpha * exp(v)       if v < 0
```
Smooth at v=0, negative saturation at -alpha.

---

### Loss Functions (nn_core/losses.py)

```python
lf = LossFn()

# MSE — for regression
E      = lf.output("mse", t, y)                  # 0.5*(t-y)²  element-wise
dE_dy  = lf.output("mse", t, y, derivative=True) # -(t-y)

# BCE — for binary classification
E      = lf.output("binary_cross_entropy", t, y)
dE_dy  = lf.output("binary_cross_entropy", t, y, derivative=True)
```

**MSE derivative:** `dE/dy = -(t-y)` — positive if underpredicting, negative if overpredicting.

**BCE with sigmoid:** delta at output layer simplifies to `y - t` (chain rule cancellation).
This is why BCE+sigmoid converges faster than MSE+sigmoid for classification.

---

### Backpropagation (nn_core/network.py)

**Forward pass** (`forward_propagate`):
```
For each layer l:
  v^(l) = W^(l) @ y^(l-1)    [append 1 to y^(l-1) if bias=True]
  y^(l) = f^(l)(v^(l))
```

**Backward pass** (`backward_propagate`):
```
Output layer L:
  δ^(L) = (∂E/∂y^(L)) · f'^(L)(v^(L))

Hidden layer l < L:
  δ^(l) = (W^(l+1).T @ δ^(l+1)) · f'^(l)(v^(l))
```

**Weight update** (`_update_weights_clipped`):
```
W^(l) = W^(l) - η · outer(δ^(l), y^(l-1))
```
If grad_clip is set: `δ = δ · (grad_clip / ‖δ‖)` when `‖δ‖ > grad_clip`.

**Training returns a tuple:**
```python
final_loss, history_train, history_test = model.train(net, X, Y, ...)
```

---

### Data Preparation (modules/data_utils.py)

```python
# Synthetic datasets (same as exercises)
X, Y = make_regression_data(n=200, noise=0.1, seed=42)   # sin(2x)+cos(x)+5+noise
X, Y, idx0, idx1 = make_classification_data(n_per_class=100)  # 2D two-class

# CSV loading without pandas
X, Y = load_csv("data.csv", target_col=-1, has_header=True)
# multi-feature: load_csv("data.csv", feature_cols=[0,1,2], target_col=3)

# Train/test split + normalization
X_tr, Y_tr, X_te, Y_te = train_test_split(X, Y, test_ratio=0.2, seed=42)
X_tr_n, mu, sig = normalize(X_tr)
X_te_n, _, _    = normalize(X_te, mean=mu, std=sig)  # MUST reuse train stats

# k-fold CV
for X_tr, Y_tr, X_val, Y_val in k_fold_split(X, Y, k=5, seed=42):
    # train one fold
```

---

### Evaluation Metrics (modules/metrics.py)

```python
# Regression
mse_val  = mse(y_true, y_pred)
rmse_val = rmse(y_true, y_pred)    # same units as target
r2       = r2_score(y_true, y_pred) # 1=perfect, 0=baseline, <0=worse than baseline
print(regression_summary(y_true, y_pred))

# Classification
acc   = accuracy(y_true, y_pred)   # default threshold=0.5
f1    = f1_score(y_true, y_pred)
prec  = precision(y_true, y_pred)
rec   = recall(y_true, y_pred)
cm    = confusion_matrix(y_true, y_pred)  # [[TN,FP],[FN,TP]]
print(classification_report(y_true, y_pred))

# All-in-one dict
results = evaluate(y_true, y_pred, task="regression")
results = evaluate(y_true, y_pred, task="classification")
```

---

## Security Constraints

When generating code for this project:

1. Use `np.savez(path, **dict)` and `np.load(path)` — NEVER `np.save(..., allow_pickle=True)` or `np.load(..., allow_pickle=True)`
2. Raise exceptions (`ValueError`, `RuntimeError`) — NEVER `sys.exit()` in nn_core or modules
3. Use `os.environ.get("MPLBACKEND", "")` guard before `import matplotlib.pyplot as plt`
4. No secrets, API keys, or credentials in any file
5. Paths in scripts: always compute from `__file__` — never hardcode absolute paths
6. File I/O: use `open(..., encoding="utf-8")` always

---

## Common Patterns

### Add a new activation function:
1. Add `_myact(layer)` and `_d_myact(layer)` as `@staticmethod` in `ActivationFn` (nn_core/activations.py)
2. Register in `self._dispatch` in `__init__`
3. If parameterized (like alpha): follow Leaky ReLU pattern
4. Add tests in `tests/test_activations.py`

### Add a new loss function:
1. Add `_myloss(t, y)` and `_d_myloss(t, y)` as `@staticmethod` in `LossFn` (nn_core/losses.py)
2. Register in `self._dispatch`
3. Add tests in `tests/test_losses.py`

### Train a classification model:
```python
model = NeuralNetwork()
structure = [
    {"type": "input",  "units": 2},
    {"type": "dense",  "units": 16, "activation_function": "tanh",    "bias": True},
    {"type": "dense",  "units": 1,  "activation_function": "sigmoid", "bias": True},
]
net = model.create_network(structure)
loss, _, _ = model.train(net, X_tr, Y_tr, loss_function="binary_cross_entropy",
                         l_rate=0.01, n_epoch=1000, verbose=0)
preds = np.array(model.predict(net, X_te)).flatten()
print(f"Accuracy: {accuracy(Y_te.flatten(), preds)*100:.1f}%")
```

---

## File Map for Code Generation

| Task | File to edit |
|------|-------------|
| New activation function | `nn_core/activations.py` |
| New loss function | `nn_core/losses.py` |
| Change training loop | `nn_core/network.py` — `train()` |
| New data generator | `modules/data_utils.py` |
| New plot type | `modules/plot_utils.py` |
| New metric | `modules/metrics.py` |
| New validation util | `modules/general_utils.py` |
| Test for nn_core | `tests/test_*.py` |
| Runnable example | `examples/0N_description.py` |
