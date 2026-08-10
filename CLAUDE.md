# mlp-study-kit — Claude / Claude Code Reference

> Read this file to understand the full project before making any changes.
> Author: Brijesh Rathod (bgrathod00@gmail.com) | Python 3.10+

---

## Project Overview

A progressive study kit that builds a Multi-Layer Perceptron (MLP) from scratch in pure NumPy, from a single neuron to TensorFlow/Keras. Two installable packages at the project root, plus exercises, homework, tools, examples, and tests.

---

## Project Structure

```
mlp-study-kit/                  ← flat layout (no src/ wrapper)
├── nn_core/                    ← MLP building blocks (installable)
│   ├── activations.py          ← ActivationFn: all 7 activations + derivatives
│   ├── losses.py               ← LossFn: MSE + BCE (forward + derivative)
│   ├── network.py              ← NeuralNetwork: build/train/save/load/summary
│   └── logger.py               ← ObjLogger: ANSI-coloured logger
├── modules/                    ← Helper utilities (installable)
│   ├── data_utils.py           ← make_regression_data, load_csv, k_fold_split, normalize
│   ├── plot_utils.py           ← plot_loss_history, plot_predictions, plot_activations
│   ├── general_utils.py        ← ensure_directory, as_float_array, print_matrices
│   └── metrics.py              ← mse, rmse, r2_score, accuracy, f1_score, evaluate
├── tests/                      ← pytest suite — 191 tests, all passing
│   └── conftest.py             ← adds project root to sys.path, sets MPLBACKEND=Agg
├── exercises/                  ← Standalone lecture exercises (ex06→ex12)
├── homework/                   ← hw_01, hw_02, hw_03
├── tools/                      ← Backprop debug tools
├── experiments/                ← Date-stamped exploration scripts
├── examples/                   ← Runnable end-to-end scripts
├── notebooks/                  ← Jupyter notebooks
└── outputs/                    ← Generated plots/weights (gitignored)
```

---

## Key Commands

```bash
# Install (editable — makes 'import nn_core' and 'import modules' work)
pip install -e .

# Verify install
python -c "from nn_core import NeuralNetwork; from modules import make_regression_data; print('OK')"

# Run tests
pytest tests/ -v                              # 191 tests
pytest --cov=nn_core --cov=modules            # with coverage

# Run examples
MPLBACKEND=Agg python examples/01_regression.py
MPLBACKEND=Agg python examples/02_classification.py

# Security
bandit -r nn_core/ modules/ -c pyproject.toml
pip-audit -r requirements.txt --skip-editable
```

---

## Coding Conventions

- **Python 3.10+** — use `X | Y` union types, `match` statements are OK
- **No `src/` layout** — both packages are at the project root, not inside `src/`
- **No `allow_pickle=True`** — weights use `.npz` format (np.savez / np.load)
- **No `sys.exit()`** in library code — raise `ValueError` or `RuntimeError`
- **MPLBACKEND=Agg** — always set before importing matplotlib in tests/CI
- **`raise ValueError`** not `sys.exit()` for bad inputs in `nn_core/` and `modules/`
- **Imports**: stdlib → numpy → nn_core → modules; no circular imports
- **Path setup** in scripts: `sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..")))` pointing to project root
- **Tests**: add to `tests/test_*.py`, use `tmp_path` fixture for file I/O
- **Git**: only `bgrathod00@gmail.com` as author

---

## Neural Network Theory

### Activation Functions (nn_core/activations.py)

All activations follow the same API:
```python
af = ActivationFn()
layer = {"activation_potential": v_arr, "output": None}
y  = af.output(layer, "relu")                        # forward
layer["output"] = y
dy = af.output(layer, "relu", derivative=True)       # derivative
```

| Name | Formula f(v) | Derivative f'(v) | Use |
|------|-------------|-----------------|-----|
| `linear` | v | 1 | Regression output |
| `sigmoid` | 1/(1+exp(-v)) | y(1-y) | Binary classification output |
| `tanh` | (exp(v)-exp(-v))/(exp(v)+exp(-v)) | 1-y² | Hidden layers |
| `relu` | max(0,v) | 1 if v≥0 else 0 | Hidden layers (fast) |
| `leaky_relu` | v if v≥0 else α·v (α=0.01) | 1 if v≥0 else α | Hidden layers (no dying) |
| `elu` | v if v≥0 else α(exp(v)-1) (α=1.0) | 1 if v≥0 else α·exp(v) | Hidden layers (smooth) |

**Critical:** derivatives use `layer["output"]` (sigmoid, tanh) or `layer["activation_potential"]` (relu, leaky_relu, elu). Both must be populated before calling with `derivative=True`.

**Vanishing gradient**: sigmoid and tanh saturate for |v|>4, gradient→0 → deep networks train slowly. ReLU avoids this for positive inputs. Use ReLU/Leaky-ReLU for hidden layers.

**Dying ReLU**: ~20% of neurons permanently output 0 if weights push v<0. Switch to Leaky ReLU or use Glorot init to mitigate.

---

### Loss Functions (nn_core/losses.py)

```python
lf = LossFn()
loss = lf.output("mse", t, y)                  # forward: 0.5*(t-y)²
grad = lf.output("mse", t, y, derivative=True) # backward: -(t-y)
```

**MSE** (`"mse"`): For regression. Formula: `E = 0.5*(t-y)²`. Derivative: `dE/dy = -(t-y)`.

**Binary Cross-Entropy** (`"binary_cross_entropy"`): For classification. Formula: `E = -(t·log(y) + (1-t)·log(1-y))`. Combined with sigmoid output: `delta = y - t` (clean, no saturation issue).

**Choose by task:**
- Regression → `loss_function="mse"` + `activation_function="linear"` on output
- Binary classification → `loss_function="binary_cross_entropy"` + `activation_function="sigmoid"` on output

---

### Backpropagation (nn_core/network.py)

Three methods, always called in this order per sample:
```python
model.forward_propagate(nnetwork, x_sample)
model.backward_propagate(loss_function, nnetwork, y_sample)
model._update_weights_clipped(nnetwork, x_sample, l_rate, grad_clip)
```

**Forward**: v^(l) = W^(l) @ y^(l-1), y^(l) = f(v^(l)) — fills `activation_potential` and `output`

**Backward delta chain**:
- Output layer: `δ^(L) = (∂E/∂y) · f'(v^(L))`
- Hidden layers: `δ^(l) = (W^(l+1).T @ δ^(l+1)) · f'(v^(l))`

**Weight update**: `W^(l) -= η · outer(δ^(l), y^(l-1))`

**Gradient clipping**: if `‖δ‖ > grad_clip`, scale `δ = δ · (grad_clip / ‖δ‖)` — prevents NaN

---

### Training API

```python
model = NeuralNetwork()
net   = model.create_network(structure)
print(model.summary(net))                   # shows architecture + param count

final_loss, h_train, h_test = model.train(
    net, X_train, Y_train,
    x_test=X_test, y_test=Y_test,
    l_rate=0.02, n_epoch=1000,
    loss_function="mse",
    epsilon=1e-5,                           # early stopping threshold
    grad_clip=1.0,                          # prevent exploding gradients / NaN
    verbose=1,
    save_plot="outputs/loss.png",           # headless-safe
    save_history="outputs/history.csv",     # loss per epoch
)

model.save_weights(net, "outputs/model")    # .npz, no pickle
model.load_weights(net, "outputs/model")    # shape-validated
```

**NaN loss**: raises `RuntimeError` immediately — reduce `l_rate` or add `grad_clip=1.0`

---

### Data Utilities (modules/data_utils.py)

```python
# Generate synthetic data
X, Y = make_regression_data(n=200, noise=0.1, seed=42)
X, Y, idx0, idx1 = make_classification_data(n_per_class=100)

# Load CSV (no pandas)
X, Y = load_csv("data.csv", target_col=-1)

# Split and normalize (ALWAYS use train stats for test set)
X_tr, Y_tr, X_te, Y_te = train_test_split(X, Y, test_ratio=0.2)
X_tr_n, mu, sig = normalize(X_tr)
X_te_n, _, _    = normalize(X_te, mean=mu, std=sig)

# Cross-validation
for X_tr, Y_tr, X_val, Y_val in k_fold_split(X, Y, k=5):
    ...
```

---

### Evaluation Metrics (modules/metrics.py)

```python
from modules.metrics import mse, rmse, r2_score, accuracy, f1_score, evaluate

# Regression
print(f"RMSE: {rmse(y_true, y_pred):.4f}")
print(f"R²:   {r2_score(y_true, y_pred):.4f}")

# Classification
print(f"Accuracy: {accuracy(y_true, y_pred)*100:.1f}%")
print(classification_report(y_true, y_pred))

# Everything at once
results = evaluate(y_true, y_pred, task="regression")   # dict of all metrics
```

---

## Security Rules

1. **Never** use `np.save(..., allow_pickle=True)` — use `np.savez`
2. **Never** use `sys.exit()` in library code — raise an exception
3. **Never** commit `AGENTS.md`, `.env`, `outputs/*`, `logs/*`
4. **No secrets** in code — check with `bandit -r nn_core/ modules/`
5. Weights in `.npz` format are safe to share (no pickle, no arbitrary code execution)

---

## File Naming Conventions

- Exercises: `ex{NN}_{description}.py`
- Tests: `test_{module}.py` in `tests/`
- Outputs: `outputs/{name}.png`, `outputs/{name}.npz`, `outputs/{name}.csv`
- Scripts: run from project root; `__file__`-relative paths only
