# GitHub Copilot Instructions — mlp-study-kit

> This file gives Copilot context about the project structure, coding conventions,
> and the neural network theory implemented here.

---

## Project Summary

`mlp-study-kit` is a Python study kit that builds a Multi-Layer Perceptron from scratch
in pure NumPy. It has two installable packages at the **project root** (flat layout):
- `nn_core/` — MLP building blocks (activations, losses, network, logger)
- `modules/` — helper utilities (data, plotting, metrics, array utils)

Plus: `exercises/` (ex06→ex12), `homework/`, `tools/`, `examples/`, `tests/`, `experiments/`

**Python 3.10+ | Author: bgrathod00@gmail.com | MIT License**

---

## Package Imports (After `pip install -e .`)

```python
# Core MLP components
from nn_core import NeuralNetwork, ActivationFn, LossFn, ObjLogger

# Helper utilities
from modules.data_utils    import make_regression_data, train_test_split, normalize, load_csv, k_fold_split
from modules.plot_utils    import plot_loss_history, plot_predictions, plot_activations, plot_decision_boundary
from modules.general_utils import ensure_directory, as_float_array, describe_array, print_matrices
from modules.metrics       import mse, rmse, mae, r2_score, accuracy, precision, recall, f1_score, evaluate
```

---

## Coding Conventions

- **Flat layout** — packages are at project root, not `src/`. `pyproject.toml` uses `where=["."]`
- **Path setup** in non-installed scripts: `sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..")))`
- **Weights** use `.npz` format (`np.savez`/`np.load`) — **never** `allow_pickle=True`
- **Errors** raise `ValueError` or `RuntimeError` — **never** `sys.exit()` in library code
- **Matplotlib** — always set `MPLBACKEND=Agg` before import in tests/CI, or use `save_plot=` param
- **Tests** — add to `tests/test_*.py`, run with `pytest tests/ -v` (191 tests currently)
- **git author** — only `bgrathod00@gmail.com`

---

## Activation Functions

All 7 activations in `nn_core/activations.py`. API:

```python
af = ActivationFn()
layer = {"activation_potential": v, "output": None}
y  = af.output(layer, "tanh")                      # forward pass
layer["output"] = y
dy = af.output(layer, "tanh", derivative=True)     # derivative for backprop
```

| Function | Formula | Derivative (using output y) | Best for |
|----------|---------|---------------------------|---------|
| `linear` | v | 1 | Regression output layer |
| `sigmoid` | 1/(1+e^-v) | y(1-y) | Binary classification output |
| `tanh` | (e^v-e^-v)/(e^v+e^-v) | 1-y² | Hidden layers |
| `relu` | max(0,v) | 1 if v≥0 else 0 | Hidden layers (fast) |
| `leaky_relu` | v if v≥0 else αv (α=0.01) | 1 if v≥0 else α | Hidden (no dying neurons) |
| `elu` | v if v≥0 else α(e^v-1) (α=1.0) | 1 if v≥0 else αe^v | Hidden (smooth gradient) |

**Key:** sigmoid/tanh derivatives use `layer["output"]`. ReLU/Leaky/ELU use `layer["activation_potential"]`.
Both are populated by `forward_propagate()` before `backward_propagate()` is called.

**Vanishing gradients:** sigmoid and tanh gradient → 0 for |v|>4 → deep networks train slowly.
ReLU solves this for positive v but has dying neuron problem (~20% dead with bad init).
Leaky ReLU / ELU fully fix both problems.

---

## Loss Functions

```python
lf = LossFn()
loss = lf.output("mse", t, y)                     # forward: 0.5*(t-y)²
grad = lf.output("mse", t, y, derivative=True)    # backward: -(t-y)
```

**MSE** — regression, unbounded output (linear activation):
- `E = 0.5*(t-y)²`, derivative: `dE/dy = -(t-y)`

**Binary Cross-Entropy (BCE)** — classification, sigmoid output:
- `E = -(t·log(y) + (1-t)·log(1-y))`, derivative: `dE/dy = -(t/y - (1-t)/(1-y))`
- With sigmoid output: combined delta = `y - t` (much cleaner gradient than MSE+sigmoid)

**Rule:** regression → MSE + linear output. Binary classification → BCE + sigmoid output.

---

## Backpropagation

Three methods in `nn_core/network.py`, always in this order per sample:

```python
model.forward_propagate(nnetwork, x)          # fills activation_potential, output
model.backward_propagate("mse", nnetwork, t)  # fills delta in each layer
model._update_weights_clipped(nnetwork, x, lr, grad_clip)  # W -= lr * outer(delta, prev)
```

**Delta computation:**
- Output: `δ = (dE/dy) · f'(v)` — loss derivative times activation derivative
- Hidden: `δ^(l) = (W^(l+1).T @ δ^(l+1)) · f'(v^(l))` — propagate backward through weights

**Weight update:** `W -= lr * outer(δ, prev_layer_output)` — outer product gives gradient for all weights

**Gradient clipping:** if `‖δ‖ > grad_clip`, scale delta to have norm = grad_clip (prevents NaN)

---

## Training

```python
model = NeuralNetwork()
net   = model.create_network([
    {"type": "input",  "units": 1},
    {"type": "dense",  "units": 32, "activation_function": "tanh",   "bias": True},
    {"type": "dense",  "units": 1,  "activation_function": "linear", "bias": True},
])

# train() returns (final_loss, history_train, history_test)
loss, h_tr, h_te = model.train(
    net, X_train, Y_train,
    x_test=X_test, y_test=Y_test,
    l_rate=0.02, n_epoch=1000,
    loss_function="mse",
    epsilon=1e-5,          # early stopping
    grad_clip=1.0,         # prevent NaN
    save_plot="out.png",   # headless safe
    save_history="out.csv",
    verbose=0,
)

print(model.summary(net))                    # architecture + param count
model.save_weights(net, "weights")           # saves as weights.npz
model.load_weights(net, "weights")           # loads + validates shapes
```

---

## Data Utilities

```python
# Generate (same data as exercises)
X, Y = make_regression_data(n=200, noise=0.1)         # sin(2x)+cos(x)+5
X, Y, i0, i1 = make_classification_data(n_per_class=100)  # 2D two-class

# Load your own CSV (no pandas)
X, Y = load_csv("data.csv", target_col=-1, has_header=True)

# Split + normalize (MUST use train stats for test normalization)
X_tr, Y_tr, X_te, Y_te = train_test_split(X, Y, test_ratio=0.2)
X_tr_n, mu, sig = normalize(X_tr)
X_te_n, _, _    = normalize(X_te, mean=mu, std=sig)

# k-fold cross-validation
for X_tr, Y_tr, X_val, Y_val in k_fold_split(X, Y, k=5):
    net = model.create_network(structure)
    model.train(net, X_tr, Y_tr, verbose=0)
```

---

## Tests

```bash
pytest tests/ -v                          # 191 tests, all passing
pytest tests/test_activations.py -v      # 19 activation tests
pytest tests/test_network.py -v          # 27 network tests (incl. NaN, grad_clip, summary)
pytest tests/test_metrics.py -v          # 58 metrics tests
```

Tests use `conftest.py` which: adds project root to `sys.path` and sets `MPLBACKEND=Agg`.

---

## Security

- `save_weights` / `load_weights` use `.npz` (NOT pickle) — safe to share
- `bandit -r nn_core/ modules/` — no security issues should be introduced
- Never commit `AGENTS.md`, `.env`, `outputs/`, `logs/`
