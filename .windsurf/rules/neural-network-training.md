# mlp-study-kit — Training Workflow Reference

## 1. Complete Workflow Overview

Building and training a network in this codebase follows a fixed sequence:

```
1. Prepare data       → modules/data_utils.py
2. Define architecture → list of layer dicts (structure)
3. Initialise weights → NeuralNetwork.create_network(structure)
4. Train              → NeuralNetwork.train(net, X_train, Y_train, ...)
5. Evaluate           → modules/metrics.py + model.predict()
6. Save               → model.save_weights(net, path)
```

Each step has dedicated utilities. Skipping or reordering steps (especially normalisation before splitting) causes subtle bugs that are difficult to diagnose.

## 2. Data Preparation — modules/data_utils.py

All data utilities are pure NumPy — no pandas, no sklearn. They work with 2D arrays where rows are samples and columns are features.

### Synthetic Data Generators

```python
from modules.data_utils import make_regression_data, make_classification_data

# Regression: y = sin(2x) + cos(x) + 5 + Gaussian noise
X, T = make_regression_data(n=200, noise=0.3, x_range=(-3, 3), seed=42)
# X shape: (200, 1), T shape: (200, 1)

# Classification: 2D two-class dataset (two Gaussian clusters)
X, T = make_classification_data(n_per_class=100, seed=42)
# X shape: (200, 2), T shape: (200, 1), T values: 0.0 or 1.0
```

### Loading Real CSV Data

```python
from modules.data_utils import load_csv

X, T = load_csv(
    path="data/my_dataset.csv",
    feature_cols=[0, 1, 2],    # column indices for inputs
    target_col=3,              # column index for target
    has_header=True,           # skip the first row
)
```

No pandas dependency. Reads with the `csv` module and converts to NumPy. Handles missing values by raising informative errors.

### Train/Test Split

```python
from modules.data_utils import train_test_split

X_train, X_test, T_train, T_test = train_test_split(X, T, test_ratio=0.2, seed=42)
```

Pure NumPy shuffle + slice. Always set a `seed` for reproducibility.

### Normalisation

```python
from modules.data_utils import normalize

# Compute statistics from training data ONLY
X_train_norm, mean, std = normalize(X_train)

# Apply the SAME statistics to test data
X_test_norm, _, _ = normalize(X_test, mean=mean, std=std)
```

**Critical rule: always fit normalisation parameters on training data only, then apply to test data using those same parameters.** Fitting on all data (including test) causes data leakage — the model implicitly sees test set statistics during training.

The function returns `(X_normalised, mean, std)`. When you pass `mean` and `std` explicitly, it uses them instead of computing new ones — this is the pattern for applying train stats to test data.

### K-Fold Cross-Validation

```python
from modules.data_utils import k_fold_split

folds = k_fold_split(X, T, k=5, seed=42)
for fold_idx, (X_tr, X_val, T_tr, T_val) in enumerate(folds):
    # train and evaluate on each fold
    pass
```

Returns a list of `(X_train, X_val, T_train, T_val)` tuples. Useful for hyperparameter selection when data is limited.

## 3. Architecture Definition

The network architecture is defined as a Python list of layer dicts called `structure`. Pass this to `NeuralNetwork.create_network()` to build `net` — the live network with weights allocated.

```python
structure = [
    {"type": "input",  "units": n_features},
    {"type": "dense",  "units": 64,  "activation_function": "tanh",    "bias": True},
    {"type": "dense",  "units": 32,  "activation_function": "relu",    "bias": True},
    {"type": "dense",  "units": 1,   "activation_function": "linear",  "bias": True},
]
```

### Architecture Rules

**Input layer:** `type="input"`, `units` = number of input features. No `activation_function` or `bias` — it just passes data through.

**Hidden layers:** `type="dense"`, choose an activation from `{"tanh", "relu", "leaky_relu", "elu"}`. Always include `"bias": True` unless you have a specific reason not to (bias allows the network to shift activations, improving expressivity without costing much).

**Output layer:** activation depends on the task:

| Task | Output activation | Units |
|------|------------------|-------|
| Regression | `"linear"` | Number of output values |
| Binary classification | `"sigmoid"` | 1 |

**Sizing guidelines:**
- Start with 1–2 hidden layers.
- For simple regression: 16–64 units per layer.
- For simple classification: 32–128 units per layer.
- Prefer wider networks over very deep ones for tabular data.
- More capacity = more risk of overfitting — add regularisation (early stopping) accordingly.

**Creating the live network:**

```python
from nn_core.network import NeuralNetwork

model = NeuralNetwork()
net = model.create_network(structure)
```

`create_network()` allocates weight matrices and returns `net` — the list of layer dicts with `weights`, `activation_potential`, `output`, and `delta` keys initialised.

## 4. Weight Initialisation

Weights are initialised in `create_network()`. Poor initialisation causes slow convergence, vanishing gradients, or exploding gradients even before training begins.

### Default Initialisation

```python
W = np.random.randn(n_out, n_in) * 0.2
```

Small random weights centred at 0. The `0.2` scale is a conservative heuristic that works reasonably well for shallow networks with tanh/sigmoid activations. It avoids saturation at initialisation.

### Glorot Normal (Xavier) Initialisation

```python
sigma = np.sqrt(2.0 / (n_in + n_out))
W = np.random.randn(n_out, n_in) * sigma
```

Derived by setting the variance of activations and gradients to be equal across layers, under the assumption of linear activations (used as an approximation for tanh/sigmoid). The factor `2 / (n_in + n_out)` balances the fan-in and fan-out.

- **Best for:** sigmoid and tanh activations.
- Keeps signal variance roughly constant across layers — prevents vanishing/exploding gradients at initialisation.
- Implemented in `experiments/20251224_v2.py` as a parameter to network construction.

### He Initialisation (for ReLU)

```python
sigma = np.sqrt(2.0 / n_in)
W = np.random.randn(n_out, n_in) * sigma
```

Accounts for the fact that ReLU zeros out half of its inputs, effectively halving the signal. The factor `2 / n_in` compensates. Not yet built into `nn_core` — add manually if using deep ReLU networks and observing slow early convergence.

**Quick rule:** use Glorot for tanh/sigmoid, He for ReLU/Leaky ReLU.

## 5. Complete train() API

```python
final_loss, history_train, history_test = model.train(
    net,                              # live network (list of layer dicts)
    X_train,                          # input array, shape (n_samples, n_features)
    Y_train,                          # target array, shape (n_samples, n_outputs)
    x_test=X_test,                    # optional — enables test loss tracking
    y_test=Y_test,                    # optional — required if x_test is given
    l_rate=0.02,                      # learning rate η
    n_epoch=1000,                     # maximum number of training epochs
    loss_function="mse",              # "mse" or "bce"
    epsilon=1e-5,                     # early stopping threshold (set None to disable)
    grad_clip=1.0,                    # gradient clipping norm (set None to disable)
    save_plot="outputs/loss.png",     # optional — saves training curve plot
    save_history="outputs/history.csv", # optional — saves epoch losses to CSV
    verbose=1,                        # 0=silent, 1=progress every 100 epochs
)
```

### Return Value

`train()` always returns a **3-tuple**:

```python
final_loss, history_train, history_test = model.train(...)
```

- `final_loss` — scalar loss on the training set at the final epoch
- `history_train` — list of training loss values, one per epoch
- `history_test` — list of test loss values, one per epoch (empty list if no test data given)

**Always unpack all three values.** Ignoring the return value discards the history.

### Early Stopping

When `epsilon` is set, training stops if the improvement in test loss between consecutive epochs falls below `epsilon`. This prevents overfitting without manually choosing `n_epoch`. Requires `x_test` and `y_test` to be provided.

```python
# Will stop before 5000 epochs if test loss plateaus
model.train(net, X_train, Y_train, x_test=X_test, y_test=Y_test,
            n_epoch=5000, epsilon=1e-6)
```

### Gradient Clipping

`grad_clip=1.0` is a safe default for most problems. If you see NaN losses, this is the first thing to add. If training seems too slow (gradients are being clipped even for small errors), try `grad_clip=5.0` or disable it.

## 6. NeuralNetwork Extras

### Architecture Summary

```python
model.summary(net)
```

Prints a formatted table showing each layer's type, units, activation function, weight matrix shape, and parameter count. Use this to verify the architecture before training and to report parameter counts.

### Saving and Loading Weights

```python
# Save — writes net weights to outputs/model.npz (NumPy compressed archive)
model.save_weights(net, "outputs/model")

# Load — reads weights back into the net structure
model.load_weights(net, "outputs/model")
```

Weights are saved as a `.npz` file (NumPy's native compressed format). This avoids pickle security issues and remains readable without this codebase — you can inspect weights with `np.load("outputs/model.npz")`.

Loading performs **shape validation** — if the saved weights don't match the current `net` structure, an informative error is raised. Always use the same `structure` to recreate `net` before loading.

### Repr

```python
print(model)   # calls __repr__
```

Displays a compact architecture summary string.

### Prediction

```python
y_pred = model.predict(net, X_test)
# Returns array of shape (n_samples, n_outputs)
```

Runs forward propagation for each sample and collects outputs. For classification, threshold the output:

```python
y_class = (y_pred >= 0.5).astype(int)
```

## 7. Hyperparameter Tuning Guide

### Learning Rate — the Most Important Hyperparameter

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| Loss barely moves after 100 epochs | `l_rate` too small | Increase by 10× |
| Loss is NaN after first epoch | `l_rate` too large or exploding gradients | Decrease by 10× and/or add `grad_clip=1.0` |
| Loss oscillates up and down | `l_rate` slightly too large | Decrease by 2–5× |
| Loss decreases then plateaus early | Good rate, but model underfits | Increase model size or train longer |

Good starting values: `0.001` (conservative), `0.01` (typical), `0.1` (aggressive). Start at `0.01` and adjust based on the first 100-epoch loss curve.

### Not Learning at All

Checklist (in order of likelihood):

1. Are inputs normalised? Unnormalised features (e.g. pixel values 0–255 mixed with binary flags) create an extremely ill-conditioned loss landscape.
2. Is the output activation correct? ReLU on the output of a regression network clips all negative predictions.
3. Is the loss function correct? BCE requires probabilities; MSE requires real values.
4. Are weights initialised too large? Try smaller scale (`* 0.1` instead of `* 0.2`).
5. Is the architecture deep enough to represent the function?

### NaN Loss

1. Add `grad_clip=1.0` first.
2. Reduce `l_rate` by 10×.
3. Normalise inputs.
4. Check for NaN or Inf values in the raw data (`np.isnan(X).any()`).

### Overfitting

1. Enable early stopping (`epsilon=1e-5`).
2. Reduce model size (fewer layers or units).
3. Collect more data.
4. Add data augmentation (if applicable to the problem domain).

### Underfitting

1. Increase model capacity (more hidden units or more layers).
2. Train for more epochs.
3. Increase learning rate slightly.
4. Verify the problem is learnable — check that train loss also decreases.

## 8. Key Files

| File | Role |
|------|------|
| `nn_core/network.py` | `NeuralNetwork` class — `create_network`, `forward_propagate`, `backward_propagate`, `train`, `predict`, `save_weights`, `load_weights`, `summary` |
| `modules/data_utils.py` | All data preparation utilities — generators, CSV loader, split, normalise, k-fold |
| `modules/metrics.py` | All evaluation metrics — regression (MSE, RMSE, MAE, R²) and classification (accuracy, precision, recall, F1, confusion matrix) |
| `modules/plot_utils.py` | `plot_loss_history()`, `plot_predictions()`, `plot_activations()`, `plot_decision_boundary()` |
| `examples/01_regression.py` | End-to-end regression example — synthetic data, tanh hidden + linear output, MSE |
| `examples/02_classification.py` | End-to-end classification example — synthetic data, tanh hidden + sigmoid output, BCE |
| `examples/03_custom_csv_data.py` | End-to-end example with real CSV data and `load_csv()` |
