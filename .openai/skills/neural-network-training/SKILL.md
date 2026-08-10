# Codex Skill: Neural Network Training â€” Weight Init, Optimizers & Practical Guide

## When to apply this context
Invoke when working on:
- Setting up a new training run (architecture choice, hyperparameters)
- Weight initialization in `experiments/20251224_v2.py` (Glorot/Xavier)
- Adagrad optimizer in `experiments/20251224_v2.py`
- Early stopping in `nn_core/network.py`
- `modules/data_utils.py` â€” normalize, train_test_split, k_fold_split
- Understanding why training is slow, oscillating, or diverging
- Evaluating results with `modules/metrics.py`

---

## 1. The End-to-End Training Workflow

```
1. Data preparation
   â””â”€ load â†’ split â†’ normalize

2. Architecture design
   â””â”€ input units â†’ hidden layers â†’ output units + activations

3. Weight initialization
   â””â”€ random, Glorot Normal, or fixed

4. Training loop
   â””â”€ for epoch in n_epochs:
        for sample in dataset:          # online SGD
          forward_propagate()
          compute_loss()
          backward_propagate()
          update_weights()
        if early_stop: break

5. Evaluation
   â””â”€ MSE / RMSE / RÂ² for regression
      accuracy / F1 for classification

6. Persist
   â””â”€ save_weights() â†’ outputs/model.npz
```

---

## 2. Data Preparation

### 2.1 Generating Synthetic Data (modules/data_utils.py)

**Regression (sin+cos target):**
```python
from modules.data_utils import make_regression_data
X, Y = make_regression_data(n=200, noise=0.15, x_range=(-3, 3), seed=42)
# X.shape = (200, 1), Y.shape = (200, 1)
# Y = sin(2x) + cos(x) + 5 + noise
```

**Linear regression:**
```python
from modules.data_utils import make_linear_data
X, Y = make_linear_data(n=100, slope=2.0, intercept=-1.0, noise=0.3)
```

**Binary classification:**
```python
from modules.data_utils import make_classification_data
X, Y, idx0, idx1 = make_classification_data(n_per_class=100, seed=42)
# X.shape = (200, 2), Y.shape = (200,)
# Class 0: x in [0,2]x[0,2], Class 1: x in [1,3]x[2,4]
```

**Load from CSV:**
```python
from modules.data_utils import load_csv
X, Y = load_csv("my_data.csv", target_col=-1, has_header=True)
# No pandas required â€” stdlib csv module only
```

### 2.2 Train/Test Split

```python
from modules.data_utils import train_test_split
X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_ratio=0.2, seed=42)
```

Always split BEFORE normalizing â€” compute stats on train set only.

### 2.3 Normalization (Critical for Training Stability)

```python
from modules.data_utils import normalize
X_train_n, mu, sigma = normalize(X_train)
X_test_n, _, _       = normalize(X_test, mean=mu, std=sigma)   # reuse train stats!
```

**Why normalize?**
Without normalization, features on different scales cause uneven gradient magnitudes.
A feature in [0, 1000] will dominate features in [0, 1], making training slow and
sensitive to learning rate.

**What normalize does:**
```
X_norm = (X - mean) / (std + eps)
```
After: mean â‰ˆ 0, std â‰ˆ 1 per feature

**CRITICAL: Apply train mean/std to test set, NOT test mean/std.** If you normalize
the test set independently, you leak future data into training statistics.

### 2.4 K-Fold Cross-Validation

```python
from modules.data_utils import k_fold_split
folds = k_fold_split(X, Y, k=5, seed=42)
val_losses = []
for X_tr, Y_tr, X_val, Y_val in folds:
    net = model.create_network(structure)
    loss, _, _ = model.train(net, X_tr, Y_tr, verbose=0, n_epoch=500)
    preds = np.array(model.predict(net, X_val))
    val_losses.append(float(np.mean((preds - Y_val)**2)))
print(f"Mean CV MSE: {np.mean(val_losses):.4f} +/- {np.std(val_losses):.4f}")
```

---

## 3. Architecture Design

### 3.1 Structure Definition

```python
structure = [
    {"type": "input",  "units": n_features},
    {"type": "dense",  "units": 32, "activation_function": "tanh",   "bias": True},
    {"type": "dense",  "units": 16, "activation_function": "tanh",   "bias": True},
    {"type": "dense",  "units": n_outputs, "activation_function": "linear", "bias": True},
]
```

**Rules of thumb:**

| Task | Output units | Output activation | Loss |
|------|-------------|------------------|------|
| Regression | 1 | linear | mse |
| Binary classification | 1 | sigmoid | binary_cross_entropy |
| Multi-output regression | k | linear | mse |

**Hidden layer sizing:**
- Start with 1-2 hidden layers
- Start with 16-64 neurons per layer (rule of thumb: 2x input features)
- Wider/deeper = more capacity, but more data needed
- More layers = captures hierarchical features

### 3.2 Bias Terms

Setting `"bias": True` adds one extra input (always value 1) to each neuron:
```
v_j = w_j0*1 + w_j1*y_1 + ... + w_jn*y_n
```
The `w_j0` is the bias weight. Without bias, the decision boundary must pass through
the origin â€” with bias, it can shift freely.

**Always use bias in practice** unless you have a specific reason not to.

---

## 4. Weight Initialization

### 4.1 Why Initialization Matters

All-zeros: every neuron computes the same gradient â†’ symmetric weights â†’ network
fails to learn (symmetry breaking problem).

All-large: activations saturate (sigmoid/tanh) â†’ vanishing gradient.

Ideal: initial weights should keep activation variances stable across layers.

### 4.2 Random Normal (current default in nn_core)

```python
w = np.random.randn(n, m + int(bias)) * 0.2
```

Small random weights centered on 0. The `* 0.2` scale is a heuristic.

**Limitation:** Not theoretically grounded. May need tuning per architecture.

### 4.3 Glorot Normal (Xavier Normal) â€” used in experiments/20251224_v2.py

```python
sigma = np.sqrt(2.0 / (n_in + n_out))   # Glorot formula
w = rng.normal(0.0, sigma, size=(in_features, out_features))
```

**Theory:** Designed to keep the variance of activations and gradients the same
across layers for linear activations. Derivation:

For layer l with n_in inputs and n_out outputs, after weight initialization:
```
Var(y^(l)) = n_in * Var(w) * Var(y^(l-1))
```
For variance to stay 1, we need:
```
Var(w) = 1/n_in   (forward pass stability)
Var(w) = 1/n_out  (backward pass stability)
Var(w) = 2/(n_in + n_out)  (Glorot: compromise between the two)
sigma  = sqrt(2/(n_in + n_out))
```

**Best for:** sigmoid, tanh activations.

### 4.4 He Initialization (Kaiming Normal)

```python
sigma = np.sqrt(2.0 / n_in)   # He formula (factor 2 for ReLU)
w = rng.normal(0.0, sigma, size=(out, n_in))
```

**Theory:** For ReLU activations, roughly half the neurons are set to 0 (negative v),
so we need twice the variance to compensate:
```
Var(w) = 2/n_in   â†’  sigma = sqrt(2/n_in)
```

**Best for:** ReLU and its variants.
**Not in nn_core yet** â€” add if training deep ReLU networks.

---

## 5. Optimizers

### 5.1 SGD â€” Stochastic Gradient Descent (in nn_core/network.py)

```python
W = W - lr * gradient
```

**Pros:** Simple, well-understood, works well with proper lr schedule.
**Cons:** Sensitive to learning rate. One global lr for all weights.

**Online SGD** (one update per sample â€” what nn_core does):
- High variance in updates, but fast convergence in early epochs
- Good for noisy problems (noise helps escape local minima)

**Mini-batch SGD** (not in nn_core â€” could add by accumulating gradients):
- Balance between online and batch
- Better GPU utilization

### 5.2 Adagrad â€” Adaptive Gradient (in experiments/20251224_v2.py)

```python
G_w += dW * dW                          # accumulate squared gradients
W   -= (lr / (sqrt(G_w) + eps)) * dW   # per-parameter adaptive learning rate
```

**Idea:** Weights that receive large gradients (frequently-updated features) get
smaller effective learning rate. Rarely-updated features get larger effective lr.

**Advantage:** Automatically adjusts lr per parameter â€” useful for sparse features.
**Disadvantage:** G_w grows monotonically â†’ effective lr eventually â†’ 0, training stops.

### 5.3 Adam (NOT in this codebase â€” would add as follows)

```python
m = beta1 * m + (1 - beta1) * dW        # momentum (1st moment)
v = beta2 * v + (1 - beta2) * dW**2     # RMS (2nd moment)
m_hat = m / (1 - beta1**t)              # bias correction
v_hat = v / (1 - beta2**t)
W -= lr * m_hat / (sqrt(v_hat) + eps)
```

Typical: lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8

**Most commonly used in practice.** If you add it, add to `nn_core/network.py` as
an alternative update rule (similar to Adagrad pattern in `20251224_v2.py`).

---

## 6. Early Stopping

Stops training when the model stops improving on the validation set â€” prevents overfitting.

```python
# Implementation in nn_core/network.py
def basic_early_stop(self, history_test, epsilon) -> bool:
    improvement = history_test[-2] - history_test[-1]
    return improvement < epsilon   # True = stop (loss improvement below threshold)
```

**Parameters:**
```python
model.train(net, X_tr, Y_tr, X_te, Y_te,
    epsilon=1e-4,   # stop when test improvement < 1e-4
    n_epoch=2000,   # max epochs (safety bound)
)
```

**Triggering after epoch 3**: the condition only fires after `epoch > 3` to avoid
stopping immediately on noisy early epochs.

**Tuning epsilon:**
- Too large (e.g., 0.1): stops too early, underfitting
- Too small (e.g., 1e-8): rarely triggers, like no early stopping
- Good range: 1e-3 to 1e-5 depending on the loss scale

---

## 7. Gradient Clipping

Prevents exploding gradients by normalizing the gradient vector if its norm exceeds a threshold:

```python
norm = np.linalg.norm(delta)
if norm > grad_clip:
    delta = delta * (grad_clip / norm)   # scale to exactly grad_clip length
```

**When to use:**
- High learning rates (lr > 0.1)
- Deep networks (5+ layers)
- When NaN appears in loss
- Recurrent networks (not in this codebase but common there)

**Setting `grad_clip`:**
```python
model.train(net, X, Y, grad_clip=1.0)   # clip to norm 1.0
```

Start with 1.0 â€” if gradients are well-behaved, it rarely triggers.

---

## 8. Complete Training Call

```python
import numpy as np
from nn_core import NeuralNetwork
from modules.data_utils import make_regression_data, train_test_split, normalize
from modules.metrics import mse, rmse, r2_score
from modules.plot_utils import plot_predictions, plot_loss_history

# 1. Data
X, Y = make_regression_data(n=200, noise=0.1, seed=42)
X_tr, Y_tr, X_te, Y_te = train_test_split(X, Y, test_ratio=0.2)
X_tr_n, mu, sig = normalize(X_tr)
X_te_n, _, _    = normalize(X_te, mean=mu, std=sigma)

# 2. Architecture
structure = [
    {"type": "input",  "units": 1},
    {"type": "dense",  "units": 32, "activation_function": "tanh",   "bias": True},
    {"type": "dense",  "units": 16, "activation_function": "tanh",   "bias": True},
    {"type": "dense",  "units": 1,  "activation_function": "linear", "bias": True},
]
model = NeuralNetwork()
net   = model.create_network(structure)

# 3. Inspect before training
print(model.summary(net))

# 4. Train
final_loss, h_train, h_test = model.train(
    net, X_tr_n, Y_tr,
    x_test=X_te_n, y_test=Y_te,
    l_rate=0.02,
    n_epoch=1000,
    loss_function="mse",
    epsilon=1e-5,        # early stopping
    grad_clip=1.0,       # prevent NaN
    verbose=1,
    save_plot="outputs/loss.png",
    save_history="outputs/history.csv",
)

# 5. Evaluate
preds = np.array(model.predict(net, X_te_n)).flatten()
Y_te_flat = Y_te.flatten()
print(f"MSE:  {mse(Y_te_flat, preds):.4f}")
print(f"RMSE: {rmse(Y_te_flat, preds):.4f}")
print(f"RÂ²:   {r2_score(Y_te_flat, preds):.4f}")

# 6. Visualise
plot_predictions(X_te, Y_te, preds, save_path="outputs/fit.png")

# 7. Persist
model.save_weights(net, "outputs/my_model")    # saves as .npz
```

---

## 9. Hyperparameter Tuning Guide

| Hyperparameter | Default | If loss not decreasing | If loss oscillates | If diverges (NaN) |
|----------------|---------|----------------------|-------------------|-------------------|
| `l_rate` | 0.01 | Try 10x larger | Try 10x smaller | Try 100x smaller |
| `n_epoch` | 100 | Increase | - | - |
| hidden units | 16 | Increase | - | - |
| `epsilon` | 0.0 | Decrease | - | - |
| `grad_clip` | None | - | Add 1.0 | Add 1.0 |
| normalization | off | Always normalize! | Always normalize! | Check for inf |

---

## 10. Key Code Locations

| Task | File | Function/Method |
|------|------|----------------|
| Complete training pipeline | `examples/01_regression.py` | Full script |
| Training loop | `nn_core/network.py` | `train()` |
| Gradient clipping | `nn_core/network.py` | `_update_weights_clipped()` |
| Early stopping | `nn_core/network.py` | `basic_early_stop()` |
| Model summary | `nn_core/network.py` | `summary()` |
| Save/load weights | `nn_core/network.py` | `save_weights()`, `load_weights()` |
| Data generation | `modules/data_utils.py` | `make_*`, `train_test_split`, `normalize` |
| Cross-validation | `modules/data_utils.py` | `k_fold_split()` |
| CSV loading | `modules/data_utils.py` | `load_csv()` |
| Evaluation metrics | `modules/metrics.py` | `mse`, `r2_score`, `accuracy`, etc. |
| Plots | `modules/plot_utils.py` | `plot_loss_history`, `plot_predictions`, etc. |
| Glorot init + Adagrad | `experiments/20251224_v2.py` | `NeuralNetCore` class |

