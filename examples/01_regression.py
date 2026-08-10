"""
Example 1 — Regression: learn sin(2x)+cos(x)+5 from noisy data

What this script demonstrates
──────────────────────────────
  • Generating a 1-D nonlinear regression dataset with make_regression_data()
  • Splitting into train / test sets with train_test_split()
  • Z-score normalising both inputs (X) and targets (Y) for stable training
  • Building a 1 → 32 → 16 → 1 MLP with tanh hidden layers and linear output
  • Running a manual training loop so the loss history is available for plotting
  • Applying early stopping based on test-loss plateau detection
  • Denormalising model predictions back to the original data scale
  • Saving the trained weights and plots to outputs/

Run from the project root:
    python examples/01_regression.py
"""

# ── 1. Standard library ───────────────────────────────────────────────────────
import os
import sys

# Suppress interactive matplotlib windows before any import touches the backend.
# Setting this early guarantees Agg is active even if plot_utils is imported
# inside another module.  Override by setting MPLBACKEND in your shell.
os.environ.setdefault("MPLBACKEND", "Agg")

# ── 2. Path setup: makes src/ importable without pip install -e . ─────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── 3. Third-party ───────────────────────────────────────────────────────────
import numpy as np

# ── 4. Application — nn_core ──────────────────────────────────────────────────
from nn_core import NeuralNetwork, LossFn, ObjLogger
from nn_core.logger import title_message

# ── 5. Application — modules ──────────────────────────────────────────────────
from modules.data_utils import make_regression_data, train_test_split, normalize
from modules.general_utils import ensure_directory
from modules.plot_utils import plot_loss_history, plot_predictions

# =============================================================================
# Logger — all progress messages go through this; only final results use print()
# =============================================================================
log = ObjLogger("01_regression")

# =============================================================================
# Hyperparameters — change these to experiment
# =============================================================================
N_SAMPLES  = 200      # total data points generated
NOISE      = 0.15     # Gaussian noise std added to targets
TEST_RATIO = 0.20     # 20 % held out for testing
L_RATE     = 0.02     # SGD learning rate
N_EPOCH    = 800      # maximum training epochs
EPSILON    = 1e-5     # early-stopping threshold (stop when Δloss < EPSILON)
LOSS_FN    = "mse"    # loss function — "mse" is Mean Squared Error
SEED       = 42       # reproducibility

# =============================================================================
# Output directory — all plots, weights, and logs land here
# =============================================================================
_HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.normpath(os.path.join(_HERE, "..", "outputs"))
ensure_directory(OUT_DIR)   # creates outputs/ if it doesn't exist
log(f"Outputs will be saved to: {OUT_DIR}", color="cyan")

# =============================================================================
# Step 1 — Generate dataset
# =============================================================================
title_message("Step 1: Generate Data", color="blue")

# Target function:  y = sin(2x) + cos(x) + 5 + Gaussian_noise
# This is a smooth nonlinear function that an MLP can approximate well.
X, Y = make_regression_data(n=N_SAMPLES, noise=NOISE, seed=SEED)

log(f"Generated  X: shape={X.shape}  Y: shape={Y.shape}", color="cyan")
log(f"X range: [{X.min():.2f}, {X.max():.2f}]  |  Y range: [{Y.min():.2f}, {Y.max():.2f}]", color="cyan")

# =============================================================================
# Step 2 — Train / test split
# =============================================================================
title_message("Step 2: Train/Test Split", color="blue")

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_ratio=TEST_RATIO, seed=SEED)

# Keep originals for denormalised plotting after training
X_test_orig = X_test.copy()
Y_test_orig  = Y_test.copy()

log(f"Train set: {len(X_train)} samples  |  Test set: {len(X_test)} samples", color="cyan")

# =============================================================================
# Step 3 — Normalize inputs and targets
# =============================================================================
title_message("Step 3: Normalize", color="blue")

# Normalising X: maps [-3, 3] → roughly [-1.5, 1.5] (zero-mean, unit-variance).
# Normalising Y: maps [~3, ~7]  → roughly [-2, 2] (same idea).
# This gives the optimiser a smoother loss landscape and speeds up convergence.
# We store the training-set statistics (mu, sigma) so we can apply them to the
# test set and then REVERSE them on the predictions.

X_train_n, X_mu, X_sig = normalize(X_train)
X_test_n, _, _          = normalize(X_test, mean=X_mu, std=X_sig)

Y_train_n, Y_mu, Y_sig = normalize(Y_train)
Y_test_n, _, _          = normalize(Y_test, mean=Y_mu, std=Y_sig)

log(f"X  after normalise: mean~{X_train_n.mean():.3f}  std~{X_train_n.std():.3f}", color="cyan")
log(f"Y  after normalise: mean~{Y_train_n.mean():.3f}  std~{Y_train_n.std():.3f}", color="cyan")

# =============================================================================
# Step 4 — Build network:  1 → 32 → 16 → 1
# =============================================================================
title_message("Step 4: Build Network", color="blue")

#  Input layer:  1 feature  (x)
#  Hidden 1:     32 neurons  tanh activation + bias
#  Hidden 2:     16 neurons  tanh activation + bias
#  Output layer:  1 neuron  linear activation (regression → unbounded output)
#
#  bias=True appends a constant "1" to each layer's input, giving each
#  neuron its own trainable intercept term.

structure = [
    {"type": "input",  "units": 1},
    {"type": "dense",  "units": 32, "activation_function": "tanh",   "bias": True},
    {"type": "dense",  "units": 16, "activation_function": "tanh",   "bias": True},
    {"type": "dense",  "units": 1,  "activation_function": "linear", "bias": True},
]

model = NeuralNetwork()
net   = model.create_network(structure)

# __str__ / __repr__ prints a formatted architecture table
print(model)

# =============================================================================
# Step 5 — Training loop
# =============================================================================
title_message("Step 5: Train", color="blue")
log(f"lr={L_RATE}  epochs={N_EPOCH}  eps={EPSILON}  loss={LOSS_FN}", color="yellow")

# We run the loop manually (instead of model.train()) so we can collect the
# per-epoch loss history for plotting.  The three model methods used are:
#   model.forward_propagate  — run inputs through the network
#   model.backward_propagate — compute per-neuron deltas (error signals)
#   model.update_weights     — apply the SGD update  W ← W − lr * δ ⊗ a_prev

loss_fn       = LossFn()        # used to compute epoch loss values for history
history_train = []              # training loss per epoch
history_test  = []              # test loss per epoch (needed for early stopping)

for epoch in range(N_EPOCH):

    # ── Training pass: forward + backward + weight update ────────────────────
    err_sum = 0.0
    for x_row, y_row in zip(X_train_n, Y_train_n):
        x_arr = np.asarray(x_row, dtype=float)
        y_arr = np.asarray(y_row, dtype=float)

        model.forward_propagate(net, x_arr)
        model.backward_propagate(LOSS_FN, net, y_arr)
        model.update_weights(net, x_arr, L_RATE)

        # Accumulate per-sample loss for the epoch average
        err_sum += float(np.sum(loss_fn.output(LOSS_FN, y_arr, net[-1]["output"])))

    history_train.append(err_sum / len(X_train_n))

    # ── Test pass: forward only (no weight update) ────────────────────────────
    t_err = 0.0
    for x_row, y_row in zip(X_test_n, Y_test_n):
        model.forward_propagate(net, np.asarray(x_row, dtype=float))
        t_err += float(
            np.sum(
                loss_fn.output(LOSS_FN, np.asarray(y_row, dtype=float), net[-1]["output"])
            )
        )
    history_test.append(t_err / len(X_test_n))

    # ── Progress log every 100 epochs ─────────────────────────────────────────
    if (epoch + 1) % 100 == 0:
        log(
            f"  epoch {epoch + 1:>4}  "
            f"train_loss={history_train[-1]:.5f}  "
            f"test_loss={history_test[-1]:.5f}",
            color="white",
        )

    # ── Early stopping: stop when test-loss improvement plateaus ──────────────
    # basic_early_stop returns True when  (loss[-2] − loss[-1]) < epsilon.
    # We wait for at least 4 epochs before checking so the history is stable.
    if epoch > 3 and EPSILON > 0:
        if model.basic_early_stop(history_test, EPSILON):
            delta = history_test[-2] - history_test[-1]
            log(
                f"Early stop at epoch {epoch + 1}  "
                f"(d_loss={delta:.2e} < eps={EPSILON})",
                color="yellow",
            )
            break

log(f"Training complete.  Ran {len(history_train)} epochs.", color="green")

# =============================================================================
# Step 6 — Evaluate: predict and denormalise
# =============================================================================
title_message("Step 6: Evaluate", color="blue")

# Predict on the *normalised* test inputs → get normalised predictions
preds_norm = model.predict(net, X_test_n)           # list of (1,) arrays

# Flatten each prediction to a scalar, then stack into a 1-D array
preds_norm_vals = np.array([p.flatten()[0] for p in preds_norm])

# Denormalise: reverse the Y normalisation applied in Step 3
#   y_original = y_normalised * sigma_Y + mu_Y
y_sigma = float(Y_sig.flatten()[0])
y_mu    = float(Y_mu.flatten()[0])
preds_denorm = preds_norm_vals * y_sigma + y_mu

# MSE on the ORIGINAL (non-normalised) scale — the meaningful metric
test_mse = np.mean((preds_denorm - Y_test_orig.flatten()) ** 2)

log(f"Test MSE (original scale): {test_mse:.6f}", color="green")

# =============================================================================
# Step 7 — Save plots
# =============================================================================
title_message("Step 7: Save Plots", color="blue")

# Scatter the true test values and overlay the model's predicted curve.
# X_test_orig and Y_test_orig are in the original (un-normalised) space.
plot_predictions(
    X_test_orig,
    Y_test_orig,
    preds_denorm,
    title="Regression fit — sin(2x) + cos(x) + 5",
    save_path=os.path.join(OUT_DIR, "01_regression_fit.png"),
)

# Loss curves let you diagnose over/underfitting at a glance.
# A large gap between train and test loss → overfitting.
# Both curves flat early → underfitting (try a wider network or more epochs).
plot_loss_history(
    history_train,
    history_test,
    title="Training history — regression",
    save_path=os.path.join(OUT_DIR, "01_regression_loss.png"),
)

# =============================================================================
# Step 8 — Save trained weights
# =============================================================================
title_message("Step 8: Save Weights", color="blue")

# Weights are saved as a .npz archive (pure numpy, no pickle).
# Reload them with:  model.load_weights(net, "outputs/01_regression_weights")
# save_weights uses np.savez; the .npz extension is added automatically.
# Pass the base name (no extension) for a clean filename: 01_regression_weights.npz
weights_path = os.path.join(OUT_DIR, "01_regression_weights")
model.save_weights(net, weights_path)

# =============================================================================
# Final summary (plain print — no logger colouring so it's easy to copy)
# =============================================================================
print("\n" + "=" * 55)
print("EXAMPLE 1 - REGRESSION  (final results)")
print("=" * 55)
print(f"  Dataset           : sin(2x)+cos(x)+5  n={N_SAMPLES}, noise={NOISE}")
print(f"  Network           : 1 -> 32 -> 16 -> 1  (tanh/tanh/linear)")
print(f"  Epochs run        : {len(history_train)}")
print(f"  Final train loss  : {history_train[-1]:.6f}  (normalised MSE)")
print(f"  Final test loss   : {history_test[-1]:.6f}  (normalised MSE)")
print(f"  Test MSE (orig.)  : {test_mse:.6f}")
print(f"  Fit plot          : outputs/01_regression_fit.png")
print(f"  Loss plot         : outputs/01_regression_loss.png")
print(f"  Weights           : outputs/01_regression_weights.npz")
print("=" * 55 + "\n")
