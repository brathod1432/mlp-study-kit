"""
Example 2 — Binary classification: learn a 2D decision boundary

What this script demonstrates
──────────────────────────────
  • Generating a 2-class, 2-feature synthetic dataset with make_classification_data()
  • Splitting and normalising 2-D inputs
  • Building a 2 → 16 → 8 → 1 MLP with sigmoid output for binary classification
  • Training with Binary Cross-Entropy loss (the right choice for 0/1 targets)
  • Computing accuracy and a full confusion matrix using pure numpy
  • Visualising the learned decision boundary with plot_decision_boundary()
  • Understanding what makes a good vs. bad decision boundary from the plots

Run from the project root:
    python examples/02_classification.py
"""

# ── 1. Standard library ───────────────────────────────────────────────────────
import os
import sys

# Force Agg backend before any code path can import matplotlib interactively.
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
from modules.data_utils import make_classification_data, train_test_split, normalize
from modules.general_utils import ensure_directory
from modules.plot_utils import plot_decision_boundary, plot_loss_history

# =============================================================================
# Logger — progress via ObjLogger; final numbers via plain print()
# =============================================================================
log = ObjLogger("02_classification")

# =============================================================================
# Hyperparameters
# =============================================================================
N_PER_CLASS = 100     # samples per class (total = 2 × N_PER_CLASS = 200)
TEST_RATIO  = 0.25    # 25 % held out → 50 test samples
L_RATE      = 0.01    # learning rate  (lower than regression — BCE gradients can be larger)
N_EPOCH     = 1000    # maximum training epochs
EPSILON     = 1e-5    # early-stopping threshold
LOSS_FN     = "binary_cross_entropy"   # BCE is the correct loss for binary labels
THRESHOLD   = 0.5     # sigmoid output ≥ 0.5 → predicted class 1
SEED        = 42

# =============================================================================
# Output directory
# =============================================================================
_HERE   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(_HERE, "..", "outputs"))
ensure_directory(OUT_DIR)
log(f"Outputs will be saved to: {OUT_DIR}", color="cyan")

# =============================================================================
# Step 1 — Generate dataset
# =============================================================================
title_message("Step 1: Generate Data", color="blue")

# The dataset has two overlapping clusters in 2-D space:
#   Class 0: Uniform([0,2] × [0,2])  — lower-left region
#   Class 1: Uniform([1,3] × [2,4])  — upper-right region
# The partial overlap makes the boundary non-trivial (and realistic).
#
# make_classification_data also returns idx0 and idx1 — index arrays that
# let you select only class-0 or class-1 rows from X.  Useful for plotting,
# though we don't plot the raw data separately here.
X, Y, idx0, idx1 = make_classification_data(n_per_class=N_PER_CLASS, seed=SEED)

log(f"X: shape={X.shape}  Y: shape={Y.shape}", color="cyan")
log(f"Class 0: {(Y == 0).sum()} samples  |  Class 1: {(Y == 1).sum()} samples", color="cyan")

# =============================================================================
# Step 2 — Train / test split
# =============================================================================
title_message("Step 2: Train/Test Split", color="blue")

# Y is 1-D here (shape (200,)).  train_test_split handles any shape.
X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_ratio=TEST_RATIO, seed=SEED)

log(f"Train set: {len(X_train)} samples  |  Test set: {len(X_test)} samples", color="cyan")

# Reshape Y to column vectors so each sample's label is a (1,) array,
# which matches the network's single-neuron output layer.
Y_train = Y_train.reshape(-1, 1)   # (N_train, 1)
Y_test  = Y_test.reshape(-1, 1)    # (N_test,  1)

# =============================================================================
# Step 3 — Normalize features
# =============================================================================
title_message("Step 3: Normalize", color="blue")

# Normalise the 2-D feature matrix.  axis=0 computes a separate mean/std
# for each column (feature), so x₁ and x₂ are scaled independently.
# We use training-set statistics for both train and test to prevent data leakage.
X_train_n, X_mu, X_sig = normalize(X_train)
X_test_n, _, _          = normalize(X_test, mean=X_mu, std=X_sig)

log(f"X after normalise: mean~{np.round(X_train_n.mean(axis=0), 3)}  std~{np.round(X_train_n.std(axis=0), 3)}", color="cyan")

# =============================================================================
# Step 4 — Build network:  2 → 16 → 8 → 1
# =============================================================================
title_message("Step 4: Build Network", color="blue")

#  Input layer:  2 features  (x₁, x₂)
#  Hidden 1:     16 neurons  tanh  — learns intermediate feature combinations
#  Hidden 2:      8 neurons  tanh  — compresses to a richer abstract rep
#  Output layer:  1 neuron   sigmoid → produces a value in (0, 1)
#
#  The sigmoid output is interpreted as P(class=1 | x).
#  Decision boundary:  sigmoid(v) = 0.5  ↔  v = 0  (where v = W·x + b)

structure = [
    {"type": "input",  "units": 2},
    {"type": "dense",  "units": 16, "activation_function": "tanh",    "bias": True},
    {"type": "dense",  "units": 8,  "activation_function": "tanh",    "bias": True},
    {"type": "dense",  "units": 1,  "activation_function": "sigmoid", "bias": True},
]

model = NeuralNetwork()
net   = model.create_network(structure)
print(model)   # formatted architecture table

# =============================================================================
# Step 5 — Training loop
# =============================================================================
title_message("Step 5: Train", color="blue")
log(f"lr={L_RATE}  epochs={N_EPOCH}  eps={EPSILON}  loss={LOSS_FN}", color="yellow")

# Binary Cross-Entropy:  L = −[t·log(ŷ) + (1−t)·log(1−ŷ)]
# Its gradient pushes the output toward 0 when t=0 and toward 1 when t=1.
# BCE is numerically more appropriate for classification than MSE because
# it penalises confident wrong predictions exponentially.

loss_fn       = LossFn()
history_train = []
history_test  = []

for epoch in range(N_EPOCH):

    # ── Training pass ─────────────────────────────────────────────────────────
    err_sum = 0.0
    for x_row, y_row in zip(X_train_n, Y_train):
        x_arr = np.asarray(x_row, dtype=float)
        y_arr = np.asarray(y_row, dtype=float)

        model.forward_propagate(net, x_arr)
        model.backward_propagate(LOSS_FN, net, y_arr)
        model.update_weights(net, x_arr, L_RATE)

        err_sum += float(np.sum(loss_fn.output(LOSS_FN, y_arr, net[-1]["output"])))

    history_train.append(err_sum / len(X_train_n))

    # ── Test evaluation (forward only — no weight updates) ────────────────────
    t_err = 0.0
    for x_row, y_row in zip(X_test_n, Y_test):
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

    # ── Early stopping ────────────────────────────────────────────────────────
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
# Step 6 — Evaluate: accuracy and confusion matrix
# =============================================================================
title_message("Step 6: Evaluate", color="blue")

# Run the trained network on each test sample.
# model.predict() returns a list of (1,) arrays.
preds = model.predict(net, X_test_n)

# Convert predictions to a flat binary array: 1 if output ≥ THRESHOLD else 0
pred_labels = (np.array(preds).flatten() >= THRESHOLD).astype(int)
true_labels = Y_test.flatten().astype(int)

# ── Accuracy ──────────────────────────────────────────────────────────────────
accuracy = np.mean((np.array(preds) >= THRESHOLD).flatten() == Y_test.flatten())
log(f"Test accuracy: {accuracy * 100:.1f}%  ({int(accuracy * len(true_labels))}/{len(true_labels)} correct)", color="green")

# ── Confusion matrix — pure numpy, no sklearn ─────────────────────────────────
#
#                  Predicted
#               ┌──────┬──────┐
#               │  TN  │  FP  │  ← Actual 0
#               ├──────┼──────┤
#               │  FN  │  TP  │  ← Actual 1
#               └──────┴──────┘
#
#  TP  True Positive:  model said 1, truth is 1  (correct)
#  TN  True Negative:  model said 0, truth is 0  (correct)
#  FP  False Positive: model said 1, truth is 0  (wrong — "false alarm")
#  FN  False Negative: model said 0, truth is 1  (wrong — "missed")

TP = int(np.sum((pred_labels == 1) & (true_labels == 1)))
TN = int(np.sum((pred_labels == 0) & (true_labels == 0)))
FP = int(np.sum((pred_labels == 1) & (true_labels == 0)))
FN = int(np.sum((pred_labels == 0) & (true_labels == 1)))

log("Confusion matrix:", color="magenta")
log(f"  TP={TP}  TN={TN}  FP={FP}  FN={FN}", color="magenta")

# Derived metrics (safe against division by zero)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

log(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}", color="magenta")

# =============================================================================
# Step 7 — Save plots
# =============================================================================
title_message("Step 7: Save Plots", color="blue")

# Decision boundary: the network is evaluated on a dense 2-D grid.
# Regions where sigmoid(v) ≥ 0.5 are coloured as Class 1 (orange),
# and regions < 0.5 as Class 0 (blue).  The black contour is the boundary.
# Note: we pass X_test_n (normalised) since the network was trained on it.
plot_decision_boundary(
    model,
    net,
    X_test_n,
    Y_test.flatten(),
    title="Decision boundary — 2-D binary classification",
    save_path=os.path.join(OUT_DIR, "02_classification_boundary.png"),
)

# Loss curves: BCE loss should decrease smoothly.
# If the test curve rises after a certain epoch → overfitting (network memorising train data).
plot_loss_history(
    history_train,
    history_test,
    title="Training history — binary cross-entropy",
    save_path=os.path.join(OUT_DIR, "02_classification_loss.png"),
)

# =============================================================================
# Final summary
# =============================================================================
print("\n" + "=" * 55)
print("EXAMPLE 2 - CLASSIFICATION  (final results)")
print("=" * 55)
print(f"  Dataset           : 2-class 2-D  n_per_class={N_PER_CLASS}")
print(f"  Network           : 2 -> 16 -> 8 -> 1  (tanh/tanh/sigmoid)")
print(f"  Epochs run        : {len(history_train)}")
print(f"  Final train loss  : {history_train[-1]:.6f}  (BCE)")
print(f"  Final test loss   : {history_test[-1]:.6f}  (BCE)")
print(f"  Test accuracy     : {accuracy * 100:.1f}%")
print(f"  Confusion matrix  : TP={TP}  TN={TN}  FP={FP}  FN={FN}")
print(f"  Precision / Recall: {precision:.3f} / {recall:.3f}  F1={f1:.3f}")
print(f"  Boundary plot     : outputs/02_classification_boundary.png")
print(f"  Loss plot         : outputs/02_classification_loss.png")
print("=" * 55 + "\n")
