"""
Example 3 — Load your own CSV data and train a regression model

What this script demonstrates
──────────────────────────────
  • Writing and reading CSV files using Python's built-in csv module (no pandas)
  • Loading arbitrary numeric CSV data into numpy arrays
  • Training a 1 → 16 → 1 regression MLP on the loaded data
  • Saving predictions and loss plots to outputs/

HOW TO USE YOUR OWN CSV
────────────────────────
  1. FORMAT
       Your CSV must have a header row with column names, and
       all data rows must contain numeric values only.

       Example — two columns:
           x,y
           0.01,2.43
           0.02,2.71
           ...

       Example — multiple features + one target:
           area,rooms,distance,price
           52.5,2,1.2,180000
           75.0,3,0.8,245000
           ...

  2. POINT THIS SCRIPT AT YOUR FILE
       Change the variable CSV_PATH to the path of your file:
           CSV_PATH = "/path/to/your/data.csv"

  3. CHOOSE FEATURE AND TARGET COLUMNS
       Change the variables below:
           X_COLS = ["area", "rooms", "distance"]   # input features
           Y_COL  = "price"                          # prediction target

       For a single feature use a list with one element:
           X_COLS = ["x"]
           Y_COL  = "y"

  4. MULTIPLE INPUT FEATURES
       If X_COLS has N columns, change the input layer in the network
       structure from  {"units": 1}  to  {"units": N}.
       Everything else stays the same.

  5. MISSING VALUES / NON-NUMERIC ROWS
       Add a filter when loading:
           if "" in row.values():   continue   # skip blanks
           if any(...):             continue   # custom check

Run from the project root:
    python examples/03_custom_csv_data.py
"""

# ── 1. Standard library ───────────────────────────────────────────────────────
import csv
import os
import sys

# Force Agg (non-interactive) matplotlib backend before any import touches it.
os.environ.setdefault("MPLBACKEND", "Agg")

# ── 2. Path setup: makes src/ importable without pip install -e . ─────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── 3. Third-party ───────────────────────────────────────────────────────────
import numpy as np

# ── 4. Application — nn_core ──────────────────────────────────────────────────
from nn_core import NeuralNetwork, LossFn, ObjLogger
from nn_core.logger import title_message

# ── 5. Application — modules ──────────────────────────────────────────────────
from modules.data_utils import normalize, train_test_split
from modules.general_utils import ensure_directory
from modules.plot_utils import plot_loss_history, plot_predictions

# =============================================================================
# Logger
# =============================================================================
log = ObjLogger("03_custom_csv")

# =============================================================================
# ── CONFIGURE THESE to point at YOUR data ────────────────────────────────────
#
#   CSV_PATH : path to your CSV file (absolute or relative to project root)
#   X_COLS   : list of column names to use as INPUT features
#   Y_COL    : column name to use as the regression TARGET
#
# The defaults below generate a sample CSV and immediately load it,
# so the script works out-of-the-box with no real data required.
# =============================================================================
_HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.normpath(os.path.join(_HERE, "..", "outputs"))
ensure_directory(OUT_DIR)

CSV_PATH = os.path.join(OUT_DIR, "sample_data.csv")   # ← change to your file
X_COLS   = ["x"]                                       # ← your feature columns
Y_COL    = "y"                                         # ← your target column

# =============================================================================
# Hyperparameters
# =============================================================================
N_POINTS   = 150     # number of synthetic data points to generate for the demo
TEST_RATIO = 0.20    # 20 % held out for testing
L_RATE     = 0.015   # SGD learning rate
N_EPOCH    = 600     # maximum training epochs
EPSILON    = 1e-5    # early-stopping threshold
LOSS_FN    = "mse"   # Mean Squared Error
SEED       = 99      # controls data generation reproducibility

# =============================================================================
# Step 1 — Generate a sample CSV
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
#   SKIP THIS SECTION when using your own file.
#   Just set CSV_PATH above and jump to Step 2.
# ─────────────────────────────────────────────────────────────────────────────
title_message("Step 1: Generate Sample CSV", color="blue")

# Target function: y = 0.5·x² − x + 2 + noise
# This is a simple parabola — easy to visualise and fast to train.
rng    = np.random.default_rng(SEED)
x_vals = np.linspace(-4.0, 4.0, N_POINTS)
y_vals = 0.5 * x_vals ** 2 - x_vals + 2.0 + rng.normal(0.0, 0.4, N_POINTS)

# Write as CSV with a header row.
# csv.writer handles quoting and line endings correctly on all platforms.
with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(["x", "y"])                        # header row (column names)
    for xv, yv in zip(x_vals, y_vals):
        writer.writerow([round(float(xv), 6), round(float(yv), 6)])

log(f"Sample CSV written: {CSV_PATH}  ({N_POINTS} rows)", color="cyan")

# =============================================================================
# Step 2 — Load the CSV back using Python's csv module
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
#   THIS IS THE SECTION TO ADAPT for your own data.
#   Point CSV_PATH, X_COLS, Y_COL at your file and columns.
# ─────────────────────────────────────────────────────────────────────────────
title_message("Step 2: Load CSV", color="blue")

# Lists to collect each column while iterating
x_lists = {col: [] for col in X_COLS}   # one list per feature column
y_list  = []                             # one list for the target column

# csv.DictReader turns every row into an {column_name: value} dict,
# so we can look up columns by name rather than position.
with open(CSV_PATH, newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)

    # Validate the header: make sure the columns we asked for actually exist.
    expected_cols = set(X_COLS) | {Y_COL}
    missing = expected_cols - set(reader.fieldnames or [])
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {missing}\n"
            f"  Found columns: {reader.fieldnames}"
        )

    for row_num, row in enumerate(reader, start=2):   # row 1 is the header
        try:
            for col in X_COLS:
                x_lists[col].append(float(row[col]))
            y_list.append(float(row[Y_COL]))
        except ValueError as exc:
            # Skip rows that cannot be converted to float (e.g. text, blanks).
            log(f"  Skipping row {row_num}: {exc}", color="yellow")
            continue

n_loaded = len(y_list)
log(f"Loaded {n_loaded} rows from {os.path.basename(CSV_PATH)}", color="cyan")

# =============================================================================
# Step 3 — Build numpy arrays from the loaded lists
# =============================================================================
title_message("Step 3: Prepare Arrays", color="blue")

# Stack feature columns into a 2-D matrix (n_samples × n_features).
# If there is only one feature, X still has shape (n_samples, 1) — the MLP
# expects 2-D input arrays, not flat scalars.
if len(X_COLS) == 1:
    X = np.array(x_lists[X_COLS[0]], dtype=np.float64).reshape(-1, 1)
else:
    X = np.column_stack(
        [np.array(x_lists[col], dtype=np.float64) for col in X_COLS]
    )

Y = np.array(y_list, dtype=np.float64).reshape(-1, 1)   # always (n_samples, 1)

log(f"X: shape={X.shape}  range=[{X.min():.3f}, {X.max():.3f}]", color="cyan")
log(f"Y: shape={Y.shape}  range=[{Y.min():.3f}, {Y.max():.3f}]", color="cyan")

n_features = X.shape[1]
log(f"Number of input features: {n_features}", color="cyan")

# =============================================================================
# Step 4 — Split and normalise
# =============================================================================
title_message("Step 4: Split and Normalise", color="blue")

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_ratio=TEST_RATIO, seed=SEED)
X_test_orig = X_test.copy()   # keep originals for plotting (before normalisation)
Y_test_orig  = Y_test.copy()

# Normalise X and Y so the network sees zero-mean, unit-variance data.
X_train_n, X_mu, X_sig = normalize(X_train)
X_test_n, _, _          = normalize(X_test, mean=X_mu, std=X_sig)

Y_train_n, Y_mu, Y_sig = normalize(Y_train)
Y_test_n, _, _          = normalize(Y_test, mean=Y_mu, std=Y_sig)

log(f"Train: {len(X_train)} samples  |  Test: {len(X_test)} samples", color="cyan")

# =============================================================================
# Step 5 — Build network
# =============================================================================
# The input layer units must equal n_features (the number of columns in X).
# If you have 3 features, change units=1 to units=3 here:
title_message("Step 5: Build Network", color="blue")

structure = [
    {"type": "input",  "units": n_features},           # ← auto-set from data
    {"type": "dense",  "units": 16, "activation_function": "tanh",   "bias": True},
    {"type": "dense",  "units": 1,  "activation_function": "linear", "bias": True},
]

model = NeuralNetwork()
net   = model.create_network(structure)
print(model)

# =============================================================================
# Step 6 — Training loop
# =============================================================================
title_message("Step 6: Train", color="blue")
log(f"lr={L_RATE}  epochs={N_EPOCH}  eps={EPSILON}  loss={LOSS_FN}", color="yellow")

loss_fn       = LossFn()
history_train = []
history_test  = []

for epoch in range(N_EPOCH):

    # ── Training pass ─────────────────────────────────────────────────────────
    err_sum = 0.0
    for x_row, y_row in zip(X_train_n, Y_train_n):
        x_arr = np.asarray(x_row, dtype=float)
        y_arr = np.asarray(y_row, dtype=float)

        model.forward_propagate(net, x_arr)
        model.backward_propagate(LOSS_FN, net, y_arr)
        model.update_weights(net, x_arr, L_RATE)

        err_sum += float(np.sum(loss_fn.output(LOSS_FN, y_arr, net[-1]["output"])))

    history_train.append(err_sum / len(X_train_n))

    # ── Test evaluation ────────────────────────────────────────────────────────
    t_err = 0.0
    for x_row, y_row in zip(X_test_n, Y_test_n):
        model.forward_propagate(net, np.asarray(x_row, dtype=float))
        t_err += float(
            np.sum(
                loss_fn.output(LOSS_FN, np.asarray(y_row, dtype=float), net[-1]["output"])
            )
        )
    history_test.append(t_err / len(X_test_n))

    # ── Progress ──────────────────────────────────────────────────────────────
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
# Step 7 — Evaluate and denormalise
# =============================================================================
title_message("Step 7: Evaluate", color="blue")

preds_norm = model.predict(net, X_test_n)
preds_norm_vals = np.array([p.flatten()[0] for p in preds_norm])

# Reverse Y normalisation:  y_orig = y_norm * sigma_Y + mu_Y
y_sigma = float(Y_sig.flatten()[0])
y_mu_val = float(Y_mu.flatten()[0])
preds_denorm = preds_norm_vals * y_sigma + y_mu_val

test_mse  = np.mean((preds_denorm - Y_test_orig.flatten()) ** 2)
test_rmse = np.sqrt(test_mse)

log(f"Test MSE  (orig. scale): {test_mse:.6f}", color="green")
log(f"Test RMSE (orig. scale): {test_rmse:.6f}", color="green")

# =============================================================================
# Step 8 — Save plots
# =============================================================================
title_message("Step 8: Save Plots", color="blue")

# plot_predictions works for single-feature problems (x-axis = X, y-axis = Y).
# For multi-feature problems you would need a different visualisation
# (e.g. predicted vs actual scatter, or per-feature partial plots).
if n_features == 1:
    plot_predictions(
        X_test_orig,
        Y_test_orig,
        preds_denorm,
        title="Custom CSV fit — 0.5·x² − x + 2",
        xlabel=X_COLS[0],
        ylabel=Y_COL,
        save_path=os.path.join(OUT_DIR, "03_custom_fit.png"),
    )
else:
    # For multi-feature data: plot predicted vs actual as a scatter
    log("Multi-feature data: plotting predicted vs actual (not x vs y)", color="yellow")
    from modules.plot_utils import _get_plt            # internal helper
    plt = _get_plt()
    plt.figure(figsize=(6, 5))
    plt.scatter(Y_test_orig.flatten(), preds_denorm, s=20, alpha=0.7)
    perfect = [Y_test_orig.min(), Y_test_orig.max()]
    plt.plot(perfect, perfect, "r--", label="Perfect fit")
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.legend(); plt.grid(True, alpha=0.4)
    plt.title("Predicted vs Actual (multi-feature)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_custom_fit.png"), bbox_inches="tight", dpi=100)
    plt.close()
    log(f"Predicted-vs-actual plot saved", color="cyan")

plot_loss_history(
    history_train,
    history_test,
    title="Training history — custom CSV regression",
    save_path=os.path.join(OUT_DIR, "03_custom_loss.png"),
)

# =============================================================================
# Final summary
# =============================================================================
print("\n" + "=" * 60)
print("EXAMPLE 3 - CUSTOM CSV REGRESSION  (final results)")
print("=" * 60)
print(f"  CSV file         : {CSV_PATH}")
print(f"  Features (X)     : {X_COLS}")
print(f"  Target   (Y)     : {Y_COL}")
print(f"  Samples loaded   : {n_loaded}  (train={len(X_train)}, test={len(X_test)})")
print(f"  Network          : {n_features} -> 16 -> 1  (tanh/linear)")
print(f"  Epochs run       : {len(history_train)}")
print(f"  Final train loss : {history_train[-1]:.6f}  (normalised MSE)")
print(f"  Final test loss  : {history_test[-1]:.6f}  (normalised MSE)")
print(f"  Test MSE (orig.) : {test_mse:.6f}")
print(f"  Test RMSE (orig.): {test_rmse:.6f}")
print(f"  Fit plot         : outputs/03_custom_fit.png")
print(f"  Loss plot        : outputs/03_custom_loss.png")
print("=" * 60 + "\n")

# =============================================================================
# ── QUICK REFERENCE: adapting this script to YOUR CSV ────────────────────────
#
#  Scenario A — single numeric feature "temperature", target "power":
#      CSV_PATH = "data/solar.csv"
#      X_COLS   = ["temperature"]
#      Y_COL    = "power"
#      # Network input layer units=1 (already correct)
#
#  Scenario B — three features "area", "rooms", "distance", target "price":
#      CSV_PATH = "data/housing.csv"
#      X_COLS   = ["area", "rooms", "distance"]
#      Y_COL    = "price"
#      # Change input layer to: {"type": "input", "units": 3}
#      # (the script does this automatically via n_features)
#
#  Scenario C — raw data with some non-numeric rows:
#      # The try/except block in Step 2 already skips bad rows silently.
#      # To fail loudly instead, remove the try/except and let ValueError raise.
#
#  Scenario D — classification from CSV (labels 0/1):
#      # Change LOSS_FN = "binary_cross_entropy"
#      # Change output layer activation from "linear" to "sigmoid"
#      # Interpret output >= 0.5 as class 1
#
# =============================================================================
