# examples/ — Runnable End-to-End Examples

Three standalone Python scripts that demonstrate practical, real-world usage of
the `nn_core` and `modules` packages.  Each script runs without any additional
setup — no `pip install -e .` required.

---

## Overview

| # | Script | Task | What you learn | Est. runtime |
|---|--------|------|----------------|--------------|
| 1 | `01_regression.py` | Regress `sin(2x)+cos(x)+5` from noisy data | Data normalisation, 1-D regression, denormalisation, weight saving | ~30 s |
| 2 | `02_classification.py` | Binary 2-D decision boundary | BCE loss, sigmoid output, accuracy, confusion matrix, boundary plot | ~45 s |
| 3 | `03_custom_csv_data.py` | Load your own CSV and train | CSV I/O without pandas, multi-feature extension, RMSE | ~20 s |

> Runtimes measured on a typical laptop CPU (single core).  They vary
> with early-stopping — scripts may finish faster when loss plateaus early.

---

## How to run

All commands are issued from the **project root** (the folder that contains
`src/`, `examples/`, `outputs/`, etc.):

```bash
# Script 1 — Regression
python examples/01_regression.py

# Script 2 — Classification
python examples/02_classification.py

# Script 3 — Custom CSV regression
python examples/03_custom_csv_data.py
```

PowerShell equivalent:

```powershell
python examples\01_regression.py
python examples\02_classification.py
python examples\03_custom_csv_data.py
```

---

## What each script produces

All artefacts land in the `outputs/` directory at the project root.

### `01_regression.py`

| File | Description |
|------|-------------|
| `outputs/01_regression_fit.png` | Scatter of test data overlaid with the model's predicted curve |
| `outputs/01_regression_loss.png` | Train vs test MSE over epochs |
| `outputs/01_regression_weights.npz` | Trained weight matrices (reload with `model.load_weights`) |

### `02_classification.py`

| File | Description |
|------|-------------|
| `outputs/02_classification_boundary.png` | Colour-filled decision regions with test-set scatter overlay |
| `outputs/02_classification_loss.png` | Train vs test BCE loss over epochs |

### `03_custom_csv_data.py`

| File | Description |
|------|-------------|
| `outputs/sample_data.csv` | Auto-generated demo CSV (`x`, `y` columns) |
| `outputs/03_custom_fit.png` | Predicted curve vs true test points |
| `outputs/03_custom_loss.png` | Train vs test MSE over epochs |

---

## Headless / CI environments

Every script sets `MPLBACKEND=Agg` (non-interactive renderer) at the top using:

```python
os.environ.setdefault("MPLBACKEND", "Agg")
```

`setdefault` means your shell's `MPLBACKEND` variable takes priority — you can
override it from outside:

```bash
# Force a specific backend (e.g. TkAgg for an interactive window)
MPLBACKEND=TkAgg python examples/01_regression.py
```

In CI pipelines (`MPLBACKEND` not set) the scripts silently write PNG files
instead of opening windows.

---

## Adapting script 3 to your own data

Open `03_custom_csv_data.py` and change three lines near the top:

```python
CSV_PATH = "/path/to/your/data.csv"   # absolute or project-root-relative
X_COLS   = ["col_a", "col_b"]         # list of feature column names
Y_COL    = "target"                   # target column name
```

If you have **N input features**, also change the input layer in the network
structure from `"units": 1` to `"units": N` — the script does this
automatically via the `n_features` variable detected from your data.

See the "QUICK REFERENCE" section at the bottom of the script for worked
examples covering single features, multiple features, and binary classification.

---

## Re-loading saved weights

Script 1 saves its trained weights.  To reload them later:

```python
import sys, os
sys.path.insert(0, "src")

from nn_core import NeuralNetwork

structure = [
    {"type": "input",  "units": 1},
    {"type": "dense",  "units": 32, "activation_function": "tanh",   "bias": True},
    {"type": "dense",  "units": 16, "activation_function": "tanh",   "bias": True},
    {"type": "dense",  "units": 1,  "activation_function": "linear", "bias": True},
]
model = NeuralNetwork()
net   = model.create_network(structure)
model.load_weights(net, "outputs/01_regression_weights")

# Run inference
import numpy as np
preds = model.predict(net, np.array([[0.0], [1.0], [2.0]]))
```

---

## Notes

- The `outputs/` directory is tracked in git via `outputs/.gitkeep`, but its
  contents (PNG files, `.npy` weights, CSV files) are gitignored.
- The `examples/` directory does **not** contain an `__init__.py` — these are
  standalone scripts, not a package.
- Import ordering in every script follows the project's always-on rule:
  stdlib → path setup → third-party → nn_core → modules.
