"""
modules — shared helper package for mlp-study-kit.

Four focused sub-modules, each importable independently:

    from modules.general_utils import ensure_directory, as_float_array
    from modules.plot_utils    import plot_loss_history, plot_predictions
    from modules.data_utils    import make_regression_data, train_test_split
    from modules.metrics       import mse, r2_score, evaluate

Or import the top-level shortcuts:

    from modules import ensure_directory, plot_loss_history, make_regression_data
    from modules import mse, rmse, mae, r2_score, accuracy, evaluate

Companion to nn_core: while nn_core provides the MLP building blocks
(activations, losses, network, logger), modules provides the surrounding
utilities — data preparation, visualisation, array helpers, and evaluation
metrics.
"""

from __future__ import annotations

from modules.general_utils import (
    as_float_array,
    describe_array,
    ensure_directory,
    ensure_matrix,
    ensure_vector,
    print_matrices,
)
from modules.plot_utils import (
    guard_backend,
    plot_activations,
    plot_decision_boundary,
    plot_loss_history,
    plot_predictions,
)
from modules.data_utils import (
    k_fold_split,
    load_csv,
    make_classification_data,
    make_linear_data,
    make_regression_data,
    normalize,
    train_test_split,
)
from modules.metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
    evaluate,
    f1_score,
    mae,
    mse,
    precision,
    r2_score,
    recall,
    regression_summary,
    rmse,
)

__version__ = "0.4.0"
__author__  = "Brijesh Rathod"

__all__ = [
    # general_utils
    "as_float_array",
    "describe_array",
    "ensure_directory",
    "ensure_matrix",
    "ensure_vector",
    "print_matrices",
    # plot_utils
    "guard_backend",
    "plot_activations",
    "plot_decision_boundary",
    "plot_loss_history",
    "plot_predictions",
    # data_utils
    "k_fold_split",
    "load_csv",
    "make_classification_data",
    "make_linear_data",
    "make_regression_data",
    "normalize",
    "train_test_split",
    # metrics
    "accuracy",
    "classification_report",
    "confusion_matrix",
    "evaluate",
    "f1_score",
    "mae",
    "mse",
    "precision",
    "r2_score",
    "recall",
    "regression_summary",
    "rmse",
    # version
    "__version__",
]
