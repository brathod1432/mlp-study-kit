"""
modules.metrics — Model evaluation metrics for regression and classification.

NumPy-only implementation; no scikit-learn dependency.

Imports::

    from modules.metrics import mse, rmse, mae, r2_score
    from modules.metrics import accuracy, precision, recall, f1_score
    from modules.metrics import confusion_matrix, classification_report
    from modules.metrics import regression_summary, evaluate

Usage::

    import numpy as np
    from modules.metrics import mse, r2_score, evaluate

    # --- Regression ---
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    print(mse(y_true, y_pred))           # 0.0275
    print(r2_score(y_true, y_pred))      # ≈ 0.983
    print(regression_summary(y_true, y_pred))

    metrics = evaluate(y_true, y_pred, task="regression")
    # {'mse': 0.0275, 'rmse': 0.1658..., 'mae': 0.15, 'r2': 0.983...}

    # --- Classification ---
    y_true = np.array([1, 0, 1, 1, 0, 0])
    y_pred = np.array([0.9, 0.2, 0.8, 0.3, 0.1, 0.7])

    print(accuracy(y_true, y_pred))      # 0.8333...
    print(f1_score(y_true, y_pred))      # ...
    print(classification_report(y_true, y_pred))

    metrics = evaluate(y_true, y_pred, task="classification")
    # {'accuracy': 0.833, 'precision': ..., 'recall': ..., 'f1': ...,
    #  'confusion_matrix': array(...)}
"""

from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _prepare(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten and coerce both arrays to float64 1-D vectors."""
    return (
        np.asarray(y_true, dtype=np.float64).flatten(),
        np.asarray(y_pred, dtype=np.float64).flatten(),
    )


def _binarise(y_pred: np.ndarray, threshold: float) -> np.ndarray:
    """Apply *threshold* to *y_pred* → int32 array of 0s and 1s."""
    return (y_pred >= threshold).astype(np.int32)


# ──────────────────────────────────────────────────────────────────────────────
# Regression metrics
# ──────────────────────────────────────────────────────────────────────────────

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error.

    .. math::

        \\text{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2

    Args:
        y_true: Ground-truth target values, any shape (will be flattened).
        y_pred: Predicted values, same number of elements as *y_true*.

    Returns:
        MSE as a Python ``float``.

    Example::

        >>> import numpy as np
        >>> from modules.metrics import mse
        >>> mse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        0.0
        >>> mse(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        1.0
    """
    yt, yp = _prepare(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    .. math::

        \\text{RMSE} = \\sqrt{\\text{MSE}(y, \\hat{y})}

    Args:
        y_true: Ground-truth target values, any shape (will be flattened).
        y_pred: Predicted values, same number of elements as *y_true*.

    Returns:
        RMSE as a Python ``float``.

    Example::

        >>> import numpy as np
        >>> from modules.metrics import rmse
        >>> rmse(np.array([0.0, 0.0, 0.0, 0.0]), np.array([2.0, 2.0, 2.0, 2.0]))
        2.0
        >>> rmse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        0.0
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    .. math::

        \\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|

    Args:
        y_true: Ground-truth target values, any shape (will be flattened).
        y_pred: Predicted values, same number of elements as *y_true*.

    Returns:
        MAE as a Python ``float``.

    Example::

        >>> import numpy as np
        >>> from modules.metrics import mae
        >>> mae(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0]))
        0.6666666666666666
        >>> mae(np.array([5.0]), np.array([5.0]))
        0.0
    """
    yt, yp = _prepare(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R-squared (coefficient of determination).

    .. math::

        R^2 = 1 - \\frac{SS_{\\text{res}}}{SS_{\\text{tot}}}

    where :math:`SS_{\\text{res}} = \\sum (y - \\hat{y})^2` and
    :math:`SS_{\\text{tot}} = \\sum (y - \\bar{y})^2`.

    Interpretation:

    - ``1.0``  — perfect fit; all variance explained.
    - ``0.0``  — model does no better than predicting the training mean.
    - ``< 0``  — model is worse than the constant-mean baseline.

    Special case: if all *y_true* values are identical (``SS_tot == 0``),
    the metric is undefined; ``0.0`` is returned by convention.

    Args:
        y_true: Ground-truth target values, any shape (will be flattened).
        y_pred: Predicted values, same number of elements as *y_true*.

    Returns:
        R² as a Python ``float``.

    Example::

        >>> import numpy as np
        >>> from modules.metrics import r2_score
        >>> r2_score(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        1.0
        >>> r2_score(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0]))
        0.0
    """
    yt, yp = _prepare(y_true, y_pred)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ──────────────────────────────────────────────────────────────────────────────
# Classification metrics  (binary, labels 0 / 1)
# ──────────────────────────────────────────────────────────────────────────────

def accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Classification accuracy.

    *y_pred* may be raw probabilities (threshold applied internally) or
    already-binarised 0/1 labels.

    Args:
        y_true:    Ground-truth binary labels (0 or 1), any shape.
        y_pred:    Predicted probabilities or 0/1 labels, same size as *y_true*.
        threshold: Decision threshold applied to *y_pred* (default ``0.5``).

    Returns:
        Fraction of correctly classified samples as a Python ``float`` in
        ``[0.0, 1.0]``.

    Example::

        >>> import numpy as np
        >>> from modules.metrics import accuracy
        >>> accuracy(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.8, 0.4]))
        1.0
        >>> accuracy(np.array([1, 0, 1, 0]), np.array([0, 1, 0, 1]))
        0.0
    """
    yt, yp = _prepare(y_true, y_pred)
    yp_bin = _binarise(yp, threshold)
    yt_bin = yt.astype(np.int32)
    return float(np.mean(yt_bin == yp_bin))


def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Precision = TP / (TP + FP).

    Returns ``0.0`` when there are no positive predictions (avoids
    zero-division without raising an exception).

    Args:
        y_true:    Ground-truth binary labels (0 or 1), any shape.
        y_pred:    Predicted probabilities or 0/1 labels.
        threshold: Decision threshold (default ``0.5``).

    Returns:
        Precision as a Python ``float`` in ``[0.0, 1.0]``.

    Example::

        >>> import numpy as np
        >>> from modules.metrics import precision
        >>> precision(np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0]))
        1.0
        >>> precision(np.array([1, 0, 1, 0]), np.array([0, 0, 0, 0]))
        0.0
    """
    yt, yp = _prepare(y_true, y_pred)
    yp_bin = _binarise(yp, threshold)
    yt_bin = yt.astype(np.int32)
    tp = int(np.sum((yp_bin == 1) & (yt_bin == 1)))
    fp = int(np.sum((yp_bin == 1) & (yt_bin == 0)))
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    Recall = TP / (TP + FN).

    Returns ``0.0`` when there are no actual positives in *y_true*.

    Args:
        y_true:    Ground-truth binary labels (0 or 1), any shape.
        y_pred:    Predicted probabilities or 0/1 labels.
        threshold: Decision threshold (default ``0.5``).

    Returns:
        Recall as a Python ``float`` in ``[0.0, 1.0]``.

    Example::

        >>> import numpy as np
        >>> from modules.metrics import recall
        >>> recall(np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0]))
        1.0
        >>> recall(np.array([1, 1, 1]), np.array([0, 0, 0]))
        0.0
    """
    yt, yp = _prepare(y_true, y_pred)
    yp_bin = _binarise(yp, threshold)
    yt_bin = yt.astype(np.int32)
    tp = int(np.sum((yp_bin == 1) & (yt_bin == 1)))
    fn = int(np.sum((yp_bin == 0) & (yt_bin == 1)))
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """
    F1 score — harmonic mean of precision and recall.

    .. math::

        F_1 = \\frac{2 \\cdot P \\cdot R}{P + R}

    Returns ``0.0`` when both precision and recall are zero (no division by
    zero raised).

    Args:
        y_true:    Ground-truth binary labels (0 or 1), any shape.
        y_pred:    Predicted probabilities or 0/1 labels.
        threshold: Decision threshold (default ``0.5``).

    Returns:
        F1 as a Python ``float`` in ``[0.0, 1.0]``.

    Example::

        >>> import numpy as np
        >>> from modules.metrics import f1_score
        >>> f1_score(np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0]))
        1.0
        >>> f1_score(np.array([1, 0, 1]), np.array([0, 1, 0]))
        0.0
    """
    p = precision(y_true, y_pred, threshold)
    r = recall(y_true, y_pred, threshold)
    denom = p + r
    return float(2.0 * p * r / denom) if denom > 0.0 else 0.0


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Binary confusion matrix.

    Layout::

        [[TN, FP],
         [FN, TP]]

    - Rows represent the **actual** class (0 = negative, 1 = positive).
    - Columns represent the **predicted** class.

    Args:
        y_true:    Ground-truth binary labels (0 or 1), any shape.
        y_pred:    Predicted probabilities or 0/1 labels.
        threshold: Decision threshold (default ``0.5``).

    Returns:
        2×2 ``np.ndarray`` of ``int32`` counts.

    Example::

        >>> import numpy as np
        >>> from modules.metrics import confusion_matrix
        >>> confusion_matrix(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
        array([[2, 0],
               [1, 1]], dtype=int32)
    """
    yt, yp = _prepare(y_true, y_pred)
    yp_bin = _binarise(yp, threshold)
    yt_bin = yt.astype(np.int32)
    tn = int(np.sum((yp_bin == 0) & (yt_bin == 0)))
    fp = int(np.sum((yp_bin == 1) & (yt_bin == 0)))
    fn = int(np.sum((yp_bin == 0) & (yt_bin == 1)))
    tp = int(np.sum((yp_bin == 1) & (yt_bin == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=np.int32)


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> str:
    """
    Human-readable classification report.

    Includes accuracy, precision, recall, F1 score, and the full 2×2
    confusion matrix, all presented in a bordered box.

    Args:
        y_true:    Ground-truth binary labels (0 or 1), any shape.
        y_pred:    Predicted probabilities or 0/1 labels.
        threshold: Decision threshold (default ``0.5``).

    Returns:
        Multi-line string suitable for ``print()``.

    Example::

        >>> import numpy as np
        >>> from modules.metrics import classification_report
        >>> y_true = np.array([1, 0, 1, 0, 1])
        >>> y_pred = np.array([0.9, 0.1, 0.8, 0.3, 0.4])
        >>> print(classification_report(y_true, y_pred))
        ┌─ Classification Report ────────────────────┐
        │ Threshold  : 0.50                           │
        │ Accuracy   : 0.8000                         │
        ...
    """
    acc  = accuracy(y_true, y_pred, threshold)
    prec = precision(y_true, y_pred, threshold)
    rec  = recall(y_true, y_pred, threshold)
    f1   = f1_score(y_true, y_pred, threshold)
    cm   = confusion_matrix(y_true, y_pred, threshold)

    # Box geometry: total line width W=46, content area inner=44
    inner = 44

    def _row(text: str) -> str:
        return f"│ {text:<{inner - 1}}│"

    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    lines = [
        f"┌─ Classification Report {'─' * (inner - 24)}┐",
        _row(f"Threshold  : {threshold:.2f}"),
        _row(f"Accuracy   : {acc:.4f}"),
        _row(f"Precision  : {prec:.4f}"),
        _row(f"Recall     : {rec:.4f}"),
        _row(f"F1 Score   : {f1:.4f}"),
        _row(""),
        _row("Confusion Matrix  [[TN, FP], [FN, TP]]"),
        _row(f"  TN={tn:<6d}  FP={fp:<6d}"),
        _row(f"  FN={fn:<6d}  TP={tp:<6d}"),
        f"└{'─' * inner}┘",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Summary / convenience functions
# ──────────────────────────────────────────────────────────────────────────────

def regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Formatted table of all four regression metrics.

    Args:
        y_true: Ground-truth target values, any shape (will be flattened).
        y_pred: Predicted values, same number of elements as *y_true*.

    Returns:
        Multi-line string with MSE, RMSE, MAE, and R2 in a bordered box.
        The string is returned (not printed); pass it to ``print()`` to display.

    Example::

        >>> import numpy as np
        >>> from modules.metrics import regression_summary
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 1.9, 3.0])
        >>> print(regression_summary(y_true, y_pred))
        ┌─ Regression Summary ───────────────────────┐
        │ MSE   : 0.006667                            │
        │ RMSE  : 0.081650                            │
        │ MAE   : 0.066667                            │
        │ R2    : 0.990000                            │
        └────────────────────────────────────────────┘
    """
    _mse  = mse(y_true, y_pred)
    _rmse = rmse(y_true, y_pred)
    _mae  = mae(y_true, y_pred)
    _r2   = r2_score(y_true, y_pred)

    # Box geometry: total line width W=46, content area inner=44
    inner = 44

    def _row(text: str) -> str:
        return f"│ {text:<{inner - 1}}│"

    lines = [
        f"┌─ Regression Summary {'─' * (inner - 21)}┐",
        _row(f"MSE   : {_mse:.6f}"),
        _row(f"RMSE  : {_rmse:.6f}"),
        _row(f"MAE   : {_mae:.6f}"),
        _row(f"R2    : {_r2:.6f}"),
        f"└{'─' * inner}┘",
    ]
    return "\n".join(lines)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "regression",
    threshold: float = 0.5,
) -> dict:
    """
    Compute all metrics for the given task and return them as a dictionary.

    Args:
        y_true:    Ground-truth values, any shape (will be flattened).
        y_pred:    Predicted values, same number of elements as *y_true*.
        task:      ``"regression"`` or ``"classification"``
                   (default ``"regression"``).
        threshold: Decision threshold used for classification
                   (default ``0.5``).  Ignored when *task* is
                   ``"regression"``.

    Returns:
        Dict mapping metric name → value:

        - **regression** keys: ``"mse"``, ``"rmse"``, ``"mae"``, ``"r2"``
        - **classification** keys: ``"accuracy"``, ``"precision"``,
          ``"recall"``, ``"f1"``, ``"confusion_matrix"``

    Raises:
        ValueError: if *task* is not ``"regression"`` or ``"classification"``.

    Example::

        >>> import numpy as np
        >>> from modules.metrics import evaluate
        >>> y_t = np.array([1.0, 2.0, 3.0])
        >>> y_p = np.array([1.0, 2.0, 3.0])
        >>> evaluate(y_t, y_p, task="regression")
        {'mse': 0.0, 'rmse': 0.0, 'mae': 0.0, 'r2': 1.0}
        >>> y_tc = np.array([1, 0, 1, 0])
        >>> y_pc = np.array([1, 0, 1, 0])
        >>> evaluate(y_tc, y_pc, task="classification")
        {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
         'confusion_matrix': array([[2, 0], [0, 2]], dtype=int32)}
    """
    task_lower = task.strip().lower()
    if task_lower == "regression":
        return {
            "mse":  mse(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mae":  mae(y_true, y_pred),
            "r2":   r2_score(y_true, y_pred),
        }
    elif task_lower == "classification":
        return {
            "accuracy":         accuracy(y_true, y_pred, threshold),
            "precision":        precision(y_true, y_pred, threshold),
            "recall":           recall(y_true, y_pred, threshold),
            "f1":               f1_score(y_true, y_pred, threshold),
            "confusion_matrix": confusion_matrix(y_true, y_pred, threshold),
        }
    else:
        raise ValueError(
            f"task must be 'regression' or 'classification', got {task!r}."
        )


__all__ = [
    # regression
    "mse",
    "rmse",
    "mae",
    "r2_score",
    # classification
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "confusion_matrix",
    "classification_report",
    # summary / convenience
    "regression_summary",
    "evaluate",
]
