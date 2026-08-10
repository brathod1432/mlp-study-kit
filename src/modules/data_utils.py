"""
modules.data_utils — dataset generators and data-preparation helpers.

All generators accept a ``seed`` argument for reproducibility and return
plain NumPy arrays (no external dependencies beyond NumPy).

The regression and classification datasets mirror exactly what the exercise
series (ex09, ex10) and experiment scripts use, so they can be imported
directly instead of re-defining the functions in every file.

Usage::

    from modules.data_utils import (
        make_regression_data,
        make_classification_data,
        make_linear_data,
        train_test_split,
        normalize,
    )

    X, Y = make_regression_data(n=200, noise=0.15, seed=42)
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_ratio=0.2)
    X_norm, mean, std = normalize(X_train)
"""

from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Regression datasets
# ──────────────────────────────────────────────────────────────────────────────

def make_regression_data(
    n:       int   = 100,
    *,
    noise:   float = 0.1,
    x_range: tuple[float, float] = (-3.0, 3.0),
    seed:    int   = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a 1-D noisy regression dataset.

    Target function:  ``y = sin(2x) + cos(x) + 5 + noise``

    This is the same dataset used in ``exercises/ex09_full_backprop.py``,
    ``exercises/ex10_bias_early_stop.py``, and ``experiments/20251224_v2.py``.

    Args:
        n:       number of samples (default 100).
        noise:   standard deviation of additive Gaussian noise (default 0.1).
        x_range: ``(x_min, x_max)`` interval (default ``(-3, 3)``).
        seed:    random seed for reproducibility.

    Returns:
        ``(X, Y)`` where ``X.shape == (n, 1)`` and ``Y.shape == (n, 1)``.

    Example::

        X, Y = make_regression_data(n=200, noise=0.2, seed=0)
        # X: inputs in [-3, 3], Y: noisy sin(2x)+cos(x)+5
    """
    rng = np.random.default_rng(seed)
    X   = np.linspace(x_range[0], x_range[1], n, dtype=np.float64).reshape(-1, 1)
    Y   = np.sin(2.0 * X) + np.cos(X) + 5.0
    Y  += rng.normal(0.0, noise, size=(n, 1))
    return X, Y.astype(np.float64)


def make_linear_data(
    n:          int   = 50,
    *,
    slope:      float = 2.0,
    intercept:  float = 1.0,
    noise:      float = 0.5,
    x_range:    tuple[float, float] = (0.0, 10.0),
    seed:       int   = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple linear regression dataset.

    Target function:  ``y = slope * x + intercept + noise``

    Matches the dataset used in ``exercises/ex12_keras_intro.py``.

    Args:
        n:         number of samples.
        slope:     true slope (default 2.0).
        intercept: true intercept (default 1.0).
        noise:     noise standard deviation (default 0.5).
        x_range:   ``(x_min, x_max)`` interval.
        seed:      random seed.

    Returns:
        ``(X, Y)`` where ``X.shape == (n, 1)`` and ``Y.shape == (n, 1)``.
        Y is NOT normalised — call :func:`normalize` if needed.

    Example::

        X, Y = make_linear_data(n=100, slope=3.0, intercept=-1.0)
    """
    rng = np.random.default_rng(seed)
    X   = np.linspace(x_range[0], x_range[1], n, dtype=np.float64).reshape(-1, 1)
    Y   = slope * X + intercept + rng.normal(0.0, noise, size=(n, 1))
    return X, Y.astype(np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# Classification dataset
# ──────────────────────────────────────────────────────────────────────────────

def make_classification_data(
    n_per_class: int   = 50,
    *,
    seed:        int   = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 2-D two-class synthetic classification dataset.

    Class 0 samples are drawn from ``Uniform([0, 2] × [0, 2])``.
    Class 1 samples are drawn from ``Uniform([1, 3] × [2, 4])``.
    The classes have partial overlap, making the task non-trivial.

    Matches the classification dataset from ``exercises/ex10_bias_early_stop.py``
    and ``experiments/20251224_v2.py``.

    Args:
        n_per_class: samples per class (default 50).
        seed:        random seed.

    Returns:
        ``(X, Y, idx_class0, idx_class1)`` where:

        - ``X.shape  == (2 * n_per_class, 2)``  — feature matrix
        - ``Y.shape  == (2 * n_per_class,)``    — labels (0 or 1)
        - ``idx_class0`` — indices of class-0 samples (for plotting)
        - ``idx_class1`` — indices of class-1 samples (for plotting)

    Example::

        X, Y, idx0, idx1 = make_classification_data(n_per_class=100)
        # scatter(X[idx0, 0], X[idx0, 1])  → class 0
        # scatter(X[idx1, 0], X[idx1, 1])  → class 1
    """
    rng = np.random.default_rng(seed)

    # Class 0
    X0 = rng.uniform(0.0, 2.0, size=(n_per_class, 2)).astype(np.float64)
    Y0 = np.zeros(n_per_class, dtype=np.float64)

    # Class 1
    x1 = rng.uniform(1.0, 3.0, size=(n_per_class, 1))
    y1 = rng.uniform(2.0, 4.0, size=(n_per_class, 1))
    X1 = np.hstack([x1, y1]).astype(np.float64)
    Y1 = np.ones(n_per_class, dtype=np.float64)

    X = np.vstack([X0, X1])
    Y = np.concatenate([Y0, Y1])

    idx0 = np.where(Y == 0)[0]
    idx1 = np.where(Y == 1)[0]
    return X, Y, idx0, idx1


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────────────

def train_test_split(
    X:          np.ndarray,
    T:          np.ndarray,
    *,
    test_ratio: float = 0.3,
    seed:       int   = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly split arrays *X* and *T* into train and test sets.

    NumPy-only (no scikit-learn dependency).  Uses a seeded RNG for
    reproducible splits.

    Args:
        X:          feature array, shape ``(N, …)``.
        T:          target array, shape ``(N, …)``.
        test_ratio: fraction of samples held out for testing (default 0.3).
        seed:       random seed.

    Returns:
        ``(X_train, T_train, X_test, T_test)`` — all ``float64``.

    Raises:
        ValueError: if *X* and *T* have different first dimensions, or
                    *test_ratio* is outside ``(0, 1)``.

    Example::

        X, Y = make_regression_data()
        X_tr, Y_tr, X_te, Y_te = train_test_split(X, Y, test_ratio=0.2)
    """
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(
            f"test_ratio must be in (0, 1), got {test_ratio}."
        )
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    if X.shape[0] != T.shape[0]:
        raise ValueError(
            f"X and T must have the same number of rows; "
            f"got X={X.shape[0]}, T={T.shape[0]}."
        )

    rng     = np.random.default_rng(seed)
    idx     = rng.permutation(X.shape[0])
    n_test  = max(1, int(round(test_ratio * X.shape[0])))
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    return (
        X[train_idx], T[train_idx],
        X[test_idx],  T[test_idx],
    )


def normalize(
    X:     np.ndarray,
    *,
    axis:  int   = 0,
    eps:   float = 1e-8,
    mean:  np.ndarray | None = None,
    std:   np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Zero-mean, unit-variance normalisation.

    If *mean* and *std* are provided (e.g. computed on the training set),
    they are applied directly — useful for normalising a test set with
    training-set statistics.

    Args:
        X:    array to normalise.
        axis: axis along which to compute statistics (default 0 = per feature).
        eps:  small constant to avoid division by zero.
        mean: pre-computed mean; if ``None`` computed from *X*.
        std:  pre-computed standard deviation; if ``None`` computed from *X*.

    Returns:
        ``(X_norm, mean, std)`` — normalised array plus the statistics used.

    Example::

        X_tr_norm, mu, sigma = normalize(X_train)
        X_te_norm, _, _      = normalize(X_test, mean=mu, std=sigma)
    """
    X    = np.asarray(X, dtype=np.float64)
    mu   = mean if mean is not None else X.mean(axis=axis, keepdims=True)
    sig  = std  if std  is not None else X.std( axis=axis, keepdims=True)
    return (X - mu) / (sig + eps), mu, sig


# ──────────────────────────────────────────────────────────────────────────────
# CSV data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(
    path: str,
    *,
    feature_cols: list[int] | None = None,
    target_col:   int               = -1,
    has_header:   bool              = True,
    delimiter:    str               = ",",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a numeric CSV file into feature matrix X and target vector Y.

    Uses only the Python standard library (``csv`` module) — no pandas needed.

    Args:
        path:         path to the CSV file.
        feature_cols: list of column indices to use as features.
                      Defaults to all columns except *target_col*.
        target_col:   column index of the target variable (default ``-1``,
                      i.e. the last column).
        has_header:   if ``True`` (default), skip the first row.
        delimiter:    field delimiter (default ``","``).

    Returns:
        ``(X, Y)`` where ``X.shape == (N, n_features)``
        and ``Y.shape == (N, 1)``, both ``float64``.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ValueError: if any row contains non-numeric values.

    Example::

        # CSV with header: x,y
        # 1.0,2.5
        # 2.0,4.1
        X, Y = load_csv("data/my_dataset.csv", target_col=-1)

        # Multi-feature CSV, use columns 0,1,2 as features, column 3 as target
        X, Y = load_csv("data/multi.csv",
                        feature_cols=[0, 1, 2], target_col=3)
    """
    import csv  # noqa: PLC0415

    rows: list[list[float]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            next(reader, None)   # skip header row
        for line_no, row in enumerate(reader, start=2 if has_header else 1):
            if not row or all(cell.strip() == "" for cell in row):
                continue   # skip blank lines
            try:
                rows.append([float(cell.strip()) for cell in row])
            except ValueError as exc:
                raise ValueError(
                    f"{path}: non-numeric value at row {line_no}: {row}"
                ) from exc

    if not rows:
        raise ValueError(f"{path}: no data rows found.")

    data    = np.array(rows, dtype=np.float64)
    n_cols  = data.shape[1]
    t_col   = target_col % n_cols          # normalise negative index

    if feature_cols is None:
        feature_cols = [c for c in range(n_cols) if c != t_col]

    X = data[:, feature_cols]
    Y = data[:, t_col].reshape(-1, 1)
    return X, Y


# ──────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────────────────────────────────────

def k_fold_split(
    X:      np.ndarray,
    T:      np.ndarray,
    k:      int  = 5,
    *,
    seed:   int  = 42,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split data into *k* stratified folds for cross-validation.

    Each call to ``k_fold_split`` returns *k* ``(X_train, T_train, X_val, T_val)``
    tuples — one per fold.  In each fold the validation set is one of the *k*
    equal-sized partitions; the remaining ``k-1`` partitions form the training set.

    Args:
        X:    feature array, shape ``(N, …)``.
        T:    target array, shape ``(N, …)``.
        k:    number of folds (default 5).  Must satisfy ``2 ≤ k ≤ N``.
        seed: random seed for the initial shuffle.

    Returns:
        List of *k* tuples: ``[(X_tr, T_tr, X_val, T_val), …]``.
        All arrays are ``float64``.

    Raises:
        ValueError: if ``k < 2``, ``k > N``, or arrays have mismatched rows.

    Example::

        folds = k_fold_split(X, Y, k=5)
        val_losses = []
        for X_tr, Y_tr, X_val, Y_val in folds:
            net = model.create_network(structure)
            loss, _, _ = model.train(net, X_tr, Y_tr, verbose=0)
            preds = model.predict(net, X_val)
            val_losses.append(np.mean((np.array(preds) - Y_val) ** 2))
        print(f"Mean CV MSE: {np.mean(val_losses):.4f}")
    """
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    N = X.shape[0]
    if X.shape[0] != T.shape[0]:
        raise ValueError(
            f"X and T must have the same number of rows; "
            f"got X={X.shape[0]}, T={T.shape[0]}."
        )
    if k < 2 or k > N:
        raise ValueError(f"k must be between 2 and N={N}, got k={k}.")

    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(N)

    # Split shuffled indices into k roughly equal folds
    fold_sizes = np.full(k, N // k, dtype=int)
    fold_sizes[: N % k] += 1      # distribute remainder across first folds
    boundaries = np.concatenate([[0], np.cumsum(fold_sizes)])

    folds: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for fold_idx in range(k):
        val_idx   = idx[boundaries[fold_idx]: boundaries[fold_idx + 1]]
        train_idx = np.concatenate([
            idx[: boundaries[fold_idx]],
            idx[boundaries[fold_idx + 1] :],
        ])
        folds.append((
            X[train_idx], T[train_idx],
            X[val_idx],   T[val_idx],
        ))
    return folds


__all__ = [
    "k_fold_split",
    "load_csv",
    "make_classification_data",
    "make_linear_data",
    "make_regression_data",
    "normalize",
    "train_test_split",
]
