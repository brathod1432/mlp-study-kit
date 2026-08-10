"""
modules.general_utils — array validation and filesystem helpers.

Adapted from Robotics_Guide/modules/GeneralUtils.py for mlp-study-kit.
Uses nn_core.logger instead of the Robotics_Guide logger chain.

Usage::

    from modules.general_utils import (
        ensure_directory, as_float_array, ensure_vector,
        ensure_matrix, describe_array, print_matrices,
    )

    ensure_directory("logs/run_01")
    weights = as_float_array([0.1, 0.2, 0.3], name="weights", ndim=1)
    print_matrices([weights, weights.reshape(1, -1)])
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from nn_core.logger import ObjLogger

_log = ObjLogger("modules.general_utils")


# ──────────────────────────────────────────────────────────────────────────────
# Filesystem
# ──────────────────────────────────────────────────────────────────────────────

def ensure_directory(path_like: str | Path) -> Path:
    """
    Create *path_like* (and any missing parents) if it does not already exist.

    Args:
        path_like: directory path as string or :class:`pathlib.Path`.

    Returns:
        The resolved :class:`pathlib.Path` object.

    Example::

        logs_dir = ensure_directory("logs/experiment_01")
    """
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Array validation
# ──────────────────────────────────────────────────────────────────────────────

def as_float_array(
    values: object,
    *,
    name: str = "value",
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """
    Convert *values* to a float64 NumPy array and optionally validate its
    dimensionality and shape.

    Args:
        values: any array-like (list, tuple, ndarray, scalar …).
        name:   label used in error messages (default ``"value"``).
        ndim:   expected number of dimensions; ``None`` = no check.
        shape:  expected exact shape tuple; ``None`` = no check.

    Returns:
        ``np.ndarray`` with ``dtype=float64``.

    Raises:
        ValueError: if *ndim* or *shape* constraints are violated.

    Example::

        w = as_float_array([[0.1, 0.2], [0.3, 0.4]], name="weights", ndim=2)
    """
    array = np.asarray(values, dtype=float)

    if ndim is not None and array.ndim != ndim:
        raise ValueError(
            f"'{name}' must have {ndim} dimension(s), got {array.ndim}."
        )
    if shape is not None and array.shape != shape:
        raise ValueError(
            f"'{name}' must have shape {shape}, got {array.shape}."
        )
    return array


def ensure_vector(
    values: object,
    *,
    name: str = "vector",
    length: int | None = None,
) -> np.ndarray:
    """
    Validate that *values* is (or can be cast to) a 1-D float array.

    Args:
        values: array-like.
        name:   label for error messages.
        length: expected number of elements; ``None`` = no check.

    Returns:
        1-D ``np.ndarray`` with ``dtype=float64``.

    Example::

        v = ensure_vector([1.0, 2.0, 3.0], name="bias", length=3)
    """
    vector = as_float_array(values, name=name, ndim=1)
    if length is not None and vector.size != length:
        raise ValueError(
            f"'{name}' must contain {length} elements, got {vector.size}."
        )
    return vector


def ensure_matrix(
    values: object,
    *,
    name: str = "matrix",
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Validate that *values* is (or can be cast to) a 2-D float array.

    Args:
        values: array-like.
        name:   label for error messages.
        shape:  expected ``(rows, cols)``; ``None`` = no check.

    Returns:
        2-D ``np.ndarray`` with ``dtype=float64``.

    Example::

        W = ensure_matrix([[0.1, 0.2], [0.3, 0.4]], name="W1", shape=(2, 2))
    """
    return as_float_array(values, name=name, ndim=2, shape=shape)


# ──────────────────────────────────────────────────────────────────────────────
# Description helpers
# ──────────────────────────────────────────────────────────────────────────────

def describe_array(values: object) -> str:
    """
    Return a compact human-readable shape string.

    Examples::

        describe_array(5.0)           # → "scalar"
        describe_array([1, 2, 3])     # → "1×3"
        describe_array([[1, 2],[3,4]])# → "2×2"
    """
    arr = np.asarray(values)
    if arr.ndim == 0:
        return "scalar"
    return "×".join(str(d) for d in arr.shape)


def _select_display_slice(array: np.ndarray) -> tuple[np.ndarray, str]:
    """Return a 2-D slice of *array* for display, plus a shape label."""
    if array.ndim == 0:
        return array.reshape(1, 1), "scalar"
    if array.ndim == 1:
        return array.reshape(1, -1), f"1×{array.shape[0]}"
    if array.ndim == 2:
        return array, f"{array.shape[0]}×{array.shape[1]}"
    # higher-dimensional: show the first 2-D slice
    selector: tuple = (0,) * (array.ndim - 2) + (slice(None), slice(None))
    display = array[selector]
    return display, f"{describe_array(array)} (first 2-D slice {display.shape})"


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ──────────────────────────────────────────────────────────────────────────────

def print_matrices(
    arrays: Sequence[object],
    *,
    names: Sequence[str] | None = None,
    color: str = "cyan",
    logger_instance: ObjLogger | None = None,
) -> None:
    """
    Pretty-print a sequence of array-like objects using the logger.

    Args:
        arrays:          sequence of array-like values to print.
        names:           optional list of labels (same length as *arrays*).
        color:           ANSI colour for the output (default ``"cyan"``).
        logger_instance: use a specific :class:`ObjLogger`; defaults to the
                         module-level logger.

    Example::

        W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
        W2 = np.array([[0.5, 0.6]])
        print_matrices([W1, W2], names=["W1", "W2"])
    """
    log = logger_instance or _log
    for idx, arr_like in enumerate(arrays, start=1):
        label = names[idx - 1] if (names and idx <= len(names)) else f"[{idx}]"
        try:
            arr = np.asarray(arr_like)
            display, shape_str = _select_display_slice(arr)
        except Exception as exc:
            log(f"{label}: cannot render — {exc}", color="red")
            continue
        log(f"{label}  shape={shape_str}", color=color)
        log(f"{display}", color=color)


__all__ = [
    "as_float_array",
    "describe_array",
    "ensure_directory",
    "ensure_matrix",
    "ensure_vector",
    "print_matrices",
]
