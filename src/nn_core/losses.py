"""
nn_core.losses -- canonical LossFn class for mlp-study-kit.

Consolidates the Loss_fcn classes from ex07-ex10, all of which were
identical except for the gradual addition of the derivative flag.

Supported loss functions:
    "mse"                  -- Mean Squared Error  (0.5 * (t - y)^2)
    "binary_cross_entropy" -- Binary Cross-Entropy (-t*log(y) - (1-t)*log(1-y))

Usage:
    from nn_core.losses import LossFn

    loss = LossFn()

    value = loss.output("mse", expected, predicted)
    grad  = loss.output("mse", expected, predicted, derivative=True)
"""

from __future__ import annotations

import sys

import numpy as np


class LossFn:
    """
    Loss function registry supporting forward and derivative modes.

    The API intentionally mirrors the Loss_fcn used in ex07-ex10 so
    that exercises remain readable alongside the canonical version.
    """

    def __init__(self) -> None:
        self._dispatch: dict = {
            "mse":                  (self._mse,  self._d_mse),
            "binary_cross_entropy": (self._bce,  self._d_bce),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def output(
        self,
        name: str,
        expected: np.ndarray,
        predicted: np.ndarray,
        derivative: bool = False,
    ) -> np.ndarray:
        """
        Compute loss value or its derivative.

        Args:
            name:       "mse" or "binary_cross_entropy"
            expected:   target array  (t)
            predicted:  network output (y)
            derivative: if True return dL/dy  (gradient w.r.t. output)
        """
        key = name.strip().lower()
        if key not in self._dispatch:
            sys.exit(f"[LossFn] Unknown loss '{name}'. "
                     f"Supported: {list(self._dispatch)}")

        fwd, deriv = self._dispatch[key]
        t = np.asarray(expected,  dtype=float)
        y = np.asarray(predicted, dtype=float)
        return deriv(t, y) if derivative else fwd(t, y)

    # ------------------------------------------------------------------
    # Implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _mse(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """0.5 * (t - y)^2  -- element-wise, summed by caller."""
        return 0.5 * np.power(t - y, 2)

    @staticmethod
    def _d_mse(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """dMSE/dy = -(t - y)"""
        return -(t - y)

    @staticmethod
    def _bce(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Binary cross-entropy: -t*log(y) - (1-t)*log(1-y)"""
        return -t * np.log(y) - (1.0 - t) * np.log(1.0 - y)

    @staticmethod
    def _d_bce(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """dBCE/dy = -(t/y - (1-t)/(1-y))"""
        return -(t / y - (1.0 - t) / (1.0 - y))
