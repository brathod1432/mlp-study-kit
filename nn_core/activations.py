"""
nn_core.activations -- canonical ActivationFn class for mlp-study-kit.

Consolidates every activation function seen across the exercise series:
  ex06  -- standalone functions (linear, tanh, relu)
  ex07  -- Activation_fcn class (linear, logistic, tanh, relu)
  ex08  -- added derivative=False flag
  hw_02 -- extended with leaky_relu and elu (20251224_v2.py)

All activations support both forward and derivative modes via a
single .output() call, matching the API used in ex08-ex10.

Usage:
    from nn_core.activations import ActivationFn

    af = ActivationFn()
    layer = {"activation_potential": z_array, "output": y_array}

    y  = af.output(layer, "tanh")           # forward
    dy = af.output(layer, "tanh", derivative=True)  # derivative
"""

from __future__ import annotations

import numpy as np


class ActivationFn:
    """
    Activation function registry with forward and derivative support.

    Supported names:
        "linear"      -- identity f(v) = v
        "sigmoid"     -- logistic  f(v) = 1 / (1 + exp(-v))
        "logistic"    -- alias for sigmoid
        "tanh"        -- hyperbolic tangent
        "relu"        -- rectified linear unit
        "leaky_relu"  -- leaky ReLU  (alpha default 0.01)
        "elu"         -- exponential linear unit (alpha default 1.0)
    """

    def __init__(self) -> None:
        self._dispatch: dict = {
            "linear":     (self._linear,     self._d_linear),
            "sigmoid":    (self._logistic,   self._d_logistic),
            "logistic":   (self._logistic,   self._d_logistic),
            "tanh":       (self._tanh,       self._d_tanh),
            "relu":       (self._relu,       self._d_relu),
            "leaky_relu": (self._leaky_relu, self._d_leaky_relu),
            "elu":        (self._elu,        self._d_elu),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def output(self, layer: dict, name: str, derivative: bool = False, alpha: float | None = None) -> np.ndarray:
        """
        Compute activation (or its derivative) for a layer dict.

        Args:
            layer:      dict with keys 'activation_potential' and 'output'
            name:       activation function name (see class docstring)
            derivative: if True return the analytical derivative
            alpha:      optional alpha for leaky_relu / elu
        """
        key = name.strip().lower()
        if key not in self._dispatch:
            raise ValueError(
                f"[ActivationFn] Unknown activation '{name}'. "
                f"Supported: {list(self._dispatch)}"
            )

        fwd, deriv = self._dispatch[key]

        if key in ("leaky_relu", "elu"):
            a = float(alpha) if alpha is not None else (0.01 if key == "leaky_relu" else 1.0)
            return deriv(layer, a) if derivative else fwd(layer, a)

        return deriv(layer) if derivative else fwd(layer)

    # ------------------------------------------------------------------
    # Forward implementations  (take layer dict, return ndarray)
    # ------------------------------------------------------------------

    @staticmethod
    def _linear(layer: dict) -> np.ndarray:
        return np.asarray(layer["activation_potential"], dtype=float)

    @staticmethod
    def _logistic(layer: dict) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.asarray(layer["activation_potential"], dtype=float)))

    @staticmethod
    def _tanh(layer: dict) -> np.ndarray:
        v = np.asarray(layer["activation_potential"], dtype=float)
        ep, em = np.exp(v), np.exp(-v)
        return (ep - em) / (ep + em)

    @staticmethod
    def _relu(layer: dict) -> np.ndarray:
        return np.maximum(0.0, np.asarray(layer["activation_potential"], dtype=float))

    @staticmethod
    def _leaky_relu(layer: dict, alpha: float = 0.01) -> np.ndarray:
        v = np.asarray(layer["activation_potential"], dtype=float)
        return np.where(v >= 0.0, v, alpha * v)

    @staticmethod
    def _elu(layer: dict, alpha: float = 1.0) -> np.ndarray:
        v = np.asarray(layer["activation_potential"], dtype=float)
        return np.where(v >= 0.0, v, alpha * (np.exp(v) - 1.0))

    # ------------------------------------------------------------------
    # Derivative implementations  (take layer dict, return ndarray)
    # ------------------------------------------------------------------

    @staticmethod
    def _d_linear(layer: dict) -> np.ndarray:
        return np.ones_like(np.asarray(layer["activation_potential"], dtype=float))

    @staticmethod
    def _d_logistic(layer: dict) -> np.ndarray:
        y = np.asarray(layer["output"], dtype=float)
        return y * (1.0 - y)

    @staticmethod
    def _d_tanh(layer: dict) -> np.ndarray:
        y = np.asarray(layer["output"], dtype=float)
        return 1.0 - np.power(y, 2)

    @staticmethod
    def _d_relu(layer: dict) -> np.ndarray:
        v = np.asarray(layer["activation_potential"], dtype=float)
        return (v >= 0.0).astype(float)

    @staticmethod
    def _d_leaky_relu(layer: dict, alpha: float = 0.01) -> np.ndarray:
        v = np.asarray(layer["activation_potential"], dtype=float)
        return np.where(v >= 0.0, 1.0, alpha).astype(float)

    @staticmethod
    def _d_elu(layer: dict, alpha: float = 1.0) -> np.ndarray:
        v = np.asarray(layer["activation_potential"], dtype=float)
        return np.where(v >= 0.0, 1.0, alpha * np.exp(v)).astype(float)
