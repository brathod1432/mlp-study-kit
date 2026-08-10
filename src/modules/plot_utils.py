"""
modules.plot_utils — matplotlib helpers for mlp-study-kit.

All public functions respect the ``MPLBACKEND`` environment variable
(set to ``Agg`` for headless/CI environments) and accept an optional
``save_path`` argument so plots can be saved to file instead of displayed
interactively.

Usage::

    import os
    os.environ["MPLBACKEND"] = "Agg"          # suppress windows

    from modules.plot_utils import (
        plot_loss_history,
        plot_predictions,
        plot_activations,
        plot_decision_boundary,
    )

    plot_loss_history(train_losses, test_losses, save_path="loss.png")
    plot_predictions(X, Y_true, Y_pred, save_path="fit.png")
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Backend guard
# ──────────────────────────────────────────────────────────────────────────────

def guard_backend() -> None:
    """
    Apply the ``MPLBACKEND`` environment variable before matplotlib initialises.

    Call this **once** at the top of any script that imports matplotlib,
    before ``import matplotlib.pyplot as plt``.

    If ``MPLBACKEND`` is not set, this is a no-op.

    Example::

        from modules.plot_utils import guard_backend
        guard_backend()
        import matplotlib.pyplot as plt
    """
    backend = os.environ.get("MPLBACKEND", "")
    if backend:
        import matplotlib
        matplotlib.use(backend)


def _get_plt():
    """Lazy-import matplotlib.pyplot after applying the backend guard."""
    guard_backend()
    import matplotlib.pyplot as plt  # noqa: PLC0415
    return plt


def _finish(plt, save_path: str | None, title: str) -> None:
    """Save or show a completed figure."""
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
        plt.close()
        print(f"[plot_utils] Saved: {save_path}")
    else:
        plt.show(block=False)
        plt.pause(0.1)


# ──────────────────────────────────────────────────────────────────────────────
# Training history
# ──────────────────────────────────────────────────────────────────────────────

def plot_loss_history(
    train_losses: Sequence[float],
    test_losses:  Sequence[float] | None = None,
    *,
    title:     str        = "Training history",
    xlabel:    str        = "Epoch",
    ylabel:    str        = "Loss",
    save_path: str | None = None,
) -> None:
    """
    Plot training (and optionally test) loss over epochs.

    Args:
        train_losses: per-epoch training loss values.
        test_losses:  per-epoch test loss values; omit for train-only plot.
        title:        plot title.
        xlabel:       x-axis label (default ``"Epoch"``).
        ylabel:       y-axis label (default ``"Loss"``).
        save_path:    file path to save the figure (PNG, PDF, …).
                      If ``None``, the figure is shown interactively.

    Example::

        from modules.plot_utils import plot_loss_history
        plot_loss_history(history_train, history_test, save_path="loss.png")
    """
    plt = _get_plt()
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train loss", linewidth=1.8)
    if test_losses is not None:
        plt.plot(test_losses, label="Test loss", linewidth=1.8, linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.4)
    _finish(plt, save_path, title)


# ──────────────────────────────────────────────────────────────────────────────
# Regression predictions
# ──────────────────────────────────────────────────────────────────────────────

def plot_predictions(
    x:          np.ndarray,
    y_true:     np.ndarray,
    y_pred:     np.ndarray | Sequence,
    *,
    title:      str        = "Predictions vs ground truth",
    xlabel:     str        = "x",
    ylabel:     str        = "y",
    save_path:  str | None = None,
) -> None:
    """
    Scatter the training data and overlay the model's predictions.

    Args:
        x:         input values (1-D or 2-D column vector).
        y_true:    ground-truth target values.
        y_pred:    model predictions (same length as *x*).
        title:     plot title.
        save_path: file path to save; ``None`` = interactive.

    Example::

        preds = model.predict(net, X)
        plot_predictions(X, Y, preds, save_path="predictions.png")
    """
    plt = _get_plt()
    x_flat    = np.asarray(x).flatten()
    y_flat    = np.asarray(y_true).flatten()
    pred_flat = np.asarray(y_pred).flatten()

    # Sort by x for a clean prediction line
    order = np.argsort(x_flat)

    plt.figure(figsize=(9, 4))
    plt.scatter(x_flat, y_flat, s=20, alpha=0.7, label="True", zorder=3)
    plt.plot(x_flat[order], pred_flat[order],
             color="red", linewidth=2, label="Predicted")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.4)
    _finish(plt, save_path, title)


# ──────────────────────────────────────────────────────────────────────────────
# Activation functions overview
# ──────────────────────────────────────────────────────────────────────────────

def plot_activations(
    names:     Sequence[str] | None = None,
    *,
    v_range:   tuple[float, float] = (-4.0, 4.0),
    n_points:  int        = 300,
    save_path: str | None = None,
) -> None:
    """
    Plot forward pass and derivative of each activation function in a grid.

    Args:
        names:     activation names to include.  Defaults to all 7 supported
                   by ``nn_core.ActivationFn``:
                   ``["linear", "sigmoid", "tanh", "relu", "leaky_relu", "elu"]``.
        v_range:   (min, max) range for the input axis.
        n_points:  number of sample points.
        save_path: file path to save; ``None`` = interactive.

    Example::

        from modules.plot_utils import plot_activations
        plot_activations(save_path="activations.png")
        plot_activations(["sigmoid", "tanh", "relu"])
    """
    from nn_core.activations import ActivationFn  # noqa: PLC0415

    if names is None:
        names = ["linear", "sigmoid", "tanh", "relu", "leaky_relu", "elu"]

    af  = ActivationFn()
    v   = np.linspace(v_range[0], v_range[1], n_points)
    plt = _get_plt()

    ncols = min(len(names), 3)
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows),
                             sharey=False)
    axes_flat = np.array(axes).flatten() if nrows * ncols > 1 else [axes]

    for ax, name in zip(axes_flat, names):
        layer = {"activation_potential": v, "output": None}
        y  = af.output(layer, name)
        layer["output"] = y
        dy = af.output(layer, name, derivative=True)

        ax.plot(v, y,  linewidth=2,   label="f(v)")
        ax.plot(v, dy, linewidth=2,   label="f′(v)", linestyle="--", alpha=0.8)
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
        ax.axvline(0, color="k", linewidth=0.5, alpha=0.4)
        ax.set_title(name, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # hide unused axes
    for ax in axes_flat[len(names):]:
        ax.set_visible(False)

    plt.suptitle("Activation functions — forward pass and derivative",
                 fontsize=13, y=1.02)
    _finish(plt, save_path, "activations")


# ──────────────────────────────────────────────────────────────────────────────
# Decision boundary (2-D classification)
# ──────────────────────────────────────────────────────────────────────────────

def plot_decision_boundary(
    model,
    net:        list,
    X:          np.ndarray,
    Y:          np.ndarray,
    *,
    h:          float      = 0.1,
    threshold:  float      = 0.5,
    title:      str        = "Decision boundary",
    save_path:  str | None = None,
) -> None:
    """
    Visualise the decision boundary of a trained binary classifier.

    Works for 2-D input features only.

    Args:
        model:     a trained :class:`~nn_core.network.NeuralNetwork` instance.
        net:       the network layer list (from ``model.create_network()``).
        X:         input data, shape ``(N, 2)``.
        Y:         labels (0 or 1), shape ``(N,)``.
        h:         grid step size (smaller = higher resolution; default 0.1).
        threshold: decision threshold (default 0.5).
        title:     plot title.
        save_path: file path to save; ``None`` = interactive.

    Example::

        from modules.plot_utils import plot_decision_boundary
        plot_decision_boundary(model, net, X_train, Y_train,
                               save_path="boundary.png")
    """
    plt = _get_plt()
    X   = np.asarray(X)
    Y   = np.asarray(Y).flatten()

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy       = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))

    grid    = np.c_[xx.ravel(), yy.ravel()]
    preds   = np.array(model.predict(net, grid)).flatten()
    Z       = (preds >= threshold).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.35, cmap="coolwarm")
    plt.contour( xx, yy, Z, levels=[0.5], colors="black", linewidths=1)

    for cls, marker, color, label in [
        (0, "o", "steelblue",  "Class 0"),
        (1, "^", "darkorange", "Class 1"),
    ]:
        mask = Y == cls
        plt.scatter(X[mask, 0], X[mask, 1],
                    marker=marker, c=color,
                    edgecolors="white", s=40, zorder=3, label=label)

    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    _finish(plt, save_path, title)


__all__ = [
    "guard_backend",
    "plot_activations",
    "plot_decision_boundary",
    "plot_loss_history",
    "plot_predictions",
]
