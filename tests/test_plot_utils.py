"""Tests for modules.plot_utils (non-visual, headless-safe)."""
import os
import numpy as np
import pytest

# Force Agg backend for all tests — no display required
os.environ.setdefault("MPLBACKEND", "Agg")

from modules.plot_utils import (
    guard_backend,
    plot_activations,
    plot_decision_boundary,
    plot_loss_history,
    plot_predictions,
)
from nn_core import NeuralNetwork


class TestGuardBackend:
    def test_runs_without_error(self):
        guard_backend()   # Agg is already set; should be a no-op


class TestPlotLossHistory:
    def test_saves_to_file(self, tmp_path):
        path = str(tmp_path / "loss.png")
        plot_loss_history([1.0, 0.5, 0.3], save_path=path)
        assert os.path.isfile(path)

    def test_train_and_test(self, tmp_path):
        path = str(tmp_path / "loss.png")
        plot_loss_history([1.0, 0.8], test_losses=[0.9, 0.7], save_path=path)
        assert os.path.isfile(path)

    def test_custom_labels(self, tmp_path):
        path = str(tmp_path / "loss.png")
        plot_loss_history([0.5, 0.4], title="My loss", xlabel="Step",
                          ylabel="MSE", save_path=path)
        assert os.path.isfile(path)


class TestPlotPredictions:
    def test_saves_to_file(self, tmp_path):
        path = str(tmp_path / "pred.png")
        X = np.linspace(0, 1, 20)
        plot_predictions(X, X ** 2, X ** 2 + 0.1, save_path=path)
        assert os.path.isfile(path)

    def test_accepts_2d_input(self, tmp_path):
        path = str(tmp_path / "pred.png")
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        Y = X ** 2
        plot_predictions(X, Y, Y + 0.05, save_path=path)
        assert os.path.isfile(path)


class TestPlotActivations:
    def test_default_all_activations(self, tmp_path):
        path = str(tmp_path / "act.png")
        plot_activations(save_path=path)
        assert os.path.isfile(path)

    def test_subset_of_activations(self, tmp_path):
        path = str(tmp_path / "act.png")
        plot_activations(["sigmoid", "tanh"], save_path=path)
        assert os.path.isfile(path)

    def test_custom_range(self, tmp_path):
        path = str(tmp_path / "act.png")
        plot_activations(["relu"], v_range=(-2.0, 2.0), save_path=path)
        assert os.path.isfile(path)


class TestPlotDecisionBoundary:
    def test_saves_to_file(self, tmp_path):
        np.random.seed(0)
        # Build and train a tiny classifier
        model = NeuralNetwork()
        structure = [
            {"type": "input", "units": 2},
            {"type": "dense", "units": 4, "activation_function": "tanh",    "bias": True},
            {"type": "dense", "units": 1, "activation_function": "sigmoid", "bias": True},
        ]
        net = model.create_network(structure)

        # Simple linearly-separable data
        X = np.array([[0.0, 0.0], [0.1, 0.1], [2.0, 2.0], [2.1, 2.1]])
        Y = np.array([0.0, 0.0, 1.0, 1.0])

        model.train(net, X, Y, l_rate=0.1, n_epoch=10, verbose=0)

        path = str(tmp_path / "boundary.png")
        plot_decision_boundary(model, net, X, Y, h=0.5, save_path=path)
        assert os.path.isfile(path)
