"""
nn_core.network -- canonical NeuralNetwork class for mlp-study-kit.

This is the fully-featured version of the network that evolved across:
  ex07  -- create_network, forward_propagate, predict
  ex08  -- derivative flag added throughout (backprop stubs)
  ex09  -- full backprop, update_weights, train loop
  ex10  -- bias terms, train/test split, basic_early_stop

Layer dict structure (matches exercises):
    {
        "weights":              np.ndarray  (n_units, n_prev [+ bias])
        "bias":                 bool        (True = appends 1 to input)
        "activation_function":  str
        "activation_potential": np.ndarray  (filled during forward)
        "output":               np.ndarray  (filled during forward)
        "delta":                np.ndarray  (filled during backward)
    }

Usage:
    from nn_core.network import NeuralNetwork

    structure = [
        {"type": "input", "units": 1},
        {"type": "dense", "units": 32, "activation_function": "tanh",   "bias": True},
        {"type": "dense", "units": 32, "activation_function": "tanh",   "bias": True},
        {"type": "dense", "units": 1,  "activation_function": "linear", "bias": True},
    ]
    model = NeuralNetwork()
    net   = model.create_network(structure)
    model.train(net, X_train, Y_train, X_test, Y_test,
                l_rate=0.01, n_epoch=500, loss_function="mse")
    preds = model.predict(net, X_test)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from nn_core.activations import ActivationFn
from nn_core.losses import LossFn


class NeuralNetwork:
    """
    Fully-connected MLP with optional bias, early stopping, and
    train / test loss tracking.

    Matches the exercise API so exercises can import this class
    directly once they are ready to stop re-defining it.
    """

    def __init__(self, structure: list | None = None) -> None:
        self.af = ActivationFn()
        self.loss = LossFn()
        self.nnetwork: list = []
        if structure:
            self.create_network(structure)

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def create_network(self, structure: list) -> list:
        """Build the layer list from a structure spec."""
        self.nnetwork = [structure[0]]
        for i in range(1, len(structure)):
            layer_spec = structure[i]
            n = layer_spec["units"]
            m = structure[i - 1]["units"]
            bias = layer_spec.get("bias", False)
            # weight matrix: n rows x (m + 1 if bias else m) cols
            w = np.random.randn(n, m + int(bias)) * 0.2
            self.nnetwork.append({
                "weights":              w,
                "bias":                 bias,
                "activation_function":  layer_spec["activation_function"],
                "activation_potential": None,
                "output":               None,
                "delta":                None,
            })
        return self.nnetwork

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward_propagate(self, nnetwork: list, inputs: np.ndarray) -> np.ndarray:
        inp = inputs.copy().flatten()
        for i in range(1, len(nnetwork)):
            layer = nnetwork[i]
            if layer["bias"]:
                inp = np.append(inp, 1.0)
            layer["activation_potential"] = np.matmul(layer["weights"], inp).flatten()
            layer["output"] = self.af.output(layer, layer["activation_function"])
            inp = layer["output"]
        return inp

    def predict(self, nnetwork: list, inputs) -> list:
        return [self.forward_propagate(nnetwork, np.asarray(x)) for x in inputs]

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward_propagate(
        self, loss_function: str, nnetwork: list, expected: np.ndarray
    ) -> None:
        N = len(nnetwork) - 1
        for i in range(N, 0, -1):
            if i < N:
                w = nnetwork[i + 1]["weights"]
                if nnetwork[i + 1]["bias"]:
                    w = w[:, :-1]   # strip bias column
                errors = np.matmul(nnetwork[i + 1]["delta"], w)
            else:
                errors = self.loss.output(
                    loss_function, expected, nnetwork[-1]["output"], derivative=True
                )
            nnetwork[i]["delta"] = np.multiply(
                errors,
                self.af.output(nnetwork[i], nnetwork[i]["activation_function"], derivative=True),
            )

    def update_weights(self, nnetwork: list, inputs: np.ndarray, l_rate: float) -> None:
        inp = inputs.copy().flatten()
        for i in range(1, len(nnetwork)):
            if nnetwork[i]["bias"]:
                inp = np.append(inp, 1.0)
            nnetwork[i]["weights"] -= l_rate * np.outer(nnetwork[i]["delta"], inp)
            inp = nnetwork[i]["output"]

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------

    def basic_early_stop(self, history_test: list, epsilon: float) -> bool:
        """Return True if test loss improved by more than epsilon last epoch."""
        return (history_test[-2] - history_test[-1]) > epsilon

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        nnetwork: list,
        x_train,
        y_train,
        x_test=None,
        y_test=None,
        l_rate: float = 0.01,
        n_epoch: int = 100,
        loss_function: str = "mse",
        epsilon: float = 0.0,
        verbose: int = 1,
    ) -> float:
        """
        Train for up to n_epoch epochs with optional early stopping.

        If x_test / y_test are not supplied, only train loss is tracked.
        epsilon > 0 enables early stopping on test loss.
        """
        history_train: list[float] = []
        history_test:  list[float] = []
        use_test = x_test is not None and y_test is not None

        for epoch in range(n_epoch):
            # -- train pass --
            err_sum = 0.0
            for x_row, y_row in zip(x_train, y_train):
                self.forward_propagate(nnetwork, np.asarray(x_row))
                self.backward_propagate(loss_function, nnetwork, np.asarray(y_row))
                self.update_weights(nnetwork, np.asarray(x_row), l_rate)
                err_sum += float(
                    np.sum(self.loss.output(loss_function, np.asarray(y_row),
                                            nnetwork[-1]["output"]))
                )
            history_train.append(err_sum / len(x_train))

            # -- test pass (optional) --
            if use_test:
                t_err = 0.0
                for x_row, y_row in zip(x_test, y_test):
                    self.forward_propagate(nnetwork, np.asarray(x_row))
                    t_err += float(
                        np.sum(self.loss.output(loss_function, np.asarray(y_row),
                                                nnetwork[-1]["output"]))
                    )
                history_test.append(t_err / len(x_test))

            if verbose > 0:
                if use_test:
                    print(f">epoch={epoch+1}, loss_train={history_train[-1]:.4f}, "
                          f"loss_test={history_test[-1]:.4f}")
                else:
                    print(f">epoch={epoch+1}, loss={history_train[-1]:.4f}")

            # -- early stop --
            if use_test and epoch > 3 and epsilon > 0:
                if self.basic_early_stop(history_test, epsilon):
                    print("Early stopping triggered.")
                    break

        # -- final plot --
        if verbose > 0:
            plt.figure()
            plt.plot(history_train, label="Train loss")
            if history_test:
                plt.plot(history_test, label="Test loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training history")
            plt.legend()
            plt.grid(True)
            plt.show()

        return history_train[-1]
