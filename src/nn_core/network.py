"""
nn_core.network -- canonical NeuralNetwork class for mlp-study-kit.

Full MLP (fully-connected feed-forward network) built on the layer-dict
representation used across the exercise series:

Layer dict keys
---------------
weights             np.ndarray  shape (n_units, n_prev [+ 1 if bias])
bias                bool        True → a bias neuron (value 1) is appended to input
activation_function str         name understood by ActivationFn
activation_potential np.ndarray filled during forward_propagate()
output              np.ndarray  filled during forward_propagate()
delta               np.ndarray  filled during backward_propagate()

Evolution across exercises
--------------------------
ex07  create_network, forward_propagate, predict
ex08  derivative flag added throughout; backward/update remain stubs
ex09  full backprop, update_weights, training loop
ex10  bias terms, train/test split, basic_early_stop

Usage
-----
    from nn_core.network import NeuralNetwork

    structure = [
        {"type": "input",  "units": 1},
        {"type": "dense",  "units": 32, "activation_function": "tanh",   "bias": True},
        {"type": "dense",  "units": 1,  "activation_function": "linear", "bias": True},
    ]
    model = NeuralNetwork()
    net   = model.create_network(structure)
    model.train(net, X_train, Y_train, X_test, Y_test, l_rate=0.01, n_epoch=500)
    preds = model.predict(net, X_test)

    # Persist learned weights
    model.save_weights(net, "weights.npy")
    model.load_weights(net, "weights.npy")

    print(model)   # shows architecture summary
"""

from __future__ import annotations

import os

import numpy as np

from nn_core.activations import ActivationFn
from nn_core.losses import LossFn


class NeuralNetwork:
    """
    Fully-connected MLP with optional per-layer bias, backpropagation,
    early stopping, weight persistence, and a human-readable repr.
    """

    def __init__(self, structure: list | None = None) -> None:
        self.af: ActivationFn = ActivationFn()
        self.loss: LossFn = LossFn()
        self.nnetwork: list[dict] = []
        if structure:
            self.create_network(structure)

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def create_network(self, structure: list[dict]) -> list[dict]:
        """
        Build the layer list from a structure specification.

        Args:
            structure: list of dicts. First entry is the input spec
                       (``{"type": "input", "units": N}``). Remaining
                       entries are dense layers with keys:
                       ``units``, ``activation_function``, ``bias`` (bool).

        Returns:
            The populated ``self.nnetwork`` list (also stored on ``self``).
        """
        self.nnetwork = [structure[0]]
        for i in range(1, len(structure)):
            spec = structure[i]
            n    = spec["units"]
            m    = structure[i - 1]["units"]
            bias = spec.get("bias", False)
            # Weight matrix: n_units rows × (n_prev + bias_column) cols
            w = np.random.randn(n, m + int(bias)) * 0.2
            self.nnetwork.append({
                "weights":              w,
                "bias":                 bias,
                "activation_function":  spec["activation_function"],
                "activation_potential": None,
                "output":               None,
                "delta":                None,
            })
        return self.nnetwork

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward_propagate(self, nnetwork: list[dict], inputs: np.ndarray) -> np.ndarray:
        """
        Propagate ``inputs`` through the network and return the output
        of the final layer.

        Side effects: fills ``activation_potential`` and ``output`` in
        every layer dict (needed by :meth:`backward_propagate`).

        Args:
            nnetwork: layer list returned by :meth:`create_network`.
            inputs:   1-D input array (shape ``(n_inputs,)``).

        Returns:
            Output of the last layer as a 1-D ``np.ndarray``.
        """
        inp = np.asarray(inputs, dtype=float).flatten()
        for i in range(1, len(nnetwork)):
            layer = nnetwork[i]
            if layer["bias"]:
                inp = np.append(inp, 1.0)
            layer["activation_potential"] = np.matmul(layer["weights"], inp).flatten()
            layer["output"] = self.af.output(layer, layer["activation_function"])
            inp = layer["output"]
        return inp

    def predict(
        self, nnetwork: list[dict], inputs: list | np.ndarray
    ) -> list[np.ndarray]:
        """
        Run forward propagation on a collection of samples.

        Args:
            nnetwork: layer list from :meth:`create_network`.
            inputs:   iterable of 1-D input arrays.

        Returns:
            List of output arrays, one per input sample.
        """
        return [self.forward_propagate(nnetwork, np.asarray(x)) for x in inputs]

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward_propagate(
        self,
        loss_function: str,
        nnetwork: list[dict],
        expected: np.ndarray,
    ) -> None:
        """
        Compute deltas via backpropagation and store them in each layer dict.

        Delta computation (lecture notation):
            Output layer:  δ = ∂L/∂y · f′(v)
            Hidden layers: δ = (W_next.T @ δ_next) · f′(v)

        where v = activation_potential, y = output, f′ = activation derivative.

        Args:
            loss_function: loss name (e.g. ``"mse"``).
            nnetwork:      layer list (must have been forward-propagated first).
            expected:      target output array.
        """
        N = len(nnetwork) - 1
        for i in range(N, 0, -1):
            if i < N:
                w = nnetwork[i + 1]["weights"]
                if nnetwork[i + 1]["bias"]:
                    w = w[:, :-1]   # strip bias column before propagating error
                errors = np.matmul(nnetwork[i + 1]["delta"], w)
            else:
                errors = self.loss.output(
                    loss_function,
                    np.asarray(expected, dtype=float),
                    nnetwork[-1]["output"],
                    derivative=True,
                )
            nnetwork[i]["delta"] = np.multiply(
                errors,
                self.af.output(nnetwork[i], nnetwork[i]["activation_function"], derivative=True),
            )

    def update_weights(
        self, nnetwork: list[dict], inputs: np.ndarray, l_rate: float
    ) -> None:
        """
        Apply the SGD weight update rule to every layer.

        Update rule:  W ← W − η · δ ⊗ a_prev
        where ⊗ is the outer product, η is the learning rate, and
        δ is the delta stored by :meth:`backward_propagate`.

        Args:
            nnetwork: layer list (must have been backward-propagated first).
            inputs:   the original input sample (same as passed to forward_propagate).
            l_rate:   learning rate η.
        """
        inp = np.asarray(inputs, dtype=float).flatten()
        for i in range(1, len(nnetwork)):
            if nnetwork[i]["bias"]:
                inp = np.append(inp, 1.0)
            nnetwork[i]["weights"] -= l_rate * np.outer(nnetwork[i]["delta"], inp)
            inp = nnetwork[i]["output"]

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------

    def basic_early_stop(self, history_test: list[float], epsilon: float) -> bool:
        """
        Return ``True`` when training should stop (loss has plateaued).

        Stopping criterion: improvement < epsilon
            improvement = prev_loss − curr_loss

        If the improvement drops below ``epsilon`` (or loss is increasing,
        giving negative improvement) training is no longer beneficial.

        Args:
            history_test: list of per-epoch test losses (at least 2 entries).
            epsilon:      minimum improvement threshold.
        """
        improvement = history_test[-2] - history_test[-1]
        return improvement < epsilon

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        nnetwork: list[dict],
        x_train: list | np.ndarray,
        y_train: list | np.ndarray,
        x_test:  list | np.ndarray | None = None,
        y_test:  list | np.ndarray | None = None,
        l_rate:      float = 0.01,
        n_epoch:     int   = 100,
        loss_function: str = "mse",
        epsilon:     float = 0.0,
        verbose:     int   = 1,
        save_plot:   str | None = None,
    ) -> float:
        """
        Train for up to ``n_epoch`` epochs with optional early stopping.

        Args:
            x_train / y_train:  training data (iterable of arrays).
            x_test  / y_test:   optional held-out test data. Required for
                                early stopping.
            l_rate:             learning rate η.
            n_epoch:            maximum training epochs.
            loss_function:      ``"mse"`` or ``"binary_cross_entropy"``.
            epsilon:            early-stop threshold; 0 = disabled. Training
                                stops when test-loss improvement < epsilon.
            verbose:            1 = print per-epoch loss and show/save plot;
                                0 = silent.
            save_plot:          if given, saves the loss history plot to this
                                file path instead of showing an interactive
                                window. Useful for CI / headless environments.
                                Example: ``save_plot="loss.png"``.
                                Set ``MPLBACKEND=Agg`` to suppress all windows.

        Returns:
            Final training loss of the last completed epoch.
        """
        history_train: list[float] = []
        history_test:  list[float] = []
        use_test = x_test is not None and y_test is not None

        for epoch in range(n_epoch):
            # ── train pass ──────────────────────────────────────────────
            err_sum = 0.0
            for x_row, y_row in zip(x_train, y_train):
                x_arr = np.asarray(x_row, dtype=float)
                y_arr = np.asarray(y_row, dtype=float)
                self.forward_propagate(nnetwork, x_arr)
                self.backward_propagate(loss_function, nnetwork, y_arr)
                self.update_weights(nnetwork, x_arr, l_rate)
                err_sum += float(
                    np.sum(self.loss.output(loss_function, y_arr, nnetwork[-1]["output"]))
                )
            history_train.append(err_sum / len(x_train))

            # ── test pass (optional) ─────────────────────────────────────
            if use_test:
                t_err = 0.0
                for x_row, y_row in zip(x_test, y_test):
                    self.forward_propagate(nnetwork, np.asarray(x_row, dtype=float))
                    t_err += float(
                        np.sum(self.loss.output(
                            loss_function,
                            np.asarray(y_row, dtype=float),
                            nnetwork[-1]["output"],
                        ))
                    )
                history_test.append(t_err / len(x_test))

            if verbose > 0:
                if use_test:
                    print(
                        f">epoch={epoch + 1}, "
                        f"loss_train={history_train[-1]:.4f}, "
                        f"loss_test={history_test[-1]:.4f}"
                    )
                else:
                    print(f">epoch={epoch + 1}, loss={history_train[-1]:.4f}")

            # ── early stop ───────────────────────────────────────────────
            if use_test and epoch > 3 and epsilon > 0:
                if self.basic_early_stop(history_test, epsilon):
                    print(f"Early stopping at epoch {epoch + 1} (improvement < {epsilon})")
                    break

        # ── optional loss plot ───────────────────────────────────────────
        if verbose > 0:
            self._plot_history(history_train, history_test, save_plot)

        return history_train[-1]

    # ------------------------------------------------------------------
    # Weight persistence
    # ------------------------------------------------------------------

    def save_weights(self, nnetwork: list[dict], path: str) -> None:
        """
        Save the trained weight matrices to a NumPy ``.npy`` file.

        Only the weight arrays are saved (not biases flags or other
        layer metadata). Load back with :meth:`load_weights`.

        Args:
            nnetwork: trained layer list.
            path:     file path, e.g. ``"my_model.npy"``.

        Example::

            model.train(net, X_train, Y_train, ...)
            model.save_weights(net, "weights.npy")
        """
        weights = np.array(
            [layer["weights"] for layer in nnetwork[1:]],
            dtype=object,
        )
        np.save(path, weights, allow_pickle=True)
        print(f"[NeuralNetwork] Weights saved to: {path}")

    def load_weights(self, nnetwork: list[dict], path: str) -> None:
        """
        Load previously saved weight matrices into an existing network.

        The network must have been created with the same structure as
        when the weights were saved (shapes must match exactly).

        Args:
            nnetwork: layer list (created by :meth:`create_network`).
            path:     file path produced by :meth:`save_weights`.

        Example::

            model.load_weights(net, "weights.npy")
            preds = model.predict(net, X_test)
        """
        weights = np.load(path, allow_pickle=True)
        if len(weights) != len(nnetwork) - 1:
            raise ValueError(
                f"Weight count mismatch: file has {len(weights)} layers, "
                f"network has {len(nnetwork) - 1} dense layers."
            )
        for i, w in enumerate(weights):
            expected = nnetwork[i + 1]["weights"].shape
            if np.asarray(w).shape != expected:
                raise ValueError(
                    f"Shape mismatch at layer {i + 1}: "
                    f"file={np.asarray(w).shape}, network={expected}"
                )
            nnetwork[i + 1]["weights"] = np.asarray(w, dtype=float)
        print(f"[NeuralNetwork] Weights loaded from: {path}")

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self.nnetwork:
            return "NeuralNetwork(uninitialised)"
        lines = ["NeuralNetwork("]
        for i, layer in enumerate(self.nnetwork):
            if layer.get("type") == "input" or "weights" not in layer:
                lines.append(f"  [{i}] input   units={layer.get('units', '?')}")
            else:
                w     = layer["weights"]
                bias  = layer.get("bias", False)
                act   = layer.get("activation_function", "?")
                lines.append(
                    f"  [{i}] dense   units={w.shape[0]:<4} "
                    f"act={act:<12} bias={str(bias):<5} "
                    f"weights={w.shape}"
                )
        lines.append(")")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _plot_history(
        history_train: list[float],
        history_test:  list[float],
        save_plot:     str | None,
    ) -> None:
        """Render the training/test loss history. Lazy-imports matplotlib."""
        # Honour MPLBACKEND env var before importing pyplot
        _backend = os.environ.get("MPLBACKEND", "")
        import matplotlib                           # noqa: PLC0415 (lazy import intentional)
        if _backend:
            matplotlib.use(_backend)
        import matplotlib.pyplot as plt            # noqa: PLC0415

        plt.figure()
        plt.plot(history_train, label="Train loss")
        if history_test:
            plt.plot(history_test, label="Test loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training history")
        plt.legend()
        plt.grid(True)

        if save_plot:
            plt.savefig(save_plot, bbox_inches="tight")
            print(f"[NeuralNetwork] Loss plot saved to: {save_plot}")
            plt.close()
        else:
            plt.show(block=False)
            plt.pause(0.1)
