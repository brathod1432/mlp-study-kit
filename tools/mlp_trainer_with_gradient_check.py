import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Literal
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nn_core.logger import ObjLogger, title_message
import numpy as np
logger = ObjLogger("MLPDebug")
LossType = Literal["mse_sigmoid", "cross_entropy_sigmoid"]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime_from_output(y: np.ndarray) -> np.ndarray:
    """Sigmoid'(z) using y = sigmoid(z)."""
    return y * (1.0 - y)


def mse_loss(t: np.ndarray, y: np.ndarray) -> float:
    """1/2 sum squared error."""
    return float(0.5 * np.sum((t - y) ** 2))


def cross_entropy_loss_sigmoid(t: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """CE for sigmoid outputs (multi-label/binary)."""
    y = np.clip(y, eps, 1.0 - eps)
    return float(-np.sum(t * np.log(y) + (1.0 - t) * np.log(1.0 - y)))


@dataclass
class MLP:
    """
    Fully-connected MLP with sigmoid at every layer.
    We store weights as W[l] of shape (n_l, n_{l-1}) and biases b[l] of shape (n_l,).
    """
    W: List[np.ndarray]
    b: List[np.ndarray]

    @staticmethod
    def init(layer_sizes: List[int], seed: int = 7, scale: float = 0.1) -> "MLP":
        """Initialize weights for sizes like [D, H1, H2, K]."""
        rng = np.random.default_rng(seed)
        W: List[np.ndarray] = []
        b: List[np.ndarray] = []
        for i in range(1, len(layer_sizes)):
            W.append(scale * rng.standard_normal((layer_sizes[i], layer_sizes[i - 1])))
            b.append(np.zeros((layer_sizes[i],), dtype=float))
        return MLP(W=W, b=b)

    def forward(self, x: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """
        Forward pass: returns activations a[0..L] and pre-activations z[1..L].
        a[0]=x
        """
        a: List[np.ndarray] = [x]
        z: List[np.ndarray] = []
        for l in range(len(self.W)):
            zl = self.W[l] @ a[-1] + self.b[l]    # pre-activation
            al = sigmoid(zl)                      # activation
            z.append(zl)
            a.append(al)
        return {"a": a, "z": z}

    def loss(self, t: np.ndarray, y: np.ndarray, loss_type: LossType) -> float:
        """Compute loss."""
        if loss_type == "mse_sigmoid":
            return mse_loss(t, y)
        if loss_type == "cross_entropy_sigmoid":
            return cross_entropy_loss_sigmoid(t, y)
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    def backward(
        self,
        cache: Dict[str, List[np.ndarray]],
        t: np.ndarray,
        loss_type: LossType,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward pass producing gradients dW, db.
        Uses lecture-style delta propagation.
        """
        a = cache["a"]
        # z = cache["z"]  # not required since we use sigmoid'(a)

        L = len(self.W)  # number of layers with weights
        dW = [np.zeros_like(Wl) for Wl in self.W]
        db = [np.zeros_like(bl) for bl in self.b]

        y = a[-1]  # output activation

        # Output delta
        if loss_type == "mse_sigmoid":
            delta = (t - y) * sigmoid_prime_from_output(y)
        else:
            # CE+sigmoid under lecture-style "w += lr * delta * input"
            delta = (t - y)

        # Last layer gradients
        dW[L - 1] = np.outer(delta, a[L - 1])
        db[L - 1] = delta

        # Hidden layers (backward)
        for l in range(L - 2, -1, -1):
            delta = (self.W[l + 1].T @ delta) * sigmoid_prime_from_output(a[l + 1])
            dW[l] = np.outer(delta, a[l])
            db[l] = delta

        return dW, db

    def sgd_step(
        self,
        x: np.ndarray,
        t: np.ndarray,
        lr: float,
        loss_type: LossType,
        verbose: bool = False,
    ) -> float:
        """One SGD step on a single sample."""
        cache = self.forward(x)
        y = cache["a"][-1]
        E = self.loss(t, y, loss_type)

        dW, db = self.backward(cache, t, loss_type)

        # Update
        for l in range(len(self.W)):
            self.W[l] += lr * dW[l]
            self.b[l] += lr * db[l]

        if verbose:
            logger(f"Loss={E:.8f}", "yellow")
        return E


def finite_difference_gradient_check(
    model: MLP,
    x: np.ndarray,
    t: np.ndarray,
    loss_type: LossType = "mse_sigmoid",
    eps: float = 1e-5,
    max_params_to_check: int = 20,
) -> None:
    """
    Numerical gradient check: compare backprop gradients with finite differences.
    Great for debugging sign mistakes, wrong layer indexing, or wrong delta formulas.
    """
    title_message("GRADIENT CHECK", "magenta")

    cache = model.forward(x)
    y = cache["a"][-1]
    base_loss = model.loss(t, y, loss_type)

    dW, db = model.backward(cache, t, loss_type)

    checks_done = 0
    worst_rel_err = 0.0

    # Check some weights
    for l in range(len(model.W)):
        Wl = model.W[l]
        for i in range(Wl.shape[0]):
            for j in range(Wl.shape[1]):
                if checks_done >= max_params_to_check:
                    break

                original = Wl[i, j]

                Wl[i, j] = original + eps
                loss_plus = model.loss(t, model.forward(x)["a"][-1], loss_type)

                Wl[i, j] = original - eps
                loss_minus = model.loss(t, model.forward(x)["a"][-1], loss_type)

                Wl[i, j] = original  # restore

                num_grad = (loss_plus - loss_minus) / (2.0 * eps)
                bp_grad = -dW[l][i, j]  # NOTE: because our update uses W += lr*dW, dW is "ascent direction"
                # Convert to comparable "dLoss/dW": dLoss/dW ≈ -dW (under our convention)

                denom = max(1e-12, abs(num_grad) + abs(bp_grad))
                rel_err = abs(num_grad - bp_grad) / denom

                worst_rel_err = max(worst_rel_err, rel_err)
                logger(
                    f"Layer {l} W[{i},{j}] | num={num_grad:.6e} vs bp={bp_grad:.6e} | rel_err={rel_err:.3e}",
                    "cyan" if rel_err < 1e-3 else "red",
                )
                checks_done += 1
            if checks_done >= max_params_to_check:
                break
        if checks_done >= max_params_to_check:
            break

    logger(f"Worst relative error: {worst_rel_err:.3e}", "green" if worst_rel_err < 1e-3 else "red")
    title_message("CHECK COMPLETE", "green")


def demo_train_and_check() -> None:
    """Demo training + gradient check."""
    title_message("DEMO: TRAIN + CHECK", "blue")

    # XOR-like dataset
    X = np.array([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]], dtype=float)
    T = np.array([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]], dtype=float)

    model = MLP.init([2, 4, 1], seed=7, scale=0.5)

    # Gradient check one sample before training
    finite_difference_gradient_check(model, X[1], T[1], loss_type="mse_sigmoid", eps=1e-5, max_params_to_check=15)

    # Train
    lr = 0.8
    epochs = 2000
    for epoch in range(1, epochs + 1):
        losses = []
        for x, t in zip(X, T):
            losses.append(model.sgd_step(x, t, lr=lr, loss_type="mse_sigmoid", verbose=False))
        if epoch % 200 == 0:
            logger(f"Epoch {epoch}/{epochs} | Avg Loss = {float(np.mean(losses)):.6f}", "yellow")

    # Predictions
    title_message("PREDICTIONS", "cyan")
    for x, t in zip(X, T):
        y = model.forward(x)["a"][-1]
        logger(f"x={x} target={t[0]:.0f} pred={y[0]:.4f}", "white")


if __name__ == "__main__":
    demo_train_and_check()
