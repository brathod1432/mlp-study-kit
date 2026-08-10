import datetime
from dataclasses import dataclass
from typing import Dict, Tuple, Literal, Optional
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nn_core.logger import ObjLogger, title_message
import numpy as np
logger = ObjLogger("BackpropCalc")
LossType = Literal["mse_sigmoid", "cross_entropy_sigmoid"]

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-z))  # elementwise


def sigmoid_prime_from_output(y: np.ndarray) -> np.ndarray:
    """Sigmoid derivative using output y: y*(1-y)."""
    return y * (1.0 - y)  # elementwise


def mse_loss(t: np.ndarray, y: np.ndarray) -> float:
    """Classic 1/2 MSE used in many lectures."""
    return float(0.5 * np.sum((t - y) ** 2))  # scalar


def cross_entropy_loss_sigmoid(t: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Binary/multi-label cross entropy with sigmoid outputs.
    E = -sum[ t*log(y) + (1-t)*log(1-y) ].
    """
    y = np.clip(y, eps, 1.0 - eps)  # avoid log(0)
    return float(-np.sum(t * np.log(y) + (1.0 - t) * np.log(1.0 - y)))  # scalar


@dataclass
class OneHiddenLayerNet:
    """
    x -> hidden(sigmoid) -> out(sigmoid)
    This matches your lecture + exercise backprop pattern cleanly.
    """
    w1: np.ndarray  # (H, D)
    b1: np.ndarray  # (H,)
    w2: np.ndarray  # (K, H)
    b2: np.ndarray  # (K,)

    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Forward pass returning intermediates for manual-style calculation."""
        z1 = self.w1 @ x + self.b1               # (H,)
        y1 = sigmoid(z1)                         # (H,)
        z2 = self.w2 @ y1 + self.b2              # (K,)
        y2 = sigmoid(z2)                         # (K,)
        return {"x": x, "z1": z1, "y1": y1, "z2": z2, "y2": y2}

    def backprop_one_sample(
        self,
        x: np.ndarray,
        t: np.ndarray,
        lr: float,
        loss_type: LossType = "mse_sigmoid",
        verbose: bool = True,
    ) -> Tuple["OneHiddenLayerNet", Dict[str, np.ndarray]]:
        """
        One full backprop step: forward -> delta -> gradients -> update.
        Returns updated network + a dict of all values (for notebook/exam checks).
        """
        try:
            title_message("BACKPROP STEP (ONE SAMPLE)", "magenta")

            cache = self.forward(x)  # forward
            y1, y2 = cache["y1"], cache["y2"]

            # ---- Loss ----
            if loss_type == "mse_sigmoid":
                E = mse_loss(t, y2)  # scalar
            elif loss_type == "cross_entropy_sigmoid":
                E = cross_entropy_loss_sigmoid(t, y2)  # scalar
            else:
                raise ValueError(f"Unsupported loss_type: {loss_type}")

            if verbose:
                logger(f"x shape: {x.shape}, target shape: {t.shape}", "cyan")
                logger(f"y2(pred) = {y2}", "cyan")
                logger(f"Loss ({loss_type}) = {E:.8f}", "yellow")

            # ---- Deltas (Professor style) ----
            # Lecture-style for sigmoid output:
            # MSE: delta2 = (t - y2) * y2*(1-y2)
            # CE(sigmoid): delta2 = (t - y2)   (common simplification), but we keep a clear form:
            # dE/dz = (t - y) for CE with sigmoid when using gradient-ascent form used in some notes.
            if loss_type == "mse_sigmoid":
                delta2 = (t - y2) * sigmoid_prime_from_output(y2)  # (K,)
            else:
                # For CE+sigmoid, derivative w.r.t z is (t - y) if using "w += lr * delta * input" convention.
                delta2 = (t - y2)  # (K,)

            delta1 = (self.w2.T @ delta2) * sigmoid_prime_from_output(y1)  # (H,)

            if verbose:
                logger(f"delta2 (output layer) = {delta2}", "white")
                logger(f"delta1 (hidden layer) = {delta1}", "white")

            # ---- Gradients (exact exam pattern) ----
            # Δw2 = lr * outer(delta2, y1)
            # Δw1 = lr * outer(delta1, x)
            dw2 = np.outer(delta2, y1)  # (K,H)
            db2 = delta2                # (K,)
            dw1 = np.outer(delta1, x)   # (H,D)
            db1 = delta1                # (H,)

            if verbose:
                logger(f"dw2 shape {dw2.shape}, dw1 shape {dw1.shape}", "cyan")
                logger(f"dw2 =\n{dw2}", "cyan")
                logger(f"dw1 =\n{dw1}", "cyan")

            # ---- Update (matches lecture convention w += lr * delta * input) ----
            new_net = OneHiddenLayerNet(
                w1=self.w1 + lr * dw1,
                b1=self.b1 + lr * db1,
                w2=self.w2 + lr * dw2,
                b2=self.b2 + lr * db2,
            )

            # Post-check: loss after update (sanity)
            new_cache = new_net.forward(x)
            new_y2 = new_cache["y2"]
            new_E = mse_loss(t, new_y2) if loss_type == "mse_sigmoid" else cross_entropy_loss_sigmoid(t, new_y2)

            if verbose:
                logger(f"Loss AFTER update = {new_E:.8f}", "green" if new_E < E else "red")
                if not np.isfinite(new_E):
                    logger("Loss became NaN/Inf. Reduce lr or check overflow.", "red")

            out = {
                **cache,
                "t": t,
                "E_before": np.array([E]),
                "delta2": delta2,
                "delta1": delta1,
                "dw2": dw2,
                "db2": db2,
                "dw1": dw1,
                "db1": db1,
                "E_after": np.array([new_E]),
            }
            title_message("STEP COMPLETE", "green")
            return new_net, out

        except Exception as e:
            logger(f"Error in backprop_one_sample: {e}", "red")
            raise


def demo_exam_style() -> None:
    """
    Demo with small sizes.
    Replace these numbers with your exam question values.
    """
    title_message("DEMO: EXAM STYLE INPUT", "blue")

    # Example dimensions: D=2 inputs, H=2 hidden, K=1 output
    w1 = np.array([[0.10, -0.20],
                   [0.40,  0.30]], dtype=float)
    b1 = np.array([0.00, 0.00], dtype=float)

    w2 = np.array([[0.20, -0.50]], dtype=float)
    b2 = np.array([0.00], dtype=float)

    net = OneHiddenLayerNet(w1=w1, b1=b1, w2=w2, b2=b2)

    x = np.array([1.0, 0.5], dtype=float)   # input sample
    t = np.array([1.0], dtype=float)        # target
    lr = 0.5

    net2, details = net.backprop_one_sample(x=x, t=t, lr=lr, loss_type="mse_sigmoid", verbose=True)
    logger(f"Updated w2:\n{net2.w2}", "yellow")
    logger(f"Updated w1:\n{net2.w1}", "yellow")


if __name__ == "__main__":
    demo_exam_style()



"""
Put your given weights, biases, x, t, η into demo_exam_style().

Run once. It prints:

forward outputs (hidden + final),

δ for output + hidden,

gradients for each weight,

updated weights,

loss before/after.

"""
