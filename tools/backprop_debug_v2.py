"""
Generic Backprop (ONE SAMPLE, ONE ITERATION) — Any layer sizes (NO bias by default)

You can set:
- number of inputs (D)
- hidden layer count (L_hidden)
- neurons per hidden layer (list)
- number of outputs (K)
- provide weights as matrices (per layer)
Then it will run:
forward pass + backprop deltas + gradients + updated weights
with FULL debug logs.

Notation (lecture-friendly):
- a[0] = x (inputs)
- For layer l = 0..(num_layers-1):
    net[l] = W[l] @ a[l]
    a[l+1] = sigmoid(net[l])
- Output delta (MSE + sigmoid):
    delta_last = (t - y) * y*(1-y)
- Hidden deltas:
    delta[l] = (W[l+1].T @ delta[l+1]) * a[l+1]*(1-a[l+1])

Update rule (same as your earlier script):
    W[l] := W[l] + lr * outer(delta[l], a[l])

IMPORTANT:
- Shapes: W[l] is (n_{l+1}, n_l)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nn_core.logger import ObjLogger, title_message

logger = ObjLogger("Test")

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime_from_output(y: np.ndarray) -> np.ndarray:
    return y * (1.0 - y)


def sse_half_loss(t: np.ndarray, y: np.ndarray) -> float:
    return float(0.5 * np.sum((t - y) ** 2))


# -----------------------------
# Generic feedforward net (NO bias)
# -----------------------------
@dataclass
class GenericNetNoBias:
    """
    W_list: list of weight matrices
      W_list[l] shape = (n_{l+1}, n_l)

    Example:
      D=2, hidden=[2,2], K=1
      sizes = [2,2,2,1]
      W0: (2,2), W1: (2,2), W2:(1,2)
    """
    W_list: List[np.ndarray]

    def _sizes_from_weights(self) -> List[int]:
        sizes = [self.W_list[0].shape[1]]
        for W in self.W_list:
            sizes.append(W.shape[0])
        return sizes

    def check_shapes(self) -> None:
        sizes = self._sizes_from_weights()
        for l, W in enumerate(self.W_list):
            expected = (sizes[l + 1], sizes[l])
            if W.shape != expected:
                raise ValueError(
                    f"W_list[{l}] shape {W.shape} does not match expected {expected} "
                    f"from sizes {sizes}"
                )

    def forward(self, x: np.ndarray, verbose: bool = True) -> Dict[str, List[np.ndarray]]:
        """
        Returns:
          a_list: activations, a_list[0]=x, a_list[-1]=y
          net_list: net inputs per layer (same length as W_list)
        """
        self.check_shapes()

        a_list: List[np.ndarray] = [x.astype(float)]
        net_list: List[np.ndarray] = []

        if verbose:
            title_message("FORWARD PASS")

        for l, W in enumerate(self.W_list):
            net = W @ a_list[l]
            a = sigmoid(net)

            net_list.append(net)
            a_list.append(a)

            if verbose:
                logger(f"Layer {l}:")
                logger(f"  W[{l}] shape = {W.shape}")
                logger(f"  a[{l}] (input to this layer) shape = {a_list[l].shape}  value = {a_list[l]}")
                logger(f"  net[{l}] = W[{l}] @ a[{l}]  shape = {net.shape}  value = {net}")
                logger(f"  a[{l+1}] = sigmoid(net[{l}]) shape = {a.shape}  value = {a}")

        return {"a_list": a_list, "net_list": net_list}

    def backprop_one_sample(
        self,
        x: np.ndarray,
        t: np.ndarray,
        lr: float,
        verbose: bool = True,
    ) -> Tuple["GenericNetNoBias", Dict[str, object]]:
        """
        One full iteration:
          forward -> deltas -> gradients -> update -> loss before/after

        Uses MSE (1/2 SSE) + sigmoid output.
        Update rule: W := W + lr * outer(delta, a_prev)
        """
        self.check_shapes()

        # -------------------------
        # Forward
        # -------------------------
        cache = self.forward(x, verbose=verbose)
        a_list: List[np.ndarray] = cache["a_list"]
        y = a_list[-1]

        E_before = sse_half_loss(t, y)

        if verbose:
            title_message("LOSS BEFORE")
            logger(f"t (target) = {t}")
            logger(f"y (output) = {y}")
            logger(f"E_before (1/2 SSE) = {E_before}")

        # -------------------------
        # Backward (deltas)
        # -------------------------
        num_layers = len(self.W_list)  # how many weight layers
        delta_list: List[Optional[np.ndarray]] = [None] * num_layers

        # Output delta (last layer)
        delta_last = (t - y) * sigmoid_prime_from_output(y)
        delta_list[-1] = delta_last

        if verbose:
            title_message("BACKWARD PASS (DELTAS)")
            logger(f"delta[{num_layers-1}] (output layer) = {delta_last}")

        # Hidden deltas (from last-1 down to 0)
        for l in range(num_layers - 2, -1, -1):
            # a_list[l+1] is activation of layer l (its output)
            a_curr = a_list[l + 1]
            delta_next = delta_list[l + 1]
            W_next = self.W_list[l + 1]

            delta = (W_next.T @ delta_next) * sigmoid_prime_from_output(a_curr)
            delta_list[l] = delta

            if verbose:
                logger(f"delta[{l}] = (W[{l+1}].T @ delta[{l+1}]) * a[{l+1}]*(1-a[{l+1}])")
                logger(f"  W[{l+1}].T shape = {W_next.T.shape}")
                logger(f"  delta[{l+1}] shape = {delta_next.shape}  value = {delta_next}")
                logger(f"  a[{l+1}] shape = {a_curr.shape}  value = {a_curr}")
                logger(f"  delta[{l}] shape = {delta.shape}  value = {delta}")

        # -------------------------
        # Gradients + Update
        # -------------------------
        if verbose:
            title_message("GRADIENTS + WEIGHT UPDATES")

        new_W_list: List[np.ndarray] = []

        dW_list: List[np.ndarray] = []
        for l in range(num_layers):
            delta = delta_list[l]          # shape (n_{l+1},)
            a_prev = a_list[l]             # shape (n_l,)
            dW = np.outer(delta, a_prev)   # shape (n_{l+1}, n_l)
            W_new = self.W_list[l] + lr * dW

            dW_list.append(dW)
            new_W_list.append(W_new)

            if verbose:
                logger(f"\nLayer {l} update:")
                logger(f"  a_prev = a[{l}] (input to layer)        = {a_prev}")
                logger(f"  delta  = delta[{l}] (this layer delta) = {delta}")
                logger(f"  dW[{l}] = outer(delta[{l}], a[{l}]) =\n{dW}")
                logger(f"  W[{l}] BEFORE =\n{self.W_list[l]}")
                logger(f"  W[{l}] AFTER  =\n{W_new}")

        new_net = GenericNetNoBias(W_list=new_W_list)

        # -------------------------
        # Loss after (sanity)
        # -------------------------
        y_after = new_net.forward(x, verbose=False)["a_list"][-1]
        E_after = sse_half_loss(t, y_after)

        if verbose:
            title_message("LOSS AFTER (SANITY CHECK)")
            logger(f"y_after = {y_after}")
            logger(f"E_after = {E_after}")
            logger("If E_after > E_before, try smaller lr or check target scale vs sigmoid outputs.")

        details = {
            "a_list": a_list,
            "net_list": cache["net_list"],
            "delta_list": delta_list,
            "dW_list": dW_list,
            "E_before": E_before,
            "E_after": E_after,
            "y_before": y,
            "y_after": y_after,
        }
        return new_net, details


# -----------------------------
# Helper: create empty matrices for your sizes
# -----------------------------
def make_weight_shapes(num_inputs: int, hidden_sizes: List[int], num_outputs: int) -> List[Tuple[int, int]]:
    """
    Returns list of shapes for W matrices:
      sizes = [D] + hidden_sizes + [K]
      W[l] shape = (sizes[l+1], sizes[l])
    """
    sizes = [num_inputs] + hidden_sizes + [num_outputs]
    shapes = []
    for l in range(len(sizes) - 1):
        shapes.append((sizes[l + 1], sizes[l]))
    return shapes


def prompt_or_edit_weights_example() -> List[np.ndarray]:
    """
    EXAMPLE ONLY (edit these matrices).
    This example corresponds to:
      inputs=2
      hidden_layers=2 with [2,2]
      outputs=1
    """
    # sizes = [2,2,2,1]
    W0 = np.array([
        [0.10, -0.20],
        [0.40,  0.30],
    ], dtype=float)  # (2,2)

    W1 = np.array([
        [0.20, -0.50],
        [0.30,  0.80],
    ], dtype=float)  # (2,2)

    W2 = np.array([
        [0.70, -0.10],
    ], dtype=float)  # (1,2)

    return [W0, W1, W2]


# -----------------------------
# MAIN: change only this section for your exam questions
# -----------------------------
def main() -> None:
    title_message("CONFIG (EDIT HERE)")

    # 1) Tell the network structure
    num_inputs = 2
    hidden_sizes = [2]      # e.g. [2] means 1 hidden layer with 2 neurons
    num_outputs = 2         # e.g. 2 outputs

    # Print what shapes you MUST provide
    shapes = make_weight_shapes(num_inputs, hidden_sizes, num_outputs)
    logger(f"Network sizes: {[num_inputs] + hidden_sizes + [num_outputs]}")
    logger("Required weight matrix shapes (W[l] = (next_layer, prev_layer)):")
    for i, sh in enumerate(shapes):
        logger(f"  W[{i}] shape = {sh}")

    # 2) Provide x, t, lr
    # NOTE: targets must match output size
    x = np.array([4.0, 4.0], dtype=float)      # inputs
    t = np.array([2.0, 2.0], dtype=float)      # targets
    lr = 0.5

    # 3) Provide weight matrices in the SAME order as shapes list
    # For your Task2 (2-2-2), you need TWO matrices:
    #   W[0] shape (2,2)  input->hidden
    #   W[1] shape (2,2)  hidden->output
    W0 = np.array([
        [0.9, 0.4],   # hidden neuron 1: [w^(0)_11, w^(0)_21]
        [0.6, 0.6],   # hidden neuron 2: [w^(0)_12, w^(0)_22]
    ], dtype=float)

    W1 = np.array([
        [0.4, 0.2],   # output neuron 1: [w^(1)_11, w^(1)_21]
        [0.5, 0.7],   # output neuron 2: [w^(1)_12, w^(1)_22]
    ], dtype=float)

    W_list = [W0, W1]

    # 4) Run one iteration
    net = GenericNetNoBias(W_list=W_list)
    net2, details = net.backprop_one_sample(x=x, t=t, lr=lr, verbose=True)

    title_message("DONE")


if __name__ == "__main__":
    main()
