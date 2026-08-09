import argparse
import datetime
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nn_core.logger import ObjLogger, title_message
logger = ObjLogger("BackpropDebugger")
LossType = Literal["mse_sigmoid", "cross_entropy_sigmoid"]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-z))  # elementwise sigmoid


def sigmoid_prime_from_output(a: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid using activation output: a*(1-a)."""
    return a * (1.0 - a)  # elementwise derivative


def mse_loss(t: np.ndarray, y: np.ndarray) -> float:
    """Half-sum-squared error: 1/2 * sum((t-y)^2)."""
    return float(0.5 * np.sum((t - y) ** 2))  # scalar


def cross_entropy_loss_sigmoid(t: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Binary/multi-label cross-entropy for sigmoid outputs."""
    y = np.clip(y, eps, 1.0 - eps)  # numerical safety
    return float(-np.sum(t * np.log(y) + (1.0 - t) * np.log(1.0 - y)))  # scalar


def compute_loss(loss_type: LossType, t: np.ndarray, y: np.ndarray) -> float:
    """Dispatch loss function by type."""
    if loss_type == "mse_sigmoid":
        return mse_loss(t, y)  # MSE
    if loss_type == "cross_entropy_sigmoid":
        return cross_entropy_loss_sigmoid(t, y)  # CE
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def pretty_array(name: str, arr: np.ndarray, color: str = "cyan") -> None:
    """Pretty-print numpy arrays for debugging."""
    logger(f"{name} | shape={arr.shape}\n{arr}", color=color)


@dataclass
class DebugMLP:
    """
    Multi-layer perceptron with sigmoid activations at every layer.
    W[l] shape: (n_l, n_{l-1})
    b[l] shape: (n_l,)
    """
    W: List[np.ndarray]
    b: List[np.ndarray]

    def forward(self, x: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """
        Forward pass producing:
        a[0]=x, z[0]=z1, a[1]=a1, ... until output.
        """
        a: List[np.ndarray] = [x]
        z: List[np.ndarray] = []
        for l in range(len(self.W)):
            zl = self.W[l] @ a[-1] + self.b[l]        # pre-activation
            al = sigmoid(zl)                           # activation
            z.append(zl)
            a.append(al)
        return {"a": a, "z": z}

    def backward(
        self,
        cache: Dict[str, List[np.ndarray]],
        t: np.ndarray,
        loss_type: LossType,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Backward pass using lecture-style deltas.
        Returns (dW, db, deltas) where deltas[l] corresponds to layer l output neurons.
        """
        a = cache["a"]
        L = len(self.W)

        dW = [np.zeros_like(Wl) for Wl in self.W]     # gradients for W
        db = [np.zeros_like(bl) for bl in self.b]     # gradients for b
        deltas = [np.zeros_like(bl) for bl in self.b] # delta per layer

        y = a[-1]  # output activation

        # Output delta (lecture-friendly with update W += lr*dW)
        if loss_type == "mse_sigmoid":
            delta = (t - y) * sigmoid_prime_from_output(y)   # output delta for MSE+sigmoid
        else:
            delta = (t - y)                                  # common simplification for CE+sigmoid in this convention

        deltas[L - 1] = delta

        # Last layer gradients
        dW[L - 1] = np.outer(delta, a[L - 1])                # delta * previous activation
        db[L - 1] = delta

        # Hidden layers backwards
        for l in range(L - 2, -1, -1):
            delta = (self.W[l + 1].T @ delta) * sigmoid_prime_from_output(a[l + 1])
            deltas[l] = delta
            dW[l] = np.outer(delta, a[l])
            db[l] = delta

        return dW, db, deltas

    def one_iteration_debug(
        self,
        x: np.ndarray,
        t: np.ndarray,
        lr: float,
        loss_type: LossType,
        show_updates: bool = True,
    ) -> "DebugMLP":
        """
        Perform exactly one iteration: forward -> backward -> update.
        Prints all intermediate values for exam-style debugging.
        """
        title_message("ONE ITERATION: FORWARD -> BACKWARD -> UPDATE", "magenta")

        # --- Forward ---
        title_message("FORWARD PASS", "blue")
        cache = self.forward(x)
        a, z = cache["a"], cache["z"]

        pretty_array("a[0] (input x)", a[0], "cyan")
        for l in range(len(z)):
            pretty_array(f"z[{l}] (pre-activation layer {l+1})", z[l], "cyan")
            pretty_array(f"a[{l+1}] (activation layer {l+1})", a[l + 1], "cyan")

        y = a[-1]
        E_before = compute_loss(loss_type, t, y)
        logger(f"Loss BEFORE update ({loss_type}) = {E_before:.10f}", "yellow")

        # --- Backward ---
        title_message("BACKWARD PASS (DELTAS + GRADIENTS)", "blue")
        dW, db, deltas = self.backward(cache, t, loss_type)

        for l in range(len(self.W)):
            pretty_array(f"delta[{l}] (layer {l+1})", deltas[l], "white")
            pretty_array(f"dW[{l}]", dW[l], "cyan")
            pretty_array(f"db[{l}]", db[l], "cyan")

        # --- Update ---
        title_message("WEIGHT UPDATES", "blue")
        new_W: List[np.ndarray] = []
        new_b: List[np.ndarray] = []

        for l in range(len(self.W)):
            delta_W = lr * dW[l]                             # ΔW
            delta_b = lr * db[l]                             # Δb
            if show_updates:
                pretty_array(f"ΔW[{l}] = lr*dW[{l}]", delta_W, "yellow")
                pretty_array(f"Δb[{l}] = lr*db[{l}]", delta_b, "yellow")

            new_W.append(self.W[l] + delta_W)                # update rule (lecture-style)
            new_b.append(self.b[l] + delta_b)

        new_model = DebugMLP(W=new_W, b=new_b)

        # --- Sanity check ---
        title_message("SANITY CHECK", "blue")
        new_y = new_model.forward(x)["a"][-1]
        E_after = compute_loss(loss_type, t, new_y)
        logger(f"Loss AFTER update  ({loss_type}) = {E_after:.10f}", "green" if E_after < E_before else "red")

        title_message("ITERATION COMPLETE", "green")
        return new_model


def load_config(path: str) -> Dict[str, Any]:
    """Load JSON configuration from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_and_build_model(cfg: Dict[str, Any]) -> Tuple[DebugMLP, np.ndarray, np.ndarray, float, LossType]:
    """
    Validate config and build model + numpy inputs.
    Raises clear errors for shape mismatches (very exam-useful).
    """
    required = ["x", "t", "lr", "loss_type", "W", "b"]
    for k in required:
        if k not in cfg:
            raise ValueError(f"Missing required key in config: '{k}'")

    x = np.array(cfg["x"], dtype=float).reshape(-1)          # input vector
    t = np.array(cfg["t"], dtype=float).reshape(-1)          # target vector
    lr = float(cfg["lr"])                                    # learning rate
    loss_type: LossType = cfg["loss_type"]

    W_list = [np.array(W, dtype=float) for W in cfg["W"]]    # weights
    b_list = [np.array(b, dtype=float).reshape(-1) for b in cfg["b"]]  # biases

    if len(W_list) != len(b_list):
        raise ValueError("Config error: len(W) must equal len(b).")

    # Shape validation
    prev_dim = x.shape[0]
    for l, (Wl, bl) in enumerate(zip(W_list, b_list)):
        if Wl.ndim != 2:
            raise ValueError(f"W[{l}] must be 2D, got shape {Wl.shape}")
        if bl.ndim != 1:
            raise ValueError(f"b[{l}] must be 1D, got shape {bl.shape}")
        if Wl.shape[1] != prev_dim:
            raise ValueError(f"Shape mismatch at layer {l}: W[{l}].shape[1]={Wl.shape[1]} must equal prev_dim={prev_dim}")
        if Wl.shape[0] != bl.shape[0]:
            raise ValueError(f"Shape mismatch at layer {l}: W[{l}].shape[0]={Wl.shape[0]} must equal b[{l}].shape[0]={bl.shape[0]}")
        prev_dim = Wl.shape[0]

    if prev_dim != t.shape[0]:
        raise ValueError(f"Target dimension mismatch: output_dim={prev_dim} but t has shape {t.shape}")

    return DebugMLP(W=W_list, b=b_list), x, t, lr, loss_type

def main() -> None:
    """
    One full BACKPROP iteration with an embedded example (lecture-style notation).

    Lecture mapping (keep this in mind while editing):
      - x            = y^(0)   (input layer outputs / input vector)
      - t            = t       (target / desired output)
      - lr           = η (eta) (learning rate)
      - W[0]         = w^(1)   (weights from layer 0 -> layer 1 : input -> hidden)
      - b[0]         = b^(1)   (biases of layer 1 : hidden)
      - W[1]         = w^(2)   (weights from layer 1 -> layer 2 : hidden -> output)
      - b[1]         = b^(2)   (biases of layer 2 : output)

    Internally during debug prints you will see:
      - z[l] = v^(l+1)  (net input / induced local field in lecture)
      - a[l] = y^(l)    (layer output after activation)
      - delta[l] = δ^(l+1) (layer delta / error signal)
    """
    title_message("USING EMBEDDED (LECTURE-NOTATION) CONFIG", "magenta")

    # -------------------- EDIT THESE VALUES FOR YOUR QUESTION --------------------
    EXAMPLE_CONFIG: Dict[str, Any] = {
        # y^(0) (input sample) = [x1, x2, ...]
        # Example: if question gives x1=1.0, x2=0.5 -> y^(0) = [1.0, 0.5]
        "x": [1.0, 0.5],

        # t (target) = desired output(s)
        # Example: if question gives target t=1 -> t = [1.0]
        "t": [1.0],

        # η (eta) = learning rate
        # Example: η = 0.5 -> lr = 0.5
        "lr": 0.5,

        # Loss/Output setting (choose as per lecture/question):
        # - "mse_sigmoid"           : MSE with sigmoid output (common backprop derivation)
        # - "cross_entropy_sigmoid" : Cross-Entropy with sigmoid output (if specified)
        "loss_type": "mse_sigmoid",

        # Weights w^(n): list of matrices (one per layer transition)
        #
        # Shape rule (lecture): w^(n) maps from y^(n-1) -> y^(n)
        # In code: W[layer] shape = (neurons_in_current_layer, neurons_in_previous_layer)
        "W": [
            # w^(1): INPUT -> HIDDEN
            # Example here: 2 inputs -> 2 hidden neurons, so w^(1) is 2x2.
            # Row 0 = weights into hidden neuron 1 from [x1, x2]
            # Row 1 = weights into hidden neuron 2 from [x1, x2]
            [[0.10, -0.20],
             [0.40,  0.30]],

            # w^(2): HIDDEN -> OUTPUT
            # Example here: 2 hidden neurons -> 1 output neuron, so w^(2) is 1x2.
            # Row 0 = weights into output neuron 1 from [hidden1, hidden2]
            [[0.20, -0.50]]
        ],

        # Biases b^(n): list of vectors (one per layer)
        #
        # Shape rule: b^(n) length = number of neurons in layer n
        "b": [
            # b^(1): biases for hidden layer neurons [b_h1, b_h2]
            [0.0, 0.0],

            # b^(2): bias for output neuron [b_o1]
            [0.0]
        ]
    }
    # ----------------------------------------------------------------------------

    try:
        model, x, t, lr, loss_type = validate_and_build_model(EXAMPLE_CONFIG)
        logger(f"Config loaded. η(lr)={lr}, loss_type={loss_type}", "green")

        # show_updates=True prints Δw^(n) and Δb^(n) clearly (best for exam practice)
        _ = model.one_iteration_debug(
            x=x,
            t=t,
            lr=lr,
            loss_type=loss_type,
            show_updates=True,
        )

    except Exception as e:
        logger(f"FAILED: {e}", "red")
        raise


if __name__ == "__main__":
    main()
