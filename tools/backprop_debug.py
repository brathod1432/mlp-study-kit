"""
Backprop (ONE ITERATION) — Task 2 (2-2-2 network, NO bias)

Matches the exact question:
- Inputs: x1=4, x2=4
- Targets: t1=2, t2=2
- Learning rate: η=0.5
- Sigmoid activation everywhere
- Loss: E = 1/2 * Σ (t_i - y_i)^2
- Weight update convention (common lecture style):
    w := w + η * delta * input
  where:
    delta_output = (t - y) * y*(1-y)
    delta_hidden = (Σ w_hidden_to_out * delta_out) * h*(1-h)

This script prints ALL forward values + deltas + gradients + updated weights.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from modules.GeneralUtils import ObjLogger, title_message

logger = ObjLogger("Test")

# -----------------------------
# Math functions
# -----------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime_from_output(y: np.ndarray) -> np.ndarray:
    # derivative using output: y*(1-y)
    return y * (1.0 - y)


def sse_half_loss(t: np.ndarray, y: np.ndarray) -> float:
    # E = 1/2 * sum (t-y)^2
    return float(0.5 * np.sum((t - y) ** 2))


# -----------------------------
# Network container (NO bias)
# Shapes:
#   w0: (H, D) = (2, 2)  input -> hidden
#   w1: (K, H) = (2, 2)  hidden -> output
# -----------------------------
@dataclass
class NetNoBias_2_2_2:
    w0: np.ndarray  # (2,2)
    w1: np.ndarray  # (2,2)

    def forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        # Hidden
        net_h = self.w0 @ x               # (2,)
        h = sigmoid(net_h)                # (2,)

        # Output
        net_o = self.w1 @ h               # (2,)
        y = sigmoid(net_o)                # (2,)

        return {
            "x": x,
            "net_h": net_h,
            "h": h,
            "net_o": net_o,
            "y": y,
        }

    def backprop_one_iteration(
        self,
        x: np.ndarray,
        t: np.ndarray,
        lr: float,
        verbose: bool = True,
    ) -> Tuple["NetNoBias_2_2_2", Dict[str, np.ndarray]]:

        title_message("BACKPROP — ONE ITERATION (NO BIAS)")

        # -------------------------
        # Forward pass
        # -------------------------
        cache = self.forward(x)
        net_h, h = cache["net_h"], cache["h"]
        net_o, y = cache["net_o"], cache["y"]

        E_before = sse_half_loss(t, y)

        if verbose:
            title_message("STEP 1 — FORWARD PASS")
            logger(f"x (inputs)           = {x}")
            logger(f"t (targets)          = {t}")
            logger(f"net_h (hidden nets)  = {net_h}   # [net_h1, net_h2]")
            logger(f"h (hidden outputs)   = {h}       # [h1, h2]")
            logger(f"net_o (out nets)     = {net_o}   # [net_o1, net_o2]")
            logger(f"y (final outputs)    = {y}       # [y1, y2]")
            logger(f"E_before (1/2 SSE)   = {E_before}")

        # -------------------------
        # Backward pass (deltas)
        # -------------------------
        # Output layer delta (lecture style):
        # delta_o = (t - y) * y*(1-y)
        delta_o = (t - y) * sigmoid_prime_from_output(y)  # (2,)

        # Hidden layer delta:
        # delta_h = (w1^T @ delta_o) * h*(1-h)
        delta_h = (self.w1.T @ delta_o) * sigmoid_prime_from_output(h)  # (2,)

        if verbose:
            title_message("STEP 2 — BACKWARD PASS (DELTAS)")
            logger(f"delta_o (output layer) = {delta_o}   # [δo1, δo2]")
            logger(f"delta_h (hidden layer) = {delta_h}   # [δh1, δh2]")

        # -------------------------
        # Gradients (as outer products)
        # Using update rule: w := w + lr * delta * input
        # So "dw" here is the raw delta*input (WITHOUT lr)
        # -------------------------
        dw1 = np.outer(delta_o, h)  # (2,2)  out <- hidden
        dw0 = np.outer(delta_h, x)  # (2,2)  hidden <- input

        if verbose:
            title_message("STEP 3 — GRADIENTS (RAW, BEFORE lr)")
            logger("dw1 (for w1 = hidden->output) = outer(delta_o, h)")
            logger(f"dw1 =\n{dw1}")
            logger("dw0 (for w0 = input->hidden)  = outer(delta_h, x)")
            logger(f"dw0 =\n{dw0}")

        # -------------------------
        # Weight updates
        # -------------------------
        w1_new = self.w1 + lr * dw1
        w0_new = self.w0 + lr * dw0
        new_net = NetNoBias_2_2_2(w0=w0_new, w1=w1_new)

        # Post-check: loss after update
        new_cache = new_net.forward(x)
        y_after = new_cache["y"]
        E_after = sse_half_loss(t, y_after)

        if verbose:
            title_message("STEP 4 — UPDATED WEIGHTS")
            logger("Updated w0 (input->hidden):")
            logger(f"{w0_new}")
            logger("Updated w1 (hidden->output):")
            logger(f"{w1_new}")

            title_message("STEP 5 — SANITY CHECK (LOSS AFTER UPDATE)")
            logger(f"y_after  = {y_after}")
            logger(f"E_after  = {E_after}")
            logger("NOTE: E_after should be smaller than E_before (usually).")

        details = {
            **cache,
            "E_before": np.array([E_before]),
            "delta_o": delta_o,
            "delta_h": delta_h,
            "dw1": dw1,
            "dw0": dw0,
            "w0_new": w0_new,
            "w1_new": w1_new,
            "y_after": y_after,
            "E_after": np.array([E_after]),
        }

        return new_net, details


# -----------------------------
# Exam-style mapping (matches diagram notation)
# -----------------------------
def build_net_from_question_weights() -> NetNoBias_2_2_2:
    """
    Diagram indices (same meaning as in your image):

    Layer 0 (input -> hidden):
      w^(0)_11 : x1 -> h1
      w^(0)_21 : x2 -> h1
      w^(0)_12 : x1 -> h2
      w^(0)_22 : x2 -> h2

    Layer 1 (hidden -> output):
      w^(1)_11 : h1 -> y1
      w^(1)_21 : h2 -> y1
      w^(1)_12 : h1 -> y2
      w^(1)_22 : h2 -> y2
    """

    # Given initial weights
    w0_11 = 0.9
    w0_12 = 0.6
    w0_21 = 0.4
    w0_22 = 0.6

    w1_11 = 0.4
    w1_12 = 0.5
    w1_21 = 0.2
    w1_22 = 0.7

    # Build matrices with exact mapping above:
    # w0 rows = hidden neurons [h1, h2], cols = inputs [x1, x2]
    w0 = np.array([
        [w0_11, w0_21],  # h1 gets x1*w0_11 + x2*w0_21
        [w0_12, w0_22],  # h2 gets x1*w0_12 + x2*w0_22
    ], dtype=float)

    # w1 rows = outputs [y1, y2], cols = hidden [h1, h2]
    w1 = np.array([
        [w1_11, w1_21],  # y1 gets h1*w1_11 + h2*w1_21
        [w1_12, w1_22],  # y2 gets h1*w1_12 + h2*w1_22
    ], dtype=float)

    return NetNoBias_2_2_2(w0=w0, w1=w1)


def print_weight_table_like_question(net_before: NetNoBias_2_2_2, net_after: NetNoBias_2_2_2) -> None:
    """
    Prints weights back in the SAME scalar names as the question.
    """

    w0b, w1b = net_before.w0, net_before.w1
    w0a, w1a = net_after.w0, net_after.w1

    # Unpack "before"
    w0_11_b, w0_21_b = w0b[0, 0], w0b[0, 1]
    w0_12_b, w0_22_b = w0b[1, 0], w0b[1, 1]

    w1_11_b, w1_21_b = w1b[0, 0], w1b[0, 1]
    w1_12_b, w1_22_b = w1b[1, 0], w1b[1, 1]

    # Unpack "after"
    w0_11_a, w0_21_a = w0a[0, 0], w0a[0, 1]
    w0_12_a, w0_22_a = w0a[1, 0], w0a[1, 1]

    w1_11_a, w1_21_a = w1a[0, 0], w1a[0, 1]
    w1_12_a, w1_22_a = w1a[1, 0], w1a[1, 1]

    title_message("FINAL UPDATED WEIGHTS (QUESTION NAMING)")

    logger("Layer 0 (Input -> Hidden)")
    logger(f"w^(0)_11 (x1->h1): {w0_11_b}  ->  {w0_11_a}")
    logger(f"w^(0)_12 (x1->h2): {w0_12_b}  ->  {w0_12_a}")
    logger(f"w^(0)_21 (x2->h1): {w0_21_b}  ->  {w0_21_a}")
    logger(f"w^(0)_22 (x2->h2): {w0_22_b}  ->  {w0_22_a}")

    logger("\nLayer 1 (Hidden -> Output)")
    logger(f"w^(1)_11 (h1->y1): {w1_11_b}  ->  {w1_11_a}")
    logger(f"w^(1)_12 (h1->y2): {w1_12_b}  ->  {w1_12_a}")
    logger(f"w^(1)_21 (h2->y1): {w1_21_b}  ->  {w1_21_a}")
    logger(f"w^(1)_22 (h2->y2): {w1_22_b}  ->  {w1_22_a}")


# -----------------------------
# Main: runs Task 2 exactly
# -----------------------------
def main() -> None:
    # Given x, t, lr (from the question)
    x = np.array([4.0, 4.0], dtype=float)   # [x1, x2]
    t = np.array([ 0], dtype=float)   # [t1, t2]
    lr = 0.5

    net = build_net_from_question_weights()

    # Run one iteration with full debug prints
    net2, details = net.backprop_one_iteration(x=x, t=t, lr=lr, verbose=True)

    # Print final weights using the same scalar names as in the diagram
    print_weight_table_like_question(net, net2)


if __name__ == "__main__":
    main()
