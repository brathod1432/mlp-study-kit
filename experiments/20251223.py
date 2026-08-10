#!/usr/bin/env python3.11



import sys, os, argparse, datetime, math
import numpy as np

import sys as _sys, os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from nn_core.logger import ObjLogger, title_message

logger = ObjLogger("Task_set_12")


# ======================================================================
# HW2 - Task: Activation functions + derivatives (LeakyReLU, ELU)
# plus forward + backprop demo network (NumPy only)
# ======================================================================
class Activation_fcn:
    """
    Activation functions utility (as in exercise-style code).

    API:
        output(v, name, derivative=False, alpha=None)

    Required:
        - Leaky ReLU (alpha default 0.01)
        - ELU       (alpha default 1.0)
        - Must be selectable by name: "leaky_relu", "elu"
    """

    def __init__(self):
        logger("Activation_fcn initialized\t(ready: relu, leaky_relu, elu, linear)", color="magenta")

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize activation name: lower + underscore->space."""
        return name.strip().lower().replace("_", " ")

    @staticmethod
    def _alpha(alpha, default: float) -> float:
        """Return alpha as float or default; raise ValueError if invalid."""
        if alpha is None:
            return float(default)
        try:
            return float(alpha)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid alpha\t(alpha must be numeric)\tgot={alpha}") from e

    # --- Existing simple activations (kept minimal) ---
    @staticmethod
    def relu(v: np.ndarray) -> np.ndarray:
        """ReLU: max(0, v)."""
        return np.maximum(0.0, v)

    @staticmethod
    def d_relu(v: np.ndarray) -> np.ndarray:
        """d(ReLU): 1 for v>=0 else 0 (course-style convention)."""
        return (v >= 0.0).astype(np.float64)

    @staticmethod
    def linear(v: np.ndarray) -> np.ndarray:
        """Linear: v."""
        return v

    @staticmethod
    def d_linear(v: np.ndarray) -> np.ndarray:
        """d(Linear): 1."""
        return np.ones_like(v, dtype=np.float64)

    # --- NEW: Leaky ReLU ---
    @staticmethod
    def leaky_relu(v: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        Leaky ReLU:
            f(v)=v            if v>=0
            f(v)=alpha*v      if v<0
        """
        return np.where(v >= 0.0, v, alpha * v)

    @staticmethod
    def d_leaky_relu(v: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        d(Leaky ReLU):
            f'(v)=1           if v>=0
            f'(v)=alpha       if v<0
        """
        return np.where(v >= 0.0, 1.0, alpha).astype(np.float64)

    # --- NEW: ELU ---
    @staticmethod
    def elu(v: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        ELU:
            f(v)=v                      if v>=0
            f(v)=alpha*(exp(v)-1)       if v<0
        """
        return np.where(v >= 0.0, v, alpha * (np.exp(v) - 1.0))

    @staticmethod
    def d_elu(v: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        d(ELU):
            f'(v)=1                     if v>=0
            f'(v)=alpha*exp(v)          if v<0
        """
        return np.where(v >= 0.0, 1.0, alpha * np.exp(v)).astype(np.float64)

    def output(self, v, name: str, derivative: bool = False, alpha=None) -> np.ndarray:
        """
        Compute activation output or derivative by name.

        Args:
            v: scalar/list/np.ndarray (converted to np.float64 array)
            name: "relu", "leaky_relu", "elu", "linear"
            derivative: if True return analytical derivative
            alpha: optional alpha (LeakyReLU/ELU); defaults apply if None
        """
        if not isinstance(name, str):
            raise TypeError(f"Activation name must be str\tgot={type(name).__name__}")

        v_arr = np.asarray(v, dtype=np.float64)
        key = self._normalize_name(name)

        if key == "relu":
            return self.d_relu(v_arr) if derivative else self.relu(v_arr)

        if key == "linear":
            return self.d_linear(v_arr) if derivative else self.linear(v_arr)

        if key == "leaky relu":
            a = self._alpha(alpha, default=0.01)
            return self.d_leaky_relu(v_arr, a) if derivative else self.leaky_relu(v_arr, a)

        if key == "elu":
            a = self._alpha(alpha, default=1.0)
            return self.d_elu(v_arr, a) if derivative else self.elu(v_arr, a)

        raise ValueError(f"Unknown activation name\t'{name}'\tAllowed: relu, leaky_relu, elu, linear")


# ------------------------------------------------
def init_layer(in_features: int, out_features: int, fcn: str, alpha=None, seed: int = 123) -> dict:
    """
    Create a layer dict similar to exercise-style code.

    Task 2: Glorot Normal (Xavier Normal) initialization:
        W ~ N(0, sigma^2),  sigma = sqrt(2 / (n_in + n_out))

    Task 3: Adagrad support:
        Add gradient accumulator G (same shape as weights) initialized to zeros.
        Store it under key 'gradient_accumulation' as per task description.

    Notes:
        - Keeps float64 everywhere for numeric consistency.
        - Adds 'weights' alias for compliance with task wording (points to same ndarray).
    """
    if not isinstance(in_features, int) or not isinstance(out_features, int):
        raise TypeError(
            f"init_layer expects ints\tin_features={type(in_features).__name__}\tout_features={type(out_features).__name__}"
        )  # validation
    if in_features <= 0 or out_features <= 0:
        raise ValueError(f"init_layer expects positive sizes\tin_features={in_features}\tout_features={out_features}")  # validation

    sigma = float(np.sqrt(2.0 / (in_features + out_features)))  # Glorot sigma

    rng = np.random.default_rng(seed)  # reproducible RNG
    w = rng.normal(0.0, sigma, size=(in_features, out_features)).astype(np.float64)  # weights
    b = np.zeros((1, out_features), dtype=np.float64)  # bias

    G_w = np.zeros_like(w, dtype=np.float64)  # Adagrad accumulator for weights
    G_b = np.zeros_like(b, dtype=np.float64)  # Adagrad accumulator for bias

    logger(f"InitLayer\tin={in_features}\tout={out_features}\tsigma={sigma:.6f}\tact={fcn}", color="cyan")

    return {
        "w": w,  # internal key used by forward/backprop
        "weights": w,  # alias key (task wording)
        "b": b,
        "fcn": fcn,
        "alpha": alpha,
        "gradient_accumulation": G_w,  # Adagrad accumulator for weights (task wording)
        "gradient_accumulation_b": G_b,  # Adagrad accumulator for bias (kept simple & consistent)
    }

def nn_forward(X: np.ndarray, layers: list, af: Activation_fcn) -> tuple:
    """
    Forward pass through all layers.

    Returns:
        V_list: list of pre-activations (v)
        Y_list: list of activations (y), includes input as Y_list[0]
    """
    V_list = []
    Y_list = [X]

    for idx, layer in enumerate(layers):
        v = (Y_list[-1] @ layer["w"]) + layer["b"]  # pre-activation
        y = af.output(v, name=layer["fcn"], derivative=False, alpha=layer.get("alpha", None))  # activation
        V_list.append(v)
        Y_list.append(y)

        logger(f"Forward\tLayer={idx}\tV.shape={v.shape}\tY.shape={y.shape}\tAct={layer['fcn']}", color="cyan")

    return V_list, Y_list


def nn_backward_update(X: np.ndarray, T: np.ndarray, layers: list, af: Activation_fcn, lr: float, eps: float = 1e-8) -> float:
    """
    Backpropagation + weight update (batch) using MSE loss.

    Task 3 change:
        Replace vanilla SGD update with Adagrad:
            g_t = dL/dw
            G_t = G_{t-1} + g_t^2
            w <- w - (lr / sqrt(G_t + eps)) * g_t

    Notes:
        - Uses in-place updates (w -= ...) to keep 'w' and 'weights' alias consistent.
        - Uses float64 everywhere for numeric consistency.
    """
    if eps <= 0:
        raise ValueError(f"eps must be > 0\teps={eps}")  # validation

    n = int(X.shape[0])  # batch size

    V_list, Y_list = nn_forward(X, layers, af)  # forward pass
    Y_out = Y_list[-1].astype(np.float64)  # output

    E = (Y_out - T).astype(np.float64)  # error
    loss = float(np.mean(E * E))  # MSE

    deltas = [None] * len(layers)  # delta per layer

    # Output layer delta  # comment
    last = len(layers) - 1
    d_act = af.output(
        V_list[last],
        name=layers[last]["fcn"],
        derivative=True,
        alpha=layers[last].get("alpha", None),
    )  # dY/dV
    deltas[last] = (2.0 / n) * E * d_act  # delta_L

    logger(f"Backprop\tLayer={last}\tDelta.shape={deltas[last].shape}", color="yellow")

    # Hidden deltas  # comment
    for i in range(last - 1, -1, -1):
        d_act_i = af.output(
            V_list[i],
            name=layers[i]["fcn"],
            derivative=True,
            alpha=layers[i].get("alpha", None),
        )  # dY/dV
        W_next = layers[i + 1].get("w", layers[i + 1]["weights"])  # safe access
        deltas[i] = (deltas[i + 1] @ W_next.T) * d_act_i  # chain rule
        logger(f"Backprop\tLayer={i}\tDelta.shape={deltas[i].shape}", color="yellow")

    # Adagrad update  # comment
    for i in range(len(layers)):
        W = layers[i].get("w", layers[i]["weights"])  # weights
        B = layers[i]["b"]  # bias

        dW = (Y_list[i].T @ deltas[i]).astype(np.float64)  # grad W
        dB = np.sum(deltas[i], axis=0, keepdims=True).astype(np.float64)  # grad b

        # --- Accumulate squared gradients (Adagrad) ---  # comment
        layers[i]["gradient_accumulation"] += dW * dW  # G_w = G_w + dW^2
        layers[i]["gradient_accumulation_b"] += dB * dB  # G_b = G_b + dB^2

        G_w = layers[i]["gradient_accumulation"]  # accumulator weights
        G_b = layers[i]["gradient_accumulation_b"]  # accumulator bias

        # --- Adaptive step (elementwise) ---  # comment
        W_step = (lr / (np.sqrt(G_w + eps))) * dW  # lr / sqrt(G + eps) * grad
        B_step = (lr / (np.sqrt(G_b + eps))) * dB  # lr / sqrt(G + eps) * grad

        # --- In-place update to preserve alias keys ---  # comment
        W -= W_step  # w <- w - step
        B -= B_step  # b <- b - step

        layers[i]["w"] = W  # keep key present
        layers[i]["weights"] = W  # keep alias consistent

        logger(f"Update(Adagrad)\tLayer={i}\tdW.shape={dW.shape}\tG_mean={float(np.mean(G_w)):.6e}", color="green")

    return loss


def train_mlp_mse(X: np.ndarray, T: np.ndarray, layers: list, epochs: int = 200, lr: float = 0.05) -> None:
    """
    Train MLP using MSE loss.
    """
    title_message("Training MLP\t(Forward + Backprop)", color="magenta")

    af = Activation_fcn()

    for ep in range(1, epochs + 1):
        loss = nn_backward_update(X, T, layers, af, lr=lr)

        if ep == 1 or ep % max(1, epochs // 10) == 0:
            logger(f"Epoch\t{ep}/{epochs}\tLoss\t{loss:.6f}", color="white")

    logger("Training complete.", color="green")

def split_train_test_ex10(X: np.ndarray, T: np.ndarray, test_ratio: float = 0.3, seed: int = 123) -> tuple:
    """
    Exercise-10 style train/test split (NumPy-only, no sklearn).

    Args:
        X: input matrix (N, d)
        T: targets      (N, k)
        test_ratio: fraction for test set
        seed: RNG seed

    Returns:
        X_train, T_train, X_test, T_test
    """
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in (0,1)\ttest_ratio={test_ratio}")
    if X.shape[0] != T.shape[0]:
        raise ValueError(f"X and T must have same N\tX.N={X.shape[0]}\tT.N={T.shape[0]}")

    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)

    n_test = int(round(test_ratio * X.shape[0]))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    X_train = X[train_idx].astype(np.float64)
    T_train = T[train_idx].astype(np.float64)
    X_test = X[test_idx].astype(np.float64)
    T_test = T[test_idx].astype(np.float64)

    logger(f"SplitEx10\tN={X.shape[0]}\ttrain={X_train.shape[0]}\ttest={X_test.shape[0]}", color="cyan")
    return X_train, T_train, X_test, T_test

def make_regression_data_ex10(n_samples: int = 200, seed: int = 123, noise_std: float = 0.10) -> tuple:
    """
    Exercise-10 Regression dataset (based on course description):
        y = sin(2x) + cos(x) + 5 + noise

    Returns:
        X: (N, 1)
        T: (N, 1)
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0\tn_samples={n_samples}")
    if noise_std < 0:
        raise ValueError(f"noise_std must be >= 0\tnoise_std={noise_std}")

    rng = np.random.default_rng(seed)

    X = np.linspace(-3.0, 3.0, n_samples, dtype=np.float64).reshape(-1, 1)
    noise = rng.normal(0.0, noise_std, size=(n_samples, 1)).astype(np.float64)
    T = (np.sin(2.0 * X) + np.cos(X) + 5.0 + noise).astype(np.float64)

    logger(f"DataEx10\tRegression\tX.shape={X.shape}\tT.shape={T.shape}\tnoise_std={noise_std}", color="cyan")
    return X, T

def make_classification_data_ex10(n_per_class: int = 200, seed: int = 123) -> tuple:
    """
    Exercise-10 Classification dataset (simple 2D two-class synthetic data).

    Produces two rectangular clusters:
        class 0: x in [0, 2], y in [0, 2]
        class 1: x in [1, 3], y in [2, 4]

    Returns:
        X: (2*n_per_class, 2)
        T: (2*n_per_class, 1) with labels in {0,1}
    """
    if n_per_class <= 0:
        raise ValueError(f"n_per_class must be > 0\tn_per_class={n_per_class}")

    rng = np.random.default_rng(seed)

    # Class 0  # comment
    x0 = rng.uniform(0.0, 2.0, size=(n_per_class, 1)).astype(np.float64)
    y0 = rng.uniform(0.0, 2.0, size=(n_per_class, 1)).astype(np.float64)
    X0 = np.hstack([x0, y0]).astype(np.float64)
    T0 = np.zeros((n_per_class, 1), dtype=np.float64)

    # Class 1  # comment
    x1 = rng.uniform(1.0, 3.0, size=(n_per_class, 1)).astype(np.float64)
    y1 = rng.uniform(2.0, 4.0, size=(n_per_class, 1)).astype(np.float64)
    X1 = np.hstack([x1, y1]).astype(np.float64)
    T1 = np.ones((n_per_class, 1), dtype=np.float64)

    X = np.vstack([X0, X1]).astype(np.float64)
    T = np.vstack([T0, T1]).astype(np.float64)

    logger(f"DataEx10\tClassification\tX.shape={X.shape}\tT.shape={T.shape}", color="cyan")
    return X, T

def init_layer_fixedstd_ex10(in_features: int, out_features: int, fcn: str, std: float = 0.5, alpha=None, seed: int = 123) -> dict:
    """
    Exercise-10 BASELINE initializer (pre-Task2 style):
        W ~ N(0, std)

    Used in Task 4 demos to compare against Glorot Normal init.
    """
    if in_features <= 0 or out_features <= 0:
        raise ValueError(f"init_layer_fixedstd_ex10 expects positive sizes\tin={in_features}\tout={out_features}")
    if std <= 0:
        raise ValueError(f"std must be > 0\tstd={std}")

    rng = np.random.default_rng(seed)

    w = rng.normal(0.0, std, size=(in_features, out_features)).astype(np.float64)
    b = np.zeros((1, out_features), dtype=np.float64)

    # Keep keys compatible with your current pipeline  # comment
    return {
        "w": w,
        "weights": w,
        "b": b,
        "fcn": fcn,
        "alpha": alpha,
        "gradient_accumulation": np.zeros_like(w, dtype=np.float64),       # exists but unused in SGD baseline
        "gradient_accumulation_b": np.zeros_like(b, dtype=np.float64),     # exists but unused in SGD baseline
    }

def nn_backward_update_sgd_ex10(X: np.ndarray, T: np.ndarray, layers: list, af: Activation_fcn, lr: float) -> float:
    """
    Exercise-10 BASELINE optimizer (pre-Task3 style):
        Vanilla SGD update:
            w <- w - lr * dW
            b <- b - lr * dB

    Used in Task 4 demos to compare against Adagrad.
    """
    n = int(X.shape[0])

    V_list, Y_list = nn_forward(X, layers, af)
    Y_out = Y_list[-1].astype(np.float64)

    E = (Y_out - T).astype(np.float64)
    loss = float(np.mean(E * E))

    deltas = [None] * len(layers)
    last = len(layers) - 1

    d_act = af.output(V_list[last], name=layers[last]["fcn"], derivative=True, alpha=layers[last].get("alpha", None))
    deltas[last] = (2.0 / n) * E * d_act

    for i in range(last - 1, -1, -1):
        d_act_i = af.output(V_list[i], name=layers[i]["fcn"], derivative=True, alpha=layers[i].get("alpha", None))
        W_next = layers[i + 1].get("w", layers[i + 1]["weights"])
        deltas[i] = (deltas[i + 1] @ W_next.T) * d_act_i

    for i in range(len(layers)):
        W = layers[i].get("w", layers[i]["weights"])
        B = layers[i]["b"]

        dW = (Y_list[i].T @ deltas[i]).astype(np.float64)
        dB = np.sum(deltas[i], axis=0, keepdims=True).astype(np.float64)

        W -= lr * dW
        B -= lr * dB

        layers[i]["w"] = W
        layers[i]["weights"] = W

    return loss

def train_mlp_ex10(
    X_train: np.ndarray,
    T_train: np.ndarray,
    layers: list,
    epochs: int,
    lr: float,
    update_fn,
    eps: float = 1e-8,
    log_every: int = 0,
) -> list:
    """
    Generic Exercise-10 training loop that returns loss history.

    Args:
        update_fn: function that performs one training step and returns loss.
            - For baseline SGD: nn_backward_update_sgd_ex10(X, T, layers, af, lr)
            - For Adagrad:      nn_backward_update(X, T, layers, af, lr, eps=...)
    """
    af = Activation_fcn()
    losses = []

    for ep in range(1, epochs + 1):
        if update_fn.__name__ == "nn_backward_update":
            loss = update_fn(X_train, T_train, layers, af, lr, eps=eps)  # Adagrad version signature
        else:
            loss = update_fn(X_train, T_train, layers, af, lr)           # SGD version signature

        losses.append(float(loss))

        if log_every > 0 and (ep == 1 or ep % log_every == 0):
            logger(f"TrainEx10\tEpoch\t{ep}/{epochs}\tLoss\t{loss:.6f}\tUpdate\t{update_fn.__name__}", color="yellow")

    return losses

def demo_task4_regression_ex10(
    epochs: int = 300,
    lr: float = 0.05,
    seed: int = 123,
    show_plot: bool = True,
) -> None:
    """
    Task 4 Demonstration (Exercise 10 - Regression):
        Compare baseline (SGD + fixed std init + ReLU) vs improved (Adagrad + Glorot + ELU/LeakyReLU).
    """
    title_message("Task4\tEx10\tRegression Demo", color="magenta")

    X, T = make_regression_data_ex10(n_samples=200, seed=seed, noise_std=0.10)
    X_train, T_train, X_test, T_test = split_train_test_ex10(X, T, test_ratio=0.3, seed=seed)

    # -------- Baseline: fixed std init + SGD + ReLU --------
    title_message("Ex10 Regression\tBaseline\t(SGD + fixed_init + ReLU)", color="cyan")
    layers_base = [
        init_layer_fixedstd_ex10(1, 10, fcn="relu", std=0.5, seed=10),
        init_layer_fixedstd_ex10(10, 1, fcn="linear", std=0.5, seed=20),
    ]
    loss_base = train_mlp_ex10(X_train, T_train, layers_base, epochs=epochs, lr=lr, update_fn=nn_backward_update_sgd_ex10, log_every=max(1, epochs // 10))

    # Evaluate baseline  # comment
    af = Activation_fcn()
    _, Yb_tr = nn_forward(X_train, layers_base, af)
    _, Yb_te = nn_forward(X_test, layers_base, af)
    mse_base_tr = float(np.mean((Yb_tr[-1] - T_train) ** 2))
    mse_base_te = float(np.mean((Yb_te[-1] - T_test) ** 2))
    logger(f"EvalEx10\tBaseline\tMSE_train={mse_base_tr:.6f}\tMSE_test={mse_base_te:.6f}", color="green")

    # -------- Improved: Glorot init + Adagrad + ELU --------
    title_message("Ex10 Regression\tImproved\t(Adagrad + Glorot + ELU)", color="cyan")
    layers_imp = [
        init_layer(1, 10, fcn="elu", alpha=1.0, seed=30),     # uses Glorot + Adagrad accum (your current init_layer)
        init_layer(10, 1, fcn="linear", alpha=None, seed=40),
    ]
    loss_imp = train_mlp_ex10(X_train, T_train, layers_imp, epochs=epochs, lr=lr, update_fn=nn_backward_update, eps=1e-8, log_every=max(1, epochs // 10))

    # Evaluate improved  # comment
    _, Yi_tr = nn_forward(X_train, layers_imp, af)
    _, Yi_te = nn_forward(X_test, layers_imp, af)
    mse_imp_tr = float(np.mean((Yi_tr[-1] - T_train) ** 2))
    mse_imp_te = float(np.mean((Yi_te[-1] - T_test) ** 2))
    logger(f"EvalEx10\tImproved\tMSE_train={mse_imp_tr:.6f}\tMSE_test={mse_imp_te:.6f}", color="green")

    # Plot (optional)  # comment
    if show_plot:
        import matplotlib
        import os as _mpl_os
        _mpl_backend = _mpl_os.environ.get("MPLBACKEND", "")
        if _mpl_backend:
            matplotlib.use(_mpl_backend)
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(loss_base, label="Baseline: SGD + fixed_init + ReLU")
        plt.plot(loss_imp, label="Improved: Adagrad + Glorot + ELU")
        plt.title("Ex10 Regression - Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Prediction plot  # comment
        X_sorted_idx = np.argsort(X_test[:, 0])
        Xs = X_test[X_sorted_idx]
        Ts = T_test[X_sorted_idx]

        _, Yp = nn_forward(Xs, layers_imp, af)
        _, Yp_b = nn_forward(Xs, layers_base, af)

        plt.figure()
        plt.scatter(X_train[:, 0], T_train[:, 0], label="Train samples")
        plt.scatter(X_test[:, 0], T_test[:, 0], label="Test samples")
        plt.plot(Xs[:, 0], Yp_b[-1][:, 0], label="Baseline prediction")
        plt.plot(Xs[:, 0], Yp[-1][:, 0], label="Improved prediction")
        plt.title("Ex10 Regression - Predictions")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.show()

def demo_task4_classification_ex10(
    epochs: int = 300,
    lr: float = 0.05,
    seed: int = 123,
    show_plot: bool = True,
) -> None:
    """
    Task 4 Demonstration (Exercise 10 - Classification):
        Compare baseline (SGD + fixed std init + ReLU) vs improved (Adagrad + Glorot + LeakyReLU/ELU).

    Note:
        - Training uses MSE to targets {0,1} with a linear output neuron.
        - For evaluation we convert scores to probabilities using sigmoid (outside the network)
          and threshold at 0.5.
    """
    title_message("Task4\tEx10\tClassification Demo", color="magenta")

    X, T = make_classification_data_ex10(n_per_class=200, seed=seed)
    X_train, T_train, X_test, T_test = split_train_test_ex10(X, T, test_ratio=0.3, seed=seed)

    def sigmoid_ex10(z: np.ndarray) -> np.ndarray:
        """Sigmoid helper for evaluation only."""
        return 1.0 / (1.0 + np.exp(-z))

    def accuracy_ex10(y_score: np.ndarray, t_true: np.ndarray) -> float:
        """Compute accuracy with sigmoid + 0.5 threshold."""
        p = sigmoid_ex10(y_score)
        y_hat = (p >= 0.5).astype(np.float64)
        return float(np.mean((y_hat == t_true).astype(np.float64)))

    # -------- Baseline: fixed std init + SGD + ReLU --------
    title_message("Ex10 Classification\tBaseline\t(SGD + fixed_init + ReLU)", color="cyan")
    layers_base = [
        init_layer_fixedstd_ex10(2, 10, fcn="relu", std=0.5, seed=10),
        init_layer_fixedstd_ex10(10, 1, fcn="linear", std=0.5, seed=20),
    ]
    loss_base = train_mlp_ex10(X_train, T_train, layers_base, epochs=epochs, lr=lr, update_fn=nn_backward_update_sgd_ex10, log_every=max(1, epochs // 10))

    af = Activation_fcn()
    _, Yb_tr = nn_forward(X_train, layers_base, af)
    _, Yb_te = nn_forward(X_test, layers_base, af)
    acc_base_tr = accuracy_ex10(Yb_tr[-1], T_train)
    acc_base_te = accuracy_ex10(Yb_te[-1], T_test)
    logger(f"EvalEx10\tBaseline\tAcc_train={acc_base_tr:.4f}\tAcc_test={acc_base_te:.4f}", color="green")

    # -------- Improved: Glorot init + Adagrad + LeakyReLU --------
    title_message("Ex10 Classification\tImproved\t(Adagrad + Glorot + LeakyReLU)", color="cyan")
    layers_imp = [
        init_layer(2, 10, fcn="leaky_relu", alpha=0.01, seed=30),
        init_layer(10, 1, fcn="linear", alpha=None, seed=40),
    ]
    loss_imp = train_mlp_ex10(X_train, T_train, layers_imp, epochs=epochs, lr=lr, update_fn=nn_backward_update, eps=1e-8, log_every=max(1, epochs // 10))

    _, Yi_tr = nn_forward(X_train, layers_imp, af)
    _, Yi_te = nn_forward(X_test, layers_imp, af)
    acc_imp_tr = accuracy_ex10(Yi_tr[-1], T_train)
    acc_imp_te = accuracy_ex10(Yi_te[-1], T_test)
    logger(f"EvalEx10\tImproved\tAcc_train={acc_imp_tr:.4f}\tAcc_test={acc_imp_te:.4f}", color="green")

    # Plot (optional)  # comment
    if show_plot:
        import matplotlib
        import os as _mpl_os
        _mpl_backend = _mpl_os.environ.get("MPLBACKEND", "")
        if _mpl_backend:
            matplotlib.use(_mpl_backend)
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(loss_base, label="Baseline: SGD + fixed_init + ReLU")
        plt.plot(loss_imp, label="Improved: Adagrad + Glorot + LeakyReLU")
        plt.title("Ex10 Classification - Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Scatter with predicted labels (improved)  # comment
        _, Y_pred = nn_forward(X_test, layers_imp, af)
        y_prob = sigmoid_ex10(Y_pred[-1])
        y_hat = (y_prob >= 0.5).astype(np.float64)

        plt.figure()
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_hat[:, 0])
        plt.title("Ex10 Classification - Predicted Classes (Improved)")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.grid(True)
        plt.show()





# ------------------------------
# Existing Homework 1 functions
# ------------------------------
def compute_cost(food: list, stock: dict, prices: dict, raiseException: bool = False):
    title_message("Starting Cost Calculation", color="magenta")

    if not isinstance(food, list) or not isinstance(stock, dict) or not isinstance(prices, dict):
        logger(f"ERROR:\tInvalid input types for compute_cost.\tExpected(list, dict, dict)\tGot({type(food).__name__},{type(stock).__name__},{type(prices).__name__})", color="red")
        if raiseException:
            raise TypeError("Input validation failed: compute_cost requires (list, dict, dict).")
        logger("raiseException=False\tNot raising error", color="red")
        return []

    total_cost = 0.0
    logger(f"Items to check:\t{food}", color="cyan")

    for item in food:
        if item in stock and item in prices:
            item_stock = stock[item]
            item_price = prices[item]
            if item_stock is None or item_price is None or item_stock < 0 or item_price < 0:
                logger(f"Skipping\t'{item}'\tInvalid stock/price\titem_stock={item_stock}\titem_price={item_price}", color="yellow")
                continue
            cost = float(item_price) * float(item_stock)
            total_cost += cost
            logger(f"Processing\t'{item}'\tStock={item_stock}\tPrice=${item_price:.2f}\tCost=${cost:.2f}", color="blue")
        else:
            logger(f"Skipping\t'{item}'\tNot found in stock/prices", color="red")

    logger(f"Total Cost:\t${total_cost:.2f}", color="green")
    title_message("Cost Calculation Complete\tReturning Total Cost", color="magenta")
    return total_cost


def prime_range(x, raise_exception: bool = False) -> list:
    if not isinstance(x, int) or x < 0:
        error_msg = f"Invalid input\tMust be non-negative int\tGot type={type(x).__name__}\tvalue={x}"
        if raise_exception:
            raise ValueError(f"Prime Range Error:\t{error_msg}")
        logger(f"Input validation failed\tReturning []\t{error_msg}", color="red")
        return []

    if x < 2:
        logger(f"Range upper limit\t{x}\tNo primes", color="cyan")
        return []

    title_message(f"Finding Primes up to\t{x}", color="cyan")

    primes = []
    for num in range(2, x + 1):
        is_prime = True
        limit = int(math.sqrt(num)) + 1
        for i in range(2, limit):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)

    logger(f"Found primes\tcount={len(primes)}\trange=[1..{x}]", color="green")
    logger(f"Primes List:\t{primes}", color="blue")
    return primes


# ------------------------------
# Main tests
# ------------------------------
if __name__ == "__main__":

    # -------------------------
    # HW1 Tests (unchanged flow)
    # -------------------------
    foods_to_buy = ["banana", "orange", "apple"]
    inventory_stock = {"banana": 4, "apple": 0, "orange": 9}
    unit_prices = {"banana": 3.0, "apple": 2.0, "orange": 2.5}

    try:
        title_message("Demonstration Run 1\tSuccessful Calculation", color="green")
        compute_cost(food=foods_to_buy, stock=inventory_stock, prices=unit_prices, raiseException=True)
    except Exception as e:
        logger(f"Unexpected error\tRun1\t{e}", color="red")

    try:
        title_message("Demonstration Run 2\tInput Validation (Expected Failure)", color="green")
        compute_cost(food="Not a list", stock=inventory_stock, prices=unit_prices, raiseException=True)
    except TypeError as e:
        logger(f"Successfully caught expected error\t{e}", color="green")
    except Exception as e:
        logger(f"Caught unexpected error\t{e}", color="red")

    logger("Scenario 1\tPrime up to 30", color="magenta")
    primes_up_to_30 = prime_range(x=30)
    logger(f"Final primes returned\t{primes_up_to_30}", color="green")

    # -------------------------
    # HW2 Tests: Forward + Backprop
    # -------------------------
    title_message("HW2\tForward + Backprop\t(LeakyReLU / ELU)", color="blue")

    # Shared dataset (simple regression): y = sin(x)
    X = np.linspace(-2.0, 2.0, 64, dtype=np.float64).reshape(-1, 1)
    T = np.sin(X).astype(np.float64)

    # -------- Scenario 1 (POSITIVE): Leaky ReLU hidden, Linear output --------
    try:
        title_message("Scenario 1\tPOSITIVE\tLeakyReLU + Backprop", color="green")

        layers_1 = [
            init_layer(in_features=1, out_features=10, fcn="leaky_relu", alpha=0.01, seed=10),
            init_layer(in_features=10, out_features=1, fcn="linear", alpha=None, seed=20),
        ]

        train_mlp_mse(X=X, T=T, layers=layers_1, epochs=200, lr=0.05)
        logger("Scenario 1 completed successfully.", color="green")
    except Exception as e:
        logger(f"Scenario 1 failed unexpectedly\t{e}", color="red")
        raise

    # -------- Scenario 2 (POSITIVE): ELU hidden, Linear output --------
    try:
        title_message("Scenario 2\tPOSITIVE\tELU + Backprop", color="green")

        layers_2 = [
            init_layer(in_features=1, out_features=10, fcn="elu", alpha=1.0, seed=30),
            init_layer(in_features=10, out_features=1, fcn="linear", alpha=None, seed=40),
        ]

        train_mlp_mse(X=X, T=T, layers=layers_2, epochs=200, lr=0.05)
        logger("Scenario 2 completed successfully.", color="green")
    except Exception as e:
        logger(f"Scenario 2 failed unexpectedly\t{e}", color="red")
        raise

    # -------- Scenario 3 (NEGATIVE): Invalid activation name --------
    try:
        title_message("Scenario 3\tNEGATIVE\tInvalid Activation Name", color="yellow")

        layers_bad = [
            init_layer(in_features=1, out_features=5, fcn="unknown_activation", alpha=None, seed=50),
            init_layer(in_features=5, out_features=1, fcn="linear", alpha=None, seed=60),
        ]

        _ = nn_backward_update(X, T, layers_bad, Activation_fcn(), lr=0.01)
        logger("ERROR\tScenario 3 unexpectedly succeeded (should fail).", color="red")
        raise RuntimeError("Scenario 3 unexpectedly succeeded: invalid activation name should have failed.")
    except ValueError as e:
        logger(f"Successfully caught expected ValueError\t{e}", color="green")
    except Exception as e:
        logger(f"Caught unexpected error type\t{e}", color="red")
        raise

    # ======================================================================
    # HW2 - Task 4: Demonstrate impact on Ex10 Regression + Classification
    # ======================================================================
    try:
        title_message("HW2\tTask4\tEx10 Demonstrations", color="blue")

        demo_task4_regression_ex10(epochs=300, lr=0.05, seed=123, show_plot=True)
        demo_task4_classification_ex10(epochs=300, lr=0.05, seed=123, show_plot=True)

        logger("HW2 Task4 completed successfully.", color="green")

    except Exception as e:
        logger(f"HW2 Task4 failed unexpectedly\t{e}", color="red")
        raise



    title_message("HW2 Demonstrations Complete!", color="blue")
