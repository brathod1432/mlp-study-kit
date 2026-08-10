#!/usr/bin/env python3.11

import sys
import os
import datetime
import math
import numpy as np
# Resolve nn_core from project root (works with or without pip install -e .)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from nn_core.logger import ObjLogger as ObjLogger2

import torch
try:
    import torch_directml
    DEVICE = torch_directml.device()          # AMD GPU via DirectML
    BACKEND = "DirectML (AMD GPU)"
except Exception:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BACKEND = "CUDA" if DEVICE.type == "cuda" else "CPU"

print(f"[Device] {BACKEND} -> {DEVICE}")

class ObjLogger:
    ANSI_COLORS = {
        "blue": "\033[34m",
        "cyan": "\033[36m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "green": "\033[32m",
        "magenta": "\033[35m",
        "white": "\033[37m",
        "reset": "\033[0m",
    }

    def __init__(self, name: str = "Logger"):
        self.name = name

    def __call__(self, message: str, color: str = "white"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = f"{timestamp}\t\t[{self.name}]\t"
        color_code = self.ANSI_COLORS.get(color.lower(), self.ANSI_COLORS["white"])
        reset_code = self.ANSI_COLORS["reset"]
        log_message = f"{prefix}{color_code}{message}{reset_code}"
        print(log_message)

logger = ObjLogger2("Task_set_12")

def title_message(msg, color="blue"):
    """
    Prints a formatted title box using logger.
    """
    border = "#" * (len(msg) + 10)
    logger(border, color=color)
    logger(f"#\t{msg}\t#", color=color)
    logger(border, color=color)

# ==================================================================
# HW2 - Task 1: Activation functions + derivatives (LeakyReLU, ELU)
# =========================================================================

class Activation_fcn:
    """
    Activation functions

    API:
        output(v, name, derivative=False, alpha=None)

    Required:
        - Leaky ReLU (alpha default 0.01)
        - ELU       (alpha default 1.0)
        - Must be by name: "leaky_relu", "elu"
    """

    def __init__(self):
        logger("Activation_fcn initialized\t(ready: relu, leaky_relu, elu, linear)", color="magenta")

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize activation name: lower + underscore->space."""
        return name.strip().lower().replace("_", " ")

    @staticmethod
    def _alpha(alpha, default: float) -> float:
        """
        Return alpha as float ;
        raise ValueError if invalid"""
        if alpha is None:
            return float(default)
        try:
            return float(alpha)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid alpha\t(alpha must be numeric)\tgot={alpha}") from e

    @staticmethod
    def relu(v: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, v)

    @staticmethod
    def d_relu(v: np.ndarray) -> np.ndarray:
        """d(ReLU): 1 for v>=0 else 0"""
        return (v >= 0.0).astype(np.float64)

    @staticmethod
    def linear(v: np.ndarray) -> np.ndarray:
        """Linear: v."""
        return v

    @staticmethod
    def d_linear(v: np.ndarray) -> np.ndarray:
        """d(Linear): 1."""
        return np.ones_like(v, dtype=np.float64)

    @staticmethod
    def leaky_relu(v: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        Leaky ReLU:
            f(v)=v    if v>=0
            f(v)=alpha*v   if v<0
        """
        return np.where(v >= 0.0, v, alpha * v)

    @staticmethod
    def d_leaky_relu(v: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        d(Leaky ReLU):
            f'(v)=1   if v>=0
            f'(v)=alpha   if v<0
        """
        return np.where(v >= 0.0, 1.0, alpha).astype(np.float64)

    @staticmethod
    def elu(v: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        ELU:
            f(v)=v      if v>=0
            f(v)=alpha*(exp(v)-1)   if v<0
        """
        return np.where(v >= 0.0, v, alpha * (np.exp(v) - 1.0))

    @staticmethod
    def d_elu(v: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        d(ELU):
            f'(v)=1     if v>=0
            f'(v)=alpha*exp(v)   if v<0
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


# ==========================================================
# HW2 == Core: init/forward/backprop/train
# =================================
class NeuralNetCore:
    """
    Refactoring container for HW2 NN functions.
    """
    @staticmethod
    def init_layer(in_features: int, out_features: int, fcn: str, alpha=None, seed: int = 123) -> dict:
        """
        Create a layer dict similar to exercise-style code.

        Task 2: Glorot Normal (Xavier Normal) initialization:
            W ~ N(0, sigma^2),  sigma = sqrt(2 / (n_in + n_out))

        Task 3: Adagrad support:
            Add gradient accumulator G (same shape as weights) initialized to zeros.

        To-Do: Need to fix the Shaping error for input * weights (2025-12-20)
        """
        if not isinstance(in_features, int) or not isinstance(out_features, int):
            raise TypeError(
                f"init_layer expects ints\tin_features={type(in_features).__name__}\tout_features={type(out_features).__name__}"
            )
        if in_features <= 0 or out_features <= 0:
            raise ValueError(f"init_layer expects positive sizes\tin_features={in_features}\tout_features={out_features}")

        sigma = float(np.sqrt(2.0 / (in_features + out_features)))  # Glorot sigma == Task 2
        rng = np.random.default_rng(seed)  # reproducible rng

        w = rng.normal(0.0, sigma, size=(in_features, out_features)).astype(np.float64)  # weights
        b = np.zeros((1, out_features), dtype=np.float64)  # bias

        G_w = np.zeros_like(w, dtype=np.float64)  # Adagrad for weights
        G_b = np.zeros_like(b, dtype=np.float64)  # Adagrad for bias

        logger(f"InitLayer\tin={in_features}\tout={out_features}\tsigma={sigma:.6f}\tact={fcn}", color="cyan")
        return {
            "w": w,  # internal key used by forward/backprop
            "weights": w,  # alias key (task wording)
            "b": b,
            "fcn": fcn,
            "alpha": alpha,
            "gradient_accumulation": G_w,  # Adagrad for weights (task wording)
            "gradient_accumulation_b": G_b,  # Adagrad for bias (kept simple & consistent)
        }

    @staticmethod
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
            logger(f"Forward\tLayer={idx}\tV.shape={v.shape}\tY.shape={y.shape}\tAct={layer['fcn']}")
        return V_list, Y_list

    @staticmethod
    def nn_backward_update(X: np.ndarray, T: np.ndarray, layers: list, af: Activation_fcn, lr: float, eps: float = 1e-8) -> float:
        """
        Backpropagation + weight update (batch) using MSE loss.
        Task 3 == change: Adagrad update rule.
        """
        if eps <= 0:
            raise ValueError(f"eps must be > 0\teps={eps}")

        n = int(X.shape[0])  # batch size
        V_list, Y_list = NeuralNetCore.nn_forward(X, layers, af)  # forward pass
        Y_out = Y_list[-1].astype(np.float64)

        E = (Y_out - T).astype(np.float64)  # error
        loss = float(np.mean(E * E))  # MSE

        deltas = [None] * len(layers)  # delta per layer

        # Output layer delta
        last = len(layers) - 1
        d_act = af.output(
            V_list[last],
            name=layers[last]["fcn"],
            derivative=True,
            alpha=layers[last].get("alpha", None),
        )
        deltas[last] = (2.0 / n) * E * d_act  # delta_L
        logger(f"Backprop\tLayer={last}\tDelta.shape={deltas[last].shape}", color="yellow")

        # Hidden del
        for i in range(last - 1, -1, -1):
            d_act_i = af.output(
                V_list[i],
                name=layers[i]["fcn"],
                derivative=True,
                alpha=layers[i].get("alpha", None),
            )
            W_next = layers[i + 1].get("w", layers[i + 1]["weights"])
            deltas[i] = (deltas[i + 1] @ W_next.T) * d_act_i
            logger(f"Backprop\tLayer={i}\tDelta.shape={deltas[i].shape}", color="yellow")

        # Adagrad update
        for i in range(len(layers)):
            W = layers[i].get("w", layers[i]["weights"])  # weights
            B = layers[i]["b"]  # bias

            dW = (Y_list[i].T @ deltas[i]).astype(np.float64)
            dB = np.sum(deltas[i], axis=0, keepdims=True).astype(np.float64)

            layers[i]["gradient_accumulation"] += dW * dW  # G_w = G_w + dW^2
            layers[i]["gradient_accumulation_b"] += dB * dB  # G_b = G_b + dB^2

            G_w = layers[i]["gradient_accumulation"]
            G_b = layers[i]["gradient_accumulation_b"]

            W_step = (lr / (np.sqrt(G_w + eps))) * dW
            B_step = (lr / (np.sqrt(G_b + eps))) * dB

            W -= W_step
            B -= B_step

            layers[i]["w"] = W
            layers[i]["weights"] = W

            logger(f"Update(Adagrad)\tLayer={i}\tdW.shape={dW.shape}\tG_mean={float(np.mean(G_w)):.6e}", color="green")
        return loss

    @staticmethod
    def train_mlp_mse(X: np.ndarray, T: np.ndarray, layers: list, epochs: int = 200, lr: float = 0.05) -> None:
        """
        Train MLP using MSE loss.
        """
        title_message("Training MLP\t(Forward + Backprop)", color="magenta")
        af = Activation_fcn()
        for ep in range(1, epochs + 1):
            loss = NeuralNetCore.nn_backward_update(X, T, layers, af, lr=lr)

            if ep == 1 or ep % max(1, epochs // 10) == 0:
                logger(f"Epoch\t{ep}/{epochs}\tLoss\t{loss:.6f}", color="white")

        logger("Training complete.")

    @staticmethod
    def split_train_test_ex10(X: np.ndarray, T: np.ndarray, test_ratio: float = 0.3, seed: int = 123) -> tuple:
        """
        for backward compatibility if previously called it from core.
        Exercise10Data.split_train_test_ex10.
        """
        return Exercise10Data.split_train_test_ex10(X, T, test_ratio=test_ratio, seed=seed)

    @staticmethod
    def make_regression_data_ex10(n_samples: int = 200, seed: int = 123, noise_std: float = 0.10) -> tuple:
        """
        for backward compatibility if previously called from core.
        Exercise10Data.make_regression_data_ex10.
        """
        return Exercise10Data.make_regression_data_ex10(n_samples=n_samples, seed=seed, noise_std=noise_std)

    @staticmethod
    def make_classification_data_ex10(n_per_class: int = 200, seed: int = 123) -> tuple:
        """
        for backward compatibility if previously called from core.
        Exercise10Data.make_classification_data_ex10 for clarity.
        """
        return Exercise10Data.make_classification_data_ex10(n_per_class=n_per_class, seed=seed)

    @staticmethod
    def init_layer_fixedstd_ex10(in_features: int, out_features: int, fcn: str, std: float = 0.5, alpha=None, seed: int = 123) -> dict:
        """
        Exercise-10 BASELINE initializer:
            W ~ N(0, std)
        """
        if in_features <= 0 or out_features <= 0:
            raise ValueError(f"init_layer_fixedstd_ex10 expects positive sizes\tin={in_features}\tout={out_features}")
        if std <= 0:
            raise ValueError(f"std must be > 0\tstd={std}")

        rng = np.random.default_rng(seed)
        w = rng.normal(0.0, std, size=(in_features, out_features)).astype(np.float64)
        b = np.zeros((1, out_features), dtype=np.float64)

        return {
            "w": w,
            "weights": w,
            "b": b,
            "fcn": fcn,
            "alpha": alpha,
            "gradient_accumulation": np.zeros_like(w, dtype=np.float64),
            "gradient_accumulation_b": np.zeros_like(b, dtype=np.float64),
        }

    @staticmethod
    def nn_backward_update_sgd_ex10(X: np.ndarray, T: np.ndarray, layers: list, af: Activation_fcn, lr: float) -> float:
        """
        Exercise-10 BASELINE optimizer
        """
        n = int(X.shape[0])

        V_list, Y_list = NeuralNetCore.nn_forward(X, layers, af)
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

    @staticmethod
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
        Exercise-10 training loop that returns loss history.
        """
        af = Activation_fcn()
        losses = []

        for ep in range(1, epochs + 1):
            if update_fn.__name__ == "nn_backward_update":
                loss = update_fn(X_train, T_train, layers, af, lr, eps=eps)
            else:
                loss = update_fn(X_train, T_train, layers, af, lr)

            losses.append(float(loss))

            if log_every > 0 and (ep == 1 or ep % log_every == 0):
                logger(f"TrainEx10\tEpoch\t{ep}/{epochs}\tLoss\t{loss:.6f}\tUpdate\t{update_fn.__name__}", color="yellow")

        return losses


# ======================================
# Exercise 10 datasets + split
# ============================

class Exercise10Data:
    """
    Refactoring container for Exercise-10 dataset helpers
    """
    @staticmethod
    def split_train_test_ex10(X: np.ndarray, T: np.ndarray, test_ratio: float = 0.3, seed: int = 123) -> tuple:
        """
        Exercise-10 train/test split (NumPy-only).
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

    @staticmethod
    def make_regression_data_ex10(n_samples: int = 200, seed: int = 123, noise_std: float = 0.10) -> tuple:
        """
        Exercise-10 Regression dataset:
            y = sin(2x) + cos(x) + 5 + noise
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

    @staticmethod
    def make_classification_data_ex10(n_per_class: int = 200, seed: int = 123) -> tuple:
        """
        Exercise-10 Classification dataset (simple 2D two-class synthetic data).
        """
        if n_per_class <= 0:
            raise ValueError(f"n_per_class must be > 0\tn_per_class={n_per_class}")

        rng = np.random.default_rng(seed)

        x0 = rng.uniform(0.0, 2.0, size=(n_per_class, 1)).astype(np.float64)
        y0 = rng.uniform(0.0, 2.0, size=(n_per_class, 1)).astype(np.float64)
        X0 = np.hstack([x0, y0]).astype(np.float64)
        T0 = np.zeros((n_per_class, 1), dtype=np.float64)

        x1 = rng.uniform(1.0, 3.0, size=(n_per_class, 1)).astype(np.float64)
        y1 = rng.uniform(2.0, 4.0, size=(n_per_class, 1)).astype(np.float64)
        X1 = np.hstack([x1, y1]).astype(np.float64)
        T1 = np.ones((n_per_class, 1), dtype=np.float64)

        X = np.vstack([X0, X1]).astype(np.float64)
        T = np.vstack([T0, T1]).astype(np.float64)

        logger(f"DataEx10\tClassification\tX.shape={X.shape}\tT.shape={T.shape}", color="cyan")
        return X, T

    @staticmethod
    def make_regression_data_from_hw1_compute_cost(
        n_samples: int = 300,
        food_items: list = None,
        max_stock: int = 10,
        price_low: float = 0.5,
        price_high: float = 5.0,
        seed: int = 123,
    ) -> tuple:
        """
        Build a regression dataset using HW1 compute_cost() as the target function.

        X features per sample:
            [stock(item1..k), price(item1..k)]  -> shape (N, 2k)
        T target per sample:
            total_cost -> shape (N, 1)
        """
        if food_items is None:
            food_items = ["banana", "orange", "apple"]
        if not isinstance(food_items, list) or len(food_items) == 0:
            raise ValueError(f"food_items must be a non-empty list\tfood_items={food_items}")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be > 0\tn_samples={n_samples}")
        if max_stock < 0:
            raise ValueError(f"max_stock must be >= 0\tmax_stock={max_stock}")
        if price_low <= 0 or price_high <= 0 or price_low >= price_high:
            raise ValueError(f"invalid price range\tprice_low={price_low}\tprice_high={price_high}")

        rng = np.random.default_rng(seed)
        k = len(food_items)

        X = np.zeros((n_samples, 2 * k), dtype=np.float64)
        T = np.zeros((n_samples, 1), dtype=np.float64)

        for i in range(n_samples):
            stock = {item: int(rng.integers(0, max_stock + 1)) for item in food_items}
            prices = {item: float(rng.uniform(price_low, price_high)) for item in food_items}

            stocks_vec = np.array([stock[item] for item in food_items], dtype=np.float64)
            prices_vec = np.array([prices[item] for item in food_items], dtype=np.float64)

            X[i, :k] = stocks_vec
            X[i, k:] = prices_vec

            total = Homework1Utils.compute_cost(food=food_items, stock=stock, prices=prices, raiseException=True)
            T[i, 0] = float(total)

        logger(f"DataHW1\tcompute_cost\tX.shape={X.shape}\tT.shape={T.shape}\titems={food_items}", color="cyan")
        return X, T


# ==============================================
# HW2 Task4 demos
# ==========================================
class Homework2Demos:
    """
    Refactoring container for HW2 demo functions.
    """
    @staticmethod
    def demo_task4_regression_ex10(
        epochs: int = 300,
        lr: float = 0.05,
        seed: int = 123,
        show_plot: bool = True,
        use_hw1_compute_cost: bool = False,
    ) -> None:
        """
        Task 4 Demonstration: Ex10 Regression.
        """
        title_message("Task4\tRegression Demo", color="magenta")
        if use_hw1_compute_cost:
            X, T = Exercise10Data.make_regression_data_from_hw1_compute_cost(
                n_samples=300,
                food_items=["banana", "orange", "apple"],
                max_stock=10,
                price_low=0.5,
                price_high=5.0,
                seed=seed,
            )
        else:
            X, T = Exercise10Data.make_regression_data_ex10(n_samples=200, seed=seed, noise_std=0.10)

        X_train, T_train, X_test, T_test = Exercise10Data.split_train_test_ex10(X, T, test_ratio=0.3, seed=seed)

        title_message("Regression\tBaseline\t(SGD + fixed_init + ReLU)", color="cyan")
        in_dim = int(X_train.shape[1])
        layers_base = [
            NeuralNetCore.init_layer_fixedstd_ex10(in_dim, 10, fcn="relu", std=0.5, seed=10),
            NeuralNetCore.init_layer_fixedstd_ex10(10, 1, fcn="linear", std=0.5, seed=20),
        ]
        loss_base = NeuralNetCore.train_mlp_ex10(
            X_train, T_train, layers_base,
            epochs=epochs, lr=lr,
            update_fn=NeuralNetCore.nn_backward_update_sgd_ex10,
            log_every=max(1, epochs // 10),
        )

        af = Activation_fcn()
        _, Yb_tr = NeuralNetCore.nn_forward(X_train, layers_base, af)
        _, Yb_te = NeuralNetCore.nn_forward(X_test, layers_base, af)
        mse_base_tr = float(np.mean((Yb_tr[-1] - T_train) ** 2))
        mse_base_te = float(np.mean((Yb_te[-1] - T_test) ** 2))
        logger(f"Eval\tBaseline\tMSE_train={mse_base_tr:.6f}\tMSE_test={mse_base_te:.6f}", color="green")

        # Improved: To-Do//done: Glorot init + Adagrad + ELU
        title_message("Regression\tImproved\t(Adagrad + Glorot + ELU)", color="cyan")
        layers_imp = [
            NeuralNetCore.init_layer(in_dim, 10, fcn="elu", alpha=1.0, seed=30),
            NeuralNetCore.init_layer(10, 1, fcn="linear", alpha=None, seed=40),
        ]
        loss_imp = NeuralNetCore.train_mlp_ex10(
            X_train, T_train, layers_imp,
            epochs=epochs, lr=lr,
            update_fn=NeuralNetCore.nn_backward_update,
            eps=1e-8,
            log_every=max(1, epochs // 10),
        )

        _, Yi_tr = NeuralNetCore.nn_forward(X_train, layers_imp, af)
        _, Yi_te = NeuralNetCore.nn_forward(X_test, layers_imp, af)
        mse_imp_tr = float(np.mean((Yi_tr[-1] - T_train) ** 2))
        mse_imp_te = float(np.mean((Yi_te[-1] - T_test) ** 2))
        logger(f"Eval\tImproved\tMSE_train={mse_imp_tr:.6f}\tMSE_test={mse_imp_te:.6f}", color="green")

        if show_plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(loss_base, label="Baseline: SGD + fixed_init + ReLU")
            plt.plot(loss_imp, label="Improved: Adagrad + Glorot + ELU")
            plt.title("Regression - Loss Curves")
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.legend()
            plt.grid(True)
            plt.show()

    @staticmethod
    def demo_task4_classification_ex10(
        epochs: int = 300,
        lr: float = 0.05,
        seed: int = 123,
        show_plot: bool = True,
    ) -> None:
        """
        Task 4 Demonstration (Exercise 10 - Classification).
        """
        title_message("Task4\tEx10\tClassification Demo", color="magenta")
        X, T = Exercise10Data.make_classification_data_ex10(n_per_class=200, seed=seed)
        X_train, T_train, X_test, T_test = Exercise10Data.split_train_test_ex10(X, T, test_ratio=0.3, seed=seed)

        def sigmoid_ex10(z: np.ndarray) -> np.ndarray:
            """Sigmoid helper == evaluation only."""
            return 1.0 / (1.0 + np.exp(-z))

        def accuracy_ex10(y_score: np.ndarray, t_true: np.ndarray) -> float:
            """Compute accuracy == sigmoid + 0.5 threshold."""
            p = sigmoid_ex10(y_score)
            y_hat = (p >= 0.5).astype(np.float64)
            return float(np.mean((y_hat == t_true).astype(np.float64)))

        # Baseline
        title_message("Ex10 Classification\tBaseline\t(SGD + fixed_init + ReLU)", color="cyan")
        layers_base = [
            NeuralNetCore.init_layer_fixedstd_ex10(2, 10, fcn="relu", std=0.5, seed=10),
            NeuralNetCore.init_layer_fixedstd_ex10(10, 1, fcn="linear", std=0.5, seed=20),
        ]
        loss_base = NeuralNetCore.train_mlp_ex10(
            X_train, T_train, layers_base, epochs=epochs, lr=lr,
            update_fn=NeuralNetCore.nn_backward_update_sgd_ex10,
            log_every=max(1, epochs // 10),
        )

        af = Activation_fcn()
        _, Yb_tr = NeuralNetCore.nn_forward(X_train, layers_base, af)
        _, Yb_te = NeuralNetCore.nn_forward(X_test, layers_base, af)
        acc_base_tr = accuracy_ex10(Yb_tr[-1], T_train)
        acc_base_te = accuracy_ex10(Yb_te[-1], T_test)
        logger(f"EvalEx10\tBaseline\tAcc_train={acc_base_tr:.4f}\tAcc_test={acc_base_te:.4f}", color="green")

        # Improved
        title_message("Ex10 Classification\tImproved\t(Adagrad + Glorot + LeakyReLU)", color="cyan")
        layers_imp = [
            NeuralNetCore.init_layer(2, 10, fcn="leaky_relu", alpha=0.01, seed=30),
            NeuralNetCore.init_layer(10, 1, fcn="linear", alpha=None, seed=40),
        ]
        loss_imp = NeuralNetCore.train_mlp_ex10(
            X_train, T_train, layers_imp, epochs=epochs, lr=lr,
            update_fn=NeuralNetCore.nn_backward_update,
            eps=1e-8,
            log_every=max(1, epochs // 10),
        )

        _, Yi_tr = NeuralNetCore.nn_forward(X_train, layers_imp, af)
        _, Yi_te = NeuralNetCore.nn_forward(X_test, layers_imp, af)
        acc_imp_tr = accuracy_ex10(Yi_tr[-1], T_train)
        acc_imp_te = accuracy_ex10(Yi_te[-1], T_test)
        logger(f"EvalEx10\tImproved\tAcc_train={acc_imp_tr:.4f}\tAcc_test={acc_imp_te:.4f}", color="green")

        if show_plot:
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

            _, Y_pred = NeuralNetCore.nn_forward(X_test, layers_imp, af)
            y_prob = sigmoid_ex10(Y_pred[-1])
            y_hat = (y_prob >= 0.5).astype(np.float64)

            plt.figure()
            plt.scatter(X_test[:, 0], X_test[:, 1], c=y_hat[:, 0])
            plt.title("Ex10 Classification - Predicted Classes (Improved)")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.grid(True)
            plt.show()

class Homework1Utils:
    """
    Container claass for HW1 functions.
    Function names preserved.
    """
    @staticmethod
    def compute_cost(food: list, stock: dict, prices: dict, raiseException: bool = False):
        title_message("Starting Cost Calculation", color="magenta")

        if not isinstance(food, list) or not isinstance(stock, dict) or not isinstance(prices, dict):
            logger(f"ERROR:\tInvalid input types for compute_cost.\tExpected(list, dict, dict)\tGot({type(food).__name__},{type(stock).__name__},{type(prices).__name__})",
                color="red")
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

    @staticmethod
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

class Homework3VGG16:
    """
    Homework 3:
        Task 1: Manual VGG16 architecture (Keras) and (Torch optional earlier)
        Task 2: Train VGG16 (Keras) on Cats vs Dogs (TFDS), metrics + plots + qualitative results
    """

    # ============================================================
    # TASK 1 (Keras): Manual VGG16 builder (no pre-built loaders)
    # ============================================================

    @staticmethod
    def _vgg_conv_block_keras(
        x,
        filters: int,
        conv_count: int,
        block_name: str,
        kernel_size: tuple = (3, 3),
        activation: str = "relu",
        padding: str = "same",
        kernel_initializer: str = "glorot_uniform",
    ):
        """
        Creates VGG-style block:
            (Conv2D + ReLU) repeated conv_count times, then MaxPool2D.
        """
        try:
            from tensorflow.keras import layers

            for i in range(conv_count):
                x = layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding=padding,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    bias_initializer="zeros",
                    name=f"{block_name}_conv{i+1}",
                )(x)

            x = layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name=f"{block_name}_pool",
            )(x)

            return x

        except Exception as e:
            logger(f"VGG block creation failed\tblock={block_name}\t{e}", color="red")
            raise

    @staticmethod
    def build_vgg16_keras(
        input_shape: tuple = (224, 224, 3),
        num_classes: int = 1000,
        include_top: bool = True,
        dropout_rate: float = 0.5,
        kernel_initializer: str = "glorot_uniform",
        model_name: str = "VGG16_Manual",
    ):
        """
        Manual VGG16 (Keras) per original design:
            Block1: 64,64 + pool
            Block2: 128,128 + pool
            Block3: 256,256,256 + pool
            Block4: 512,512,512 + pool
            Block5: 512,512,512 + pool
            Top: Flatten -> 4096 -> 4096 -> num_classes(softmax)
        """
        title_message("HW3\tTask1\tBuilding VGG16 (Keras) Manually", color="blue")

        try:
            from tensorflow.keras import layers, models

            if not (isinstance(input_shape, tuple) and len(input_shape) == 3):
                raise ValueError(f"Invalid input_shape\tExpected (H,W,C)\tGot={input_shape}")
            if not isinstance(num_classes, int) or num_classes <= 1:
                raise ValueError(f"Invalid num_classes\tExpected int > 1\tGot={num_classes}")
            if not (0.0 <= float(dropout_rate) <= 1.0):
                raise ValueError(f"Invalid dropout_rate\tExpected [0,1]\tGot={dropout_rate}")

            inputs = layers.Input(shape=input_shape, name="input")
            logger(f"Input shape\t{input_shape}", color="cyan")

            x = Homework3VGG16._vgg_conv_block_keras(
                inputs, filters=64, conv_count=2, block_name="block1", kernel_initializer=kernel_initializer
            )
            x = Homework3VGG16._vgg_conv_block_keras(
                x, filters=128, conv_count=2, block_name="block2", kernel_initializer=kernel_initializer
            )
            x = Homework3VGG16._vgg_conv_block_keras(
                x, filters=256, conv_count=3, block_name="block3", kernel_initializer=kernel_initializer
            )
            x = Homework3VGG16._vgg_conv_block_keras(
                x, filters=512, conv_count=3, block_name="block4", kernel_initializer=kernel_initializer
            )
            x = Homework3VGG16._vgg_conv_block_keras(
                x, filters=512, conv_count=3, block_name="block5", kernel_initializer=kernel_initializer
            )

            if include_top:
                x = layers.Flatten(name="flatten")(x)
                x = layers.Dense(
                    4096,
                    activation="relu",
                    kernel_initializer=kernel_initializer,
                    bias_initializer="zeros",
                    name="fc1",
                )(x)
                x = layers.Dropout(rate=dropout_rate, name="dropout1")(x)
                x = layers.Dense(
                    4096,
                    activation="relu",
                    kernel_initializer=kernel_initializer,
                    bias_initializer="zeros",
                    name="fc2",
                )(x)
                x = layers.Dropout(rate=dropout_rate, name="dropout2")(x)
                outputs = layers.Dense(
                    num_classes,
                    activation="softmax",
                    kernel_initializer=kernel_initializer,
                    bias_initializer="zeros",
                    name="predictions",
                )(x)
            else:
                outputs = x  # feature extractor output

            model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
            logger(f"VGG16 model created\tname={model_name}", color="green")
            logger(f"Model parameters\t{model.count_params():,}", color="cyan")

            return model

        except Exception as e:
            logger(f"VGG16 build failed\t{e}", color="red")
            raise

    @staticmethod
    def log_vgg16_summary(model) -> None:
        """
        Prints the Keras model summary.
        """
        title_message("HW3\tTask1\tVGG16 Summary", color="magenta")

        try:
            if model is None:
                raise ValueError("Model is None\tCannot print summary")

            logger("Printing model.summary() ...", color="yellow")
            model.summary()
            logger("Summary printed.", color="green")

        except Exception as e:
            logger(f"Model summary failed\t{e}", color="red")
            raise

    # ============================================================
    # TASK 2 (Keras): TFDS Cats vs Dogs training pipeline
    # ============================================================

    @staticmethod
    def _prepare_tfds_cats_vs_dogs(
            image_size: tuple = (224, 224),
            batch_size: int = 32,
            shuffle_buffer: int = 2048,
            seed: int = 42,
            data_dir: str = None,
    ):
        """
        TFDS Cats vs Dogs:
            - 80/20 split
            - VGG preprocessing (RGB->BGR + mean subtraction)
        """
        title_message("HW3\tTask2\tTFDS Load + VGG Preprocess", color="blue")

        try:
            import tensorflow as tf
            import tensorflow_datasets as tfds
            from tensorflow.keras.applications.vgg16 import preprocess_input

            if batch_size <= 0:
                raise ValueError(f"batch_size must be > 0\tGot={batch_size}")
            if not (isinstance(image_size, tuple) and len(image_size) == 2):
                raise ValueError(f"image_size must be (H,W)\tGot={image_size}")

            (train_raw, val_raw), ds_info = tfds.load(
                "cats_vs_dogs",
                split=["train[:80%]", "train[80%:]"],
                as_supervised=True,
                with_info=True,
                data_dir=data_dir,
            )

            class_names = ds_info.features["label"].names
            logger(f"Classes\t{class_names}", color="cyan")

            def _preprocess(img, label):
                img = tf.image.resize(img, image_size)
                img = tf.cast(img, tf.float32)
                img = preprocess_input(img)
                return img, label

            train_ds = train_raw.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
            train_ds = train_ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
            train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            val_ds = val_raw.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
            val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            logger("TFDS datasets ready.", color="green")
            return train_ds, val_ds, class_names

        except Exception as e:
            logger(f"TFDS load failed\t{e}", color="red")
            raise

    @staticmethod
    def _prepare_local_cats_vs_dogs_dir(
            dataset_dir: str,
            image_size: tuple = (224, 224),
            batch_size: int = 32,
            seed: int = 42,
            cache: bool = False,
    ):
        """
        Local PetImages folder:
            dataset_dir/
                Cat/
                Dog/
        """
        title_message("HW3\tTask2\tLocal Dir Load + VGG Preprocess", color="blue")

        try:
            import os
            import tensorflow as tf
            from tensorflow.keras.applications.vgg16 import preprocess_input

            if not isinstance(dataset_dir, str) or len(dataset_dir.strip()) == 0:
                raise ValueError("dataset_dir must be a non-empty string")
            if not os.path.isdir(dataset_dir):
                raise ValueError(f"dataset_dir not found\t{dataset_dir}")

            train_ds = tf.keras.utils.image_dataset_from_directory(
                dataset_dir,
                labels="inferred",
                label_mode="int",
                validation_split=0.2,
                subset="training",
                seed=seed,
                image_size=image_size,
                batch_size=batch_size,
            )

            val_ds = tf.keras.utils.image_dataset_from_directory(
                dataset_dir,
                labels="inferred",
                label_mode="int",
                validation_split=0.2,
                subset="validation",
                seed=seed,
                image_size=image_size,
                batch_size=batch_size,
            )

            class_names = list(train_ds.class_names)
            logger(f"Classes\t{class_names}", color="cyan")

            def _vgg_map(x, y):
                x = tf.cast(x, tf.float32)
                x = preprocess_input(x)
                return x, y

            options = tf.data.Options()
            options.experimental_deterministic = False  # faster input pipeline

            train_ds = train_ds.map(_vgg_map, num_parallel_calls=tf.data.AUTOTUNE)
            train_ds = train_ds.ignore_errors()
            train_ds = train_ds.with_options(options)
            if cache:
                train_ds = train_ds.cache()
            train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

            val_ds = val_ds.map(_vgg_map, num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.ignore_errors()
            val_ds = val_ds.with_options(options)
            if cache:
                val_ds = val_ds.cache()
            val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

            logger("Local datasets ready.", color="green")
            return train_ds, val_ds, class_names

        except Exception as e:
            logger(f"Local dataset prep failed\t{e}", color="red")
            raise

    @staticmethod
    def clean_corrupted_petimages_files(
            petimages_dir: str,
            max_delete: int = 10_000,
    ):
        """
        Deletes corrupted image files from:
            petimages_dir/Cat
            petimages_dir/Dog
        """
        title_message("HW3\tTask2\tClean Corrupted Images", color="magenta")

        try:
            import os
            from PIL import Image

            if not os.path.isdir(petimages_dir):
                raise ValueError(f"petimages_dir not found\t{petimages_dir}")

            folders = ["Cat", "Dog"]
            deleted = 0
            scanned = 0

            for folder in folders:
                class_dir = os.path.join(petimages_dir, folder)
                if not os.path.isdir(class_dir):
                    raise ValueError(f"Missing class folder\t{class_dir}")

                for fname in os.listdir(class_dir):
                    fpath = os.path.join(class_dir, fname)
                    if not os.path.isfile(fpath):
                        continue
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue

                    scanned += 1
                    try:
                        with Image.open(fpath) as img:
                            img.verify()  # verifies file integrity
                    except Exception:
                        try:
                            os.remove(fpath)
                            deleted += 1
                            if deleted >= max_delete:
                                logger(f"Reached max_delete={max_delete}", color="yellow")
                                logger(f"Scanned\t{scanned}\tDeleted\t{deleted}", color="cyan")
                                return deleted
                        except Exception:
                            pass

            logger(f"Scanned\t{scanned}\tDeleted\t{deleted}", color="green")
            return deleted

        except Exception as e:
            logger(f"Corrupt clean failed\t{e}", color="red")
            raise

    @staticmethod
    def _deprocess_vgg16_for_display(x):
        """
        Undo VGG preprocessing for visualization:
            - add ImageNet means (BGR order)
            - convert to RGB
            - scale to [0,1]
        """
        try:
            import tensorflow as tf

            means_bgr = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
            x = x + means_bgr
            x = x[..., ::-1]  # BGR -> RGB
            x = tf.clip_by_value(x / 255.0, 0.0, 1.0)
            return x

        except Exception as e:
            logger(f"Deprocess failed\t{e}", color="red")
            raise

    @staticmethod
    def _compute_val_metrics_keras(model, val_ds, class_names: list):
        """
        Metrics:
            - overall accuracy
            - per-class precision
        """
        title_message("HW3\tTask2\tValidation Metrics", color="blue")

        try:
            import tensorflow as tf
            from sklearn.metrics import accuracy_score, precision_score

            y_true = []
            y_pred = []

            for xb, yb in val_ds:
                probs = model(xb, training=False)
                preds = tf.argmax(probs, axis=1)
                y_true.extend(yb.numpy().tolist())
                y_pred.extend(preds.numpy().tolist())

            acc = float(accuracy_score(y_true, y_pred))
            prec = precision_score(
                y_true,
                y_pred,
                average=None,
                labels=list(range(len(class_names))),
                zero_division=0,
            )

            metrics_dict = {"val_accuracy": acc, "class_names": class_names}
            for i, cname in enumerate(class_names):
                metrics_dict[f"precision_{cname}"] = float(prec[i])

            logger(f"Val Accuracy\t{acc:.4f}", color="green")
            for cname in class_names:
                logger(f"Precision\t{cname}\t{metrics_dict[f'precision_{cname}']:.4f}", color="green")

            return metrics_dict

        except Exception as e:
            logger(f"Metric computation failed\t{e}", color="red")
            raise

    @staticmethod
    def _plot_train_val_loss(train_loss: list, val_loss: list, title: str):
        """
        Plot train/val loss over epochs.
        """
        title_message("HW3\tTask2\tPlot Loss Curves", color="blue")

        try:
            import matplotlib.pyplot as plt

            if len(train_loss) == 0:
                logger("No loss data found to plot.", color="red")
                return

            epochs = list(range(1, len(train_loss) + 1))
            plt.figure()
            plt.plot(epochs, train_loss, label="train_loss")
            if len(val_loss) == len(train_loss):
                plt.plot(epochs, val_loss, label="val_loss")
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

            logger("Loss curves plotted.", color="green")

        except Exception as e:
            logger(f"Loss plot failed\t{e}", color="red")
            raise

    @staticmethod
    def _show_qualitative_keras(model, val_ds, class_names: list, count: int = 12):
        """
        Display images with predicted vs true labels (validation batch).
        """
        title_message("HW3\tTask2\tQualitative Results", color="blue")

        try:
            import tensorflow as tf
            import numpy as np
            import matplotlib.pyplot as plt

            xb, yb = next(iter(val_ds))
            probs = model(xb, training=False)
            preds = tf.argmax(probs, axis=1).numpy()

            images_disp = Homework3VGG16._deprocess_vgg16_for_display(xb).numpy()
            labels = yb.numpy()

            n = min(count, images_disp.shape[0])
            cols = 4
            rows = int(np.ceil(n / cols))

            plt.figure(figsize=(12, 3 * rows))
            for i in range(n):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(images_disp[i])
                t = class_names[int(labels[i])]
                p = class_names[int(preds[i])]
                plt.title(f"T:{t}  P:{p}")
                plt.axis("off")
            plt.tight_layout()
            plt.show()

            logger("Qualitative samples displayed.", color="green")

        except Exception as e:
            logger(f"Qualitative display failed\t{e}", color="red")
            raise

    @staticmethod
    def train_vgg16_cats_vs_dogs_keras(
            input_shape: tuple = (224, 224, 3),
            batch_size: int = 16,
            epochs: int = 5,
            learning_rate: float = 1e-3,
            momentum: float = 0.9,
            dropout_rate: float = 0.5,
            seed: int = 42,
            use_early_stopping: bool = True,
            early_stop_patience: int = 4,
            show_qualitative: bool = True,
            qualitative_count: int = 12,
            dataset_source: str = "local",
            data_dir: str = None,
            local_dataset_dir: str = None,
            auto_download_on_tfds_fail: bool = True,
            auto_download_root_dir: str = ".",
            force_redownload: bool = False,
            steps_per_epoch: int = None,
            validation_steps: int = None,
            log_every_n_batches: int = 50,
            verbose: int = 1,
            force_device: str = "auto",  # "auto" | "gpu" | "cpu"
    ):
        """
        Task 2:
            - Supports TFDS or local dataset
            - Optional steps_per_epoch/validation_steps for faster runs
            - Batch logging so epoch doesn't look "stuck"
            - Device selection: GPU if available (or forced)
        """
        title_message("HW3\tTask2\tTrain VGG16 (Keras) Cats vs Dogs", color="magenta")

        try:
            import tensorflow as tf
            import numpy as np

            if not (isinstance(input_shape, tuple) and len(input_shape) == 3):
                raise ValueError(f"input_shape must be (H,W,C)\tGot={input_shape}")
            if epochs <= 0:
                raise ValueError(f"epochs must be > 0\tGot={epochs}")

            tf.random.set_seed(seed)
            np.random.seed(seed)

            # ---- Device selection ----
            gpus = tf.config.list_physical_devices("GPU")
            if force_device not in ("auto", "gpu", "cpu"):
                raise ValueError(f"force_device must be 'auto'|'gpu'|'cpu'\tGot={force_device}")

            if force_device == "gpu":
                if not gpus:
                    raise RuntimeError("force_device='gpu' but TensorFlow reports no GPU devices.")
                device_name = "/GPU:0"
            elif force_device == "cpu":
                device_name = "/CPU:0"
            else:
                device_name = "/GPU:0" if gpus else "/CPU:0"

            if gpus:
                # Safe default: allow TF to grow memory usage rather than pre-allocating all VRAM
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
                logger(f"TensorFlow GPU detected\tcount={len(gpus)}\tUsing\t{device_name}", color="green")
            else:
                logger(f"No TensorFlow GPU detected\tUsing\t{device_name}", color="yellow")

            # --- Data ---
            if dataset_source.lower() == "tfds":
                try:
                    train_ds, val_ds, class_names = Homework3VGG16._prepare_tfds_cats_vs_dogs(
                        image_size=(input_shape[0], input_shape[1]),
                        batch_size=batch_size,
                        seed=seed,
                        data_dir=data_dir,
                    )
                except Exception as e_tfds:
                    if not auto_download_on_tfds_fail:
                        raise
                    logger(f"TFDS failed -> switching to PetImages\t{e_tfds}", color="yellow")
                    petimages_dir = Homework3VGG16.download_cats_vs_dogs_petimages_to_dir(
                        target_root_dir=auto_download_root_dir,
                        force_redownload=force_redownload,
                    )
                    train_ds, val_ds, class_names = Homework3VGG16._prepare_local_cats_vs_dogs_dir(
                        dataset_dir=petimages_dir,
                        image_size=(input_shape[0], input_shape[1]),
                        batch_size=batch_size,
                        seed=seed,
                    )
            elif dataset_source.lower() == "local":
                if local_dataset_dir is None:
                    raise ValueError("local_dataset_dir is required when dataset_source='local'")
                train_ds, val_ds, class_names = Homework3VGG16._prepare_local_cats_vs_dogs_dir(
                    dataset_dir=local_dataset_dir,
                    image_size=(input_shape[0], input_shape[1]),
                    batch_size=batch_size,
                    seed=seed,
                )
            else:
                raise ValueError("dataset_source must be 'tfds' or 'local'")

            # Repeat only when step limits are used (prevents iterator exhaustion)
            if steps_per_epoch is not None:
                train_ds = train_ds.repeat()
                logger(f"Using steps_per_epoch={steps_per_epoch}", color="yellow")
            if validation_steps is not None:
                val_ds = val_ds.repeat()
                logger(f"Using validation_steps={validation_steps}", color="yellow")

            # --- Model + Train on chosen device ---
            with tf.device(device_name):
                model = Homework3VGG16.build_vgg16_keras(
                    input_shape=input_shape,
                    num_classes=2,
                    include_top=True,
                    dropout_rate=dropout_rate,
                    model_name="VGG16_Manual_CatsDogs_Keras",
                )

                optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

                model.compile(
                    optimizer=optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )

                logger(
                    f"Params\tdevice={device_name}\tbatch={batch_size}\tepochs={epochs}\tlr={learning_rate}\t"
                    f"momentum={momentum}\tdropout={dropout_rate}\tsource={dataset_source}",
                    color="cyan",
                )

                callbacks = []

                if use_early_stopping:
                    callbacks.append(
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss",
                            patience=early_stop_patience,
                            restore_best_weights=True,
                        )
                    )
                    logger(f"EarlyStopping\tpatience={early_stop_patience}", color="yellow")

                class BatchLogger(tf.keras.callbacks.Callback):
                    def on_train_batch_end(self, batch, logs=None):
                        if log_every_n_batches and (batch + 1) % log_every_n_batches == 0:
                            loss_v = float(logs.get("loss", 0.0)) if logs else 0.0
                            acc_v = float(logs.get("accuracy", 0.0)) if logs else 0.0
                            logger(f"Batch {batch + 1}\tloss={loss_v:.4f}\tacc={acc_v:.4f}", color="cyan")

                callbacks.append(BatchLogger())

                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=epochs,
                    callbacks=callbacks,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    verbose=verbose,
                )

            metrics_dict = Homework3VGG16._compute_val_metrics_keras(
                model=model,
                val_ds=val_ds.take(validation_steps) if validation_steps else val_ds,
                class_names=class_names,
            )

            Homework3VGG16._plot_train_val_loss(
                train_loss=history.history.get("loss", []),
                val_loss=history.history.get("val_loss", []),
                title="VGG16 (Keras) Cats vs Dogs - Loss Curves",
            )

            if show_qualitative:
                Homework3VGG16._show_qualitative_keras(
                    model=model,
                    val_ds=val_ds.take(1),
                    class_names=class_names,
                    count=qualitative_count,
                )

            logger("HW3 Task2 completed successfully.", color="green")
            return model, history, metrics_dict

        except Exception as e:
            logger(f"HW3 Task2 failed\t{e}", color="red")
            raise

    @staticmethod
    def download_cats_vs_dogs_petimages_to_dir(
            target_root_dir: str = ".",
            force_redownload: bool = False,
            url: str = None,
    ) -> str:
        """
        Downloads Cats vs Dogs zip and extracts it into:
            {target_root_dir}/cats_vs_dogs_data/extracted/PetImages

        Skip rules:
            - If PetImages exists -> skip download + extraction
            - Else if zip exists -> skip download, do extraction
            - force_redownload=True -> wipe work_dir and redo everything
        """
        title_message("HW3\tTask2\tDownload Cats vs Dogs (PetImages)", color="magenta")

        try:
            import os
            import zipfile
            import shutil
            import urllib.request

            root_dir = os.path.abspath(target_root_dir)
            work_dir = os.path.join(root_dir, "cats_vs_dogs_data")
            zip_path = os.path.join(work_dir, "kagglecatsanddogs_5340.zip")
            extract_dir = os.path.join(work_dir, "extracted")
            petimages_dir = os.path.join(extract_dir, "PetImages")

            if url is None:
                url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

            # Force redownload resets everything
            if force_redownload and os.path.exists(work_dir):
                logger("force_redownload=True -> removing existing dataset folder", color="yellow")
                shutil.rmtree(work_dir)

            os.makedirs(work_dir, exist_ok=True)

            # If already extracted and valid, skip everything
            cat_dir = os.path.join(petimages_dir, "Cat")
            dog_dir = os.path.join(petimages_dir, "Dog")
            if os.path.isdir(cat_dir) and os.path.isdir(dog_dir):
                logger(f"PetImages already present -> skipping\t{petimages_dir}", color="green")
                return petimages_dir

            # Download only if zip doesn't exist
            if not os.path.exists(zip_path):
                logger(f"Downloading\t{url}", color="yellow")
                urllib.request.urlretrieve(url, zip_path)
                logger(f"Downloaded\t{zip_path}", color="green")
            else:
                logger(f"Zip already exists -> skipping download\t{zip_path}", color="cyan")

            # Extract only if PetImages is missing
            if not os.path.exists(petimages_dir):
                os.makedirs(extract_dir, exist_ok=True)
                logger("Extracting zip...", color="yellow")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(extract_dir)
                logger(f"Extracted\t{extract_dir}", color="green")
            else:
                logger(f"Extract folder exists -> skipping extraction\t{petimages_dir}", color="cyan")

            # Validate structure
            if not (os.path.isdir(cat_dir) and os.path.isdir(dog_dir)):
                raise ValueError(f"Invalid structure\tExpected Cat/ and Dog/\tGot={petimages_dir}")

            logger(f"PetImages ready\t{petimages_dir}", color="green")
            return petimages_dir

        except Exception as e:
            logger(f"Download/extract failed\t{e}", color="red")
            raise

    @staticmethod
    def run_task2_with_auto_download(
            input_shape: tuple = (224, 224, 3),
            batch_size: int = 32,
            epochs: int = 5,
            learning_rate: float = 1e-3,
            momentum: float = 0.9,
            dropout_rate: float = 0.5,
            seed: int = 42,
            use_early_stopping: bool = True,
            early_stop_patience: int = 3,
            show_qualitative: bool = True,
            qualitative_count: int = 12,
            target_root_dir: str = ".",
            force_redownload: bool = False,
            clean_corrupt: bool = True,
    ):
        """
        - Downloads + extracts PetImages into project folder
        - Optionally deletes corrupted images once
        - Trains using local loader
        """
        title_message("HW3\tTask2\tAuto-Download + Train", color="magenta")

        try:
            petimages_dir = Homework3VGG16.download_cats_vs_dogs_petimages_to_dir(
                target_root_dir=target_root_dir,
                force_redownload=force_redownload,
            )

            if clean_corrupt:
                deleted = Homework3VGG16.clean_corrupted_petimages_files(petimages_dir)
                logger(f"Corrupt files deleted\t{deleted}", color="yellow")

            return Homework3VGG16.train_vgg16_cats_vs_dogs_keras(
                input_shape=input_shape,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                momentum=momentum,
                dropout_rate=dropout_rate,
                seed=seed,
                use_early_stopping=use_early_stopping,
                early_stop_patience=early_stop_patience,
                show_qualitative=show_qualitative,
                qualitative_count=qualitative_count,
                dataset_source="local",
                local_dataset_dir=petimages_dir,
            )

        except Exception as e:
            logger(f"Auto-download train failed\t{e}", color="red")
            raise


if __name__ == "__main__":
    # -------------------------
    # HW1 Tests
    # -------------------------
    foods_to_buy = ["banana", "orange", "apple"]
    inventory_stock = {"banana": 4, "apple": 0, "orange": 9}
    unit_prices = {"banana": 3.0, "apple": 2.0, "orange": 2.5}
    # try:
    #     title_message("Demonstration Run 1\tSuccessful Calculation", color="green")
    #     Homework1Utils.compute_cost(food=foods_to_buy, stock=inventory_stock, prices=unit_prices, raiseException=True)
    # except Exception as e:
    #     logger(f"Unexpected error\tRun1\t{e}", color="red")
    #
    # try:
    #     title_message("Demonstration Run 2\tInput Validation (Expected Failure)", color="green")
    #     Homework1Utils.compute_cost(food="Not a list", stock=inventory_stock, prices=unit_prices, raiseException=True)
    # except TypeError as e:
    #     logger(f"Successfully caught expected error\t{e}", color="green")
    # except Exception as e:
    #     logger(f"Caught unexpected error\t{e}", color="red")
    #
    # logger("Scenario 1\tPrime up to 30", color="magenta")
    # primes_up_to_30 = Homework1Utils.prime_range(x=30)
    # logger(f"Final primes returned\t{primes_up_to_30}", color="green")
    #
    # # ----------
    # # HW2 Tests: Forward + Backprop
    # # -------------------------
    # title_message("HW2\tForward + Backprop\t(LeakyReLU / ELU)", color="blue")
    #
    # X = np.linspace(-2.0, 2.0, 64, dtype=np.float64).reshape(-1, 1)
    # T = np.sin(X).astype(np.float64)
    #
    # # Scenario 1: LeakyReLU
    # try:
    #     title_message("Scenario 1\tPOSITIVE\tLeakyReLU + Backprop", color="green")
    #     layers_1 = [
    #         NeuralNetCore.init_layer(in_features=1, out_features=100, fcn="leaky_relu", alpha=0.01, seed=10),
    #         NeuralNetCore.init_layer(in_features=100, out_features=100, fcn="leaky_relu", alpha=None, seed=20),
    #         NeuralNetCore.init_layer(in_features=100, out_features=1, fcn="linear", alpha=None, seed=20),
    #     ] # Setting two init_layers in between input and output for testing == curious about change in Loss and Epoch
    #     # observation: the more layers, the less loss
    #     NeuralNetCore.train_mlp_mse(X=X, T=T, layers=layers_1, epochs=200, lr=0.05)
    #     logger("Scenario 1 completed successfully.", color="green")
    # except Exception as e:
    #     logger(f"Scenario 1 failed unexpectedly\t{e}", color="red")
    #     raise
    #
    # # Scenario 2: ELU
    # try:
    #     title_message("Scenario 2\tPOSITIVE\tELU + Backprop", color="green")
    #     layers_2 = [
    #         NeuralNetCore.init_layer(in_features=1, out_features=10, fcn="elu", alpha=1.0, seed=30),
    #         NeuralNetCore.init_layer(in_features=10, out_features=1, fcn="linear", alpha=None, seed=40),
    #     ]
    #     NeuralNetCore.train_mlp_mse(X=X, T=T, layers=layers_2, epochs=200, lr=0.05)
    #     logger("Scenario 2 completed successfully.", color="green")
    # except Exception as e:
    #     logger(f"Scenario 2 failed unexpectedly\t{e}", color="red")
    #     raise
    #
    # # Scenario 3: Invalid activation name
    # try:
    #     title_message("Scenario 3\tNEGATIVE\tInvalid Activation Name", color="yellow")
    #     layers_bad = [
    #         NeuralNetCore.init_layer(in_features=1, out_features=5, fcn="unknown_activation", alpha=None, seed=50),
    #         NeuralNetCore.init_layer(in_features=5, out_features=1, fcn="linear", alpha=None, seed=60),
    #     ]
    #     _ = NeuralNetCore.nn_backward_update(X, T, layers_bad, Activation_fcn(), lr=0.01)
    #     logger("ERROR\tScenario 3 unexpectedly succeeded (should fail).", color="red")
    #     raise RuntimeError("Scenario 3 unexpectedly succeeded: invalid activation name should have failed.")
    # except ValueError as e:
    #     logger(f"Successfully caught expected ValueError\t{e}", color="green")
    # except Exception as e:
    #     logger(f"Caught unexpected error type\t{e}", color="red")
    #     raise
    #
    # # HW2 Task4 demos
    # try:
    #     title_message("HW2\tTask4\tEx10 Demonstrations", color="blue")
    #     Homework2Demos.demo_task4_regression_ex10(epochs=300, lr=0.05, seed=123, show_plot=True)
    #     Homework2Demos.demo_task4_classification_ex10(epochs=300, lr=0.05, seed=123, show_plot=True)
    #     logger("HW2 Task4 completed successfully.", color="green")
    # except Exception as e:
    #     logger(f"HW2 Task4 failed unexpectedly\t{e}", color="red")
    #     raise
    #
    # title_message("HW2 Demonstrations Complete!", color="blue")

    # model = Homework3VGG16.build_vgg16_keras(input_shape=(224,224,3), num_classes=1000, include_top=True)
    # Homework3VGG16.log_vgg16_summary(model)
    #
    # model_k, hist_k, met_k = Homework3VGG16.train_vgg16_cats_vs_dogs_keras(
    #     dataset_source="tfds",
    #     auto_download_on_tfds_fail=True,
    #     auto_download_root_dir=".",
    #     force_redownload=False,
    #     epochs=5,
    # )

    # model_k, hist_k, met_k = Homework3VGG16.run_task2_with_auto_download(
    #     batch_size=8,
    #     epochs=3,
    #     force_redownload=False,
    #     clean_corrupt=True,
    # )

    # model_k, hist_k, met_k = Homework3VGG16.train_vgg16_cats_vs_dogs_keras(
    #     dataset_source="local",
    #     local_dataset_dir=r".\cats_vs_dogs_data\extracted\PetImages",
    #     batch_size=8,
    #     epochs=3,
    #     steps_per_epoch=100,
    #     validation_steps=50,
    #     log_every_n_batches=10,
    #     verbose=1,
    # )

    title_message("HW3\tMAIN\tDemo: Task1 + Task2", color="magenta")

    # ----------------------------
    # Task 1: Manual VGG16 sanity tests
    # ----------------------------
    title_message("HW3\tMAIN\tTask1 Tests", color="blue")

    import numpy as np
    import tensorflow as tf

    vgg_model = Homework3VGG16.build_vgg16_keras(
        input_shape=(224, 224, 3),
        num_classes=2,
        include_top=True,
        dropout_rate=0.5,
        model_name="VGG16_Manual_Task1_Test",
    )

    Homework3VGG16.log_vgg16_summary(vgg_model)

    x_dummy = np.random.rand(2, 224, 224, 3).astype(np.float32)
    y_dummy = vgg_model(x_dummy, training=False)
    logger(f"Task1 forward pass OK\tinput={x_dummy.shape}\toutput={tuple(y_dummy.shape)}", color="green")

    # ----------------------------
    # Task 2: Dataset quick test + training quick run
    # ----------------------------
    title_message("HW3\tMAIN\tTask2 Tests", color="blue")

    petimages_dir = Homework3VGG16.download_cats_vs_dogs_petimages_to_dir(
        target_root_dir=".",
        force_redownload=False,
    )

    deleted = Homework3VGG16.clean_corrupted_petimages_files(
        petimages_dir=petimages_dir,
        max_delete=10000,
    )
    logger(f"Corrupt cleanup done\tdeleted={deleted}", color="yellow")

    model_k, hist_k, met_k = Homework3VGG16.train_vgg16_cats_vs_dogs_keras(
        input_shape=(160, 160, 3),
        dataset_source="local",
        local_dataset_dir=petimages_dir,

        batch_size=16,
        epochs=15,
        # steps_per_epoch=100,
        # validation_steps=50,

        use_early_stopping=True,
        early_stop_patience=4,

        log_every_n_batches=50,
        verbose=1,
        show_qualitative=True,
        qualitative_count=12,
    )

    logger(f"Task2 metrics\t{met_k}", color="green")

    title_message("HW3\tMAIN\tCompleted", color="green")

    # torch_vgg = Homework3VGG16.build_vgg16_torch(input_channels=3, num_classes=1000, include_top=True)
    # Homework3VGG16.log_vgg16_torch_summary(torch_vgg, input_shape=(1, 3, 224, 224))

