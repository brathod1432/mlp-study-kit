# experiments/ — Exploration Scripts

Date-stamped working scripts written during the course. Each one extends
the previous, adding new concepts or alternative implementations.

These are **not** polished exercises — they are authentic development
snapshots showing how the code evolved. Read them chronologically.

---

## Script Index

| File | Date | What's New vs Previous |
|------|------|----------------------|
| `20251215.py` | 2025-12-15 | First standalone MLP experiments: activation functions (LeakyReLU, ELU), basic forward + backprop. Basis for HW2 tasks. |
| `20251223.py` | 2025-12-23 | Extended training loop; regression demo (`sin(2x)+cos(x)+5`) and 2-class classification with decision boundary plotting. |
| `20251224.py` | 2025-12-24 | Pre-HW2 iteration: cleaner train/test split, improved classification dataset generator. |
| `20251224_v2.py` | 2025-12-24 (v2) | **Most evolved.** Full HW2 implementation: `NeuralNetCore` (static methods, batch-first `(N, features)`), Glorot Normal init, **Adagrad** optimizer, `Exercise10Data` class, regression dataset from `compute_cost()`. |

---

## Key Concepts Across the Series

### `20251215.py` → `20251224.py`
- `Activation_fcn` extended with `leaky_relu` (α=0.01) and `elu` (α=1.0)
- Full analytical derivatives for both
- Forward + backprop with MSE loss

### `20251224_v2.py` — HW2 Final
Introduces several production-grade patterns:

**Glorot Normal (Xavier) init:**
```python
sigma = sqrt(2 / (n_in + n_out))
w = rng.normal(0, sigma, size=(in_features, out_features))
```

**Adagrad optimizer:**
```python
G_w += dW ** 2
W -= (lr / sqrt(G_w + eps)) * dW
```

**Batch-first convention:**
All weight matrices use shape `(in_features, out_features)` rather than
the `(out_features, in_features)` convention in exercises ex06–ex10.
This is consistent with PyTorch / scikit-learn conventions.

---

## Running

```bash
# All scripts use MPLBACKEND env var to suppress display windows
export MPLBACKEND=Agg          # headless / CI
$env:MPLBACKEND = "Agg"        # PowerShell

python experiments/20251224_v2.py   # most complete
```

---

## Relation to `src/nn_core/`

`nn_core/` consolidates the best patterns from across this series:
- `nn_core/activations.py` — final `ActivationFn` (all 7 activations from `20251224_v2`)
- `nn_core/network.py` — `NeuralNetwork` with bias + early stop (from ex10 + refinements)
- `nn_core/logger.py` — `ObjLogger` (deduplicated from all 4 experiment files)

The experiment files now import `ObjLogger` and `title_message` from `nn_core` directly.
