"""Tests for modules.data_utils."""
import csv
import os
import numpy as np
import pytest

from modules.data_utils import (
    k_fold_split,
    load_csv,
    make_classification_data,
    make_linear_data,
    make_regression_data,
    normalize,
    train_test_split,
)


class TestMakeRegressionData:
    def test_output_shapes(self):
        X, Y = make_regression_data(n=50)
        assert X.shape == (50, 1)
        assert Y.shape == (50, 1)

    def test_reproducible_with_seed(self):
        X1, Y1 = make_regression_data(seed=0)
        X2, Y2 = make_regression_data(seed=0)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(Y1, Y2)

    def test_different_seeds_differ(self):
        _, Y1 = make_regression_data(seed=0)
        _, Y2 = make_regression_data(seed=1)
        assert not np.allclose(Y1, Y2)

    def test_x_range(self):
        X, _ = make_regression_data(n=100, x_range=(-5.0, 5.0))
        assert X.min() >= -5.0
        assert X.max() <= 5.0

    def test_noise_affects_y(self):
        _, Y_low  = make_regression_data(noise=0.0, seed=42)
        _, Y_high = make_regression_data(noise=2.0, seed=42)
        # higher noise = larger variance in Y
        assert Y_high.std() > Y_low.std()

    def test_dtype_float64(self):
        X, Y = make_regression_data()
        assert X.dtype == np.float64
        assert Y.dtype == np.float64


class TestMakeLinearData:
    def test_output_shapes(self):
        X, Y = make_linear_data(n=30)
        assert X.shape == (30, 1)
        assert Y.shape == (30, 1)

    def test_zero_noise_is_linear(self):
        X, Y = make_linear_data(n=10, slope=3.0, intercept=-1.0, noise=0.0, seed=0)
        expected = 3.0 * X - 1.0
        np.testing.assert_array_almost_equal(Y, expected)

    def test_reproducible(self):
        X1, Y1 = make_linear_data(seed=7)
        X2, Y2 = make_linear_data(seed=7)
        np.testing.assert_array_equal(Y1, Y2)


class TestMakeClassificationData:
    def test_output_shapes(self):
        X, Y, idx0, idx1 = make_classification_data(n_per_class=40)
        assert X.shape == (80, 2)
        assert Y.shape == (80,)

    def test_class_balance(self):
        X, Y, idx0, idx1 = make_classification_data(n_per_class=50)
        assert len(idx0) == 50
        assert len(idx1) == 50

    def test_labels_are_0_and_1(self):
        _, Y, _, _ = make_classification_data()
        assert set(Y.tolist()) == {0.0, 1.0}

    def test_reproducible(self):
        X1, Y1, _, _ = make_classification_data(seed=3)
        X2, Y2, _, _ = make_classification_data(seed=3)
        np.testing.assert_array_equal(Y1, Y2)


class TestTrainTestSplit:
    def test_output_shapes(self):
        X = np.ones((100, 2))
        T = np.ones((100, 1))
        Xtr, Ttr, Xte, Tte = train_test_split(X, T, test_ratio=0.3)
        assert Xtr.shape[0] + Xte.shape[0] == 100
        assert Ttr.shape[0] + Tte.shape[0] == 100

    def test_test_ratio_respected(self):
        n = 200
        X = np.ones((n, 1))
        T = np.ones((n, 1))
        Xtr, _, Xte, _ = train_test_split(X, T, test_ratio=0.2)
        assert Xte.shape[0] == round(0.2 * n)

    def test_reproducible(self):
        X = np.arange(100).reshape(-1, 1).astype(float)
        T = np.arange(100).reshape(-1, 1).astype(float)
        Xtr1, _, Xte1, _ = train_test_split(X, T, seed=0)
        Xtr2, _, Xte2, _ = train_test_split(X, T, seed=0)
        np.testing.assert_array_equal(Xtr1, Xtr2)

    def test_invalid_ratio_raises(self):
        X, T = np.ones((10, 1)), np.ones((10, 1))
        with pytest.raises(ValueError, match="test_ratio"):
            train_test_split(X, T, test_ratio=1.5)

    def test_shape_mismatch_raises(self):
        X = np.ones((10, 1))
        T = np.ones((9, 1))
        with pytest.raises(ValueError, match="rows"):
            train_test_split(X, T)


class TestNormalize:
    def test_zero_mean(self):
        X = np.array([[1.0], [2.0], [3.0]])
        X_n, mu, _ = normalize(X)
        assert abs(X_n.mean()) < 1e-6

    def test_unit_std(self):
        X = np.random.default_rng(0).normal(5.0, 2.0, (100, 3))
        X_n, _, _ = normalize(X)
        np.testing.assert_array_almost_equal(X_n.std(axis=0), np.ones(3), decimal=4)

    def test_apply_existing_stats(self):
        X_train = np.array([[1.0], [2.0], [3.0]])
        X_test  = np.array([[4.0], [5.0]])
        _, mu, sigma = normalize(X_train)
        X_te_norm, _, _ = normalize(X_test, mean=mu, std=sigma)
        # test values should be shifted by the train mean
        expected = (X_test - mu) / (sigma + 1e-8)
        np.testing.assert_array_almost_equal(X_te_norm, expected)

    def test_returns_float64(self):
        X = np.array([[1, 2, 3]], dtype=np.int32)
        X_n, _, _ = normalize(X)
        assert X_n.dtype == np.float64


# ──────────────────────────────────────────────────────────────────────────────
# load_csv
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadCsv:
    def _write_csv(self, tmp_path, rows, header=None):
        p = tmp_path / "data.csv"
        with open(p, "w", newline="") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
            writer.writerows(rows)
        return str(p)

    def test_basic_load(self, tmp_path):
        path = self._write_csv(tmp_path,
                               [[1.0, 2.0], [3.0, 4.0]],
                               header=["x", "y"])
        X, Y = load_csv(path)
        assert X.shape == (2, 1)
        assert Y.shape == (2, 1)
        np.testing.assert_almost_equal(Y.flatten(), [2.0, 4.0])

    def test_no_header(self, tmp_path):
        path = self._write_csv(tmp_path, [[1.0, 5.0], [2.0, 6.0]])
        X, Y = load_csv(path, has_header=False)
        assert X.shape == (2, 1)

    def test_multi_feature(self, tmp_path):
        path = self._write_csv(tmp_path,
                               [[1, 2, 3, 99], [4, 5, 6, 88]],
                               header=["a", "b", "c", "target"])
        X, Y = load_csv(path, feature_cols=[0, 1, 2], target_col=3)
        assert X.shape == (2, 3)
        assert Y.shape == (2, 1)

    def test_last_col_default_target(self, tmp_path):
        path = self._write_csv(tmp_path, [[1, 2, 3]], header=["a","b","c"])
        X, Y = load_csv(path)
        assert X.shape == (1, 2)   # cols 0, 1
        np.testing.assert_almost_equal(Y[0, 0], 3.0)

    def test_float64_dtype(self, tmp_path):
        path = self._write_csv(tmp_path, [[1, 2]], header=["x","y"])
        X, Y = load_csv(path)
        assert X.dtype == np.float64
        assert Y.dtype == np.float64

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path/data.csv")

    def test_non_numeric_raises(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("x,y\nhello,1.0\n")
        with pytest.raises(ValueError, match="non-numeric"):
            load_csv(str(p))


# ──────────────────────────────────────────────────────────────────────────────
# k_fold_split
# ──────────────────────────────────────────────────────────────────────────────

class TestKFoldSplit:
    def test_returns_k_folds(self):
        X = np.ones((50, 2))
        T = np.ones((50, 1))
        folds = k_fold_split(X, T, k=5)
        assert len(folds) == 5

    def test_train_val_sizes(self):
        n, k = 100, 5
        X = np.ones((n, 1))
        T = np.ones((n, 1))
        for Xtr, Ttr, Xval, Tval in k_fold_split(X, T, k=k):
            assert Xtr.shape[0] + Xval.shape[0] == n
            assert Xval.shape[0] == n // k

    def test_all_samples_covered(self):
        n = 30
        X = np.arange(n, dtype=float).reshape(-1, 1)
        T = np.zeros((n, 1))
        folds = k_fold_split(X, T, k=3)
        all_val = np.concatenate([Xval.flatten() for _, _, Xval, _ in folds])
        assert len(all_val) == n
        assert set(all_val.tolist()) == set(range(n))

    def test_reproducible(self):
        X = np.arange(20, dtype=float).reshape(-1, 1)
        T = np.zeros((20, 1))
        f1 = k_fold_split(X, T, k=4, seed=0)
        f2 = k_fold_split(X, T, k=4, seed=0)
        for (Xtr1, _, Xv1, _), (Xtr2, _, Xv2, _) in zip(f1, f2):
            np.testing.assert_array_equal(Xv1, Xv2)

    def test_invalid_k_raises(self):
        X = np.ones((10, 1))
        T = np.ones((10, 1))
        with pytest.raises(ValueError, match="k must be between"):
            k_fold_split(X, T, k=1)
        with pytest.raises(ValueError, match="k must be between"):
            k_fold_split(X, T, k=11)

    def test_shape_mismatch_raises(self):
        X = np.ones((10, 1))
        T = np.ones((9, 1))
        with pytest.raises(ValueError, match="rows"):
            k_fold_split(X, T, k=3)
