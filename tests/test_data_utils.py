"""Tests for modules.data_utils."""
import numpy as np
import pytest

from modules.data_utils import (
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
