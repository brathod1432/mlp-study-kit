"""Tests for nn_core.losses -- LossFn forward + derivative."""
import numpy as np
import pytest
from nn_core.losses import LossFn


@pytest.fixture
def lf():
    return LossFn()


class TestMSE:
    def test_zero_error(self, lf):
        t = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        loss = lf.output("mse", t, y)
        np.testing.assert_array_equal(loss, [0.0, 0.0, 0.0])

    def test_known_value(self, lf):
        t = np.array([1.0])
        y = np.array([0.0])
        loss = lf.output("mse", t, y)
        np.testing.assert_almost_equal(loss[0], 0.5)   # 0.5 * (1-0)^2

    def test_derivative_sign(self, lf):
        t = np.array([1.0])
        y = np.array([0.5])
        d = lf.output("mse", t, y, derivative=True)
        # dMSE/dy = -(t - y) = -(1 - 0.5) = -0.5
        np.testing.assert_almost_equal(d[0], -0.5)

    def test_derivative_at_optimum(self, lf):
        t = np.array([2.0])
        y = np.array([2.0])
        d = lf.output("mse", t, y, derivative=True)
        np.testing.assert_almost_equal(d[0], 0.0)


class TestBCE:
    def test_perfect_prediction_class1(self, lf):
        t = np.array([1.0])
        y = np.array([0.9999])
        loss = lf.output("binary_cross_entropy", t, y)
        assert loss[0] < 0.001

    def test_perfect_prediction_class0(self, lf):
        t = np.array([0.0])
        y = np.array([0.0001])
        loss = lf.output("binary_cross_entropy", t, y)
        assert loss[0] < 0.001

    def test_derivative_direction(self, lf):
        t = np.array([1.0])
        y = np.array([0.5])
        d = lf.output("binary_cross_entropy", t, y, derivative=True)
        # For t=1, y=0.5: -(t/y - (1-t)/(1-y)) = -(1/0.5 - 0) = -2
        np.testing.assert_almost_equal(d[0], -2.0)


class TestUnknownLoss:
    def test_raises_value_error(self, lf):
        t = np.array([1.0])
        y = np.array([0.5])
        with pytest.raises(ValueError, match="Unknown loss"):
            lf.output("not_a_loss", t, y)
