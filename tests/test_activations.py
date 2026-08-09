"""Tests for nn_core.activations -- ActivationFn forward + derivative."""
import numpy as np
import pytest
from nn_core.activations import ActivationFn


def make_layer(v):
    """Helper: create a minimal layer dict with pre-activation v."""
    af = ActivationFn()
    v_arr = np.asarray(v, dtype=float)
    # compute output for derivative tests
    layer = {"activation_potential": v_arr, "output": None}
    return layer


@pytest.fixture
def af():
    return ActivationFn()


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------
class TestLinear:
    def test_forward(self, af):
        layer = {"activation_potential": np.array([1.0, -2.0, 3.0]), "output": None}
        out = af.output(layer, "linear")
        np.testing.assert_array_almost_equal(out, [1.0, -2.0, 3.0])

    def test_derivative(self, af):
        layer = {"activation_potential": np.array([5.0, -3.0]), "output": None}
        d = af.output(layer, "linear", derivative=True)
        np.testing.assert_array_equal(d, [1.0, 1.0])


# ---------------------------------------------------------------------------
# Sigmoid / Logistic
# ---------------------------------------------------------------------------
class TestSigmoid:
    def test_forward_zero(self, af):
        layer = {"activation_potential": np.array([0.0]), "output": None}
        out = af.output(layer, "sigmoid")
        np.testing.assert_almost_equal(out[0], 0.5)

    def test_forward_large_positive(self, af):
        layer = {"activation_potential": np.array([100.0]), "output": None}
        out = af.output(layer, "sigmoid")
        assert out[0] > 0.999

    def test_forward_large_negative(self, af):
        layer = {"activation_potential": np.array([-100.0]), "output": None}
        out = af.output(layer, "sigmoid")
        assert out[0] < 0.001

    def test_derivative_at_zero(self, af):
        layer = {"activation_potential": np.array([0.0]), "output": np.array([0.5])}
        d = af.output(layer, "sigmoid", derivative=True)
        np.testing.assert_almost_equal(d[0], 0.25)   # 0.5 * (1 - 0.5)

    def test_logistic_alias(self, af):
        layer = {"activation_potential": np.array([1.0]), "output": None}
        out_s = af.output(layer, "sigmoid")
        out_l = af.output(layer, "logistic")
        np.testing.assert_array_equal(out_s, out_l)


# ---------------------------------------------------------------------------
# Tanh
# ---------------------------------------------------------------------------
class TestTanh:
    def test_forward_zero(self, af):
        layer = {"activation_potential": np.array([0.0]), "output": None}
        out = af.output(layer, "tanh")
        np.testing.assert_almost_equal(out[0], 0.0)

    def test_forward_range(self, af):
        layer = {"activation_potential": np.linspace(-5, 5, 50), "output": None}
        out = af.output(layer, "tanh")
        assert (out >= -1.0).all() and (out <= 1.0).all()

    def test_derivative(self, af):
        # tanh'(0) = 1 - tanh(0)^2 = 1 - 0 = 1
        layer = {"activation_potential": np.array([0.0]), "output": np.array([0.0])}
        d = af.output(layer, "tanh", derivative=True)
        np.testing.assert_almost_equal(d[0], 1.0)


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------
class TestReLU:
    def test_forward_positive(self, af):
        layer = {"activation_potential": np.array([3.0, 0.0, -2.0]), "output": None}
        out = af.output(layer, "relu")
        np.testing.assert_array_equal(out, [3.0, 0.0, 0.0])

    def test_derivative_step(self, af):
        layer = {"activation_potential": np.array([2.0, -1.0, 0.0]), "output": None}
        d = af.output(layer, "relu", derivative=True)
        np.testing.assert_array_equal(d, [1.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Leaky ReLU
# ---------------------------------------------------------------------------
class TestLeakyReLU:
    def test_forward_default_alpha(self, af):
        layer = {"activation_potential": np.array([-4.0, 2.0]), "output": None}
        out = af.output(layer, "leaky_relu")
        np.testing.assert_almost_equal(out[0], -0.04)   # 0.01 * -4
        np.testing.assert_almost_equal(out[1], 2.0)

    def test_forward_custom_alpha(self, af):
        layer = {"activation_potential": np.array([-2.0]), "output": None}
        out = af.output(layer, "leaky_relu", alpha=0.1)
        np.testing.assert_almost_equal(out[0], -0.2)

    def test_derivative(self, af):
        layer = {"activation_potential": np.array([1.0, -1.0]), "output": None}
        d = af.output(layer, "leaky_relu", derivative=True)
        np.testing.assert_array_equal(d, [1.0, 0.01])


# ---------------------------------------------------------------------------
# ELU
# ---------------------------------------------------------------------------
class TestELU:
    def test_forward_positive(self, af):
        layer = {"activation_potential": np.array([3.0]), "output": None}
        out = af.output(layer, "elu")
        np.testing.assert_almost_equal(out[0], 3.0)

    def test_forward_negative(self, af):
        layer = {"activation_potential": np.array([-1.0]), "output": None}
        out = af.output(layer, "elu")
        expected = 1.0 * (np.exp(-1.0) - 1.0)
        np.testing.assert_almost_equal(out[0], expected)

    def test_derivative_positive(self, af):
        layer = {"activation_potential": np.array([2.0]), "output": None}
        d = af.output(layer, "elu", derivative=True)
        np.testing.assert_almost_equal(d[0], 1.0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestUnknownActivation:
    def test_raises_value_error(self, af):
        layer = {"activation_potential": np.array([0.0]), "output": None}
        with pytest.raises(ValueError, match="Unknown activation"):
            af.output(layer, "not_an_activation")
