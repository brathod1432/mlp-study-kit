"""Tests for nn_core.network -- NeuralNetwork forward pass, backprop, early stopping."""
import numpy as np
import pytest
from nn_core.network import NeuralNetwork


STRUCTURE_SIMPLE = [
    {"type": "input", "units": 2},
    {"type": "dense", "units": 4, "activation_function": "tanh",   "bias": False},
    {"type": "dense", "units": 1, "activation_function": "linear", "bias": False},
]

STRUCTURE_WITH_BIAS = [
    {"type": "input", "units": 1},
    {"type": "dense", "units": 8,  "activation_function": "tanh",   "bias": True},
    {"type": "dense", "units": 1,  "activation_function": "linear", "bias": True},
]


@pytest.fixture
def model():
    np.random.seed(42)
    return NeuralNetwork()


class TestCreateNetwork:
    def test_returns_list_of_correct_length(self, model):
        net = model.create_network(STRUCTURE_SIMPLE)
        # input layer + 2 dense layers
        assert len(net) == 3

    def test_weight_shapes_no_bias(self, model):
        net = model.create_network(STRUCTURE_SIMPLE)
        # layer 1: (4 units, 2 inputs)
        assert net[1]["weights"].shape == (4, 2)
        # layer 2: (1 unit, 4 inputs)
        assert net[2]["weights"].shape == (1, 4)

    def test_weight_shapes_with_bias(self, model):
        net = model.create_network(STRUCTURE_WITH_BIAS)
        # layer 1: (8 units, 1 input + 1 bias)
        assert net[1]["weights"].shape == (8, 2)
        # layer 2: (1 unit, 8 inputs + 1 bias)
        assert net[2]["weights"].shape == (1, 9)


class TestForwardPropagate:
    def test_output_shape(self, model):
        net = model.create_network(STRUCTURE_SIMPLE)
        x = np.array([1.0, 2.0])
        out = model.forward_propagate(net, x)
        assert out.shape == (1,)

    def test_output_is_float(self, model):
        net = model.create_network(STRUCTURE_SIMPLE)
        x = np.array([0.5, -0.5])
        out = model.forward_propagate(net, x)
        assert np.issubdtype(out.dtype, np.floating)

    def test_zero_weights_produce_zero_output_linear(self):
        model = NeuralNetwork()
        structure = [
            {"type": "input",  "units": 2},
            {"type": "dense",  "units": 1, "activation_function": "linear", "bias": False},
        ]
        net = model.create_network(structure)
        net[1]["weights"] = np.zeros((1, 2))
        out = model.forward_propagate(net, np.array([5.0, 3.0]))
        np.testing.assert_almost_equal(out[0], 0.0)


class TestPredict:
    def test_predict_batch(self, model):
        net = model.create_network(STRUCTURE_SIMPLE)
        X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        preds = model.predict(net, X)
        assert len(preds) == 3

    def test_predict_consistent_with_forward(self, model):
        net = model.create_network(STRUCTURE_SIMPLE)
        x = [1.0, 2.0]
        pred = model.predict(net, [x])[0]
        fwd  = model.forward_propagate(net, np.asarray(x))
        np.testing.assert_array_almost_equal(pred, fwd)


class TestEarlyStopping:
    def test_stops_when_plateau(self, model):
        # improvement = 0.001 - 0.001 = 0.0 < 0.01 → should stop
        history = [1.0, 0.5, 0.3, 0.2, 0.101, 0.100]
        assert model.basic_early_stop(history, epsilon=0.01) is True

    def test_continues_when_improving(self, model):
        # improvement = 0.5 - 0.1 = 0.4 > 0.01 → should NOT stop
        history = [1.0, 0.5]
        assert model.basic_early_stop(history, epsilon=0.01) is False

    def test_stops_on_increasing_loss(self, model):
        # loss went UP: improvement = 0.1 - 0.2 = -0.1 < 0.01 → stop
        history = [0.1, 0.2]
        assert model.basic_early_stop(history, epsilon=0.01) is True


class TestTraining:
    def test_loss_decreases_on_simple_regression(self):
        """Train a tiny net on a linear function -- loss must drop."""
        np.random.seed(0)
        model = NeuralNetwork()
        structure = [
            {"type": "input",  "units": 1},
            {"type": "dense",  "units": 4, "activation_function": "tanh",   "bias": True},
            {"type": "dense",  "units": 1, "activation_function": "linear", "bias": True},
        ]
        net = model.create_network(structure)
        X = np.linspace(-1, 1, 30).reshape(-1, 1)
        Y = 2.0 * X + 1.0

        # run 1 epoch to get initial loss
        initial_loss = float(np.mean([
            np.sum(0.5 * (model.forward_propagate(net, x) - y) ** 2)
            for x, y in zip(X, Y)
        ]))

        # train 200 epochs silently
        final_loss = model.train(net, X, Y, l_rate=0.05, n_epoch=200,
                                 loss_function="mse", verbose=0)

        assert final_loss < initial_loss, (
            f"Expected loss to decrease, got initial={initial_loss:.4f} final={final_loss:.4f}"
        )
