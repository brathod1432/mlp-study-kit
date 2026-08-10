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


class TestRepr:
    def test_repr_uninitialised(self):
        model = NeuralNetwork()
        assert "uninitialised" in repr(model)

    def test_repr_shows_layers(self, model):
        net = model.create_network(STRUCTURE_SIMPLE)
        r = repr(model)
        assert "NeuralNetwork(" in r
        assert "tanh" in r
        assert "linear" in r
        assert "input" in r

    def test_str_equals_repr(self, model):
        net = model.create_network(STRUCTURE_SIMPLE)
        assert str(model) == repr(model)


class TestSaveLoadWeights:
    def test_save_load_roundtrip(self, tmp_path, model):
        net = model.create_network(STRUCTURE_SIMPLE)
        original = [layer["weights"].copy() for layer in net[1:]]

        # save_weights now uses .npz format (no pickle)
        path = str(tmp_path / "weights")   # extension added automatically
        model.save_weights(net, path)

        # Scramble weights then reload
        for layer in net[1:]:
            layer["weights"] = np.zeros_like(layer["weights"])

        model.load_weights(net, path)

        for i, layer in enumerate(net[1:]):
            np.testing.assert_array_almost_equal(layer["weights"], original[i])

    def test_npz_file_created(self, tmp_path, model):
        net  = model.create_network(STRUCTURE_SIMPLE)
        path = str(tmp_path / "w")
        model.save_weights(net, path)
        import os
        assert os.path.isfile(path + ".npz")

    def test_load_with_explicit_npz_extension(self, tmp_path, model):
        net  = model.create_network(STRUCTURE_SIMPLE)
        path = str(tmp_path / "weights")
        model.save_weights(net, path)
        # Loading with explicit .npz should also work
        model.load_weights(net, path + ".npz")

    def test_load_shape_mismatch_raises(self, tmp_path, model):
        net  = model.create_network(STRUCTURE_SIMPLE)
        path = str(tmp_path / "weights")
        model.save_weights(net, path)

        net2 = model.create_network([
            {"type": "input", "units": 3},
            {"type": "dense", "units": 4, "activation_function": "tanh",    "bias": False},
            {"type": "dense", "units": 1, "activation_function": "linear",  "bias": False},
        ])
        with pytest.raises(ValueError, match="Shape mismatch"):
            model.load_weights(net2, path)


class TestSummary:
    def test_summary_contains_param_count(self, model):
        net = model.create_network(STRUCTURE_SIMPLE)
        s = model.summary(net)
        assert "Total trainable parameters" in s
        assert "params=" in s

    def test_summary_shows_all_layers(self, model):
        net = model.create_network(STRUCTURE_SIMPLE)
        s = model.summary(net)
        assert "tanh" in s
        assert "linear" in s
        assert "input" in s

    def test_summary_uninitialised(self):
        m = NeuralNetwork()
        assert "uninitialised" in m.summary()

    def test_summary_param_count_correct(self):
        """1 input → 4 tanh (bias=False) → 1 linear (bias=False).
        Layer1 weights: (4,1)=4, Layer2 weights: (1,4)=4  → total 8.
        """
        m = NeuralNetwork()
        structure = [
            {"type": "input",  "units": 1},
            {"type": "dense",  "units": 4, "activation_function": "tanh",   "bias": False},
            {"type": "dense",  "units": 1, "activation_function": "linear", "bias": False},
        ]
        net = m.create_network(structure)
        s = m.summary(net)
        assert "8" in s   # total params


class TestGradientClipping:
    def test_grad_clip_does_not_crash(self):
        np.random.seed(7)
        m = NeuralNetwork()
        structure = [
            {"type": "input",  "units": 1},
            {"type": "dense",  "units": 4, "activation_function": "tanh",   "bias": True},
            {"type": "dense",  "units": 1, "activation_function": "linear", "bias": True},
        ]
        net = m.create_network(structure)
        X = np.linspace(-1, 1, 10).reshape(-1, 1)
        Y = 2.0 * X
        loss, _, _ = m.train(net, X, Y, l_rate=0.01, n_epoch=5,
                             verbose=0, grad_clip=1.0)
        assert not np.isnan(loss)


class TestSaveHistory:
    def test_csv_created(self, tmp_path):
        import os
        m   = NeuralNetwork()
        structure = [
            {"type": "input",  "units": 1},
            {"type": "dense",  "units": 4, "activation_function": "tanh",   "bias": True},
            {"type": "dense",  "units": 1, "activation_function": "linear", "bias": True},
        ]
        net = m.create_network(structure)
        X = np.linspace(-1, 1, 10).reshape(-1, 1)
        Y = 2.0 * X
        csv_path = str(tmp_path / "history.csv")
        m.train(net, X, Y, l_rate=0.01, n_epoch=5, verbose=0,
                save_history=csv_path)
        assert os.path.isfile(csv_path)

    def test_csv_has_correct_columns(self, tmp_path):
        import csv, os
        m   = NeuralNetwork()
        structure = [
            {"type": "input",  "units": 1},
            {"type": "dense",  "units": 2, "activation_function": "tanh",   "bias": False},
            {"type": "dense",  "units": 1, "activation_function": "linear", "bias": False},
        ]
        net = m.create_network(structure)
        X   = np.linspace(-1, 1, 8).reshape(-1, 1)
        Y   = X
        X_te, Y_te = X[:2], Y[:2]
        csv_path = str(tmp_path / "h.csv")
        m.train(net, X, Y, X_te, Y_te, l_rate=0.01, n_epoch=3,
                verbose=0, save_history=csv_path)
        with open(csv_path) as f:
            header = next(csv.reader(f))
        assert header == ["epoch", "loss_train", "loss_test"]


class TestNaNDetection:
    def test_nan_raises_runtime_error(self):
        """Inject NaN into weights to force NaN loss."""
        m   = NeuralNetwork()
        structure = [
            {"type": "input",  "units": 1},
            {"type": "dense",  "units": 2, "activation_function": "tanh",   "bias": False},
            {"type": "dense",  "units": 1, "activation_function": "linear", "bias": False},
        ]
        net = m.create_network(structure)
        # Corrupt weights with NaN to force divergence
        net[1]["weights"][:] = np.nan
        X = np.linspace(-1, 1, 5).reshape(-1, 1)
        Y = X
        with pytest.raises(RuntimeError, match="NaN"):
            m.train(net, X, Y, l_rate=0.1, n_epoch=3, verbose=0)


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

        # train() now returns (final_loss, history_train, history_test)
        final_loss, history_tr, history_te = model.train(
            net, X, Y, l_rate=0.05, n_epoch=200, loss_function="mse", verbose=0
        )

        assert final_loss < initial_loss, (
            f"Expected loss to decrease, got initial={initial_loss:.4f} final={final_loss:.4f}"
        )
