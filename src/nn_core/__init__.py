"""
nn_core — MLP building blocks for mlp-study-kit.

Core components (activations, losses, network, logger):

    from nn_core import NeuralNetwork, ActivationFn, LossFn, ObjLogger

See also the ``modules`` companion package for surrounding utilities:

    from modules.data_utils    import make_regression_data, train_test_split
    from modules.plot_utils    import plot_loss_history, plot_predictions
    from modules.general_utils import ensure_directory, as_float_array
"""

from nn_core.activations import ActivationFn
from nn_core.logger import ObjLogger
from nn_core.losses import LossFn
from nn_core.network import NeuralNetwork

__all__ = ["ActivationFn", "LossFn", "NeuralNetwork", "ObjLogger"]
