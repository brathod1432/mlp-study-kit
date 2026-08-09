"""
nn_core -- shared building blocks for mlp-study-kit.

Provides the canonical, deduplicated implementations of every
component that appears (in progressive form) across the exercises
and homework assignments.

Imports:
    from nn_core.logger      import ObjLogger
    from nn_core.activations import ActivationFn
    from nn_core.losses      import LossFn
    from nn_core.network     import NeuralNetwork
"""

from nn_core.logger import ObjLogger
from nn_core.activations import ActivationFn
from nn_core.losses import LossFn
from nn_core.network import NeuralNetwork

__all__ = ["ObjLogger", "ActivationFn", "LossFn", "NeuralNetwork"]
