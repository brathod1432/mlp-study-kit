# Author: Andrzej Kordecki
# Mail: andrzej.kordecki@pw.edu.pl
# Neural Networks - Exercise 09
# Division of Theory of Machines and Robots
# Institute of Aeronautics and Applied Mechanics
# Faculty of Power and Aeronautical Engineering
# Warsaw University of Technology

import os
import numpy as np
import matplotlib
_mpl_backend = os.environ.get("MPLBACKEND", "")
if _mpl_backend:
    matplotlib.use(_mpl_backend)
import matplotlib.pyplot as plt

# Neuron strcutre
neuron = {"weights": None,
          "activation_potential": None,
          "activation_function": "sigmoid",
          "output": None}

np.random.seed(4)
print(np.random.rand(4))
print(np.random.randn(4))

# Print every time we run the program:
# [0.96702984 0.54723225 0.97268436 0.71481599]
# [-0.99590893  0.69359851 -0.41830152 -1.58457724]

def generate_weights(neuron, number):
    neuron["weights"] = [np.random.randn() for i in range(number)]
    return neuron

generate_weights(neuron, 2)
# Print
# {'weights': [-0.6477067671218505, 0.5985751739673772],
#  'activation_potential': None,
#  'activation_function': 'sigmoid',
#  'output': None}

def neuron_activation_potential(neuron, inputs):
    activation = 0
    for i, weight in enumerate(neuron["weights"]):
        activation += weight * inputs[i]
    # We can also use matrix operation
    # neuron["activation_potential"] = np.matmul(neuron["weights"], inputs)
    return neuron

# Inputs
input = np.array([1, 2])
neuron_activation_potential(neuron, input)
# Print
# {'weights': [0.42507239648702144, 0.33225314537233536],
#  'activation_potential': np.float64(1.0895786872316922),
#  'activation_function': 'sigmoid',
#  'output': None}

def neuron_linear(neuron):
    return neuron['activation_potential']

neuron = {"weights": None,
          "activation_potential": np.linspace(-5, 5, 100),
          "activation_function": "sigmoid",
          "output": None}

plt.figure
plt.plot(neuron['activation_potential'], neuron_linear(neuron))
plt.show()

def neuron_tanh(neuron):
    out = (np.exp(neuron['activation_potential']) \
        - np.exp(-neuron['activation_potential'])) \
        / (np.exp(neuron['activation_potential']) \
        + np.exp(-neuron['activation_potential']))
    return out

neuron = {"weights": None,
          "activation_potential": np.linspace(-5, 5, 100),
          "activation_function": "sigmoid",
          "output": None}

plt.figure
plt.plot(neuron['activation_potential'], neuron_tanh(neuron))
plt.show()

def neuron_relu(neuron):
    out = np.maximum(0, neuron['activation_potential'])
    return out

neuron = {"weights": None,
          "activation_potential": np.linspace(-5, 5, 100),
          "activation_function": "sigmoid",
          "output": None}

plt.figure
plt.plot(neuron['activation_potential'], neuron_relu(neuron))
plt.show()

# def loss_fcn(loss, expected, outputs):
#     loss = str.lower(loss) # convert to lower case
#     error_sum = 0
#     if loss == 'mse':
#         error_sum = mse(expected,
#                         outputs)
#     elif loss == "binary_cross_entropy":
#         error_sum = binary_cross_entropy(expected,
#                                          outputs)
#     return error_sum

def loss_MSE(outputs, expected):
  return (expected - outputs) ** 2

expected = np.zeros((100, ))
outputs = np.linspace(-5, 5, 100)

plt.figure
plt.plot(outputs, loss_MSE(outputs, expected))
plt.show()

def loss_BCE(outputs, expected):
  return -expected * np.log(outputs) - (1-expected) * np.log(1 - outputs)

expected1 = np.ones((100, ))
outputs1 = np.linspace(0, 1, 100)
expected2 = np.zeros((100, ))
outputs2 = np.linspace(0, 1, 100)

plt.figure
plt.plot(outputs, loss_BCE(outputs1, expected1), "r")
plt.plot(outputs, loss_BCE(outputs2, expected2), "b")
plt.show()