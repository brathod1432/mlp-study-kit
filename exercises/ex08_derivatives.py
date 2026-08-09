# Author: Andrzej Kordecki
# Mail: andrzej.kordecki@pw.edu.pl
# Neural Networks - Exercise 07
# Division of Theory of Machines and Robots
# Institute of Aeronautics and Applied Mechanics
# Faculty of Power and Aeronautical Engineering
# Warsaw University of Technology

import sys
import numpy as np
import matplotlib.pyplot as plt

np. random.seed(100)

#  Neuron class = calculation of: output, activation potential, activation functions
class Activation_fcn:
    def __init__(self):
        self.functions = {
            'linear': self.linear,
            'sigmoid': self.logistic,
            'logistic': self.logistic,
            'tanh': self.tanh,
            'relu': self.relu            
        }
    
    # Claculate neuron output
    def output(self, layer, name, derivative=False):
        if name in self.functions:
            return self.functions[name](layer, derivative)
        else:
            sys.exit(f"Error: Activation function '{name}' not found.")

    # Identity activation function
    def linear(self, layer, derivative=False):
        return layer['activation_potential'] if not derivative else np.ones_like(layer['activation_potential'])

    # Logistic (sigmoid) activation function
    def logistic(self, layer, derivative=False):
        if not derivative:
            return 1.0 / (1.0 + np.exp(-layer['activation_potential']))
        else:
            return layer['output'] * (1.0 - layer['output'])

    # Hyperbolic tangent activation function  
    def tanh(self, layer, derivative=False):
        if not derivative:
            exp_sum = np.exp(layer['activation_potential']) + np.exp(-layer['activation_potential'])
            return (np.exp(layer['activation_potential']) - np.exp(-layer['activation_potential'])) / exp_sum
        else:
            return 1.0 - np.power(layer['output'], 2)

    #  ReLU activation function 
    def relu(self, layer, derivative=False):
        return np.maximum(0, layer['activation_potential']) if not derivative else (layer['activation_potential'] >= 0).astype(int)
    
#  Loss function class
class Loss_fcn:
    def __init__(self):
        self.functions = {
            'mse': self.mse,
            'binary_cross_entropy': self.binary_cross_entropy
        }

    # Loss/error value calculated for all input data sample
    def output(self, name, expected, outputs, derivative):
        if name in self.functions:
            return self.functions[name](expected, outputs, derivative)
        else:
            sys.exit(f"Error: Loss function '{name}' not found.")

    # Mean Square Error loss function
    def mse(self, expected, outputs, derivative=False):
        if not derivative:
            return 0.5 * np.power(expected - outputs, 2)
        else:
            return -(expected - outputs)

    # Cross-entropy loss function
    def binary_cross_entropy(self, expected, outputs, derivative=False):
        if not derivative:
            return -expected * np.log(outputs) - (1 - expected) * np.log(1 - outputs)
        else:
            return -(expected / outputs - (1 - expected) / (1 - outputs))

# Initialize a network
class Neural_network:
    def __init__(self):
        self.af = Activation_fcn()
        self.loss = Loss_fcn()

    def create_network(self, structure, init_weight="rand"):
        self.nnetwork = [structure[0]]
        
        for i in range(1, len(structure)):
            n, m = structure[i]['units'], structure[i-1]['units']
            
            match init_weight:
                case "rand":
                    weight = np.random.randn(n, m)
                case "one":
                    weight = np.ones((n, m))
                case "zero":
                    weight = np.zeros((n, m))
                
            new_layer = {
                'weights': weight,
                'activation_function': structure[i]['activation_function'],
                'activation_potential': None,
                'delta': None,
                'output': None}
            self.nnetwork.append(new_layer)
        return self.nnetwork
    
    # Forward propagate input to a network output
    def forward_propagate(self, nnetwork, inputs):
        # Network input values from dataset
        inp = inputs.copy()
        for i in range(1, len(nnetwork)):
            # Storage of network outputs from present layer of network
            nnetwork[i]['activation_potential'] = np.matmul(nnetwork[i]['weights'], inp).flatten()
            nnetwork[i]['output'] = self.af.output(nnetwork[i], nnetwork[i]['activation_function'])
            inp = nnetwork[i]['output']
        return inp

    # Backpropagate error and store it in neuron
    def backward_propagate(self, loss_function, nnetwork, expected):
        None
        
    # Update network weights with error
    def update_weights(self, nnetwork, inputs, l_rate):
        None
        
    # Train a network for a fixed number of epochs
    def train(self, nnetwork, x_train, y_train, l_rate=0.01, n_epoch=100, loss_function='mse', verbose=1):
        for epoch in range(n_epoch):
            None
        
    # Calculate network output
    def predict(self, inputs):
        out = []
        for input in inputs:
            out.append(self.forward_propagate(self.nnetwork, input))
        return out

structure = [
{'type': 'input', 'units': 1},
{'type': 'dense', 'units': 2, 'activation_function': 'linear'},
{'type': 'dense', 'units': 2, 'activation_function': 'linear'},
{'type': 'dense', 'units': 1, 'activation_function': 'linear'}]

model = Neural_network()
model.create_network(structure, "rand")

n = 5
X = np.linspace(-5, 5, n).reshape(-1, 1)

predicted = model.predict(X)
print("Input = {}".format(X,))
print("Prediction = {}".format(predicted))