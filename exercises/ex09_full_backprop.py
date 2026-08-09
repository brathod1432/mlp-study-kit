# Author: Andrzej Kordecki
# Mail: andrzej.kordecki@pw.edu.pl
# Neural Networks - Exercise 08
# Division of Theory of Machines and Robots
# Institute of Aeronautics and Applied Mechanics
# Faculty of Power and Aeronautical Engineering
# Warsaw University of Technology

import sys
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

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
class Neural_network(object):
    def __init__(self):
        self.af = Activation_fcn()
        self.loss = Loss_fcn()

    def create_network(self, structure):
        self.nnetwork = [structure[0]]
        for i in range(1, len(structure)):
            new_layer = {
                'weights': np.random.randn(structure[i]['units'], structure[i-1]['units']),
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
            nnetwork[i]['output'] = self.af.output(nnetwork[i], nnetwork[i]['activation_function'], derivative=False)
            inp = nnetwork[i]['output']
        return inp

    # Backpropagate error and store it in neuron
    def backward_propagate(self, loss_function, nnetwork, expected):
        # Error prooagation will start from last layer
        N = len(nnetwork)-1
        for i in range(N, 0, -1):
            # Storage of error values from present layer
            errors = []
            # Calculation of error values for other layers than last layer 
            if i<N:
                weights = nnetwork[i+1]['weights']
                errors = np.matmul(nnetwork[i+1]['delta'], weights)
            # Calculation of error values for last layer
            else:                
                errors = self.loss.output(loss_function, expected, nnetwork[-1]['output'], derivative=True)
            
            nnetwork[i]['delta'] = np.multiply(errors, self.af.output(nnetwork[i], nnetwork[i]['activation_function'], derivative=True))

    # Update network weights with error
    def update_weights(self, nnetwork, inputs, l_rate):
        inp = inputs
        for i in range(1, len(nnetwork)):
            nnetwork[i]['weights'] -= l_rate * np.matmul(nnetwork[i]['delta'].reshape(-1,1), inp.reshape(1,-1))
            inp = nnetwork[i]['output']
            
    # Train a network for a fixed number of epochs
    def train(self, nnetwork, x_train, y_train, l_rate=0.01, n_epoch=100, loss_function='mse', verbose=1):
        history = []
        for epoch in range(n_epoch):
            sum_error = 0
            for iter, (x_row, y_row) in enumerate(zip(x_train, y_train)):

                self.forward_propagate(nnetwork, x_row)

                self.backward_propagate(loss_function, nnetwork, y_row)

                self.update_weights(nnetwork, x_row, l_rate)        
                
                error = np.sum(self.loss.output(loss_function, y_row, nnetwork[-1]['output'], derivative=False))
                sum_error += error       

            sum_error = sum_error/len(x_train)
            history.append(sum_error)

            if verbose > 0:
                print('>epoch=%d, loss=%.3f' % (epoch + 1, sum_error))
                
        if verbose > 0:
            print('Results: epoch=%d, loss=%.3f' % (epoch + 1, sum_error))
            plt.figure()
            plt.plot(history, label="Train loss")
            plt.xlabel("Epoches")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid()
            plt.show()
        return sum_error

    # Calculate network output
    def predict(self, nnetwork, inputs):
        out = []
        for input in inputs:
            out.append(self.forward_propagate(nnetwork, input))
        return out

# Number of samples
n = 30
# Generate regression dataset
X = np.linspace(-5, 5, n).reshape(-1, 1)
y = np.sin(2 * X) + np.cos(X)
# simulate noise
data_noise = np.random.normal(0, 0.05, n).reshape(-1, 1)
# Generate training data
Y = y + data_noise

# Create network
model = Neural_network()
structure = [{'type': 'input', 'units': 1},
            {'type': 'dense', 'units': 16, 'activation_function': 'tanh'},
            {'type': 'dense', 'units': 16, 'activation_function': 'tanh'},
            {'type': 'dense', 'units': 16, 'activation_function': 'relu'},
            {'type': 'dense', 'units': 1, 'activation_function': 'linear'}]

network = model.create_network(structure)

model.train(network, X, Y, 0.001, 1000, 'mse', 1)

predicted = model.predict(network, X)
std = np.std(predicted - Y)
print("\nError standard deviation = {}".format(std))

X_test = np.linspace(-7, 7, 100).reshape(-1, 1)
X_test = np.array(X_test).tolist()
predicted = model.predict(network, X_test)

plt.plot(X, Y, 'r--o', label="Training data")
plt.plot(X_test, predicted, 'b--x', label="Predicted")
plt.legend()
plt.grid()
plt.show()

