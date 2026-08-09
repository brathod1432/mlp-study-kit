# Author: Andrzej Kordecki
# Mail: andrzej.kordecki@pw.edu.pl
# Neural Networks - Exercise 10
# Division of Theory of Machines and Robots
# Institute of Aeronautics and Applied Mechanics
# Faculty of Power and Aeronautical Engineering
# Warsaw University of Technology

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generating a dataset
X = np.linspace(0, 10, 30)
Y = 2 * X + 4*np.random.rand(np.size(X))-2
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)
Y = (Y - np.mean(Y))/np.std(Y)

# Creating a neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='linear', input_shape=(1, ))
])

# Setting the neural network optimization parameters
model.compile(optimizer='SGD', loss=tf.keras.losses.MAE)

# Starting the learning process
model.fit(X, Y, epochs=1000)

# Checking the correct operation of the neural network
predX1 = np.linspace(min(X), max(X), 100)
predY1 = model.predict(predX1)
plt.figure()
plt.scatter(X, Y, label='Dane uczące', color='blue')
plt.plot(predX1, predY1, label='MAE', color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Regresja")
plt.legend()
plt.grid(True)
plt.show()