import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        # Initialisation des poids et du biais
        self.weights = np.zeros(n_inputs)
        self.bias = 0.0
        self.learning_rate = learning_rate

    def activation_function(self, x):
        # Fonction d'activation échelon de Heaviside
        return 1 if x >= 0 else 0

    def predict(self, x):
        # Calcul de la sortie
        linear_output = np.dot(self.weights, x) + self.bias
        return self.activation_function(linear_output)

    def train(self, X, y, epochs):
        # Entraînement du perceptron
        for epoch in range(epochs):
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                # Mise à jour des poids et du biais
                update = self.learning_rate * (yi - y_pred)
                self.weights += update * xi
                self.bias += update
