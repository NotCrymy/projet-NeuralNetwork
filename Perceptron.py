import numpy as np

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = np.zeros(n_inputs)
        self.bias = 0.0
        self.learning_rate = learning_rate

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        linear_output = np.dot(self.weights, x) + self.bias
        return self.activation_function(linear_output)

    def train(self, X, y, epochs, callback=None):
        errors = []
        for epoch in range(epochs):
            total_error = 0
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                update = self.learning_rate * (yi - y_pred)
                self.weights += update * xi
                self.bias += update
                total_error += abs(yi - y_pred)
            errors.append(total_error)
            if callback:
                callback(epoch, self, errors)