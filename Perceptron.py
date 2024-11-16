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
        self.weights_history = []  # Pour stocker l'évolution des poids
        for epoch in range(epochs):
            total_error = 0
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)
                update = self.learning_rate * (yi - y_pred)
                self.weights += update * xi
                self.bias += update
                total_error += abs(yi - y_pred)
            
            # Enregistrer les erreurs et l'évolution des poids
            errors.append(total_error)
            self.weights_history.append(self.weights.copy())  # Stocke une copie des poids actuels
            
            # Appel du callback s'il existe
            if callback:
                callback(epoch, self, errors)
            
            # Vérifier si l'erreur totale est inférieure à 40
            if total_error < 40:
                print(f"Entraînement arrêté à l'époque {epoch+1} avec une erreur totale de {total_error}.")
                break