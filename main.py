import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
from visualization import plot_decision_boundary

# Données d'entrée (exemple avec la porte OR)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Étiquettes correspondantes
y = np.array([0, 1, 1, 1])

# Initialisation du perceptron
perceptron = Perceptron(n_inputs=2, learning_rate=0.1)

# Fonction de callback pour la visualisation
def callback(epoch, perceptron_instance, errors):
    plot_decision_boundary(epoch, perceptron_instance, errors, X, y)

# Entraînement du perceptron avec visualisation en direct
plt.ion()
perceptron.train(X, y, epochs=10, callback=callback)
plt.ioff()
plt.show()
