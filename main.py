from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron
from visualization import plot_decision_boundary, plot_weights_evolution

# Fonction de callback pour la visualisation
def callback(epoch, perceptron_instance, errors):
    plot_decision_boundary(epoch, perceptron_instance, errors, X, y)

# Générer le dataset en forme de deux lunes
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# Initialisation du perceptron avec différentes valeurs de learning_rate
learning_rates = [0.1]

for lr in learning_rates:
    print(f"Training with learning_rate = {lr}")
    perceptron = Perceptron(n_inputs=2, learning_rate=lr)
    
    plt.ion()  # Active le mode interactif de matplotlib
    perceptron.train(X, y, epochs=400, callback=callback)

    # Tracer l'évolution des poids après l'entraînement
    plot_weights_evolution(perceptron)

# Désactiver le mode interactif et afficher les graphiques
plt.ioff()
plt.show()