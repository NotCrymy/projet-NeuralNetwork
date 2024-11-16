from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

# Générer le dataset en forme de deux lunes
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# Initialisation et entraînement du perceptron multi-couches (MLP) avec des paramètres ajustés
mlp = MLPClassifier(hidden_layer_sizes=(100, 10, 100), max_iter=1000, learning_rate_init=0.01, tol=1e-6, random_state=42)
mlp.fit(X, y)

# Visualisation de la frontière de décision
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Tracé des données et de la frontière de décision
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.title("Frontière de décision du MLP")
plt.xlabel("Entrée x1")
plt.ylabel("Entrée x2")
plt.show()

# Visualiser l'évolution de la fonction de perte (erreur) au cours des époques
plt.plot(mlp.loss_curve_)
plt.title('Évolution de la perte au cours des époques')
plt.xlabel('Époque')
plt.ylabel('Erreur (loss)')
plt.grid()
plt.show()
