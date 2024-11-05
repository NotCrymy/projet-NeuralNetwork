from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Charger le dataset d'Iris
iris = load_iris()
X, y = iris.data, iris.target

# Diviser les données en ensemble d'entraînement et de test (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données pour rendre l'entraînement plus stable
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialiser et entraîner le perceptron multi-couches (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=2000, learning_rate_init=0.01, random_state=42)
mlp.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = mlp.predict(X_test)

# Calculer la précision
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy * 100:.2f}%")

# Réduire les données à 2 dimensions avec PCA pour visualisation
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Réentraîner un MLP sur les données réduites (juste pour visualisation)
mlp_pca = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=2000, learning_rate_init=0.01, random_state=42)
mlp_pca.fit(X_train_pca, y_train)

# Visualisation de la frontière de décision
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = mlp_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Tracé des données et de la frontière de décision
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
plt.title("Frontière de décision du MLP sur les données PCA")
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.show()

# Nouvelle observation (fleur)
nouvelle_fleur = [[5.0, 3.6, 1.4, 0.2]]

# Normaliser la nouvelle observation (avec le même scaler utilisé pour l'entraînement)
nouvelle_fleur_scaled = scaler.transform(nouvelle_fleur)

# Faire la prédiction
prediction = mlp.predict(nouvelle_fleur_scaled)

# Afficher l'espèce prédite
print(f"L'espèce prédite est : {iris.target_names[prediction][0]}")