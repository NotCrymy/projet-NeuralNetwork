import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(epoch, perceptron, errors, X, y):
    plt.clf()
    plt.subplot(1, 2, 1)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = np.array([perceptron.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, levels=[-1, 0, 1], cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.title(f"Perceptron - Époque {epoch+1}")
    plt.xlabel("Entrée x1")
    plt.ylabel("Entrée x2")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch+2), errors, marker='o')
    plt.title('Erreur au cours du temps')
    plt.xlabel('Époque')
    plt.ylabel('Erreur totale')
    plt.tight_layout()
    plt.pause(0.1)

def plot_weights_evolution(perceptron):
    plt.figure(figsize=(10, 5))
    weights = np.array(perceptron.weights_history)
    for i in range(weights.shape[1]):
        plt.plot(weights[:, i], label=f'Poids {i + 1}')
    plt.xlabel('Époque')
    plt.ylabel('Valeur des poids')
    plt.title('Évolution des poids du perceptron')
    plt.legend()
    plt.grid()
    plt.show()
