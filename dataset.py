import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.datasets import make_blobs


def map_color(d):
    return [{0: "b", 1: "g", 2: "r"}.get(p) for p in d]


def cloud(n_samples=30):
    X, Y = make_blobs(n_samples=n_samples, centers=3, cluster_std=1.5)
    Y_one_hot = np.zeros((Y.size, 3))
    Y_one_hot[np.arange(Y.size), Y] = 1
    plot(X, Y_one_hot, title="Data set")
    return X, Y_one_hot


def plot(X, y, new_classification=None, title="Data Set"):
    original_classification = y if type(y) is list else np.argmax(y, axis=1)
    if new_classification is None:
        new_classification = original_classification

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=map_color(new_classification),
        edgecolors=map_color(original_classification),
        linewidths=2,
        s=200,
        alpha=0.5,
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def plot_loss(x, y):
    plt.plot(x, y, c="cyan")
    plt.title("Errors")
    plt.xlabel("Epochs")
    plt.ylabel("Errors")
    plt.show()


def plot_network_topology(layer_sizes):
    G = nx.DiGraph()
    pos = {}
    node_count = 0
    max_nodes = max(layer_sizes)

    for i, size in enumerate(layer_sizes):
        x_pos = i
        for j in range(size):
            y_pos = max_nodes / 2 - size / 2 + j
            pos[node_count] = (x_pos, y_pos)
            node_count += 1

    # Conectar nodos entre capas
    node_counter = 0
    for i in range(len(layer_sizes) - 1):
        for current_node in range(layer_sizes[i]):
            for next_node in range(layer_sizes[i + 1]):
                G.add_edge(
                    node_counter + current_node,
                    node_counter + layer_sizes[i] + next_node,
                )
        node_counter += layer_sizes[i]

    # Dibujar la red
    plt.figure(figsize=(12, 6))
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=400,
        node_color="lightblue",
        edge_color="gray",
    )
    plt.title("Topología de la red neuronal")
    plt.show()


def plot_decision_boundary(
    model, X, y, preds, weights=None, biases=None, title="Decision Boundary"
):
    # Crear un grid de puntos para evaluar las predicciones
    # en todo el espacio de entrada
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Y = np.argmax(y, axis=1)
    # Obtener las predicciones para cada punto del grid
    if weights is not None:
        preds_for_boundary = model.predict(grid, weights, biases)
    else:
        preds_for_boundary = model.predict(grid)
    Z = preds_for_boundary.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=map_color(preds),
        edgecolors=map_color(Y),
        linewidths=2,
        s=200,
        alpha=0.5,
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# Función para evaluar y graficar la frontera de decisión
def plot_decision_boundary_pytorch(model, X, Y, preds, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    preds_for_boundary = model.predict(grid)
    Z = preds_for_boundary.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=map_color(preds),
        edgecolors=map_color(Y),
        linewidths=2,
        s=200,
        alpha=0.5,
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
