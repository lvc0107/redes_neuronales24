import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.datasets import make_blobs


def map_color(d):
    return [{0: "b", 1: "g", 2: "r"}.get(p) for p in d]


def cloud(n_samples):
    X, Y = make_blobs(n_samples=30, centers=3, cluster_std=1.5)
    Y_one_hot = np.zeros((Y.size, 3))
    Y_one_hot[np.arange(Y.size), Y] = 1
    plot(X, Y_one_hot, title="Data set")
    return X, Y_one_hot


def plot(X, y, new_classification=None, title="Data Set"):
    original_classification = y if type(y) is list else np.argmax(y, axis=1)
    if new_classification is None:
        new_classification = original_classification

    size_dot = 200
    plt.scatter(
        X[:, 0],
        X[:, 1],
        alpha=0.5,
        linewidths=2,
        c=map_color(new_classification),
        edgecolors=map_color(original_classification),
        s=size_dot,
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
    # Crear un grid de puntos para evaluar las predicciones en todo el espacio de entrada
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Obtener las predicciones para cada punto del grid
    if weights and biases:
        Z = model.predict(grid_points, weights, biases)
    else:
        Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Crear el gráfico
    plt.contourf(xx, yy, Z, alpha=0.8)
    # plt.colorbar()
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=map_color(preds),
        linewidths=2,
        edgecolors=map_color(np.argmax(y, axis=1)),
        s=200,
        alpha=0.5,
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# Función para evaluar y graficar la frontera de decisión
def plot_decision_boundary_pytorch(model, X, y, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    preds = model.predict(grid)
    plt.contourf(xx, yy, preds.reshape(xx.shape), alpha=0.3, cmap="viridis")
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y.numpy(),
        edgecolors=map_color(y.tolist()),
        s=200,
        alpha=0.5,
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
