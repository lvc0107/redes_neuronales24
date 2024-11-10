import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
import networkx as nx


def cloud(n_samples):
        
    X, Y = make_blobs(n_samples=30, centers=3, random_state=42, cluster_std=4)
    Y_one_hot = np.zeros((Y.size, 3))
    Y_one_hot[np.arange(Y.size), Y] = 1
    plot(X, Y_one_hot, title="Data set")
    return X, Y_one_hot
    


def plot(X, y, new_classification=None, title="Data Set"):
    original_classification = np.argmax(y, axis=1)
    if new_classification is None:
        new_classification = original_classification

    def map_color(d):
        return [{0: "b", 1: "g", 2: "r"}.get(p) for p in d]

    size_dot = 200
    plt.scatter(
        X[:, 0],
        X[:, 1],
        alpha=0.5,
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

