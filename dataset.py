import matplotlib.pyplot as plt
import numpy as np


def cloud(num_points_per_class, bias=False):
    # Class 0
    x0 = np.random.randn(num_points_per_class, 2) + np.array([0, 0])
    if bias:
        x0 = np.random.randn(num_points_per_class, 3) + np.array([0, 0, 0])
        for i in range(num_points_per_class):
            x0[i][3] = -1

    y0 = np.zeros(num_points_per_class)
    # Class 1
    x1 = np.random.randn(num_points_per_class, 2) + np.array([4, 4])
    if bias:
        x1 = np.random.randn(num_points_per_class, 3) + np.array([4, 4, 0])
        for i in range(num_points_per_class):
            x1[i][2] = -1
    y1 = np.ones(num_points_per_class)

    # Class 2
    x2 = np.random.randn(num_points_per_class, 2) + np.array([0, 4])
    if bias:
        for i in range(num_points_per_class):
            x2 = np.random.randn(num_points_per_class, 3) + np.array([0, 4, 4])
            x2[i][2] = -1
    y2 = np.ones(num_points_per_class) * 2

    # Combine dataset
    x = np.vstack([x0, x1, x2])
    y = np.hstack([y0, y1, y2])
    # One-hot encoding
    num_classes = len(np.unique(y))
    y_one_hot = np.eye(num_classes)[y.astype(int)]

    return x, y_one_hot


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
