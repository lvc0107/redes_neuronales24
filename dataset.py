import matplotlib.pyplot as plt
import numpy as np


def cloud(num_points_per_class):
    # Class 0
    x0 = np.random.randn(num_points_per_class, 3) + np.array([0, 0, 0])
    for i in range(num_points_per_class):
        x0[i][2] = -1
    y0 = np.zeros(num_points_per_class)

    # Class 1
    x1 = np.random.randn(num_points_per_class, 3) + np.array([4, 4, 0])
    for i in range(num_points_per_class):
        x1[i][2] = -1
    y1 = np.ones(num_points_per_class)

    # Class 2
    x2 = np.random.randn(num_points_per_class, 3) + np.array([0, 4, 0])
    for i in range(num_points_per_class):
        x2[i][2] = -1
    y2 = np.ones(num_points_per_class) * 2

    # Combine dataset
    x = np.vstack([x0, x1, x2])
    y = np.hstack([y0, y1, y2])
    # One-hot encoding
    num_classes = len(np.unique(y))
    y_one_hot = np.eye(num_classes)[y.astype(int)]

    classification = ["b"] * len(x0) + ["g"] * len(x1) + ["r"] * len(x2)
    return x, y_one_hot, classification


def plot(x, original_classification, new_classification, title):
    size_dot = 200
    plt.scatter(
        x[:, 0],
        x[:, 1],
        c=new_classification,
        alpha=0.5,
        s=size_dot,
        linewidth=3,
        edgecolors=original_classification,
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def plot_loss(x, y):
    plt.plot(range(x), y, c="cyan")
    plt.title("Errors")
    plt.xlabel("Epochs")
    plt.ylabel("Errors")
    plt.show()
