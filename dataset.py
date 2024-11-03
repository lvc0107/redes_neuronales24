import matplotlib.pyplot as plt
import numpy as np

# Generar un dataset de ejemplo
# np.random.seed(0)
num_points_per_class = 10

# Clase 0
x0 = np.random.randn(num_points_per_class, 2) + np.array([0, 0])
y0 = np.zeros(num_points_per_class)

# Clase 1
x1 = np.random.randn(num_points_per_class, 2) + np.array([4, 4])
y1 = np.ones(num_points_per_class)

# Clase 2
x2 = np.random.randn(num_points_per_class, 2) + np.array([0, 4])
y2 = np.ones(num_points_per_class) * 2

# Combinar el dataset
x = np.vstack([x0, x1, x2])


y = np.hstack([y0, y1, y2])
# One-hot encoding de las etiquetas
num_classes = len(np.unique(y))
y_one_hot = np.eye(num_classes)[y.astype(int)]

size_dot = 200
dataset_colors = ["b"] * len(x0) + ["g"] * len(x1) + ["r"] * len(x2)


def plot(preds=None):
    if preds is None:
        title = "Dots before run Perceptron"
        colors = dataset_colors
    else:
        title = "Clasificated dots with ReLU model"
        colors = preds

    plt.scatter(
        x[:, 0],
        x[:, 1],
        c=colors,
        alpha=0.5,
        s=size_dot,
        linewidth=3,
        edgecolors=dataset_colors,
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()