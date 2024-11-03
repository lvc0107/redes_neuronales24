import matplotlib.pyplot as plt
import numpy as np

num_points_per_class = 30

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

size_dot = 200
dataset_colors = ["b"] * len(x0) + ["g"] * len(x1) + ["r"] * len(x2)


mapping = {0: "b", 1: "g", 2: "r"}


def plot(preds=None, title="XXX"):
    if preds is None:
        colors = dataset_colors
    else:
        new_colors = []
        for p in preds:
            new_colors.append(mapping.get(p))
        colors = new_colors

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
