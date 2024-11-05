import matplotlib.pyplot as plt
import numpy as np

from dataset import plot_loss
from practico10_escalar import (
    derivative_relu,
    derivative_sigmoid,
    gradient_descent,
    plot_preds_scalar,
    relu,
    sigmoid,
)

# Xor

e1 = (0, 0, -1)
e2 = (0, 1, -1)
e3 = (1, 0, -1)
e4 = (1, 1, -1)

x = np.array([e1, e2, e3, e4])
colors = ["b", "r", "r", "b"]

plt.scatter(x[:, 0], x[:, 1], c=colors, s=200)
plt.title("Dots before run Perceptron")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# expected results
y_one_hot = np.array([(1, 0), (0, 1), (0, 1), (1, 0)])

num_epochs = 10000
learning_rate = 0.02
# Train model
w1, w2, aggregated_loss = gradient_descent(
    x,
    y_one_hot,
    relu,
    derivative_relu,
    sigmoid,
    derivative_sigmoid,
    learning_rate,
    num_epochs,
)
print(f"weights layer 1: {w1}")
print(f"weights layer 2: {w2}")
title = "Clasificated dots with 2 layers. Scalar version"
plot_preds_scalar(x, y_one_hot, colors, w1, w2, relu, sigmoid, title)
plot_loss(range(num_epochs), aggregated_loss)


# Not bias
e1 = (0, 0)
e2 = (0, 1)
e3 = (1, 0)
e4 = (1, 1)

# Train model
w1, w2, aggregated_loss = gradient_descent(
    x,
    y_one_hot,
    relu,
    derivative_relu,
    sigmoid,
    derivative_sigmoid,
    learning_rate,
    num_epochs,
)
print(f"weights layer 1: {w1}")
print(f"weights layer 2: {w2}")
title = "Clasificated dots with 2 layers. Scalar version"
plot_preds_scalar(x, y_one_hot, colors, w1, w2, relu, sigmoid, title)
plot_loss(range(num_epochs), aggregated_loss)
