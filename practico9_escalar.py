import numpy as np

from dataset import cloud, plot
from practico9_gpt import (
    gradient_descent as gd_chatgpt,
    plot_preds_chat_gpt,
    relu_vectorial,
)


def relu(h):
    return h if h > 0 else 0


def derivative_relu(h):
    return 1 if h > 0 else 0


def sigmoid(h):
    return 1 / (1 + np.exp(-h))


def derivative_sigmoid(h):
    return sigmoid(h) * (1 - sigmoid(h))


def hyperbolic_tangent(h):
    return np.sinh(h) / np.cosh(h)


def derivative_hyperbolic_tangent(h):
    return 1 - hyperbolic_tangent(h) ** 2


def gradient_descent(x, y_one_hot, g, dg=None, learning_rate=0.02, num_epochs=10000):
    num_samples, num_features = x.shape
    num_classes = y_one_hot.shape[1]

    # Set weights
    w = np.random.randn(num_classes, num_features) * 0.01
    print(f"Initial weights: {w}")
    h = np.zeros((num_samples, num_classes))

    def f_h(u, i, w):
        h[u, i] = 0.0
        for k in range(num_features):
            h[u, i] += w[i, k] * x[u, k]
        return h[u, i]

    # Gradient descent iterations
    for epoch in range(num_epochs):
        loss = 0.0
        for u in range(num_samples):
            for i in range(num_classes):
                h_ui = f_h(u, i, w)
                y_ui = g(h_ui)
                # error of the i-th output on the u-th example
                error_ui = y_ui - y_one_hot[u, i]
                loss += error_ui**2  # compute square error
                gradient_component = error_ui * dg(h_ui) if dg else error_ui
                """
                for relu, its derivative returns 0 when h < 0 so the product operation
                becomes 0 and the w[i,k] is not updated
                """
                for k in range(num_features):
                    w[i, k] -= learning_rate * gradient_component * x[u, k]

        loss *= 0.5
        if epoch % (num_epochs / 10) == 0:
            print(f"Iteration epoch: {epoch}, Loss: {loss:.16f}")
    return w


def pred(w, g, x):
    n_s = w.shape[0]
    n_e = len(x)

    y = np.zeros(n_s)
    for i in range(n_s):
        h = 0.0
        for k in range(n_e):
            h += w[i, k] * x[k]
        y[i] = g(h)
    return y


def plot_preds_scalar(x, original_classification, weights, g, title):
    num_samples, _ = x.shape
    preds = np.zeros(num_samples)

    for m in range(num_samples):
        y = pred(weights, g, x[m, :])
        preds[m] = np.argmax(y)

    new_clasification = [{0: "b", 1: "g", 2: "r"}.get(p) for p in preds]
    # Visualize dots and classes
    plot(x, original_classification, new_clasification, title)


learning_rate = 0.02
num_epochs = 10000
x, y_one_hot, c = cloud(num_points_per_class=10)
plot(x, c, c, title="Data set")


# =========================================
weights1, biases = gd_chatgpt(x, y_one_hot, relu_vectorial, learning_rate, num_epochs)
print(f"Final weights: {weights1}")
title = (
    f"Clasificated dots with {relu_vectorial.__name__} activation function (chatGPT)"
)
plot_preds_chat_gpt(x, c, weights1, relu_vectorial, title)


# ===========================================

# Train model
weights2 = gradient_descent(x, y_one_hot, relu, None, learning_rate, num_epochs)
print(f"Final weights: {weights2}")
title = f"Clasificated dots with {relu.__name__} activation function. Scalar version"
plot_preds_scalar(x, c, weights2, relu, title)


# ===========================================

# Train model
weights3 = gradient_descent(
    x, y_one_hot, sigmoid, derivative_sigmoid, learning_rate, num_epochs
)
print(f"Final weights: {weights3}")
title = f"Clasificated dots with {sigmoid.__name__} activation function. Scalar version"
plot_preds_scalar(x, c, weights3, sigmoid, title)


# ===========================================

# Train model
weights4 = gradient_descent(
    x,
    y_one_hot,
    hyperbolic_tangent,
    derivative_hyperbolic_tangent,
    learning_rate,
    num_epochs,
)
print(f"Final weights: {weights4}")
title = f"Clasificated dots with {hyperbolic_tangent.__name__} activation function. Scalar version"
plot_preds_scalar(x, c, weights4, hyperbolic_tangent, title)
