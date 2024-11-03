import numpy as np

from dataset import plot, x, y_one_hot
from practico9_gpt import gradient_descent as gd2, plot_preds_chat_gpt, relu_vectorial

plot(title="Data set")


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
    h = np.zeros((num_classes, num_samples))

    def f_h(i, u, w):
        h[i, u] = 0.0
        for k in range(num_features):
            h[i, u] += w[i, k] * x[u, k]
        return h[i, u]

    # Gradient descent iterations
    for epoch in range(num_epochs):
        # compute g(h) = h for each dot (P) and for each expected output (n_s)
        for u in range(num_samples):
            for i in range(num_classes):
                h[i, u] = f_h(i, u, w)

        loss = 0.0
        for u in range(num_samples):
            for i in range(num_classes):
                h_iu = f_h(i, u, w)
                y_iu = g(h_iu)
                # error of the i-th output on the u-th example
                error_iu = y_iu - y_one_hot[u, i]
                loss += error_iu**2  # compute square error
                gradient_component = error_iu * dg(h_iu) if dg else error_iu
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


# Visualize dots and classes
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


def plot_preds_scalar(title, relu, x, weights):
    num_samples, num_features = x.shape
    preds = np.zeros(num_samples)

    for m in range(num_samples):
        y = pred(weights, relu, x[m, :])
        preds[m] = np.argmax(y)

    plot(preds, title)


learning_rate = 0.02
num_epochs = 10000

# =========================================
weights1, biases = gd2(x, y_one_hot, relu_vectorial, learning_rate, num_epochs)
print(f"Final weights: {weights1}")
title = f"Clasificated dots with {relu_vectorial.__name__} model (chatGPT)"
plot_preds_chat_gpt(title, relu_vectorial, x, weights1, biases)


# ===========================================

# Train model
weights1 = gradient_descent(x, y_one_hot, relu, None, learning_rate, num_epochs)
print(f"Final weights: {weights1}")
title = f"Clasificated dots with {relu.__name__} model. Scalar version"
plot_preds_scalar(title, relu, x, weights1)


# ===========================================

# Train model
weights2 = gradient_descent(
    x, y_one_hot, sigmoid, derivative_sigmoid, learning_rate, num_epochs
)
print(f"Final weights: {weights2}")
title = f"Clasificated dots with {sigmoid.__name__} model. Scalar version"
plot_preds_scalar(title, sigmoid, x, weights2)


# ===========================================

# Train model
weights3 = gradient_descent(
    x,
    y_one_hot,
    hyperbolic_tangent,
    derivative_hyperbolic_tangent,
    learning_rate,
    num_epochs,
)
print(f"Final weights: {weights3}")
title = f"Clasificated dots with {hyperbolic_tangent.__name__} model. Scalar version"
plot_preds_scalar(title, hyperbolic_tangent, x, weights3)
