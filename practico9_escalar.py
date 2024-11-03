import numpy as np

from dataset import plot, x, y_one_hot
from practico9_gpt import gradient_descent as gd2, plot_preds_chat_gpt, relu_vectorial

plot(title="Data set")


def relu(h):
    return h if h > 0 else 0


def relu_derivative(h):
    return 1 if h > 0 else 0


def gradient_descent(x, y_one_hot, g, dg=None, learning_rate=0.02, num_epochs=10000):
    num_samples, num_features = x.shape
    num_classes = y_one_hot.shape[1]

    # Set weights
    w = np.random.randn(num_classes, num_features) * 0.01
    print(f"Initial weights: {w}")

    h = np.zeros((num_classes, num_samples))

    # Gradient descent iterations
    for epoch in range(num_epochs):
        # compute g(h) = h for each dot (P) and for each expected output (n_s)
        for u in range(num_samples):
            for i in range(num_classes):
                h[i, u] = 0.0
                for k in range(num_features):
                    h[i, u] += w[i, k] * x[u, k]

        loss = 0.0
        for u in range(num_samples):
            for i in range(num_classes):
                y_iu = g(h[i, u])
                # error of the i-th output on the u-th example
                error_iu = y_iu - y_one_hot[u, i]
                loss += error_iu**2  # compute squared error
                # gradient_component = error_iu  # * dg(h[i, u])
                for k in range(num_features):
                    w[i, k] -= learning_rate * error_iu * x[u, k]
        loss *= 0.5
        if epoch % (num_epochs / 10) == 0:
            print(f"Iteration epoch: {epoch}, Loss: {loss:.4f}")
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
weights2, biases = gd2(x, y_one_hot, relu_vectorial, learning_rate, num_epochs)
print(f"Final weights: {weights2}")
title = f"Dots classificated with {relu_vectorial.__name__} model (chatGPT)"
plot_preds_chat_gpt(title, relu_vectorial, x, weights2, biases)


# ===========================================

# Train model
weights = gradient_descent(
    x, y_one_hot, relu, relu_derivative, learning_rate, num_epochs
)
print(f"Final weights: {weights}")
title = f"Clasificated dots with {relu.__name__} model. Scalar version"
plot_preds_scalar(title, relu, x, weights)
