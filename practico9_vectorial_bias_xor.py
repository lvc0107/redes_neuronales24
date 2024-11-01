import matplotlib.pyplot as plt
import numpy as np

e1 = (0, 0)
e2 = (0, 1)
e3 = (1, 0)
e4 = (1, 1)

x = np.array([e1, e2, e3, e4])
colors = ["b", "r", "r", "b"]

plt.scatter(x[:, 0], x[:, 1], c=colors, s=200)
plt.title("Dots before run Perceptron")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# expected results
s = np.array([(1, 0), (0, 1), (0, 1), (1, 0)])
num_classes = 2


def relu(h):
    return np.maximum(0, h)


def relu_derivative(h):
    return np.where(h > 0, 1, 0)


def gradient_descent(x, s, learning_rate=0.02, num_iterations=1000):
    num_samples, num_features = x.shape
    num_classes = s.shape[1]

    # Set weights
    weights = np.random.randn(num_features, num_classes) * 0.01
    biases = np.zeros((1, num_classes))

    # Gradient descent iterations
    for _ in range(num_iterations):
        # Calculate h
        h = np.dot(x, weights) + biases
        # Apply ReLU activaton function
        activated = relu(h)

        # Calculate probabilities using softmax
        exp_h = np.exp(activated - np.max(activated, axis=1, keepdims=True))
        probabilities = exp_h / exp_h.sum(axis=1, keepdims=True)

        # Calculate gradients
        d_h = probabilities - s
        dW = np.dot(x.T, d_h) / num_samples
        db = np.sum(d_h, axis=0, keepdims=True) / num_samples

        # Updates weight
        weights -= learning_rate * dW
        biases -= learning_rate * db

    return weights, biases


eta = 0.02
epoch = 1000

# Train model
weights, biases = gradient_descent(x, s, learning_rate=eta, num_iterations=epoch)


# Visualize dots and classes
h = np.dot(x, weights) + biases
activated = relu(h)
exp_h = np.exp(activated - np.max(activated, axis=1, keepdims=True))
preds = np.argmax(exp_h / exp_h.sum(axis=1, keepdims=True), axis=1)

plt.scatter(
    x[:, 0],
    x[:, 1],
    c=preds,
    cmap="viridis",
    alpha=0.5,
    s=200,
    linewidth=3,
    edgecolors=colors,
)
plt.title("Clasificated dots with ReLU model")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


"""
Explicar por que no funciona xor

"""
