import matplotlib.pyplot as plt
import numpy as np


def generate_cloud(num_points_per_class, sigma=0.1):
    range_cloud = 2
    center = (np.random.randint(range_cloud), np.random.randint(range_cloud))
    cloud = []
    for _ in range(num_points_per_class):
        x = center[0] + np.random.normal(scale=sigma)
        y = center[1] + np.random.normal(scale=sigma)
        new_point = (x, y)
        cloud.append(new_point)

    return cloud


cloud_1 = generate_cloud(num_points_per_class=3)
plt.scatter(*zip(*cloud_1), color="b")

cloud_2 = generate_cloud(num_points_per_class=2)
plt.scatter(*zip(*cloud_2), color="g")

cloud_3 = generate_cloud(num_points_per_class=3)
plt.scatter(*zip(*cloud_3), color="r")
plt.title("Dots before run Perceptron")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


x = np.array(cloud_1 + cloud_2 + cloud_3)


y_classes = np.array(
    [(1, 0, 0) for _ in cloud_1]
    + [(0, 1, 0) for _ in cloud_2]
    + [(0, 0, 1) for _ in cloud_3]
)


def relu(h):
    return np.maximum(0, h)


def relu_derivative(h):
    return np.where(h > 0, 1, 0)


def gradient_descent(x, y_classes, learning_rate=0.02, num_iterations=1000):
    num_samples, num_features = x.shape
    num_classes = y_classes.shape[1]

    # Set weights and biases
    weights = np.random.randn(num_features, num_classes) * 0.01
    biases = np.zeros((1, num_classes))

    # Gradient descent iterations
    for i in range(num_iterations):
        # Calculate logits
        logits = np.dot(x, weights) + biases
        # Apply ReLU
        activated = relu(logits)

        # Calculate probabilities using softmax
        exp_logits = np.exp(activated - np.max(activated, axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Calculate gradients
        d_logits = probabilities - y_classes
        dW = np.dot(x.T, d_logits) / num_samples
        db = np.sum(d_logits, axis=0, keepdims=True) / num_samples

        # Updates weights and biases
        weights -= learning_rate * dW
        biases -= learning_rate * db


    return weights, biases


eta = 0.02
epoch = 1000

# Train model
weights, biases = gradient_descent(
    x, y_classes, learning_rate=eta, num_iterations=epoch
)

# Visualize dots and classes
logits = np.dot(x, weights) + biases
activated = relu(logits)
exp_logits = np.exp(activated - np.max(activated, axis=1, keepdims=True))
preds = np.argmax(exp_logits / exp_logits.sum(axis=1, keepdims=True), axis=1)

plt.scatter(x[:, 0], x[:, 1], c=preds, cmap="winter", alpha=0.5)
plt.title("Dots clasificated with ReLU model")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
