import matplotlib.pyplot as plt
import numpy as np


def generate_cloud(num_points_per_class, sigma=0.1):
    range_cloud = 5
    center = (np.random.randint(range_cloud), np.random.randint(range_cloud))
    cloud = []
    for _ in range(num_points_per_class):
        x = center[0] + np.random.normal(scale=sigma)
        y = center[1] + np.random.normal(scale=sigma)
        new_point = (x, y)
        cloud.append(new_point)

    return cloud


num_points_per_class = 10
cloud_1 = generate_cloud(num_points_per_class)
plt.scatter(*zip(*cloud_1), color="b", s=200)

cloud_2 = generate_cloud(num_points_per_class)
plt.scatter(*zip(*cloud_2), color="g", s=200)

cloud_3 = generate_cloud(num_points_per_class)
plt.scatter(*zip(*cloud_3), color="r", s=200)
plt.title("Dots before run Perceptron")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


x = np.array(cloud_1 + cloud_2 + cloud_3)
colors = (
    ["blue" for _ in cloud_1] + ["green" for _ in cloud_2] + ["red" for _ in cloud_3]
)

# expected results
s = np.array(
    [(1, 0, 0) for _ in cloud_1]
    + [(0, 1, 0) for _ in cloud_2]
    + [(0, 0, 1) for _ in cloud_3]
)
num_classes = 3
s2 = np.eye(num_classes)


def relu(h):
    return np.maximum(0, h)


def relu_derivative(h):
    return np.where(h > 0, 1, 0)


def gradient_descent(x, s, learning_rate=0.02, num_iterations=1000):
    num_samples, num_features = x.shape
    num_classes = s.shape[1]

    # Set weights
    weights = np.random.randn(num_features, num_classes) * 0.01
    # Gradient descent iterations
    for _ in range(num_iterations):
        # Calculate h
        h = np.dot(x, weights)
        # Apply ReLU activaton function
        activated = relu(h)

        # Calculate probabilities using softmax
        exp_h = np.exp(activated - np.max(activated, axis=1, keepdims=True))
        probabilities = exp_h / exp_h.sum(axis=1, keepdims=True)

        # Calculate gradients
        d_h = probabilities - s
        dW = np.dot(x.T, d_h) / num_samples

        # Updates weight
        weights -= learning_rate * dW

    return weights


eta = 0.02
epoch = 1000

# Train model
weights = gradient_descent(x, s, learning_rate=eta, num_iterations=epoch)


# Visualize dots and classes
h = np.dot(x, weights)
activated = relu(h)
exp_h = np.exp(activated - np.max(activated, axis=1, keepdims=True))
preds = np.argmax(exp_h / exp_h.sum(axis=1, keepdims=True), axis=1)

plt.scatter(
    x[:, 0],
    x[:, 1],
    c=preds,
    cmap="winter",
    alpha=0.5,
    s=200,
    linewidth=3,
    edgecolors=colors,
)
plt.title("Dots clasificated with ReLU model")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
