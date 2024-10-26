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


def relu(h):
    return h if h > 0 else 0


def relu_derivative(h):
    return 1 if h > 0 else 0


def gradient_descent(e, s, eta=0.02, num_epochs=1000):
    M, n_e = e.shape
    n_s = s.shape[1]

    # Set weights
    w = np.random.randn(n_s, n_e) * 0.1

    h = np.zeros((n_s, M))

    # Gradient descent iterations
    for _ in range(num_epochs):
        for m in range(M):
            for j in range(n_s):
                h[j, m] = 0.0
                for i in range(n_e):
                    h[j, m] += w[j, i] * e[m, i]

        E = 0.0
        for m in range(M):
            for p in range(n_s):
                A = relu(h[p, m]) - s[m, p]
                B = A * relu_derivative(h[p, m])
                E += A * A
                for q in range(n_e):
                    w[p, q] -= eta * B * e[m, q]
    return w


eta = 0.02
epochs = 1000

# Train model
w = gradient_descent(x, s, eta=eta, num_epochs=epochs)

# Visualize dots and classes


def pred(w, g, x):
    n_s = w.shape[0]
    n_e = len(x)

    y = np.zeros(n_s)
    for j in range(n_s):
        h = 0.0
        for i in range(n_e):
            h += w[j, i] * x[i]

        y[j] = g(h)
    return y


M, n_e = x.shape
n_s = s.shape[1]
c = np.zeros(M)
b = np.zeros(M)

for m in range(M):
    y = pred(w, relu, x[m, :])
    b[m] = np.argmax(y)

plt.scatter(
    x[:, 0],
    x[:, 1],
    c=b,
    cmap="winter",
    alpha=0.5,
    linewidth=3,
    edgecolors=colors,
    s=200,
)
plt.title("Dots clasificated with ReLU model")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
