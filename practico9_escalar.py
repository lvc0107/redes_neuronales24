import matplotlib.pyplot as plt
import numpy as np


def generate_cloud(num_points_per_class, sigma=0.1):
    range_cloud = 5
    center = (
        int(np.random.normal() * range_cloud),
        int(np.random.normal() * range_cloud),
    )
    cloud = []
    for _ in range(num_points_per_class):
        x = center[0] + np.random.normal(scale=sigma)
        y = center[1] + np.random.normal(scale=sigma)
        new_point = (x, y, -1)
        cloud.append(new_point)

    return cloud


num_points_per_class = 2
cloud_1 = generate_cloud(num_points_per_class)
cloud_2 = generate_cloud(num_points_per_class)
cloud_3 = generate_cloud(num_points_per_class)


x = np.array(cloud_1 + cloud_2 + cloud_3)
colors = ["b"] * len(cloud_1) + ["g"] * len(cloud_2) + ["r"] * len(cloud_3)


plt.scatter(x[:, 0], x[:, 1], c=colors, s=200)
plt.title("Dots before run Perceptron")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


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


def gradient_descent(e, s, g, dg, eta=0.02, num_epochs=1000):
    P, n_e = e.shape
    n_s = s.shape[1]

    # Set weights
    w = np.random.randn(n_s, n_e) * 0.1

    h = np.zeros((n_s, P))

    # Gradient descent iterations
    for epoch in range(num_epochs):
        # compute g(h) = h for each dot (P) and for each expected output (n_s)
        for u in range(P):
            for i in range(n_s):
                h[i, u] = 0.0
                for k in range(n_e):
                    h[i, u] += w[i, k] * e[u, k]

        E = 0.0
        for u in range(P):
            for i in range(n_s):
                y_iu = g(h[i, u])
                # error of the i-th output on the u-th example
                error_iu = y_iu - s[u, i]
                E += error_iu**2  # compute squared error
                gradient_component = error_iu * dg(h[i, u])
                for k in range(n_e):
                    w[i, k] -= eta * gradient_component * e[u, k]

        if epoch % 1000 == 0:
            print(f"Iteration {epoch}, Error: {E:.4f}")
    return w


eta = 0.02
epochs = 100000


# Train model
w = gradient_descent(x, s, relu, relu_derivative, eta=eta, num_epochs=epochs)


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
new_classes = np.zeros(M)

for m in range(M):
    y = pred(w, relu, x[m, :])
    new_classes[m] = np.argmax(y)

plt.scatter(
    x[:, 0],
    x[:, 1],
    c=new_classes,
    cmap="viridis",
    alpha=0.5,
    linewidth=3,
    edgecolors=colors,
    s=200,
)
plt.title("Clasificated dots with ReLU model")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
