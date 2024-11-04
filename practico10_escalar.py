import numpy as np

from dataset import cloud, plot


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


def gradient_descent(
    x, y_one_hot, g1, dg1=None, g2=None, dg2=None, learning_rate=0.02, num_epochs=10000
):
    # for this example num_features and num_classes_layers are the same for both layers
    num_samples, num_features1 = x.shape
    num_samples, num_features2 = x.shape
    num_classes_layer_1 = y_one_hot.shape[1]
    num_classes_layer_2 = y_one_hot.shape[1]

    # Set weights
    w1 = np.random.randn(num_classes_layer_1, num_features1) * 0.01
    print(f"Initial weights first layer: {w1}")
    w2 = np.random.randn(num_classes_layer_2, num_features2) * 0.01
    print(f"Initial weights hiden layer: {w2}")
    h1 = np.zeros((num_classes_layer_1, num_samples))
    h2 = np.zeros((num_classes_layer_2, num_samples))
    # for hidden layer
    v = np.zeros((num_classes_layer_2, num_samples))

    # gradient of output layer
    grad_layer_2 = np.zeros((num_classes_layer_2, num_samples))
    v = np.zeros((num_classes_layer_2, num_samples))

    def f_h1(j, u, w):
        h1[j, u] = 0.0
        for k in range(num_features1):
            h1[j, u] += w1[j, k] * x[u, k]
        return h1[j, u]

    def f_h2(i, u, w):
        h2[i, u] = 0.0
        for k in range(num_features2):
            h2[i, u] += w2[i, k] * v[u, k]
        return h2[i, u]

    # Gradient descent iterations
    for epoch in range(num_epochs):
        loss = 0.0
        for u in range(num_samples):
            # Forward step

            # Compute hidden layer
            for j in range(num_classes_layer_1):
                h1_iu = f_h1(j, u, w1)
                v[j, u] = g1(h1_iu)
            # Compute output layer

            for i in range(num_classes_layer_2):
                h2_iu = f_h2(i, u, w2)
                y2_iu = g2(h2_iu)
                # error of the i-th output on the u-th example
                error_iu_layer2 = y2_iu - y_one_hot[u, i]
                loss += error_iu_layer2**2  # compute square error
                gradient_component_layer2 = (
                    error_iu_layer2 * dg2(h2_iu) if dg2 else error_iu_layer2
                )
                # Udpate weights output layer:
                for j in range(num_features2):
                    w2[i, j] -= learning_rate * gradient_component_layer2 * v[u, j]
                    w2[i, j] += w2[i, j]

                grad_layer_2[i, u] = gradient_component_layer2

            # Backward step
            for j in range(num_classes_layer_1):
                h1_ju = f_h1(j, u, w2)
                sum_w2_ij = sum(
                    w2[i, j] * grad_layer_2[i, u] for i in range(num_classes_layer_2)
                )
                gradient_component_layer1 = dg1(h1_ju) * sum_w2_ij if dg1 else sum_w2_ij
                for k in range(num_features1):
                    w1[j, k] -= learning_rate * gradient_component_layer1 * x[u, k]

        loss *= 0.5
        if epoch % (num_epochs / 10) == 0:
            print(f"Iteration epoch: {epoch}, Loss: {loss:.16f}")

    return w1, w2


def pred(w1, w2, g1, g2, x):
    n_s = w1.shape[0]
    n_e = len(x)
    y = np.zeros(n_s)
    for i in range(n_s):
        h = 0.0
        for k in range(n_e):
            h += w2[i, k] * g1(w1[i, k] * x[k])
        y[i] = g2(h)
    return y


def plot_preds_scalar(x, w1, w2, g1, g2, title):
    num_samples, num_features = x.shape
    preds = np.zeros(num_samples)

    for m in range(num_samples):
        y = pred(w1, w2, g1, g2, x[m, :])
        preds[m] = np.argmax(y)

    colors = [{0: "b", 1: "g", 2: "r"}.get(p) for p in preds]
    # Visualize dots and classes
    plot(x, colors, title)


learning_rate = 0.02
num_epochs = 10000
x, y_one_hot, c = cloud(num_points_per_class=10)
plot(x, colors=c, title="Data set")


# ===========================================

# Train model
w1, w2 = gradient_descent(
    x, y_one_hot, relu, None, sigmoid, derivative_sigmoid, learning_rate, num_epochs
)
print(f"weights layer 1: {w1}")
print(f"weights layer 2: {w2}")
title = f"Clasificated dots with {sigmoid.__name__} activation function. Scalar version"
plot_preds_scalar(x, w1, w2, relu, sigmoid, title)
