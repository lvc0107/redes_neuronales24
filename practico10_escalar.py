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
    # for this example num_classes_layers are the same for both layers
    num_samples, num_features = x.shape
    num_classes_layer_1 = y_one_hot.shape[1]
    num_classes_layer_2 = y_one_hot.shape[1]

    # Set weights
    w1 = np.random.randn(num_classes_layer_1, num_features) * 0.01
    print(f"Initial weights first layer: {w1}")
    w2 = np.random.randn(num_classes_layer_2, num_classes_layer_1) * 0.01
    print(f"Initial weights hidden layer: {w2}")
    h1 = np.zeros((num_samples, num_classes_layer_1))
    h2 = np.zeros((num_samples, num_classes_layer_2))
    # for hidden layer
    v = np.zeros((num_samples, num_classes_layer_1))
    # gradient of output layer
    grad_layer_2 = np.zeros((num_samples, num_classes_layer_2))

    def f_h1(u, j, w1):
        h1[u, j] = 0.0
        for k in range(num_features):
            h1[u, j] += w1[j, k] * x[u, k]
        return h1[u, j]

    def f_h2(u, i, w2):
        h2[u, i] = 0.0
        for j in range(num_classes_layer_1):
            h2[u, i] += w2[i, j] * v[u, j]
        return h2[u, i]

    # Gradient descent iterations
    aggregated_loss = []
    for epoch in range(num_epochs):
        loss = 0.0
        for u in range(num_samples):
            # Forward step

            # Compute hidden layer
            for j in range(num_classes_layer_1):
                h1_uj = f_h1(u, j, w1)
                v[u, j] = g1(h1_uj)
            # Compute output layer
            for i in range(num_classes_layer_2):
                h2_ui = f_h2(u, i, w2)
                y2_ui = g2(h2_ui)
                # error of the i-th output on the u-th example
                error_ui_layer2 = y2_ui - y_one_hot[u, i]
                loss += error_ui_layer2**2  # compute square error
                grad_layer_2[u, i] = dg2(h2_ui) * error_ui_layer2
                # Udpate weights output layer:
                for j in range(num_classes_layer_1):
                    w2[i, j] -= learning_rate * grad_layer_2[u, i] * v[u, j]

            # Backward step. Updating  weights on first layer
            for j in range(num_classes_layer_1):
                h1_uj = f_h1(u, j, w1)
                sum_w2_ij = sum(
                    w2[i, j] * grad_layer_2[u, i] for i in range(num_classes_layer_2)
                )
                gradient_component_layer1 = (
                    dg1(h1_uj) * sum_w2_ij if dg1 else h1_uj * sum_w2_ij
                )
                for k in range(num_features):
                    w1[j, k] -= learning_rate * gradient_component_layer1 * x[u, k]

        loss *= 0.5
        aggregated_loss.append(loss)
        if epoch % (num_epochs / 10) == 0:
            print(f"Iteration epoch: {epoch}, Loss: {loss:.16f}")

    return w1, w2, aggregated_loss


def pred(w1, w2, g1, g2, x, y_one_hot):
    num_features = len(x)
    num_classes_layer_1 = y_one_hot.shape[1]
    num_classes_layer_2 = y_one_hot.shape[1]

    y = np.zeros(num_classes_layer_2)
    v = np.zeros(num_classes_layer_1)

    for j in range(num_classes_layer_1):
        h1 = 0.0
        for k in range(num_features):
            h1 += w1[j, k] * x[k]
        v[j] = g1(h1)

    for i in range(num_classes_layer_2):
        h2 = 0.0
        for j in range(num_classes_layer_1):
            h2 += w2[i, j] * v[j]

        y[i] = g2(h2)
    return y


def plot_preds_scalar(x, y_one_hot, c, w1, w2, g1, g2, title):
    num_samples, _ = x.shape
    preds = np.zeros(num_samples)

    for m in range(num_samples):
        y = pred(w1, w2, g1, g2, x[m, :], y_one_hot)
        preds[m] = np.argmax(y)

    new_clasification = [{0: "b", 1: "g", 2: "r"}.get(p) for p in preds]
    # Visualize dots and classes
    plot(x, c, new_clasification, title)


learning_rate = 0.02
num_epochs = 10000
x, y_one_hot, c = cloud(num_points_per_class=10)
plot(x, c, c, title="Data set")


# # ===========================================

# # Train model
# w1, w2, aggregated_loss = gradient_descent(
#     x,
#     y_one_hot,
#     relu,
#     derivative_relu,
#     sigmoid,
#     derivative_sigmoid,
#     learning_rate,
#     num_epochs,
# )
# print(f"weights layer 1: {w1}")
# print(f"weights layer 2: {w2}")
# title = "Clasificated dots with 2 layers. Scalar version"
# plot_preds_scalar(x, y_one_hot, c, w1, w2, relu, sigmoid, title)
# plot_loss(range(num_epochs), aggregated_loss)
