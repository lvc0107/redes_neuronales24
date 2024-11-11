import matplotlib.pyplot as plt
import numpy as np

from dataset import map_color, plot_network_topology


def plot_decision_boundary(
    X, y, preds, weights=None, biases=None, title="Decision Boundary"
):
    # Crear un grid de puntos para evaluar las predicciones
    # en todo el espacio de entrada
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Y = np.argmax(y, axis=1)
    # Obtener las predicciones para cada punto del grid
    preds_for_boundary = predict(grid, weights, biases)

    Z = preds_for_boundary.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=map_color(preds),
        edgecolors=map_color(Y),
        linewidths=2,
        s=200,
        alpha=0.5,
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def relu(h):
    return np.maximum(0, h)


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


def predict(X, weights, biases):
    h = np.dot(X, weights) + biases
    activated = relu(h)
    exp_h = np.exp(activated - np.max(activated, axis=1, keepdims=True))
    preds = np.argmax(exp_h / exp_h.sum(axis=1, keepdims=True), axis=1)
    return preds


if __name__ == "__main__":
    """
    Se intenta clasificar al XOR pero no se va a poder
    por que con una red neuronal con una sola capa de salida
    no alcanza para aprender la separacion lineal
    Ver Wikipedia: https://es.wikipedia.org/wiki/Perceptr%C3%B3n
    """

    # Esta red tiene solo 2 entradas(no son neuronas) y 2 neuronas de salida
    plot_network_topology(layer_sizes=[2, 2])

    e1 = (0, 0)
    e2 = (0, 1)
    e3 = (1, 0)
    e4 = (1, 1)

    X = np.array([e1, e2, e3, e4])
    colors = ["b", "g", "g", "b"]

    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200)
    plt.title("Dots before run Perceptron")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # expected results
    Y_one_hot = np.array([(1, 0), (0, 1), (0, 1), (1, 0)])

    eta = 0.02
    epoch = 1000

    # Train model
    """
    la entrada del modelo es:

      X   Y_one_hot (lo que se espera)
     0 0    1 0
     0 1    0 1
     1 0    0 1
     1 1    1 0
    """

    weights, biases = gradient_descent(
        X, Y_one_hot, learning_rate=eta, num_iterations=epoch
    )

    # Obtener las predicciones
    preds = predict(X, weights, biases)
    print("\nDatos antes del entrenamiento:")
    # Arg max devuelve el indice del maximo valor.
    # En este caso el indice donde el valor es 1
    print(np.argmax(Y_one_hot, axis=1))
    print("\nPredicciones despues del entrenamiento:")
    print(preds)

    plot_decision_boundary(X, Y_one_hot, preds, weights, biases)
