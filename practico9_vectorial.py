import numpy as np

from dataset import cloud, plot, plot_decision_boundary, plot_network_topology


def relu(h):
    return np.maximum(0, h)


def relu_derivative(h):
    return np.where(h > 0, 1, 0)


class Perceptron:
    def gradient_descent(self, x, s, learning_rate=0.02, epochs=1000):
        num_samples, num_features = x.shape
        num_classes = s.shape[1]

        # Set weights
        weights = np.random.randn(num_features, num_classes) * 0.01
        # Gradient descent iterations
        for _ in range(epochs):
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

    def predict(self, X, weights, biases=None):
        h = np.dot(X, weights)
        activated = relu(h)
        exp_h = np.exp(activated - np.max(activated, axis=1, keepdims=True))
        preds = np.argmax(exp_h / exp_h.sum(axis=1, keepdims=True), axis=1)
        return preds


if __name__ == "__main__":
    learning_rate = 0.02
    epoch = 10000
    plot_network_topology(layer_sizes=[3, 2])
    X, Y_one_hot = cloud(n_samples=30)
    # Train model
    model = Perceptron()
    weights = model.gradient_descent(
        X, Y_one_hot, learning_rate=learning_rate, epochs=epoch
    )
    # evaluar con los nuevos pesos sinapticos
    preds = model.predict(X, weights)

    print("\nDatos antes del entrenamiento:")
    print(np.argmax(Y_one_hot, axis=1))
    # Predicciones del modelo después del entrenamiento
    print("\nPredicciones después del entrenamiento:")
    print(preds)
    # Evaluar el modelo con un gráfico
    plot(X, Y_one_hot, preds, title="Predicted Classes")
    plot_decision_boundary(
        model,
        X,
        Y_one_hot,
        preds,
        weights,
        None,
        title="Decision Boundary and Predicted Classes",
    )
