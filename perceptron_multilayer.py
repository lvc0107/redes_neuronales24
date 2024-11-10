import numpy as np

from dataset import cloud, plot, plot_decision_boundary, plot_network_topology

"""
mplementar un perceptrón multicapa (varias capas ocultas) en Python con descenso
por gradiente, utilizando la función ReLU para las capas ocultas y
la función sigmoide para la capa de salida.
Las entradas son pares de puntos agrupados en tres clases y la salida
son 3 neuronas que representan una clasificación one-hot de orden 3x3.
Implementar una función de evaluación que asigne un color a cada predicción.
Graficar los puntos y la decisión boundary.
Generar una imagen de la topología de la red.
"""



def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
    return np.sum(log_likelihood) / m


class Perceptron:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.layer_sizes = layer_sizes

    # Funci—n de inicializaci—n de pesos
    def initialize_weights(self):
        weights = []
        biases = []
        for i in range(1, len(self.layer_sizes)):
            weights.append(np.random.randn(self.layer_sizes[i - 1], self.layer_sizes[i]) * 0.01)
            biases.append(np.zeros((1, self.layer_sizes[i])))
        return weights, biases

    # Implementaci—n del forward pass
    def forward_propagation(self, X, weights, biases):
        activations = [X]
        inputs = []

        for i in range(len(weights) - 1):
            z = np.dot(activations[-1], weights[i]) + biases[i]
            inputs.append(z)
            activations.append(relu(z))

        # òltima capa con funci—n sigmoide
        z = np.dot(activations[-1], weights[-1]) + biases[-1]
        inputs.append(z)
        activations.append(sigmoid(z))

        return activations, inputs

    def backward_propagation(self, Y, activations, inputs, weights):
        grads_w = [None] * len(weights)
        grads_b = [None] * len(weights)

        # C‡lculo del error en la capa de salida
        delta = activations[-1] - Y
        grads_w[-1] = np.dot(activations[-2].T, delta) / Y.shape[0]
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True) / Y.shape[0]

        # Backpropagation para capas ocultas
        for i in reversed(range(len(weights) - 1)):
            delta = np.dot(delta, weights[i + 1].T) * relu_derivative(inputs[i])
            grads_w[i] = np.dot(activations[i].T, delta) / Y.shape[0]
            grads_b[i] = np.sum(delta, axis=0, keepdims=True) / Y.shape[0]

        return grads_w, grads_b

    def update_weights(self, weights, biases, grads_w, grads_b):
        for i in range(len(weights)):
            weights[i] -= self.learning_rate * grads_w[i]
            biases[i] -= self.learning_rate * grads_b[i]
        return weights, biases

    def train(self, X, Y, epochs):
        weights, biases = self.initialize_weights()
        for epoch in range(epochs):
            activations, inputs = self.forward_propagation(X, weights, biases)
            grads_w, grads_b = self.backward_propagation(
                Y, activations, inputs, weights
            )
            weights, biases = self.update_weights(
                weights, biases, grads_w, grads_b
            )

            if epoch % 1000 == 0:
                loss = cross_entropy_loss(Y, activations[-1])
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
        return weights, biases

    # Funci—n de predicci—n y evaluaci—n
    def predict(self, X, weights, biases):
        activations, _ = self.forward_propagation(X, weights, biases)
        return np.argmax(activations[-1], axis=1)


if __name__ == "__main__":
    learning_rate = 0.01
    epochs = 10000
    input_size = 2
    hidden_sizes = [4]
    output_size = 3
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    plot_network_topology(layer_sizes)
    X, Y_one_hot = cloud(n_samples=30)

    model = Perceptron(layer_sizes, learning_rate)
    weights, biases = model.train(X, Y_one_hot, epochs=epochs)

    print("\nDatos antes del entrenamiento:")
    print(np.argmax(Y_one_hot, axis=1))
    predictions = model.predict(X, weights, biases)
    print("\nPredicciones despues del entrenamiento:")
    print(predictions)
    plot(X, Y_one_hot, predictions, title="Predicted Classes")
    plot_decision_boundary(
        model,
        X,
        Y_one_hot,
        predictions,
        weights,
        biases,
        title="Decision Boundary and Predicted Classes",
    )
