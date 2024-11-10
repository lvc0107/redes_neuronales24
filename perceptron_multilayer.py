"""
Implementar un perceptrón multicapa (varias capas ocultas) en Python con
descenso por gradiente, utilizando la función ReLU para las capas ocultas y
la función sigmoide para la capa de salida.
Las entradas son pares de puntos  agrupados en tres clases y
la salida son 3 neuronas que representan una clasificacion one_hot de orden 3x3.
Implementar una funcion de evaluacion que asigne un color a cada prediccion.
Graficar los puntos y la dessicion boundary.
Generar una imagen de la topologia de la red
"""


import numpy as np

from dataset import cloud, plot, plot_decision_boundary, plot_network_topology


# Función ReLU y su derivada
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# Función sigmoide y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
    return np.sum(log_likelihood) / m


class Perceptron:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate

    # Función de inicialización de pesos
    def initialize_weights(self, layer_sizes):
        weights = []
        biases = []
        for i in range(1, len(layer_sizes)):
            weights.append(np.random.randn(layer_sizes[i - 1], layer_sizes[i]) * 0.01)
            biases.append(np.zeros((1, layer_sizes[i])))
        return weights, biases

    # Implementación del forward pass
    def forward_propagation(self, X, weights, biases):
        activations = [X]
        inputs = []

        for i in range(len(weights) - 1):
            z = np.dot(activations[-1], weights[i]) + biases[i]
            inputs.append(z)
            activations.append(relu(z))

        # Última capa con función sigmoide
        z = np.dot(activations[-1], weights[-1]) + biases[-1]
        inputs.append(z)
        activations.append(sigmoid(z))

        return activations, inputs

    # Implementación del backpropagation
    def backward_propagation(self, Y, activations, inputs, weights):
        grads_w = [None] * len(weights)
        grads_b = [None] * len(weights)

        # Cálculo del error en la capa de salida
        delta = activations[-1] - Y
        grads_w[-1] = np.dot(activations[-2].T, delta) / Y.shape[0]
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True) / Y.shape[0]

        # Backpropagation para capas ocultas
        for i in reversed(range(len(weights) - 1)):
            delta = np.dot(delta, weights[i + 1].T) * relu_derivative(inputs[i])
            grads_w[i] = np.dot(activations[i].T, delta) / Y.shape[0]
            grads_b[i] = np.sum(delta, axis=0, keepdims=True) / Y.shape[0]

        return grads_w, grads_b

    # Función de actualización de pesos
    def update_weights(self, weights, biases, grads_w, grads_b, learning_rate):
        for i in range(len(weights)):
            weights[i] -= learning_rate * grads_w[i]
            biases[i] -= learning_rate * grads_b[i]
        return weights, biases

    # Función de entrenamiento
    def train(self, X, Y, layer_sizes, epochs, learning_rate):
        weights, biases = self.initialize_weights(layer_sizes)
        for epoch in range(epochs):
            activations, inputs = self.forward_propagation(X, weights, biases)
            grads_w, grads_b = self.backward_propagation(
                Y, activations, inputs, weights
            )
            weights, biases = self.update_weights(
                weights, biases, grads_w, grads_b, learning_rate
            )

            # Cálculo de pérdida (opcional para monitoreo)
            if epoch % 1000 == 0:
                loss = cross_entropy_loss(Y, activations[-1])
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
        return weights, biases

    # Función de predicción y evaluación
    def predict(self, X, weights, biases):
        activations, _ = self.forward_propagation(X, weights, biases)
        return np.argmax(activations[-1], axis=1)


if __name__ == "__main__":
    learning_rate = 0.01
    epochs = 10000
    input_size = 2
    hidden_sizes = [4, 4]
    output_size = 3
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    plot_network_topology(layer_sizes)

    model = Perceptron(input_size, hidden_sizes, output_size, learning_rate)

    X, Y_one_hot = cloud(n_samples=30)

    weights, biases = model.train(
        X, Y_one_hot, layer_sizes, epochs=epochs, learning_rate=learning_rate
    )

    print("\nDatos antes del entrenamiento:")
    print(np.argmax(Y_one_hot, axis=1))
    # Predicciones del modelo después del entrenamiento
    predictions = model.predict(X, weights, biases)
    print("\nPredicciones después del entrenamiento:")
    print(predictions)
    # Evaluar el modelo con un gráfico
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
