import numpy as np

from dataset import cloud, plot, plot_decision_boundary, plot_network_topology


# Función de activación Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivada de la función Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)


# Función Softmax para la capa de salida (normalización a probabilidades)
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Evitar desbordamiento numérico
    return exp_x / exp_x.sum(axis=1, keepdims=True)


# Función de coste: Entropía cruzada (Cross-Entropy)
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
    return np.sum(log_likelihood) / m


# compute square error
def square_error(y_true, y_pred):
    m = y_true.shape[0]
    error = y_true - y_pred
    loss = error**2
    return np.sum(loss) / m * 0.5


# Implementación de un perceptrón con dos capas (red neuronal)
class Perceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Inicializar pesos y sesgos
        self.weights_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))

        self.weights_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def forward(self, X):
        # Propagación hacia adelante (capa oculta)
        self.hidden_input = np.dot(X, self.weights_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)

        # Propagación hacia adelante (capa de salida)
        self.output_input = (
            np.dot(self.hidden_output, self.weights_output) + self.bias_output
        )
        self.output = softmax(self.output_input)

        return self.output

    def backward(self, X, y):
        # Número de ejemplos
        m = X.shape[0]

        # Propagación hacia atrás para la capa de salida
        output_error = self.output - y
        # Derivada de la cross-entropy + softmax combinada
        output_delta = output_error / m

        # Propagación hacia atrás para la capa oculta
        hidden_error = output_delta.dot(self.weights_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Actualización de los pesos y sesgos de la capa de salida
        self.weights_output -= (
            self.hidden_output.T.dot(output_delta) * self.learning_rate
        )
        self.bias_output -= (
            np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        )

        # Actualización de los pesos y sesgos de la capa oculta
        self.weights_hidden -= X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden -= (
            np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
        )

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Propagación hacia adelante
            output = self.forward(X)

            # Retropropagación
            self.backward(X, y)

            # Calcular y mostrar el error cada 100 iteraciones
            if epoch % 1000 == 0:
                loss = square_error(y, output)
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


if __name__ == "__main__":
    # parametros:
    num_points_per_class = 10
    input_size = 2
    hidden_size = 4
    output_size = 3
    learning_rate = 0.01
    epochs = 10000
    layer_sizes = [input_size] + [hidden_size] + [output_size]
    plot_network_topology(layer_sizes)

    X, y_one_hot = cloud(num_points_per_class)
    # Crear el modelo de perceptrón
    model = Perceptron(input_size, hidden_size, output_size, learning_rate)
    # Entrenar el modelo
    model.train(X, y_one_hot, epochs)

    print("\nDatos antes del entrenamiento:")
    print(np.argmax(y_one_hot, axis=1))
    # Predicciones del modelo después del entrenamiento
    predictions = model.predict(X)
    print("\nPredicciones después del entrenamiento:")
    print(predictions)
    # Evaluar el modelo con un gráfico
    plot(X, y_one_hot, predictions, title="Predicted Classes")
    plot_decision_boundary(
        model,
        X,
        y_one_hot,
        predictions,
        title="Decision Boundary and Predicted Classes",
    )
