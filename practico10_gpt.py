import matplotlib.pyplot as plt
import numpy as np

from dataset import cloud, plot


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
                loss = cross_entropy_loss(y, output)
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


def plot_decision_boundary(model, X, y, c, title="Decision Boundary"):
    def map_color(d):
        return [{0: "b", 1: "g", 2: "r"}.get(p) for p in d]

    # Crear un grid de puntos para evaluar las predicciones en todo el espacio de entrada
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Obtener las predicciones para cada punto del grid
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Crear el gráfico
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.viridis)
    plt.colorbar()
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=np.argmax(y, axis=1),
        edgecolors=map_color(c),
        s=200,
        alpha=0.5,
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# Ejemplo de uso
if __name__ == "__main__":
    X, y_one_hot = cloud(num_points_per_class=10)
    # print original_classification
    original_classification = np.argmax(y_one_hot, axis=1)
    plot(X, original_classification, title="Data set")

    # Crear el modelo de perceptrón
    model = Perceptron(input_size=2, hidden_size=4, output_size=3, learning_rate=0.1)

    # Entrenar el modelo
    model.train(X, y_one_hot, epochs=10000)

    print("\nDatos antes del entrenamiento:")
    print(np.argmax(y_one_hot, axis=1))
    # Predicciones del modelo después del entrenamiento
    predictions = model.predict(X)
    print("\nPredicciones después del entrenamiento:")
    print(predictions)
    # Evaluar el modelo con un gráfico
    plot(X, original_classification, predictions, title="Predicted Classes")
    plot_decision_boundary(
        model,
        X,
        y_one_hot,
        original_classification,
        title="Decision Boundary and Predicted Classes",
    )
