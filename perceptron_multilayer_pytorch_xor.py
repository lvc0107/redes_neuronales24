"""
Prueba para XOR.

"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import plot, plot_network_topology
from perceptron_multilayer_pytorch import Perceptron, plot_decision_boundary_pytorch


# Generar datos de ejemplo
def generate_data():
    # Xor
    e1 = (0, 0)
    e2 = (0, 1)
    e3 = (1, 0)
    e4 = (1, 1)

    X = np.array([e1, e2, e3, e4])
    # expected results
    Y_one_hot = np.array([(1, 0), (0, 1), (0, 1), (1, 0)])
    return torch.FloatTensor(X), torch.LongTensor(np.argmax(Y_one_hot, axis=1))


if __name__ == "__main__":
    learning_rate = 0.01
    epochs = 10000
    input_size = 2
    hidden_sizes = [3]
    output_size = 2
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    plot_network_topology(layer_sizes)

    colors = ["b", "r", "r", "b"]
    X, Y_one_hot = generate_data()
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200)
    plt.title("Dots before run Perceptron")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    model = Perceptron(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        learning_rate=learning_rate,
    )
    model.train(X, Y_one_hot, epochs=epochs)

    print("\nDatos antes del entrenamiento")
    print(Y_one_hot.tolist())
    predictions = model.predict(X).tolist()
    print("\nPredicciones despues del entrenamiento")
    print(predictions)
    # Evaluar el modelo con un gráfico
    plot(X, Y_one_hot.tolist(), predictions, title="Predicted Classes")
    plot_decision_boundary_pytorch(model, X, Y_one_hot, predictions)
