import matplotlib.pyplot as plt
import numpy as np

from dataset import plot_decision_boundary, plot_network_topology
from perceptron_multilayer import Perceptron

"""
Prueba para XOR. Anda maso menos
"""


if __name__ == "__main__":
    learning_rate = 0.01
    epochs = 50000
    input_size = 2
    hidden_sizes = [10]
    output_size = 2
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    plot_network_topology(layer_sizes)
    # Xor
    e1 = (0, 0)
    e2 = (0, 1)
    e3 = (1, 0)
    e4 = (1, 1)

    X = np.array([e1, e2, e3, e4])
    # expected results
    Y_one_hot = np.array([(1, 0), (0, 1), (0, 1), (1, 0)])
    colors = ["b", "g", "g", "b"]

    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200)
    plt.title("Dots before run Perceptron")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    model = Perceptron(layer_sizes, learning_rate)
    weights, biases = model.train(X, Y_one_hot, epochs=epochs)

    print("\nDatos antes del entrenamiento:")
    print(np.argmax(Y_one_hot, axis=1))
    predictions = model.predict(X, weights, biases)
    print("\nPredicciones despues del entrenamiento:")
    print(predictions)
    plot_decision_boundary(
        model,
        X,
        Y_one_hot,
        predictions,
        weights,
        biases,
    )
