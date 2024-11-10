import matplotlib.pyplot as plt
import numpy as np

from practico10_gpt import (
    Perceptron,
    plot,
    plot_decision_boundary,
    plot_network_topology,
)

"""
Se espera que no funcione pues

La función XOR no puede ser aprendida por un único perceptrón puesto
que requiere al menos de dos líneas para separar las clases (0 y 1).
Debe utilizarse al menos una capa adicional de perceptrones
para permitir su aprendizaje.
https://es.wikipedia.org/wiki/Perceptr%C3%B3n

"""


if __name__ == "__main__":
    # parametros:
    num_points_per_class = 10
    input_size = 2
    hidden_size = 2
    output_size = 2
    learning_rate = 0.01
    epochs = 10000
    plot_network_topology(layer_sizes=[2, 2, 2])
    # Xor
    e1 = (0, 0)
    e2 = (0, 1)
    e3 = (1, 0)
    e4 = (1, 1)

    X = np.array([e1, e2, e3, e4])
    colors = ["b", "r", "r", "b"]

    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200)
    plt.title("Dots before run Perceptron")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # expected results
    y_one_hot = np.array([(1, 0), (0, 1), (0, 1), (1, 0)])

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
