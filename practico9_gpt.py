import matplotlib.pyplot as plt
import numpy as np

# Generar un dataset de ejemplo
np.random.seed(0)
num_points_per_class = 10

# Clase 0
x0 = np.random.randn(num_points_per_class, 2) + np.array([0, 0])
y0 = np.zeros(num_points_per_class)

# Clase 1
x1 = np.random.randn(num_points_per_class, 2) + np.array([4, 4])
y1 = np.ones(num_points_per_class)

# Clase 2
x2 = np.random.randn(num_points_per_class, 2) + np.array([0, 4])
y2 = np.ones(num_points_per_class) * 2

# Combinar el dataset
X = np.vstack([x0, x1, x2])
y = np.hstack([y0, y1, y2])


# Función ReLU
def relu(z):
    return np.maximum(0, z)


# Derivada de la función ReLU
def relu_derivative(z):
    return np.where(z > 0, 1, 0)


# Función de costo (entropía cruzada)
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))


# Implementación del descenso por el gradiente
def gradient_descent(X, y, learning_rate=0.02, num_iterations=1000):
    num_samples, num_features = X.shape
    num_classes = len(np.unique(y))

    # One-hot encoding de las etiquetas
    y_one_hot = np.eye(num_classes)[y.astype(int)]

    # Inicializar pesos y sesgos
    weights = np.random.randn(num_features, num_classes) * 0.01
    biases = np.zeros((1, num_classes))

    # Iteraciones del descenso por el gradiente
    for i in range(num_iterations):
        # Calcular logits
        logits = np.dot(X, weights) + biases
        # Aplicar ReLU
        activated = relu(logits)

        # Calcular probabilidades usando softmax
        exp_logits = np.exp(activated - np.max(activated, axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Calcular la pérdida
        loss = cross_entropy_loss(y_one_hot, probabilities)

        # Calcular los gradientes
        d_logits = probabilities - y_one_hot
        dW = np.dot(X.T, d_logits) / num_samples
        db = np.sum(d_logits, axis=0, keepdims=True) / num_samples

        # Actualizar pesos y sesgos
        weights -= learning_rate * dW
        biases -= learning_rate * db

        # Imprimir la pérdida cada 100 iteraciones
        if i % 100 == 0:
            print(f"Iteración {i}, Pérdida: {loss:.4f}")

    return weights, biases


# Entrenar el modelo
weights, biases = gradient_descent(X, y)

# Visualizar puntos y clases
logits = np.dot(X, weights) + biases
activated = relu(logits)
exp_logits = np.exp(activated - np.max(activated, axis=1, keepdims=True))
preds = np.argmax(exp_logits / exp_logits.sum(axis=1, keepdims=True), axis=1)

plt.scatter(X[:, 0], X[:, 1], c=preds, cmap="viridis", alpha=0.5)
plt.title("Puntos clasificados por modelo con ReLU")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
