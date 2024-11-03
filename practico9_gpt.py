import numpy as np

from dataset import plot, x, y_one_hot

plot()


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return np.where(z > 0, 1, 0)


# Función de costo (entropía cruzada)
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))


# Implementación del descenso por el gradiente
def gradient_descent(x, y_one_hot, learning_rate=0.02, num_epochs=10000):
    num_samples, num_features = x.shape
    num_classes = y_one_hot.shape[1]

    # Inicializar pesos y sesgos
    weights = np.random.randn(num_features, num_classes) * 0.01
    print(f"Initial weights: {weights}")
    biases = np.zeros((1, num_classes))

    # Iteraciones del descenso por el gradiente
    for epoch in range(num_epochs):
        # Calcular logits
        logits = np.dot(x, weights) + biases
        # Aplicar ReLU
        activated = relu(logits)

        # Calcular probabilidades usando softmax
        exp_logits = np.exp(activated - np.max(activated, axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Calcular la pérdida
        loss = cross_entropy_loss(y_one_hot, probabilities)

        # Calcular los gradientes
        d_logits = probabilities - y_one_hot
        dW = np.dot(x.T, d_logits) / num_samples
        db = np.sum(d_logits, axis=0, keepdims=True) / num_samples

        # Actualizar pesos y sesgos
        weights -= learning_rate * dW
        biases -= learning_rate * db

        # Imprimir la pérdida cada 100 iteraciones
        if epoch % 1000 == 0:
            print(f"Iteration epoch: {epoch}, Loss: {loss:.4f}")

    return weights, biases


# train the model
weights, biases = gradient_descent(x, y_one_hot)
print(f"Final weights: {weights}")


# Visualize dots and classes
logits = np.dot(x, weights) + biases
activated = relu(logits)
exp_logits = np.exp(activated - np.max(activated, axis=1, keepdims=True))
preds = np.argmax(exp_logits / exp_logits.sum(axis=1, keepdims=True), axis=1)


plot(preds)
