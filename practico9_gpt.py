import numpy as np

from dataset import cloud, plot, plot_network_topology


def relu_vectorial(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return np.where(z > 0, 1, 0)


# Coss Function (cross entropy)
# Usamos esta funcion en lugar del error medio cuadrado por que
# es lo que se recomendo en el teorico. Ademas de que chatgpt tambien
# propone esta funcion: Entropia Cruzada
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))


def gradient_descent(x, y_one_hot, g, dg=None, learning_rate=0.02, num_epochs=10000):
    num_samples, num_features = x.shape
    num_classes = y_one_hot.shape[1]

    # Set weights
    weights = np.random.randn(num_features, num_classes) * 0.01
    print(f"Initial weights: {weights}")
    biases = np.zeros((1, num_classes))

    for epoch in range(num_epochs):
        # Compute logits
        logits = np.dot(x, weights) + biases
        # Apply ReLU
        activated = g(logits)

        # biases probabilities using softmax
        exp_logits = np.exp(activated - np.max(activated, axis=1, keepdims=True))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Compute loss
        loss = cross_entropy_loss(y_one_hot, probabilities)

        # Compute gradients
        d_logits = probabilities - y_one_hot
        dW = np.dot(x.T, d_logits) / num_samples
        db = np.sum(d_logits, axis=0, keepdims=True) / num_samples

        # Updates weights and biases
        weights -= learning_rate * dW
        biases -= learning_rate * db

        if epoch % (num_epochs / 10) == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss:.4f}")

    return weights, biases


def plot_preds_chat_gpt(x, y_one_hot, weights, relu_vectorial, title, biases=None):
    if biases is not None:
        logits = np.dot(x, weights) + biases
    else:
        logits = np.dot(x, weights)
    activated = relu_vectorial(logits)
    exp_logits = np.exp(activated - np.max(activated, axis=1, keepdims=True))
    preds = np.argmax(exp_logits / exp_logits.sum(axis=1, keepdims=True), axis=1)
    # Visualize dots and classes
    plot(x, y_one_hot, preds, title)


learning_rate = 0.02
num_epochs = 100000
# Datos para el entrenamiento
# Genera 3 nubes con 30/3 puntos por cada nube
# cambiar a gusto
X, y_one_hot = cloud(n_samples=30)

"""
 X        J (lo que entra)     O (lo que se va calculando en cada epoca)
2 3     1 0 0                 0 0 1
3 5     1 0 0                 0 0 1
4 5     0 1 0                 0 0 1
6 7     0 1 0                 0 1 0
20 34   1 0 0                 0 1 0
.         .                     .
.         .                     .
.         .                     .
.        1 0 0                0 0 1


cuando la diferencia entre lo que entra y lo que se va generando es cada vez mas
chica es por que se esta entrenando bien y el error se vuelve mas chico

Si la losss funcion se estanca en un valor arriba de 0 es por que el gradiente
cayo en un minimo local y se quedo estancado ahi para todas las vueltas o
iteraciones que se hacen en cada epoca. Seguramente va a clasificar mal
"""


plot_network_topology(layer_sizes=[2, 3])
# train the model
# Devuelve los pesos sinapticos y los ubrales por separado.
# en una implementacion mejor va todo junto
weights, biases = gradient_descent(
    X, y_one_hot, relu_vectorial, learning_rate, num_epochs
)
print(f"Final weights: {weights}")
plot(X, y_one_hot, title="Data set")
title = f"Dots classificated with {relu_vectorial.__name__} model"

# con los pesos obtenidos por el descenso del gradiente calculamos una vez mas
# para obtener las salidas predecidas (udnado la misma funcion de activacion, en este caso la Relu)
# y las ploteamos
plot_preds_chat_gpt(X, y_one_hot, weights, relu_vectorial, title, biases)
