import numpy as np

from dataset import cloud, plot


def relu_vectorial(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return np.where(z > 0, 1, 0)


# Coss Function (cross entropy)
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
X, y_one_hot = cloud(num_points_per_class=10)

# train the model
weights, biases = gradient_descent(
    X, y_one_hot, relu_vectorial, learning_rate, num_epochs
)
print(f"Final weights: {weights}")
plot(X, y_one_hot, title="Data set")
title = f"Dots classificated with {relu_vectorial.__name__} model"
plot_preds_chat_gpt(X, y_one_hot, weights, relu_vectorial, title, biases)
