import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataset import cloud, plot_decision_boundary_pytorch, plot_network_topology, plot




# Generar datos de ejemplo
def generate_data(n_samples=30):
    X, Y_one_hot = cloud(n_samples)
    return torch.FloatTensor(X), torch.LongTensor(np.argmax(Y_one_hot, axis=1))

# Multi Layer Perceptron
class Perceptron(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=[64, 64], output_size=3, learning_rate=0.01):
        super(Perceptron, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())  # Función de salida
        self.model = nn.Sequential(*layers)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def train(self, X, y, epochs=10000):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    def predict(self, grid):
        preds = self.model(grid).detach().numpy()
        preds = np.argmax(preds, axis=1)
        return preds




if __name__ == "__main__":
    # Inicializar el modelo, datos y entrenamiento
    learning_rate = 0.01
    epochs = 10000
    input_size = 2
    hidden_sizes = [4]
    output_size = 3
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    plot_network_topology(layer_sizes)
    X, Y_one_hot = generate_data(n_samples=30)
    model = Perceptron(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size, learning_rate=learning_rate)
    model.train(X, Y_one_hot, epochs=epochs)

    print("\nDatos antes del entrenamiento")
    print(Y_one_hot.tolist())
    predictions = model.predict(X).tolist()
    print("\nPredicciones despues del entrenamiento")
    print(predictions)
    #Evaluar el modelo con un gráfico
    plot(X, Y_one_hot.tolist(), predictions, title="Predicted Classes")

    plot_decision_boundary_pytorch(model, X, Y_one_hot, predictions)
