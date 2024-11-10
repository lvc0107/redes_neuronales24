"""
Implementar un perceptron multicapa (varias capas ocultas)  usando pytorch con
descenso por gradiente, utilizando la funci—n ReLU para las capas ocultas y
la funci—n sigmoide para la capa de salida.
Las entradas son pares de puntos  agrupados en tres clases y
la salida son 3 neuronas que representan una clasificacion one_hot de orden 3x3.
Implementar una funcion de evaluacion que asigne un color a cada prediccion.
Graficar los puntos y la dessicion boundary.
Generar una imagen de la topologia de la red
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from dataset import cloud


# Generar datos de ejemplo
def generate_data(n_samples=30):

    X, Y_one_hot = cloud(n_samples)
    X = np.random.rand(n_samples, 2) * 2 - 1  # Puntos en el rango [-1, 1]
    y = np.zeros((n_samples, 3))
    for i, (x1, x2) in enumerate(X):
        if x1 ** 2 + x2 ** 2 < 0.5:  # Clase 0
            y[i, 0] = 1
        elif x1 > 0 and x2 > 0:  # Clase 1
            y[i, 1] = 1
        else:  # Clase 2
            y[i, 2] = 1
    return torch.FloatTensor(X), torch.LongTensor(np.argmax(y, axis=1))

# Definir el modelo MLP
class MLP(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=[64, 64], output_size=3):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())  # Función de salida
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    # Función para entrenar el modelo
    def train_model(self, X, y, epochs=1000, learning_rate=0.01):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

    def predict(self, grid):
        preds = self.model(grid).detach().numpy()
        preds = np.argmax(preds, axis=1)
        return preds


# Función para evaluar y graficar la frontera de decisión
def plot_decision_boundary2(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    preds = model.predict(grid)
    plt.contourf(xx, yy, preds.reshape(xx.shape), alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y.numpy(), cmap='viridis', edgecolor='k')
    plt.show()

# Inicializar el modelo, datos y entrenamiento
X, y = generate_data()
model = MLP()
model.train_model(X, y)
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
preds = model(grid).detach().numpy()
preds = np.argmax(preds, axis=1)

plot_decision_boundary2(model, X, y)
