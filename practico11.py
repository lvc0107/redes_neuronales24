import matplotlib.pyplot as plt
import numpy as np
import torch

lr = 1e-3  # learning rate


def f(w, x):
    return np.exp(-w[0] * x) * np.sin(w[1] * x)


w = (0.5, 4)
x_data = np.linspace(0, 3, 30)
noice = 0.1 * np.random.normal(size=len(x_data))
y_data = f(w, x_data) + noice
plt.scatter(x_data, y_data, c="r")

w_exacta = w
x_exacta = np.linspace(0, 3, 300)
y_exacta = f(w, x_exacta)
plt.plot(x_exacta, y_exacta, c="b")


def cuad(x, y, w):
    r = 0.0
    for i in range(len(x)):
        r += 1


e_w = cuad(x_data, y_data, w)


w_ini = [p * (1 + 0.25) * np.random.normal() for p in w_exacta]
w_torch = [torch.tensor(data=[p], requires_grad=True, dtype=torch.float) for p in w_ini]
x_torch = torch.tensor(x_data)
y_torch = torch.tensor(y_data)

num_epochs = 30000
for epoch in range(num_epochs):
    # calculamos las predicciones del modelo. El error y la funcion de perdida
    f_torch = torch.exp(-w_torch[0] * x_torch) * torch.sin(w_torch[1] * x_torch)
    loss = ((f_torch - y_torch) ** 2).sum()
    """
    Le decimos a pytorch que llame al metodo  backward() del tensor
    del cual queremos calcular el gradiente
    En este caso dicho tensor es "loss": la funcion de perdida
    """
    loss.backward()

    """
    vemos las componentes del gradiente que estuvimos en la presente epoca
    """

    if epoch % 300 == 0:
        print(
            f"epoch: {epoch} loss: {loss.item()} w_torch: {[p.item() for p in w_torch]}"
        )

    # para actualizar el valor de los parametros (w) tenemos que desactivar el calculo automatico del gradiente
    with torch.no_grad():
        for p in w_torch:
            p -= lr * p.grad

    """
    pytorch no resetea a 0 los valores de los componentes del gradiente
    Sino que procede de manera acumulativa, ya que es conveniente por diversas razones
    Por lo tanto hay  que resetearlos a 0 explicitamente

    """
    for p in w_torch:
        p.grad.zero_()


# 4.2

w_ajust1 = [p.item() for p in w_torch]
y_adjust1_pred = f(w_ajust1, x_exacta)

plt.plot(x_exacta, y_adjust1_pred, c="orange", label="pred 1")


# 5


def f2(w, x):
    return np.exp(-w[0] * x + w[2]) * np.sin(w[1] * x + w[3])


w_exacta = (0.5, 4, 0.2, 3)
w_ini = [p * (1 + 0.25) * np.random.normal() for p in w_exacta]
w_torch = [torch.tensor(data=[p], requires_grad=True, dtype=torch.float) for p in w_ini]
x_torch = torch.tensor(x_data)
y_torch = torch.tensor(y_data)


# PRED 2
num_epochs = 30000
for epoch in range(num_epochs):
    # calculamos las predicciones del modelo. El error y la funcion de perdida
    f_torch = torch.exp(-w_torch[0] * x_torch + w_torch[2]) * torch.sin(
        w_torch[1] * x_torch + w_torch[3]
    )
    loss = ((f_torch - y_torch) ** 2).sum()
    """
    Le decimos a pytorch que llame al metodo  backward() del tensor
    del cual queremos calcular el gradiente
    En este caso dicho tensor es "loss": la funcion de perdida
    """
    loss.backward()

    """
    vemos las componentes del gradiente que estuvimos en la presente epoca
    """

    if epoch % 300 == 0:
        print(
            f"epoch: {epoch} loss: {loss.item()} w_torch: {[p.item() for p in w_torch]}"
        )

    # para actualizar el valor de los parametros (w) tenemos que desactivar el calculo automatico del gradiente
    with torch.no_grad():
        for p in w_torch:
            p -= lr * p.grad

    """
    pytorch no resetea a 0 los valores de los componentes del gradiente
    Sino que procede de manera acumulativa, ya que es conveniente por diversas razones
    Por lo tanto hay  que resetearlos a 0 explicitamente

    """
    for p in w_torch:
        p.grad.zero_()


w_ajust2 = [p.item() for p in w_torch]
y_adjust2_pred = f2(w_ajust2, x_exacta)

plt.plot(x_exacta, y_adjust2_pred, c="green", label="pred 2")
plt.show()
