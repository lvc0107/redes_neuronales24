#!/usr/bin/env python3
"""
Created on Sun Nov 10 17:00:17 2024

@author: luisvargas
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device")


classes = [
    "T-shirt/top",  # 0
    "Trouser",  # 1
    "Pullover",  # 2
    "Dress",  # 3
    "Coat",  # 4
    "Sandal",  # 5
    "Shirt",  # 6
    "Sneaker",  # 7
    "Bag",  # 8
    "Ankle boot",  # 9
]

label_names = {i: classes[i] for i in range(len(classes))}
print(f"label_names = {label_names}")


def plot_some_data(training_data):
    figure = plt.figure()
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        j = torch.randint(len(training_data), size=(1,)).item()
        image, label = training_data[j]
        figure.add_subplot(rows, cols, i)
        plt.title(label_names[label])
        plt.axis("off")
        # remove channel. Not needed for classificacion
        image_to_plot = image.squeeze()
        plt.imshow(image_to_plot, cmap="Greys_r")

    plt.show()


def generate_data(batch_size):
    # transform number to [0,1]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    training_data = datasets.FashionMNIST(
        root="MNIST_data/",
        train=True,
        download=True,
        transform=transform,
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="MNIST_data/",
        train=False,
        download=True,
        transform=transform,
    )

    plot_some_data(training_data)
    train_loader = torch.utils.data.DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True
    )

    return train_loader, valid_loader


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout):
        super().__init__()
        prev_size = input_size
        layers = [nn.Flatten()]
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_loop(dataloader, model, loss_fn, optimizer, verbose=False):
    model.train()
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    sum_batch_avg_loss = 0
    sum_correct = 0
    num_processed_samples = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        batch_size = len(X)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_avg_loss = loss.item()
        sum_batch_avg_loss += batch_avg_loss
        sum_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        num_processed_samples += batch_size

        if verbose and batch % (num_batches / 10) == 0:
            processed_samples = 100 * num_processed_samples / num_samples
            print(
                f"train loop: batch={batch:>5d} batch_avg_loss={batch_avg_loss:>7f} processed_samples={processed_samples:>5f}"
            )

    avg_loss = sum_batch_avg_loss / num_batches
    precision = sum_correct / num_samples

    return avg_loss, precision


def eval_loop(dataloader, model, loss_fn, verbose=False):
    model.eval()
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    sum_batch_avg_loss = 0
    sum_correct = 0
    num_processed_samples = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            batch_size = len(X)
            pred = model(X)
            loss = loss_fn(pred, y)
            batch_avg_loss = loss.item()
            sum_batch_avg_loss += batch_avg_loss
            sum_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            num_processed_samples += batch_size

    avg_loss = sum_batch_avg_loss / num_batches
    precision = sum_correct / num_samples
    if verbose:
        print(f"eval loop precision={100*precision:>0.1f} avg loss={avg_loss:>8f}")

    return avg_loss, precision


if __name__ == "__main__":
    ####################################
    # Hiperparametros a testear:
                             # Algunos parametros para probar
    batch_size = 100         # 500, 1000,
    dropout = 0.2            # 0.1, 0.5
    lr = 1e-3                # 2e-3, 5e-3
    epochs = 30              # 15, 100
    hidden_sizes = [128, 64] # [128], [256], [64, 32] [64, 32, 32]
    optimizer_option = 2     # 1:SGD 2:Adam

    ####################################
    input_size = 28 * 28
    output_size = 10
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    verbose = True
    loss_fn = nn.CrossEntropyLoss()

    train_dataloader, valid_dataloader = generate_data(batch_size)

    model = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        dropout=dropout,
    )
    optimizer = (
        lambda x, lr: torch.optim.SGD(model.parameters(), lr=lr)
        if x == 1
        else torch.optim.Adam(model.parameters(), lr=lr, eps=1e-08)
    )(optimizer_option, lr)

    list_train_avg_loss_incorrect = []
    list_train_presicion_incorrect = []
    list_train_avg_loss = []
    list_train_presicion = []
    list_eval_avg_loss = []
    list_eval_presicion = []
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch+1}")
        print("-" * 70)
        train_avg_loss_incorrect, train_presicion_incorrect = train_loop(
            train_dataloader, model, loss_fn, optimizer, verbose
        )
        list_train_avg_loss_incorrect.append(train_avg_loss_incorrect)
        list_train_presicion_incorrect.append(train_presicion_incorrect)

        train_avg_loss, train_presicion = eval_loop(
            train_dataloader, model, loss_fn, verbose
        )
        list_train_avg_loss.append(train_avg_loss)
        list_train_presicion.append(train_presicion)

        eval_avg_loss, eval_presicion = eval_loop(
            valid_dataloader, model, loss_fn, verbose
        )
        list_eval_avg_loss.append(eval_avg_loss)
        list_eval_presicion.append(eval_presicion)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(
        range(1, len(list_train_avg_loss_incorrect) + 1),
        list_train_avg_loss_incorrect,
        c="r",
        label="train incorrect",
        linestyle="-",
    )
    plt.plot(
        range(1, len(list_train_avg_loss) + 1),
        list_train_avg_loss,
        c="g",
        label="train",
        linestyle="-.",
    )
    plt.plot(
        range(1, len(list_eval_avg_loss) + 1),
        list_eval_avg_loss,
        c="b",
        label="eval",
        linestyle="--",
    )
    plt.legend()
    plt.show()

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.axhline(y=0.9, c="red", linestyle="--")
    plt.plot(
        range(1, len(list_train_presicion_incorrect) + 1),
        list_train_presicion_incorrect,
        c="r",
        label="train incorrect",
        linestyle="-",
    )
    plt.plot(
        range(1, len(list_train_presicion) + 1),
        list_train_presicion,
        c="g",
        label="train",
        linestyle="-.",
    )
    plt.plot(
        range(1, len(list_eval_presicion) + 1),
        list_eval_presicion,
        c="b",
        label="eval",
        linestyle="--",
    )
    plt.legend()
    plt.show()
