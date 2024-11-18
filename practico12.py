#!/usr/bin/env python3
"""
Created on Sun Nov 10 17:00:17 2024

@author: luisvargas
"""
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Check that MPS is available
    # if not torch.backends.mps.is_available():
    #     if not torch.backends.mps.is_built():
    #         print("MPS not available because the current PyTorch install was not "
    #               "built with MPS enabled.")
    #     else:
    #         print("MPS not available because the current MacOS version is not 12.3+ "
    #               "and/or you do not have an MPS-enabled device on this machine.")

    # else:
    #     device = torch.device("mps")

    print(f"Using {device} device")
    return device


def plot_some_data(training_data):
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


def format_to_log(hyperparameters):
    hyperparameters_to_log = hyperparameters.copy()
    hyperparameters_to_log["Device"] = hyperparameters["Device"].type
    hyperparameters_to_log.pop("Loss Function", None)
    hyperparameters_to_log["Optimizer"] = (
        "SGD" if isinstance(hyperparameters["Optimizer"], torch.optim.SGD) else "Adam"
    )
    return hyperparameters_to_log


def log(hyperparameters, execution_time):
    hyperparameters_to_log = format_to_log(hyperparameters)
    print(f"{'Hyperparameter':<20} {'Value':<10}")
    print("-" * 30)
    for k, v in hyperparameters_to_log.items():
        print(f"{k:<20} {str(v):<10}")
    print("-" * 30)
    key = "Execution time"
    print(f"{key:<20} {execution_time:<10}")


def plot_results(
    hyperparameters,
    list_eval_avg_loss,
    list_eval_precision,
    list_train_avg_loss,
    list_train_avg_loss_incorrect,
    list_train_precision,
    list_train_precision_incorrect,
):
    hyperparameters_to_log = format_to_log(hyperparameters)

    caption = ", ".join([f"{k}: {v}" for k, v in hyperparameters_to_log.items()])
    filename = "_".join([f"{k}-{v}".lower() for k, v in hyperparameters_to_log.items()])

    ######### Loss
    metric = "loss"
    path = "trabajo_practico2"
    extension = "png"
    full_path = f"{path}/{metric}_{filename}.{extension}"
    plt.xlabel("Epocs")
    plt.ylabel("Loss")
    plt.figtext(
        0.5, -0.08, caption, wrap=True, horizontalalignment="center", fontsize=10
    )
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
    plt.savefig(full_path, bbox_inches="tight")
    plt.show()
    ######### Precision
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.figtext(
        0.5, -0.08, caption, wrap=True, horizontalalignment="center", fontsize=10
    )
    plt.axhline(y=0.9, c="red", linestyle="--")
    plt.plot(
        range(1, len(list_train_precision_incorrect) + 1),
        list_train_precision_incorrect,
        c="r",
        label="train incorrect",
        linestyle="-",
    )
    plt.plot(
        range(1, len(list_train_precision) + 1),
        list_train_precision,
        c="g",
        label="train",
        linestyle="-.",
    )
    plt.plot(
        range(1, len(list_eval_precision) + 1),
        list_eval_precision,
        c="b",
        label="eval",
        linestyle="--",
    )
    metric = "precision"
    full_path = f"{path}/{metric}_{filename}.{extension}"
    plt.legend()
    plt.savefig(full_path, bbox_inches="tight")
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


def train_loop(dataloader, model, hyperparameters, verbose=False):
    device = hyperparameters["Device"]
    loss_fn = hyperparameters["Loss Function"]
    optimizer = hyperparameters["Optimizer"]

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


def eval_loop(dataloader, model, hyperparameters, verbose=False):
    device = hyperparameters["Device"]
    loss_fn = hyperparameters["Loss Function"]

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


def train_and_eval(model, train_dataloader, valid_dataloader, hyperparameters, verbose):
    list_train_avg_loss_incorrect = []
    list_train_precision_incorrect = []
    list_train_avg_loss = []
    list_train_precision = []
    list_eval_avg_loss = []
    list_eval_precision = []

    epochs = hyperparameters["Epochs"]
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1}")
        print("-" * 70)
        train_avg_loss_incorrect, train_precision_incorrect = train_loop(
            train_dataloader, model, hyperparameters, verbose
        )
        list_train_avg_loss_incorrect.append(train_avg_loss_incorrect)
        list_train_precision_incorrect.append(train_precision_incorrect)

        train_avg_loss, train_precision = eval_loop(
            train_dataloader, model, hyperparameters, verbose
        )
        list_train_avg_loss.append(train_avg_loss)
        list_train_precision.append(train_precision)

        eval_avg_loss, eval_precision = eval_loop(
            valid_dataloader, model, hyperparameters, verbose
        )
        list_eval_avg_loss.append(eval_avg_loss)
        list_eval_precision.append(eval_precision)

    plot_results(
        hyperparameters,
        list_eval_avg_loss,
        list_eval_precision,
        list_train_avg_loss,
        list_train_avg_loss_incorrect,
        list_train_precision,
        list_train_precision_incorrect,
    )


def main():
    ####################################
    # Hyperparameters to test:
    batch_size = 1000  # 100  500, 1000,
    dropout = 0.2  # 0.1, 0.2, 0.5
    lr = 1e-3  # 1e-3, 2e-3, 5e-3
    epochs = 30  # 15, 30, 100
    hidden_sizes = [128, 64]  # [128, 64],  [128], [256], [64, 32] [64, 32, 32]
    optimizer_option = 2  # 1:SGD 2:Adam

    ####################################
    device = get_device()
    loss_fn = nn.CrossEntropyLoss()
    input_size = 28 * 28
    output_size = 10
    verbose = True

    model = NeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        dropout=dropout,
    )

    if optimizer_option == 1:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-08)

    hyperparameters = {
        "Device": device,
        "Hidden Layers": hidden_sizes,
        "Batch Size": batch_size,
        "Epochs": epochs,
        "Learning Rate": lr,
        "Loss Function": loss_fn,
        "Optimizer": optimizer,
        "Dropout": dropout,
    }

    start_time = time.perf_counter()

    train_dataloader, valid_dataloader = generate_data(batch_size)
    train_and_eval(model, train_dataloader, valid_dataloader, hyperparameters, verbose)

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    log(hyperparameters, execution_time)


if __name__ == "__main__":
    main()
