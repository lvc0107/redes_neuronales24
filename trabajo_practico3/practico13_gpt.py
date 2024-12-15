#!/usr/bin/env python3

"""
Created on Sun Nov 10 17:00:17 2024

@author: luisvargas
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms

CURRENT_PATH = Path(__file__).resolve().parent


def batch(x):
    return x.unsqueeze(0)  # -> (28, 28) => (1, 28, 28)


def unbatch(x):
    return x.squeeze().detach().cpu().numpy()  # (1, 28, 28) => (28, 28)


def plot_image_and_prediction(model, train_set, h_params):
    figure = plt.figure()
    rows, cols = 3, 2
    i = 0  # subplot index
    for row in range(1, rows + 1):
        j = torch.randint(len(train_set), size=(1,)).item()
        i += 1
        image, _ = train_set[j]
        figure.add_subplot(rows, cols, i)
        if row == 1:
            plt.title("original")
        plt.axis("off")
        plt.imshow(unbatch(image), cmap="Greys_r")

        i += 1
        figure.add_subplot(rows, cols, i)
        if row == 1:
            plt.title("predicha")
        plt.axis("off")
        pred_image = unbatch(model(batch(image)))
        plt.imshow(pred_image, cmap="Greys_r")

    extension = "png"
    full_path = f"{CURRENT_PATH}/prediccion_{h_params.filename}.{extension}"
    plt.savefig(full_path, bbox_inches="tight")
    plt.show()


def log(
    h_params=None,
    last_eval_avg_loss=None,
    last_train_avg_loss=None,
    last_train_avg_loss_incorrect=None,
    execution_time=None,
):
    extension = "p"
    filename = "results"
    full_path = f"{CURRENT_PATH}/{filename}.{extension}"

    with open(full_path, "a") as f:
        if execution_time:
            key = "Execution time"
            print("-" * 30, file=f)
            print(f"{key:<20} {execution_time:<10}", file=f)
            return

        print("-" * 30, file=f)
        print("\n", file=f)

        print(f"{'Hyperparameter':<20} {'Value':<10}", file=f)
        print("-" * 30, file=f)
        for k, v in h_params.attrs.items():
            print(f"{k:<20} {str(v):<10}", file=f)
        print("-" * 30, file=f)

        print(
            f"{'Last train avg loss incorrect':<20} {last_train_avg_loss_incorrect:>7f}",
            file=f,
        )


def plot_results(
    h_params,
    list_eval_avg_loss,
    list_train_avg_loss,
    list_train_avg_loss_incorrect,
):
    caption = ", ".join([f"{k}: {v}" for k, v in h_params.attrs.items()])

    num_samples = len(list_train_avg_loss_incorrect)
    x = range(1, num_samples + 1)
    fontsize = 12

    plt.xlabel("Épocas", size=fontsize)
    plt.ylabel("Error", size=fontsize)
    plt.grid(True)
    # plt.xlim([0, len(x) + 1])
    plt.title("Error promedio por épocas", size=fontsize)
    plt.figtext(
        0.5, -0.08, caption, wrap=True, horizontalalignment="center", fontsize=fontsize
    )

    y = list_train_avg_loss_incorrect
    plt.plot(x, y, c="r", label="Error durante entrenamiento.", linestyle="-")
    plt.plot(x[-1], y[-1], c="r", marker="o", markersize=5)

    y = list_train_avg_loss
    plt.plot(x, y, c="g", label="Evaluación en datos de entrenamiento.", linestyle="-.")
    plt.plot(x[-1], y[-1], c="g", marker="o", markersize=5)

    y = list_eval_avg_loss
    plt.plot(x, y, c="b", label="Evaluación en datos de validación.", linestyle="--")
    plt.plot(x[-1], y[-1], c="b", marker="o", markersize=5)

    plt.legend()

    metric = "loss"
    extension = "png"
    full_path = f"{CURRENT_PATH}/{metric}_{h_params.filename}.{extension}"
    plt.savefig(full_path, bbox_inches="tight")
    plt.show()


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        image, label = self.dataset[i]
        return image, image


def generate_data():
    # transform number to [0,1]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_set_orig = datasets.FashionMNIST(
        root="MNIST_data/",
        train=True,
        download=True,
        transform=transform,
    )

    # Download test data from open datasets.
    valid_set_orig = datasets.FashionMNIST(
        root="MNIST_data/",
        train=False,
        download=True,
        transform=transform,
    )

    train_set = CustomDataset(train_set_orig)
    valid_set = CustomDataset(valid_set_orig)
    return train_set, valid_set


class AutoEncoder(nn.Module):
    def __init__(self, h_params):
        super().__init__()

        input_channels = h_params.input_channels
        conv_channels = h_params.conv_channels
        kernel_size = h_params.kernel_size
        pool_size = h_params.pool_size
        dropout = h_params.dropout

        # Encoder: Convolutional + MaxPool2D

        padding = (kernel_size // 2) - 1
        self.conv2d = nn.Sequential(
            nn.Conv2d(
                input_channels,
                conv_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(pool_size),
        )

        # Flattened dimensions after convolution and pooling
        flattened_dim = conv_channels * (26 // pool_size) * (26 // pool_size)

        # Fully connected layer
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, flattened_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Decoder: ConvTranspose2D
        self.convt2d = nn.Sequential(
            nn.Unflatten(1, (conv_channels, 26 // pool_size, 26 // pool_size)),
            nn.ConvTranspose2d(
                conv_channels,
                input_channels,
                kernel_size=(kernel_size * 2),
                stride=2,
                padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv2d(x)
        x = self.linear(x)
        x = self.convt2d(x)
        return x


def train_loop(dataloader, model, h_params):
    device = h_params.device
    loss_fn = h_params.loss_fn
    optimizer = h_params.optimizer

    model.train()
    num_samples = len(dataloader.dataset)
    num_batches = len(dataloader)
    sum_batch_avg_loss = 0
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
        num_processed_samples += batch_size

        if h_params.verbose and batch % (num_batches / 10) == 0:
            processed_samples = 100 * num_processed_samples / num_samples
            print(
                f"train loop: batch={batch:>5d}"
                f" batch_avg_loss={batch_avg_loss:>7f}"
                f" processed_samples={processed_samples:>5f}"
            )

    avg_loss = sum_batch_avg_loss / num_batches

    return avg_loss


def eval_loop(dataloader, model, h_params):
    model.eval()
    num_batches = len(dataloader)
    sum_batch_avg_loss = 0
    num_processed_samples = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(h_params.device)
            y = y.to(h_params.device)
            batch_size = len(X)
            pred = model(X)
            loss = h_params.loss_fn(pred, y)
            batch_avg_loss = loss.item()
            sum_batch_avg_loss += batch_avg_loss
            num_processed_samples += batch_size

    avg_loss = sum_batch_avg_loss / num_batches
    if h_params.verbose:
        print(f"eval loop: avg loss={avg_loss:>8f}")

    return avg_loss


def train_and_eval(model, train_dataloader, valid_dataloader, h_params):
    list_train_avg_loss_incorrect = []
    list_train_avg_loss = []
    list_eval_avg_loss = []

    for epoch in range(1, h_params.epochs + 1):
        print(f"\nEpoch: {epoch}")
        print("-" * 70)
        train_avg_loss_incorrect = train_loop(train_dataloader, model, h_params)
        list_train_avg_loss_incorrect.append(train_avg_loss_incorrect)

        train_avg_loss = eval_loop(train_dataloader, model, h_params)
        list_train_avg_loss.append(train_avg_loss)

        eval_avg_loss = eval_loop(valid_dataloader, model, h_params)
        list_eval_avg_loss.append(eval_avg_loss)

    plot_results(
        h_params,
        list_eval_avg_loss,
        list_train_avg_loss,
        list_train_avg_loss_incorrect,
    )

    log(
        h_params,
        list_eval_avg_loss[-1],
        list_train_avg_loss[-1],
        list_train_avg_loss_incorrect[-1],
    )


def execute_model(h_params):
    train_set, valid_set = generate_data()
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=h_params.batch_size, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_set, batch_size=h_params.batch_size, shuffle=True
    )

    for epochs in [10, 20, 30]:
        h_params.epochs = epochs
        for conv_channels in [16, 32, 64]:
            h_params.conv_channels = conv_channels
            for kernel_size in [3]:
                h_params.kernel_size = kernel_size

                model = AutoEncoder(h_params)

                if h_params.optimizer_option == 1:
                    h_params.optimizer = torch.optim.SGD(
                        model.parameters(), lr=h_params.lr
                    )
                else:
                    h_params.optimizer = torch.optim.Adam(
                        model.parameters(), lr=h_params.lr, eps=1e-08
                    )

                start_time = time.perf_counter()
                train_and_eval(model, train_dataloader, valid_dataloader, h_params)
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                log(execution_time=execution_time)

                if h_params.verbose:
                    print(model.state_dict())
                    # torch.save(model.state_dict(), f"{h_params.filename}_learning.pt")
                plot_image_and_prediction(model, train_set, h_params)


class Hyperparameters:
    ####################################
    # Hyperparameters:
    batch_size = 100  # 128  512, 1024,
    dropout = 0.1  # 0.1, 0.2, 0.5
    lr = 1e-3  # 1e-3, 2e-3, 5e-3
    epochs = 10  # 15, 30, 100
    loss_fn_option = 1  # 1:MSE 2:CEL
    loss_fn = None
    optimizer_option = 2  # 1:SGD 2:Adam
    optimizer = None
    device = None
    verbose = False

    # Convolution Hyperparameters
    input_channels = 1
    conv_channels = 16
    kernel_size = 3
    pool_size = 2

    def __init__(self, epochs=2, batch_size=100, loss_fn_option=1):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = self.get_device()
        self.loss_fn_option = loss_fn_option

        if loss_fn_option == 1:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def get_device(self):
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

    @property
    def attrs(self):
        nn_attrs = {
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "lr": self.lr,
            "device": self.device.type,
        }

        autoencoder_attrs = {
            "epochs": self.epochs,
            "loss_fn": "MSE" if self.loss_fn_option == 1 else "CrossEntropy",
            "optimizer": "SGD" if self.optimizer_option == 1 else "ADAM",
            "conv_channels": self.conv_channels,
            "kernel_size": self.kernel_size,
            "pool_size": self.pool_size,
        }
        return autoencoder_attrs

    @property
    def filename(self):
        return "_".join([f"{k}-{v}" for k, v in self.attrs.items()])


def main():
    h_params = Hyperparameters(batch_size=100)
    execute_model(h_params)


if __name__ == "__main__":
    main()
