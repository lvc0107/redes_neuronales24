#!/usr/bin/env python3
"""
Created on Sun Dec 15 21:55:11 2024

@author: luisvargas
"""
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets, transforms

CURRENT_PATH = Path(__file__).resolve().parent


@dataclass
class Hyperparameters:
    ####################################
    # Hyperparameters:
    batch_size = 100  # 128  512, 1024,
    dropout = 0.1  # 0.1, 0.2, 0.5
    lr = 1e-3  # 1e-3, 2e-3, 5e-3
    epochs = 0
    loss_fn_option = 1  # 1:MSE 2:CEL
    loss_fn = None
    optimizer_option = 2  # 1:SGD 2:Adam
    optimizer = None
    device = None
    verbose = False

    # Convolution Hyperparameters
    epochs_convolution = 2
    input_channels = 1
    conv_channels = 16
    kernel_size = 3
    pool_size = 2

    # Classification Hyperparameters
    hidden_sizes = [128]  # [128, 64],  [128], [256], [64, 32] [64, 32, 32]
    input_size = 28 * 28
    output_size = 10
    epochs_classification = 2
    classification_stage_running = False

    def __init__(self, loss_fn_option=1):
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
    def classificator_attrs(self):
        return {
            "epochs": self.epochs_classification,
            "loss_fn": "MSE" if self.loss_fn_option == 1 else "CrossEntropy",
            "optimizer": "SGD" if self.optimizer_option == 1 else "ADAM",
            "batch_size": self.batch_size,
            "hidden_layers": self.hidden_sizes,
            "dropout": self.dropout,
            "lr": self.lr,
        }

    @property
    def autoencoder_attrs(self):
        return {
            "epochs": self.epochs_convolution,
            "loss_fn": "MSE" if self.loss_fn_option == 1 else "CrossEntropy",
            "optimizer": "SGD" if self.optimizer_option == 1 else "ADAM",
            "conv_channels": self.conv_channels,
            "kernel_size": self.kernel_size,
            "pool_size": self.pool_size,
        }

    @property
    def filename(self):
        return "_".join([f"{k}-{v}" for k, v in self.autoencoder_attrs.items()])


def plot_image_and_prediction(model, train_set, h_params):
    figure = plt.figure()
    rows, cols = 3, 2
    i = 0  # subplot index
    caption = ", ".join([f"{k}: {v}" for k, v in h_params.autoencoder_attrs.items()])

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
    plt.figtext(
        0.5, -0.08, caption, wrap=True, horizontalalignment="center", fontsize=12
    )
    plt.savefig(full_path, bbox_inches="tight")
    plt.show()


def batch(x):
    return x.unsqueeze(0)  # -> (28, 28) => (1, 28, 28)


def unbatch(x):
    return x.squeeze().detach().cpu().numpy()  # (1, 28, 28) => (28, 28)


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

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_log_handler = logging.FileHandler(full_path)
    logger.addHandler(file_log_handler)

    stdout_log_handler = logging.StreamHandler()
    logger.addHandler(stdout_log_handler)

    # nice output format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_log_handler.setFormatter(formatter)
    stdout_log_handler.setFormatter(formatter)

    if execution_time:
        key = "Execution time"
        logger.info(f"{key:<20} {execution_time:<10}")

    for k, v in h_params.autoencoder_attrs.items():
        logger.info(f"{k:<20} {str(v):<10}")
    logger.info("-" * 30)


def plot_loss(
    h_params,
    list_eval_avg_loss,
    list_train_avg_loss,
    list_train_avg_loss_incorrect,
    title=None,
):
    num_samples = len(list_train_avg_loss_incorrect)
    x = range(1, num_samples + 1)
    fontsize = 12
    if h_params.classification_stage_running:
        stage = "clasificación"
        caption = ", ".join(
            [f"{k}: {v}" for k, v in h_params.classificator_attrs.items()]
        )
    else:
        stage = "autoencoder"
        caption = ", ".join(
            [f"{k}: {v}" for k, v in h_params.autoencoder_attrs.items()]
        )

    plt.xlabel("Épocas", size=fontsize)
    plt.ylabel("Error", size=fontsize)
    if title:
        plt.title(f"Error promedio por épocas durante {stage} - {title}", size=fontsize)
    else:
        plt.title(f"Error promedio por épocas durante {stage}", size=fontsize)
    plt.grid(True)
    # plt.xlim([0, len(x) + 1])
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


def plot_accuracy(
    h_params,
    list_train_precision_incorrect,
    list_train_precision,
    list_eval_precision,
    title,
):
    num_samples = len(list_train_precision_incorrect)
    x = range(1, num_samples + 1)
    fontsize = 12

    caption = ", ".join([f"{k}: {v}" for k, v in h_params.classificator_attrs.items()])

    plt.xlabel("Épocas", size=fontsize)
    plt.ylabel("Precisión", size=fontsize)
    plt.grid(True)
    # plt.xlim([0, len(x) + 1])
    plt.figtext(
        0.5, -0.08, caption, wrap=True, horizontalalignment="center", fontsize=fontsize
    )

    y = list_train_precision_incorrect
    plt.title(
        f"Precision promedio por épocas durante clasificación - {title}", size=fontsize
    )
    plt.plot(x, y, c="r", label="Precisión durante entrenamiento.", linestyle="-")
    plt.plot(x[-1], y[-1], c="r", marker="o", markersize=5)

    y = list_train_precision
    plt.plot(x, y, c="g", label="Evaluación en datos de entrenamiento.", linestyle="-.")
    plt.plot(x[-1], y[-1], c="g", marker="o", markersize=5)

    y = list_eval_precision
    plt.plot(x, y, c="b", label="Evaluación en datos de validación.", linestyle="--")
    plt.plot(x[-1], y[-1], c="b", marker="o", markersize=5)
    plt.legend()

    metric = "accuracy"
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

    train_set_clasif = datasets.FashionMNIST(
        root="MNIST_data/",
        train=True,
        download=True,
        transform=transform,
    )

    # Download test data from open datasets.
    valid_set_clasif = datasets.FashionMNIST(
        root="MNIST_data/",
        train=False,
        download=True,
        transform=transform,
    )

    train_set_autoencoder = CustomDataset(train_set_clasif)
    valid_set_autoencoder = CustomDataset(valid_set_clasif)
    return (
        train_set_autoencoder,
        train_set_clasif,
        valid_set_autoencoder,
        valid_set_clasif,
    )


def count_time(f, *args):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        f(*args)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        log(h_params=args[1], execution_time=execution_time)

    return wrapper
