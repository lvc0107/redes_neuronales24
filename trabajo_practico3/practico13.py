#!/usr/bin/env python3

"""
Created on Sun Nov 10 17:00:17 2024

@author: luisvargas
"""
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import Dataset



CURRENT_PATH = Path(__file__).resolve().parent


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
    label_names = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }
    import json

    pretty_dict = json.dumps(label_names, indent=4, sort_keys=True)
    print(f"label_names = {pretty_dict}")

    figure = plt.figure()
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        j = torch.randint(len(training_data), size=(1,)).item()
        image, label = training_data[j]
        figure.add_subplot(rows, cols, i)
        plt.title(label_names[label])
        plt.axis("off")
        # remove channel. Not needed for classification
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


def log(
    hyperparameters=None,
    last_eval_avg_loss=None,
    last_eval_precision=None,
    last_train_avg_loss=None,
    last_train_avg_loss_incorrect=None,
    last_train_precision=None,
    last_train_precision_incorrect=None,
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

        hyperparameters_to_log = format_to_log(hyperparameters)
        print("-" * 30, file=f)
        print("\n", file=f)

        print(f"{'Hyperparameter':<20} {'Value':<10}", file=f)
        print("-" * 30, file=f)
        for k, v in hyperparameters_to_log.items():
            print(f"{k:<20} {str(v):<10}", file=f)
        print("-" * 30, file=f)

        print(
            f"{'Last train avg loss incorrect':<20} {last_train_avg_loss_incorrect:>7f}",
            file=f,
        )
        print(
            f"{'Last train precision incorrect':<20} {last_train_precision_incorrect:>7f}",
            file=f,
        )

        print(f"{'Last train avg loss':<20} {last_train_avg_loss:>7f}", file=f)
        print(f"{'Last train avg loss':<20} {last_train_precision:>7f}", file=f)

        print(f"{'Last eval avg loss':<20} {last_eval_avg_loss:>7f}", file=f)
        print(f"{'Last eval precision':<20} {last_eval_precision:>7f}", file=f)


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

    num_samples = len(list_train_avg_loss_incorrect)
    x = range(1,num_samples + 1)
    fontsize = 12

    plt.xlabel("Épocas", size=fontsize)
    plt.ylabel("Error", size=fontsize)
    plt.grid(True)
    #plt.xlim([0, len(x) + 1])
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
    full_path = f"{CURRENT_PATH}/{metric}_{filename}.{extension}"
    plt.savefig(full_path, bbox_inches="tight")
    plt.show()

    fontsize = 12
    plt.xlabel("Épocas", size=fontsize)
    plt.ylabel("Precisión", size=fontsize)
    plt.title("Precisión por épocas", size=fontsize)
    plt.grid(True)
    #plt.xlim([0, len(x)])
    plt.figtext(
        0.5, -0.08, caption, wrap=True, horizontalalignment="center", fontsize=12
    )
    plt.axhline(y=0.9, c="red", linestyle="--")

    y = list_train_precision_incorrect
    plt.plot(x, y, c="r", label="Precisión durante entrenamiento", linestyle="-")
    plt.plot(x[-1], y[-1], c="r", marker="o", markersize=5)

    y = list_train_precision
    plt.plot(x, y, c="g", label="Evaluación en datos de entrenamiento", linestyle="-.")
    plt.plot(x[-1], y[-1], c="g", marker="o", markersize=5)
 
    y = list_eval_precision
    plt.plot(x, y, c="b", label="Evaluación en datos de validación", linestyle="--")
    plt.plot(x[-1], y[-1], c="b", marker="o", markersize=5)
 
    metric = "precision"
    full_path = f"{CURRENT_PATH}/{metric}_{filename}.{extension}"
    plt.legend()
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

def generate_data(batch_size):
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

    plot_some_data(train_set_orig)


    train_set = CustomDataset(train_set_orig)
    valid_set = CustomDataset(valid_set_orig)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=True
    )


    return train_loader, valid_loader


class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout):
        super().__init__()
        

        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=0), #  (1, 28, 28) -> (16, 26, 26)
            nn.ReLU(), # activacion
            nn.Dropout(dropout),
            nn.MaxPool2d(2,2) #  (16, 26, 26) -> (16, 13, 13)            
        )
        
        self.linear = nn.Sequential(
            nn.Flatten(), #  (16, 13, 13) -> 16 * 13 * 13
            nn.Linear(16 * 13 * 13, 16 * 13 * 13), # fully connected 16 * 13 * 13  => 16 * 13 * 13
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.convt2d = nn.Sequential(
            nn.UnFlatten(1, (16, 13, 13)), #  16 * 13 * 13 -> (16, 13, 13)
            nn.ConvTranspose2d(16, 1, kernel_size=6, stride=2, padding=1), # (16, 13, 13) -> (1, 28, 28) 
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv2d(x)
        x = self.linear(x)
        x = self.convt2d(x)
        return x


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
                f"train loop: batch={batch:>5d}"
                f" batch_avg_loss={batch_avg_loss:>7f}"
                f" processed_samples={processed_samples:>5f}"
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
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch: {epoch}")
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

    log(
        hyperparameters,
        list_eval_avg_loss[-1],
        list_eval_precision[-1],
        list_train_avg_loss[-1],
        list_train_avg_loss_incorrect[-1],
        list_train_precision[-1],
        list_train_precision_incorrect[-1],
    )


def main():
    ####################################
    # Hyperparameters to test:
    batch_size = 128  # 128  512, 1024,
    dropout = 0.1  # 0.1, 0.2, 0.5
    lr = 2e-3  # 1e-3, 2e-3, 5e-3
    epochs = 15  # 15, 30, 100
    hidden_sizes = [256]  # [128, 64],  [128], [256], [64, 32] [64, 32, 32]
    optimizer_option = 2  # 1:SGD 2:Adam

    ####################################
    device = get_device()
    loss_fn = nn.CrossEntropyLoss()
    input_size = 28 * 28
    output_size = 10
    verbose = False

    model = AutoEncoder(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        dropout=dropout,
    )

    # 3*3*3*3*5*2 = 810 tests
    # 2*2*2*3*5*1 = 120 tests
    if optimizer_option == 1:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, eps=1e-08
        )

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
    train_dataloader, valid_dataloader = generate_data(
        batch_size
    )
    
    start_time = time.perf_counter()
    
    train_and_eval(
        model,
        train_dataloader,
        valid_dataloader,
        hyperparameters,
        verbose,
    )

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    log(execution_time=execution_time)


if __name__ == "__main__":
    main()
