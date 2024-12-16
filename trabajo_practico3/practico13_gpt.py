#!/usr/bin/env python3

"""
Created on Sun Nov 10 17:00:17 2024

@author: luisvargas
"""
import copy
import time

import torch
import torch.nn as nn
from utils import (
    Hyperparameters,
    generate_data,
    log,
    plot_image_and_prediction,
    plot_results,
)


class AutoEncoder(nn.Module):
    def __init__(self, h_params):
        super().__init__()

        input_channels = h_params.input_channels
        conv_channels = h_params.conv_channels
        kernel_size = h_params.kernel_size
        pool_size = h_params.pool_size
        self.dropout = h_params.dropout

        self.n = 28 * 28

        # Encoder: Convolutional + MaxPool2D
        padding = (kernel_size // 2) - 1
        self.encoder = nn.Sequential(
            nn.Conv2d(
                input_channels,
                conv_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),  # (1, 28, 28) -> (16, 26, 26)
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool2d(pool_size),  # (16, 26, 26) ->  (16, 13, 13) ->
            nn.Flatten(),
            nn.Linear(
                16 * 13 * 13, self.n
            ),  # fully connected 16 * 13 * 13  => 16 * 13 * 13
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        # Decoder: ConvTranspose2D
        self.decoder = nn.Sequential(
            nn.Linear(self.n, 16 * 13 * 13),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Unflatten(1, (16, 13, 13)),  # 16*13*13 -> (16,13,13)
            nn.ConvTranspose2d(
                16, 1, kernel_size=6, stride=2, padding=1, output_padding=0, dilation=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Classificator(nn.Module):
    def __init__(self, autoencoder, h_params):
        super().__init__()

        self.n = autoencoder.n
        self.dropout = h_params.dropout
        self.encoder = copy.deepcopy(autoencoder.encoder)
        self.classificator = nn.Sequential(nn.Linear(self.n, 10))

    def forward(self, x):
        x = self.encoder(x)
        x = self.classificator(x)
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
    sum_correct = 0
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

        if h_params.verbose and batch % (num_batches / 10) == 0:
            processed_samples = 100 * num_processed_samples / num_samples
            print(
                f"train loop: batch={batch:>5d}"
                f" batch_avg_loss={batch_avg_loss:>7f}"
                f" processed_samples={processed_samples:>5f}"
            )

    avg_loss = sum_batch_avg_loss / num_batches
    precision = sum_correct / num_samples
    return avg_loss, precision


def eval_loop(dataloader, model, h_params):
    model.eval()
    num_batches = len(dataloader)
    num_samples = len(dataloader.dataset)
    sum_batch_avg_loss = 0
    sum_correct = 0
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
            sum_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            num_processed_samples += batch_size

    avg_loss = sum_batch_avg_loss / num_batches
    precision = sum_correct / num_samples
    if h_params.verbose:
        print(f"eval loop precision={100 * precision:>0.1f} avg loss={avg_loss:>8f}")

    return avg_loss, precision


def train_and_eval(model, train_dataloader, valid_dataloader, h_params):
    list_train_avg_loss_incorrect = []
    list_train_precision_incorrect = []
    list_train_avg_loss = []
    list_train_precision = []
    list_eval_avg_loss = []
    list_eval_precision = []

    for epoch in range(1, h_params.epochs + 1):
        if h_params.classification_stage_running:
            print(f"\nEpoch classification: {epoch}")
        else:
            print(f"\nEpoch convolution: {epoch}")
        print("-" * 70)
        train_avg_loss_incorrect, train_precision_incorrect = train_loop(
            train_dataloader, model, h_params
        )
        list_train_avg_loss_incorrect.append(train_avg_loss_incorrect)
        list_train_precision_incorrect.append(train_precision_incorrect)

        train_avg_loss, train_precision = eval_loop(train_dataloader, model, h_params)
        list_train_avg_loss.append(train_avg_loss)
        list_train_precision.append(train_precision)

        eval_avg_loss, eval_precision = eval_loop(valid_dataloader, model, h_params)
        list_eval_avg_loss.append(eval_avg_loss)
        list_eval_precision.append(eval_precision)

    plot_results(
        h_params,
        list_eval_avg_loss,
        list_train_avg_loss,
        list_train_avg_loss_incorrect,
        list_train_precision_incorrect,
        list_train_precision,
        list_eval_precision,
    )

    log(
        h_params,
        list_eval_avg_loss[-1],
        list_train_avg_loss[-1],
        list_train_avg_loss_incorrect[-1],
    )


def main():
    h_params = Hyperparameters(batch_size=100)
    train_set, train_set_orig, valid_set, valid_set_orig = generate_data()
    train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=h_params.batch_size, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_set, batch_size=h_params.batch_size, shuffle=True
    )

    for conv_channels in [16]:
        h_params.conv_channels = conv_channels
        # TODO make this parameter configurable in Autoencoder class
        for kernel_size in [3]:
            h_params.kernel_size = kernel_size

            autoencoder = AutoEncoder(h_params)
            if h_params.optimizer_option == 1:
                h_params.optimizer = torch.optim.SGD(
                    autoencoder.parameters(), lr=h_params.lr
                )
            else:
                h_params.optimizer = torch.optim.Adam(
                    autoencoder.parameters(), lr=h_params.lr, eps=1e-08
                )

            for epochs in h_params.epochs_convolution_options:
                compute_convolution(
                    autoencoder, epochs, h_params, train_dataloader, valid_dataloader
                )
                plot_image_and_prediction(autoencoder, train_set, h_params)
                compute_classification(
                    autoencoder, h_params, train_set_orig, valid_set_orig
                )


def compute_classification(autoencoder, h_params, train_set_orig, valid_set_orig):
    train_dataloader = torch.utils.data.DataLoader(
        train_set_orig, batch_size=h_params.batch_size, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_set_orig, batch_size=h_params.batch_size, shuffle=True
    )
    h_params = Hyperparameters(loss_fn_option=2)
    h_params.classification_stage_running = True
    model = Classificator(autoencoder, h_params)
    h_params.optimizer = torch.optim.Adam(
        model.classificator.parameters(), lr=h_params.lr, eps=1e-08
    )
    for epochs in h_params.epochs_classification_options:
        h_params.epochs_classification = h_params.epochs = epochs

        start_time = time.perf_counter()
        train_and_eval(model, train_dataloader, valid_dataloader, h_params)
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        log(execution_time=execution_time)
        if h_params.verbose:
            print(model.state_dict())
            # torch.save(model.state_dict(), f"{h_params.filename}_learning.pt")


def compute_convolution(
    autoencoder, epochs, h_params, train_dataloader, valid_dataloader
):
    h_params.epochs_convolution = h_params.epochs = epochs

    start_time = time.perf_counter()
    train_and_eval(autoencoder, train_dataloader, valid_dataloader, h_params)
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    log(execution_time=execution_time)
    if h_params.verbose:
        print(autoencoder.state_dict())
        # torch.save(model.state_dict(), f"{h_params.filename}_learning.pt")


if __name__ == "__main__":
    main()
