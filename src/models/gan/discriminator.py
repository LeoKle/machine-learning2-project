from typing import Literal
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, dataset: Literal["MNIST", "CIFAR10"] = "MNIST"):
        super().__init__()

        if dataset == "MNIST":
            self.data_size = 1 * 28 * 28
        elif dataset == "CIFAR10":
            self.data_size = 3 * 32 * 32
        else:
            raise ValueError("Unsupported dataset: choose 'MNIST' or 'CIFAR10'")

        self.main = nn.Sequential(
            nn.Linear(self.data_size, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, self.data_size)
        return self.main(x)
