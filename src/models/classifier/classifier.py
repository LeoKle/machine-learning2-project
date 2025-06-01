from typing import Literal
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, dataset: Literal["MNIST", "CIFAR10"] = "MNIST"):
        super().__init__()

        if dataset == "MNIST":
            input_size = 1 * 28 * 28
            hidden_size = 512  # zu groß??
        elif dataset == "CIFAR10":
            input_size = 3 * 32 * 32
            hidden_size = 1024  # zu groß??
        else:
            raise ValueError("Unsupported dataset: choose 'MNIST' or 'CIFAR10'")

        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(True),
            nn.Linear(hidden_size // 2, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.main(x)
