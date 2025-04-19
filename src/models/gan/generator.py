from typing import Literal
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, dataset: Literal["MNIST", "CIFAR10"] = "MNIST"):
        super().__init__()

        self.latent_dim = latent_dim

        if dataset == "MNIST":
            self.out_channels = 1
            self.image_size = 28
        elif dataset == "CIFAR10":
            self.out_channels = 3
            self.image_size = 32
        else:
            raise ValueError("Unsupported dataset: choose 'MNIST' or 'CIFAR10'")

        self.output_dim = self.out_channels * self.image_size * self.image_size

        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, self.output_dim),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.main(z)
        return out.view(-1, self.out_channels, self.image_size, self.image_size)
