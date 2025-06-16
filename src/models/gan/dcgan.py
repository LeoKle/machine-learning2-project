from typing import Literal
import torch.nn as nn


class DCGANGenerator(nn.Module):
    def __init__(
        self, latent_dim=100, dataset: Literal["MNIST", "CIFAR10"] = "CIFAR10"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        if dataset != "CIFAR10":
            raise ValueError()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class DCGANDiscriminator(nn.Module):
    def __init__(self, dataset: Literal["MNIST", "CIFAR10"] = "CIFAR10"):
        super().__init__()

        if dataset != "CIFAR10":
            raise ValueError()

        self.net = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1)
