from typing import Literal
import torch.nn as nn


class DiscriminatorCNN(nn.Module):
    def __init__(self, dataset: Literal["MNIST", "CIFAR10"] = "MNIST"):
        super(DiscriminatorCNN, self).__init__()

        if dataset == "MNIST":
            self.img_channels = 1
            self.img_size = 28
        elif dataset == "CIFAR10":
            self.img_channels = 3
            self.img_size = 32
        else:
            raise ValueError("Unsupported dataset: choose 'MNIST' or 'CIFAR10'")

        self.main = nn.Sequential(
            nn.Conv2d(
                self.img_channels, 64, kernel_size=4, stride=2, padding=1
            ),  # -> size/2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> size/4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> size/8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.output_size = self.img_size // 8  # e.g. 28 -> 3, 32 -> 4
        self.fc = nn.Linear(256 * self.output_size**2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
