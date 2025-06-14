from typing import Literal
import torch.nn as nn
import torch.nn.functional as F


class GeneratorCNN(nn.Module):
    def __init__(self, latent_dim=100, dataset: Literal["MNIST", "CIFAR10"] = "MNIST"):
        super(GeneratorCNN, self).__init__()
        self.latent_dim = latent_dim
        if dataset == "MNIST":
            self.img_size = 28
            self.img_channels = 1
        elif dataset == "CIFAR10":
            self.img_size = 32
            self.img_channels = 3
        else:
            raise ValueError("Unsupported dataset: choose 'MNIST' or 'CIFAR10'")

        self.init_size = self.img_size // 4  # Downsampled spatial size (e.g. 7 or 8)
        self.linear_dim = 256 * self.init_size ** 2

        self.main = nn.Sequential(
            nn.Linear(latent_dim, self.linear_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(self.linear_dim),
            nn.Unflatten(1, (256, self.init_size, self.init_size)),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> 2x
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, self.img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.main(z)

        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        return x
    
class GeneratorCNN2(nn.Module):
    def __init__(self, latent_dim=100, dataset: Literal["MNIST", "CIFAR10"] = "MNIST"):
        super(GeneratorCNN2, self).__init__()
        self.latent_dim = latent_dim
        if dataset == "MNIST":
            self.img_size = 28
            self.img_channels = 1
        elif dataset == "CIFAR10":
            self.img_size = 32
            self.img_channels = 3
        else:
            raise ValueError("Unsupported dataset: choose 'MNIST' or 'CIFAR10'")

        self.init_size = 4  # Starting size (4x4)
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.main = nn.Sequential(
            # Output of linear layer reshaped to (256, 4, 4)
            nn.Unflatten(1, (256, self.init_size, self.init_size)),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Output: 32x32x3
            nn.Conv2d(128, self.img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.linear(z)
        x = self.main(x)
        return x