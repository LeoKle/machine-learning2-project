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
        self.linear_dim = 256 * self.init_size**2

        self.main = nn.Sequential(
            nn.Linear(latent_dim, self.linear_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(self.linear_dim),
            nn.Unflatten(1, (256, self.init_size, self.init_size)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> 2x
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                128, self.img_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.main(z)

        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )

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

        self.init_size = 4  # 4x4 base resolution
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_size * self.init_size),
            nn.BatchNorm1d(512 * self.init_size * self.init_size),
            nn.ReLU(True),
        )

        self.conv_blocks = nn.Sequential(
            nn.Unflatten(1, (512, self.init_size, self.init_size)),  # (512, 4, 4)
            nn.ConvTranspose2d(
                512, 256, kernel_size=4, stride=2, padding=1
            ),  # (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1
            ),  # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),  # (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(
                64, self.img_channels, kernel_size=3, stride=1, padding=1
            ),  # Final output
            nn.Tanh(),  # [-1, 1] range
        )

    def forward(self, z):
        out = self.fc(z)
        out = self.conv_blocks(out)
        return out
