import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoencoderCNN3(nn.Module):
    def __init__(self, dataset_type="CIFAR10", drop_prob=0.0, latent_dim=256):
        super().__init__()

        if dataset_type == "MNIST":
            self.img_channels = 1
            self.img_size = 28
            self.latent_dim = latent_dim if latent_dim is not None else 32
        elif dataset_type == "CIFAR10":
            self.img_channels = 3
            self.img_size = 32
            self.latent_dim = latent_dim if latent_dim is not None else 256
        else:
            raise ValueError("Unsupported dataset. Use 'MNIST' or 'CIFAR10'.")

        self.drop = nn.Dropout2d(p=drop_prob)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.img_channels, 32, 3, stride=1, padding=1),  # 32x32
            nn.ReLU(),
            self.drop,
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            self.drop,
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            self.drop,
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 4x4
            nn.ReLU(),
            self.drop,
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # 2x2
            nn.ReLU(),
            self.drop,
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, self.latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 512 * 2 * 2),
            nn.Unflatten(dim=1, unflattened_size=(512, 2, 2)),
            nn.ConvTranspose2d(
                512, 256, 3, stride=2, padding=1, output_padding=1
            ),  # 4x4
            nn.ReLU(),
            self.drop,
            nn.ConvTranspose2d(
                256, 128, 3, stride=2, padding=1, output_padding=1
            ),  # 8x8
            nn.ReLU(),
            self.drop,
            nn.ConvTranspose2d(
                128, 64, 3, stride=2, padding=1, output_padding=1
            ),  # 16x16
            nn.ReLU(),
            self.drop,
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),  # 32x32
            nn.ReLU(),
            self.drop,
            nn.Conv2d(32, self.img_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)
        decoded = self.decoder(z)
        decoded = F.interpolate(
            decoded,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return decoded
