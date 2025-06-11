import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, dataset_type="CIFAR10", drop_prob=0.0):
        super().__init__()

        if dataset_type == "MNIST":
            self.img_channels = 1
            self.img_size = 28
            self.latent_dim = 32
        elif dataset_type == "CIFAR10":
            self.img_channels = 3
            self.img_size = 32
            self.latent_dim = 128
        else:
            raise ValueError("Unsupported dataset. Use 'MNIST' or 'CIFAR10'.")

        self.drop = nn.Dropout2d(p=drop_prob)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.img_channels, 32, kernel_size=3, stride=2, padding=1),  # -> 16x16 or 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),                 # -> 8x8 or 7x7
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),                # -> 4x4 or 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, self.latent_dim) 
        )


        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 128 * 4 * 4),                                            # [B, 2048]
            nn.Unflatten(dim=1, unflattened_size=(128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.img_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  
        )

    

    def forward(self, x):
        z = self.encoder(x)
        decoded = self.decoder(z)
        #decoded = F.interpolate(decoded, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return decoded


