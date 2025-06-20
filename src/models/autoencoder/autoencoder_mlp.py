import torch.nn as nn


class Autoencoder_lin(nn.Module):
    def __init__(self, dataset_type="MNIST"):
        super().__init__()

        if dataset_type == "MNIST":
            self.image_dim = 1 * 28 * 28
            self.output_shape = (1, 28, 28)
        elif dataset_type == "CIFAR10":
            self.image_dim = 3 * 32 * 32
            self.output_shape = (3, 32, 32)
        else:
            raise ValueError("Unsupported dataset type. Use 'MNIST' or 'CIFAR10'.")

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.image_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.image_dim),
            nn.Tanh(),
            nn.Unflatten(dim=1, unflattened_size=self.output_shape),
        )

    def forward(self, x):
        z = self.encoder(x)
        decoded = self.decoder(z)
        return decoded
