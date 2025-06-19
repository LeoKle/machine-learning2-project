import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self, dataset="MNIST", num_classes=10):
        super().__init__()

        if dataset == "MNIST":
            input_channels = 1
        elif dataset == "CIFAR10":
            input_channels = 3
        else:
            raise ValueError("Dataset must be 'MNIST' or 'CIFAR10'.")

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        if dataset == "MNIST":
            self.fc = nn.Linear(256 * 1 * 1, num_classes)
        else:
            self.fc = nn.Linear(256 * 2 * 2, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.fc(x), dim=1)
