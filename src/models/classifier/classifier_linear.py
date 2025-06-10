import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int = 10):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.main(x)
