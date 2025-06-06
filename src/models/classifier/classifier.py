import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int = 10):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.main(x)
