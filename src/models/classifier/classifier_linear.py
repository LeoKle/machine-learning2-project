import torch.nn as nn

class ClassifierMLP(nn.Module):
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
    
def get_activation(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation: {name}")
    
class ClassifierMLPDeep(nn.Module):
    def __init__(self, input_size: int, num_classes: int = 10, activation="leaky_relu"):
        super().__init__()
        act = get_activation(activation)
        self.main = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            act,
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            act,
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            act,
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            act,
            nn.Dropout(0.1),

            nn.Linear(32, 32),
            act,
            nn.Dropout(0.1),

            nn.Linear(32, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.main(x)
    