import torch.nn as nn


class ClassifierMLPLarge(nn.Module):
    def __init__(self, input_size: int, num_classes: int = 10):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.main(x)


# def get_activation(name):
#     if name == "relu":
#         return nn.ReLU(inplace=True)
#     elif name == "leaky_relu":
#         return nn.LeakyReLU(0.1)
#     elif name == "tanh":
#         return nn.Tanh()
#     elif name == "sigmoid":
#         return nn.Sigmoid()
#     else:
#         raise ValueError(f"Unsupported activation: {name}")


class ClassifierResNet(nn.Module):
    def __init__(self, input_size: int, num_classes: int = 10, num_blocks: int = 8):
        super().__init__()
        # act = get_activation(activation)

        self.block1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
        )

        self.block2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
        )

        self.block3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        )

        self.output_block = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.LogSoftmax(
                dim=1
            ),  # NLLLoss: nn.Softmax, CrossEntropyLoss: nn.LogSoftmax
        )

        self.num_blocks = num_blocks

    def forward(self, x):
        x = x.view(x.size(0), -1)
        blocks = [None] * (self.num_blocks + 1)
        blocks[0] = self.block1(x)
        blocks[1] = self.block2(blocks[0])
        for i in range(2, self.num_blocks):
            blocks[i] = self.block3(blocks[i - 1] + blocks[i - 2])
        out = self.output_block(
            blocks[self.num_blocks - 1] + blocks[self.num_blocks - 2]
        )
        return out
