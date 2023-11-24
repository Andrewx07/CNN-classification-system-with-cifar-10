from torch import nn as nn


# model
class REDCN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_conv = nn.Sequential(
            nn.Conv2d(3, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 30, 5),
            nn.ReLU(),
            nn.Conv2d(30, 40, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(1),
        )

        self.relu_linear = nn.Sequential(
            nn.Linear(360, 700),
            nn.SELU(),
            nn.Linear(700, 300),
            nn.SELU(),
            # nn.Dropout(p=0.25),
            nn.Linear(300, 150),
            nn.SELU(),
            nn.Linear(150, 80),
            nn.SELU(),
            nn.Linear(80, 20),
        )

    def forward(self, x):
        conv_layers = self.relu_conv(x)
        linear_layers = self.relu_linear(conv_layers)
        return linear_layers
