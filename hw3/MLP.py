import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_featrues, num_of_classes=2):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_featrues, 100),
            nn.ReLU(),
            nn.Linear(100, num_of_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
