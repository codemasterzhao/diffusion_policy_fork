import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()

        # Define the linear layers and activation functions
        self.block1 = nn.Sequential(
            nn.Linear(23, 180),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Linear(180, 252),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Linear(252, 252),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Linear(252, 252),
            nn.ReLU()
        )

        self.block5 = nn.Sequential(
            nn.Linear(252, 10),
            nn.ReLU()
        )

    def forward(self, x):
        # Apply each block sequentially
        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.block5(x)

        return x
