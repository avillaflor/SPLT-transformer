from torch import nn


class Policy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, observation):
        pass

    def update_context(self, observation, action, reward):
        pass

    def reset(self):
        pass
