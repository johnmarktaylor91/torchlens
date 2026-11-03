"""Fixture model for the harness stub task."""

import torch
from torch import nn


class Net(nn.Module):
    """Two linear layers around a ReLU."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, 7)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(7, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.fc2(self.act(self.fc1(x)))))


def build_model() -> nn.Module:
    """Return the seeded model in eval mode."""

    torch.manual_seed(0)
    return Net().eval()


def make_inputs(seed: int) -> torch.Tensor:
    """Return 100 input rows for ``seed``."""

    return torch.randn(100, 5, generator=torch.Generator().manual_seed(seed))
