# ruff: noqa
import warnings

import torch
from torch import nn

torch.set_num_threads(1)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(8, 16), nn.Linear(16, 16), nn.Linear(16, 4)

    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


def make():
    torch.manual_seed(0)
    return MLP().eval(), torch.randn(3, 8), torch.randn(3, 8)


def hooked(model, site, edit, x):
    h = model.get_submodule(site).register_forward_hook(lambda m, a, o: edit(o))
    try:
        with torch.no_grad():
            return model(x)
    finally:
        h.remove()


def md(a, b):
    return float((a.detach() - b.detach()).abs().max())


def show_warnings(w):
    for x in w:
        print("   warning:", type(x.message).__name__, str(x.message)[:110])
