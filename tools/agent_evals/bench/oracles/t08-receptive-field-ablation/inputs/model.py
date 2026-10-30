"""Seeded spatial network with stride, padding and dilation."""
import numpy as np
import torch
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 5, 3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(5, 6, 3, dilation=2, padding=2)
        self.relu3 = nn.ReLU()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(6, 3)

    def forward(self, x, return_intermediates=False):
        a = {}
        a["relu1"] = self.relu1(self.conv1(x))
        a["pool1"] = self.pool1(a["relu1"])
        a["relu2"] = self.relu2(self.conv2(a["pool1"]))
        a["relu3"] = self.relu3(self.conv3(a["relu2"]))
        logits = self.fc(self.avg(a["relu3"]).flatten(1))
        return (logits, a) if return_intermediates else logits


def build_model(seed):
    rng = np.random.default_rng(seed)
    model = Network()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            scale = 0.1 if 'bias' in name else 1 / np.sqrt(np.prod(param.shape[1:]))
            param.copy_(torch.from_numpy((rng.standard_normal(param.shape) * scale).astype(np.float32)))
    return model.eval()


def make_inputs(seed):
    return np.random.default_rng(seed).standard_normal((8, 1, 64, 64)).astype(np.float32)
