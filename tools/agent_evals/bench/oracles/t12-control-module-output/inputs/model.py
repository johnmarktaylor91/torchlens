"""The network for this task. Weights are deterministic given weight_seed.

Build it with ``build_model(weight_seed)``; it is returned in eval mode.
"""
import numpy as np
import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.block1 = nn.Sequential()
        self.block1.add_module("conv", nn.Conv2d(16, 32, 3, padding=1))
        self.block1.add_module("bn", nn.BatchNorm2d(32))
        self.block1.add_module("relu", nn.ReLU())
        self.block1.add_module("pool", nn.MaxPool2d(2))
        self.block2 = nn.Sequential()
        self.block2.add_module("conv", nn.Conv2d(32, 64, 3, padding=1))
        self.block2.add_module("bn", nn.BatchNorm2d(64))
        self.block2.add_module("relu", nn.ReLU())
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10))

    def forward(self, x):
        return self.head(self.block2(self.block1(self.stem(x))))


def build_model(weight_seed: int) -> SmallCNN:
    rng = np.random.default_rng(weight_seed)
    model = SmallCNN()
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if name.endswith("bias"):
                vals = rng.standard_normal(p.shape) * 0.1
            else:
                fan_in = int(np.prod(p.shape[1:])) if p.dim() > 1 else 1
                vals = rng.standard_normal(p.shape) / np.sqrt(fan_in)
            p.copy_(torch.from_numpy(vals.astype(np.float32)))
        for name, b in sorted(model.named_buffers()):
            if name.endswith("running_mean"):
                b.copy_(torch.from_numpy((rng.standard_normal(b.shape) * 0.5).astype(np.float32)))
            elif name.endswith("running_var"):
                b.copy_(torch.from_numpy((0.5 + rng.random(b.shape)).astype(np.float32)))
    return model.eval()


def make_stimuli(stimulus_seed: int, n: int, shape) -> np.ndarray:
    return np.random.default_rng(stimulus_seed).standard_normal((n, *shape)).astype(np.float32)
