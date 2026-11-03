"""A small CNN mixing module calls with tensor functions."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

IDS = ["stem", "conv1", "relu1", "pool1", "conv2", "relu2", "add1", "ln1", "flat", "fc"]


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 1)
        self.conv_weight = nn.Parameter(torch.empty(6, 4, 3, 3))
        self.conv_bias = nn.Parameter(torch.empty(6))
        self.conv2 = nn.Conv2d(6, 6, 3, padding=1)
        self.norm_weight = nn.Parameter(torch.empty(6, 8, 8))
        self.norm_bias = nn.Parameter(torch.empty(6, 8, 8))
        self.fc_weight = nn.Parameter(torch.empty(5, 384))
        self.fc_bias = nn.Parameter(torch.empty(5))

    def forward(self, x, return_intermediates=False):
        a = {}
        a["stem"] = self.stem(x)
        a["conv1"] = F.conv2d(a["stem"], self.conv_weight, self.conv_bias, padding=1)
        a["relu1"] = F.relu(a["conv1"])
        a["pool1"] = F.max_pool2d(a["relu1"], 2)
        a["conv2"] = self.conv2(a["pool1"])
        a["relu2"] = F.relu(a["conv2"])
        a["add1"] = torch.add(a["pool1"], a["relu2"])
        a["ln1"] = F.layer_norm(a["add1"], (6, 8, 8), self.norm_weight, self.norm_bias, eps=1e-5)
        a["flat"] = torch.flatten(a["ln1"], 1)
        a["fc"] = F.linear(a["flat"], self.fc_weight, self.fc_bias)
        return (a["fc"], a) if return_intermediates else a["fc"]


def build_model(seed):
    rng = np.random.default_rng(seed)
    model = Network()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            if name == "norm_weight":
                values = 1 + 0.05 * rng.standard_normal(param.shape)
            else:
                scale = 0.1 if 'bias' in name else 1 / np.sqrt(np.prod(param.shape[1:]))
                values = rng.standard_normal(param.shape) * scale
            param.copy_(torch.from_numpy(values.astype(np.float32)))
    return model.eval()


def make_inputs(seed):
    return np.random.default_rng(seed).standard_normal((16, 3, 16, 16)).astype(np.float32)
