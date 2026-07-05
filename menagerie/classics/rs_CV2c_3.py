# SOURCE: vendored from mjhydri/BeatNet @ main (src/BeatNet/model.py)
# SOURCE: vendored from zhiheng-ma/Bayesian-Crowd-Counting @ master (models/vgg.py)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class BDA(nn.Module):
    """Beat/downbeat activation CRNN from BeatNet."""

    def __init__(self, dim_in, num_cells, num_layers, device):
        """Initialize the BeatNet BDA module."""

        super().__init__()

        self.dim_in = dim_in
        self.dim_hd = num_cells
        self.num_layers = num_layers
        self.device = device
        self.conv_out = 150
        self.kernelsize = 10
        self.conv1 = nn.Conv1d(1, 2, self.kernelsize)
        self.linear0 = nn.Linear(2 * int((self.dim_in - self.kernelsize + 1) / 2), self.conv_out)
        self.lstm = nn.LSTM(
            input_size=self.conv_out,
            hidden_size=self.dim_hd,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.linear = nn.Linear(in_features=self.dim_hd, out_features=3)

        self.softmax = nn.Softmax(dim=0)
        self.hidden = torch.zeros(2, 1, self.dim_hd).to(device)
        self.cell = torch.zeros(2, 1, self.dim_hd).to(device)

        self.to(device)

    def forward(self, data):
        """Run beat/downbeat activation inference."""

        x = data
        x = torch.reshape(x, (-1, self.dim_in))
        x = x.unsqueeze(0).transpose(0, 1)
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.linear0(x)
        x = torch.reshape(x, (np.shape(data)[0], np.shape(data)[1], self.conv_out))
        x, (self.hidden, self.cell) = self.lstm(x, (self.hidden, self.cell))
        out = self.linear(x)
        out = out.transpose(1, 2)
        return out

    def train_forward(self, data):
        """Forward pass for training with a stateless LSTM."""

        x = data
        x = torch.reshape(x, (-1, self.dim_in))
        x = x.unsqueeze(0).transpose(0, 1)
        x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.linear0(x)
        x = torch.reshape(x, (data.shape[0], data.shape[1], self.conv_out))
        x = self.lstm(x)[0]
        out = self.linear(x)
        out = out.transpose(1, 2)
        return out

    def final_pred(self, input):
        """Apply the BeatNet softmax prediction layer."""

        return self.softmax(input)

    def num_flat_features(self, x):
        """Count flattened non-batch features."""

        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class VGG(nn.Module):
    """VGG-based Bayesian Loss crowd-counting regressor."""

    def __init__(self, features):
        """Initialize the VGG crowd-counting regressor."""

        super().__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x):
        """Run the crowd-density regressor."""

        x = self.features(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)
        return torch.abs(x)


def make_layers(cfg, batch_norm=False):
    """Create VGG feature layers."""

    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
    ]
}


def vgg19():
    """Build the official VGG 19-layer architecture with random weights."""

    return VGG(make_layers(cfg["E"]))


def build_beatnet_bda() -> BDA:
    """Build a tiny BeatNet BDA CRNN."""

    model = BDA(dim_in=16, num_cells=4, num_layers=2, device=torch.device("cpu"))
    model.eval()
    return model


def example_input_beatnet_bda() -> torch.Tensor:
    """Create a BeatNet frame-feature input."""

    return torch.randn(1, 3, 16)


def build_bayesian_loss_vgg() -> VGG:
    """Build the Bayesian Loss crowd-counting VGG model."""

    model = vgg19()
    model.eval()
    return model


def example_input_bayesian_loss_vgg() -> torch.Tensor:
    """Create an image input for the crowd-counting VGG model."""

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("BeatNet", "build_beatnet_bda", "example_input_beatnet_bda", "2019", "CV2c_124"),
    (
        "BL (Bayesian Loss Crowd Counting)",
        "build_bayesian_loss_vgg",
        "example_input_bayesian_loss_vgg",
        "2019",
        "CV2c_139",
    ),
]
