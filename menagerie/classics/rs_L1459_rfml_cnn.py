# SOURCE: vendored from brysef/rfml @ 1.0.1 (tag)
# https://raw.githubusercontent.com/brysef/rfml/1.0.1/rfml/nn/model/cnn.py
# https://raw.githubusercontent.com/brysef/rfml/1.0.1/rfml/nn/model/base.py
# https://raw.githubusercontent.com/brysef/rfml/1.0.1/rfml/nn/layers/flatten.py
# https://raw.githubusercontent.com/brysef/rfml/1.0.1/rfml/nn/layers/power_normalization.py
# https://raw.githubusercontent.com/brysef/rfml/1.0.1/rfml/nn/F/energy.py
#
# rfml (Radio Frequency Machine Learning), Bryse Flowers' PyTorch library of RF signal
# classification networks. `CNN` is the flagship "VT_CNN2"-based automatic modulation
# classification (AMC) architecture (O'Shea et al. 2016; West/O'Shea 2017), documented
# in the repo as ported to PyTorch with bias removed on the first conv layer and the
# PowerNormalization folded into the network. Transcribed verbatim from
# `rfml/nn/model/cnn.py` (+ its `Model` base class, `Flatten`/`PowerNormalization`
# layers, and the `energy()` helper they call) with only the package-relative imports
# flattened into this single file; no architectural change.

import torch
import torch.nn as nn


# ---- rfml/nn/F/energy.py ----
def energy(x: torch.Tensor, sps: float = 1.0):
    """Calculate the average energy (per symbol if provided) for each example.

    Assumes signal structured as Batch x Channel x IQ x Time.
    """
    if len(x.shape) != 4:
        raise ValueError(
            "The inputs to the energy function must have 4 dimensions (BxCxIQxT), "
            "input shape was {}".format(x.shape)
        )
    if x.shape[2] != 2:
        raise ValueError(
            "The inputs to the energy function must be 'complex valued' by having 2 "
            "elements in the IQ dimension (BxCxIQxT), input shape was {}".format(x.shape)
        )
    iq_dim = 2
    time_dim = 3

    r, c = x.chunk(chunks=2, dim=iq_dim)
    power = (r * r) + (c * c)  # power is magnitude squared so sqrt cancels

    x = torch.mean(power, dim=time_dim) * sps
    x = x.squeeze(dim=iq_dim)

    return x


# ---- rfml/nn/layers/flatten.py ----
class Flatten(nn.Module):
    """Flatten the channel, IQ, and time dims into a single feature dim."""

    def __init__(self, preserve_time: bool = False):
        super().__init__()
        self._preserve_time = preserve_time

    def forward(self, x: torch.Tensor):
        if self._preserve_time:
            return self._flatten_preserve_time(x=x)
        else:
            return self._flatten(x=x)

    def _flatten(self, x: torch.Tensor):
        if len(x.shape) < 2:
            raise ValueError(
                "The inputs to the Flatten layer must have at least 2 dimensions (e.g. "
                "BxCxIQxT), input shape was {}".format(x.shape)
            )
        x = x.contiguous()
        x = x.view(x.size()[0], -1)
        return x

    def _flatten_preserve_time(self, x: torch.Tensor):
        if len(x.shape) != 4:
            raise ValueError(
                "The inputs to the Flatten layer must have at least 4 dimensions (e.g. "
                "BxCxIQxT), input shape was {}".format(x.shape)
            )
        channel_dim, time_dim = 1, 3

        x = x.transpose(channel_dim, time_dim)
        x = x.contiguous()
        x = x.view(x.size()[0], x.size()[1], -1)
        return x


# ---- rfml/nn/layers/power_normalization.py ----
class PowerNormalization(nn.Module):
    """Perform average energy per sample (power) normalization."""

    def forward(self, x: torch.Tensor):
        if len(x.shape) != 4:
            raise ValueError(
                "The inputs to the PowerNormalization layer must have 4 dimensions "
                "(BxCxIQxT), input shape was {}".format(x.shape)
            )
        if x.shape[2] != 2:
            raise ValueError(
                "The inputs to the PowerNormalization layer must be 'complex valued' "
                "by having 2 elements in the IQ dimension (BxCxIQxT), input shape was "
                "{}".format(x.shape)
            )

        e = energy(x)
        e = e.view([e.size()[0], e.size()[1], 1, 1])

        return x / torch.sqrt(e)


# ---- rfml/nn/model/base.py (trimmed to what CNN needs) ----
class Model(nn.Module):
    """Base class that all rfml neural network models inherit from."""

    def __init__(self, input_samples: int, n_classes: int):
        super().__init__()
        self._input_samples = input_samples
        self._n_classes = n_classes
        self._frozen = False

    @property
    def input_samples(self):
        return self._input_samples

    @property
    def n_classes(self):
        return self._n_classes


# ---- rfml/nn/model/cnn.py ----
class CNN(Model):
    """Convolutional Neural Network based on the "VT_CNN2" Architecture.

    This network is based off of a network for modulation classification first
    introduced in O'Shea et al and later updated by West/O'Shea and Hauser et al
    to have larger filter sizes.
    """

    def __init__(self, input_samples: int, n_classes: int):
        super().__init__(input_samples, n_classes)

        self.preprocess = PowerNormalization()

        # Batch x 1-channel x IQ x input_samples
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 7),
            padding=(0, 3),
            bias=False,
        )
        self.a1 = nn.ReLU()
        self.n1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=80,
            kernel_size=(2, 7),
            padding=(0, 3),
            bias=True,
        )
        self.a2 = nn.ReLU()
        self.n2 = nn.BatchNorm2d(80)

        # Flatten the input layer down to 1-d
        self.flatten = Flatten()

        # Batch x Features
        self.dense1 = nn.Linear(80 * 1 * input_samples, 256)
        self.a3 = nn.ReLU()
        self.n3 = nn.BatchNorm1d(256)

        self.dense2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.preprocess(x)

        x = self.conv1(x)
        x = self.a1(x)
        x = self.n1(x)

        x = self.conv2(x)
        x = self.a2(x)
        x = self.n2(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.a3(x)
        x = self.n3(x)

        x = self.dense2(x)

        return x


def build_rfml_cnn():
    torch.manual_seed(0)
    model = CNN(input_samples=32, n_classes=11)
    model.eval()
    return model


def example_input_rfml_cnn():
    torch.manual_seed(0)
    # Batch x 1-channel x IQ(2) x input_samples; PowerNormalization requires the IQ
    # dim to have exactly 2 elements.
    return torch.randn(2, 1, 2, 32)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("rfml-VT-CNN2", "build_rfml_cnn", "example_input_rfml_cnn", 2016, MENAGERIE_ZOO),
]
