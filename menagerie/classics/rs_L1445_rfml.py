# SOURCE: vendored from brysef/rfml @ dbd3db2f6090232ebebc9d7107d990dd04fcb323
# https://raw.githubusercontent.com/brysef/rfml/dbd3db2f6090232ebebc9d7107d990dd04fcb323/rfml/nn/model/cnn.py
# https://raw.githubusercontent.com/brysef/rfml/dbd3db2f6090232ebebc9d7107d990dd04fcb323/rfml/nn/model/cldnn.py
# https://raw.githubusercontent.com/brysef/rfml/dbd3db2f6090232ebebc9d7107d990dd04fcb323/rfml/nn/model/base.py
# https://raw.githubusercontent.com/brysef/rfml/dbd3db2f6090232ebebc9d7107d990dd04fcb323/rfml/nn/layers/flatten.py
# https://raw.githubusercontent.com/brysef/rfml/dbd3db2f6090232ebebc9d7107d990dd04fcb323/rfml/nn/layers/power_normalization.py
# https://raw.githubusercontent.com/brysef/rfml/dbd3db2f6090232ebebc9d7107d990dd04fcb323/rfml/nn/F/energy.py
#
# rfml (Bryse Flowers, "Radio Frequency Machine Learning" toolkit for RF fingerprinting
# / automatic modulation classification, VT). CNN is the "VT_CNN2" architecture from
# O'Shea et al. "Convolutional radio modulation recognition networks" (2016), updated
# by West/O'Shea and Hauser et al. to use larger filters, and ported to PyTorch by
# Bryse Flowers (dropping the first conv's bias due to vanishing-gradient issues
# observed only in the PyTorch port). CLDNN is the "Convolutional Long Deep Neural
# Network" from West/O'Shea "Deep architectures for modulation recognition" (2017),
# with the documented Flowers modifications: added BatchNorm, filter size fixed at 7
# w/ padding 3, LSTM swapped for GRU, no bias on first conv, GRU hidden size == n_classes.
#
# Both classes are transcribed verbatim from the real repo modules (rfml/nn/model/cnn.py,
# rfml/nn/model/cldnn.py) plus their real supporting layers (rfml/nn/layers/flatten.py
# PowerNormalization from rfml/nn/layers/power_normalization.py, and the energy() helper
# from rfml/nn/F/energy.py) that CNN.preprocess/CLDNN rely on. No architectural changes;
# only the package-relative imports were flattened into this single file and Model's
# save/load/predict/outputs conveniences (torch.load/save I/O, not part of the traced
# architecture) were dropped -- __init__ and forward() are unchanged. Input convention
# (Batch x Channel=1 x IQ=2 x input_samples) matches the docstrings on PowerNormalization,
# Flatten, and CNN/CLDNN's first Conv2d layers.

import torch
import torch.nn as nn


def energy(x: torch.Tensor, sps: float = 1.0) -> torch.Tensor:
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

    # This Tensor still has an unnecessary singleton dimensions in IQ
    x = x.squeeze(dim=iq_dim)

    return x


class Flatten(nn.Module):
    """Flatten the channel, IQ, and time dims into a single feature dim.

    Assumes input structured as Batch x Channel x IQ x Time.
    """

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

        # BxCxIQxT
        x = x.transpose(channel_dim, time_dim)
        # BxTxCxIQ -- Can now collapse the final two dimensions
        x = x.contiguous()
        x = x.view(x.size()[0], x.size()[1], -1)
        return x


class PowerNormalization(nn.Module):
    """Perform average energy per sample (power) normalization.

    Assumes signal structured as Batch x Channel x IQ x Time.
    """

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
        # Divide each channel of each example by the sqrt of the power of that
        # channel/example pair.
        e = e.view([e.size()[0], e.size()[1], 1, 1])

        return x / torch.sqrt(e)


class CNN(nn.Module):
    """Convolutional Neural Network based on the "VT_CNN2" Architecture.

    Network for modulation classification first introduced in O'Shea et al. and
    later updated by West/O'Shea and Hauser et al. to have larger filter sizes.
    PowerNormalization is folded into the network (rather than a pre-processing
    stage) as a simplification made by Bryse Flowers.
    """

    def __init__(self, input_samples: int, n_classes: int):
        super().__init__()
        self._input_samples = input_samples
        self._n_classes = n_classes

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


class CLDNN(nn.Module):
    """Convolutional Long Deep Neural Network (CNN + GRU + MLP).

    Network for modulation classification first introduced in West/O'Shea, with
    documented modifications by Bryse Flowers (BatchNorm added, filter size fixed
    at 7 w/ padding 3, GRU in place of LSTM, no bias on first conv, GRU hidden
    size == n_classes).
    """

    def __init__(self, input_samples: int, n_classes: int):
        super().__init__()
        self._input_samples = input_samples
        self._n_classes = n_classes

        # Batch x 1-channel x IQ x input_samples
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=50,
            kernel_size=(1, 7),
            padding=(0, 3),
            bias=False,
        )
        self.a1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(50)

        self.conv2 = nn.Conv2d(
            in_channels=50,
            out_channels=50,
            kernel_size=(1, 7),
            padding=(0, 3),
            bias=True,
        )
        self.a2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(50)

        self.conv3 = nn.Conv2d(
            in_channels=50,
            out_channels=50,
            kernel_size=(1, 7),
            padding=(0, 3),
            bias=True,
        )
        self.a3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(50)

        # Flatten along channels and I/Q
        self.flatten_preserve_time = Flatten(preserve_time=True)

        self.GRU_n_layers = 1
        self.GRU_n_directions = 1
        self.GRU_hidden_size = n_classes
        self.gru = nn.GRU(
            input_size=100 * 2,  # 100 channels after concatenation (50+50) * IQ (2)
            hidden_size=self.GRU_hidden_size,
            batch_first=True,
            num_layers=self.GRU_n_layers,
            bidirectional=False,
        )

        # Flatten everything outside of batch dimension
        self.flatten = Flatten()

        self.dense1 = nn.Linear(input_samples * self.GRU_hidden_size * self.GRU_n_directions, 256)
        self.a4 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(256)

        self.dense2 = nn.Linear(256, n_classes)

    def forward(self, x):
        channel_dim = 1
        batch_size = x.shape[0]

        # Up front "filter" with no bias
        x = self.conv1(x)
        x = self.a1(x)
        a = self.bn1(x)  # Output is concatenated back as a "skip connection" below

        # Convolutional feature extraction layers
        x = self.conv2(a)
        x = self.a2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.a3(x)
        x = self.bn3(x)

        # Concatenate the "skip connection" with the output of the rest of the CNN
        x = torch.cat((a, x), dim=channel_dim)

        # Temporal feature extraction
        x = self.flatten_preserve_time(x)  # BxTxF
        hidden = x.new_zeros(
            self.GRU_n_layers * self.GRU_n_directions, batch_size, self.GRU_hidden_size
        )
        x, _ = self.gru(x, hidden)

        # MLP Classification stage
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.a4(x)
        x = self.bn4(x)

        x = self.dense2(x)

        return x


def build_rfml_cnn():
    torch.manual_seed(0)
    model = CNN(input_samples=128, n_classes=10)
    model.eval()
    return model


def example_input_rfml_cnn():
    torch.manual_seed(0)
    # Batch x Channel=1 x IQ=2 x input_samples
    return torch.randn(2, 1, 2, 128)


def build_rfml_cldnn():
    torch.manual_seed(0)
    model = CLDNN(input_samples=128, n_classes=10)
    model.eval()
    return model


def example_input_rfml_cldnn():
    torch.manual_seed(0)
    return torch.randn(2, 1, 2, 128)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("RFML-CNN-VTCNN2", "build_rfml_cnn", "example_input_rfml_cnn", 2018, MENAGERIE_ZOO),
    ("RFML-CLDNN", "build_rfml_cldnn", "example_input_rfml_cldnn", 2018, MENAGERIE_ZOO),
]
