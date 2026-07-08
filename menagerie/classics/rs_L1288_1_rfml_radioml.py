# SOURCE: vendored from https://github.com/brysef/rfml @ master
#   Vendored files: rfml/nn/model/base.py (Model base class, trimmed to the
#   torch.nn.Module surface -- save/load/predict/outputs/__del__ helpers that touch
#   the filesystem or Variable() are dropped since they are not part of the forward
#   graph), rfml/nn/model/cnn.py (CNN / "VT_CNN2"), rfml/nn/model/cldnn.py (CLDNN),
#   rfml/nn/layers/flatten.py (Flatten), rfml/nn/layers/power_normalization.py
#   (PowerNormalization), and rfml/nn/F/energy.py (energy helper that
#   PowerNormalization calls). No architectural changes -- only relative imports
#   were flattened into this single file.
#
# "DeepSig RadioML" family: brysef/rfml (Bryse Flowers, Virginia Tech) is a PyTorch
# research library for RF modulation-recognition on the RadioML datasets. It ships
# two real classifier architectures used across the RadioML literature:
#   - CNN ("VT_CNN2"): the O'Shea et al. (2016) / West & O'Shea (2017) / Hauser et
#     al. (2017) raw-IQ 2D-CNN modulation classifier (2x conv -> 2x dense), with a
#     built-in PowerNormalization pre-processing layer.
#   - CLDNN: West & O'Shea's Convolutional Long Deep Neural Network (3x conv with a
#     concatenated skip connection -> GRU over time -> 2x dense).
#
# Ref: https://github.com/brysef/rfml/blob/master/rfml/nn/model/base.py
# Ref: https://github.com/brysef/rfml/blob/master/rfml/nn/model/cnn.py
# Ref: https://github.com/brysef/rfml/blob/master/rfml/nn/model/cldnn.py
# Ref: https://github.com/brysef/rfml/blob/master/rfml/nn/layers/flatten.py
# Ref: https://github.com/brysef/rfml/blob/master/rfml/nn/layers/power_normalization.py
# Ref: https://github.com/brysef/rfml/blob/master/rfml/nn/F/energy.py

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# From rfml/nn/F/energy.py, unmodified.
# ---------------------------------------------------------------------------
def energy(x: torch.Tensor, sps: float = 1.0):
    """Calculate the average energy (per symbol if provided) for each example.

    Assumes the signal is structured as Batch x Channel x IQ x Time.
    """
    if len(x.shape) != 4:
        raise ValueError(
            "The inputs to the energy function must have 4 dimensions (BxCxIQxT), "
            "input shape was {}".format(x.shape)
        )
    if x.shape[2] != 2:
        raise ValueError(
            "The inputs to the energy function must be 'complex valued' by having 2 "
            "elements in the IQ dimension (BxCxIQxT), input shape was "
            "{}".format(x.shape)
        )
    iq_dim = 2
    time_dim = 3

    r, c = x.chunk(chunks=2, dim=iq_dim)
    power = (r * r) + (c * c)  # power is magnitude squared so sqrt cancels

    x = torch.mean(power, dim=time_dim) * sps

    # This Tensor still has an unnecessary singleton dimensions in IQ
    x = x.squeeze(dim=iq_dim)

    return x


# ---------------------------------------------------------------------------
# From rfml/nn/layers/power_normalization.py, unmodified.
# ---------------------------------------------------------------------------
class PowerNormalization(nn.Module):
    """Perform average energy per sample (power) normalization.

    This module assumes that the signal is structured as Batch x Channel x IQ x
    Time, where the power normalization is performed along the T axis using the
    power measured in the complex-valued I/Q dimension. Output shape matches input.
    """

    def forward(self, x: torch.Tensor):
        if len(x.shape) != 4:
            raise ValueError(
                "The inputs to the PowerNormalization layer must have 4 dimensions "
                "(BxCxIQxT), input shape was {}".format(x.shape)
            )
        if x.shape[2] != 2:
            raise ValueError(
                "The inputs to the PowerNormalization layer must be 'complex "
                "valued' by having 2 elements in the IQ dimension (BxCxIQxT), "
                "input shape was {}".format(x.shape)
            )

        e = energy(x)
        # Make the dimensions match so we divide each channel of each example by
        # the sqrt of the power of that channel/example pair
        e = e.view([e.size()[0], e.size()[1], 1, 1])

        return x / torch.sqrt(e)


# ---------------------------------------------------------------------------
# From rfml/nn/layers/flatten.py, unmodified.
# ---------------------------------------------------------------------------
class Flatten(nn.Module):
    """Flatten the channel, IQ, and time dims into a single feature dim.

    Assumes the input signal is structured as Batch x Channel x IQ x Time.
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
                "The inputs to the Flatten layer must have at least 2 dimensions "
                "(e.g. BxCxIQxT), input shape was {}".format(x.shape)
            )
        x = x.contiguous()
        x = x.view(x.size()[0], -1)
        return x

    def _flatten_preserve_time(self, x: torch.Tensor):
        if len(x.shape) != 4:
            raise ValueError(
                "The inputs to the Flatten layer must have at least 4 dimensions "
                "(e.g. BxCxIQxT), input shape was {}".format(x.shape)
            )
        channel_dim, time_dim = 1, 3

        # BxCxIQxT
        x = x.transpose(channel_dim, time_dim)
        # BxTxCxIQ -- Can now collapse the final two dimensions
        x = x.contiguous()
        x = x.view(x.size()[0], x.size()[1], -1)
        return x


# ---------------------------------------------------------------------------
# From rfml/nn/model/base.py (Model), trimmed to the nn.Module forward-graph
# surface. Dropped: __del__, save()/load() (filesystem side effects), predict()/
# outputs() (call Variable()/`.to(self.device)`, not part of the traced graph).
# Kept: the properties/freeze-unfreeze hooks the two model classes rely on.
# ---------------------------------------------------------------------------
class Model(nn.Module):
    """Base class that all rfml neural network models inherit from."""

    def __init__(self, input_samples: int, n_classes: int):
        super().__init__()
        self._input_samples = input_samples
        self._n_classes = n_classes
        self._frozen = False

    def freeze(self):
        self._frozen = True
        if hasattr(self, "_freeze"):
            self._freeze()

    def unfreeze(self):
        self._frozen = False
        if hasattr(self, "_unfreeze"):
            self._unfreeze()

    @property
    def is_frozen(self):
        return self._frozen

    @property
    def input_samples(self):
        return self._input_samples

    @property
    def n_classes(self):
        return self._n_classes


# ---------------------------------------------------------------------------
# From rfml/nn/model/cnn.py (CNN / "VT_CNN2"), unmodified.
# ---------------------------------------------------------------------------
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

    def _freeze(self):
        for name, module in self.named_children():
            if "dense" not in name and "n3" not in name:
                for p in module.parameters():
                    p.requires_grad = False

    def _unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# From rfml/nn/model/cldnn.py (CLDNN), unmodified.
# ---------------------------------------------------------------------------
class CLDNN(Model):
    """Convolutional Long Deep Neural Network (CNN + GRU + MLP).

    This network is based off of a network for modulation classification first
    introduced in West/O'Shea.
    """

    def __init__(self, input_samples: int, n_classes: int):
        super().__init__(input_samples, n_classes)

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

        # Fully connected layers
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
        hidden = x.new(
            self.GRU_n_layers * self.GRU_n_directions,
            batch_size,
            self.GRU_hidden_size,
        )
        hidden.zero_()
        x, _ = self.gru(x, hidden)

        # MLP Classification stage
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.a4(x)
        x = self.bn4(x)

        x = self.dense2(x)

        return x

    def _freeze(self):
        for name, module in self.named_children():
            if "dense" not in name and "bn4" not in name:
                for p in module.parameters():
                    p.requires_grad = False

    def _unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo). RadioML modulation
# classification networks expect BxCxIQxT input (batch, 1-channel, 2-value I/Q,
# time samples); a tiny input_samples=32, n_classes=11 (the classic RadioML2016.10a
# 11-class modulation set) keeps both models fast to trace on CPU.
# ---------------------------------------------------------------------------
def build_rfml_cnn():
    torch.manual_seed(0)
    model = CNN(input_samples=32, n_classes=11)
    model.eval()
    return model


def example_input_rfml_cnn():
    torch.manual_seed(0)
    return torch.randn(2, 1, 2, 32)


def build_rfml_cldnn():
    torch.manual_seed(0)
    model = CLDNN(input_samples=32, n_classes=11)
    model.eval()
    return model


def example_input_rfml_cldnn():
    torch.manual_seed(0)
    return torch.randn(2, 1, 2, 32)


MENAGERIE_ENTRIES = [
    (
        "rfml.CNN (VT_CNN2, RadioML modulation classifier)",
        "build_rfml_cnn",
        "example_input_rfml_cnn",
        2016,
        MENAGERIE_ZOO,
    ),
    (
        "rfml.CLDNN (RadioML modulation classifier)",
        "build_rfml_cldnn",
        "example_input_rfml_cldnn",
        2017,
        MENAGERIE_ZOO,
    ),
]
