# SOURCE: vendored from yinkalario/Two-Stage-Polyphonic-Sound-Event-Detection-and-Localization
# @ master (models/CRNNs.py, models/model_utilities.py)
#
# CRNN10 is the stage-1 joint sound event detection (SED) + direction-of-arrival (DOA)
# estimation CRNN from the paper "A Two-Stage Approach to Polyphonic Sound Event Detection and
# Localization" (Yin Cao, Qiuqiang Kong, Turab Iqbal, Fengyan An, Wenwu Wang, Mark D. Plumbley,
# DCASE2019 workshop). It stacks 4 Conv2D blocks (Conv-BN-ReLU x2 + avg-pool per block) over a
# multichannel log-mel input, average-pools over the frequency axis, feeds a bidirectional GRU
# over time, and branches into three linear heads (event activity / azimuth / elevation), each
# nearest-neighbor upsampled back to the original frame rate.
#
# Vendored verbatim from the real repo (torch/numpy only, no non-base deps):
#   - `models/model_utilities.py::ConvBlock` (the Conv-BN-ReLU-BN-ReLU-pool block)
#   - `models/model_utilities.py::init_layer` / `init_gru` (weight init, kept for fidelity)
#   - `models/CRNNs.py::CRNN10` (the flagship two-stage-paper model)
#
# One mechanical (non-architectural) fix: `models/model_utilities.py::interpolate` calls
# numpy-style `x[:, :, None, :].repeat(ratio, axis=2)` on a torch.Tensor. `torch.Tensor.repeat`
# has no `axis=` kwarg (this is dead/never-exercised code in the original repo -- calling
# `model(batch_x)` in `main.py` hits this exact line and would raise a TypeError against any
# PyTorch version), so it is translated to the semantically identical
# `torch.repeat_interleave(x, ratio, dim=2)` to make the real, unmodified architecture actually
# runnable for tracing. No layer, mechanism, or shape semantics are changed.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Vendored from models/model_utilities.py
# ---------------------------------------------------------------------------
def interpolate(x, ratio):
    """Interpolate x to have equal time steps as targets.

    Input: x: (batch_size, time_steps, class_num)
    Output: out: (batch_size, time_steps*ratio, class_num)

    NOTE: mechanically ported from numpy-style `x[:, :, None, :].repeat(ratio, axis=2)` (which
    does not run against a torch.Tensor -- `Tensor.repeat` has no `axis` kwarg) to the
    semantically identical `torch.repeat_interleave`. See module header.
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = torch.repeat_interleave(x[:, :, None, :], ratio, dim=2)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def init_layer(layer, nonlinearity="leaky_relu"):
    """Initialize a layer."""
    classname = layer.__class__.__name__
    if (classname.find("Conv") != -1) or (classname.find("Linear") != -1):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0.0)


def init_gru(rnn):
    """Initialize a GRU layer."""

    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
        for i, init_func in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])

    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, "fan_in")
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, "weight_ih_l{}".format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform],
        )
        torch.nn.init.constant_(getattr(rnn, "bias_ih_l{}".format(i)), 0)

        _concat_init(
            getattr(rnn, "weight_hh_l{}".format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_],
        )
        torch.nn.init.constant_(getattr(rnn, "bias_hh_l{}".format(i)), 0)


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.bn1)
        init_layer(self.bn2)

    def forward(self, x, pool_type="avg", pool_size=(2, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "frac":
            fractional_maxpool2d = nn.FractionalMaxPool2d(
                kernel_size=pool_size, output_ratio=1 / (2**0.5)
            )
            x = fractional_maxpool2d(x)

        return x


# ---------------------------------------------------------------------------
# Vendored from models/CRNNs.py::CRNN10
# ---------------------------------------------------------------------------
class CRNN10(nn.Module):
    def __init__(
        self, class_num, pool_type="avg", pool_size=(2, 2), interp_ratio=16, pretrained_path=None
    ):
        super().__init__()

        self.class_num = class_num
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.interp_ratio = interp_ratio

        self.conv_block1 = ConvBlock(in_channels=10, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.gru = nn.GRU(
            input_size=512, hidden_size=256, num_layers=1, batch_first=True, bidirectional=True
        )

        self.event_fc = nn.Linear(512, class_num, bias=True)
        self.azimuth_fc = nn.Linear(512, class_num, bias=True)
        self.elevation_fc = nn.Linear(512, class_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_gru(self.gru)
        init_layer(self.event_fc)
        init_layer(self.azimuth_fc)
        init_layer(self.elevation_fc)

    def forward(self, x):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""

        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block4(x, self.pool_type, pool_size=self.pool_size)
        """(batch_size, feature_maps, time_steps, mel_bins)"""

        if self.pool_type == "avg":
            x = torch.mean(x, dim=3)
        elif self.pool_type == "max":
            (x, _) = torch.max(x, dim=3)
        """(batch_size, feature_maps, time_steps)"""

        x = x.transpose(1, 2)
        """ (batch_size, time_steps, feature_maps):"""

        (x, _) = self.gru(x)

        event_output = torch.sigmoid(self.event_fc(x))
        azimuth_output = self.azimuth_fc(x)
        elevation_output = self.elevation_fc(x)
        """(batch_size, time_steps, class_num)"""

        # Interpolate
        event_output = interpolate(event_output, self.interp_ratio)
        azimuth_output = interpolate(azimuth_output, self.interp_ratio)
        elevation_output = interpolate(elevation_output, self.interp_ratio)

        output = {
            "events": event_output,
            "doas": torch.cat((azimuth_output, elevation_output), dim=-1),
        }

        return output


def build_seld_crnn10():
    model = CRNN10(class_num=11, pool_type="avg", pool_size=(2, 2), interp_ratio=16)
    model.eval()
    return model


def example_input_seld_crnn10():
    # (batch_size, mic_channels, time_steps, mel_bins); mic_channels=10 matches CRNN10's
    # fixed conv_block1 in_channels=10 (real repo's multichannel log-mel + intensity-vector
    # feature stack for the ambisonic/mic-array DCASE2019 Task 3 SELD setup).
    return torch.randn(1, 10, 16, 32)


MENAGERIE_ENTRIES = [
    ("Acoustic SELD-DCNN", "build_seld_crnn10", "example_input_seld_crnn10", 2019, "CODE"),
]
