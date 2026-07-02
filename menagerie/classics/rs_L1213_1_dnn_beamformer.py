# SOURCE: vendored from pytorch/audio @ release/2.1 (examples/dnn_beamformer/model.py)
#
# The DNNBeamformer architecture in the original example composes:
#   - torchaudio.transforms.Spectrogram / InverseSpectrogram / PSD / SoudenMVDR (base lib, used as-is)
#   - a TDConvNet mask-estimation network from the `asteroid` package (NOT installed as a base
#     lib here). `asteroid` is not otherwise needed by torchlens/menagerie; rather than adding a
#     new dependency, the TDConvNet mask network (Conv-TasNet's temporal conv-net, "Conv-TasNet:
#     Surpassing ideal time-frequency magnitude masking for speech separation", Luo & Mesgarani,
#     TASLP 2019) is vendored verbatim below from asteroid-team/asteroid @ master
#     (asteroid/masknn/convolutional.py, asteroid/masknn/norms.py, asteroid/masknn/activations.py,
#     asteroid/utils/generic_utils.py::has_arg). These vendored pieces have ONLY torch/numpy
#     dependencies (no architectural changes -- straight copy of the real classes/functions).
#
# MENAGERIE_ZOO = "vendored-pytorch"

from functools import partial
from typing import List

import numpy as np
import torch
from torch import nn
from torchaudio.transforms import PSD, InverseSpectrogram, SoudenMVDR, Spectrogram

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# Vendored from asteroid/utils/generic_utils.py (asteroid-team/asteroid @ master)
# ---------------------------------------------------------------------------
def has_arg(fn, name):
    """Checks if a callable accepts a given keyword argument."""
    import inspect

    signature = inspect.signature(fn)
    parameter = signature.parameters.get(name)
    if parameter is None:
        return False
    return parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


# ---------------------------------------------------------------------------
# Vendored from asteroid/masknn/norms.py (asteroid-team/asteroid @ master)
# Only the pieces reachable from norm_type="gLN" (DNNBeamformer's default) are kept.
# ---------------------------------------------------------------------------
def z_norm(x, dims: List[int], eps: float = 1e-8):
    mean = x.mean(dim=dims, keepdim=True)
    var2 = torch.var(x, dim=dims, keepdim=True, unbiased=False)
    value = (x - mean) / torch.sqrt((var2 + eps))
    return value


def _glob_norm(x, eps: float = 1e-8):
    dims: List[int] = list(range(1, len(x.shape)))
    return z_norm(x, dims, eps)


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """Assumes input of size `[batch, chanel, *]`."""
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x, EPS: float = 1e-8):
        value = _glob_norm(x, eps=EPS)
        return self.apply_gain_and_bias(value)


def norms_get(identifier):
    """Returns a norm class from a string (mirrors asteroid.masknn.norms.get)."""
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        registry = {"gLN": GlobLN, "GlobLN": GlobLN}
        cls = registry.get(identifier)
        if cls is None:
            raise ValueError("Could not interpret normalization identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret normalization identifier: " + str(identifier))


# ---------------------------------------------------------------------------
# Vendored from asteroid/masknn/activations.py (asteroid-team/asteroid @ master)
# Only the pieces reachable from mask_act="relu"/"linear" (DNNBeamformer usage) are kept.
# ---------------------------------------------------------------------------
def linear():
    return nn.Identity()


def relu():
    return nn.ReLU()


def activations_get(identifier):
    """Returns an activation function from a string (mirrors asteroid.masknn.activations.get)."""
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        registry = {"linear": linear, "relu": relu}
        cls = registry.get(identifier)
        if cls is None:
            raise ValueError("Could not interpret activation identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret activation identifier: " + str(identifier))


# ---------------------------------------------------------------------------
# Vendored from asteroid/masknn/convolutional.py (asteroid-team/asteroid @ master)
# ---------------------------------------------------------------------------
class _Chop1d(nn.Module):
    """To ensure the output length is the same as the input."""

    def __init__(self, chop_size):
        super().__init__()
        self.chop_size = chop_size

    def forward(self, x):
        return x[..., : -self.chop_size].contiguous()


class Conv1DBlock(nn.Module):
    """One dimensional convolutional block, as proposed in Conv-TasNet."""

    def __init__(
        self,
        in_chan,
        hid_chan,
        skip_out_chan,
        kernel_size,
        padding,
        dilation,
        norm_type="gLN",
        causal=False,
    ):
        super(Conv1DBlock, self).__init__()
        self.skip_out_chan = skip_out_chan
        conv_norm = norms_get(norm_type)
        in_conv1d = nn.Conv1d(in_chan, hid_chan, 1)
        depth_conv1d = nn.Conv1d(
            hid_chan, hid_chan, kernel_size, padding=padding, dilation=dilation, groups=hid_chan
        )
        if causal:
            depth_conv1d = nn.Sequential(depth_conv1d, _Chop1d(padding))
        self.shared_block = nn.Sequential(
            in_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
            depth_conv1d,
            nn.PReLU(),
            conv_norm(hid_chan),
        )
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)
        if skip_out_chan:
            self.skip_conv = nn.Conv1d(hid_chan, skip_out_chan, 1)

    def forward(self, x):
        r"""Input shape $(batch, feats, seq)$."""
        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        if not self.skip_out_chan:
            return res_out
        skip_out = self.skip_conv(shared_out)
        return res_out, skip_out


class TDConvNet(nn.Module):
    """Temporal Convolutional network used in ConvTasnet.

    Reference: "Conv-TasNet: Surpassing ideal time-frequency magnitude masking for speech
    separation" TASLP 2019, Yi Luo, Nima Mesgarani. https://arxiv.org/abs/1809.07454
    """

    def __init__(
        self,
        in_chan,
        n_src,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="gLN",
        mask_act="relu",
        causal=False,
    ):
        super(TDConvNet, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.skip_chan = skip_chan
        self.conv_kernel_size = conv_kernel_size
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.causal = causal

        layer_norm = norms_get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                if not causal:
                    padding = (conv_kernel_size - 1) * 2**x // 2
                else:
                    padding = (conv_kernel_size - 1) * 2**x
                self.TCN.append(
                    Conv1DBlock(
                        bn_chan,
                        hid_chan,
                        skip_chan,
                        conv_kernel_size,
                        padding=padding,
                        dilation=2**x,
                        norm_type=norm_type,
                        causal=causal,
                    )
                )
        mask_conv_inp = skip_chan if skip_chan else bn_chan
        mask_conv = nn.Conv1d(mask_conv_inp, n_src * out_chan, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)
        mask_nl_class = activations_get(mask_act)
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        batch, _, n_frames = mixture_w.size()
        output = self.bottleneck(mixture_w)
        skip_connection = torch.tensor([0.0], device=output.device)
        for layer in self.TCN:
            tcn_out = layer(output)
            if self.skip_chan:
                residual, skip = tcn_out
                skip_connection = skip_connection + skip
            else:
                residual = tcn_out
            output = output + residual
        mask_inp = skip_connection if self.skip_chan else output
        score = self.mask_net(mask_inp)
        score = score.view(batch, self.n_src, self.out_chan, n_frames)
        est_mask = self.output_act(score)
        return est_mask


# ---------------------------------------------------------------------------
# Real class, vendored from pytorch/audio @ release/2.1
# examples/dnn_beamformer/model.py::DNNBeamformer (only the Lightning training wrapper is
# dropped -- not part of the architecture).
# ---------------------------------------------------------------------------
class DNNBeamformer(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, ref_channel: int = 0):
        super().__init__()
        self.stft = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
        self.istft = InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)
        self.mask_net = TDConvNet(
            n_fft // 2 + 1,
            2,
            out_chan=n_fft // 2 + 1,
            causal=False,
            mask_act="linear",
            norm_type="gLN",
        )
        self.beamformer = SoudenMVDR()
        self.psd = PSD()
        self.ref_channel = ref_channel

    def forward(self, mixture) -> torch.Tensor:
        spectrum = self.stft(mixture)  # (batch, channel, time, freq)
        batch, _, freq, time = spectrum.shape
        input_feature = torch.log(spectrum[:, self.ref_channel].abs() + 1e-8)  # (batch, freq, time)
        mask = torch.nn.functional.relu(self.mask_net(input_feature))  # (batch, 2, freq, time)
        mask_speech = mask[:, 0]
        mask_noise = mask[:, 1]
        psd_speech = self.psd(spectrum, mask_speech)
        psd_noise = self.psd(spectrum, mask_noise)
        enhanced_stft = self.beamformer(spectrum, psd_speech, psd_noise, self.ref_channel)
        enhanced_waveform = self.istft(enhanced_stft, length=mixture.shape[-1])
        return enhanced_waveform


def build_dnn_beamformer():
    # Tiny n_fft keeps the TDConvNet mask network (8 blocks x 3 repeats by default) cheap;
    # override n_blocks/n_repeats down for tracing speed via a small TDConvNet substitution.
    model = DNNBeamformer(n_fft=64, hop_length=32, ref_channel=0)
    # Shrink the vendored TDConvNet to a tiny size for fast tracing (architecture unchanged,
    # only depth/width hyperparameters -- same knobs the original constructor exposes).
    model.mask_net = TDConvNet(
        64 // 2 + 1,
        2,
        out_chan=64 // 2 + 1,
        n_blocks=2,
        n_repeats=1,
        bn_chan=8,
        hid_chan=16,
        skip_chan=8,
        causal=False,
        mask_act="linear",
        norm_type="gLN",
    )
    model.eval()
    return model


def example_input_dnn_beamformer():
    # (batch, mic_channels, time) multi-channel waveform, as consumed by Spectrogram(power=None).
    return torch.randn(1, 4, 2048)


MENAGERIE_ENTRIES = [
    (
        "Acoustic Beamformer via Complex-valued DNN",
        "build_dnn_beamformer",
        "example_input_dnn_beamformer",
        2021,
        "CODE",
    ),
]
