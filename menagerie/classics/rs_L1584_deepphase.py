# SOURCE: vendored from sebastianstarke/AI4Animation @ master
# https://raw.githubusercontent.com/sebastianstarke/AI4Animation/master/AI4Animation/SIGGRAPH_2022/PyTorch/PAE/PAE.py
# https://raw.githubusercontent.com/sebastianstarke/AI4Animation/master/AI4Animation/SIGGRAPH_2022/PyTorch/Library/Utility.py
# (LN_v2, the layer-norm variant PAE.py imports as `utility.LN_v2`)
#
# DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds (Starke et
# al., SIGGRAPH 2022). `Model` is the real Periodic Autoencoder: a 1D-conv
# encoder over a temporal window of per-joint motion curves, an FFT-based phase
# extraction per latent channel (frequency/amplitude/offset/phase via
# `torch.fft.rfft`), reconstruction of a parametric sinusoidal signal from those
# phase parameters, and a symmetric 1D-conv decoder back to the input curves.
# Vendored verbatim from the official repo's `SIGGRAPH_2022/PyTorch/PAE/PAE.py`
# (the module class only; the `Network.py` training-loop/plotting driver is not
# part of the architecture and is omitted) plus `LN_v2` from `Library/Utility.py`.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

MENAGERIE_ZOO = "vendored-pytorch"


# ---- Library/Utility.py (LN_v2, vendored verbatim) ----
class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


# ---- PAE/PAE.py (Model, vendored verbatim) ----
class Model(nn.Module):
    def __init__(self, input_channels, embedding_channels, time_range, window):
        super(Model, self).__init__()
        self.input_channels = input_channels
        self.embedding_channels = embedding_channels
        self.time_range = time_range
        self.window = window

        self.tpi = Parameter(
            torch.from_numpy(np.array([2.0 * np.pi], dtype=np.float32)), requires_grad=False
        )
        self.args = Parameter(
            torch.from_numpy(
                np.linspace(-self.window / 2, self.window / 2, self.time_range, dtype=np.float32)
            ),
            requires_grad=False,
        )
        self.freqs = Parameter(
            torch.fft.rfftfreq(time_range)[1:] * time_range / self.window, requires_grad=False
        )  # Remove DC frequency

        intermediate_channels = int(input_channels / 3)

        self.conv1 = nn.Conv1d(
            input_channels,
            intermediate_channels,
            time_range,
            stride=1,
            padding=int((time_range - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        self.norm1 = LN_v2(time_range)
        self.conv2 = nn.Conv1d(
            intermediate_channels,
            embedding_channels,
            time_range,
            stride=1,
            padding=int((time_range - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )

        self.fc = torch.nn.ModuleList()
        for _ in range(embedding_channels):
            self.fc.append(nn.Linear(time_range, 2))

        self.deconv1 = nn.Conv1d(
            embedding_channels,
            intermediate_channels,
            time_range,
            stride=1,
            padding=int((time_range - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )
        self.denorm1 = LN_v2(time_range)
        self.deconv2 = nn.Conv1d(
            intermediate_channels,
            input_channels,
            time_range,
            stride=1,
            padding=int((time_range - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )

    # Returns the frequency for a function over a time window in s
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:, :, 1:]  # Spectrum without DC component
        power = spectrum**2

        # Frequency
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)

        # Amplitude
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

        # Offset
        offset = rfft.real[:, :, 0] / self.time_range  # DC component

        return freq, amp, offset

    def forward(self, x):
        y = x

        # Signal Embedding
        y = y.reshape(y.shape[0], self.input_channels, self.time_range)

        y = self.conv1(y)
        y = self.norm1(y)
        y = F.elu(y)

        y = self.conv2(y)

        latent = y  # Save latent for returning

        # Frequency, Amplitude, Offset
        f, a, b = self.FFT(y, dim=2)

        # Phase
        p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
        for i in range(self.embedding_channels):
            v = self.fc[i](y[:, i, :])
            p[:, i] = torch.atan2(v[:, 1], v[:, 0]) / self.tpi

        # Parameters
        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
        params = [p, f, a, b]  # Save parameters for returning

        # Latent Reconstruction
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b

        signal = y  # Save signal for returning

        # Signal Reconstruction
        y = self.deconv1(y)
        y = self.denorm1(y)
        y = F.elu(y)

        y = self.deconv2(y)

        y = y.reshape(y.shape[0], self.input_channels * self.time_range)

        return y, latent, signal, params


# ---- staging wrapper ----
# Real default config (Network.py): window=2.0s, fps=60, joints=26 -> frames=121,
# input_channels=3*26=78, phase_channels=5. Shrunk here to a tiny window/joint
# count for a fast trace; the architecture (conv encoder -> FFT phase extraction
# -> parametric-sinusoid reconstruction -> conv decoder) is unmodified.
_WINDOW = 1.0
_FPS = 10
_JOINTS = 4
_PHASE_CHANNELS = 2


def build_deepphase():
    frames = int(_WINDOW * _FPS) + 1
    input_channels = 3 * _JOINTS
    model = Model(
        input_channels=input_channels,
        embedding_channels=_PHASE_CHANNELS,
        time_range=frames,
        window=_WINDOW,
    )
    model.eval()
    return model


def example_input_deepphase():
    frames = int(_WINDOW * _FPS) + 1
    input_channels = 3 * _JOINTS
    batch = 2
    torch.manual_seed(0)
    # PAE.forward reshapes x to (batch, input_channels, time_range) internally,
    # so the raw input is flattened exactly as `LoadBatches` produces it in the
    # real training script.
    x = torch.randn(batch, input_channels * frames)
    return x


MENAGERIE_ENTRIES = [
    ("DeepPhase", "build_deepphase", "example_input_deepphase", 2022, "vendored-pytorch"),
]
