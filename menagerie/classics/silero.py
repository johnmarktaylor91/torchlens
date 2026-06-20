"""Silero audio models: Silero-VAD (voice activity detection) and Silero-STT.

Source: snakers4/silero-vad and snakers4/silero-models.  The published weights
are JIT-packaged, but the architectures are public; a random-init reimplementation
of the architecture is the faithful atlas artifact.

**Silero-VAD (v5):** a fixed-DFT STFT front-end (a Conv1d whose kernel is the DFT
basis -> magnitude), a 4-block Conv1d+ReLU encoder, a single-layer LSTM decoder,
then a 1x1 conv + sigmoid producing a per-frame speech probability.  ~roughly
300K parameters.

**Silero-STT:** a QuartzNet-style (Kriman et al. 2019, arXiv:1910.10261) 1D-conv
CTC acoustic model.  Time-channel-separable Conv1d (depthwise + pointwise) +
BatchNorm + ReLU modules, repeated R times per block, with a residual added before
the last ReLU; B unique blocks (the canonical "15x5" = 5 blocks repeated 3x = 15
TCSConv blocks of R=5) plus a strided prologue and pointwise epilogue ending in a
CTC vocabulary projection.

Random init, CPU, forward-only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Silero-VAD (v5)
# ============================================================


class STFTFrontEnd(nn.Module):
    """Fixed-basis STFT: a Conv1d with DFT kernels -> magnitude spectrum."""

    def __init__(self, n_fft: int = 256, hop: int = 128, n_freq: int = 129) -> None:
        super().__init__()
        self.n_freq = n_freq
        # Conv1d producing 2*n_freq channels (real, imag), kernel = DFT basis.
        weight = torch.zeros(2 * n_freq, 1, n_fft)
        idx = torch.arange(n_fft)
        for f in range(n_freq):
            weight[f, 0] = torch.cos(2 * math.pi * f * idx / n_fft)
            weight[n_freq + f, 0] = -torch.sin(2 * math.pi * f * idx / n_fft)
        self.conv = nn.Conv1d(1, 2 * n_freq, kernel_size=n_fft, stride=hop, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        self.conv.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L)
        spec = self.conv(x)  # (B, 2*n_freq, frames)
        re, im = spec[:, : self.n_freq], spec[:, self.n_freq :]
        return torch.sqrt(re * re + im * im + 1e-9)  # magnitude (B, n_freq, frames)


class SileroVAD(nn.Module):
    """Silero VAD v5: STFT -> 4 conv blocks -> LSTM -> 1x1 conv -> sigmoid."""

    def __init__(self, n_freq: int = 129) -> None:
        super().__init__()
        self.stft = STFTFrontEnd(n_fft=256, hop=128, n_freq=n_freq)
        self.enc0 = nn.Conv1d(n_freq, 128, 3, stride=1, padding=1)
        self.enc1 = nn.Conv1d(128, 64, 3, stride=2, padding=1)
        self.enc2 = nn.Conv1d(64, 64, 3, stride=2, padding=1)
        self.enc3 = nn.Conv1d(64, 128, 3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.out = nn.Conv1d(128, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L) raw audio
        h = self.stft(x)
        h = F.relu(self.enc0(h))
        h = F.relu(self.enc1(h))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))  # (B, 128, frames)
        h = h.transpose(1, 2)  # (B, frames, 128)
        h, _ = self.lstm(h)
        h = F.relu(h).transpose(1, 2)  # (B, 128, frames)
        return torch.sigmoid(self.out(h))  # (B, 1, frames)


# ============================================================
# Silero-STT (QuartzNet 15x5)
# ============================================================


class TCSConv(nn.Module):
    """Time-channel-separable Conv1d: depthwise + pointwise + BN + (ReLU)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        dilation: int = 1,
        relu: bool = True,
    ) -> None:
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.depthwise = nn.Conv1d(
            in_ch,
            in_ch,
            kernel,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=in_ch,
            bias=False,
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.pointwise(self.depthwise(x)))
        return F.relu(x) if self.relu else x


class QuartzBlock(nn.Module):
    """B_k block: R TCSConv modules with a residual added before the final ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, repeats: int = 5) -> None:
        super().__init__()
        mods = []
        for r in range(repeats):
            ic = in_ch if r == 0 else out_ch
            # last module: no internal ReLU (residual added before ReLU)
            mods.append(TCSConv(ic, out_ch, kernel, relu=(r != repeats - 1)))
        self.modules_list = nn.ModuleList(mods)
        self.res_pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.res_bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.res_bn(self.res_pw(x))
        h = x
        for m in self.modules_list:
            h = m(h)
        return F.relu(h + res)


class SileroSTT(nn.Module):
    """QuartzNet-15x5 CTC acoustic model (Silero-STT structural template)."""

    def __init__(self, n_mels: int = 64, vocab: int = 32) -> None:
        super().__init__()
        # C1 prologue: strided separable conv
        self.c1 = TCSConv(n_mels, 256, kernel=33, stride=2)
        # 5 unique blocks (canonical channels/kernels), each repeated 3x -> 15
        block_cfg = [(256, 256, 33), (256, 256, 39), (256, 512, 51), (512, 512, 63), (512, 512, 75)]
        blocks = []
        for out_first, out_ch, k in block_cfg:
            in_ch = out_first
            for rep in range(3):
                blocks.append(QuartzBlock(in_ch if rep == 0 else out_ch, out_ch, k, repeats=5))
                in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        # epilogue
        self.c2 = TCSConv(512, 512, kernel=87, dilation=2)
        self.c3 = nn.Conv1d(512, 1024, kernel_size=1, bias=False)
        self.c3_bn = nn.BatchNorm1d(1024)
        self.c4 = nn.Conv1d(1024, vocab, kernel_size=1)  # CTC logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_mels, T)
        h = self.c1(x)
        for b in self.blocks:
            h = b(h)
        h = self.c2(h)
        h = F.relu(self.c3_bn(self.c3(h)))
        return self.c4(h)  # (B, vocab, T')


def build_silero_vad() -> nn.Module:
    return SileroVAD()


def build_silero_stt() -> nn.Module:
    return SileroSTT()


def example_input() -> torch.Tensor:
    """Raw audio ``(1, 1, 2048)`` for Silero-VAD."""
    return torch.randn(1, 1, 2048)


def example_input_stt() -> torch.Tensor:
    """Log-mel feature ``(1, 64, 200)`` for Silero-STT."""
    return torch.randn(1, 64, 200)


MENAGERIE_ENTRIES = [
    (
        "Silero-VAD (STFT + conv + LSTM voice activity detector)",
        "build_silero_vad",
        "example_input",
        "2021",
        "DC",
    ),
    (
        "Silero-STT (QuartzNet-15x5 CTC speech-to-text)",
        "build_silero_stt",
        "example_input_stt",
        "2021",
        "DC",
    ),
]
