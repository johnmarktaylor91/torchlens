# FAITHFUL PORT of anicolson/DeepXi @ master (original framework: TensorFlow 2 / Keras
# -- `tensorflow.keras.layers.Conv1D/LayerNormalization`; this is Keras/TF, not the
# torch base-lib set this menagerie targets, so the architecture is transcribed
# faithfully into self-contained torch below, layer for layer, rather than vendored).
#
# The queue candidate "MMSE-Net"/DeepMMSE refers to Zhang, Nicolson, Wang, Paliwal &
# Wang, "DeepMMSE: A Deep Learning Approach to MMSE-based Noise Power Spectral
# Density Estimation" (IEEE/ACM TASLP 2020). The paper's own repo
# (yunzqq/DeepMMSE) ships only a README pointing to the reference model
# implementation: anicolson/DeepXi ("Deep Xi" -- the a priori SNR / noise-PSD
# estimation network family the DeepMMSE paper builds its MMSE-based noise PSD
# estimator on top of). File used: deepxi/network/tcn.py -> class `ResNet`, the
# base residual dilated-TCN network (matching this repo's `resnet-1.0c`/`resnet-1.1c`/
# `resnet-1.1n` reference checkpoints in `data/` and `log/`) used to map a per-frame
# spectral input (e.g. log-power-spectrum / MFCC-style features) to a per-bin
# a-priori-SNR-derived target (from which the MMSE noise PSD estimate is computed).
#
# ResNet stacks `n_blocks` bottleneck residual blocks over 1D convolutions along the
# time axis: each block is Pre-LN -> ReLU -> 1x1 Conv1D (bottleneck-down, no bias) ->
# Pre-LN -> ReLU -> dilated Conv1D (bottleneck width, cyclic dilation rate
# `2**(i % (log2(max_d_rate)+1))`, causal padding) -> Pre-LN -> ReLU -> 1x1 Conv1D
# (bottleneck-up, WITH bias) -> residual add. A frame-wise (LayerNormalization over
# the feature axis, per-timestep) feedforward stem projects the input to `d_model`
# channels before the block stack, and a final 1x1 Conv1D projects to `n_outp`
# channels with a configurable output activation (Sigmoid/ReLU/Linear). Causal
# ("CAUSAL") padding on the dilated convs is the paper's real-time-capable design
# choice (kept here). Architecture (bottleneck residual block, cyclic dilation
# schedule, frame-wise LayerNorm placement, causal dilated convs) is reproduced
# faithfully from the real Keras code; only the framework substrate changes.

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class FrameLayerNorm(nn.Module):
    """Keras `LayerNormalization(axis=2, epsilon=1e-6)` on a (B, T, C) tensor
    normalizes over the channel axis per (batch, timestep) -- equivalent to
    torch's nn.LayerNorm(C) applied on the same (B, T, C) layout."""

    def __init__(self, d_model, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: (B, C, T) torch-conv layout -> normalize over C
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def causal_conv1d(in_ch, out_ch, k, dilation, bias):
    """Keras Conv1D(padding='CAUSAL') left-pads so output[t] only depends on
    input[<=t]; replicate with explicit left padding + a plain valid Conv1d."""
    conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=dilation, bias=bias)
    pad = (k - 1) * dilation
    return conv, pad


class ResNetUnit(nn.Module):
    """tcn.py: ResNet.unit -- LN -> ReLU -> Conv1D(causal-padded when k>1)."""

    def __init__(self, in_ch, n_filt, k, d_rate, use_bias, padding="causal"):
        super().__init__()
        self.norm = FrameLayerNorm(in_ch)
        self.relu = nn.ReLU()
        self.k = k
        self.padding = padding
        if padding == "causal" and k > 1:
            self.conv, self.left_pad = causal_conv1d(in_ch, n_filt, k, d_rate, use_bias)
        else:
            self.left_pad = ((k - 1) * d_rate) // 2
            self.conv = nn.Conv1d(
                in_ch, n_filt, kernel_size=k, dilation=d_rate, padding=self.left_pad, bias=use_bias
            )
            self.left_pad = 0  # padding built into conv already ('same')

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        if self.left_pad > 0:
            x = nn.functional.pad(x, (self.left_pad, 0))
        x = self.conv(x)
        return x


class ResNetBlock(nn.Module):
    """tcn.py: ResNet.block -- bottleneck residual block."""

    def __init__(self, d_model, d_f, k, d_rate, padding="causal"):
        super().__init__()
        self.conv_1 = ResNetUnit(d_model, d_f, 1, 1, False, padding)
        self.conv_2 = ResNetUnit(d_f, d_f, k, d_rate, False, padding)
        self.conv_3 = ResNetUnit(d_f, d_model, 1, 1, True, padding)

    def forward(self, x):
        h = self.conv_1(x)
        h = self.conv_2(h)
        h = self.conv_3(h)
        return x + h


class ResNet(nn.Module):
    """tcn.py: class ResNet -- residual TCN with bottleneck blocks + cyclic
    dilation rate and frame-wise layer normalisation.

    Input/output layout here is (B, C, T) (torch-conv-native); the real Keras
    code uses (B, T, C) -- purely a layout convention, not an architectural change.
    """

    def __init__(
        self,
        n_feat,
        n_outp,
        n_blocks,
        d_model,
        d_f,
        k,
        max_d_rate,
        padding="causal",
        outp_act="Sigmoid",
    ):
        super().__init__()
        self.stem_conv = nn.Conv1d(n_feat, d_model, kernel_size=1, bias=False)
        self.stem_norm = FrameLayerNorm(d_model)
        self.stem_relu = nn.ReLU()

        n_cycle = int(math.log2(max_d_rate)) + 1
        self.blocks = nn.ModuleList(
            [ResNetBlock(d_model, d_f, k, 2 ** (i % n_cycle), padding) for i in range(n_blocks)]
        )

        self.outp_conv = nn.Conv1d(d_model, n_outp, kernel_size=1, bias=True)
        if outp_act == "Sigmoid":
            self.outp_act = nn.Sigmoid()
        elif outp_act == "ReLU":
            self.outp_act = nn.ReLU()
        elif outp_act == "Linear":
            self.outp_act = nn.Identity()
        else:
            raise ValueError("Invalid outp_act")

    def forward(self, x):
        """x: (B, n_feat, T) -- per-frame spectral features (e.g. log-power
        spectrum) along the time axis."""
        h = self.stem_conv(x)
        h = self.stem_norm(h)
        h = self.stem_relu(h)
        for block in self.blocks:
            h = block(h)
        out = self.outp_conv(h)
        out = self.outp_act(out)
        return out


# ---- staging entry points ----


def build_deepmmse_resnet():
    """DeepXi/DeepMMSE ResNet at tiny size for tracing (2 bottleneck blocks, small
    d_model/d_f, max_d_rate=2 -> single dilation cycle). Architecture is unmodified
    from the real repo."""
    torch.manual_seed(0)
    model = ResNet(
        n_feat=16,
        n_outp=16,
        n_blocks=2,
        d_model=8,
        d_f=4,
        k=3,
        max_d_rate=2,
        padding="causal",
        outp_act="Sigmoid",
    )
    model.eval()
    return model


def example_input_deepmmse_resnet():
    """Matches ResNet.forward(x): a batch of per-frame spectral feature sequences
    (B, n_feat, T)."""
    torch.manual_seed(0)
    return (torch.randn(2, 16, 20),)


MENAGERIE_ENTRIES = [
    (
        "DeepMMSE-ResNet",
        "build_deepmmse_resnet",
        "example_input_deepmmse_resnet",
        2020,
        MENAGERIE_ZOO,
    ),
]
