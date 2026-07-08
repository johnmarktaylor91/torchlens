# FAITHFUL PORT of Altaheri/EEG-ATCNet @ main (models.py, TCNet_Fusion/EEGNet/TCN_block)
# (original framework: TensorFlow/Keras)
#
# TCNet-Fusion architecture from:
#   Musallam, Y.K., AlFassam, N.I., Muhammad, G., Amin, S.U., Alsulaiman, M., Abdul, W.,
#   Altaheri, H., Bencherif, M.A. and Algabri, M. (2021). "Electroencephalography-based
#   motor imagery classification using temporal convolutional network fusion."
#   Biomedical Signal Processing and Control, 69, p.102826. https://doi.org/10.1016/j.bspc.2021.102826
#
# The original paper released no public code. The canonical, author-attributed reproduction
# lives in Altaheri/EEG-ATCNet (King Saud University, Apache-2.0), which is the exact repo
# named in queue.tsv's TCNet-Fusion notes ("also benchmarked in Altaheri/EEG-ATCNet"). That
# repo's TCNet_Fusion() is Keras/TensorFlow (tensorflow.keras.layers.*), so per the model
# ladder it is transcribed here FAITHFULLY into torch: every layer/mechanism (EEGNet
# feature-extraction block -> TCN residual dilated-conv block -> triple-concatenation fusion
# head) preserved, only the tensor-layout idiom changed (Keras channels_last -> torch
# channels_first; both compute identical convolutions/pooling/dilation math).
#
# Reference EEGNet sub-block:
#   Lawhern, V. J., et al. (2018). "EEGNet: A Compact Convolutional Network for EEG-based
#   Brain-Computer Interfaces." arXiv:1611.08024. (github.com/vlawhern/arl-eegmodels)
# Reference TCN sub-block:
#   Bai, S., Kolter, J. Z., & Koltun, V. (2018). "An empirical evaluation of generic
#   convolutional and recurrent networks for sequence modeling." arXiv:1803.01271.
#   (this TCN_block variant is the one used verbatim in iis-eth-zurich/eeg-tcnet and
#   reproduced in Altaheri/EEG-ATCNet)

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalConv1d(nn.Module):
    """Conv1d with left-only ('causal') zero padding, matching Keras Conv1D(padding='causal')."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class _EEGNetBlock(nn.Module):
    """Faithful port of EEGNet(input_layer, F1, kernLength, D, Chans, dropout) from
    Altaheri/EEG-ATCNet models.py (itself a reproduction of Lawhern et al. 2018 EEGNet).

    Keras layout: input (N, Samples, Chans, 1) channels_last.
    Torch layout used here: input (N, 1, Samples, Chans) channels_first -- identical math,
    H=Samples, W=Chans, C=1, matching the Keras Permute((3,2,1)) applied before this block.
    """

    def __init__(
        self, chans: int, f1: int = 24, kern_length: int = 32, d: int = 2, dropout: float = 0.3
    ):
        super().__init__()
        f2 = f1 * d
        # block1: Conv2D(F1, (kernLength, 1), padding='same')
        self.conv1 = nn.Conv2d(1, f1, (kern_length, 1), padding=(kern_length // 2, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        # block2: DepthwiseConv2D((1, Chans)) -- kernel spans the full channel axis (valid pad)
        self.depthwise = nn.Conv2d(f1, f1 * d, (1, chans), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1 * d)
        self.pool1 = nn.AvgPool2d((8, 1))
        self.drop1 = nn.Dropout(dropout)
        # block3: SeparableConv2D(F2, (16, 1), padding='same') = depthwise + pointwise
        self.sep_depthwise = nn.Conv2d(
            f1 * d, f1 * d, (16, 1), padding=(8, 0), groups=f1 * d, bias=False
        )
        self.sep_pointwise = nn.Conv2d(f1 * d, f2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((8, 1))
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (N, 1, Samples, Chans)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)  # -> (N, F1*D, Samples, 1)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)  # -> (N, F2, Samples/8, 1)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x  # (N, F2, Samples/64, 1)


class _TCNBlock(nn.Module):
    """Faithful port of TCN_block(input_layer, input_dimension, depth, kernel_size, filters,
    dropout, activation) from Altaheri/EEG-ATCNet models.py (dilated causal residual stack,
    originally from iis-eth-zurich/eeg-tcnet, itself adapting Bai et al. 2018 TCN)."""

    def __init__(
        self,
        input_dimension: int,
        depth: int,
        kernel_size: int,
        filters: int,
        dropout: float,
        activation=F.elu,
    ):
        super().__init__()
        self.activation = activation
        self.input_dimension = input_dimension
        self.filters = filters

        self.conv_a1 = _CausalConv1d(input_dimension, filters, kernel_size, dilation=1)
        self.bn_a1 = nn.BatchNorm1d(filters)
        self.conv_a2 = _CausalConv1d(filters, filters, kernel_size, dilation=1)
        self.bn_a2 = nn.BatchNorm1d(filters)
        self.drop = nn.Dropout(dropout)
        self.resid_proj = None
        if input_dimension != filters:
            self.resid_proj = nn.Conv1d(input_dimension, filters, kernel_size=1)

        self.depth = depth
        self.rest_conv1 = nn.ModuleList()
        self.rest_bn1 = nn.ModuleList()
        self.rest_conv2 = nn.ModuleList()
        self.rest_bn2 = nn.ModuleList()
        for i in range(depth - 1):
            dilation = 2 ** (i + 1)
            self.rest_conv1.append(_CausalConv1d(filters, filters, kernel_size, dilation=dilation))
            self.rest_bn1.append(nn.BatchNorm1d(filters))
            self.rest_conv2.append(_CausalConv1d(filters, filters, kernel_size, dilation=dilation))
            self.rest_bn2.append(nn.BatchNorm1d(filters))

    def forward(self, x):
        # x: (N, C_in, L)
        block = self.conv_a1(x)
        block = self.bn_a1(block)
        block = self.activation(block)
        block = self.drop(block)
        block = self.conv_a2(block)
        block = self.bn_a2(block)
        block = self.activation(block)
        block = self.drop(block)
        residual = self.resid_proj(x) if self.resid_proj is not None else x
        out = self.activation(block + residual)

        for i in range(self.depth - 1):
            block = self.rest_conv1[i](out)
            block = self.rest_bn1[i](block)
            block = self.activation(block)
            block = self.drop(block)
            block = self.rest_conv2[i](block)
            block = self.rest_bn2[i](block)
            block = self.activation(block)
            block = self.drop(block)
            out = self.activation(block + out)
        return out  # (N, filters, L)


class TCNetFusion(nn.Module):
    """Faithful port of TCNet_Fusion(n_classes, Chans, Samples, layers, kernel_s, filt,
    dropout, activation, F1, D, kernLength, dropout_eeg) from Altaheri/EEG-ATCNet models.py.

    Musallam et al. 2021, "EEG-based motor imagery classification using temporal
    convolutional network fusion" (BSPC 69:102826).
    """

    def __init__(
        self,
        n_classes: int,
        chans: int = 22,
        samples: int = 1125,
        layers: int = 2,
        kernel_s: int = 4,
        filt: int = 12,
        dropout: float = 0.3,
        f1: int = 24,
        d: int = 2,
        kern_length: int = 32,
        dropout_eeg: float = 0.3,
    ):
        super().__init__()
        f2 = f1 * d
        self.eegnet = _EEGNetBlock(
            chans=chans, f1=f1, kern_length=kern_length, d=d, dropout=dropout_eeg
        )
        self.tcn = _TCNBlock(
            input_dimension=f2, depth=layers, kernel_size=kernel_s, filters=filt, dropout=dropout
        )
        seq_len = samples // 64  # two (8,1) poolings after 'same'-padded convs
        # Con1 = concat([block2, outs]) over channel axis (Keras concat axis=-1 on
        # (N, seq_len, F2) and (N, seq_len, filt) channels_last == channel-dim concat here)
        # out = Flatten(Con1); FC = Flatten(block2); Con2 = concat([out, FC])
        flat_dim = seq_len * (f2 + filt) + seq_len * f2
        self.dense = nn.Linear(flat_dim, n_classes)

    def forward(self, x):
        # x: (N, 1, Chans, Samples) -- matches Keras Input(shape=(1, Chans, Samples))
        # Keras: input2 = Permute((3,2,1))(input1) -> (Samples, Chans, 1) channels_last
        # Torch equivalent (channels_first): (N, 1, Samples, Chans)
        x = x.permute(0, 1, 3, 2)  # (N, 1, Samples, Chans)
        block2 = self.eegnet(x)  # (N, F2, seq_len, 1)
        block2 = block2.squeeze(
            -1
        )  # (N, F2, seq_len) -- matches Lambda(x[:, :, -1, :]) selecting the W=1 axis
        fc = torch.flatten(block2, start_dim=1)

        outs = self.tcn(block2)  # (N, filt, seq_len)

        con1 = torch.cat([block2, outs], dim=1)  # channel-axis concat
        out = torch.flatten(con1, start_dim=1)
        con2 = torch.cat([out, fc], dim=1)
        logits = self.dense(con2)
        return F.softmax(logits, dim=-1)


MENAGERIE_ZOO = "ported-pytorch"


def build_tcnet_fusion():
    return TCNetFusion(
        n_classes=4, chans=8, samples=256, layers=2, kernel_s=4, filt=12, f1=8, d=2, kern_length=16
    )


def example_input_tcnet_fusion():
    return torch.randn(1, 1, 8, 256)


MENAGERIE_ENTRIES = [
    (
        "TCNet-Fusion",
        build_tcnet_fusion,
        example_input_tcnet_fusion,
        2021,
        MENAGERIE_ZOO,
    ),
]
