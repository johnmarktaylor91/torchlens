# FAITHFUL PORT of calico/basenji @ 0.2 (original framework: TensorFlow 1.x)
#
# Source: https://github.com/calico/basenji/blob/0.2/basenji/seqnn.py (`SeqNN.build`)
# config: https://github.com/calico/basenji/blob/0.2/tutorials/models/params_med.txt
#
# Basenji (Kelley et al. 2018, Genome Research) predicts cell-type-specific
# chromatin accessibility / gene expression tracks directly from one-hot DNA
# sequence. The original v0.2 `SeqNN.build()` is TensorFlow-1.x graph-mode code
# (tf.placeholder, tf.contrib.layers.xavier_initializer, tf.layers.conv1d/
# batch_normalization) that cannot run in this repo's base env and predates the
# later Keras-params-driven `basenji/blocks.py` refactor (which targets the
# distinct, larger "Basenji2"/Enformer-family architecture) -- so the *original*
# published Basenji model is transcribed faithfully here rather than vendored.
#
# The real `SeqNN.build()` loop, replicated 1:1 per CNN layer `li`:
#   conv1d(filters=cnn_filters[li], kernel=cnn_filter_sizes[li],
#          stride=cnn_strides[li], dilation=cnn_dilation[li], padding='same',
#          bias=False) -> BatchNorm -> ReLU
#     -> optional MaxPool1d(cnn_pool[li]) if cnn_pool[li] > 1
#     -> optional Dropout(cnn_dropout[li]) if cnn_dropout[li] > 0
#     -> if cnn_dense[li]: concat this layer's output onto the running
#        representation (DenseNet-style channel growth), else replace it.
# followed by a final 1x1 conv projecting to `num_targets` channels and a
# `link` nonlinearity (paper's tutorial config uses `softplus`).
#
# Hyperparameters below are taken verbatim from the paper's own published
# tutorial config `tutorials/models/params_med.txt` (the "medium" Basenji model
# used in the original manuscript): 6 strided/pooled conv layers, then 7
# dilated (2,4,8,...,128) DenseNet-style residual conv layers, then a wide
# 512-filter conv, then a final 1x1 projection with softplus link, batch_buffer
# trimming, and target_pool average pooling of the targets (target_pool=128
# in the paper; not reproduced here since TorchLens traces the model, not the
# label pipeline). Sequence length reduced from the paper's 131072bp to a small
# multiple of the total pooling factor (4*4*4*2=128) for tracing speed.

import torch
import torch.nn as nn
import torch.nn.functional as F


# (filters, filter_size, stride, dilation, pool, dropout, dense) per CNN layer,
# taken directly from tutorials/models/params_med.txt.
_LAYER_CFG = [
    (196, 22, 1, 1, 1, 0.05, False),
    (196, 1, 1, 1, 2, 0.05, False),
    (235, 6, 1, 1, 4, 0.05, False),
    (282, 6, 1, 1, 4, 0.05, False),
    (338, 6, 1, 1, 4, 0.05, False),
    (384, 3, 1, 1, 1, 0.05, False),
    (64, 3, 1, 2, 1, 0.05, True),
    (64, 3, 1, 4, 1, 0.05, True),
    (64, 3, 1, 8, 1, 0.05, True),
    (64, 3, 1, 16, 1, 0.05, True),
    (64, 3, 1, 32, 1, 0.05, True),
    (64, 3, 1, 64, 1, 0.05, True),
    (64, 3, 1, 128, 1, 0.05, True),
    (512, 3, 1, 1, 1, 0.1, False),
]


class BasenjiConvBlock(nn.Module):
    """One `cnn%d` block from the original `SeqNN.build()` loop."""

    def __init__(self, in_channels, filters, filter_size, stride, dilation, pool, dropout, dense):
        super().__init__()
        pad = ((filter_size - 1) * dilation) // 2
        self.conv = nn.Conv1d(
            in_channels,
            filters,
            kernel_size=filter_size,
            stride=stride,
            dilation=dilation,
            padding=pad,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(filters, momentum=0.1, eps=1e-3)
        self.pool = pool
        self.dropout_p = dropout
        self.dense = dense
        if pool > 1:
            self.maxpool = nn.MaxPool1d(kernel_size=pool, stride=pool, ceil_mode=True)
        if dropout > 0:
            self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (batch, channels, length)
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        if self.pool > 1:
            out = self.maxpool(out)
        if self.dropout_p > 0:
            out = self.drop(out)

        if self.dense:
            # DenseNet-style concat: grow channel dim by concatenating onto
            # the running representation (matches `tf.concat([seqs_repr,
            # seqs_repr_next], axis=2)` in the channels-last original).
            if x.shape[-1] != out.shape[-1]:
                # a dense (non-pooling) layer's input/output length always
                # matches in the original config (pool==1 whenever dense=1),
                # but guard defensively for other configs.
                x = F.adaptive_avg_pool1d(x, out.shape[-1])
            return torch.cat([x, out], dim=1)
        return out


class Basenji(nn.Module):
    """Port of `SeqNN.build()` from basenji/seqnn.py @ tag 0.2, instantiated
    with the paper's `params_med.txt` layer config."""

    def __init__(self, seq_depth=4, num_targets=39, link="softplus", batch_buffer_bp=0):
        super().__init__()
        self.link = link
        self.batch_buffer_bp = batch_buffer_bp

        blocks = []
        in_channels = seq_depth
        for filters, fsize, stride, dilation, pool, dropout, dense in _LAYER_CFG:
            block = BasenjiConvBlock(
                in_channels, filters, fsize, stride, dilation, pool, dropout, dense
            )
            blocks.append(block)
            in_channels = in_channels + filters if dense else filters
        self.blocks = nn.ModuleList(blocks)

        # final 1x1 conv projecting to targets, matches `final/conv1d`.
        self.final_conv = nn.Conv1d(in_channels, num_targets, kernel_size=1, bias=True)

    def forward(self, seq):
        # seq: (batch, length, 4) one-hot DNA, matches the original
        # `tf.placeholder(shape=(batch, batch_length, seq_depth))`.
        x = seq.transpose(1, 2)  # -> (batch, channels, length)
        for block in self.blocks:
            x = block(x)

        if self.batch_buffer_bp > 0:
            x = x[:, :, self.batch_buffer_bp : -self.batch_buffer_bp]

        out = self.final_conv(x)  # (batch, num_targets, length)

        if self.link == "softplus":
            out = F.softplus(out.clamp(min=-50, max=50))
        elif self.link == "relu":
            out = F.relu(out)
        elif self.link == "exp":
            out = torch.exp(out.clamp(min=-50, max=50))
        # 'identity'/'linear' -> no-op

        return out.transpose(1, 2)  # (batch, length, num_targets), channels-last like the original


MENAGERIE_ZOO = "ported-pytorch"

_SEQ_LEN = 256  # small multiple of the total pooling factor (2*4*4*4=128) for fast tracing


def build_basenji():
    return Basenji(seq_depth=4, num_targets=39, link="softplus")


def example_input_basenji():
    return torch.randn(1, _SEQ_LEN, 4)


MENAGERIE_ENTRIES = [
    (
        "Basenji",
        build_basenji,
        example_input_basenji,
        2018,
        MENAGERIE_ZOO,
    ),
]
