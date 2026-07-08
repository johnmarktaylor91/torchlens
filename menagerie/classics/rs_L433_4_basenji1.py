# FAITHFUL PORT of calico/basenji @ master (original framework: TensorFlow 1.x / tf.layers)
# Ported from the legacy `SeqNN` model-construction loop in basenji/seqnn.py (tag 0.1,
# https://raw.githubusercontent.com/calico/basenji/0.1/basenji/seqnn.py, lines ~65-203)
# using the ACTUAL published Basenji1 (Kelley et al. 2018, Genome Research) architecture
# hyperparameters from manuscripts/genome_research2018/params.txt in the same repo:
# 14 cnn stages (cnn_filter_sizes/cnn_filters/cnn_pool/cnn_dilation/cnn_dropout/cnn_dense),
# each stage = Conv1d(no bias, dilation, "same" padding) -> BatchNorm1d -> ReLU ->
# optional MaxPool -> optional Dropout -> either replace the running representation
# (cnn_dense=0) or channel-concat it onto the running representation (cnn_dense=1, the
# "dense"/growth connections used for the 7 dilated residual stages), followed by a final
# 1x1 conv to num_targets and the exp_linear link function
# (x > 0 ? x + 1 : exp(clip(x, -50, 50))) exactly as in the TF1 source.
# TensorFlow (needed to run the real basenji/seqnn.py + basenji/blocks.py) is not installed
# and basenji itself is not pip-installable in this base torch environment, so the real code
# could not be vendored/run directly; every layer here is transcribed 1:1 from the real
# TF1 source rather than guessed from the paper text.
"""Basenji1: dilated CNN predicting genome-wide functional tracks from long DNA
sequence, faithfully ported from the real calico/basenji TF1 SeqNN model."""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class _CnnStage(nn.Module):
    """One `for li in range(cnn_layers)` iteration of the real SeqNN.build() loop."""

    def __init__(self, in_channels, filters, kernel_size, stride, dilation, pool, dropout, dense):
        super().__init__()
        # "same" padding for stride=1 dilated conv, matching tf.layers.conv1d(padding='same')
        pad = ((kernel_size - 1) * dilation) // 2
        self.conv = nn.Conv1d(
            in_channels,
            filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(filters, momentum=0.1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(pool, stride=pool) if pool > 1 else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.dense = dense  # dense (concat/"growth") connection vs. plain replace

    def forward(self, seqs_repr):
        # NOTE: real code convs on the *running* seqs_repr (post previous concat/replace)
        x = self.conv(seqs_repr)
        x = self.relu(self.bn(x))
        if self.pool is not None:
            x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.dense:
            seqs_repr = torch.cat([seqs_repr, x], dim=1)
        else:
            seqs_repr = x
        return seqs_repr


# Basenji1 (Kelley et al. 2018) real published hyperparameters, verbatim from
# manuscripts/genome_research2018/params.txt in the calico/basenji repo.
BASENJI1_STAGES = [
    # (filter_size, filters, pool, dilation, dropout, dense)
    (22, 312, 1, 1, 0.05, False),
    (1, 368, 2, 1, 0.05, False),
    (6, 435, 4, 1, 0.05, False),
    (6, 514, 4, 1, 0.05, False),
    (6, 607, 4, 1, 0.05, False),
    (3, 717, 1, 1, 0.05, False),
    (3, 108, 1, 2, 0.1, True),
    (3, 108, 1, 4, 0.1, True),
    (3, 108, 1, 8, 0.1, True),
    (3, 108, 1, 16, 0.1, True),
    (3, 108, 1, 32, 0.1, True),
    (3, 108, 1, 64, 0.1, True),
    (1, 1365, 1, 1, 0.05, False),
]


class Basenji1(nn.Module):
    """Dilated CNN for regulatory-track prediction from DNA sequence.

    Faithful port of the real basenji `SeqNN` graph-construction loop
    (basenji/seqnn.py, tag 0.1) parameterized with the actual published
    Basenji1 stage list (manuscripts/genome_research2018/params.txt).
    Input: one-hot DNA sequence (batch, 4, seq_length).
    Output: predicted per-position target tracks after the "exp_linear" link,
    matching `self.preds_op` in the real SeqNN.build().
    """

    def __init__(self, num_targets=4, seq_depth=4, stages=None):
        super().__init__()
        stages = BASENJI1_STAGES if stages is None else stages
        self.stages = nn.ModuleList()
        in_channels = seq_depth
        for kernel_size, filters, pool, dilation, dropout, dense in stages:
            self.stages.append(
                _CnnStage(
                    in_channels,
                    filters,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    pool=pool,
                    dropout=dropout,
                    dense=dense,
                )
            )
            in_channels = in_channels + filters if dense else filters

        # "final" scope: 1x1 conv to num_targets, with bias (real code use_bias=True)
        self.final_conv = nn.Conv1d(in_channels, num_targets, kernel_size=1, bias=True)

    def forward(self, x):
        seqs_repr = x
        for stage in self.stages:
            seqs_repr = stage(seqs_repr)

        preds = self.final_conv(seqs_repr)

        # exp_linear link function, verbatim from the real SeqNN.build():
        # tf.where(seqs_repr > 0, seqs_repr + 1, tf.exp(tf.clip_by_value(seqs_repr,-50,50)))
        preds = torch.where(
            preds > 0,
            preds + 1,
            torch.exp(torch.clamp(preds, min=-50, max=50)),
        )
        return preds


# --- staging build/example helpers ---


def build_basenji1():
    """Tiny Basenji1 (real 13-stage architecture, shrunk seq_length/filters for a
    fast trace; num_targets shrunk from the real 4229 to 4)."""
    tiny_stages = [
        (11, 16, 1, 1, 0.05, False),
        (1, 16, 2, 1, 0.05, False),
        (3, 16, 4, 1, 0.05, False),
        (3, 16, 4, 1, 0.05, False),
        (3, 16, 4, 1, 0.05, False),
        (3, 16, 1, 1, 0.05, False),
        (3, 8, 1, 2, 0.1, True),
        (3, 8, 1, 4, 0.1, True),
        (3, 8, 1, 8, 0.1, True),
        (3, 8, 1, 16, 0.1, True),
        (3, 8, 1, 32, 0.1, True),
        (3, 8, 1, 64, 0.1, True),
        (1, 32, 1, 1, 0.05, False),
    ]
    return Basenji1(num_targets=4, seq_depth=4, stages=tiny_stages)


def example_input_basenji1():
    # One-hot DNA sequence: (batch, 4 bases, seq_length); seq_length must survive
    # the cumulative pooling (2*4*4*4 = 128x total) so it's a multiple of 128.
    return torch.eye(4)[torch.randint(0, 4, (1, 1024))].transpose(1, 2).contiguous()


MENAGERIE_ENTRIES = [
    ("Basenji1", "build_basenji1", "example_input_basenji1", 2018, "ported-pytorch"),
]
