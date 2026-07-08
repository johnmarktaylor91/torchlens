# FAITHFUL REIMPLEMENTATION from Kumar, A., Singh, J., Paliwal, K., Singh, J., Zhou, Y.
# (2020) "Single-sequence and profile-based prediction of RNA solvent accessibility
# using dilated convolutional neural network", Bioinformatics, btaa652,
# https://doi.org/10.1093/bioinformatics/btaa652 (no public code)
#
# RNAsnap2 (jaswindersingh2/RNAsnap2) predicts per-nucleotide RNA solvent accessibility
# (RSA) with a dilated-convolutional residual network over a per-position feature matrix
# (one-hot sequence + Infernal profile + LinearPartition base-pair probability +
# secondary-structure features). The repo's own inference script
# (utils/rna-snap2.py) is TensorFlow-1.14 code that restores a frozen
# `tf.compat.v1.train.import_meta_graph` checkpoint (`models/tensorflow_model_profile.meta`)
# -- there is no `nn.Module`/`tf.keras.Model` architecture class anywhere in the repo to
# vendor or port; the layer graph is baked entirely into the (unavailable) binary
# checkpoint. The repo's own README figure (docs/RNAsnap2_architecture.png, reproduced
# in the header comment on RNAsnap2Net below) is the most detailed architecture
# description available and is transcribed faithfully here as fresh, randomly
# initialized torch:
#
#   Input (L x 10) -> Conv1D(k=3) -> 64 channels
#   -> [x3 residual dilated block, dilation DF = 2^i for block index i in {0,1,2}]:
#        Conv1D(k=5, dilation=DF, dropout=0.4) -> BatchInstanceNorm1D -> ELU
#        Conv1D(k=7, dilation=DF, dropout=0.4) -> BatchInstanceNorm1D -> ELU
#        Conv1D(k=3, dilation=DF, dropout=0.4) -> BatchInstanceNorm1D -> ELU
#        (+ residual add from block input)
#   -> Dropout(0.4) -> BatchInstanceNorm1D -> ELU
#   -> Sigmoid output layer (per-position RSA in [0, 1])
#
# "Batch Instance Norm" (BIN) is the gated mix of BatchNorm1d and InstanceNorm1d from
# Nam & Kim, "Batch-Instance Normalization for Adaptively Style-Invariant Neural
# Networks" (NeurIPS 2018) -- the standard technique that name refers to; implemented
# here per that paper's Eq. (2)-(3) (per-channel learned gate rho blending BN and IN
# statistics, followed by an affine transform), since the RNAsnap2 paper itself only
# names the technique (no equations) and cites no other BIN variant.
#
# MENAGERIE_ZOO = "reimpl-pytorch"

from __future__ import annotations

import torch
import torch.nn as nn

MENAGERIE_ZOO = "reimpl-pytorch"


class BatchInstanceNorm1d(nn.Module):
    """Batch-Instance Normalization (Nam & Kim, NeurIPS 2018), 1D-conv variant.

    out = rho * BN(x) + (1 - rho) * IN(x), then per-channel affine (gamma, beta),
    with rho in [0, 1] a learned per-channel gate (clamped after each update, as in
    the reference implementation).
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=False)
        self.in_ = nn.InstanceNorm1d(num_features, eps=eps, affine=False)
        self.rho = nn.Parameter(torch.ones(num_features) * 0.5)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        rho = self.rho.clamp(0, 1).view(1, -1, 1)
        out = rho * self.bn(x) + (1 - rho) * self.in_(x)
        return out * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)


class DilatedResidualBlock(nn.Module):
    """One '[x3]' dashed-box unit from the RNAsnap2 architecture figure: three dilated
    Conv1D layers (k=5, k=7, k=3) at a shared dilation factor, each followed by
    dropout + BatchInstanceNorm1d + ELU, with a residual add from the block input."""

    def __init__(self, channels, dilation, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=5, dilation=dilation, padding=(5 - 1) * dilation // 2
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=7, dilation=dilation, padding=(7 - 1) * dilation // 2
        )
        self.conv3 = nn.Conv1d(
            channels, channels, kernel_size=3, dilation=dilation, padding=(3 - 1) * dilation // 2
        )

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

        self.bin1 = BatchInstanceNorm1d(channels)
        self.bin2 = BatchInstanceNorm1d(channels)
        self.bin3 = BatchInstanceNorm1d(channels)

        self.act = nn.ELU()

    def forward(self, x):
        residual = x

        h = self.act(self.bin1(self.drop1(self.conv1(x))))
        h = self.act(self.bin2(self.drop2(self.conv2(h))))
        h = self.act(self.bin3(self.drop3(self.conv3(h))))

        return h + residual


class RNAsnap2Net(nn.Module):
    """Dilated-CNN RNA solvent-accessibility predictor (RNAsnap2), faithfully
    reimplemented from the paper's architecture figure. `in_features=10` matches the
    figure's `(1's pre-padding + one-hot encoding + Infernal features + base-pair
    probability of SS)` per-position feature stack; `n_blocks=3` and the per-block
    dilation schedule `DF = 2^i` also come directly from the figure."""

    def __init__(self, in_features=10, hidden_channels=64, n_blocks=3, dropout=0.4):
        super().__init__()

        self.stem = nn.Conv1d(in_features, hidden_channels, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList(
            [
                DilatedResidualBlock(hidden_channels, dilation=2**i, dropout=dropout)
                for i in range(n_blocks)
            ]
        )

        self.out_drop = nn.Dropout(dropout)
        self.out_bin = BatchInstanceNorm1d(hidden_channels)
        self.out_act = nn.ELU()

        self.output_layer = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, length, in_features) -> conv wants (batch, in_features, length)
        x = x.transpose(1, 2)

        h = self.stem(x)

        for block in self.blocks:
            h = block(h)

        h = self.out_act(self.out_bin(self.out_drop(h)))

        rsa = self.output_act(self.output_layer(h))
        return rsa.squeeze(1)


# ---------------------------------------------------------------------------
# staging glue (not part of the original architecture)
# ---------------------------------------------------------------------------
def build_rnasnap2():
    return RNAsnap2Net(in_features=10, hidden_channels=64, n_blocks=3, dropout=0.4)


def example_input_rnasnap2():
    batch, length, in_features = 2, 24, 10
    return torch.randn(batch, length, in_features)


MENAGERIE_ENTRIES = [
    ("RNAsnap2", "build_rnasnap2", "example_input_rnasnap2", 2020, "reimpl-pytorch"),
]
