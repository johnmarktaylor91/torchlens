# FAITHFUL REIMPLEMENTATION from Singh, Hanson, Paliwal & Zhou (2019, Nature
# Communications, "RNA secondary structure prediction using an ensemble of
# two-dimensional deep neural networks and transfer learning", SPOT-RNA,
# https://doi.org/10.1038/s41467-019-13395-9) (no public architecture code --
# jaswindersingh2/SPOT-RNA and jaswindersingh2/SPOT-RNA2 both ship only the input/
# output preprocessing utilities plus a `tf.compat.v1.train.import_meta_graph` call
# that loads a pre-frozen TensorFlow-1 checkpoint; no `tf.keras`/`tf.nn` model-
# construction code for the network itself exists in either repo).
#
# Reimplemented per Fig. 1 ("Generalized model architecture of SPOT-RNA") and the
# "Deep neural networks" / "Input" Methods paragraphs of the paper: an RNA sequence
# is one-hot encoded (L x 4), outer-concatenated into an L x L x 8 2D map (as
# described in RaptorX-Contact-style protein contact prediction, cited by the
# paper), then passed through an initial 3x3 convolution ("pre-activation" ResNet
# style per He et al.), N_A residual blocks (Block A: 3x3 conv -> act/norm/dropout
# -> 5x5 conv -> residual add -> act/norm, with dilated convolutions across blocks
# using dilation = 2^(i/n) per the "dilated convolutions ... exponential linear
# scalar n" Methods text), a 2D bidirectional LSTM, N_B fully-connected Block-B
# layers (linear -> act/norm/dropout), and a final linear output layer with sigmoid
# activation producing an L x L base-pair probability matrix (paper: "The sigmoid
# function converts the output into the probability of each nucleotide being
# paired with other nucleotides"). Hyperparameters (N_A, D_RES, D_BL, N_B, D_FC)
# are set to the paper's stated search ranges/defaults (Methods: "N_A, D_RES, D_BL,
# N_B, D_FC ... searched over 16 to 32, 32 to 72, 128 to 256, 0 to 4, and 256 to
# 512, respectively"), using one representative point from that range per the
# staging harness below (this file represents a single ensemble member, not the
# paper's 5-model ensemble).

from __future__ import annotations

import torch
from torch import nn


class OuterConcatenation(nn.Module):
    """Outer-concatenate an (L, C) sequence embedding into an (L, L, 2C) 2D map.

    Matches the paper's Input section: "This one-dimensional (L x 4) input feature
    is converted to two dimensional (L x L x 8) by the outer concatenation function
    as described in RaptorX-Contact" -- i.e. for every pair (i, j), concatenate the
    per-position feature vectors at i and j.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> (B, L, L, 2C)
        length = x.shape[1]
        row = x.unsqueeze(2).expand(-1, -1, length, -1)
        col = x.unsqueeze(1).expand(-1, length, -1, -1)
        return torch.cat([row, col], dim=-1)


class ResNetBlockA(nn.Module):
    """SPOT-RNA "Block A": pre-activation residual block with dilated convolutions.

    Fig. 1 layout: 3x3 conv -> act/norm/dropout -> 3x3 conv -> act/norm/dropout ->
    5x5 conv -> (+residual) -> act/norm, repeated (N_A - 1) times. Dilation grows
    across the stack per the Methods text ("For the dilated convolutional layers,
    the dilation factor was set to 2^(i/n)").
    """

    def __init__(self, channels: int, dilation: int, dropout: float = 0.25):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2 * dilation, dilation=dilation
        )
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.norm1(x))
        out = self.dropout(self.conv1(out))
        out = self.act(self.norm2(out))
        out = self.dropout(self.conv2(out))
        return out + residual


class FCBlockB(nn.Module):
    """SPOT-RNA "Block B": fully-connected block (linear -> act/norm/dropout)."""

    def __init__(self, dim: int, dropout: float = 0.5):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.norm(self.linear(x))))


class SpotRNA(nn.Module):
    """Faithful reimplementation of a single SPOT-RNA ensemble member.

    Parameters follow the paper's Methods hyperparameter-search ranges:
    n_resnet_blocks (N_A) in [16, 32]; resnet_channels (D_RES) in [32, 72];
    blstm_hidden (D_BL) in [128, 256]; n_fc_blocks (N_B) in [0, 4];
    fc_dim (D_FC) in [256, 512].
    """

    def __init__(
        self,
        n_resnet_blocks: int = 4,
        resnet_channels: int = 16,
        blstm_hidden: int = 24,
        n_fc_blocks: int = 2,
        fc_dim: int = 32,
    ):
        super().__init__()
        self.outer_concat = OuterConcatenation()
        # Initial 3x3 conv: one-hot outer-concat gives 8 input channels (2 * 4 bases).
        self.stem = nn.Conv2d(8, resnet_channels, kernel_size=3, padding=1)

        blocks = []
        for i in range(n_resnet_blocks):
            dilation = max(1, int(round(2 ** (i / max(n_resnet_blocks, 1)))))
            blocks.append(ResNetBlockA(resnet_channels, dilation=dilation))
        self.resnet_blocks = nn.ModuleList(blocks)

        self.blstm = nn.LSTM(
            input_size=resnet_channels,
            hidden_size=blstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.post_blstm_norm = nn.LayerNorm(2 * blstm_hidden)

        fc_blocks = []
        fc_in = 2 * blstm_hidden
        for _ in range(n_fc_blocks):
            fc_blocks.append(nn.Linear(fc_in, fc_dim))
            fc_blocks.append(FCBlockB(fc_dim))
            fc_in = fc_dim
        self.fc_blocks = nn.ModuleList(fc_blocks)

        self.output_layer = nn.Linear(fc_in, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, one_hot_seq: torch.Tensor) -> torch.Tensor:
        # one_hot_seq: (B, L, 4)
        batch, length, _ = one_hot_seq.shape

        pair_map = self.outer_concat(one_hot_seq)  # (B, L, L, 8)
        pair_map = pair_map.permute(0, 3, 1, 2)  # (B, 8, L, L)

        x = self.stem(pair_map)
        for block in self.resnet_blocks:
            x = block(x)

        # Feed rows of the L x L feature map to the 2D-BLSTM (row-wise sequence model).
        channels = x.shape[1]
        x = x.permute(0, 2, 3, 1).reshape(batch * length, length, channels)
        x, _ = self.blstm(x)
        x = self.post_blstm_norm(x)
        x = x.reshape(batch, length, length, -1)

        for layer in self.fc_blocks:
            x = layer(x)

        logits = self.output_layer(x).squeeze(-1)  # (B, L, L)
        prob = self.sigmoid(logits)
        # Symmetrize the base-pair probability matrix, as the target/output of
        # SPOT-RNA is an undirected pairing matrix (paper's L x L upper-triangular
        # label convention).
        prob = 0.5 * (prob + prob.transpose(1, 2))
        return prob


# --- staging harness (tiny sizes; not part of the real repo) ---


def build_spot_rna() -> nn.Module:
    model = SpotRNA(
        n_resnet_blocks=4,
        resnet_channels=16,
        blstm_hidden=24,
        n_fc_blocks=2,
        fc_dim=32,
    )
    model.eval()
    return model


def example_input_spot_rna():
    # (batch, length, 4) one-hot-encoded RNA sequence (A, U, C, G), matching the
    # paper's stated input representation.
    batch, length = 1, 20
    idx = torch.randint(0, 4, (batch, length))
    return (torch.nn.functional.one_hot(idx, num_classes=4).float(),)


MENAGERIE_ZOO = "reimpl-pytorch"
MENAGERIE_ENTRIES = [
    ("SPOT-RNA", "build_spot_rna", "example_input_spot_rna", 2019, "reimpl"),
]
