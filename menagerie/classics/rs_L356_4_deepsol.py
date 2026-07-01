# FAITHFUL PORT of sameerkhurana10/DSOL_rv0.2 @ master (original framework: Keras/TF1.x)
#   scripts/dsol/Models_dsol1.py, Models_dsol2.py, Models_dsol3.py (class DeepSol)
# The real repo is Keras (`keras.layers.Conv1D/Embedding/GRU/LSTM`, functional API,
# `keras.models.Model`) using an old Keras/TF1.x API (`keras.layers.convolutional`,
# `keras.layers.recurrent`) incompatible with the installed torch/TF stack. This ports
# all three DeepSol variants (from Khurana et al. 2018, Bioinformatics) faithfully, using
# the ACTUAL default hyperparameters from the repo's own `parameters.json`:
#   deepsol1: Embedding -> parallel multi-kernel Conv1D bank (14 kernel sizes 2..15,
#     feature maps 64x7+128x7) each globally max-pooled -> concat -> FC(64) -> sigmoid head.
#   deepsol2: same conv-bank front end (7 kernel sizes 3..15) on the protein sequence,
#     concatenated with a raw biological-feature vector (57-dim, from SCRATCH) -> FC(64).
#   deepsol3: deepsol1's 14-kernel conv bank on the sequence branch, PLUS the biological
#     features first passed through their own small FC sub-network (bio_dnn_config =
#     "256,0.2": one Dense(256)+Dropout(0.2)+ReLU layer) before concatenation -> FC(64).
# Ported 1:1 from the real Keras functional-API graphs above; only the framework
# changes (Keras -> torch) and Conv1D's channel-last->channel-first convention.
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def _parse_cnn_config(cnn_config: str):
    """Parses the repo's `cnn_config` string, e.g.
    "2:3:4:...:15,64:64:...:128,global" -> (kernel_sizes, feature_maps, pool_mode)."""
    kernel_str, feature_str, pool_mode = cnn_config.split(",")
    kernel_sizes = [int(k) for k in kernel_str.split(":")]
    feature_maps = [int(f) for f in feature_str.split(":")]
    return kernel_sizes, feature_maps, pool_mode


class ConvBank(nn.Module):
    """Ports the single CNN "layer" (k=0 in the real code's `for k in
    range(len(cnn_config))` loop, which for all 3 DeepSol variants is a single entry)
    from Models_dsol{1,2,3}.py: a bank of parallel Conv1D branches (one per kernel size
    in `kernel_sizes`, matched with a per-branch `feature_maps` width), each
    padding="same" + ReLU, each followed by a pool_size="global" -> global max pool ->
    flatten (this is the last/only CNN layer with no RNN afterwards in the default
    configs, so per the real code `pool_size.lower()=='global' and rnn_config is None`
    triggers global max pooling + flatten for every branch), concatenated on the
    feature axis.
    """

    def __init__(self, in_channels: int, kernel_sizes: List[int], feature_maps: List[int]):
        super().__init__()
        assert len(kernel_sizes) == len(feature_maps)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels, fm, kernel_size=k, padding="same")
                for k, fm in zip(kernel_sizes, feature_maps)
            ]
        )

    def forward(self, x):
        # x: (batch, in_channels, seq_len) -- torch Conv1d channel-first convention
        branch_outs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch, fm, seq_len)
            pooled, _ = conv_out.max(dim=2)  # global max pool -> (batch, fm)
            branch_outs.append(pooled)
        return torch.cat(branch_outs, dim=1)


class BioDNN(nn.Module):
    """Ports the small biological-feature sub-network used in deepsol3
    (`bio_dnn_config = "256,0.2"`): Dense(256) -> Dropout(0.2) -> ReLU."""

    def __init__(self, in_dim: int, hidden_dims: List[int], dropouts: List[float]):
        super().__init__()
        layers = []
        prev = in_dim
        for h, p in zip(hidden_dims, dropouts):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Dropout(p))
            prev = h
        self.layers = nn.ModuleList(layers)
        self.out_dim = prev

    def forward(self, x):
        i = 0
        while i < len(self.layers):
            x = self.layers[i](x)  # Linear
            x = self.layers[i + 1](x)  # Dropout
            x = F.relu(x)
            i += 2
        return x


class DeepSol1(nn.Module):
    """Faithful port of DSOL_rv0.2 Models_dsol1.py::DeepSol (sequence-only variant)."""

    def __init__(
        self,
        maxlen: int = 200,
        vocab_size: int = 23,
        embedding_dim: int = 64,
        em_drop: float = 0.2,
        num_classes: int = 2,
        cnn_config: str = (
            "2:3:4:5:6:7:8:9:10:11:12:13:14:15,"
            "64:64:64:64:64:64:64:128:128:128:128:128:128:128,global"
        ),
        fc_config: str = "64,0.2",
    ):
        super().__init__()
        kernel_sizes, feature_maps, _pool_mode = _parse_cnn_config(cnn_config)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.em_dropout = nn.Dropout(em_drop)
        self.conv_bank = ConvBank(embedding_dim, kernel_sizes, feature_maps)
        conv_out_dim = sum(feature_maps)

        fc_dims_str, fc_drops_str = fc_config.split(",")
        # fc_config supports "-"-joined multi-layer specs; here it's a single layer.
        fc_dims = [int(d) for d in fc_dims_str.split("-")]
        fc_drops = [float(d) for d in fc_drops_str.split("-")]
        fc_layers = []
        prev = conv_out_dim
        for dim, drop in zip(fc_dims, fc_drops):
            fc_layers.append(nn.Linear(prev, dim))
            fc_layers.append(nn.Dropout(drop))
            prev = dim
        self.fc_layers = nn.ModuleList(fc_layers)
        self.output_layer = nn.Linear(prev, num_classes)

    def forward(self, input_protein_seq):
        # input_protein_seq: (batch, maxlen) long token ids
        embedded = self.embedding(input_protein_seq)  # (batch, maxlen, embed_dim)
        embedded = self.em_dropout(embedded.transpose(1, 2))  # spatial dropout on channel axis
        x = self.conv_bank(embedded)  # (batch, sum(feature_maps))

        i = 0
        while i < len(self.fc_layers):
            x = self.fc_layers[i](x)
            x = self.fc_layers[i + 1](x)
            x = F.relu(x)
            i += 2

        main_output = torch.sigmoid(self.output_layer(x))
        return main_output


class DeepSol2(nn.Module):
    """Faithful port of DSOL_rv0.2 Models_dsol2.py::DeepSol (sequence + raw
    biological features, concatenated directly -- no bio sub-network)."""

    def __init__(
        self,
        maxlen: int = 200,
        vocab_size: int = 23,
        embedding_dim: int = 64,
        em_drop: float = 0.2,
        num_classes: int = 2,
        num_bio_feats: int = 57,
        cnn_config: str = "3:5:7:9:11:13:15,64:64:64:128:128:128:128,global",
        fc_config: str = "64,0.2",
    ):
        super().__init__()
        kernel_sizes, feature_maps, _pool_mode = _parse_cnn_config(cnn_config)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.em_dropout = nn.Dropout(em_drop)
        self.conv_bank = ConvBank(embedding_dim, kernel_sizes, feature_maps)
        conv_out_dim = sum(feature_maps)

        fc_dims_str, fc_drops_str = fc_config.split(",")
        fc_dims = [int(d) for d in fc_dims_str.split("-")]
        fc_drops = [float(d) for d in fc_drops_str.split("-")]
        fc_layers = []
        prev = conv_out_dim + num_bio_feats  # concat(x, bio_feats) before FC layers
        for dim, drop in zip(fc_dims, fc_drops):
            fc_layers.append(nn.Linear(prev, dim))
            fc_layers.append(nn.Dropout(drop))
            prev = dim
        self.fc_layers = nn.ModuleList(fc_layers)
        self.output_layer = nn.Linear(prev, num_classes)

    def forward(self, input_protein_seq, input_bio_feats):
        embedded = self.embedding(input_protein_seq)
        embedded = self.em_dropout(embedded.transpose(1, 2))
        x = self.conv_bank(embedded)
        x = torch.cat([x, input_bio_feats], dim=1)

        i = 0
        while i < len(self.fc_layers):
            x = self.fc_layers[i](x)
            x = self.fc_layers[i + 1](x)
            x = F.relu(x)
            i += 2

        main_output = torch.sigmoid(self.output_layer(x))
        return main_output


class DeepSol3(nn.Module):
    """Faithful port of DSOL_rv0.2 Models_dsol3.py::DeepSol (sequence conv-bank +
    biological features first passed through their own small FC sub-network, per
    `bio_dnn_config`, before concatenation)."""

    def __init__(
        self,
        maxlen: int = 200,
        vocab_size: int = 23,
        embedding_dim: int = 64,
        em_drop: float = 0.2,
        num_classes: int = 2,
        num_bio_feats: int = 57,
        cnn_config: str = (
            "2:3:4:5:6:7:8:9:10:11:12:13:14:15,"
            "64:64:64:64:64:64:64:128:128:128:128:128:128:128,global"
        ),
        fc_config: str = "64,0.2",
        bio_dnn_config: str = "256,0.2",
    ):
        super().__init__()
        kernel_sizes, feature_maps, _pool_mode = _parse_cnn_config(cnn_config)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.em_dropout = nn.Dropout(em_drop)
        self.conv_bank = ConvBank(embedding_dim, kernel_sizes, feature_maps)
        conv_out_dim = sum(feature_maps)

        bio_dims_str, bio_drops_str = bio_dnn_config.split(",")
        bio_dims = [int(d) for d in bio_dims_str.split("-")]
        bio_drops = [float(d) for d in bio_drops_str.split("-")]
        self.bio_dnn = BioDNN(num_bio_feats, bio_dims, bio_drops)

        fc_dims_str, fc_drops_str = fc_config.split(",")
        fc_dims = [int(d) for d in fc_dims_str.split("-")]
        fc_drops = [float(d) for d in fc_drops_str.split("-")]
        fc_layers = []
        prev = conv_out_dim + self.bio_dnn.out_dim
        for dim, drop in zip(fc_dims, fc_drops):
            fc_layers.append(nn.Linear(prev, dim))
            fc_layers.append(nn.Dropout(drop))
            prev = dim
        self.fc_layers = nn.ModuleList(fc_layers)
        self.output_layer = nn.Linear(prev, num_classes)

    def forward(self, input_protein_seq, input_bio_feats):
        embedded = self.embedding(input_protein_seq)
        embedded = self.em_dropout(embedded.transpose(1, 2))
        x = self.conv_bank(embedded)

        y = self.bio_dnn(input_bio_feats)
        x = torch.cat([x, y], dim=1)

        i = 0
        while i < len(self.fc_layers):
            x = self.fc_layers[i](x)
            x = self.fc_layers[i + 1](x)
            x = F.relu(x)
            i += 2

        main_output = torch.sigmoid(self.output_layer(x))
        return main_output


# ---------------------------------------------------------------------------
# Menagerie staging glue
# ---------------------------------------------------------------------------
_MAXLEN = 40  # shrunk from real default (200/1200) for a fast tiny-init trace
_VOCAB = 23  # real default (20 amino acids + X/U/pad-like tokens)


def build_deepsol1():
    torch.manual_seed(0)
    model = DeepSol1(
        maxlen=_MAXLEN,
        vocab_size=_VOCAB,
        embedding_dim=16,
        num_classes=2,
        cnn_config="2:3:4,8:8:8,global",  # shrunk kernel bank, same structure/format
        fc_config="16,0.2",
    )
    model.eval()
    return model


def example_input_deepsol1():
    torch.manual_seed(0)
    batch_size = 2
    return torch.randint(0, _VOCAB, (batch_size, _MAXLEN))


def build_deepsol2():
    torch.manual_seed(0)
    model = DeepSol2(
        maxlen=_MAXLEN,
        vocab_size=_VOCAB,
        embedding_dim=16,
        num_classes=2,
        num_bio_feats=10,
        cnn_config="3:5:7,8:8:8,global",
        fc_config="16,0.2",
    )
    model.eval()
    return model


def example_input_deepsol2():
    torch.manual_seed(0)
    batch_size = 2
    seq = torch.randint(0, _VOCAB, (batch_size, _MAXLEN))
    bio = torch.randn(batch_size, 10)
    return (seq, bio)


def build_deepsol3():
    torch.manual_seed(0)
    model = DeepSol3(
        maxlen=_MAXLEN,
        vocab_size=_VOCAB,
        embedding_dim=16,
        num_classes=2,
        num_bio_feats=10,
        cnn_config="2:3:4,8:8:8,global",
        fc_config="16,0.2",
        bio_dnn_config="12,0.2",
    )
    model.eval()
    return model


def example_input_deepsol3():
    torch.manual_seed(0)
    batch_size = 2
    seq = torch.randint(0, _VOCAB, (batch_size, _MAXLEN))
    bio = torch.randn(batch_size, 10)
    return (seq, bio)


MENAGERIE_ENTRIES = [
    ("DeepSol1", build_deepsol1, example_input_deepsol1, 2018, "REIMPLEMENT"),
    ("DeepSol2", build_deepsol2, example_input_deepsol2, 2018, "REIMPLEMENT"),
    ("DeepSol3", build_deepsol3, example_input_deepsol3, 2018, "REIMPLEMENT"),
]
