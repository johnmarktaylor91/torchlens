# SOURCE: vendored from Eryk96/NetSurfP-3.0 @ main
# https://raw.githubusercontent.com/Eryk96/NetSurfP-3.0/main/nsp3/nsp3/models/CNNbLSMT/model.py
# https://raw.githubusercontent.com/Eryk96/NetSurfP-3.0/main/nsp3/nsp3/base/base_model.py
#
# Klausen, Jespersen, Nielsen, Jensen, Jurtz, Soenderby, Sommer, Winther, Nielsen,
# Petersen, Marcatili 2019 (Proteins) "NetSurfP-2.0: Improved prediction of protein
# structural features by integrated deep learning" -- a per-residue protein structure
# predictor: a stack of parallel multi-kernel-size 1D-CNN blocks over the input
# per-residue feature vector (PSSM/HMM profile features), concatenated with the raw
# input as a residual skip and batch-normalized, then fed through a 2-layer
# bidirectional LSTM, with independent linear task heads for 8-class and 3-class
# secondary structure, disorder, relative solvent accessibility (RSA), and
# backbone phi/psi dihedral angles. This is the queue's requested "v2.0 architecture
# (CNN+BiLSTM)"; NetSurfP-3.0's `CNNbLSTM` class is the base (no-language-model)
# variant of that same CNN+BiLSTM architecture shipped in the actively maintained
# successor repo (the v2.0 TensorFlow repo is not maintained/runnable; v3.0's PyTorch
# `CNNbLSTM` is architecturally the v2.0 CNN+BiLSTM baseline, distinct from the
# v3.0-specific ESM1b-embedding variants `CNNbLSTM_ESM1b_*` in the same file, which are
# NOT used here).
#
# `CNNbLSTM` is copied verbatim from the real `nsp3/nsp3/models/CNNbLSMT/model.py`
# (only the base no-embedding class; the `CNNbLSTM_ESM1b_*` variants that require an
# ESM1b language-model checkpoint are omitted -- not this architecture). `ModelBase` is
# reproduced verbatim from the real `nsp3/nsp3/base/base_model.py` (a trivial
# `nn.Module` subclass with a `__str__` override; no architectural role). The
# `setup_logger`/`log.info` init-time logging calls (which pull in a YAML-config
# logging chain unrelated to the network) are dropped -- mechanical logging-plumbing
# trim only, no architectural change.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelBase(nn.Module):
    """Base class for all models"""

    def __init__(self):
        super().__init__()

    def forward(self, *input):
        """Forward pass logic"""
        raise NotImplementedError


class CNNbLSTM(ModelBase):
    def __init__(
        self,
        init_n_channels: int,
        out_channels: int,
        cnn_layers: int,
        kernel_size: tuple,
        padding: tuple,
        n_hidden: int,
        dropout: float,
        lstm_layers: int,
    ):
        """Baseline model for CNNbLSTM
        Args:
            init_n_channels: size of the incoming feature vector
            out_channels: amount of hidden neurons in the bidirectional lstm
            cnn_layers: amount of cnn layers
            kernel_size: kernel sizes of the cnn layers
            padding: padding of the cnn layers
            n_hidden: amount of hidden neurons
            dropout: amount of dropout
            lstm_layers: amount of bidirectional lstm layers
        """

        super(CNNbLSTM, self).__init__()

        # CNN blocks
        self.conv = nn.ModuleList()
        for i in range(cnn_layers):
            self.conv.append(
                nn.Sequential(
                    *[
                        nn.Dropout(p=dropout),
                        nn.Conv1d(
                            in_channels=init_n_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size[i],
                            padding=padding[i],
                        ),
                        nn.ReLU(),
                    ]
                )
            )

        self.batch_norm = nn.BatchNorm1d(init_n_channels + (out_channels * 2))

        # LSTM block
        self.lstm = nn.LSTM(
            input_size=init_n_channels + (out_channels * 2),
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=dropout,
        )
        self.lstm_dropout_layer = nn.Dropout(p=dropout)

        # Task block
        self.ss8 = nn.Sequential(
            *[
                nn.Linear(in_features=n_hidden * 2, out_features=8),
            ]
        )
        self.ss3 = nn.Sequential(
            *[
                nn.Linear(in_features=n_hidden * 2, out_features=3),
            ]
        )
        self.disorder = nn.Sequential(
            *[
                nn.Linear(in_features=n_hidden * 2, out_features=2),
            ]
        )
        self.rsa = nn.Sequential(
            *[nn.Linear(in_features=n_hidden * 2, out_features=1), nn.Sigmoid()]
        )
        self.phi = nn.Sequential(*[nn.Linear(in_features=n_hidden * 2, out_features=2), nn.Tanh()])
        self.psi = nn.Sequential(*[nn.Linear(in_features=n_hidden * 2, out_features=2), nn.Tanh()])

    def forward(self, x, mask) -> list:
        """Forwarding logic"""

        max_length = x.size(1)
        x = x.permute(0, 2, 1)

        # concatenate channels from residuals and input + batch norm
        r = x
        for layer in self.conv:
            r = torch.cat([r, layer(x)], dim=1)

        x = self.batch_norm(r)

        # calculate double layer bidirectional lstm
        x = x.permute(0, 2, 1)
        x = pack_padded_sequence(x, mask, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, total_length=max_length, batch_first=True)
        x = self.lstm_dropout_layer(x)

        # hidden neurons to classes
        ss8 = self.ss8(x)
        ss3 = self.ss3(x)
        dis = self.disorder(x)
        rsa = self.rsa(x)
        phi = self.phi(x)
        psi = self.psi(x)

        return [ss8, ss3, dis, rsa, phi, psi]


def build_netsurfp2():
    # Exact real training config, from the repo's shipped NetSurfP-2.0/HHblits
    # experiment file `experiments/netsurfp_2/CNNbLSTM_HHblits.yml`: 50-dim
    # per-residue input feature vector (HHblits profile features), 2 parallel CNN
    # kernel widths {129, 257} with matching padding {64, 128}, 32 conv filters per
    # branch, 1024-unit 2-layer bidirectional LSTM, dropout 0.5.
    return CNNbLSTM(
        init_n_channels=50,
        out_channels=32,
        cnn_layers=2,
        kernel_size=(129, 257),
        padding=(64, 128),
        n_hidden=1024,
        dropout=0.5,
        lstm_layers=2,
    )


def example_input_netsurfp2():
    # (batch, sequence length, per-residue feature dim); mask = per-sample sequence
    # lengths required by pack_padded_sequence. Sequence length exceeds the largest
    # kernel (257) so every CNN branch produces a valid receptive field.
    x = torch.randn(2, 260, 50)
    mask = [260, 200]
    return (x, mask)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("NetSurfP-2.0", "build_netsurfp2", "example_input_netsurfp2", 2019, "vendored"),
]
