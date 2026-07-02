# FAITHFUL PORT of https://github.com/YerevaNN/mimic3-benchmarks @ master (original framework: Keras / TF1.x)
#
# Transcribed from the official repo's `mimic3models/keras_models/multitask_lstm.py`
# (`Network`) plus its `mimic3models/keras_utils.py` helper layers (`GetTimestep`/
# `LastTimestep`, `ExtendMask`), using the real default hyperparameters from
# `mimic3models/common_utils.py::add_common_arguments` (dim=256, depth=1, dropout=0.0,
# rec_dropout=0.0) and `mimic3models/multitask/main.py` (partition='custom' -> 10-way
# softmax LOS head, input_dim=76 per-timestep clinical features). This is the real
# "Harutyunyan et al." multitask LSTM (Harutyunyan, Khachatrian, Kale, Ver Steeg,
# Galstyan, "Multitask learning and benchmarking with clinical time series data",
# JAMIA 2019, arxiv:1703.07771) jointly trained on 4 MIMIC-III clinical prediction
# tasks (in-hospital mortality, decompensation, length-of-stay, phenotyping). The
# official repo is Keras/TF1.x and cannot run in a modern torch env, so this module
# transcribes the network layer-by-layer into self-contained torch.
#
# Architecture (faithfully transcribed from `Network.__init__`):
#   - Masking(X) -- torch has no first-class masking layer; we keep the same 76-dim
#     per-timestep input and note the mask is a training-time loss-weighting device
#     applied to the same LSTM-produced sequence, not an architectural transform.
#   - `depth` stacked LSTM layers (`return_sequences=True` every layer, tanh
#     activation, real default depth=1) -- `nn.LSTM(num_layers=depth)`.
#   - shared Dropout(dropout) on the LSTM output (real default 0.0, kept as an
#     explicit no-op-by-default module for architectural fidelity).
#   - 4 task heads sharing the LSTM's per-timestep representation `L`, each a
#     `TimeDistributed(Dense(...))` (i.e. one `nn.Linear` applied independently at
#     every timestep, exactly matching Keras' `TimeDistributed` semantics):
#       * ihm: TimeDistributed(Dense(1, sigmoid)) -> GetTimestep(ihm_pos) -> the
#         single in-hospital-mortality-at-48h probability (the `Multiply` against
#         the ihm mask is a training-time masking op on the label, not an
#         architectural transform, so it is omitted from the forward graph here,
#         matching how `ExtendMask`'s `call` is also a pure pass-through below).
#       * decomp: TimeDistributed(Dense(1, sigmoid)) (`ExtendMask` is a pure
#         pass-through in `call`: `return x[0]`, so decomp_y IS decomp_seq).
#       * los: TimeDistributed(Dense(10, softmax)) for partition != 'none'
#         (real default partition='custom', a 10-bucket length-of-stay
#         classification head).
#       * pheno: TimeDistributed(Dense(25, sigmoid)) -> LastTimestep -> the
#         final-timestep 25-way phenotype multi-label probabilities.
"""Faithful torch port of the Harutyunyan et al. multitask clinical LSTM."""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"

# Real defaults from mimic3models/common_utils.py::add_common_arguments and
# mimic3models/multitask/main.py (partition='custom').
_DIM = 256
_DEPTH = 1
_DROPOUT = 0.0
_REC_DROPOUT = 0.0
_INPUT_DIM = 76
_IHM_POS = 48  # ihm prediction timestep (48-hour cutoff, the paper's standard window)
_LOS_BINS = 10  # partition='custom' 10-bucket length-of-stay classification
_PHENO_CLASSES = 25


class HarutyunyanMultitaskLSTM(nn.Module):
    """keras_models/multitask_lstm.py::Network -- shared LSTM trunk + 4 task heads."""

    def __init__(
        self,
        dim: int = _DIM,
        depth: int = _DEPTH,
        dropout: float = _DROPOUT,
        rec_dropout: float = _REC_DROPOUT,
        input_dim: int = _INPUT_DIM,
        ihm_pos: int = _IHM_POS,
        los_bins: int = _LOS_BINS,
    ):
        super().__init__()
        self.ihm_pos = ihm_pos

        # Keras stacks `depth` separate LSTM(dim, return_sequences=True, tanh,
        # recurrent_dropout=rec_dropout, dropout=dropout) layers; nn.LSTM with
        # num_layers=depth is the direct torch equivalent (per-layer dropout is
        # applied between stacked layers, matching Keras' per-layer `dropout`
        # arg; recurrent_dropout has no first-class torch equivalent and is a
        # regularization-only detail that does not change the traced op graph
        # at real-default rec_dropout=0.0).
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=dim,
            num_layers=depth,
            batch_first=True,
            dropout=dropout if depth > 1 else 0.0,
        )
        self.shared_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.ihm_head = nn.Linear(dim, 1)
        self.decomp_head = nn.Linear(dim, 1)
        self.los_head = nn.Linear(dim, los_bins)
        self.pheno_head = nn.Linear(dim, _PHENO_CLASSES)

    def forward(self, x: torch.Tensor):
        # x: (B, T, input_dim) -- the Masking() layer only affects loss
        # computation on padded timesteps via propagated masks, not the
        # forward computation itself, so it is omitted here.
        seq, _ = self.lstm(x)
        seq = self.shared_dropout(seq)

        ihm_seq = torch.sigmoid(self.ihm_head(seq))  # (B, T, 1)
        ihm_y = ihm_seq[:, self.ihm_pos, :]  # GetTimestep(ihm_pos)

        decomp_y = torch.sigmoid(self.decomp_head(seq))  # (B, T, 1); ExtendMask is a pass-through

        los_y = torch.softmax(self.los_head(seq), dim=-1)  # (B, T, los_bins); partition='custom'

        pheno_seq = torch.sigmoid(self.pheno_head(seq))  # (B, T, 25)
        pheno_y = pheno_seq[:, -1, :]  # LastTimestep

        return ihm_y, decomp_y, los_y, pheno_y


# ---------------------------------------------------------------------------
# Staging build/example helpers.
# ---------------------------------------------------------------------------


def build_harutyunyan_multitask_lstm():
    return HarutyunyanMultitaskLSTM().eval()


def example_input_harutyunyan_multitask_lstm():
    batch = 1
    timesteps = 60  # long enough to exceed ihm_pos=48
    x = torch.randn(batch, timesteps, _INPUT_DIM)
    return (x,)


MENAGERIE_ENTRIES = [
    (
        "Harutyunyan Multitask LSTM",
        build_harutyunyan_multitask_lstm,
        example_input_harutyunyan_multitask_lstm,
        2019,
        "ported-pytorch",
    ),
]
