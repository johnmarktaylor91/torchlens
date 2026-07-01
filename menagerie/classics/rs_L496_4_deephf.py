# FAITHFUL PORT of izhangcd/DeepHF (original framework: Keras 2.1.6 + TensorFlow 1.8)
# Original repo (izhangcd/DeepHF) has been deleted from GitHub (confirmed via
# ericmalekos/crisprware:crisprware/scorers/deephf.py, which notes "Original
# GitHub repo (`izhangcd/DeepHF`) has been deleted; weights are ..."). The
# real training code is preserved verbatim in a downstream package mirror:
# https://github.com/crisprVerse/crisprScore/blob/devel/inst/python/deephf/deephf/training_util.py
# https://github.com/crisprVerse/crisprScore/blob/devel/inst/python/deephf/deephf/prediction_util.py
#
# DeepHF: optimized CRISPR guide RNA on-target efficiency prediction for
# high-fidelity SpCas9 variants (Wang et al., Nature Communications 2019,
# "DeepHF"). The real ``lstm_model`` function in training_util.py builds a
# Keras functional-API graph: a character Embedding of the 22-length guide
# sequence (21mer + START token, per ``sequence_input = Input(shape=(22,))``
# and ``make_data`` in prediction_util.py) -> SpatialDropout1D -> a
# Bidirectional LSTM -> Flatten -> concatenated with an 11-dim biological
# feature vector (``lst_features = [0, 1, 2, 3, -7, -6, -5, -4, -3, -2, -1]``
# selected out of ``get_embedding_data`` in prediction_util.py) -> a 3-layer
# ELU-activated Dense stack with Dropout -> a single linear-activation output
# (predicted edit efficiency). Keras 2.1.6/TensorFlow 1.8 is not installable
# in this environment (API incompatible with modern TF; the repo weights are
# also unrecoverable), so this module transcribes the exact layer sequence
# and default hyperparameters from ``lstm_model`` into self-contained torch:
# nn.Embedding + SpatialDropout(1D-style channel dropout) + a bidirectional
# nn.LSTM + Flatten + feature concatenation + the same 3-hidden-layer ELU
# Dense stack + linear output head.

import torch
import torch.nn as nn


class SpatialDropout1D(nn.Module):
    """Matches keras.layers.SpatialDropout1D: drops entire embedding
    channels (not individual timesteps) -- same mechanism as
    nn.Dropout2d applied over the channel dimension of a (B, C, T) view."""

    def __init__(self, p: float):
        super().__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T, 1) -> Dropout2d over channel dim -> back
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.dropout(x)
        return x.squeeze(-1).permute(0, 2, 1)


class DeepHFModel(nn.Module):
    """Faithful port of training_util.py::lstm_model's Keras graph.

    Default hyperparameters mirror the ``lstm_model`` function signature:
    batch_size=90, epochs=50, em_dim=44, em_drop=0.2, rnn_units=60,
    rnn_drop=0.6, rnn_rec_drop=0.1, fc_num_hidden_layers=3, fc_num_units=320,
    fc_drop=0.4, fc_activation='elu' (the '0' entry in fc_activation_dict,
    the default `fc_activation` argument value).
    """

    def __init__(
        self,
        vocab_size: int = 7,  # Embedding(7, em_dim, input_length=22): PAD/START + A/T/C/G (+2 spare)
        seq_len: int = 22,
        biofeat_dim: int = 11,
        em_dim: int = 44,
        em_drop: float = 0.2,
        rnn_units: int = 60,
        rnn_drop: float = 0.6,
        rnn_rec_drop: float = 0.1,
        fc_num_hidden_layers: int = 3,
        fc_num_units: int = 320,
        fc_drop: float = 0.4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, em_dim)
        self.spatial_dropout = SpatialDropout1D(em_drop)
        # keras LSTM(rnn_units, dropout=rnn_drop, recurrent_dropout=rnn_rec_drop,
        # return_sequences=True) wrapped in Bidirectional(...)
        self.lstm = nn.LSTM(em_dim, rnn_units, batch_first=True, bidirectional=True)
        self.lstm_out_dropout = nn.Dropout(rnn_drop)
        self.rnn_rec_drop = (
            rnn_rec_drop  # kept for provenance; recurrent-dropout has no direct nn.LSTM analog
        )

        in_dim = rnn_units * 2 * seq_len + biofeat_dim
        fc_layers = []
        for _ in range(fc_num_hidden_layers):
            fc_layers.append(nn.Linear(in_dim, fc_num_units))
            fc_layers.append(nn.ELU())
            fc_layers.append(nn.Dropout(fc_drop))
            in_dim = fc_num_units
        self.fc = nn.Sequential(*fc_layers)
        self.mix_output = nn.Linear(in_dim, 1)  # Dense(1, activation='linear', name='mix_output')

    def forward(self, seq_input: torch.Tensor, bio_input: torch.Tensor) -> torch.Tensor:
        x = self.embedding(seq_input)
        x = self.spatial_dropout(x)
        x, _ = self.lstm(x)
        x = self.lstm_out_dropout(x)
        x = x.reshape(x.size(0), -1)  # Flatten()
        x = torch.cat([x, bio_input], dim=-1)  # keras.layers.concatenate([x, biological_input])
        x = self.fc(x)
        return self.mix_output(x)


MENAGERIE_ZOO = "ported-pytorch"

_SEQ_LEN = 22
_BIOFEAT_DIM = 11
_VOCAB_SIZE = 7


def build_deephf():
    return DeepHFModel(vocab_size=_VOCAB_SIZE, seq_len=_SEQ_LEN, biofeat_dim=_BIOFEAT_DIM)


def example_input_deephf():
    seq_input = torch.randint(0, _VOCAB_SIZE, (2, _SEQ_LEN))
    bio_input = torch.randn(2, _BIOFEAT_DIM)
    return (seq_input, bio_input)


MENAGERIE_ENTRIES = [
    (
        "DeepHF",
        "build_deephf",
        "example_input_deephf",
        2019,
        MENAGERIE_ZOO,
    ),
]
