# SOURCE: vendored from https://github.com/luoyunan/ECNet @ main (ecnet/model.py)
# ECNet: an evolutionary context-integrated deep learning framework for protein
# engineering (Luo et al., Nature Communications 2021). The model class below
# (`LSTMPredictor` and its submodules) is copied verbatim from the official repo's
# ecnet/model.py, which is BSD-3-Clause licensed. Only import paths were adjusted
# (no relative-package imports were present) and the module was renamed for staging.
"""Vendored ECNet model definition (LSTMPredictor + submodules)."""

import collections
import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class PositionalEmbedding(nn.Module):
    """
    Modified from Annotated Transformer
    http://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    def __init__(self, d_model, max_len=1024):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros((max_len, d_model), requires_grad=False).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class InputPositionEmbedding(nn.Module):
    def __init__(
        self, vocab_size=None, embed_dim=None, dropout=0.1, init_weight=None, seq_len=None
    ):
        super(InputPositionEmbedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.position_embed = PositionalEmbedding(embed_dim, max_len=seq_len)
        self.reproject = nn.Identity()
        if init_weight is not None:
            self.embed = nn.Embedding.from_pretrained(init_weight)
            self.reproject = nn.Linear(init_weight.size(1), embed_dim)

    def forward(self, inputs):
        x = self.embed(inputs)
        x = x + self.position_embed(inputs)
        x = self.reproject(x)
        x = self.dropout(x)
        return x


class AggregateLayer(nn.Module):
    def __init__(self, d_model=None, dropout=0.1):
        super(AggregateLayer, self).__init__()
        self.attn = nn.Sequential(
            collections.OrderedDict(
                [
                    ("layernorm", nn.LayerNorm(d_model)),
                    ("fc", nn.Linear(d_model, 1, bias=False)),
                    ("dropout", nn.Dropout(dropout)),
                    ("softmax", nn.Softmax(dim=1)),
                ]
            )
        )

    def forward(self, context):
        """
        Parameters
        ----------
        context: token embedding from encoder (Transformer/LSTM)
                (batch_size, seq_len, embed_dim)
        """

        weight = self.attn(context)
        # (batch_size, seq_len, embed_dim).T * (batch_size, seq_len, 1) *  ->
        # (batch_size, embed_dim, 1)
        output = torch.bmm(context.transpose(1, 2), weight)
        output = output.squeeze(2)
        return output


class GlobalPredictor(nn.Module):
    def __init__(self, d_model=None, d_h=None, d_out=None, dropout=0.5):
        super(GlobalPredictor, self).__init__()
        self.predict_layer = nn.Sequential(
            collections.OrderedDict(
                [
                    ("batchnorm", nn.BatchNorm1d(d_model)),
                    ("fc1", nn.Linear(d_model, d_h)),
                    ("tanh", nn.Tanh()),
                    ("dropout", nn.Dropout(dropout)),
                    ("fc2", nn.Linear(d_h, d_out)),
                ]
            )
        )

    def forward(self, x):
        x = self.predict_layer(x)
        return x


class SequenceLSTM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        d_input=None,
        d_embed=20,
        d_model=128,
        vocab_size=None,
        seq_len=None,
        dropout=0.1,
        lstm_dropout=0,
        nlayers=1,
        bidirectional=False,
        proj_loc_config=None,
    ):
        super(SequenceLSTM, self).__init__()

        self.embed = InputPositionEmbedding(
            vocab_size=vocab_size, seq_len=seq_len, embed_dim=d_embed
        )

        self.lstm = nn.LSTM(
            input_size=d_input,
            hidden_size=d_model // 2 if bidirectional else d_model,
            num_layers=nlayers,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )
        self.drop = nn.Dropout(dropout)
        self.proj_loc_layer = proj_loc_config["layer"](
            proj_loc_config["d_in"], proj_loc_config["d_out"]
        )

    def forward(self, x, loc_feat=None):
        x = self.embed(x)
        if loc_feat is not None:
            loc_feat = self.proj_loc_layer(loc_feat)
            x = torch.cat([x, loc_feat], dim=2)
        x = x.transpose(0, 1).contiguous()
        x, _ = self.lstm(x)
        x = x.transpose(0, 1).contiguous()
        x = self.drop(x)
        return x


class LSTMPredictor(nn.Module):
    def __init__(
        self,
        d_embed=20,
        d_model=128,
        d_h=128,
        d_out=1,
        vocab_size=None,
        seq_len=None,
        dropout=0.1,
        lstm_dropout=0,
        nlayers=1,
        bidirectional=False,
        use_loc_feat=True,
        use_glob_feat=True,
        proj_loc_config=None,
        proj_glob_config=None,
    ):
        super(LSTMPredictor, self).__init__()
        self.seq_lstm = SequenceLSTM(
            d_input=d_embed + (proj_loc_config["d_out"] if use_loc_feat else 0),
            d_embed=d_embed,
            d_model=d_model,
            vocab_size=vocab_size,
            seq_len=seq_len,
            dropout=dropout,
            lstm_dropout=lstm_dropout,
            nlayers=nlayers,
            bidirectional=bidirectional,
            proj_loc_config=proj_loc_config,
        )
        self.proj_glob_layer = proj_glob_config["layer"](
            proj_glob_config["d_in"], proj_glob_config["d_out"]
        )
        self.aggragator = AggregateLayer(
            d_model=d_model + (proj_glob_config["d_out"] if use_glob_feat else 0)
        )
        self.predictor = GlobalPredictor(
            d_model=d_model + (proj_glob_config["d_out"] if use_glob_feat else 0),
            d_h=d_h,
            d_out=d_out,
        )

    def forward(self, x, glob_feat=None, loc_feat=None):
        x = self.seq_lstm(x, loc_feat=loc_feat)
        if glob_feat is not None:
            glob_feat = self.proj_glob_layer(glob_feat)
            x = torch.cat([x, glob_feat], dim=2)
        x = self.aggragator(x)
        output = self.predictor(x)
        return output


# ---------------------------------------------------------------------------
# Staging build/example helpers (tiny config, matches the repo's own
# `if __name__ == "__main__":` smoke-test shapes, scaled down for fast tracing).
# ---------------------------------------------------------------------------

_SEQ_LEN = 32
_GLOB_DIM = 64
_LOC_DIM = _SEQ_LEN


def build_ecnet():
    return LSTMPredictor(
        d_embed=20,
        d_model=32,
        d_h=32,
        d_out=1,
        vocab_size=21,
        seq_len=_SEQ_LEN,
        nlayers=1,
        bidirectional=False,
        proj_glob_config={"layer": nn.Linear, "d_in": _GLOB_DIM, "d_out": 32},
        proj_loc_config={"layer": nn.Linear, "d_in": _LOC_DIM, "d_out": 32},
    )


class _ECNetInputWrapper(nn.Module):
    """Wraps LSTMPredictor so tracing can pass a single example-input tuple."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, glob_feat, loc_feat):
        return self.model(x, glob_feat=glob_feat, loc_feat=loc_feat)


def build_ecnet_wrapped():
    return _ECNetInputWrapper(build_ecnet())


def example_input_ecnet():
    batch = 2
    x = torch.randint(0, 21, (batch, _SEQ_LEN))
    glob_feat = torch.rand((batch, _SEQ_LEN, _GLOB_DIM))
    loc_feat = torch.rand((batch, _SEQ_LEN, _LOC_DIM))
    return (x, glob_feat, loc_feat)


MENAGERIE_ENTRIES = [
    ("ECNet", "build_ecnet_wrapped", "example_input_ecnet", 2021, "vendored-pytorch"),
]
