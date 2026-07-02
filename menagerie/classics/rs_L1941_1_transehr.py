# SOURCE: vendored from SigmaTsing/TransEHR @ main
# https://raw.githubusercontent.com/SigmaTsing/TransEHR/main/ts_transformer/ts_transformer.py
#
# TransEHR (Zhang et al., "Self-Supervised Pre-Training for Transformer-Based Person
# Activity Recognition and Clinical Time Series", PMLR 2023 / MLHC). A self-supervised
# transformer for clinical EHR time series ("TSTransformerEncoder") pre-trained with a
# masked-language-modeling-style objective, then fine-tuned for downstream classification
# via "DownstreamClassifier". This module vendors the real classes verbatim (only the
# cross-file `from ... import` is inlined -- there are none, `ts_transformer.py` is
# self-contained on base torch already). `mimic.py` / `pipeline/thp_utils.py` in the real
# repo construct exactly this `DownstreamClassifier(TSTransformerEncoder(...), ...)`
# composition (see `pipeline/thp_utils.py` lines 48-56) for the clinical-time-series
# classification task, which is the architectural contribution this staging module
# exercises end to end.
#
# Minimal torch-version compat fix: `TransformerBatchNormEncoderLayer.forward` gained
# an accepted-but-unused `is_causal` kwarg, since torch>=2.x's `nn.TransformerEncoder`
# now always forwards that kwarg to each layer -- the original 2020s-era custom layer
# predates it. No architectural change (still the real BatchNorm-instead-of-LayerNorm
# encoder layer with the same self-attn/FFN/residual mechanism).

from typing import Optional

import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import (
    MultiheadAttention,
    Linear,
    Dropout,
    BatchNorm1d,
    TransformerEncoderLayer,
)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
    in the sequence. The positional encodings have the same dimension as
    the embeddings, so that the two can be summed. Here, we use sine and cosine
    functions of different frequencies.
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer(
            "pe", pe
        )  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(
            torch.empty(max_len, 1, d_model)
        )  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError(
        "pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding)
    )


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(
            d_model, eps=1e-5
        )  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> Tensor:
        # `is_causal` accepted (and unused) only for forward-compat with newer
        # torch.nn.TransformerEncoder, which always forwards it to each layer's
        # forward() as of torch>=2.x; the original repo (pre-dates that kwarg)
        # never passed or consumed it, so this is a signature-only compat shim,
        # not an architectural change.
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoder(nn.Module):
    def __init__(
        self,
        feat_dim,
        max_len,
        d_model,
        n_heads,
        num_layers,
        dim_feedforward,
        dropout=0.1,
        pos_encoding="fixed",
        activation="gelu",
        norm="BatchNorm",
        freeze=False,
    ):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout, max_len=max_len)

        if norm == "LayerNorm":
            encoder_layer = TransformerEncoderLayer(
                d_model, self.n_heads, dim_feedforward, dropout, activation=activation
            )
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(
                d_model, self.n_heads, dim_feedforward, dropout, activation=activation
            )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at
                this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim].
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp)
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with MultiHeadAttention convention
        output = self.transformer_encoder(
            inp, src_key_padding_mask=~padding_masks
        )  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the transformer output embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        return output


class DownstreamClassifier(nn.Module):
    def __init__(self, encoder, d_model, num_classes, aggr="max"):
        super(DownstreamClassifier, self).__init__()

        self.encoder = encoder
        self.linear0 = torch.nn.Linear(d_model, 64)
        self.linear = torch.nn.Linear(64, num_classes)
        self.aggregation = aggr

    def forward(self, X, padding_masks, statics=None):
        output = self.encoder(X, padding_masks)
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        if self.aggregation == "max":
            output, _ = torch.max(output, dim=1)
        elif self.aggregation == "mean":
            output = torch.mean(output, dim=1)
        else:
            pass
        if statics is not None:
            output = torch.cat((output, statics), axis=1)
        output = self.linear0(output)
        output = F.gelu(output)
        return self.linear(output)


MENAGERIE_ZOO = "vendored-pytorch"


def build_transehr():
    torch.manual_seed(0)
    feat_dim = 16  # number of clinical time-series channels
    max_len = 20  # sequence length
    d_model = 32
    n_heads = 4
    num_layers = 2
    dim_feedforward = 64
    num_classes = 2
    encoder = TSTransformerEncoder(
        feat_dim,
        max_len,
        d_model,
        n_heads,
        num_layers,
        dim_feedforward,
        dropout=0.1,
        pos_encoding="fixed",
        activation="gelu",
        norm="BatchNorm",
    )
    model = DownstreamClassifier(encoder, d_model, num_classes, aggr="max")
    model.eval()
    return model


def example_input_transehr():
    torch.manual_seed(0)
    batch, seq_len, feat_dim = 2, 20, 16
    X = torch.randn(batch, seq_len, feat_dim)
    padding_masks = torch.ones(batch, seq_len, dtype=torch.bool)
    return (X, padding_masks)


MENAGERIE_ENTRIES = [
    ("TransEHR", "build_transehr", "example_input_transehr", 2023, MENAGERIE_ZOO),
]
