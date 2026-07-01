# SOURCE: vendored from liu-bioinfo-lab/EPCOT @ a8f49cf072a05542c3d8a760dd2c094686f1f6da
#
# Vendored files (concatenated, imports rewritten to be self-contained, architecture
# unmodified):
#   pretraining/layers.py       (CNN backbone used by the pretraining model)
#   pretraining/transformer.py  (DETR-style Transformer encoder/decoder, copied from
#                                 Query2label/DETR per the source file's own header
#                                 comment)
#   pretraining/model.py        (GroupWiseLinear, Tranmodel, build_backbone,
#                                 build_transformer, build_model)
#
# EPCOT's pretraining stage (`Tranmodel`) is a CNN backbone (1D conv tower over a
# 5-channel one-hot-DNA + DNase-accessibility track) feeding a DETR-style
# encoder-decoder transformer with learned per-epigenomic-feature query embeddings
# (Query2label-style multi-label classification head), producing per-feature logits.
#
# `Tranmodel.forward` in the real source hardcodes `.cuda()` on `label_input`
# (`label_inputs=self.label_input.repeat(src.size(0),1).cuda()`), so this recipe
# builds and traces on CUDA (unmodified real code) rather than patching that line.
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn, Tensor

MENAGERIE_ZOO = "vendored-pytorch"

# ================================ pretraining/layers.py ==============================


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        conv_kernel_size1 = 10
        conv_kernel_size2 = 8
        pool_kernel_size1 = 5
        pool_kernel_size2 = 4
        self.conv_net = nn.Sequential(
            nn.Conv1d(5, 256, kernel_size=conv_kernel_size1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv1d(256, 256, kernel_size=conv_kernel_size1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size1, stride=pool_kernel_size1),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.1),
            nn.Conv1d(256, 360, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv1d(360, 360, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size2, stride=pool_kernel_size2),
            nn.BatchNorm1d(360),
            nn.Dropout(p=0.1),
            nn.Conv1d(360, 512, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv1d(512, 512, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
        )
        self.num_channels = 512

    def forward(self, x):
        out = self.conv_net(x)
        return out


# ============================= pretraining/transformer.py ============================
# "most of the codes below are copied from Query2label and DETR" -- source header.


def _get_activation_fn(activation):
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


def _get_clones(module, N):
    import copy

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        if num_decoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation
            )
            encoder_norm = nn.LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed=None, mask=None):
        bs, c, w = src.shape
        src = src.permute(2, 0, 1)
        query_embed = query_embed.transpose(0, 1)
        if mask is not None:
            mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(
            query_embed, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=None
        )
        return hs.transpose(0, 1)


# ================================ pretraining/model.py ================================
# "most of the codes below are copied from Query2label" -- source header.


class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class Tranmodel(nn.Module):
    def __init__(self, backbone, transfomer, num_class):
        super().__init__()
        self.backbone = backbone
        self.transformer = transfomer
        self.num_class = num_class
        hidden_dim = transfomer.d_model
        self.label_input = torch.Tensor(np.arange(num_class)).view(1, -1).long()
        self.input_proj = nn.Conv1d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_class, hidden_dim)
        self.fc = GroupWiseLinear(num_class, hidden_dim, bias=True)

    def forward(self, input):
        src = self.backbone(input)
        label_inputs = self.label_input.repeat(src.size(0), 1).cuda()
        label_embed = self.query_embed(label_inputs)
        src = self.input_proj(src)
        hs = self.transformer(src, label_embed)
        out = self.fc(hs)
        return out


class _Args:
    """Tiny stand-in for the real argparse.Namespace passed to build_model."""

    def __init__(
        self,
        num_class,
        hidden_dim,
        nheads,
        dim_feedforward,
        enc_layers,
        dec_layers,
        dropout,
        load_backbone=False,
    ):
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.dropout = dropout
        self.load_backbone = load_backbone


def build_backbone(args):
    model = CNN()
    # Real code loads a pretrained backbone checkpoint here when
    # `args.load_backbone` is True; that path needs external weights, so this
    # recipe always builds with `load_backbone=False` (random init).
    return model


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
    )


def build_model(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = Tranmodel(
        backbone=backbone,
        transfomer=transformer,
        num_class=args.num_class,
    )
    return model


# =================================== recipe glue ======================================


def build_epcot_pretraining():
    """EPCOT pretraining model (CNN backbone + DETR-style transformer head).

    Hyperparameters mirror `pretraining/pre_train.py`'s `parser_args` defaults
    (num_class=245, seq_length=1600, nheads=4, hidden_dim=512, dim_feedforward=1024,
    enc_layers=1, dec_layers=2), except `num_class` is shrunk for a fast smoke trace
    and `load_backbone` is forced False (no external checkpoint).
    """
    args = _Args(
        num_class=8,
        hidden_dim=512,
        nheads=4,
        dim_feedforward=1024,
        enc_layers=1,
        dec_layers=2,
        dropout=0.2,
        load_backbone=False,
    )
    model = build_model(args)
    assert torch.cuda.is_available(), (
        "EPCOT's real Tranmodel.forward hardcodes `.cuda()` on the label input "
        "tensor; this recipe traces the unmodified source, so it requires CUDA."
    )
    return model.cuda()


def example_input_epcot():
    torch.manual_seed(0)
    # (batch, 5, seq_length): one-hot ACGT (4 channels) + DNase accessibility
    # (1 channel), matching `pretraining/dataset.py`'s Task1Dataset input convention.
    # seq_length=1600 reproduces `pretraining/pre_train.py`'s --seq_length default.
    return torch.randn(1, 5, 1600).cuda()


MENAGERIE_ENTRIES = [
    (
        "EPCOT-pretraining",
        "build_epcot_pretraining",
        "example_input_epcot",
        2023,
        "SOURCE_AVAILABLE",
    ),
]
