# SOURCE: vendored from https://github.com/HC-Guo/LogFormer @ main (model.py)
# LogFormer: pre-train + adapter-tune transformer for log anomaly detection
# (Guo, Yuan, Wu, Zhao, Zhu, Kang, Zhang, Xu, Bo, AAAI 2024).
#
# Vendored real repo code (the `Model` nn.Module from model.py, plus its
# `PositionalEncoding`, `Adapter`, and adapter-augmented
# `TransformerEncoderLayer` classes). This is genuinely a modified
# architecture, not a bare `nn.TransformerEncoder`: each encoder layer wraps
# the real `nn.MultiheadAttention` in an `Adapter` bottleneck
# (Linear -> GELU -> Linear, added residually) after self-attention and after
# the feed-forward block, matching the paper's parameter-efficient
# adapter-tuning stage. Only the unused `LearnedPositionEncoding` class,
# training-mode toggles (`train_adapter`/`train_classifier`, which just flip
# `requires_grad`, no architecture), and the `if __name__ == "__main__"` demo
# stub were left out of the traced entry point (kept in-file, unused). No
# layer, dimension, or dataflow inside `Model`/`TransformerEncoderLayer` was
# changed. Constructed in "adapter" mode (`mode='adapter'`), the real
# adapter-tuning architecture the paper contributes.

import math

import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        z = self.gelu(self.linear1(x))
        z = self.linear2(z)
        return x + z


class TransformerEncoderLayer(nn.Module):
    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        adapter_size=64,
        dim_feedforward=3072,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs
        )
        self.dropout1 = nn.Dropout(dropout)
        self.adapter1 = Adapter(d_model, d_model, hidden_dim=adapter_size)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.adapter2 = Adapter(d_model, d_model, hidden_dim=adapter_size)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **_unused_kwargs):
        # **_unused_kwargs absorbs newer-torch-only args (e.g. `is_causal`)
        # that `nn.TransformerEncoder.forward` started forwarding to layers
        # after this repo was written; not part of the architecture.
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src2 = self.dropout1(src2)
        src2 = self.adapter1(src2)
        src = self.norm1(src + src2)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.dropout2(src2)
        src2 = self.adapter2(src2)
        src = self.norm2(src + src2)

        return src

    def activate_adapter(self):
        tune_layers = [self.adapter1, self.adapter2, self.norm1, self.norm2]
        for layer in tune_layers:
            for param in layer.parameters():
                param.requires_grad = True


class Model(nn.Module):
    def __init__(
        self,
        mode,
        num_layers=4,
        adapter_size=64,
        dim=768,
        window_size=100,
        nhead=8,
        dim_feedforward=3072,
        dropout=0.1,
    ):
        super(Model, self).__init__()
        if mode == "adapter":
            encoder_layer = TransformerEncoderLayer(
                dim,
                nhead,
                adapter_size=adapter_size,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                dim, nhead, dim_feedforward, dropout, batch_first=True
            )

        self.trans_encder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers
        )
        self.pos_encoder1 = PositionalEncoding(d_model=768)
        self.fc1 = nn.Linear(dim * window_size, 2)

    def forward(self, x):
        B, _, _ = x.size()
        x = self.pos_encoder1(x)
        x = self.trans_encder(x)
        x = x.contiguous().view(B, -1)
        x = self.fc1(x)
        return x

    def train_adapter(self):
        for param in self.parameters():
            param.requires_grad = False
        for layer in self.trans_encder.layers:
            layer.activate_adapter()
        for param in self.fc1.parameters():
            param.requires_grad = True

    def train_classifier(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.fc1.parameters():
            param.requires_grad = True


def build_logformer():
    # dim is pinned at 768 because the real repo's `Model.__init__`
    # hardcodes `self.pos_encoder1 = PositionalEncoding(d_model=768)`
    # regardless of the `dim` constructor arg (a quirk of the actual code,
    # reproduced faithfully rather than silently fixed). num_layers,
    # adapter_size, window_size, and dim_feedforward are shrunk from the
    # paper's num_layers=4/window_size=100/dim_feedforward=3072 defaults to
    # keep the trace small; architecture (adapter-tuning transformer
    # encoder) is unchanged.
    return Model(
        "adapter",
        num_layers=2,
        adapter_size=16,
        dim=768,
        window_size=4,
        nhead=4,
        dim_feedforward=64,
    )


def example_input_logformer():
    # (batch, window_size, dim=768) pre-computed log-event embeddings,
    # matching DataGenerator's `mix_features_pad` layout in the real repo's
    # dataloader.
    return torch.randn(2, 4, 768)


MENAGERIE_ENTRIES = [
    (
        "LogFormer",
        build_logformer,
        example_input_logformer,
        2024,
        MENAGERIE_ZOO,
    ),
]
