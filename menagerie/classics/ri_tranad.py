"""TranAD-benchmark MTS anomaly-detection models (the TranAD* dual-decoder transformer family).

Tuli, Casale, Jennings; "TranAD: Deep Transformer Networks for Anomaly Detection in
Multivariate Time Series Data", VLDB 2022.
Paper: https://arxiv.org/abs/2201.07284
Source: https://github.com/imperial-qore/TranAD/blob/main/src/models.py (+ src/dlutils.py)

This file reimplements the TranAD core and its four ablation variants plus the simple
Attention baseline, all from the imperial-qore/TranAD ``src/models.py`` benchmark suite.

Distinctive primitives reproduced faithfully (compact, random-init):
  * TranAD: a Transformer ENCODER over the window concatenated with a self-conditioning
    focus-score channel ``c`` (so d_model = 2*feats), then TWO parallel Transformer
    DECODERS sharing one encoder memory. Two inference passes: pass-1 uses c=0, pass-2
    sets c = (x1 - W)^2 (adversarial self-conditioning / "focus score"). Final FCN -> feats.
  * The repo's bespoke TransformerEncoderLayer/DecoderLayer use LeakyReLU FFN and a tiny
    dim_feedforward=16 -- reproduced here rather than torch's stock layers.
  * PositionalEncoding adds sin AND cos into the SAME buffer (the repo's quirk) -- kept.
  * Ablations: TranAD_Basic (single decoder, no self-conditioning, d_model=feats),
    TranAD_Transformer (MLP-only "transformer", dual decoders, self-conditioning),
    TranAD_Adversarial (single decoder + self-conditioning, no second decoder),
    TranAD_SelfConditioning (dual decoders, NO second adversarial pass), Attention (the
    learned per-feature attention-matrix baseline).

Each model normally takes (src, tgt) windows; we wrap each in a single-tensor adapter that
synthesizes tgt internally so the unrolled graph is forward-able from one example tensor.
All compact: feats=7, window=10. trace+draw verified.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------
# dlutils primitives (ported verbatim from imperial-qore/TranAD src/dlutils.py)
# --------------------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Sinusoidal PE that sums sin and cos into the SAME buffer (the repo's quirk)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        x = x + self.pe[pos : pos + x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Repo's bespoke encoder layer: MHA + LeakyReLU FFN (dim_feedforward=16)."""

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None) -> torch.Tensor:
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    """Repo's bespoke decoder layer: self-attn + cross-attn + LeakyReLU FFN."""

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int = 16, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ) -> torch.Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


# --------------------------------------------------------------------------------------
# Attention baseline
# --------------------------------------------------------------------------------------


class Attention(nn.Module):
    """Learned per-feature attention-matrix baseline (window flattened -> feats x feats)."""

    def __init__(self, feats: int) -> None:
        super().__init__()
        self.name = "Attention"
        self.n_feats = feats
        self.n_window = 5
        self.n = self.n_feats * self.n_window
        self.atts = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.n, feats * feats), nn.ReLU(True)) for _ in range(1)]
        )

    def forward(self, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ats = g
        for at in self.atts:
            ats = at(g.reshape(-1)).reshape(self.n_feats, self.n_feats)
            g = torch.matmul(g, ats)
        return g, ats


# --------------------------------------------------------------------------------------
# TranAD core + ablations
# --------------------------------------------------------------------------------------


class TranAD_Basic(nn.Module):
    """Single encoder + single decoder, d_model = feats, no self-conditioning."""

    def __init__(self, feats: int) -> None:
        super().__init__()
        self.name = "TranAD_Basic"
        self.n_feats = feats
        self.n_window = 10
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        self.transformer_encoder = TransformerEncoderLayer(
            d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder = TransformerDecoderLayer(
            d_model=feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.fcn = nn.Sigmoid()

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        x = self.transformer_decoder(tgt, memory)
        x = self.fcn(x)
        return x


class TranAD_Transformer(nn.Module):
    """MLP-only 'transformer': dual decoders + self-conditioning, no real attention."""

    def __init__(self, feats: int) -> None:
        super().__init__()
        self.name = "TranAD_Transformer"
        self.n_feats = feats
        self.n_hidden = 8
        self.n_window = 10
        self.n = 2 * self.n_feats * self.n_window
        self.transformer_encoder = nn.Sequential(
            nn.Linear(self.n, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n),
            nn.ReLU(True),
        )
        self.transformer_decoder1 = nn.Sequential(
            nn.Linear(self.n, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, 2 * feats),
            nn.ReLU(True),
        )
        self.transformer_decoder2 = nn.Sequential(
            nn.Linear(self.n, self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, 2 * feats),
            nn.ReLU(True),
        )
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src: torch.Tensor, c: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src = torch.cat((src, c), dim=2)
        src = src.permute(1, 0, 2).flatten(start_dim=1)
        return self.transformer_encoder(src)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.zeros_like(src)
        x1 = self.transformer_decoder1(self.encode(src, c, tgt))
        x1 = x1.reshape(-1, 1, 2 * self.n_feats).permute(1, 0, 2)
        x1 = self.fcn(x1)
        c = (x1 - src) ** 2
        x2 = self.transformer_decoder2(self.encode(src, c, tgt))
        x2 = x2.reshape(-1, 1, 2 * self.n_feats).permute(1, 0, 2)
        x2 = self.fcn(x2)
        return x1, x2


class TranAD_Adversarial(nn.Module):
    """Single decoder + adversarial self-conditioning (two passes, no second decoder)."""

    def __init__(self, feats: int) -> None:
        super().__init__()
        self.name = "TranAD_Adversarial"
        self.n_feats = feats
        self.n_window = 10
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        self.transformer_encoder = TransformerEncoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode_decode(self, src: torch.Tensor, c: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        x = self.transformer_decoder(tgt, memory)
        x = self.fcn(x)
        return x

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        c = torch.zeros_like(src)
        x = self.encode_decode(src, c, tgt)
        c = (x - src) ** 2
        x = self.encode_decode(src, c, tgt)
        return x


class TranAD_SelfConditioning(nn.Module):
    """Dual parallel decoders + self-conditioning, but NO second adversarial pass."""

    def __init__(self, feats: int) -> None:
        super().__init__()
        self.name = "TranAD_SelfConditioning"
        self.n_feats = feats
        self.n_window = 10
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        self.transformer_encoder = TransformerEncoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder1 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder2 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src: torch.Tensor, c: torch.Tensor, tgt: torch.Tensor):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


class TranAD(nn.Module):
    """Full TranAD: 1 encoder (d_model=2*feats) + 2 parallel decoders, adversarial 2-pass."""

    def __init__(self, feats: int) -> None:
        super().__init__()
        self.name = "TranAD"
        self.n_feats = feats
        self.n_window = 10
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        self.transformer_encoder = TransformerEncoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder1 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder2 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src: torch.Tensor, c: torch.Tensor, tgt: torch.Tensor):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2


# --------------------------------------------------------------------------------------
# Single-tensor adapters: each takes the window (n_window, 1, feats) and synthesizes tgt.
# In the TranAD training loop tgt == the same window, so we set tgt = window.
# --------------------------------------------------------------------------------------

_FEATS = 7
_WINDOW = 10


class _WindowSrcTgt(nn.Module):
    """Adapter: window (W,1,F) -> model(src=window, tgt=window). tgt==window per TranAD loop."""

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, window: torch.Tensor):
        return self.core(window, window)


class _AttentionWrapper(nn.Module):
    """Adapter for the Attention baseline: a flat (n_window, feats) window in."""

    def __init__(self, core: Attention) -> None:
        super().__init__()
        self.core = core

    def forward(self, window: torch.Tensor):
        return self.core(window)


def build_tranad() -> nn.Module:
    """Full TranAD (dual-decoder transformer w/ adversarial self-conditioning)."""
    return _WindowSrcTgt(TranAD(_FEATS)).eval()


def build_tranad_basic() -> nn.Module:
    """TranAD_Basic ablation (single encoder + single decoder, d_model=feats)."""
    return _WindowSrcTgt(TranAD_Basic(_FEATS)).eval()


def build_tranad_transformer() -> nn.Module:
    """TranAD_Transformer ablation (MLP-only 'transformer', dual decoders)."""
    return _WindowSrcTgt(TranAD_Transformer(_FEATS)).eval()


def build_tranad_adversarial() -> nn.Module:
    """TranAD_Adversarial ablation (single decoder + adversarial 2-pass)."""
    return _WindowSrcTgt(TranAD_Adversarial(_FEATS)).eval()


def build_tranad_selfconditioning() -> nn.Module:
    """TranAD_SelfConditioning ablation (dual decoders, no adversarial 2nd pass)."""
    return _WindowSrcTgt(TranAD_SelfConditioning(_FEATS)).eval()


def build_attention() -> nn.Module:
    """Attention baseline (learned per-feature attention matrix)."""
    return _AttentionWrapper(Attention(_FEATS)).eval()


def example_input_window() -> torch.Tensor:
    """A (n_window=10, batch=1, feats=7) MTS window for the TranAD family."""
    return torch.randn(_WINDOW, 1, _FEATS)


def example_input_attention() -> torch.Tensor:
    """A (n_window=5, feats=7) flat window for the Attention baseline."""
    return torch.randn(5, _FEATS)


MENAGERIE_ENTRIES = [
    (
        "TranAD (dual-decoder transformer w/ adversarial self-conditioning)",
        "build_tranad",
        "example_input_window",
        "2022",
        "DC",
    ),
    (
        "TranAD_Basic (single encoder-decoder transformer ablation)",
        "build_tranad_basic",
        "example_input_window",
        "2022",
        "DC",
    ),
    (
        "TranAD_Transformer (MLP-only dual-decoder ablation)",
        "build_tranad_transformer",
        "example_input_window",
        "2022",
        "DC",
    ),
    (
        "TranAD_Adversarial (single decoder + adversarial self-conditioning ablation)",
        "build_tranad_adversarial",
        "example_input_window",
        "2022",
        "DC",
    ),
    (
        "TranAD_SelfConditioning (dual decoder, no adversarial pass ablation)",
        "build_tranad_selfconditioning",
        "example_input_window",
        "2022",
        "DC",
    ),
    (
        "Attention (learned per-feature attention-matrix MTS baseline)",
        "build_attention",
        "example_input_attention",
        "2022",
        "DC",
    ),
]
