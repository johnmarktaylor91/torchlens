"""WavTokenizer (neural audio codec) + WuKong (stacked-FM recommendation).

Two architectures with no prior in-tree implementation, reproduced as compact random-init
classics. (Filename uses the agent's `ri_vision_a` namespace though these are audio/reco.)

  * WavTokenizer (Ji et al., 2024, arXiv:2408.16532, github jishengpeng/WavTokenizer)
      - A neural audio codec / discrete tokenizer: 1D-conv ENCODER downsamples a raw
        waveform to a latent sequence, a VECTOR-QUANTIZATION bottleneck snaps each latent
        to the nearest codebook entry (the "tokens"), with a straight-through estimator
        (q + (x - q).detach()) so gradients flow; a 1D conv-transpose DECODER reconstructs
        the waveform. WavTokenizer's claim is an extremely compact single-codebook codec
        (40/75 tokens per second). The distinctive primitive reproduced here is the
        encoder -> VQ codebook (cdist argmin + straight-through) -> decoder codec path.

  * WuKong (Zhang et al., Meta, 2024, arXiv:2403.02545)
      - A recommendation / CTR architecture built PURELY from stacked Factorization
        Machines, designed to give recommendation models an LLM-style scaling law.
      - Sparse categorical fields -> embeddings (one row per field).
      - Each WuKong layer = a Factorization-Machine Block (FMB) in parallel with a Linear
        Compress Block (LCB), with residual + LayerNorm:
          * FMB: compute the pairwise interaction matrix (E @ E^T over fields), flatten it,
            pass through an MLP, reshape back to `lcb_features` field-embeddings -- captures
            any-order interactions by stacking.
          * LCB: a linear (field-mixing) compression of the input embeddings -> the same
            embedding shape, providing a linear pass-through path.
      - Stacking taller/wider WuKong layers captures higher-order interactions; a final MLP
        head maps the pooled interaction embeddings to a CTR logit.

Compact random-init reimplementations; small audio length / few fields & embed dim. The
distinctive primitives (VQ codec path; stacked FMB+LCB interaction blocks) are faithful.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ===========================================================================
# WavTokenizer: conv encoder -> VQ codebook -> conv decoder
# ===========================================================================
class _VQBottleneck(nn.Module):
    """Vector-quantization bottleneck: nearest-codebook snap with straight-through estimator."""

    def __init__(self, d: int = 64, n_codes: int = 256) -> None:
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(n_codes, d) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, t = x.shape
        xf = x.permute(0, 2, 1).reshape(-1, d)  # (B*T, D)
        dists = torch.cdist(xf.unsqueeze(0), self.codebook.unsqueeze(0)).squeeze(0)
        idx = dists.argmin(-1)  # nearest code index = the discrete token
        q = self.codebook[idx].reshape(b, t, d).permute(0, 2, 1)
        return q + (x - q).detach()  # straight-through estimator


class _WavTokenizer(nn.Module):
    def __init__(self, d: int = 64, n_codes: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, d, 7, padding=3),
            nn.ReLU(True),
            nn.Conv1d(d, d, 4, stride=4),
            nn.ReLU(True),
            nn.Conv1d(d, d, 4, stride=4),
        )
        self.vq = _VQBottleneck(d, n_codes)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(d, d, 4, stride=4),
            nn.ReLU(True),
            nn.ConvTranspose1d(d, d, 4, stride=4),
            nn.ReLU(True),
            nn.Conv1d(d, 1, 7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        zq = self.vq(z)
        return self.decoder(zq)


def build_wavtokenizer() -> nn.Module:
    """WavTokenizer single-codebook audio codec (encoder -> VQ -> decoder)."""
    return _WavTokenizer(d=64, n_codes=256)


def example_input_wavtokenizer() -> torch.Tensor:
    """Raw mono waveform (1, 1, 4096)."""
    return torch.randn(1, 1, 4096)


# ===========================================================================
# WuKong: stacked Factorization-Machine + Linear-Compress blocks
# ===========================================================================
class _FMB(nn.Module):
    """Factorization-Machine Block: pairwise field interactions -> flatten -> MLP -> reshape."""

    def __init__(self, num_fields: int, embed_dim: int, out_fields: int, hidden: int = 64) -> None:
        super().__init__()
        self.out_fields = out_fields
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(num_fields * num_fields, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, out_fields * embed_dim),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # emb: (B, F, E) -> pairwise interaction matrix (B, F, F)
        interact = torch.bmm(emb, emb.transpose(1, 2))
        flat = interact.flatten(1)  # (B, F*F)
        out = self.proj(flat)
        return out.view(emb.shape[0], self.out_fields, self.embed_dim)


class _LCB(nn.Module):
    """Linear Compress Block: linear field-mixing of the embeddings (pass-through path)."""

    def __init__(self, num_fields: int, out_fields: int) -> None:
        super().__init__()
        self.mix = nn.Linear(num_fields, out_fields)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # mix over the field dimension: (B, F, E) -> (B, F', E)
        x = emb.transpose(1, 2)  # (B, E, F)
        x = self.mix(x)  # (B, E, F')
        return x.transpose(1, 2)


class _WuKongLayer(nn.Module):
    """One WuKong layer: FMB || LCB, summed, residual + LayerNorm."""

    def __init__(self, num_fields: int, embed_dim: int) -> None:
        super().__init__()
        self.fmb = _FMB(num_fields, embed_dim, num_fields)
        self.lcb = _LCB(num_fields, num_fields)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        out = self.fmb(emb) + self.lcb(emb)
        return self.norm(out + emb)  # residual


class _WuKong(nn.Module):
    """WuKong CTR model: field embeddings -> stacked FMB+LCB layers -> MLP -> logit."""

    def __init__(
        self, num_fields: int = 4, vocab: int = 128, embed_dim: int = 8, num_layers: int = 3
    ) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.embeddings = nn.ModuleList([nn.Embedding(vocab, embed_dim) for _ in range(num_fields)])
        self.layers = nn.ModuleList(
            [_WuKongLayer(num_fields, embed_dim) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(num_fields * embed_dim, 32), nn.ReLU(True), nn.Linear(32, 1)
        )

    def forward(self, field_ids: torch.Tensor) -> torch.Tensor:
        # field_ids: (B, num_fields) integer category ids
        emb = torch.stack(
            [self.embeddings[i](field_ids[:, i]) for i in range(self.num_fields)], dim=1
        )  # (B, F, E)
        for layer in self.layers:
            emb = layer(emb)
        pooled = emb.flatten(1)
        return torch.sigmoid(self.head(pooled))


def build_wukong() -> nn.Module:
    """WuKong stacked-Factorization-Machine recommendation/CTR network."""
    return _WuKong(num_fields=4, vocab=128, embed_dim=8, num_layers=3)


def example_input_wukong() -> torch.Tensor:
    """Categorical field ids (1, 4) int64."""
    return torch.randint(0, 128, (1, 4), dtype=torch.int64)


MENAGERIE_ENTRIES = [
    (
        "WavTokenizer (single-codebook VQ neural audio codec)",
        "build_wavtokenizer",
        "example_input_wavtokenizer",
        "2024",
        "DC",
    ),
    (
        "WuKong (stacked-Factorization-Machine recommendation)",
        "build_wukong",
        "example_input_wukong",
        "2024",
        "DC",
    ),
]
