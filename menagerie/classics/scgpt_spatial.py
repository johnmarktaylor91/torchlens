"""scGPT / scGPT-spatial: gene-token transformer for single-cell omics.

Cui et al., "scGPT: toward building a foundation model for single-cell
multi-omics using generative AI", Nature Methods 2024.
scGPT-spatial: Wang et al., bioRxiv 2025.
Source: https://github.com/bowang-lab/scGPT (scgpt/model/model.py),
        https://github.com/bowang-lab/scGPT-spatial (scgpt_spatial/model/).

scGPT-spatial "ceilings" in the source catalog because the package is not
installed and the forward takes structured single-cell inputs (gene ids +
expression values, plus spatial coordinates) rather than a plain tensor.  This
is a faithful pure-torch reimplementation of the core forward consuming a small
fixed tuple ``[gene_ids (B,G), values (B,G), coords (B,2)]``.

Architecture (faithful to the source):
  - **GeneEncoder**: ``nn.Embedding(num_genes, D) -> LayerNorm`` on gene ids.
  - **ContinuousValueEncoder**: ``Linear(1,D) -> ReLU -> Linear(D,D) -> LayerNorm``
    on (clamped) expression values.
  - token = gene_emb + value_emb (elementwise sum; no positional encoding).
  - standard ``nn.TransformerEncoderLayer`` x nlayers (post-norm, batch_first).
  - **ExprDecoder** (base scGPT) vs **MoE expression decoder** (scGPT-spatial:
    4-expert top-2 mixture of the same expression MLP).  The cell embedding is
    ``output[:, 0, :]`` (cls token at index 0).

Note (source-verified): scGPT-spatial does NOT add a coordinate embedding to
tokens -- coordinates enter only via ``cdist`` neighbour selection in a training
loss.  The coords tensor is accepted for signature faithfulness and threaded
through a neighbour-distance op so it appears in the graph, but it is not summed
into the token stream (matching the published forward).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class GeneEncoder(nn.Module):
    def __init__(self, num_genes: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_genes, d_model, padding_idx=0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, gene_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.embedding(gene_ids))


class ContinuousValueEncoder(nn.Module):
    def __init__(self, d_model: int, max_value: float = 512.0) -> None:
        super().__init__()
        self.max_value = max_value
        self.linear1 = nn.Linear(1, d_model)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        x = values.unsqueeze(-1).clamp(max=self.max_value)
        x = self.act(self.linear1(x))
        x = self.linear2(x)
        return self.norm(x)


class ExprDecoder(nn.Module):
    """Per-gene expression prediction MLP (base scGPT)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)  # (B,G)


class MoEDecoder(nn.Module):
    """scGPT-spatial mixture-of-experts expression decoder (4 experts, top-2)."""

    def __init__(self, d_model: int, n_experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.top_k = top_k
        self.experts = nn.ModuleList([ExprDecoder(d_model) for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)  # (B,G,E)
        probs = torch.softmax(logits, dim=-1)
        topw, topi = torch.topk(probs, self.top_k, dim=-1)
        topw = topw / topw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        gate = torch.zeros_like(probs).scatter(-1, topi, topw)  # (B,G,E)
        out = torch.zeros(x.shape[:-1], device=x.device)
        for e, expert in enumerate(self.experts):
            out = out + gate[..., e] * expert(x)
        return out


class ClsDecoder(nn.Module):
    """Cell-type head over the cls token."""

    def __init__(self, d_model: int, n_cls: int, n_layers: int = 2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model)]
        layers.append(nn.Linear(d_model, n_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.net(cls)


class ScGPTSpatial(nn.Module):
    """scGPT(-spatial) core: gene+value token transformer + MoE expr head + cls head."""

    def __init__(
        self,
        num_genes: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        nlayers: int = 2,
        d_hid: int = 64,
        n_cls: int = 8,
        use_moe: bool = True,
    ) -> None:
        super().__init__()
        self.gene_encoder = GeneEncoder(num_genes, d_model)
        self.value_encoder = ContinuousValueEncoder(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_hid, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.expr_decoder = MoEDecoder(d_model) if use_moe else ExprDecoder(d_model)
        self.cls_decoder = ClsDecoder(d_model, n_cls)

    def forward(
        self, gene_ids: torch.Tensor, values: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        token = self.gene_encoder(gene_ids) + self.value_encoder(values)  # (B,G,D)
        # Coordinates enter only via a neighbour-distance op (matching the source);
        # not summed into the token stream.
        coord_dist = torch.cdist(coords, coords)  # (B?,...) -- keeps coords in the graph
        token = token + 0.0 * coord_dist.sum()
        output = self.transformer(token)  # (B,G,D)
        expr = self.expr_decoder(output)  # (B,G) reconstructed expression
        cls = self.cls_decoder(output[:, 0, :])  # (B,n_cls) cell-type logits over cls token
        return expr, cls


def build_scgpt_spatial() -> nn.Module:
    return ScGPTSpatial(
        num_genes=16, d_model=64, nhead=4, nlayers=2, d_hid=64, n_cls=8, use_moe=True
    )


def example_input_scgpt_spatial() -> List[torch.Tensor]:
    """``[gene_ids (1,16), values (1,16), coords (1,2)]`` (index 0 = cls token)."""
    g = 16
    gene_ids = torch.randint(1, 16, (1, g))
    gene_ids[:, 0] = 0  # cls token at index 0
    values = torch.rand(1, g) * 6.0
    values[:, 0] = 0.0
    coords = torch.randn(1, 2)
    return [gene_ids, values, coords]


MENAGERIE_ENTRIES = [
    (
        "scGPT-spatial (gene-token transformer + MoE expression head)",
        "build_scgpt_spatial",
        "example_input_scgpt_spatial",
        "2024",
        "DC",
    ),
]
