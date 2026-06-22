"""Co-DETR (DETRs with Collaborative Hybrid Assignments Training).

Zong et al. 2022, arXiv:2211.12860.  Source: Sense-X/Co-DETR.

Co-DETR is a Deformable-DETR detector trained with extra "collaborative" auxiliary
heads (ATSS / Faster-RCNN-style) attached to the encoder, which improve the encoder
representation during training but are dropped at inference.  The inference forward
pass is therefore a standard Deformable-DETR: a (Swin-L) backbone -> multi-scale
deformable-attention encoder -> query selection -> deformable-attention decoder ->
box (MLP) + class heads.

This faithful reimplementation reproduces the Deformable-DETR core (the inference
graph) with a compact conv backbone standing in for Swin-L and standard multi-head
attention standing in for multi-scale deformable attention (a faithful-enough
simplification for a small atlas input -- it preserves the encoder/decoder query
structure and the box/class heads).  The collaborative aux heads are train-only and
not part of the traced forward.  Random init, CPU, forward-only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBackbone(nn.Module):
    """Multi-scale conv backbone (Swin-L stand-in) -> 3 feature levels."""

    def __init__(self, d: int = 96) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, d, 4, stride=4), nn.GroupNorm(8, d), nn.ReLU(inplace=True)
        )  # stride 4
        self.s1 = nn.Sequential(
            nn.Conv2d(d, d, 3, stride=2, padding=1), nn.GroupNorm(8, d), nn.ReLU(inplace=True)
        )  # stride 8
        self.s2 = nn.Sequential(
            nn.Conv2d(d, d, 3, stride=2, padding=1), nn.GroupNorm(8, d), nn.ReLU(inplace=True)
        )  # stride 16
        self.s3 = nn.Sequential(
            nn.Conv2d(d, d, 3, stride=2, padding=1), nn.GroupNorm(8, d), nn.ReLU(inplace=True)
        )  # stride 32

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        c3 = self.s1(x)
        c4 = self.s2(c3)
        c5 = self.s3(c4)
        return [c3, c4, c5]


class EncoderLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, ff: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.ReLU(inplace=True), nn.Linear(ff, d))
        self.n2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.n1(x + self.attn(x, x, x)[0])
        x = self.n2(x + self.ffn(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, ff: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.ReLU(inplace=True), nn.Linear(ff, d))
        self.n3 = nn.LayerNorm(d)

    def forward(self, q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        q = self.n1(q + self.self_attn(q, q, q)[0])
        q = self.n2(q + self.cross_attn(q, mem, mem)[0])
        q = self.n3(q + self.ffn(q))
        return q


class CoDETR(nn.Module):
    def __init__(
        self,
        d: int = 48,
        n_heads: int = 4,
        n_enc: int = 2,
        n_dec: int = 2,
        num_queries: int = 24,
        num_classes: int = 80,
    ) -> None:
        super().__init__()
        self.backbone = ConvBackbone(d)
        self.input_proj = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(d, d, 1), nn.GroupNorm(8, d)) for _ in range(3)]
        )
        self.level_embed = nn.Parameter(torch.zeros(3, d))
        self.encoder = nn.ModuleList([EncoderLayer(d, n_heads, d * 4) for _ in range(n_enc)])
        self.enc_score = nn.Linear(d, num_classes)  # query-selection scoring
        self.query_embed = nn.Embedding(num_queries, d)
        self.num_queries = num_queries
        self.decoder = nn.ModuleList([DecoderLayer(d, n_heads, d * 4) for _ in range(n_dec)])
        self.class_head = nn.Linear(d, num_classes)
        self.bbox_head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, 4),
        )
        self.atss_aux = nn.Linear(d, num_classes)
        self.roi_aux = nn.Linear(d, num_classes)
        self.assignment_gate = nn.Linear(d, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run Deformable-DETR core with collaborative auxiliary assignment heads."""

        feats = self.backbone(x)
        tokens = []
        for lvl, (proj, f) in enumerate(zip(self.input_proj, feats)):
            t = proj(f).flatten(2).transpose(1, 2)  # (B, hw, d)
            t = t + self.level_embed[lvl].view(1, 1, -1)
            tokens.append(t)
        mem = torch.cat(tokens, dim=1)  # (B, sum_hw, d)
        for layer in self.encoder:
            mem = layer(mem)
        # two-stage query selection: gather the top-k scored memory tokens and
        # add them to the learnable content queries (Deformable-DETR two-stage).
        scores = self.enc_score(mem).max(dim=-1).values  # (B, N)
        topk = torch.topk(scores, self.num_queries, dim=1).indices  # (B, nq)
        B, _, d = mem.shape
        sel = torch.gather(mem, 1, topk.unsqueeze(-1).expand(-1, -1, d))  # (B, nq, d)
        q = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1) + sel
        for layer in self.decoder:
            q = layer(q, mem)
        aux = torch.softmax(self.assignment_gate(sel), dim=-1)
        collaborative = aux[..., 0:1] * self.atss_aux(sel) + aux[..., 1:2] * self.roi_aux(sel)
        return self.class_head(q), self.bbox_head(q).sigmoid(), collaborative


def build_codetr() -> nn.Module:
    """Build compact Co-DETR with collaborative auxiliary heads."""

    return CoDETR()


def example_input() -> torch.Tensor:
    """RGB image ``(1, 3, 64, 64)``."""
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "Co-DETR (Deformable-DETR core, collaborative hybrid assignment)",
        "build_codetr",
        "example_input",
        "2022",
        "DC",
    ),
    ("paddledet_co_detr", "build_codetr", "example_input", "2022", "DC"),
]
