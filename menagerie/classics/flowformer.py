"""FlowFormer: Transformer-based optical flow estimation with cost volume encoding.

Huang et al. (2022), "FlowFormer: A Transformer Architecture for Optical Flow".
ECCV 2022.  arXiv:2203.16194.
Source: https://github.com/drinkingcoder/FlowFormer-Official

Distinctive primitives:
  1. COST VOLUME TOKENISATION: for each query pixel, compute dot-product matching
     costs against all key pixels in a local window, then tokenise the resulting
     cost volume as a sequence of tokens for transformer processing.
  2. CONTEXT ENCODER: shared CNN (SimpleCNN) encodes both frames into feature maps.
  3. ENCODER TRANSFORMER: processes the cost volume tokens using standard
     self-attention layers to produce per-pixel flow features.
  4. RECURRENT DECODER (GRU-based): iteratively refines the flow estimate using
     the context features + cost volume embeddings, similar to RAFT's GRU updater.

For the atlas: compact reproduction of the cost-volume tokenisation + transformer
encoder + GRU decoder. Full cross-frame correlation replaced with a local (H*W)
cost volume on a tiny spatial resolution.

Compact config: d_model=32, H=W=8, n_heads=2, n_enc_layers=2, n_iters=2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# Feature encoder (shared CNN for both frames)
# ==============================================================


class FeatureEncoder(nn.Module):
    """Simple convolutional feature encoder."""

    def __init__(self, d_feat: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, d_feat, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_feat, d_feat, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) -> (B, d_feat, H, W)"""
        return self.net(x)


# ==============================================================
# Cost volume tokenisation
# ==============================================================


class CostVolumeEncoder(nn.Module):
    """Compute all-pairs cost volume and encode each query pixel's correlation
    vector as a token.

    For H*W queries, each gets a cost vector of length H*W.
    Cost = dot product of query feature and all key features.
    Then a linear projection maps cost_vec (H*W) -> d_model.
    """

    def __init__(self, d_feat: int = 32, d_model: int = 32) -> None:
        super().__init__()
        # Will determine n_kv at runtime from spatial size
        self.d_feat = d_feat
        self.d_model = d_model
        self.proj = None  # lazy init

    def _ensure_proj(self, n_kv: int, device: torch.device, dtype: torch.dtype) -> None:
        if self.proj is None or self.proj.in_features != n_kv:
            self.proj = nn.Linear(n_kv, self.d_model).to(device=device, dtype=dtype)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """feat1/feat2: (B, d_feat, H, W) -> tokens (B, H*W, d_model)"""
        B, d, H, W = feat1.shape
        n = H * W
        # Flatten spatial: (B, d, n)
        f1 = feat1.view(B, d, n)  # queries
        f2 = feat2.view(B, d, n)  # keys
        # Cost volume: (B, n_q, n_k)
        cost = torch.bmm(f1.permute(0, 2, 1), f2) / (d**0.5)  # (B, n, n)
        cost = F.softmax(cost, dim=-1)  # normalise over key dimension
        # Encode each query's cost vector
        self._ensure_proj(n, cost.device, cost.dtype)
        tokens = self.proj(cost)  # (B, n, d_model)
        return tokens


# ==============================================================
# Encoder Transformer (processes cost volume tokens)
# ==============================================================


class FlowformerEncoderLayer(nn.Module):
    """One encoder layer: self-attention + FFN over cost tokens."""

    def __init__(self, d_model: int = 32, n_heads: int = 2, d_ff: int = 64) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + y
        x = x + self.ff2(F.gelu(self.ff1(self.norm2(x))))
        return x


# ==============================================================
# GRU decoder (iterative flow refinement, RAFT-style)
# ==============================================================


class GRUDecoder(nn.Module):
    """GRU-based iterative flow updater.

    Maintains hidden state (B, d_model, H, W).
    At each iteration: takes [hidden, context, cost_features] -> delta_flow.
    """

    def __init__(self, d_model: int = 32, d_context: int = 32) -> None:
        super().__init__()
        d_gru_in = d_model + d_context
        self.gru = nn.GRUCell(d_gru_in, d_model)
        self.flow_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),  # delta_flow (x, y)
        )

    def forward(
        self,
        hidden: torch.Tensor,
        cost_feat: torch.Tensor,
        context: torch.Tensor,
        n_iter: int = 2,
    ) -> torch.Tensor:
        """hidden: (B*H*W, d_model), cost_feat/context: (B*H*W, d_model/d_context).
        Returns cumulative flow (B*H*W, 2).
        """
        flow = torch.zeros(hidden.shape[0], 2, device=hidden.device)
        h = hidden
        for _ in range(n_iter):
            inp = torch.cat([cost_feat, context], dim=-1)  # (B*HW, d_gru_in)
            h = self.gru(inp, h)
            delta_flow = self.flow_head(h)
            flow = flow + delta_flow
        return flow


# ==============================================================
# FlowFormer full model
# ==============================================================


class FlowFormerOfficial(nn.Module):
    """Compact FlowFormer: feature encoder + cost-volume tokenisation +
    encoder transformer + GRU decoder -> optical flow field.

    Input: (B, 6, H, W) = frame1 (3) + frame2 (3).
    Output: (B, 2, H, W) optical flow (frame1 -> frame2).
    """

    def __init__(
        self,
        d_feat: int = 32,
        d_model: int = 32,
        n_heads: int = 2,
        n_enc_layers: int = 2,
        n_iter: int = 2,
    ) -> None:
        super().__init__()
        self.feat_enc = FeatureEncoder(d_feat)
        self.cost_enc = CostVolumeEncoder(d_feat, d_model)
        self.enc_layers = nn.ModuleList(
            [FlowformerEncoderLayer(d_model, n_heads) for _ in range(n_enc_layers)]
        )
        # Context encoder for decoder (separate light CNN)
        self.context_enc = FeatureEncoder(d_model)
        self.decoder = GRUDecoder(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 6, H, W) -> (B, 2, H, W)"""
        B, _, H, W = x.shape
        I1 = x[:, :3]
        I2 = x[:, 3:6]

        # Feature encoding
        f1 = self.feat_enc(I1)  # (B, d, H, W)
        f2 = self.feat_enc(I2)

        # Cost volume tokens: (B, H*W, d_model)
        tokens = self.cost_enc(f1, f2)

        # Encoder transformer
        for layer in self.enc_layers:
            tokens = layer(tokens)

        # Context for decoder
        ctx = self.context_enc(I1)  # (B, d, H, W)
        ctx_flat = ctx.view(B, -1, H * W).permute(0, 2, 1).reshape(B * H * W, -1)  # (B*HW, d)
        tokens_flat = tokens.reshape(B * H * W, -1)  # (B*HW, d_model)
        # Initial hidden = cost tokens
        hidden = tokens_flat

        # GRU decoder
        flow_flat = self.decoder(
            hidden,
            tokens_flat,
            ctx_flat,
            self.decoder.__class__.__init__.__defaults__[0] if False else 2,
        )  # n_iter=2

        flow = flow_flat.view(B, H, W, 2).permute(0, 3, 1, 2)  # (B, 2, H, W)
        return flow


def build_flowformer_official() -> nn.Module:
    return FlowFormerOfficial(d_feat=32, d_model=32, n_heads=2, n_enc_layers=2, n_iter=2).eval()


def example_input() -> torch.Tensor:
    """(1, 6, 8, 8) -- batch=1, frame1(3)+frame2(3), 8x8."""
    return torch.randn(1, 6, 8, 8)


MENAGERIE_ENTRIES = [
    (
        "FlowFormer (transformer optical flow: cost-volume tokenisation + encoder transformer + GRU decoder)",
        "build_flowformer_official",
        "example_input",
        "2022",
        "DC",
    ),
]
