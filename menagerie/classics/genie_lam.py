"""Genie Latent Action Model (LAM): unsupervised discrete inter-frame actions.

Bruce, Dennis, Edwards, Parker-Holder et al. (Google DeepMind), 2024,
arXiv:2402.15391 ("Genie: Generative Interactive Environments").

Genie is a foundation world model with three components: (1) a spatiotemporal
(ST) video tokenizer, (2) the **Latent Action Model (LAM)** -- the distinctive
primitive reproduced here -- and (3) a MaskGIT-style ST-transformer dynamics
model.  The LAM is a VQ-VAE-style model that infers a small set of DISCRETE
latent actions between consecutive frames in a fully UNSUPERVISED way:

  - **Encoder**: an ST-transformer over the frame sequence ``x_1..x_{t+1}``
    (spatial self-attention within each frame, then causal temporal
    self-attention across frames), producing a per-transition continuous action
    embedding.
  - **VQ quantizer**: the continuous action is snapped to the nearest of |A|
    codebook vectors (the paper uses a tiny codebook, |A|=8) via a
    straight-through nearest-neighbour lookup -- this is what forces the actions
    to be a small discrete vocabulary.
  - **Decoder**: takes the past frames ``x_1..x_t`` and the quantized latent
    action and reconstructs the next frame ``x_{t+1}``.  Trained end-to-end, the
    latent-action bottleneck is the only path the future can flow through, so it
    must carry the controllable change between frames.

This is a faithful COMPACT random-init reimplementation: tiny frames
(T=4, C=3, H=8, W=8), patch size 4 -> 4 tokens/frame, ``d_model`` small, |A|=8.
The distinctive structural primitive -- ST-transformer (spatial-then-temporal
attention) + VQ over inter-frame latent actions -- is reproduced exactly; only
the widths/lengths are shrunk so the unrolled graph stays renderable.  Pure
torch, CPU, forward-only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """Self-attention over the spatial tokens WITHIN each frame (per-frame)."""

    def __init__(self, d_model: int, n_head: int = 4) -> None:
        super().__init__()
        self.h = n_head
        self.hd = d_model // n_head
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, d_model) -- N spatial tokens per frame.
        B, T, N, d = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, N, self.h, self.hd).permute(0, 1, 3, 2, 4)
        k = k.view(B, T, N, self.h, self.hd).permute(0, 1, 3, 2, 4)
        v = v.view(B, T, N, self.h, self.hd).permute(0, 1, 3, 2, 4)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.hd**0.5), dim=-1)
        out = torch.matmul(att, v).permute(0, 1, 3, 2, 4).reshape(B, T, N, d)
        return x + self.o(out)


class TemporalAttention(nn.Module):
    """Causal self-attention across frames for a FIXED spatial token position."""

    def __init__(self, d_model: int, n_head: int = 4) -> None:
        super().__init__()
        self.h = n_head
        self.hd = d_model // n_head
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, d_model) -- attend along T, per spatial position.
        B, T, N, d = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        # bring N to the batch axis, attend over T
        q = q.view(B, T, N, self.h, self.hd).permute(0, 2, 3, 1, 4)
        k = k.view(B, T, N, self.h, self.hd).permute(0, 2, 3, 1, 4)
        v = v.view(B, T, N, self.h, self.hd).permute(0, 2, 3, 1, 4)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.hd**0.5)
        causal = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        att = torch.softmax(scores + causal, dim=-1)
        out = torch.matmul(att, v).permute(0, 3, 1, 2, 4).reshape(B, T, N, d)
        return x + self.o(out)


class STBlock(nn.Module):
    """Genie ST-transformer block: spatial attn -> temporal attn -> FFN."""

    def __init__(self, d_model: int, n_head: int = 4) -> None:
        super().__init__()
        self.spatial = SpatialAttention(d_model, n_head)
        self.temporal = TemporalAttention(d_model, n_head)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        x = self.temporal(x)
        x = x + self.ff(self.ff_norm(x))
        return x


class VectorQuantizer(nn.Module):
    """Straight-through nearest-codebook quantizer (|A| discrete latent actions)."""

    def __init__(self, n_codes: int, dim: int) -> None:
        super().__init__()
        self.codebook = nn.Embedding(n_codes, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, T-1, dim) continuous action embeddings; quantize each.
        flat = z.reshape(-1, z.shape[-1])
        cb = self.codebook.weight  # (n_codes, dim)
        dist = flat.pow(2).sum(-1, keepdim=True) - 2 * flat @ cb.t() + cb.pow(2).sum(-1)
        idx = torch.argmin(dist, dim=-1)
        q = self.codebook(idx).view_as(z)
        # straight-through estimator: gradients flow to the encoder
        return z + (q - z).detach()


class PatchEmbed(nn.Module):
    """Patchify each frame independently into spatial tokens."""

    def __init__(self, in_ch: int, d_model: int, patch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, C, H, W) -> tokens (B, T, N, d_model)
        B, T, C, H, W = frames.shape
        x = self.proj(frames.reshape(B * T, C, H, W))  # (B*T, d, h', w')
        d = x.shape[1]
        x = x.flatten(2).transpose(1, 2)  # (B*T, N, d)
        return x.view(B, T, -1, d)


class GenieLAM(nn.Module):
    """Genie Latent Action Model: encoder ST-transformer + VQ + decoder.

    forward(frames) where frames = (B, T, C, H, W):
      - encoder ST-transformer over all T frames -> per-transition action
        embedding (T-1 actions), VQ-quantized to the |A|-code vocabulary;
      - decoder ST-transformer over the past frames conditioned on the quantized
        latent action reconstructs the next-frame patch tokens.
    Returns (recon_next_frame_tokens, quantized_actions).
    """

    def __init__(
        self,
        in_ch: int = 3,
        d_model: int = 32,
        patch: int = 4,
        n_actions: int = 8,
        n_enc: int = 2,
        n_dec: int = 2,
        n_head: int = 4,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(in_ch, d_model, patch)
        self.pos = nn.Parameter(torch.zeros(1, 1, 64, d_model))  # spatial pos (<=64 tokens)
        # --- encoder: ST-transformer over the full frame sequence ---
        self.enc_blocks = nn.ModuleList([STBlock(d_model, n_head) for _ in range(n_enc)])
        self.to_action = nn.Linear(d_model, d_model)
        self.vq = VectorQuantizer(n_actions, d_model)
        # --- decoder: ST-transformer over past frames, conditioned on action ---
        self.action_proj = nn.Linear(d_model, d_model)
        self.dec_blocks = nn.ModuleList([STBlock(d_model, n_head) for _ in range(n_dec)])
        self.dec_norm = nn.LayerNorm(d_model)
        self.to_pixels = nn.Linear(d_model, d_model)

    def forward(self, frames: torch.Tensor):
        B, T, C, H, W = frames.shape
        tok = self.patch_embed(frames)  # (B, T, N, d)
        N = tok.shape[2]
        tok = tok + self.pos[:, :, :N, :]

        # Encoder ST-transformer over ALL frames (x_1..x_{t+1}).
        enc = tok
        for blk in self.enc_blocks:
            enc = blk(enc)
        # Per-transition action: pool spatial tokens, take inter-frame deltas.
        frame_summary = enc.mean(dim=2)  # (B, T, d)
        action_cont = self.to_action(frame_summary[:, 1:] - frame_summary[:, :-1])  # (B, T-1, d)
        action_q = self.vq(action_cont)  # (B, T-1, d) discrete latent actions

        # Decoder: condition the PAST frames on the latent action, reconstruct next.
        past = tok[:, :-1]  # (B, T-1, N, d) -- frames x_1..x_t
        cond = self.action_proj(action_q).unsqueeze(2)  # (B, T-1, 1, d)
        dec = past + cond
        for blk in self.dec_blocks:
            dec = blk(dec)
        # next-frame reconstruction: read the last decoded frame's tokens.
        recon = self.to_pixels(self.dec_norm(dec[:, -1]))  # (B, N, d)
        return recon, action_q


def build() -> nn.Module:
    return GenieLAM()


def example_input() -> torch.Tensor:
    """Tiny video clip ``(1, T=4, C=3, H=8, W=8)`` -- 4 frames, 3 inter-frame actions."""
    return torch.randn(1, 4, 3, 8, 8)


MENAGERIE_ENTRIES = [
    (
        "Genie-style-LAM",
        "build",
        "example_input",
        "2024",
        "DC",
    ),
]
