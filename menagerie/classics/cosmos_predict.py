"""Cosmos-Predict: NVIDIA world-foundation video diffusion transformer (DiT).

NVIDIA (Agarwal et al.), "Cosmos World Foundation Model Platform for Physical AI",
2025, arXiv:2501.03575.
Source: https://github.com/nvidia-cosmos/cosmos-predict1
        (cosmos_predict1/diffusion/networks/general_dit.py, module/blocks.py)

Cosmos-Predict's diffusion world model is a **video DiT**. Its distinctive primitive
(reproduced faithfully here at tiny scale):

  - **3D patchify** of a continuous video latent ``(B, C, T, H, W)`` from the Cosmos
    continuous tokenizer into non-overlapping cubes of shape ``(p_t, p_h, p_w)``
    (the real model uses ``p_t=1, p_h=p_w=2``), flattened into a 1D spatiotemporal
    token sequence of length ``T*H*W / (p_t*p_h*p_w)``.
  - A stack of **DiT blocks**, each = self-attention -> cross-attention -> MLP, where
      * **self-attention** uses **factorized 3D RoPE** (the feature dim is split into
        three chunks, one each for temporal / height / width axes) and **QK-RMSNorm**;
      * **cross-attention** attends to a (T5-XXL) **text-conditioning** sequence;
      * every sub-layer is wrapped in **adaLN-LoRA** modulation driven by the diffusion
        **timestep** embedding: ``norm -> x*(1+scale)+shift -> sublayer -> x + gate*out``,
        with (shift, scale, gate) produced by a low-rank ``SiLU->Linear->Linear`` head.
  - A **final adaLN layer + linear unpatchify** back to ``(B, C, T, H, W)``.

This is a faithful COMPACT random-init reimpl: pure ``torch`` (no diffusers), tiny hidden
sizes and a tiny video latent ``(1, C=4, T=2, H=4, W=4)`` so the unrolled graph stays
renderable. The real models are 4B/7B/14B; we keep the block topology, not the size.
CPU, forward-only. (The autoregressive Cosmos variant — a GPT over FSQ-discrete video
tokens — is a separate family; this module reimplements the *diffusion* Predict DiT.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Per-head RMSNorm used for query/key normalization (Cosmos 'qkv_norm=RRI')."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def _build_rope_3d(t: int, h: int, w: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Factorized 3D RoPE: split head_dim into 3 axis chunks (temporal/height/width).

    Returns (cos, sin) of shape (t*h*w, head_dim), built once for the token grid.
    Mirrors VideoRopePosition3DEmb (three approximately-equal feature chunks).
    """
    # split head_dim into 3 even chunks (pad the temporal chunk to absorb remainder)
    base = head_dim // 3
    dims = [base + (head_dim - 3 * base), base, base]  # (t_dim, h_dim, w_dim), sums to head_dim
    grids = [
        torch.arange(t).view(t, 1, 1).expand(t, h, w).reshape(-1),
        torch.arange(h).view(1, h, 1).expand(t, h, w).reshape(-1),
        torch.arange(w).view(1, 1, w).expand(t, h, w).reshape(-1),
    ]
    cos_parts, sin_parts = [], []
    for axis_dim, pos in zip(dims, grids):
        half = axis_dim // 2
        if half == 0:
            cos_parts.append(torch.ones(pos.shape[0], axis_dim))
            sin_parts.append(torch.zeros(pos.shape[0], axis_dim))
            continue
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half).float() / half))
        ang = pos.float().unsqueeze(-1) * inv_freq.unsqueeze(0)  # (L, half)
        cos = torch.cos(ang).repeat_interleave(2, dim=-1)  # (L, 2*half)
        sin = torch.sin(ang).repeat_interleave(2, dim=-1)
        if 2 * half < axis_dim:  # odd axis_dim -> pad identity for the leftover slot
            cos = torch.cat([cos, torch.ones(pos.shape[0], axis_dim - 2 * half)], dim=-1)
            sin = torch.cat([sin, torch.zeros(pos.shape[0], axis_dim - 2 * half)], dim=-1)
        cos_parts.append(cos)
        sin_parts.append(sin)
    return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, L, Dh); cos/sin: (L, Dh). Standard rotate-half RoPE."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    return x * cos + rot * sin


class AdaLNLoRA(nn.Module):
    """Low-rank adaLN head producing (shift, scale, gate) from the timestep embedding."""

    def __init__(self, d_model: int, lora_dim: int = 8) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.down = nn.Linear(d_model, lora_dim, bias=False)
        self.up = nn.Linear(lora_dim, 3 * d_model)

    def forward(self, emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.up(self.down(self.act(emb))).chunk(3, dim=-1)


class SelfAttention3D(nn.Module):
    """Self-attention with factorized 3D RoPE + per-head QK RMSNorm."""

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        self.h = n_heads
        self.hd = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.q_norm(q.view(B, L, self.h, self.hd)).transpose(1, 2)
        k = self.k_norm(k.view(B, L, self.h, self.hd)).transpose(1, 2)
        v = v.view(B, L, self.h, self.hd).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.hd**0.5), dim=-1)
        out = torch.matmul(att, v).transpose(1, 2).reshape(B, L, -1)
        return self.o(out)


class CrossAttentionText(nn.Module):
    """Cross-attention: video queries attend to text (T5-XXL) keys/values."""

    def __init__(self, d_model: int, d_text: int, n_heads: int = 4) -> None:
        super().__init__()
        self.h = n_heads
        self.hd = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_text, d_model, bias=False)
        self.v = nn.Linear(d_text, d_model, bias=False)
        self.q_norm = RMSNorm(self.hd)
        self.k_norm = RMSNorm(self.hd)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        S = text.shape[1]
        q = self.q_norm(self.q(x).view(B, L, self.h, self.hd)).transpose(1, 2)
        k = self.k_norm(self.k(text).view(B, S, self.h, self.hd)).transpose(1, 2)
        v = self.v(text).view(B, S, self.h, self.hd).transpose(1, 2)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.hd**0.5), dim=-1)
        out = torch.matmul(att, v).transpose(1, 2).reshape(B, L, -1)
        return self.o(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, mult: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * mult)
        self.fc2 = nn.Linear(d_model * mult, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CosmosDITBlock(nn.Module):
    """Cosmos DiT block: self-attn(3D-RoPE) -> cross-attn(text) -> MLP, each adaLN-LoRA gated."""

    def __init__(self, d_model: int, d_text: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = SelfAttention3D(d_model, n_heads)
        self.ada1 = AdaLNLoRA(d_model)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cross_attn = CrossAttentionText(d_model, d_text, n_heads)
        self.ada2 = AdaLNLoRA(d_model)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = MLP(d_model)
        self.ada3 = AdaLNLoRA(d_model)

    def forward(self, x, text, emb, cos, sin):
        s, sc, g = self.ada1(emb)
        x = x + g.unsqueeze(1) * self.self_attn(_modulate(self.norm1(x), s, sc), cos, sin)
        s, sc, g = self.ada2(emb)
        x = x + g.unsqueeze(1) * self.cross_attn(_modulate(self.norm2(x), s, sc), text)
        s, sc, g = self.ada3(emb)
        x = x + g.unsqueeze(1) * self.mlp(_modulate(self.norm3(x), s, sc))
        return x


class CosmosPredictDiT(nn.Module):
    """Compact Cosmos-Predict diffusion video transformer.

    Forward consumes a single video-latent tensor (B, C, T, H, W); the diffusion
    timestep and a synthesized text-conditioning sequence are produced internally so
    the module is traceable from one positional input.
    """

    def __init__(
        self,
        in_channels: int = 4,
        d_model: int = 48,
        d_text: int = 32,
        n_blocks: int = 3,
        n_heads: int = 4,
        patch_t: int = 1,
        patch_h: int = 2,
        patch_w: int = 2,
        text_len: int = 6,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.pt, self.ph, self.pw = patch_t, patch_h, patch_w
        self.d_model = d_model
        patch_dim = in_channels * patch_t * patch_h * patch_w
        # 3D patchify = linear over flattened (C * p_t * p_h * p_w) cube
        self.x_embedder = nn.Linear(patch_dim, d_model)
        # timestep embedding -> conditioning vector that drives adaLN-LoRA
        self.t_embedder = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        # synthetic learnable text (stands in for T5-XXL embeddings)
        self.text = nn.Parameter(torch.randn(1, text_len, d_text) * 0.02)
        self.blocks = nn.ModuleList(
            [CosmosDITBlock(d_model, d_text, n_heads) for _ in range(n_blocks)]
        )
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.unpatch = nn.Linear(d_model, patch_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = latent.shape
        pt, ph, pw = self.pt, self.ph, self.pw
        nt, nh, nw = T // pt, H // ph, W // pw
        # 3D patchify into non-overlapping (pt,ph,pw) cubes -> token sequence
        x = latent.view(B, C, nt, pt, nh, ph, nw, pw)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, nt * nh * nw, C * pt * ph * pw)
        x = self.x_embedder(x)  # (B, L, d_model)

        # diffusion timestep -> adaLN conditioning embedding
        tval = torch.full((B, 1), 0.5, device=latent.device, dtype=latent.dtype)
        emb = self.t_embedder(tval)  # (B, d_model)

        text = self.text.expand(B, -1, -1)
        head_dim = self.d_model // self.blocks[0].self_attn.h
        cos, sin = _build_rope_3d(nt, nh, nw, head_dim)
        cos = cos.to(latent.dtype)
        sin = sin.to(latent.dtype)

        for blk in self.blocks:
            x = blk(x, text, emb, cos, sin)

        shift, scale = self.final_ada(emb).chunk(2, dim=-1)
        x = _modulate(self.final_norm(x), shift, scale)
        x = self.unpatch(x)  # (B, L, C*pt*ph*pw)

        # unpatchify back to (B, C, T, H, W)
        x = x.view(B, nt, nh, nw, C, pt, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(B, C, T, H, W)
        return x


def build() -> nn.Module:
    return CosmosPredictDiT()


def example_input() -> torch.Tensor:
    """Tiny video latent ``(1, C=4, T=2, H=4, W=4)`` from a Cosmos-style tokenizer."""
    return torch.randn(1, 4, 2, 4, 4)


MENAGERIE_ENTRIES = [
    (
        "Cosmos-Predict (NVIDIA world-foundation video DiT)",
        "build",
        "example_input",
        "2025",
        "DC",
    ),
]
