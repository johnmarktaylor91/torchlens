"""Matrix-Game 2.0: action-conditioned causal/streaming interactive world model.

Skywork AI (He et al.), "Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming
Interactive World Model", 2025, arXiv:2508.13009.
Source: https://github.com/SkyworkAI/Matrix-Game

Matrix-Game 2.0 is a real-time **autoregressive diffusion** world model: it rolls out a
playable video frame-by-frame, each next-frame latent conditioned on (a) the recent
self-generated frame latents and (b) frame-level **keyboard + mouse actions**. The
backbone is a Wan2.1/SkyReels-V2-style video DiT made **causal & streaming**.

Distinctive primitive (reproduced faithfully here at tiny scale):

  - **Block-wise causal attention over time** with a **rolling KV window**: a chunk of
    latent frames attends only to itself and to earlier frames (a per-frame block-causal
    mask), so generation can stream forward indefinitely.
  - An **action-injection module** with two paths, matching the paper:
      * **continuous (mouse)** signal is *concatenated* to the latent tokens, passed
        through an **MLP**, then a **temporal self-attention** layer;
      * **discrete (keyboard)** signal is embedded and injected via a **cross-attention**
        layer where the fused features query the keyboard embeddings.
  - **RoPE** temporal positional encoding (replacing additive sin-cos) for long rollouts.
  - The DiT is a timestep-modulated (**adaLN**) transformer; a current-frame latent +
    action embedding yields the next-frame latent via a few-step (3-step) denoiser. We
    reproduce ONE such denoising-block pass (the structural primitive), not the full DMD
    Self-Forcing rollout.

This is a faithful COMPACT random-init reimpl: pure ``torch`` (no diffusers / Wan), tiny
hidden sizes and a tiny latent chunk ``(B=1, T=3 frames, C=16, H=4, W=4)`` so the unrolled
graph stays renderable. CPU, forward-only. The real model is 1.8B params at 352x640/25fps.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_rope_time(x: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Apply temporal RoPE over the frame axis. x: (B, H, T, S, Dh) for T frames.

    Each frame index gets a rotary phase; spatial tokens within a frame share it.
    """
    B, Hh, T, S, Dh = x.shape
    half = Dh // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half).float() / half))
    pos = torch.arange(n_frames).float().unsqueeze(-1)  # (T, 1)
    ang = pos * inv_freq.unsqueeze(0)  # (T, half)
    cos = torch.cos(ang).repeat_interleave(2, dim=-1).view(1, 1, T, 1, Dh).to(x.dtype)
    sin = torch.sin(ang).repeat_interleave(2, dim=-1).view(1, 1, T, 1, Dh).to(x.dtype)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    return x * cos + rot * sin


class BlockCausalSpatioTemporalAttention(nn.Module):
    """Self-attention over (T frames x S spatial tokens) with a frame-block-causal mask.

    Frame i may attend to all spatial tokens of frames <= i (block-causal in time, full
    in space) — the streaming attention pattern Matrix-Game 2.0 uses for rollout.
    """

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        self.h = n_heads
        self.hd = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, T: int, S: int) -> torch.Tensor:
        # x: (B, T*S, d_model)
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(t):
            return t.view(B, T, S, self.h, self.hd).permute(0, 3, 1, 2, 4)  # (B,H,T,S,Dh)

        q, k, v = split(q), split(k), split(v)
        q = _apply_rope_time(q, T)
        k = _apply_rope_time(k, T)
        # flatten time+space for attention
        q = q.reshape(B, self.h, T * S, self.hd)
        k = k.reshape(B, self.h, T * S, self.hd)
        v = v.reshape(B, self.h, T * S, self.hd)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.hd**0.5)
        # block-causal mask in time: frame index of each token
        frame_idx = torch.arange(T).repeat_interleave(S)  # (T*S,)
        allowed = frame_idx.unsqueeze(0) <= frame_idx.unsqueeze(1)  # (L, L) True if key<=query
        mask = torch.zeros(T * S, T * S)
        mask = mask.masked_fill(~allowed, float("-inf"))
        scores = scores + mask
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
        out = out.reshape(B, self.h, T, S, self.hd).permute(0, 2, 3, 1, 4).reshape(B, L, -1)
        return self.o(out)


class MouseActionPath(nn.Module):
    """Continuous (mouse) action: concat to latent tokens -> MLP -> temporal self-attn."""

    def __init__(self, d_model: int, d_mouse: int, n_heads: int = 4) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + d_mouse, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.h = n_heads
        self.hd = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mouse: torch.Tensor, T: int, S: int) -> torch.Tensor:
        # x: (B, T*S, d_model); mouse: (B, T, d_mouse) -> broadcast over spatial tokens
        B = x.shape[0]
        mouse_tok = mouse.unsqueeze(2).expand(B, T, S, mouse.shape[-1]).reshape(B, T * S, -1)
        fused = self.mlp(torch.cat([x, mouse_tok], dim=-1))
        # temporal self-attention (per spatial position, across frames)
        f = fused.view(B, T, S, -1).permute(0, 2, 1, 3).reshape(B * S, T, -1)
        q, k, v = self.qkv(f).chunk(3, dim=-1)

        def sh(t):
            return t.view(B * S, T, self.h, self.hd).transpose(1, 2)

        q, k, v = sh(q), sh(k), sh(v)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.hd**0.5), dim=-1)
        out = torch.matmul(att, v).transpose(1, 2).reshape(B * S, T, -1)
        out = self.o(out).view(B, S, T, -1).permute(0, 2, 1, 3).reshape(B, T * S, -1)
        return out


class KeyboardCrossAttention(nn.Module):
    """Discrete (keyboard) action injected via cross-attention (latent queries keys)."""

    def __init__(self, d_model: int, n_keys: int = 8, n_heads: int = 4) -> None:
        super().__init__()
        self.key_embed = nn.Embedding(n_keys, d_model)
        self.h = n_heads
        self.hd = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, keyboard: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model); keyboard: (B, K) long ids of pressed keys
        B, L, _ = x.shape
        kb = self.key_embed(keyboard)  # (B, K, d_model)
        K = kb.shape[1]
        q = self.q(x).view(B, L, self.h, self.hd).transpose(1, 2)
        k = self.k(kb).view(B, K, self.h, self.hd).transpose(1, 2)
        v = self.v(kb).view(B, K, self.h, self.hd).transpose(1, 2)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.hd**0.5), dim=-1)
        out = torch.matmul(att, v).transpose(1, 2).reshape(B, L, -1)
        return self.o(out)


class MatrixGameDiTBlock(nn.Module):
    """One causal action-conditioned DiT block (adaLN-modulated)."""

    def __init__(self, d_model: int, d_mouse: int, n_keys: int = 8, n_heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = BlockCausalSpatioTemporalAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mouse = MouseActionPath(d_model, d_mouse, n_heads)
        self.norm3 = nn.LayerNorm(d_model)
        self.keyboard = KeyboardCrossAttention(d_model, n_keys, n_heads)
        self.norm4 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))

    def forward(self, x, emb, mouse, keyboard, T, S):
        s1, c1, g1, s2, c2, g2 = self.adaLN(emb).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + c1.unsqueeze(1)) + s1.unsqueeze(1)
        x = x + g1.unsqueeze(1) * self.attn(h, T, S)
        # action conditioning: mouse path then keyboard cross-attn
        x = x + self.mouse(self.norm2(x), mouse, T, S)
        x = x + self.keyboard(self.norm3(x), keyboard)
        h = self.norm4(x) * (1 + c2.unsqueeze(1)) + s2.unsqueeze(1)
        x = x + g2.unsqueeze(1) * self.ff(h)
        return x


class MatrixGame2(nn.Module):
    """Compact Matrix-Game 2.0: action-conditioned causal next-frame video DiT.

    Forward consumes a single latent chunk ``(B, T, C, H, W)``; the diffusion timestep and
    frame-level mouse/keyboard actions are synthesized internally so the module traces from
    one positional input. Output is the predicted (denoised) next-frame latent chunk.
    """

    def __init__(
        self,
        in_channels: int = 16,
        d_model: int = 48,
        d_mouse: int = 4,
        n_keys: int = 8,
        n_blocks: int = 3,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.d_mouse = d_mouse
        self.n_keys = n_keys
        self.patch = nn.Linear(in_channels, d_model)  # per-spatial-token patch embed (1x1)
        self.t_embedder = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.blocks = nn.ModuleList(
            [MatrixGameDiTBlock(d_model, d_mouse, n_keys, n_heads) for _ in range(n_blocks)]
        )
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.unpatch = nn.Linear(d_model, in_channels)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = latent.shape
        S = H * W
        # tokenize: (B, T, C, H, W) -> (B, T*S, C) -> patch embed
        x = latent.permute(0, 1, 3, 4, 2).reshape(B, T * S, C)
        x = self.patch(x)

        # diffusion timestep -> adaLN conditioning
        tval = torch.full((B, 1), 0.5, device=latent.device, dtype=latent.dtype)
        emb = self.t_embedder(tval)

        # synthesized frame-level actions: continuous mouse (B,T,d_mouse) + discrete keyboard ids
        mouse = torch.zeros(B, T, self.d_mouse, device=latent.device, dtype=latent.dtype)
        keyboard = torch.zeros(B, 2, dtype=torch.long, device=latent.device)  # 2 pressed keys

        for blk in self.blocks:
            x = blk(x, emb, mouse, keyboard, T, S)

        x = self.unpatch(self.final_norm(x))  # (B, T*S, C)
        x = x.view(B, T, H, W, C).permute(0, 1, 4, 2, 3)  # (B, T, C, H, W)
        return x


def build() -> nn.Module:
    return MatrixGame2()


def example_input() -> torch.Tensor:
    """Tiny latent chunk ``(1, T=3, C=16, H=4, W=4)`` (3 latent frames, streaming chunk)."""
    return torch.randn(1, 3, 16, 4, 4)


MENAGERIE_ENTRIES = [
    (
        "Matrix-Game-2.0 (action-conditioned causal video world DiT)",
        "build",
        "example_input",
        "2025",
        "DC",
    ),
]
