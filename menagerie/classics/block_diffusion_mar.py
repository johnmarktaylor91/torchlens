"""Block-Diffusion (BD3-LM) and MAR with Diffusion Loss.

Block-Diffusion (BD3-LM):
  Arriola et al., "Block Diffusion: Interpolating Between Autoregressive and
  Diffusion Language Models." arXiv:2406.09405 (2024).
  Source: https://github.com/kuleshov-group/bd3lms

MAR (Masked Autoregressive generation) with Diffusion Loss:
  Li et al., "Autoregressive Image Generation without Vector Quantization."
  arXiv:2406.11838 (2024). ICLR 2025.
  Source: https://github.com/LTH14/mar

------------------------------------------------------------------------------
BD3-LM (Block-Diffusion) distinctive primitive:
  Interpolates between autoregressive (AR) and diffusion language models by
  operating ACROSS blocks autoregressively while operating WITHIN each block
  as a diffusion model. The network uses a BLOCK-CAUSAL attention mask:

    Block i attends to ALL tokens in blocks 0..i-1  (AR dependency)
    Block i attends to ALL tokens in block i         (within-block diffusion)
    Block i does NOT attend to tokens in blocks i+1 ..  (causal)

  The transformer also receives a time embedding t (one per block, broadcast
  to its tokens) injected via adaLN-style conditioning, since within-block
  denoising is a diffusion process.

Faithful-compact simplifications:
  - 3 blocks of 4 tokens each (12 total).
  - d_model=32, 4 heads, 2 transformer layers.
  - adaLN-Zero conditioning with sinusoidal time embed.
  - Time t is one scalar per block (3,), broadcast per block's tokens.

------------------------------------------------------------------------------
MAR with Diffusion Loss distinctive primitive:
  A BIDIRECTIONAL transformer (MAE-style) autoregressively predicts tokens in
  RANDOM ORDER (masked autoregressive, not left-to-right). Each masked position
  gets a learnable mask token. The distinctive feature is the DIFFUSION LOSS
  HEAD: instead of a softmax over codebook, each predicted token embedding feeds
  a small per-token DIFFUSION MLP (a tiny UNet-like denoiser) that denoises a
  noisy continuous latent. This replaces cross-entropy and allows continuous
  token representations.

  Network structure:
    bidirectional transformer on (N tokens, some masked)
  + for each masked position: small MLP denoiser (z_t, t, condition -> z_0_hat)
  The denoiser is a 2-layer MLP conditioned on the transformer output + time.

Faithful-compact simplifications:
  - 8 tokens, 4 masked.
  - Bidirectional transformer: 2 layers, d_model=32, 4 heads.
  - Diffusion MLP denoiser: 2 hidden layers of width 32.
  - No VQ codebook, no continuous VAE latents -- uses d_model-dim vectors.
  - Random init, CPU, forward-only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Shared utilities
# =============================================================================


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        assert d % 2 == 0
        self.d = d

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (...,) float  ->  (..., d)"""
        half = self.d // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t.unsqueeze(-1) * freqs  # (..., half)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class AdaLNZero(nn.Module):
    def __init__(self, d_model: int, d_cond: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_cond, 6 * d_model))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.mlp(c).chunk(6, dim=-1)


class AdaLNBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_cond: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaln = AdaLNZero(d_model, d_cond)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(
        self,
        x: torch.Tensor,  # (T, d_model)
        c: torch.Tensor,  # (T, d_cond) OR (d_cond,) broadcast
        attn_mask: torch.Tensor | None = None,  # (T, T)
    ) -> torch.Tensor:
        T, D = x.shape
        if c.dim() == 1:
            c = c.unsqueeze(0).expand(T, -1)
        alpha_a, beta_a, gamma_a, alpha_f, beta_f, gamma_f = self.adaln(c)

        # Attention
        xn = self.norm1(x) * (1 + alpha_a) + beta_a
        qkv = self.qkv(xn).view(T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(1)
        attn = torch.einsum("ihd,jhd->ijh", q, k) / math.sqrt(self.d_head)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.unsqueeze(-1), float("-inf"))
        attn = F.softmax(attn, dim=1)
        out = torch.einsum("ijh,jhd->ihd", attn, v).reshape(T, D)
        out = self.proj(out)
        x = x + gamma_a * out

        # FFN
        xn = self.norm2(x) * (1 + alpha_f) + beta_f
        x = x + gamma_f * self.ffn(xn)
        return x


# =============================================================================
# BD3-LM (Block Diffusion)
# =============================================================================


def _block_causal_mask(n_blocks: int, block_size: int) -> torch.Tensor:
    """Build block-causal attention mask (True = blocked).

    Block i attends to blocks 0..i (causal across blocks, dense within block).
    """
    T = n_blocks * block_size
    mask = torch.ones(T, T, dtype=torch.bool)
    for i in range(n_blocks):
        for j in range(i + 1):  # block i can attend to blocks 0..i
            r0, r1 = i * block_size, (i + 1) * block_size
            c0, c1 = j * block_size, (j + 1) * block_size
            mask[r0:r1, c0:c1] = False
    return mask  # True = blocked


class BlockDiffusionLM(nn.Module):
    """BD3-LM: Block-Causal masked transformer with per-block time conditioning.

    Tokens are arranged in n_blocks of block_size.
    Block i autoregressively depends on blocks 0..i-1.
    Within-block diffusion conditioned on time t_i.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        n_blocks: int = 3,
        block_size: int = 4,
        d_time: int = 32,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.block_size = block_size
        T = n_blocks * block_size

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(T, d_model)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbed(d_time),
            nn.Linear(d_time, 4 * d_time),
            nn.SiLU(),
            nn.Linear(4 * d_time, 4 * d_time),
        )
        d_cond = 4 * d_time
        self.blocks = nn.ModuleList([AdaLNBlock(d_model, n_heads, d_cond) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.register_buffer("attn_mask", _block_causal_mask(n_blocks, block_size))

    def forward(
        self,
        tokens: torch.Tensor,  # (T,) int64
        block_times: torch.Tensor,  # (n_blocks,) float, one per block
    ) -> torch.Tensor:
        """Returns (T, vocab_size) logits."""
        T = tokens.size(0)
        x = self.tok_embed(tokens) + self.pos_embed(torch.arange(T, device=tokens.device))

        # Time conditioning: per-block time, broadcast to each block's tokens
        t_cond = self.time_embed(block_times)  # (n_blocks, d_cond)
        # Build per-token condition by repeating block time for block_size tokens
        c = t_cond.repeat_interleave(self.block_size, dim=0)  # (T, d_cond)

        for blk in self.blocks:
            x = blk(x, c, attn_mask=self.attn_mask)

        return self.head(self.norm_out(x))


def build_block_diffusion() -> nn.Module:
    return BlockDiffusionLM(
        vocab_size=64, d_model=32, n_heads=4, n_layers=2, n_blocks=3, block_size=4, d_time=32
    )


def example_input_block_diffusion() -> list[torch.Tensor]:
    torch.manual_seed(8)
    tokens = torch.randint(0, 64, (12,))  # 3 blocks * 4
    block_times = torch.rand(3)
    return [tokens, block_times]


# =============================================================================
# MAR (Masked Autoregressive with Diffusion Loss)
# =============================================================================


class DiffusionMLPHead(nn.Module):
    """Per-token diffusion-MLP denoiser (the 'diffusion loss' head in MAR).

    Takes (z_t, t, condition) where:
      z_t    : noisy token embedding  (d_model,)
      t      : scalar noise time
      cond   : transformer output for this position  (d_model,)
    Returns z_0_hat (denoised embedding, d_model,).

    This replaces the usual softmax cross-entropy loss head.
    """

    def __init__(self, d_model: int, d_time: int = 32) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbed(d_time)
        # Input: z_t + cond + time_embed all d_model-sized (concat)
        self.net = nn.Sequential(
            nn.Linear(d_model + d_model + d_time, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        z_t: torch.Tensor,  # (M, d_model)
        t: torch.Tensor,  # (M,) or scalar
        cond: torch.Tensor,  # (M, d_model) transformer outputs
    ) -> torch.Tensor:
        """Returns z_0_hat (M, d_model)."""
        M = z_t.size(0)
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(M)
        t_emb = self.time_embed(t)  # (M, d_time)
        inp = torch.cat([z_t, cond, t_emb], dim=-1)
        return self.net(inp)


class MARModel(nn.Module):
    """MAR: Masked Autoregressive model with diffusion loss.

    Forward path:
      1. Embed all tokens; replace masked positions with a learnable mask token.
      2. Run a BIDIRECTIONAL transformer (no causal mask).
      3. At masked positions, apply the diffusion-MLP denoiser.
      4. At non-masked positions, pass through as-is (conditioned on context).

    This captures the key architectural primitives:
      - Bidirectional MAE-style transformer (attends globally).
      - Diffusion loss head (per-token denoiser MLP) instead of softmax.
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_time: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.mask_token = nn.Parameter(torch.randn(d_model))

        # adaLN transformer: time condition = diffusion noise time
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbed(d_time),
            nn.Linear(d_time, 4 * d_time),
            nn.SiLU(),
            nn.Linear(4 * d_time, 4 * d_time),
        )
        d_cond = 4 * d_time
        self.blocks = nn.ModuleList([AdaLNBlock(d_model, n_heads, d_cond) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model)
        # Diffusion loss denoiser head
        self.diff_head = DiffusionMLPHead(d_model, d_time)

    def forward(
        self,
        x: torch.Tensor,  # (N, d_model) token embeddings (continuous)
        mask: torch.Tensor,  # (N,) bool: True = masked
        t: torch.Tensor,  # scalar, diffusion noise time
        z_t: torch.Tensor,  # (N, d_model) noisy targets (for masked positions)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns:
        z0_hat: (M, d_model) denoised embeddings at masked positions.
        x_out:  (N, d_model) transformer output (for unmasked positions).
        """
        N = x.size(0)
        # Replace masked tokens with learnable mask token
        x_in = x.clone()
        x_in[mask] = self.mask_token

        # Time condition (same t for all tokens in this pass)
        t_vec = t.unsqueeze(0)  # (1,)
        c = self.time_embed(t_vec).expand(N, -1)  # (N, d_cond)

        # Bidirectional transformer (no attn mask)
        h = x_in
        for blk in self.blocks:
            h = blk(h, c, attn_mask=None)
        h = self.norm_out(h)  # (N, d_model)

        # Apply diffusion-MLP head to masked positions
        if mask.any():
            cond_masked = h[mask]  # (M, d_model)
            z_t_masked = z_t[mask]  # (M, d_model)
            t_masked = t.unsqueeze(0).expand(cond_masked.size(0))
            z0_hat = self.diff_head(z_t_masked, t_masked, cond_masked)  # (M, d_model)
        else:
            z0_hat = h.new_zeros(0, self.d_model)

        return z0_hat, h


def build_mar() -> nn.Module:
    return MARModel(d_model=32, n_heads=4, n_layers=2, d_time=32)


def example_input_mar() -> list[torch.Tensor]:
    torch.manual_seed(9)
    N = 8
    x = torch.randn(N, 32)
    mask = torch.zeros(N, dtype=torch.bool)
    mask[2:6] = True  # 4 masked positions
    t = torch.tensor(0.5)
    z_t = torch.randn(N, 32)  # noisy targets
    return [x, mask, t, z_t]


# =============================================================================
# Registry
# =============================================================================

MENAGERIE_ENTRIES = [
    (
        "BD3-LM (Block Diffusion Language Model)",
        "build_block_diffusion",
        "example_input_block_diffusion",
        "2024",
        "DC",
    ),
    (
        "MAR (Masked AR with Diffusion Loss)",
        "build_mar",
        "example_input_mar",
        "2024",
        "DC",
    ),
]
