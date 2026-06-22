"""TarFlow and DyDiT: autoregressive normalizing flow and dynamic diffusion transformer.

TarFlow:
  Zhai et al., "Normalizing Flows are Capable Generative Models."
  arXiv:2412.06329 (2024).
  Source: https://github.com/google-deepmind/tarflow
  (See also: Kingma & Dhariwal, Glow; Papamakarios et al. RealNVP)

DyDiT:
  Zheng et al., "DyDiT: Dynamic Diffusion Transformers." arXiv:2410.09957 (2024).
  Source: https://github.com/NUS-HPC-AI-Lab/DyDiT

------------------------------------------------------------------------------
TarFlow distinctive primitive:
  A normalizing flow (invertible, exact log-density) built from stacked
  transformer-AR affine coupling blocks:

  1. Patch the image: treat each patch as a token.
  2. Each flow step is an AFFINE COUPLING via a CAUSAL TRANSFORMER:
       - Split tokens x = [x_1:T] (or use channel split for patches).
       - A CAUSAL AUTOREGRESSIVE TRANSFORMER maps x_{<t} -> (log_s_t, b_t).
       - Transform: y_t = x_t * exp(log_s_t) + b_t    (invertible by construction).
  3. Stack K such transformer-AR flow blocks (alternating split direction).
  4. The distinctive primitive: each affine coupling's scale+shift is computed
     by a CAUSAL transformer (not a CNN).

Faithful-compact simplifications:
  - 4 patch tokens (2x2 patches of 4-dim "image").
  - 2 flow blocks, d_model=32, 4 heads.
  - Causal mask in transformer.
  - Scale+shift MLP head on transformer output.
  - Random init, CPU, forward-only.

------------------------------------------------------------------------------
DyDiT distinctive primitive:
  DiT (Diffusion Transformer) with DYNAMIC COMPUTATION:

  1. Timestep-wise Dynamic Width (TDW): at each timestep t, a subset of
     channels is ACTIVATED based on a learned t-dependent mask.
     The mask is a binary gate per channel, computed by:
       g(t) = straight-through binarize(sigmoid(W_t * time_emb))
     Inactive channels are zeroed (not computed in principle, here simulated).

  2. Spatial-wise Token Dynamic (SDT): a learned router decides to SKIP
     tokens (spatial positions) that carry low information. Tokens with
     routing score below a threshold are replaced by a bypass (no attention).
     This gives spatially-adaptive computation.

  3. Standard DiT block otherwise: self-attention + FFN + adaLN-Zero.

Faithful-compact simplifications:
  - 16 tokens (4x4 spatial), 4 channels.
  - 2 DyDiT blocks, d_model=32, 4 heads.
  - TDW: binary mask over d_model channels per timestep.
  - SDT: routing score per token; bottom k tokens bypassed.
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
        half = self.d // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t.unsqueeze(-1) * freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)


# =============================================================================
# TarFlow
# =============================================================================


class CausalTransformer(nn.Module):
    """Causal autoregressive transformer for TarFlow affine coupling."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, d_in: int) -> None:
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.pos_embed = nn.Embedding(256, d_model)
        self.blocks = nn.ModuleList([self._make_block(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2 * d_in)  # scale and shift

    def _make_block(self, d_model: int, n_heads: int) -> nn.Module:
        return _CausalBlock(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (T, d_in) -> log_s, b each (T, d_in)"""
        T = x.size(0)
        h = self.embed(x) + self.pos_embed(torch.arange(T, device=x.device))
        # Causal mask: (T, T), True = masked (future)
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        for blk in self.blocks:
            h = blk(h, causal_mask)
        h = self.norm(h)
        out = self.head(h)  # (T, 2*d_in)
        log_s, b = out.chunk(2, dim=-1)
        return log_s, b


class _CausalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        T, D = x.shape
        res = x
        x = self.norm1(x)
        qkv = self.qkv(x).view(T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(1)
        attn = torch.einsum("ihd,jhd->ijh", q, k) / math.sqrt(self.d_head)
        attn = attn.masked_fill(mask.unsqueeze(-1), float("-inf"))
        attn = F.softmax(attn, dim=1)
        out = torch.einsum("ijh,jhd->ihd", attn, v).reshape(T, D)
        x = res + self.proj(out)
        x = x + self.ffn(self.norm2(x))
        return x


class TarFlowBlock(nn.Module):
    """One TarFlow affine coupling block using a causal transformer."""

    def __init__(self, d_token: int, d_model: int, n_heads: int, n_layers: int) -> None:
        super().__init__()
        self.transformer = CausalTransformer(d_model, n_heads, n_layers, d_token)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (T, d_token) -> y: (T, d_token), log_det: scalar"""
        # Shift x by 1 to make it strictly causal (predict y_t from x_{<t})
        # Prepend zero token (represents "no past" for position 0)
        T, D = x.shape
        x_shifted = torch.cat([x.new_zeros(1, D), x[:-1]], dim=0)
        log_s, b = self.transformer(x_shifted)  # (T, d_token) each
        log_s = torch.tanh(log_s)  # keep scale bounded
        y = x * torch.exp(log_s) + b
        log_det = log_s.sum()
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Autoregressive inversion (sequential, for completeness)."""
        T, D = y.shape
        x = y.new_zeros(T, D)
        for t in range(T):
            x_prev = torch.cat([y.new_zeros(1, D), x[:t]], dim=0)
            log_s, b = self.transformer(x_prev)
            log_s = torch.tanh(log_s)
            x[t] = (y[t] - b[t]) * torch.exp(-log_s[t])
        return x


class TarFlow(nn.Module):
    """TarFlow: stacked transformer-AR affine coupling flow."""

    def __init__(
        self,
        n_patches: int = 4,
        d_patch: int = 4,
        d_model: int = 32,
        n_heads: int = 4,
        n_flow_layers: int = 2,
    ) -> None:
        super().__init__()
        self.flow_blocks = nn.ModuleList(
            [TarFlowBlock(d_patch, d_model, n_heads, 1) for _ in range(n_flow_layers)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (T, d_patch) -> y: (T, d_patch), total_log_det: scalar"""
        total_log_det = x.new_zeros(())
        for blk in self.flow_blocks:
            x, ld = blk(x)
            total_log_det = total_log_det + ld
        return x, total_log_det


def build_tarflow() -> nn.Module:
    return TarFlow(n_patches=4, d_patch=4, d_model=32, n_heads=4, n_flow_layers=2)


def example_input_tarflow() -> torch.Tensor:
    """4 patch tokens, each 4-dim."""
    torch.manual_seed(10)
    return torch.randn(4, 4)


# =============================================================================
# DyDiT
# =============================================================================


class TimestepWidthMask(nn.Module):
    """TDW: Timestep-wise Dynamic Width mask.

    Given time embedding, predicts a binary channel mask via straight-through
    sigmoid binarization.
    """

    def __init__(self, d_model: int, d_time: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_time, d_model),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        """t_emb: (d_time,) -> mask: (d_model,) binary {0,1}"""
        logits = self.gate(t_emb)  # (d_model,)
        # Straight-through binarization
        hard = (logits > 0).float()
        soft = torch.sigmoid(logits)
        return hard - soft.detach() + soft  # (d_model,) in {0,1} approx


class TokenRouter(nn.Module):
    """SDT: Spatial-wise Token Dynamic routing.

    Scores each token; bottom-k tokens are flagged as 'skip' (low information).
    Skipped tokens pass through a bypass (identity or simple projection).
    """

    def __init__(self, d_model: int, keep_ratio: float = 0.75) -> None:
        super().__init__()
        self.keep_ratio = keep_ratio
        self.score_net = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (T, d_model) -> scores (T,), keep_mask (T,) bool"""
        scores = self.score_net(x).squeeze(-1)  # (T,)
        T = x.size(0)
        k = max(1, int(T * self.keep_ratio))
        topk = torch.topk(scores, k, sorted=False)
        keep_mask = torch.zeros(T, dtype=torch.bool, device=x.device)
        keep_mask[topk.indices] = True
        return scores, keep_mask


class DyDiTBlock(nn.Module):
    """DyDiT block: DiT with TDW channel masking + SDT token routing."""

    def __init__(self, d_model: int, n_heads: int, d_time: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        # adaLN-Zero conditioning (standard DiT)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(d_time, 6 * d_model))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

        # TDW: channel mask from timestep
        self.tdw = TimestepWidthMask(d_model, d_time)
        # SDT: token router (keep top 75%)
        self.sdt = TokenRouter(d_model, keep_ratio=0.75)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """x: (T, d_model), t_emb: (d_time,) -> (T, d_model)"""
        T, D = x.shape

        # TDW: compute dynamic channel mask from timestep
        ch_mask = self.tdw(t_emb)  # (d_model,) binary

        # SDT: route tokens
        scores, keep_mask = self.sdt(x)  # keep_mask: (T,) bool

        # adaLN conditioning
        ada = self.adaln(t_emb)  # (6*d_model,)
        alpha_a, beta_a, gamma_a, alpha_f, beta_f, gamma_f = ada.chunk(6, dim=-1)

        # Process kept tokens with full attention
        # Skipped tokens: identity (bypass)
        x_kept = x[keep_mask]  # (K, d_model)
        x_skip = x[~keep_mask]  # (S, d_model)

        T_k = x_kept.size(0)

        # Attention on kept tokens
        xn = self.norm1(x_kept) * (1 + alpha_a) + beta_a
        qkv = self.qkv(xn).view(T_k, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(1)
        attn = torch.einsum("ihd,jhd->ijh", q, k) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=1)
        out = torch.einsum("ijh,jhd->ihd", attn, v).reshape(T_k, D)
        out = self.proj(out)
        # Apply TDW channel mask to attention output
        out = out * ch_mask
        x_kept = x_kept + gamma_a * out

        # FFN on kept tokens
        xn = self.norm2(x_kept) * (1 + alpha_f) + beta_f
        ffn_out = self.ffn(xn) * ch_mask
        x_kept = x_kept + gamma_f * ffn_out

        # Recombine
        x_out = x.new_zeros(T, D)
        x_out[keep_mask] = x_kept
        x_out[~keep_mask] = x_skip  # bypass skipped tokens
        return x_out


class DyDiT(nn.Module):
    """Dynamic Diffusion Transformer (DyDiT).

    Standard DiT for image generation, augmented with TDW channel masking
    and SDT token routing for dynamic computation.
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_time: int = 32,
        in_channels: int = 4,
        n_tokens: int = 16,
    ) -> None:
        super().__init__()
        self.tok_embed = nn.Linear(in_channels, d_model)
        self.pos_embed = nn.Embedding(n_tokens, d_model)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbed(d_time),
            nn.Linear(d_time, 4 * d_time),
            nn.SiLU(),
            nn.Linear(4 * d_time, d_time),
        )
        self.blocks = nn.ModuleList([DyDiTBlock(d_model, n_heads, d_time) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, in_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: (T, in_ch) noisy tokens,  t: scalar time  ->  (T, in_ch) denoised"""
        T = x.size(0)
        h = self.tok_embed(x) + self.pos_embed(torch.arange(T, device=x.device))
        t_emb = self.time_embed(t.unsqueeze(0)).squeeze(0)  # (d_time,)
        for blk in self.blocks:
            h = blk(h, t_emb)
        h = self.norm(h)
        return self.head(h)


def build_dydit() -> nn.Module:
    return DyDiT(d_model=32, n_heads=4, n_layers=2, d_time=32, in_channels=4, n_tokens=16)


def example_input_dydit() -> list[torch.Tensor]:
    torch.manual_seed(11)
    x = torch.randn(16, 4)
    t = torch.tensor(0.3)
    return [x, t]


# =============================================================================
# Registry
# =============================================================================

MENAGERIE_ENTRIES = [
    (
        "TarFlow (Transformer AR Normalizing Flow)",
        "build_tarflow",
        "example_input_tarflow",
        "2024",
        "DC",
    ),
    (
        "DyDiT (Dynamic Diffusion Transformer)",
        "build_dydit",
        "example_input_dydit",
        "2024",
        "DC",
    ),
]
