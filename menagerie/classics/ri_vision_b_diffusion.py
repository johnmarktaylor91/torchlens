"""DiT -- Diffusion Transformer (Peebles & Xie, ICCV 2023, arXiv:2212.09748).

Source: github.com/facebookresearch/DiT.

DiT replaces the U-Net diffusion backbone with a pure TRANSFORMER operating on latent image
patches. Its DISTINCTIVE primitive is the adaLN-Zero conditioning block:

  1. PATCHIFY the (noised) latent into a sequence of patch tokens + add positional embeddings.
  2. Embed the diffusion TIMESTEP and CLASS LABEL into a conditioning vector c.
  3. Run N transformer blocks; in each block, instead of plain LayerNorm, the scale/shift
     (and a residual gate alpha) of the LayerNorm before attention and before the MLP are
     REGRESSED from c (adaptive LayerNorm). adaLN-Zero initializes the gates to zero so each
     block starts as identity -- the key trick that stabilizes training.
  4. A final adaLN layer + LINEAR head UNPATCHIFY back to noise/Sigma prediction.

dit and dit_xl_2 are the SAME architecture at different sizes (XL/2 = 28 layers, hidden 1152,
patch 2). One core, two entries (here both at a small atlas scale; the "XL/2" entry uses
slightly more depth than the base entry to reflect the size axis).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class _TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding -> MLP (the diffusion-step conditioning)."""

    def __init__(self, hidden: int, freq_dim: int = 64) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class _DiTBlock(nn.Module):
    """A DiT transformer block with adaLN-Zero conditioning (the distinctive primitive)."""

    def __init__(self, hidden: int, heads: int = 4, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, hidden)
        )
        # adaLN modulation: regress 6 params (shift/scale/gate x2) from conditioning c
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(
            6, dim=1
        )
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(h, h, h)
        x = x + gate_msa[:, None] * attn_out
        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None] * self.mlp(h)
        return x


class DiT(nn.Module):
    """Diffusion Transformer: patchify -> N adaLN-Zero blocks (cond. on t + class) -> unpatchify."""

    def __init__(
        self,
        in_ch: int = 4,
        img_size: int = 16,
        patch: int = 2,
        hidden: int = 64,
        depth: int = 4,
        heads: int = 4,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.in_ch = in_ch
        self.grid = img_size // patch
        n_patches = self.grid**2
        self.patch_embed = nn.Conv2d(in_ch, hidden, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, hidden))
        self.t_embed = _TimestepEmbedder(hidden)
        self.y_embed = nn.Embedding(num_classes, hidden)
        self.blocks = nn.ModuleList([_DiTBlock(hidden, heads) for _ in range(depth)])
        self.norm_final = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.adaLN_final = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.head = nn.Linear(hidden, patch * patch * in_ch * 2)  # predicts noise + Sigma

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        c = self.in_ch * 2
        x = x.view(b, self.grid, self.grid, self.patch, self.patch, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(
            b, c, self.grid * self.patch, self.grid * self.patch
        )
        return x

    def forward(
        self, x: torch.Tensor, t: torch.Tensor | None = None, y: torch.Tensor | None = None
    ) -> torch.Tensor:
        b = x.shape[0]
        if t is None:
            t = torch.zeros(b, dtype=torch.long, device=x.device)
        if y is None:
            y = torch.zeros(b, dtype=torch.long, device=x.device)
        h = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos  # patchify
        c = self.t_embed(t) + self.y_embed(y)  # timestep + class conditioning
        for blk in self.blocks:
            h = blk(h, c)
        shift, scale = self.adaLN_final(c).chunk(2, dim=1)
        h = _modulate(self.norm_final(h), shift, scale)
        h = self.head(h)  # (B, n_patches, p*p*2C)
        return self._unpatchify(h)


class _DiTWrapper(nn.Module):
    """DiT forwardable from a single latent tensor (default timestep=0, class=0)."""

    def __init__(self, depth: int = 4) -> None:
        super().__init__()
        self.dit = DiT(depth=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        t = torch.zeros(b, dtype=torch.long, device=x.device)
        y = torch.zeros(b, dtype=torch.long, device=x.device)
        return self.dit(x, t, y)


def build_dit() -> nn.Module:
    """DiT base: patchify -> adaLN-Zero transformer blocks -> unpatchify (compact: 4 blocks)."""
    return _DiTWrapper(depth=4).eval()


def build_dit_xl_2() -> nn.Module:
    """DiT-XL/2: same architecture, larger size axis (compact stand-in: 6 blocks)."""
    return _DiTWrapper(depth=6).eval()


def example_input() -> torch.Tensor:
    """Noised latent (1, 4, 16, 16) -- VAE latent grid for a 128px image at f=8."""
    return torch.randn(1, 4, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "DiT (Diffusion Transformer, adaLN-Zero conditioning)",
        "build_dit",
        "example_input",
        "2022",
        "DC",
    ),
    (
        "DiT-XL/2 (Diffusion Transformer, XL size, patch-2)",
        "build_dit_xl_2",
        "example_input",
        "2022",
        "DC",
    ),
]
