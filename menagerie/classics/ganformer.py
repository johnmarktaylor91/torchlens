"""GANformer: Generative Adversarial Transformers.

Hudson & Finn (Stanford) 2021.  arXiv:2103.01209.
Source: https://github.com/dorarad/gansformer

GANformer's distinctive primitive: **duplex attention** (bipartite transformer)
between a set of K global latent tokens and a 2D feature grid (H*W pixels).
  - **Simplex attention** (latents -> image): latent tokens attend over image grid
    features as keys/values, producing updated latent summaries.
  - **Duplex attention** (image -> latents): image grid positions attend over the
    latent tokens as keys/values, receiving global structure.
  - Combined, this creates a factored bipartite attention graph between K latents
    and H*W image positions, replacing the dense self-attention O((H*W)^2) cost
    with O(K*(H*W)) cross-attention, where K << H*W.

Both generator (synthesis network) and discriminator use the duplex attention block.
Here we reproduce:
  - Generator: noise z -> MLP -> K latent tokens; 4x4 const feature map; two rounds
    of duplex attention layers; upsample conv blocks to 16x16; RGB head.
  - Discriminator: image -> conv features; duplex attention pooling via K queries ->
    classifier.

Random init, CPU, compact spatial/channel for clean tracing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuplexAttention(nn.Module):
    """Bipartite duplex attention between K latent tokens and H*W grid positions.

    Step 1 (simplex, latents attend grid):
        latents = softmax(latents @ grid.T / sqrt(d)) @ grid
    Step 2 (duplex, grid attends latents):
        grid = softmax(grid @ latents.T / sqrt(d)) @ latents
    Both use multi-head attention (compact: 2 heads here).
    """

    def __init__(self, d_model: int, n_heads: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        hd = d_model // n_heads
        self.hd = hd
        # Latent-to-grid (simplex)
        self.lat_q = nn.Linear(d_model, d_model, bias=False)
        self.grid_k = nn.Linear(d_model, d_model, bias=False)
        self.grid_v = nn.Linear(d_model, d_model, bias=False)
        self.lat_out = nn.Linear(d_model, d_model, bias=False)
        # Grid-to-latent (duplex)
        self.grd_q = nn.Linear(d_model, d_model, bias=False)
        self.lat_k = nn.Linear(d_model, d_model, bias=False)
        self.lat_v = nn.Linear(d_model, d_model, bias=False)
        self.grd_out = nn.Linear(d_model, d_model, bias=False)
        # Norms
        self.lat_norm = nn.LayerNorm(d_model)
        self.grd_norm = nn.LayerNorm(d_model)

    def _mha(
        self,
        q: torch.Tensor,  # (B, Nq, d)
        k: torch.Tensor,  # (B, Nk, d)
        v: torch.Tensor,  # (B, Nk, d)
        proj_q: nn.Module,
        proj_k: nn.Module,
        proj_v: nn.Module,
        out_proj: nn.Module,
    ) -> torch.Tensor:
        B, Nq, _ = q.shape
        Nk = k.shape[1]
        H, hd = self.n_heads, self.hd
        Q = proj_q(q).view(B, Nq, H, hd).transpose(1, 2)
        K = proj_k(k).view(B, Nk, H, hd).transpose(1, 2)
        V = proj_v(v).view(B, Nk, H, hd).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (hd**0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, Nq, hd)
        out = out.transpose(1, 2).reshape(B, Nq, -1)
        return out_proj(out)

    def forward(
        self, latents: torch.Tensor, grid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # latents: (B, K, d),  grid: (B, N, d)
        # -- simplex: latents attend grid --
        delta_lat = self._mha(
            latents, grid, grid, self.lat_q, self.grid_k, self.grid_v, self.lat_out
        )
        latents = self.lat_norm(latents + delta_lat)
        # -- duplex: grid attends latents --
        delta_grd = self._mha(
            grid, latents, latents, self.grd_q, self.lat_k, self.lat_v, self.grd_out
        )
        grid = self.grd_norm(grid + delta_grd)
        return latents, grid


# ---------------------------------------------------------------------------
# GANformer Generator
# ---------------------------------------------------------------------------


class GANformerGenerator(nn.Module):
    """GANformer generator: noise -> K latents + const feature grid; duplex attention; upsample."""

    def __init__(
        self,
        z_dim: int = 32,
        n_latents: int = 4,
        d_model: int = 32,
        nf: int = 16,
    ) -> None:
        super().__init__()
        self.n_latents = n_latents
        self.d_model = d_model
        # Map z -> K latent tokens
        self.latent_map = nn.Sequential(
            nn.Linear(z_dim, d_model * n_latents),
            nn.LeakyReLU(0.2),
        )
        # Const 4x4 starting feature grid
        self.const = nn.Parameter(torch.randn(1, d_model, 4, 4))
        self.grid_proj = nn.Linear(d_model, d_model)
        # Duplex attention layers
        self.attn1 = DuplexAttention(d_model)
        self.attn2 = DuplexAttention(d_model)
        # Upsample: 4->8->16
        self.up1 = nn.Sequential(
            nn.Conv2d(d_model, nf * 2, 3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(nf * 2, nf, 3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.to_rgb = nn.Conv2d(nf, 3, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        # Build K latent tokens: (B, K, d)
        lats = self.latent_map(z).view(B, self.n_latents, self.d_model)
        # Build 4x4 feature grid: (B, d, 4, 4) -> (B, 16, d) flat grid tokens
        feat = self.const.expand(B, -1, -1, -1)  # (B, d, 4, 4)
        H, W = feat.shape[2], feat.shape[3]
        grid = feat.flatten(2).transpose(1, 2)  # (B, H*W, d)
        grid = self.grid_proj(grid)
        # Duplex attention
        lats, grid = self.attn1(lats, grid)
        lats, grid = self.attn2(lats, grid)
        # Reshape grid back to spatial, upsample
        feat = grid.transpose(1, 2).view(B, self.d_model, H, W)
        feat = F.interpolate(feat, scale_factor=2.0, mode="bilinear", align_corners=False)
        feat = self.up1(feat)
        feat = F.interpolate(feat, scale_factor=2.0, mode="bilinear", align_corners=False)
        feat = self.up2(feat)
        return torch.tanh(self.to_rgb(feat))


def build_ganformer_generator() -> nn.Module:
    return GANformerGenerator()


def example_input_generator() -> torch.Tensor:
    return torch.randn(1, 32)


# ---------------------------------------------------------------------------
# GANformer Discriminator
# ---------------------------------------------------------------------------


class GANformerDiscriminator(nn.Module):
    """GANformer discriminator: image -> conv features; duplex attention via K queries -> logit."""

    def __init__(
        self,
        nc: int = 3,
        nf: int = 16,
        n_latents: int = 4,
        d_model: int = 32,
    ) -> None:
        super().__init__()
        self.n_latents = n_latents
        self.d_model = d_model
        # Conv feature extractor: 32x32 -> 8x8
        self.conv = nn.Sequential(
            nn.Conv2d(nc, nf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
        )
        self.feat_proj = nn.Linear(nf * 2, d_model)
        # K learnable query tokens (play role of latents in discriminator)
        self.queries = nn.Parameter(torch.randn(1, n_latents, d_model))
        self.attn = DuplexAttention(d_model)
        # Classifier
        self.head = nn.Linear(n_latents * d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        feat = self.conv(x)  # (B, nf*2, H, W)
        grid = feat.flatten(2).transpose(1, 2)  # (B, H*W, nf*2)
        grid = self.feat_proj(grid)  # (B, H*W, d)
        q = self.queries.expand(B, -1, -1)  # (B, K, d)
        q, grid = self.attn(q, grid)
        return self.head(q.reshape(B, -1))


def build_ganformer_discriminator() -> nn.Module:
    return GANformerDiscriminator()


def example_input_discriminator() -> torch.Tensor:
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "GANformer Generator (duplex/bipartite attention between latent tokens and image grid)",
        "build_ganformer_generator",
        "example_input_generator",
        "2021",
        "DC",
    ),
    (
        "GANformer Discriminator (duplex attention pooling: K query tokens over feature grid)",
        "build_ganformer_discriminator",
        "example_input_discriminator",
        "2021",
        "DC",
    ),
]
