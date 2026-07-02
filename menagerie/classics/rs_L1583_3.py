# FAITHFUL REIMPLEMENTATION from Torrado, Khalifa, Green, Justesen, Risi & Togelius,
# "Bootstrapping Conditional GANs for Video Game Level Generation" (arXiv:1910.01603,
# IEEE CoG 2020) -- no public code release exists (no repo linked from the paper, IEEE
# CoG page, NSF PAR record, or ITU Copenhagen publication page; author GitHub accounts
# for Torrado/Khalifa/Green/Justesen/Risi/Togelius and the modl.ai / OriGen.ai orgs were
# checked directly, none host a CESAGAN implementation).
#
# The paper's Methodology + Experiments sections specify the architecture concretely:
#   - CESAGAN = a Self-Attention GAN (SAGAN, Zhang et al. 2018) backbone -- self-attention
#     computed from 1x1-conv query/key/value projections, softmax attention map, learned
#     gamma-scaled residual -- used in BOTH the generator and discriminator.
#   - "The CESAGAN network uses 1x1 convolutions in the discriminator and 1x1 deconvolutions
#     in the generator. Additionally, we employ batchnorm both in the generator and
#     discriminator after each layer and ReLU activations."
#   - The conditional feature vector u (per-level sprite-type counts) is mapped through an
#     embedding network -- "a multilayer perceptron (MLP) was used as the embedding network"
#     -- to t(u) = W_t u, and "the output of the embedding mapping t(u) is concatenated with
#     the output of the attention layer o(i)" in both G and D (Figure 1).
#   - Tile levels are one-hot encoded per-tile (Table 1: 8 tile classes for the Zelda
#     evaluation domain used in the paper).
# This module reimplements exactly that structure (self-attention block with query/key/value
# 1x1 convs + softmax + gamma residual; 1x1 conv/deconv trunk with batchnorm+ReLU; MLP
# embedding of the conditional count vector u concatenated post-attention) for both G and D.
# Sizes (channel widths, level grid, u-dimension) are set small for a tiny recipe instance;
# the paper does not give an exhaustive per-layer channel table, so channel counts follow
# the standard SAGAN convention (doubling/halving per stage) referenced by the paper.
import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention2d(nn.Module):
    """SAGAN self-attention block (Zhang et al. 2018): query/key/value 1x1 convs,
    softmax attention map, gamma-scaled residual -- exactly the mechanism the CESAGAN
    paper describes (query f(x)=W_f x, key g(x)=W_g x, value/output v(.), h(.))."""

    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        q = self.query(x).view(b, -1, n).permute(0, 2, 1)  # (b, n, c//8)
        k = self.key(x).view(b, -1, n)  # (b, c//8, n)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)  # (b, n, n)
        v = self.value(x).view(b, c, n)  # (b, c, n)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)
        return self.gamma * out + x


class ConditionalEmbedding(nn.Module):
    """Embedding network t(u) = W_t u for the auxiliary conditional feature vector u
    (per-tile-type counts). The paper: "a multilayer perceptron (MLP) was used as the
    embedding network"."""

    def __init__(self, u_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(u_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.mlp(u)


class CESAGANGenerator(nn.Module):
    """CESAGAN generator: latent noise z -> 1x1-deconv trunk (batchnorm+ReLU each layer)
    with a self-attention block, conditioned by concatenating the embedded feature vector
    t(u) with the post-attention feature map, then a final 1x1 conv to per-tile logits over
    a fixed level grid (height x width x n_tiles), matching the paper's tile-based level
    representation."""

    def __init__(
        self,
        z_dim: int = 32,
        u_dim: int = 8,
        embed_dim: int = 16,
        base_channels: int = 32,
        n_tiles: int = 8,
        grid_h: int = 6,
        grid_w: int = 6,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.base_channels = base_channels

        self.project = nn.Linear(z_dim, base_channels * grid_h * grid_w)

        self.deconv1 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.attn = SelfAttention2d(base_channels)

        self.embed = ConditionalEmbedding(u_dim, embed_dim)

        self.deconv2 = nn.ConvTranspose2d(base_channels + embed_dim, base_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(base_channels)

        self.to_tiles = nn.Conv2d(base_channels, n_tiles, kernel_size=1)

    def forward(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x = self.project(z).view(-1, self.base_channels, self.grid_h, self.grid_w)

        x = F.relu(self.bn1(self.deconv1(x)))
        x = self.attn(x)

        t_u = self.embed(u)  # (b, embed_dim)
        t_u_map = t_u.view(t_u.shape[0], t_u.shape[1], 1, 1).expand(
            -1, -1, self.grid_h, self.grid_w
        )
        x = torch.cat([x, t_u_map], dim=1)

        x = F.relu(self.bn2(self.deconv2(x)))
        logits = self.to_tiles(x)
        return logits


class CESAGANDiscriminator(nn.Module):
    """CESAGAN discriminator: one-hot tile level (+ conditional u) -> 1x1-conv trunk
    (batchnorm+ReLU each layer) with a self-attention block, conditioned by concatenating
    the embedded feature vector t(u), then a linear real/fake score, following equation
    (4)/Figure 1 of the paper (D conditioned on u)."""

    def __init__(
        self,
        n_tiles: int = 8,
        u_dim: int = 8,
        embed_dim: int = 16,
        base_channels: int = 32,
        grid_h: int = 6,
        grid_w: int = 6,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

        self.conv1 = nn.Conv2d(n_tiles, base_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.attn = SelfAttention2d(base_channels)

        self.embed = ConditionalEmbedding(u_dim, embed_dim)

        self.conv2 = nn.Conv2d(base_channels + embed_dim, base_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(base_channels)

        self.score = nn.Linear(base_channels * grid_h * grid_w, 1)

    def forward(self, level: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(level)))
        x = self.attn(x)

        t_u = self.embed(u)
        t_u_map = t_u.view(t_u.shape[0], t_u.shape[1], 1, 1).expand(
            -1, -1, self.grid_h, self.grid_w
        )
        x = torch.cat([x, t_u_map], dim=1)

        x = F.relu(self.bn2(self.conv2(x)))
        x = x.flatten(1)
        return self.score(x)


MENAGERIE_ZOO = "reimpl-pytorch"


def build_cesagan_generator():
    return CESAGANGenerator(
        z_dim=32, u_dim=8, embed_dim=16, base_channels=32, n_tiles=8, grid_h=6, grid_w=6
    )


def example_input_cesagan_generator():
    z = torch.randn(4, 32)
    u = torch.rand(4, 8)
    return (z, u)


def build_cesagan_discriminator():
    return CESAGANDiscriminator(
        n_tiles=8, u_dim=8, embed_dim=16, base_channels=32, grid_h=6, grid_w=6
    )


def example_input_cesagan_discriminator():
    level = torch.rand(4, 8, 6, 6)
    u = torch.rand(4, 8)
    return (level, u)


MENAGERIE_ENTRIES = [
    (
        "cesagan_generator",
        "build_cesagan_generator",
        "example_input_cesagan_generator",
        2020,
        "reimpl-pytorch",
    ),
    (
        "cesagan_discriminator",
        "build_cesagan_discriminator",
        "example_input_cesagan_discriminator",
        2020,
        "reimpl-pytorch",
    ),
]
