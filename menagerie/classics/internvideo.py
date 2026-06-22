"""InternVideo: video understanding foundation models (V1 and V2).

InternVideo1: Wang et al. (Shanghai AI Lab), 2022, arXiv:2212.03191.
  Source: https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo1
InternVideo2: Wang et al. (Shanghai AI Lab), 2024, arXiv:2403.15377.
  Source: https://github.com/OpenGVLab/InternVideo2

Both are spatiotemporal Vision Transformers (video ViTs) for video understanding.
Distinctive primitives:
  - Tubelet embedding: partition a video clip into 3D patches (T', H', W') via a
    3D (tubelet) conv; each patch becomes one token. This is the 3D analogue of
    ViT's patch embedding for static images.
  - Spatiotemporal positional embedding: separate time + space pos embeddings,
    added to the tubelet tokens.
  - Transformer blocks over the flattened spatiotemporal sequence.
  - CLS token for global representation.
  - V1 uses a standard MLP head; V2 adds an attention-pooling head (a single
    cross-attention where the CLS queries over all patch tokens).

Both variants share a common video ViT backbone; two builders cover the targets.
Compact config: 4 frames, 16x16 spatial, patch_size=8 (V1) / patch_size=8 (V2),
2 ViT blocks, 2 heads, d_model=64.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Tubelet embedding (3D patch embed)                                          #
# --------------------------------------------------------------------------- #


class TubeletEmbed(nn.Module):
    """3D conv patch embedding for video: (B, C, T, H, W) -> (B, N, d_model).

    tube_size: (t, h, w) -- spatial/temporal patch size.
    The Conv3d with stride=tube_size splits the video into non-overlapping tubelets.
    """

    def __init__(
        self,
        in_channels: int = 3,
        tube_size: tuple = (2, 8, 8),
        d_model: int = 64,
    ) -> None:
        super().__init__()
        self.tube_size = tube_size
        self.proj = nn.Conv3d(
            in_channels,
            d_model,
            kernel_size=tube_size,
            stride=tube_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        tokens = self.proj(x)  # (B, d_model, T', H', W')
        B, D, T2, H2, W2 = tokens.shape
        return tokens.flatten(2).transpose(1, 2)  # (B, T'*H'*W', d_model)


# --------------------------------------------------------------------------- #
#  Spatiotemporal positional embedding                                         #
# --------------------------------------------------------------------------- #


class SpatiotemporalPosEmbed(nn.Module):
    """Separate learnable temporal and spatial positional embeddings."""

    def __init__(self, n_time: int, n_space: int, d_model: int) -> None:
        super().__init__()
        self.time_embed = nn.Embedding(n_time, d_model)
        self.space_embed = nn.Embedding(n_space, d_model)
        self.n_time = n_time
        self.n_space = n_space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_time*n_space, d_model)
        B, N, D = x.shape
        t_idx = torch.arange(self.n_time, device=x.device).repeat_interleave(self.n_space)
        s_idx = torch.arange(self.n_space, device=x.device).repeat(self.n_time)
        pos = self.time_embed(t_idx) + self.space_embed(s_idx)  # (N, D)
        return x + pos.unsqueeze(0)


# --------------------------------------------------------------------------- #
#  Video ViT backbone (shared by V1 and V2)                                   #
# --------------------------------------------------------------------------- #


class VideoViT(nn.Module):
    """Spatiotemporal ViT backbone for video."""

    def __init__(
        self,
        in_channels: int = 3,
        tube_size: tuple = (2, 8, 8),
        n_frames: int = 4,
        img_size: int = 16,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        t_patch = n_frames // tube_size[0]
        h_patch = img_size // tube_size[1]
        w_patch = img_size // tube_size[2]
        self.n_patches = t_patch * h_patch * w_patch
        self.d_model = d_model

        self.patch_embed = TubeletEmbed(in_channels, tube_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = SpatiotemporalPosEmbed(t_patch, h_patch * w_patch, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple:
        # x: (B, C, T, H, W)
        B = x.shape[0]
        tokens = self.patch_embed(x)  # (B, N, d_model)
        tokens = self.pos_embed(tokens)  # add ST pos embed
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        seq = torch.cat([cls, tokens], dim=1)  # (B, 1+N, d_model)
        out = self.transformer(seq)  # (B, 1+N, d_model)
        out = self.norm(out)
        cls_out = out[:, 0]  # (B, d_model)
        patch_out = out[:, 1:]  # (B, N, d_model)
        return cls_out, patch_out


# --------------------------------------------------------------------------- #
#  Attention Pooling head (V2 distinctive)                                     #
# --------------------------------------------------------------------------- #


class AttentionPoolingHead(nn.Module):
    """Cross-attention where a learnable query pools over patch tokens (V2 style)."""

    def __init__(self, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, cls_token: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        # cls_token: (B, d_model); patches: (B, N, d_model)
        B = cls_token.shape[0]
        q = self.query.expand(B, -1, -1)  # (B, 1, d_model)
        # Cross-attend: query over patches
        pooled, _ = self.cross_attn(q, patches, patches)  # (B, 1, d_model)
        pooled = self.norm(pooled.squeeze(1))  # (B, d_model)
        return self.head(pooled)  # (B, num_classes)


# --------------------------------------------------------------------------- #
#  InternVideo1 -- standard MLP head                                          #
# --------------------------------------------------------------------------- #


class InternVideo1(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = VideoViT(
            tube_size=(2, 8, 8), n_frames=4, img_size=16, d_model=64, nhead=2, num_layers=2
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls, _ = self.backbone(x)
        return self.head(cls)


# --------------------------------------------------------------------------- #
#  InternVideo2 -- attention pooling head, patch14-style (patch_size=8 here) #
# --------------------------------------------------------------------------- #


class InternVideo2(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # V2 uses patch14 (14px patches); here 8px for 16x16 input
        self.backbone = VideoViT(
            tube_size=(2, 8, 8), n_frames=4, img_size=16, d_model=64, nhead=2, num_layers=2
        )
        self.attn_pool_head = AttentionPoolingHead(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls, patches = self.backbone(x)
        return self.attn_pool_head(cls, patches)


# --------------------------------------------------------------------------- #
#  Wrappers & menagerie interface                                              #
# --------------------------------------------------------------------------- #


class _V1Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = InternVideo1(num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _V2Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = InternVideo2(num_classes=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_internvideo1() -> nn.Module:
    return _V1Wrapper()


def build_internvideo2() -> nn.Module:
    return _V2Wrapper()


def example_input() -> torch.Tensor:
    """4 frames of 16x16 RGB -- (B, C, T, H, W)."""
    torch.manual_seed(0)
    return torch.randn(1, 3, 4, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "InternVideo1 (spatiotemporal ViT with tubelet embedding, MLP head)",
        "build_internvideo1",
        "example_input",
        "2022",
        "DC",
    ),
    (
        "InternVideo2 (spatiotemporal ViT with tubelet embedding, attention-pooling head)",
        "build_internvideo2",
        "example_input",
        "2024",
        "DC",
    ),
]
