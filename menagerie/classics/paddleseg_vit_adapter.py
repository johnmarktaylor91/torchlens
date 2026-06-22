"""PaddleSeg ViT-Adapter dense-prediction backbone.

Chen et al. (ICLR 2023), "Vision Transformer Adapter for Dense Predictions."
ViT-Adapter augments a plain ViT with a convolutional spatial-prior module and
interaction blocks that inject local CNN priors into tokens and extract
multi-scale dense features back from the transformer stream.  This compact
model keeps patch tokens, a spatial-prior pyramid, injector/extractor
cross-attention, and a dense segmentation head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPrior(nn.Module):
    """Convolutional spatial-prior pyramid for ViT-Adapter."""

    def __init__(self, width: int) -> None:
        """Initialize the spatial-prior convolutions.

        Parameters
        ----------
        width:
            Feature width.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
        )
        self.down1 = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(width, width, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return multi-scale CNN prior features.

        Parameters
        ----------
        x:
            Input image.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Feature maps at strides two, four, and eight.
        """

        s2 = self.stem(x)
        s4 = F.gelu(self.down1(s2))
        s8 = F.gelu(self.down2(s4))
        return s2, s4, s8


class InteractionBlock(nn.Module):
    """Injector/extractor interaction between ViT tokens and spatial priors."""

    def __init__(self, width: int, heads: int = 4) -> None:
        """Initialize the interaction block.

        Parameters
        ----------
        width:
            Token and prior width.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.inject = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )
        self.extract = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(
        self, tokens: torch.Tensor, prior: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exchange information between tokens and flattened prior features.

        Parameters
        ----------
        tokens:
            Transformer tokens.
        prior:
            Flattened CNN prior sequence.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated tokens and prior sequence.
        """

        injected, _ = self.inject(tokens, prior, prior, need_weights=False)
        tokens = tokens + injected
        tokens = tokens + self.ffn(tokens)
        extracted, _ = self.extract(prior, tokens, tokens, need_weights=False)
        prior = self.norm(prior + extracted)
        return tokens, prior


class CompactViTAdapter(nn.Module):
    """Small ViT-Adapter segmentation model."""

    def __init__(self, classes: int = 6, width: int = 32, patch: int = 8) -> None:
        """Initialize the compact ViT-Adapter.

        Parameters
        ----------
        classes:
            Number of segmentation classes.
        width:
            Embedding width.
        patch:
            Patch size for ViT tokens.
        """

        super().__init__()
        self.patch = patch
        self.spatial = SpatialPrior(width)
        self.patch_embed = nn.Conv2d(3, width, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, 64, width))
        self.blocks = nn.ModuleList([InteractionBlock(width), InteractionBlock(width)])
        self.head = nn.Sequential(
            nn.Conv2d(width * 3, width, 1),
            nn.GELU(),
            nn.Conv2d(width, classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict dense logits with adapted ViT features.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Semantic logits at input resolution.
        """

        s2, s4, s8 = self.spatial(x)
        patch_map = self.patch_embed(x)
        bsz, channels, height, width = patch_map.shape
        tokens = patch_map.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos[:, : tokens.shape[1]]
        prior = s8.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens, prior = block(tokens, prior)
        token_map = tokens.transpose(1, 2).reshape(bsz, channels, height, width)
        prior_map = prior.transpose(1, 2).reshape_as(s8)
        fused = torch.cat(
            [
                F.interpolate(token_map, size=s4.shape[2:], mode="bilinear"),
                F.interpolate(prior_map, size=s4.shape[2:], mode="bilinear"),
                s4,
            ],
            dim=1,
        )
        return F.interpolate(self.head(fused), size=x.shape[2:], mode="bilinear")


def build() -> nn.Module:
    """Build the compact PaddleSeg ViT-Adapter.

    Returns
    -------
    nn.Module
        Random-init ViT-Adapter in evaluation mode.
    """

    return CompactViTAdapter().eval()


def example_input() -> torch.Tensor:
    """Return a small image for tracing.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("paddleseg_vit_adapter", "build", "example_input", "2023", "DC"),
]
