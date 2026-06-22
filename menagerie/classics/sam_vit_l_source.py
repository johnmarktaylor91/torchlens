"""Segment Anything Model (SAM) ViT-L source-style compact reconstruction.

Kirillov et al. 2023, "Segment Anything".

The faithful structure is the three-part SAM design: a ViT image encoder with
window/global self-attention blocks, prompt encodings for sparse point prompts,
and a lightweight two-way mask decoder that cross-attends between prompt tokens
and image tokens before producing masks. This random-init version keeps the same
data flow while shrinking dimensions and image size for base-environment tracing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MLP(nn.Module):
    """Two-layer feed-forward block used by compact transformer blocks."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """Initialize the MLP.

        Parameters
        ----------
        dim:
            Input and output feature dimension.
        hidden_dim:
            Hidden expansion dimension.
        """
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply GELU feed-forward projection.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        Tensor
            Projected tensor with the same trailing dimension.
        """
        return self.fc2(F.gelu(self.fc1(x)))


class ViTBlock(nn.Module):
    """Compact SAM ViT block with optional local-window attention."""

    def __init__(self, dim: int, heads: int, window_size: int | None) -> None:
        """Initialize the attention and MLP sublayers.

        Parameters
        ----------
        dim:
            Token feature dimension.
        heads:
            Number of attention heads.
        window_size:
            Local window size, or ``None`` for global attention.
        """
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4)

    def _window_attention(self, x: Tensor, height: int, width: int) -> Tensor:
        """Apply attention independently inside non-overlapping windows.

        Parameters
        ----------
        x:
            Flattened image tokens with shape ``(batch, height * width, dim)``.
        height:
            Token-grid height.
        width:
            Token-grid width.

        Returns
        -------
        Tensor
            Locally attended tokens.
        """
        assert self.window_size is not None
        batch, _, dim = x.shape
        win = self.window_size
        grid = x.reshape(batch, height // win, win, width // win, win, dim)
        windows = grid.permute(0, 1, 3, 2, 4, 5).reshape(-1, win * win, dim)
        attended, _ = self.attn(windows, windows, windows, need_weights=False)
        grid = attended.reshape(batch, height // win, width // win, win, win, dim)
        return grid.permute(0, 1, 3, 2, 4, 5).reshape(batch, height * width, dim)

    def forward(self, x: Tensor, height: int, width: int) -> Tensor:
        """Apply one SAM image-encoder transformer block.

        Parameters
        ----------
        x:
            Flattened image tokens.
        height:
            Token-grid height.
        width:
            Token-grid width.

        Returns
        -------
        Tensor
            Updated image tokens.
        """
        residual = x
        y = self.norm1(x)
        if self.window_size is None:
            y, _ = self.attn(y, y, y, need_weights=False)
        else:
            y = self._window_attention(y, height, width)
        x = residual + y
        return x + self.mlp(self.norm2(x))


class CompactSAM(nn.Module):
    """Compact Segment Anything model with ViT encoder and mask decoder."""

    def __init__(self, dim: int = 64, heads: int = 4, patches: int = 8) -> None:
        """Initialize compact SAM components.

        Parameters
        ----------
        dim:
            Shared embedding dimension.
        heads:
            Number of attention heads.
        patches:
            Patch grid size for a 64x64 image with 8x8 patches.
        """
        super().__init__()
        self.dim = dim
        self.patches = patches
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=8, stride=8)
        self.pos_embed = nn.Parameter(torch.zeros(1, patches * patches, dim))
        self.blocks = nn.ModuleList(
            [
                ViTBlock(dim, heads, window_size=4),
                ViTBlock(dim, heads, window_size=4),
                ViTBlock(dim, heads, window_size=None),
                ViTBlock(dim, heads, window_size=4),
            ]
        )
        self.neck = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.point_embed = nn.Linear(3, dim)
        self.iou_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.prompt_to_image = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.image_to_prompt = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.hyper = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.upscale = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.iou_head = nn.Linear(dim, 1)

    def forward(self, image: Tensor, points: Tensor) -> Tensor:
        """Predict prompt-conditioned masks.

        Parameters
        ----------
        image:
            Image tensor with shape ``(batch, 3, 64, 64)``.
        points:
            Point prompts with normalized ``x, y, label`` columns.

        Returns
        -------
        Tensor
            Concatenated low-resolution mask logits and IoU score.
        """
        batch = image.shape[0]
        tokens = self.patch_embed(image).flatten(2).transpose(1, 2) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens, self.patches, self.patches)
        image_tokens = self.neck(tokens)
        prompt_tokens = torch.cat(
            (
                self.iou_token.expand(batch, -1, -1),
                self.mask_token.expand(batch, -1, -1),
                self.point_embed(points),
            ),
            dim=1,
        )
        prompt_tokens = (
            prompt_tokens
            + self.prompt_to_image(prompt_tokens, image_tokens, image_tokens, need_weights=False)[0]
        )
        image_tokens = (
            image_tokens
            + self.image_to_prompt(image_tokens, prompt_tokens, prompt_tokens, need_weights=False)[
                0
            ]
        )
        mask_weights = self.hyper(prompt_tokens[:, 1])
        feature_map = image_tokens.transpose(1, 2).reshape(
            batch, self.dim, self.patches, self.patches
        )
        feature_map = self.upscale(feature_map)
        mask = torch.einsum("bc,bchw->bhw", mask_weights, feature_map).unsqueeze(1)
        iou = (
            self.iou_head(prompt_tokens[:, 0])
            .view(batch, 1, 1, 1)
            .expand(-1, -1, 1, mask.shape[-1])
        )
        return torch.cat((mask, iou), dim=2)


def build() -> nn.Module:
    """Build a compact random-init SAM model.

    Returns
    -------
    nn.Module
        Compact SAM reconstruction.
    """
    return CompactSAM()


def example_input() -> tuple[Tensor, Tensor]:
    """Return a small image and two point prompts.

    Returns
    -------
    tuple[Tensor, Tensor]
        Image and point-prompt tensors.
    """
    image = torch.randn(1, 3, 64, 64)
    points = torch.tensor([[[0.25, 0.25, 1.0], [0.75, 0.60, 0.0]]])
    return image, points


MENAGERIE_ENTRIES = [
    ("SAM_ViT_L_Source", "build", "example_input", "2023", "E7"),
]
