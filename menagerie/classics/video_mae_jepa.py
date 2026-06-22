"""VideoMAE V2, V-JEPA, and V-JEPA 2 compact classics.

VideoMAE V2: Wang et al., CVPR 2023, scales video masked autoencoders with a
dual-masking design over tube tokens.

V-JEPA: Bardes et al., ICLR 2024, predicts masked spatio-temporal regions in
learned latent space using context and target encoders plus a predictor.

V-JEPA 2: Meta AI, 2025, keeps the video JEPA encoder/predictor pretraining
stage and adds action-conditioned world-model post-training.  The variants here
share the faithful compact encoder and differ only by named builder width/input
size to supersede catalog variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block with explicit attention and MLP."""

    def __init__(self, dim: int, num_heads: int) -> None:
        """Initialize a transformer block.

        Parameters
        ----------
        dim:
            Token width.
        num_heads:
            Attention head count.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer attention and feed-forward updates.

        Parameters
        ----------
        x:
            Token tensor ``(B, N, C)``.

        Returns
        -------
        torch.Tensor
            Updated tokens.
        """

        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed)[0]
        return x + self.ff(self.norm2(x))


class _VideoTubeEmbed(nn.Module):
    """Tubelet patch embedding for video transformers."""

    def __init__(
        self, in_channels: int = 3, dim: int = 64, tubelet: int = 2, patch: int = 8
    ) -> None:
        """Initialize tubelet embedding.

        Parameters
        ----------
        in_channels:
            Video channel count.
        dim:
            Token width.
        tubelet:
            Temporal tubelet size.
        patch:
            Spatial patch size.
        """

        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            dim,
            kernel_size=(tubelet, patch, patch),
            stride=(tubelet, patch, patch),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Tokenize a video into tube tokens.

        Parameters
        ----------
        video:
            Video tensor ``(B, C, T, H, W)``.

        Returns
        -------
        torch.Tensor
            Tube tokens ``(B, N, C)``.
        """

        tokens = self.proj(video)
        return tokens.flatten(2).transpose(1, 2)


class VideoMAEV2Pretrain(nn.Module):
    """VideoMAE V2-style tube-masked video autoencoder."""

    def __init__(self, dim: int = 64, depth: int = 3, num_heads: int = 4) -> None:
        """Initialize compact VideoMAE V2 pretraining core.

        Parameters
        ----------
        dim:
            Token width.
        depth:
            Number of encoder and decoder blocks.
        num_heads:
            Attention head count.
        """

        super().__init__()
        self.patch = 8
        self.tubelet = 2
        self.embed = _VideoTubeEmbed(dim=dim, tubelet=self.tubelet, patch=self.patch)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.encoder = nn.ModuleList([_TransformerBlock(dim, num_heads) for _ in range(depth)])
        self.decoder = nn.ModuleList([_TransformerBlock(dim, num_heads) for _ in range(2)])
        self.reconstruct = nn.Linear(dim, 3 * self.tubelet * self.patch * self.patch)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Reconstruct masked video tubelets.

        Parameters
        ----------
        x:
            Tuple ``(video, mask)`` where mask is boolean ``(B, N)``.

        Returns
        -------
        torch.Tensor
            Reconstructed tubelet pixels for every token.
        """

        video, mask = x
        tokens = self.embed(video)
        visible = tokens.masked_fill(mask.unsqueeze(-1), 0.0)
        for block in self.encoder:
            visible = block(visible)
        decoded = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(visible), visible)
        for block in self.decoder:
            decoded = block(decoded)
        return self.reconstruct(decoded)


class VideoJEPA(nn.Module):
    """Video joint-embedding predictive architecture."""

    def __init__(
        self, dim: int = 64, depth: int = 3, num_heads: int = 4, action_conditioned: bool = False
    ) -> None:
        """Initialize a compact V-JEPA/V-JEPA2 model.

        Parameters
        ----------
        dim:
            Token width.
        depth:
            Number of context/target encoder blocks.
        num_heads:
            Attention head count.
        action_conditioned:
            Whether to include V-JEPA 2-style action-conditioned world prediction.
        """

        super().__init__()
        self.action_conditioned = action_conditioned
        self.embed = _VideoTubeEmbed(dim=dim, tubelet=2, patch=8)
        self.context_blocks = nn.ModuleList(
            [_TransformerBlock(dim, num_heads) for _ in range(depth)]
        )
        self.target_blocks = nn.ModuleList(
            [_TransformerBlock(dim, num_heads) for _ in range(depth)]
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.predictor = nn.ModuleList([_TransformerBlock(dim, num_heads) for _ in range(2)])
        self.action_proj = nn.Linear(4, dim)
        self.out = nn.Linear(dim, dim)

    def _encode(self, tokens: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        """Encode tokens with a transformer stack.

        Parameters
        ----------
        tokens:
            Input tokens.
        blocks:
            Transformer block stack.

        Returns
        -------
        torch.Tensor
            Encoded tokens.
        """

        for block in blocks:
            tokens = block(tokens)
        return tokens

    def forward(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Predict target latent regions from context tokens.

        Parameters
        ----------
        x:
            Tuple ``(video, target_mask, action)``.

        Returns
        -------
        torch.Tensor
            Predicted latent tokens for the target positions.
        """

        video, target_mask, action = x
        tokens = self.embed(video)
        context_tokens = tokens.masked_fill(target_mask.unsqueeze(-1), 0.0)
        context = self._encode(context_tokens, self.context_blocks)
        with torch.no_grad():
            target = self._encode(tokens, self.target_blocks)
        predicted = torch.where(
            target_mask.unsqueeze(-1), self.mask_token.expand_as(context), context
        )
        if self.action_conditioned:
            predicted = predicted + self.action_proj(action).unsqueeze(1)
        for block in self.predictor:
            predicted = block(predicted)
        return (
            self.out(predicted) * target_mask.unsqueeze(-1).to(predicted.dtype)
            + target.detach() * 0.0
        )


def _video_example(frame_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a small video, target mask, and action vector.

    Parameters
    ----------
    frame_size:
        Spatial video size.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Video, boolean target mask, and action vector.
    """

    video = torch.randn(1, 3, 4, frame_size, frame_size)
    token_count = (4 // 2) * (frame_size // 8) * (frame_size // 8)
    mask = torch.zeros(1, token_count, dtype=torch.bool)
    mask[:, token_count // 2 :] = True
    action = torch.randn(1, 4)
    return video, mask, action


def build_videomaev2_pretrain_base_patch16_224() -> nn.Module:
    """Build compact VideoMAE V2 pretraining model.

    Returns
    -------
    nn.Module
        Random-init VideoMAE V2 compact core.
    """

    return VideoMAEV2Pretrain(dim=64, depth=3, num_heads=4)


def example_input_videomae() -> tuple[torch.Tensor, torch.Tensor]:
    """Return VideoMAE V2 example video and mask.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Video tensor and boolean tube mask.
    """

    video, mask, _action = _video_example(32)
    return video, mask


def build_vjepa_vit_large() -> nn.Module:
    """Build compact V-JEPA ViT-Large-style model.

    Returns
    -------
    nn.Module
        Random-init V-JEPA compact core.
    """

    return VideoJEPA(dim=64, depth=3, num_heads=4)


def build_vjepa_vit_huge() -> nn.Module:
    """Build compact V-JEPA ViT-Huge-style model.

    Returns
    -------
    nn.Module
        Random-init wider V-JEPA compact core.
    """

    return VideoJEPA(dim=96, depth=3, num_heads=4)


def build_vjepa2_vith16_256() -> nn.Module:
    """Build compact V-JEPA 2 H/16 256 model.

    Returns
    -------
    nn.Module
        Random-init V-JEPA 2 compact core.
    """

    return VideoJEPA(dim=80, depth=3, num_heads=4, action_conditioned=True)


def build_vjepa2_vitl16_256() -> nn.Module:
    """Build compact V-JEPA 2 L/16 256 model.

    Returns
    -------
    nn.Module
        Random-init V-JEPA 2 compact core.
    """

    return VideoJEPA(dim=64, depth=3, num_heads=4, action_conditioned=True)


def build_vjepa2_vitg16_256() -> nn.Module:
    """Build compact V-JEPA 2 G/16 256 model.

    Returns
    -------
    nn.Module
        Random-init V-JEPA 2 compact core.
    """

    return VideoJEPA(dim=96, depth=4, num_heads=4, action_conditioned=True)


def build_vjepa2_vitg16_384() -> nn.Module:
    """Build compact V-JEPA 2 G/16 384 model.

    Returns
    -------
    nn.Module
        Random-init V-JEPA 2 compact core.
    """

    return VideoJEPA(dim=96, depth=4, num_heads=4, action_conditioned=True)


def example_input_vjepa_256() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return compact 256-family V-JEPA example input.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Video, target mask, and action vector.
    """

    return _video_example(32)


def example_input_vjepa_384() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return compact 384-family V-JEPA example input.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Video, target mask, and action vector.
    """

    return _video_example(48)


MENAGERIE_ENTRIES = [
    (
        "VideoMAE V2 Pretrain Base Patch16 224 (dual-masked video MAE)",
        "build_videomaev2_pretrain_base_patch16_224",
        "example_input_videomae",
        "2023",
        "DC",
    ),
    (
        "V-JEPA ViT-Large (latent video joint-embedding predictor)",
        "build_vjepa_vit_large",
        "example_input_vjepa_256",
        "2024",
        "DC",
    ),
    (
        "V-JEPA ViT-Huge (latent video joint-embedding predictor)",
        "build_vjepa_vit_huge",
        "example_input_vjepa_256",
        "2024",
        "DC",
    ),
    (
        "V-JEPA2 ViT-H/16 256 (action-conditioned latent video predictor)",
        "build_vjepa2_vith16_256",
        "example_input_vjepa_256",
        "2025",
        "DC",
    ),
    (
        "V-JEPA2 ViT-L/16 256 (action-conditioned latent video predictor)",
        "build_vjepa2_vitl16_256",
        "example_input_vjepa_256",
        "2025",
        "DC",
    ),
    (
        "V-JEPA2 ViT-G/16 256 (action-conditioned latent video predictor)",
        "build_vjepa2_vitg16_256",
        "example_input_vjepa_256",
        "2025",
        "DC",
    ),
    (
        "V-JEPA2 ViT-G/16 384 (action-conditioned latent video predictor)",
        "build_vjepa2_vitg16_384",
        "example_input_vjepa_384",
        "2025",
        "DC",
    ),
]
