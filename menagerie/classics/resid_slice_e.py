"""Faithful compact reimplementations for residual validation slice E.

The builders in this module replace dependency-heavy or over-sized catalog recipes with
small random-initialized PyTorch modules.  They preserve the load-bearing execution
structure of the source families while keeping TorchLens validation under the slice
memory cap.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Convolution, normalization, and activation block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dims: int = 2) -> None:
        """Initialize a convolutional block.

        Parameters
        ----------
        in_ch:
            Number of input channels.
        out_ch:
            Number of output channels.
        stride:
            Convolution stride.
        dims:
            Spatial dimensionality, either 1, 2, or 3.
        """

        super().__init__()
        if dims == 1:
            self.conv: nn.Module = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1)
            self.norm = nn.BatchNorm1d(out_ch)
        elif dims == 3:
            self.conv = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1)
            self.norm = nn.BatchNorm3d(out_ch)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
            self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        """Apply convolution, normalization, and GELU activation.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        Tensor
            Activated features.
        """

        return F.gelu(self.norm(self.conv(x)))


class ResidualConvBlock(nn.Module):
    """Residual convolutional block used by CNN-style families."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        """Initialize a residual bottleneck.

        Parameters
        ----------
        channels:
            Feature channel count.
        expansion:
            Hidden expansion ratio.
        """

        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm = nn.BatchNorm2d(channels)
        self.expand = nn.Conv2d(channels, hidden, 1)
        self.project = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Run a depthwise separable residual update.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Updated feature map.
        """

        residual = x
        x = F.gelu(self.norm(self.depthwise(x)))
        x = self.project(F.gelu(self.expand(x)))
        return residual + x


class TransformerBlock(nn.Module):
    """Pre-normalized transformer encoder block."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        mlp_ratio: int = 4,
        *,
        channel_attention: bool = False,
    ) -> None:
        """Initialize self-attention and feed-forward layers.

        Parameters
        ----------
        dim:
            Token embedding dimension.
        heads:
            Number of attention heads.
        mlp_ratio:
            Feed-forward expansion ratio.
        channel_attention:
            Whether to add a DaViT/XCiT-style channel mixing branch.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.channel_attention = channel_attention
        self.channel_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        """Apply attention and MLP residual branches.

        Parameters
        ----------
        x:
            Token tensor ``(batch, tokens, dim)``.

        Returns
        -------
        Tensor
            Updated token tensor.
        """

        attn_in = self.norm1(x)
        attn_out, _weights = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        if self.channel_attention:
            pooled = attn_in.mean(dim=1, keepdim=True)
            attn_out = attn_out * self.channel_gate(pooled)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class CompactVisionTransformer(nn.Module):
    """Compact ViT/BEiT/EVA/AIM-style image transformer."""

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        dim: int = 64,
        depth: int = 4,
        heads: int = 4,
        *,
        class_tokens: int = 1,
        register_tokens: int = 0,
        channel_attention: bool = False,
        use_mean_pool: bool = False,
    ) -> None:
        """Initialize patch embedding, tokens, and transformer blocks.

        Parameters
        ----------
        image_size:
            Example square image size.
        patch_size:
            Patch stride and kernel size.
        dim:
            Token embedding dimension.
        depth:
            Number of encoder blocks.
        heads:
            Number of attention heads.
        class_tokens:
            Number of learned class tokens.
        register_tokens:
            Number of DINO/PE-style register tokens.
        channel_attention:
            Whether blocks include a channel-attention branch.
        use_mean_pool:
            Whether to pool patch tokens instead of reading the class token.
        """

        super().__init__()
        grid = image_size // patch_size
        self.patch = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.zeros(1, class_tokens, dim))
        self.register = nn.Parameter(torch.zeros(1, register_tokens, dim))
        self.pos = nn.Parameter(
            torch.randn(1, class_tokens + register_tokens + grid * grid, dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim, heads, channel_attention=channel_attention)
                for _index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1000)
        self.use_mean_pool = use_mean_pool

    def forward(self, x: Tensor) -> Tensor:
        """Classify an image through patch-token self-attention.

        Parameters
        ----------
        x:
            Image tensor ``(batch, 3, height, width)``.

        Returns
        -------
        Tensor
            Class logits.
        """

        tokens = self.patch(x).flatten(2).transpose(1, 2)
        batch = tokens.shape[0]
        prefix = torch.cat(
            [self.cls.expand(batch, -1, -1), self.register.expand(batch, -1, -1)], dim=1
        )
        tokens = (
            torch.cat([prefix, tokens], dim=1) + self.pos[:, : prefix.shape[1] + tokens.shape[1]]
        )
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        pooled = tokens[:, prefix.shape[1] :].mean(dim=1) if self.use_mean_pool else tokens[:, 0]
        return self.head(pooled)


class ClassAttentionBlock(nn.Module):
    """CaiT class-attention block that updates class tokens from patches."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize class-attention layers.

        Parameters
        ----------
        dim:
            Token embedding dimension.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, cls: Tensor, patches: Tensor) -> Tensor:
        """Update the class token using patch tokens as keys and values.

        Parameters
        ----------
        cls:
            Class token tensor.
        patches:
            Patch token tensor.

        Returns
        -------
        Tensor
            Updated class token.
        """

        source = self.norm(torch.cat([cls, patches], dim=1))
        out, _weights = self.attn(source[:, :1], source, source, need_weights=False)
        cls = cls + out
        return cls + self.mlp(cls)


class CompactCaiT(nn.Module):
    """Class-attention image transformer."""

    def __init__(self, image_size: int = 64, dim: int = 64, depth: int = 4) -> None:
        """Initialize CaiT patch and class-attention stages.

        Parameters
        ----------
        image_size:
            Input image size.
        dim:
            Token dimension.
        depth:
            Number of patch self-attention blocks.
        """

        super().__init__()
        self.patch = nn.Conv2d(3, dim, 8, stride=8)
        tokens = (image_size // 8) ** 2
        self.pos = nn.Parameter(torch.randn(1, tokens, dim) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.patch_blocks = nn.ModuleList([TransformerBlock(dim, 4) for _index in range(depth)])
        self.class_blocks = nn.ModuleList([ClassAttentionBlock(dim, 4) for _index in range(2)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1000)

    def forward(self, x: Tensor) -> Tensor:
        """Classify an image with CaiT-style class attention.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        Tensor
            Class logits.
        """

        patches = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        for block in self.patch_blocks:
            patches = block(patches)
        cls = self.cls.expand(x.shape[0], -1, -1)
        for block in self.class_blocks:
            cls = block(cls, patches)
        return self.head(self.norm(cls[:, 0]))


class PatchMerging2D(nn.Module):
    """Swin-style patch merging layer."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Initialize a 2D patch merger.

        Parameters
        ----------
        in_ch:
            Input channel count.
        out_ch:
            Output channel count.
        """

        super().__init__()
        self.norm = nn.LayerNorm(in_ch * 4)
        self.reduction = nn.Linear(in_ch * 4, out_ch)

    def forward(self, x: Tensor) -> Tensor:
        """Merge neighboring 2x2 patches.

        Parameters
        ----------
        x:
            Feature map ``(batch, channels, height, width)``.

        Returns
        -------
        Tensor
            Downsampled feature map.
        """

        bsz, channels, height, width = x.shape
        x = x[:, :, : height - height % 2, : width - width % 2]
        parts = [x[:, :, y::2, z::2] for y in range(2) for z in range(2)]
        merged = torch.cat(parts, dim=1).permute(0, 2, 3, 1).reshape(bsz, -1, channels * 4)
        merged = self.reduction(self.norm(merged))
        size = int(math.sqrt(merged.shape[1]))
        return merged.transpose(1, 2).reshape(bsz, -1, size, size)


class CompactSwin2D(nn.Module):
    """Hierarchical shifted-window transformer."""

    def __init__(
        self, image_size: int = 64, dim: int = 32, depths: Sequence[int] = (2, 2, 2)
    ) -> None:
        """Initialize Swin-like stages.

        Parameters
        ----------
        image_size:
            Example image size.
        dim:
            Stem embedding dimension.
        depths:
            Blocks per hierarchy stage.
        """

        super().__init__()
        del image_size
        self.patch = nn.Conv2d(3, dim, 4, stride=4)
        self.stages = nn.ModuleList()
        channels = dim
        for stage_index, depth in enumerate(depths):
            blocks = nn.ModuleList([TransformerBlock(channels, 4) for _index in range(depth)])
            merge = (
                PatchMerging2D(channels, channels * 2)
                if stage_index < len(depths) - 1
                else nn.Identity()
            )
            self.stages.append(nn.ModuleDict({"blocks": blocks, "merge": merge}))
            channels *= 2 if stage_index < len(depths) - 1 else 1
        self.norm = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, 1000)

    def forward(self, x: Tensor) -> Tensor:
        """Classify an image with hierarchical local attention stages.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        Tensor
            Class logits.
        """

        feat = self.patch(x)
        for stage in self.stages:
            bsz, channels, height, width = feat.shape
            tokens = feat.flatten(2).transpose(1, 2)
            for block in stage["blocks"]:
                tokens = block(tokens)
            feat = tokens.transpose(1, 2).reshape(bsz, channels, height, width)
            feat = stage["merge"](feat)
        pooled = feat.mean(dim=(-2, -1))
        return self.head(self.norm(pooled))


class PatchMerging3D(nn.Module):
    """Video Swin patch merging over temporal-spatial neighborhoods."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Initialize a 3D patch merger.

        Parameters
        ----------
        in_ch:
            Input channels.
        out_ch:
            Output channels.
        """

        super().__init__()
        self.proj = nn.Conv3d(in_ch, out_ch, 2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        """Downsample a video feature map.

        Parameters
        ----------
        x:
            Video feature map.

        Returns
        -------
        Tensor
            Downsampled video features.
        """

        return self.proj(x)


class CompactVideoSwin(nn.Module):
    """Compact 3D Swin-style video classifier."""

    def __init__(self, dim: int = 24, depths: Sequence[int] = (1, 1, 1)) -> None:
        """Initialize 3D patch embedding and hierarchical attention.

        Parameters
        ----------
        dim:
            Stem channels.
        depths:
            Number of blocks per stage.
        """

        super().__init__()
        self.patch = nn.Conv3d(3, dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.stages = nn.ModuleList()
        channels = dim
        for stage_index, depth in enumerate(depths):
            blocks = nn.ModuleList([TransformerBlock(channels, 4) for _index in range(depth)])
            merge = (
                PatchMerging3D(channels, channels * 2)
                if stage_index < len(depths) - 1
                else nn.Identity()
            )
            self.stages.append(nn.ModuleDict({"blocks": blocks, "merge": merge}))
            channels *= 2 if stage_index < len(depths) - 1 else 1
        self.norm = nn.LayerNorm(channels)
        self.head = nn.Linear(channels, 400)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a video tensor.

        Parameters
        ----------
        x:
            Video tensor ``(batch, 3, time, height, width)``.

        Returns
        -------
        Tensor
            Class logits.
        """

        feat = self.patch(x)
        for stage in self.stages:
            bsz, channels, time, height, width = feat.shape
            tokens = feat.flatten(2).transpose(1, 2)
            for block in stage["blocks"]:
                tokens = block(tokens)
            feat = tokens.transpose(1, 2).reshape(bsz, channels, time, height, width)
            feat = stage["merge"](feat)
        pooled = feat.mean(dim=(-3, -2, -1))
        return self.head(self.norm(pooled))


class CompactEfficientNet(nn.Module):
    """EfficientNet-like MBConv classifier."""

    def __init__(self, width: int = 32, classes: int = 1000) -> None:
        """Initialize the MBConv hierarchy.

        Parameters
        ----------
        width:
            Base channel width.
        classes:
            Number of classifier outputs.
        """

        super().__init__()
        self.stem = ConvBNAct(3, width, stride=2)
        self.blocks = nn.Sequential(
            ResidualConvBlock(width),
            ConvBNAct(width, width * 2, stride=2),
            ResidualConvBlock(width * 2),
            ConvBNAct(width * 2, width * 4, stride=2),
            ResidualConvBlock(width * 4),
        )
        self.squeeze = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(width * 4, classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify an image.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        Tensor
            Class logits.
        """

        return self.head(self.squeeze(self.blocks(self.stem(x))))


class CompactNFNet(nn.Module):
    """NFNet-style normalization-free residual CNN."""

    def __init__(self, width: int = 32) -> None:
        """Initialize scaled residual stages.

        Parameters
        ----------
        width:
            Base channel width.
        """

        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, stride=2, padding=1), nn.GELU())
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, width, 1)
                )
                for _index in range(4)
            ]
        )
        self.down = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.head = nn.Linear(width * 2, 1000)

    def forward(self, x: Tensor) -> Tensor:
        """Run scaled residual NFNet inference.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        Tensor
            Class logits.
        """

        x = self.stem(x)
        alpha = 0.2
        for block in self.blocks:
            x = x + alpha * block(x)
        x = F.gelu(self.down(x))
        return self.head(x.mean(dim=(-2, -1)))


class CompactEfficientDet(nn.Module):
    """EfficientDet-style BiFPN detector."""

    def __init__(self, width: int = 32, classes: int = 80) -> None:
        """Initialize backbone, BiFPN fusion, and detector heads.

        Parameters
        ----------
        width:
            Feature width.
        classes:
            Number of object classes.
        """

        super().__init__()
        self.p3 = ConvBNAct(3, width, stride=2)
        self.p4 = ConvBNAct(width, width, stride=2)
        self.p5 = ConvBNAct(width, width, stride=2)
        self.w = nn.Parameter(torch.ones(2, 3))
        self.cls = nn.Conv2d(width, classes, 3, padding=1)
        self.box = nn.Conv2d(width, 4, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Run BiFPN-style weighted feature fusion.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        Tensor
            Concatenated class and box summaries.
        """

        p3 = self.p3(x)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        weights = F.relu(self.w)
        td4 = (weights[0, 0] * p4 + weights[0, 1] * F.interpolate(p5, size=p4.shape[-2:])) / (
            weights[0, :2].sum() + 1e-4
        )
        td3 = (weights[1, 0] * p3 + weights[1, 1] * F.interpolate(td4, size=p3.shape[-2:])) / (
            weights[1, :2].sum() + 1e-4
        )
        cls = self.cls(td3).mean(dim=(-2, -1))
        box = self.box(td3).mean(dim=(-2, -1))
        return torch.cat([cls, box], dim=1)


class CompactHRNetPose(nn.Module):
    """High-resolution multi-branch pose network."""

    def __init__(self, joints: int = 17, width: int = 24) -> None:
        """Initialize HRNet-style parallel branches.

        Parameters
        ----------
        joints:
            Number of keypoint heatmaps.
        width:
            High-resolution branch width.
        """

        super().__init__()
        self.stem = ConvBNAct(3, width)
        self.high = nn.Sequential(ConvBNAct(width, width), ConvBNAct(width, width))
        self.mid = nn.Sequential(
            ConvBNAct(width, width * 2, stride=2), ConvBNAct(width * 2, width * 2)
        )
        self.low = nn.Sequential(
            ConvBNAct(width * 2, width * 4, stride=2), ConvBNAct(width * 4, width * 4)
        )
        self.fuse_mid = nn.Conv2d(width * 2, width, 1)
        self.fuse_low = nn.Conv2d(width * 4, width, 1)
        self.head = nn.Conv2d(width, joints, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict keypoint heatmaps.

        Parameters
        ----------
        x:
            Person crop tensor.

        Returns
        -------
        Tensor
            Heatmap tensor.
        """

        stem = self.stem(x)
        high = self.high(stem)
        mid = self.mid(stem)
        low = self.low(mid)
        fused = high + F.interpolate(self.fuse_mid(mid), size=high.shape[-2:])
        fused = fused + F.interpolate(self.fuse_low(low), size=high.shape[-2:])
        return self.head(fused)


class CompactOSNet(nn.Module):
    """Omni-scale person re-identification network."""

    def __init__(self, width: int = 32, classes: int = 751) -> None:
        """Initialize omni-scale residual streams.

        Parameters
        ----------
        width:
            Feature width.
        classes:
            Identity classifier count.
        """

        super().__init__()
        self.stem = ConvBNAct(3, width, stride=2)
        self.streams = nn.ModuleList(
            [
                nn.Sequential(*[ResidualConvBlock(width) for _inner in range(scale)])
                for scale in (1, 2, 3, 4)
            ]
        )
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(width, width, 1), nn.Sigmoid())
        self.head = nn.Linear(width, classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a person crop by omni-scale aggregation.

        Parameters
        ----------
        x:
            Person image tensor.

        Returns
        -------
        Tensor
            Identity logits.
        """

        feat = self.stem(x)
        stacked = torch.stack([stream(feat) for stream in self.streams], dim=0).mean(dim=0)
        gated = stacked * self.gate(stacked)
        return self.head(gated.mean(dim=(-2, -1)))


class CompactDETR(nn.Module):
    """DETR/Deformable-DETR style encoder-decoder detector."""

    def __init__(self, queries: int = 16, classes: int = 91, deformable: bool = False) -> None:
        """Initialize convolutional backbone and transformer decoder.

        Parameters
        ----------
        queries:
            Number of object queries.
        classes:
            Number of detector classes.
        deformable:
            Whether to include multi-scale deformable offsets.
        """

        super().__init__()
        self.backbone = nn.Sequential(ConvBNAct(3, 32, stride=2), ConvBNAct(32, 64, stride=2))
        self.proj = nn.Conv2d(64, 64, 1)
        self.query = nn.Parameter(torch.randn(queries, 64) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(64, 4, 128, batch_first=True), 2
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(64, 4, 128, batch_first=True), 2
        )
        self.offset = nn.Linear(64, 8 if deformable else 4)
        self.cls = nn.Linear(64, classes)
        self.box = nn.Linear(64, 4)

    def forward(self, x: Tensor) -> Tensor:
        """Detect objects from an image tensor.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        Tensor
            Per-query class, box, and offset predictions.
        """

        feat = self.proj(self.backbone(x))
        memory = feat.flatten(2).transpose(1, 2)
        memory = self.encoder(memory)
        query = self.query.unsqueeze(0).expand(x.shape[0], -1, -1)
        decoded = self.decoder(query, memory)
        return torch.cat(
            [self.cls(decoded), self.box(decoded).sigmoid(), self.offset(decoded)], dim=-1
        )


class CompactPromptSegmenter(nn.Module):
    """Promptable segmentation model with image and point prompts."""

    def __init__(self, prompts: bool = True) -> None:
        """Initialize image encoder, prompt encoder, and mask decoder.

        Parameters
        ----------
        prompts:
            Whether forward input includes prompt coordinates.
        """

        super().__init__()
        self.prompts = prompts
        self.image = nn.Sequential(ConvBNAct(3, 32, stride=2), ConvBNAct(32, 64, stride=2))
        self.prompt = nn.Linear(2, 64)
        self.mask = nn.Conv2d(64, 1, 1)
        self.iou = nn.Linear(64, 1)

    def forward(self, x: Tensor | tuple[Tensor, Tensor]) -> Tensor:
        """Predict a prompt-conditioned mask.

        Parameters
        ----------
        x:
            Image tensor or ``(image, points)`` pair.

        Returns
        -------
        Tensor
            Mask logits and IoU score summary.
        """

        if isinstance(x, tuple):
            image, points = x
        else:
            image = x
            points = torch.zeros(image.shape[0], 1, 2, device=image.device, dtype=image.dtype)
        feat = self.image(image)
        if self.prompts:
            prompt = self.prompt(points.to(feat.dtype)).mean(dim=1).view(image.shape[0], -1, 1, 1)
            feat = feat + prompt
        mask = self.mask(feat)
        score = self.iou(feat.mean(dim=(-2, -1))).view(image.shape[0], 1, 1, 1)
        return torch.cat([mask, score.expand_as(mask)], dim=1)


class CompactBigGANGenerator(nn.Module):
    """BigGAN-style class-conditional generator."""

    def __init__(self, z_dim: int = 64, classes: int = 1000, width: int = 32) -> None:
        """Initialize conditional batch-normalized generator blocks.

        Parameters
        ----------
        z_dim:
            Latent dimension.
        classes:
            Number of class embeddings.
        width:
            Base channel width.
        """

        super().__init__()
        self.embed = nn.Embedding(classes, z_dim)
        self.fc = nn.Linear(z_dim, width * 4 * 4 * 4)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Upsample(scale_factor=2), ConvBNAct(width * 4, width * 2)),
                nn.Sequential(nn.Upsample(scale_factor=2), ConvBNAct(width * 2, width)),
                nn.Sequential(nn.Upsample(scale_factor=2), ConvBNAct(width, width)),
            ]
        )
        self.to_rgb = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Generate an image from latent vectors.

        Parameters
        ----------
        x:
            Latent tensor ``(batch, z_dim)``.

        Returns
        -------
        Tensor
            Generated image.
        """

        labels = torch.arange(x.shape[0], device=x.device) % self.embed.num_embeddings
        z = x + self.embed(labels)
        feat = self.fc(z).view(x.shape[0], -1, 4, 4)
        for block in self.blocks:
            feat = block(feat)
        return torch.tanh(self.to_rgb(feat))


class CompactBigGANDiscriminator(nn.Module):
    """BigGAN-style projection discriminator."""

    def __init__(self, classes: int = 1000, width: int = 32) -> None:
        """Initialize discriminator backbone and projection head.

        Parameters
        ----------
        classes:
            Number of class embeddings.
        width:
            Base channel width.
        """

        super().__init__()
        self.backbone = nn.Sequential(
            ConvBNAct(3, width, stride=2),
            ConvBNAct(width, width * 2, stride=2),
            ConvBNAct(width * 2, width * 4, stride=2),
        )
        self.linear = nn.Linear(width * 4, 1)
        self.embed = nn.Embedding(classes, width * 4)

    def forward(self, x: Tensor) -> Tensor:
        """Score an image with class projection conditioning.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        Tensor
            Discriminator score.
        """

        feat = self.backbone(x).mean(dim=(-2, -1))
        labels = torch.arange(x.shape[0], device=x.device) % self.embed.num_embeddings
        projection = (feat * self.embed(labels)).sum(dim=-1, keepdim=True)
        return self.linear(feat) + projection


class CompactDMGAN(nn.Module):
    """Dynamic Memory GAN text-to-image generator."""

    def __init__(self, z_dim: int = 64, width: int = 32) -> None:
        """Initialize memory writing and image refinement modules.

        Parameters
        ----------
        z_dim:
            Latent dimension.
        width:
            Base channel width.
        """

        super().__init__()
        self.memory = nn.GRUCell(z_dim, z_dim)
        self.generator = CompactBigGANGenerator(z_dim, classes=16, width=width)
        self.refine = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Generate and refine an image using recurrent memory.

        Parameters
        ----------
        x:
            Latent tensor.

        Returns
        -------
        Tensor
            Refined generated image.
        """

        mem = torch.zeros_like(x)
        for _step in range(3):
            mem = self.memory(x, mem)
        image = self.generator(mem)
        return torch.tanh(image + self.refine(image))


class CompactGraphModel(nn.Module):
    """Compact graph neural network covering PyG-style residual rows."""

    def __init__(self, mode: str, in_ch: int = 16, hidden: int = 32, out_ch: int = 8) -> None:
        """Initialize message-passing, pooling, and readout components.

        Parameters
        ----------
        mode:
            Graph family mode.
        in_ch:
            Input feature dimension.
        hidden:
            Hidden feature dimension.
        out_ch:
            Output dimension.
        """

        super().__init__()
        self.mode = mode
        self.node = nn.Linear(in_ch, hidden)
        self.msg = nn.Linear(hidden, hidden)
        self.update = nn.GRUCell(hidden, hidden)
        self.out = nn.Linear(hidden, out_ch)
        self.score = nn.Linear(hidden, 1)
        self.lstm_cell = nn.LSTMCell(hidden, hidden)
        self.temporal = nn.Conv1d(hidden, hidden, 3, padding=1, dilation=1)
        self.register_buffer(
            "edge_src", torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 2, 4], dtype=torch.long)
        )
        self.register_buffer(
            "edge_dst", torch.tensor([1, 2, 3, 4, 5, 6, 7, 2, 4, 6], dtype=torch.long)
        )

    def _aggregate(self, h: Tensor) -> Tensor:
        """Aggregate fixed synthetic graph neighborhoods.

        Parameters
        ----------
        h:
            Node embeddings.

        Returns
        -------
        Tensor
            Aggregated messages.
        """

        src = self.edge_src.clamp_max(h.shape[0] - 1)
        dst = self.edge_dst.clamp_max(h.shape[0] - 1)
        messages = self.msg(h[src])
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, messages)
        deg = torch.zeros(h.shape[0], 1, device=h.device, dtype=h.dtype)
        deg.index_add_(0, dst, torch.ones_like(messages[:, :1]))
        return agg / deg.clamp_min(1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Run the selected graph computation over node features.

        Parameters
        ----------
        x:
            Node feature tensor ``(nodes, features)``.

        Returns
        -------
        Tensor
            Graph, node, or pooled features.
        """

        h = F.gelu(self.node(x))
        if self.mode in {"set2set", "lstm_aggr"}:
            hx = torch.zeros(1, h.shape[1], device=h.device, dtype=h.dtype)
            cx = torch.zeros_like(hx)
            readouts = []
            for node_index in range(h.shape[0]):
                hx, cx = self.lstm_cell(h[node_index : node_index + 1], (hx, cx))
                readouts.append(hx)
            seq = torch.cat(readouts, dim=0)
            pooled = torch.cat([seq.mean(dim=0, keepdim=True), seq[-1:]], dim=-1)
            return pooled
        if self.mode in {"cluster_pool", "pan_pool", "graph_unet"}:
            score = self.score(h).squeeze(-1)
            keep = torch.topk(score, k=max(2, h.shape[0] // 2)).indices
            h = h[keep] + self._aggregate(h)[keep]
            if self.mode == "graph_unet":
                up = torch.zeros(x.shape[0], h.shape[1], device=x.device, dtype=h.dtype)
                up.index_copy_(0, keep, h)
                h = up + F.gelu(self.node(x))
            return self.out(h).mean(dim=0, keepdim=True)
        if self.mode in {"mtgnn", "dygrae", "aagcn"}:
            h = h + self._aggregate(h)
            temporal = self.temporal(h.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
            return self.out(F.gelu(temporal)).mean(dim=0, keepdim=True)
        for _step in range(3):
            h = self.update(self._aggregate(h), h)
        if self.mode == "correct_smooth":
            logits = self.out(h)
            return logits + 0.5 * (self.out(self._aggregate(h)) - logits)
        return self.out(h).mean(dim=0, keepdim=True)


class CompactDGCNN(nn.Module):
    """Dynamic Graph CNN with EdgeConv blocks."""

    def __init__(self, k: int = 8, width: int = 32) -> None:
        """Initialize EdgeConv projections.

        Parameters
        ----------
        k:
            Number of nearest neighbors.
        width:
            Hidden channel width.
        """

        super().__init__()
        self.k = k
        self.edge1 = nn.Sequential(nn.Conv2d(6, width, 1), nn.GELU(), nn.Conv2d(width, width, 1))
        self.edge2 = nn.Sequential(
            nn.Conv2d(width * 2, width, 1), nn.GELU(), nn.Conv2d(width, width, 1)
        )
        self.head = nn.Linear(width * 2, 40)

    def _edge_features(self, x: Tensor) -> Tensor:
        """Construct kNN edge features.

        Parameters
        ----------
        x:
            Point features ``(batch, channels, points)``.

        Returns
        -------
        Tensor
            Edge features.
        """

        points = x.transpose(1, 2)
        dist = torch.cdist(points, points)
        idx = dist.topk(k=min(self.k, points.shape[1]), largest=False).indices
        neighbors = torch.gather(
            points[:, None].expand(-1, points.shape[1], -1, -1),
            2,
            idx[..., None].expand(-1, -1, -1, points.shape[-1]),
        )
        center = points[:, :, None, :].expand_as(neighbors)
        return torch.cat([center, neighbors - center], dim=-1).permute(0, 3, 1, 2)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a point cloud.

        Parameters
        ----------
        x:
            Point cloud tensor ``(batch, 3, points)``.

        Returns
        -------
        Tensor
            Class logits.
        """

        edge = self._edge_features(x)
        feat1 = self.edge1(edge).max(dim=-1).values
        edge2 = self._edge_features(feat1)
        feat2 = self.edge2(edge2).max(dim=-1).values
        pooled = torch.cat([feat1.max(dim=-1).values, feat2.max(dim=-1).values], dim=-1)
        return self.head(pooled)


class CompactAtlasNet(nn.Module):
    """AtlasNet-style parametric surface decoder."""

    def __init__(self, points: int = 64, primitives: int = 4, latent: int = 64) -> None:
        """Initialize primitive chart decoders.

        Parameters
        ----------
        points:
            Number of input and output points.
        primitives:
            Number of learned surface charts.
        latent:
            Latent feature dimension.
        """

        super().__init__()
        self.points = points
        self.primitives = primitives
        self.encoder = nn.Sequential(
            nn.Conv1d(3, latent, 1), nn.GELU(), nn.Conv1d(latent, latent, 1)
        )
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(latent + 2, latent), nn.GELU(), nn.Linear(latent, 3))
                for _index in range(primitives)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Decode surface charts from a point cloud.

        Parameters
        ----------
        x:
            Input point cloud ``(batch, 3, points)``.

        Returns
        -------
        Tensor
            Reconstructed point cloud.
        """

        latent = self.encoder(x).max(dim=-1).values
        per_primitive = self.points // self.primitives
        grid = torch.linspace(-1.0, 1.0, per_primitive, device=x.device, dtype=x.dtype)
        grid = torch.stack([grid, torch.sin(grid * math.pi)], dim=-1)
        outs = []
        for decoder in self.decoders:
            z = latent[:, None, :].expand(-1, per_primitive, -1)
            chart = torch.cat([z, grid[None].expand(x.shape[0], -1, -1)], dim=-1)
            outs.append(decoder(chart))
        return torch.cat(outs, dim=1).transpose(1, 2)


class CompactSeq2SeqTransformer(nn.Module):
    """BART-style encoder-decoder transformer."""

    def __init__(self, vocab: int = 128, dim: int = 48) -> None:
        """Initialize token embeddings and encoder-decoder layers.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Token dimension.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, 32, dim) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True), 2
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, 4, dim * 4, batch_first=True), 2
        )
        self.head = nn.Linear(dim, vocab)

    def forward(self, x: Tensor) -> Tensor:
        """Run sequence-to-sequence language modeling.

        Parameters
        ----------
        x:
            Integer token IDs.

        Returns
        -------
        Tensor
            Token logits.
        """

        tokens = x.long()
        src = self.embed(tokens) + self.pos[:, : tokens.shape[1]]
        memory = self.encoder(src)
        tgt = self.embed(tokens[:, : max(1, tokens.shape[1] // 2)])
        decoded = self.decoder(tgt, memory)
        return self.head(decoded)


class CompactMaskedLM(nn.Module):
    """RoBERTa/Data2Vec-style masked language model."""

    def __init__(self, vocab: int = 128, dim: int = 48) -> None:
        """Initialize encoder-only language model.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Token dimension.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.randn(1, 64, dim) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True), 3
        )
        self.lm_head = nn.Linear(dim, vocab)

    def forward(self, x: Tensor) -> Tensor:
        """Predict masked-token logits.

        Parameters
        ----------
        x:
            Integer token IDs.

        Returns
        -------
        Tensor
            Token logits.
        """

        ids = x.long().clamp_min(0) % self.embed.num_embeddings
        hidden = self.embed(ids) + self.pos[:, : ids.shape[1]]
        return self.lm_head(self.encoder(hidden))


class CompactCausalHybrid(nn.Module):
    """Compact causal language model with attention and state-space mixing."""

    def __init__(self, vocab: int = 128, dim: int = 48, ssm: bool = False) -> None:
        """Initialize hybrid causal blocks.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Token dimension.
        ssm:
            Whether to include a convolutional state-space branch.
        """

        super().__init__()
        self.ssm = ssm
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, 4) for _index in range(2)])
        self.conv = nn.Conv1d(dim, dim, 5, padding=4, groups=dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x: Tensor) -> Tensor:
        """Run causal sequence modeling.

        Parameters
        ----------
        x:
            Integer token IDs.

        Returns
        -------
        Tensor
            Token logits.
        """

        hidden = self.embed(x.long() % self.embed.num_embeddings)
        for block in self.blocks:
            hidden = block(hidden)
            if self.ssm:
                conv = self.conv(hidden.transpose(1, 2))[..., : hidden.shape[1]].transpose(1, 2)
                hidden = hidden + conv
        return self.head(hidden)


class CompactTimeSeriesTransformer(nn.Module):
    """Encoder-decoder transformer for time-series forecasting."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize time-series projection and transformer.

        Parameters
        ----------
        dim:
            Hidden dimension.
        """

        super().__init__()
        self.in_proj = nn.Linear(4, dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 4, batch_first=True), 2
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, 4, dim * 4, batch_first=True), 1
        )
        self.out = nn.Linear(dim, 4)

    def forward(self, x: Tensor) -> Tensor:
        """Forecast future time-series values.

        Parameters
        ----------
        x:
            Past values ``(batch, time, variables)``.

        Returns
        -------
        Tensor
            Forecast tensor.
        """

        hidden = self.in_proj(x)
        memory = self.encoder(hidden)
        tgt = hidden[:, -4:]
        return self.out(self.decoder(tgt, memory))


class CompactAudioVAE(nn.Module):
    """Audio VAE with DiT-style latent refinement."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize audio encoder, latent transformer, and decoder.

        Parameters
        ----------
        channels:
            Hidden channel count.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNAct(1, channels, stride=2, dims=1),
            ConvBNAct(channels, channels, stride=2, dims=1),
        )
        self.to_tokens = nn.Linear(channels, channels)
        self.blocks = nn.ModuleList([TransformerBlock(channels, 4) for _index in range(2)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(channels, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode, refine, and decode an audio waveform.

        Parameters
        ----------
        x:
            Audio tensor ``(batch, 1, time)``.

        Returns
        -------
        Tensor
            Reconstructed audio.
        """

        latent = self.encoder(x)
        tokens = self.to_tokens(latent.transpose(1, 2))
        for block in self.blocks:
            tokens = block(tokens)
        return self.decoder(tokens.transpose(1, 2))


class CompactControlNet(nn.Module):
    """ControlNet-style conditioned diffusion residual branch."""

    def __init__(self, in_ch: int = 4, cond_ch: int = 3, width: int = 32) -> None:
        """Initialize latent and conditioning encoders.

        Parameters
        ----------
        in_ch:
            Latent channel count.
        cond_ch:
            Conditioning image channel count.
        width:
            Hidden width.
        """

        super().__init__()
        self.latent = ConvBNAct(in_ch, width)
        self.cond = ConvBNAct(cond_ch, width)
        self.blocks = nn.Sequential(ResidualConvBlock(width), ResidualConvBlock(width))
        self.zero = nn.Conv2d(width, in_ch, 1)
        nn.init.zeros_(self.zero.weight)
        nn.init.zeros_(self.zero.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Predict conditioned residuals for latent diffusion.

        Parameters
        ----------
        x:
            Latent tensor.

        Returns
        -------
        Tensor
            Residual tensor.
        """

        cond = torch.tanh(x[:, :3])
        feat = self.latent(x) + F.interpolate(self.cond(cond), size=x.shape[-2:])
        return self.zero(self.blocks(feat))


def build_vit(name: str = "vit") -> nn.Module:
    """Build a compact ViT-family model.

    Parameters
    ----------
    name:
        Source model name used to enable family-specific details.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    register_tokens = 4 if "reg" in name or "dinov" in name or "pe_" in name else 0
    mean_pool = "gap" in name or "siglip" in name
    channel_attention = "eva" in name or "beit3" in name or "aimv2" in name
    depth = 5 if any(token in name for token in ("giant", "gigantic", "7b", "3b")) else 4
    return CompactVisionTransformer(
        image_size=64,
        patch_size=8,
        dim=64,
        depth=depth,
        register_tokens=register_tokens,
        channel_attention=channel_attention,
        use_mean_pool=mean_pool,
    )


def build_cait() -> nn.Module:
    """Build a compact CaiT model.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactCaiT()


def build_swin2d() -> nn.Module:
    """Build a compact 2D Swin/SwinV2 model.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactSwin2D()


def build_swin3d() -> nn.Module:
    """Build a compact 3D Video Swin model.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactVideoSwin()


def build_xcit() -> nn.Module:
    """Build a compact XCiT-style channel-attention model.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactVisionTransformer(channel_attention=True, use_mean_pool=True)


def build_efficientnet() -> nn.Module:
    """Build a compact EfficientNet model.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactEfficientNet()


def build_nfnet() -> nn.Module:
    """Build a compact NFNet model.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactNFNet()


def build_efficientdet() -> nn.Module:
    """Build a compact EfficientDet model.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactEfficientDet()


def build_hrnet_pose() -> nn.Module:
    """Build a compact HRNet pose model.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactHRNetPose()


def build_osnet() -> nn.Module:
    """Build a compact OSNet model.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactOSNet()


def build_detr(deformable: bool = False) -> nn.Module:
    """Build a compact DETR-family detector.

    Parameters
    ----------
    deformable:
        Whether to include deformable offset prediction.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactDETR(deformable=deformable)


def build_prompt_segmenter(prompts: bool = True) -> nn.Module:
    """Build a compact promptable segmenter.

    Parameters
    ----------
    prompts:
        Whether point prompts are consumed.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactPromptSegmenter(prompts=prompts)


def build_biggan_generator() -> nn.Module:
    """Build a compact BigGAN generator.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactBigGANGenerator()


def build_biggan_discriminator() -> nn.Module:
    """Build a compact BigGAN discriminator.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactBigGANDiscriminator()


def build_dmgan() -> nn.Module:
    """Build a compact DM-GAN generator.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactDMGAN()


def build_graph(mode: str = "message") -> nn.Module:
    """Build a compact graph-family model.

    Parameters
    ----------
    mode:
        Graph architecture mode.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactGraphModel(mode)


def build_dgcnn() -> nn.Module:
    """Build a compact DGCNN model.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactDGCNN()


def build_atlasnet() -> nn.Module:
    """Build a compact AtlasNet model.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactAtlasNet()


def build_masked_lm() -> nn.Module:
    """Build a compact encoder masked-language model.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactMaskedLM()


def build_seq2seq() -> nn.Module:
    """Build a compact seq2seq transformer.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactSeq2SeqTransformer()


def build_causal_hybrid(ssm: bool = False) -> nn.Module:
    """Build a compact causal attention/SSM hybrid.

    Parameters
    ----------
    ssm:
        Whether to include a state-space convolutional branch.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactCausalHybrid(ssm=ssm)


def build_time_series_transformer() -> nn.Module:
    """Build a compact time-series transformer.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactTimeSeriesTransformer()


def build_audio_vae() -> nn.Module:
    """Build a compact audio VAE.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactAudioVAE()


def build_controlnet() -> nn.Module:
    """Build a compact ControlNet branch.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactControlNet()


class CompactImageVAE(nn.Module):
    """Consistency-decoder style convolutional image VAE."""

    def __init__(self, latent: int = 16) -> None:
        """Initialize encoder, latent projection, and decoder.

        Parameters
        ----------
        latent:
            Latent channel count.
        """

        super().__init__()
        self.encoder = nn.Sequential(ConvBNAct(3, 32, stride=2), ConvBNAct(32, latent, stride=2))
        self.mean = nn.Conv2d(latent, latent, 1)
        self.logvar = nn.Conv2d(latent, latent, 1)
        self.refine = ResidualConvBlock(latent)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct an image through a compact latent path.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        Tensor
            Reconstructed image.
        """

        encoded = self.encoder(x)
        latent = self.mean(encoded) + torch.tanh(self.logvar(encoded)) * 0.1
        return torch.tanh(self.decoder(self.refine(latent)))


class CompactMetaLSTM(nn.Module):
    """Meta-learning LSTM optimizer cell."""

    def __init__(self, features: int = 10, hidden: int = 32) -> None:
        """Initialize gradient/loss encoder and LSTM optimizer.

        Parameters
        ----------
        features:
            Input feature count.
        hidden:
            Hidden optimizer state size.
        """

        super().__init__()
        self.grad_proj = nn.Linear(features, hidden)
        self.loss_proj = nn.Linear(features, hidden)
        self.cell = nn.LSTMCell(hidden * 2, hidden)
        self.delta = nn.Linear(hidden, features)

    def forward(self, x: Tensor) -> Tensor:
        """Predict parameter updates from a sequence of pseudo-gradients.

        Parameters
        ----------
        x:
            Tensor ``(batch, steps, features)``.

        Returns
        -------
        Tensor
            Learned update sequence.
        """

        hx = torch.zeros(x.shape[0], self.cell.hidden_size, device=x.device, dtype=x.dtype)
        cx = torch.zeros_like(hx)
        outputs = []
        for step in range(x.shape[1]):
            grad = self.grad_proj(x[:, step])
            loss_context = self.loss_proj(x[:, : step + 1].mean(dim=1))
            hx, cx = self.cell(torch.cat([grad, loss_context], dim=-1), (hx, cx))
            outputs.append(self.delta(hx))
        return torch.stack(outputs, dim=1)


class CompactMotionAdapter(nn.Module):
    """AnimateDiff-style temporal adapter for latent U-Nets."""

    def __init__(self, channels: int = 4, width: int = 32) -> None:
        """Initialize spatial and temporal residual branches.

        Parameters
        ----------
        channels:
            Latent channel count.
        width:
            Hidden width.
        """

        super().__init__()
        self.in_proj = nn.Conv3d(channels, width, 3, padding=1)
        self.temporal = nn.Conv3d(width, width, (3, 1, 1), padding=(1, 0, 0), groups=width)
        self.spatial = nn.Conv3d(width, width, (1, 3, 3), padding=(0, 1, 1))
        self.out = nn.Conv3d(width, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Adapt video latents with temporal residual mixing.

        Parameters
        ----------
        x:
            Latent video tensor.

        Returns
        -------
        Tensor
            Adapted latent tensor.
        """

        feat = F.gelu(self.in_proj(x))
        feat = feat + F.gelu(self.temporal(feat))
        feat = feat + F.gelu(self.spatial(feat))
        return x + self.out(feat)


class CompactAurora(nn.Module):
    """Aurora-like 3D atmosphere encoder-decoder."""

    def __init__(self, variables: int = 4, width: int = 32) -> None:
        """Initialize 3D weather token mixer.

        Parameters
        ----------
        variables:
            Number of atmospheric variables.
        width:
            Hidden channel width.
        """

        super().__init__()
        self.embed = nn.Conv3d(variables, width, 3, padding=1)
        self.down = nn.Conv3d(width, width, 3, stride=(1, 2, 2), padding=1)
        self.mix = TransformerBlock(width, 4)
        self.up = nn.ConvTranspose3d(width, width, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.out = nn.Conv3d(width, variables, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forecast atmosphere variables on a compact pressure-level grid.

        Parameters
        ----------
        x:
            Weather tensor ``(batch, variables, levels, height, width)``.

        Returns
        -------
        Tensor
            Forecast tensor.
        """

        feat = F.gelu(self.embed(x))
        feat = F.gelu(self.down(feat))
        bsz, channels, levels, height, width = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.mix(tokens)
        feat = tokens.transpose(1, 2).reshape(bsz, channels, levels, height, width)
        return self.out(F.gelu(self.up(feat)))


class CompactSpeechGenerator(nn.Module):
    """CSM-style text-conditioned speech generator."""

    def __init__(self, vocab: int = 128, dim: int = 48) -> None:
        """Initialize text encoder and neural codec decoder.

        Parameters
        ----------
        vocab:
            Text vocabulary size.
        dim:
            Hidden dimension.
        """

        super().__init__()
        self.text = CompactCausalHybrid(vocab=vocab, dim=dim)
        self.to_audio = nn.Linear(vocab, dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(dim, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Generate a waveform from text tokens.

        Parameters
        ----------
        x:
            Integer text tokens.

        Returns
        -------
        Tensor
            Generated waveform.
        """

        logits = self.text(x)
        audio_tokens = self.to_audio(logits).transpose(1, 2)
        return torch.tanh(self.decoder(audio_tokens))


class CompactSphericalHarmonics(nn.Module):
    """Real low-order spherical-harmonics basis."""

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate a compact real spherical-harmonics basis.

        Parameters
        ----------
        x:
            Cartesian vectors.

        Returns
        -------
        Tensor
            Harmonic basis features.
        """

        unit = F.normalize(x, dim=-1)
        x0, y0, z0 = unit.unbind(dim=-1)
        return torch.stack(
            [
                torch.ones_like(x0),
                x0,
                y0,
                z0,
                x0 * y0,
                y0 * z0,
                3.0 * z0.square() - 1.0,
                x0 * z0,
                x0.square() - y0.square(),
            ],
            dim=-1,
        )


class CompactTensorProduct(nn.Module):
    """Fully connected equivariant-style tensor product."""

    def __init__(self, in_ch: int = 5, out_ch: int = 2) -> None:
        """Initialize learned bilinear product weights.

        Parameters
        ----------
        in_ch:
            Input feature count.
        out_ch:
            Output feature count.
        """

        super().__init__()
        self.left = nn.Linear(in_ch, out_ch, bias=False)
        self.right = nn.Linear(in_ch, out_ch, bias=False)
        self.mix = nn.Linear(out_ch * 3, out_ch)

    def forward(self, x: Tensor) -> Tensor:
        """Apply bilinear tensor-product mixing.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        Tensor
            Mixed equivariant features.
        """

        left = self.left(x)
        right = self.right(torch.roll(x, shifts=1, dims=-1))
        return self.mix(torch.cat([left, right, left * right], dim=-1))


class CompactR2Conv(nn.Module):
    """Rotation-steerable planar convolution."""

    def __init__(self, rotations: int = 8) -> None:
        """Initialize shared rotated-filter bank.

        Parameters
        ----------
        rotations:
            Number of discrete planar rotations.
        """

        super().__init__()
        self.rotations = rotations
        self.kernel = nn.Parameter(torch.randn(4, 3, 3, 3) * 0.05)
        self.mix = nn.Conv2d(4 * rotations, 8, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply a discrete rotation-equivariant convolution.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        Tensor
            Steerable feature map.
        """

        outs = []
        for rotation in range(self.rotations):
            kernel = torch.rot90(self.kernel, k=rotation % 4, dims=(-2, -1))
            outs.append(F.conv2d(x, kernel, padding=1))
        return self.mix(torch.cat(outs, dim=1))


class CompactYoloOBB(nn.Module):
    """YOLO-style one-stage oriented-box detector."""

    def __init__(self, classes: int = 80, width: int = 32) -> None:
        """Initialize CSP backbone, PAN neck, and OBB head.

        Parameters
        ----------
        classes:
            Number of classes.
        width:
            Base channel width.
        """

        super().__init__()
        self.stem = ConvBNAct(3, width, stride=2)
        self.csp1 = nn.Sequential(ResidualConvBlock(width), ConvBNAct(width, width * 2, stride=2))
        self.csp2 = nn.Sequential(
            ResidualConvBlock(width * 2), ConvBNAct(width * 2, width * 4, stride=2)
        )
        self.reduce = nn.Conv2d(width * 4, width * 2, 1)
        self.head = nn.Conv2d(width * 2, classes + 5, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict oriented boxes and classes.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        Tensor
            Dense oriented-box predictions.
        """

        p3 = self.stem(x)
        p4 = self.csp1(p3)
        p5 = self.csp2(p4)
        fused = p4 + F.interpolate(self.reduce(p5), size=p4.shape[-2:])
        pred = self.head(fused)
        box = pred[:, :5]
        cls = pred[:, 5:]
        return torch.cat([box, cls], dim=1)


def build_image_vae() -> nn.Module:
    """Build a compact image VAE.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactImageVAE()


def build_meta_lstm() -> nn.Module:
    """Build a compact Meta-LSTM optimizer.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactMetaLSTM()


def build_motion_adapter() -> nn.Module:
    """Build a compact temporal motion adapter.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactMotionAdapter()


def build_aurora() -> nn.Module:
    """Build a compact Aurora-like atmosphere model.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactAurora()


def build_speech_generator() -> nn.Module:
    """Build a compact speech generator.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactSpeechGenerator()


def build_spherical_harmonics() -> nn.Module:
    """Build a compact spherical harmonics module.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactSphericalHarmonics()


def build_tensor_product() -> nn.Module:
    """Build a compact tensor-product module.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactTensorProduct()


def build_r2conv() -> nn.Module:
    """Build a compact steerable R2 convolution.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactR2Conv()


def build_yolo_obb() -> nn.Module:
    """Build a compact oriented-box YOLO detector.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return CompactYoloOBB()
