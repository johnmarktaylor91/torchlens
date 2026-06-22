"""Dependency-gated reimplementations for REIMPL5 shard 2.

Paper anchors:
IgFold, Ruffolo et al. 2023; ImageBind, Girdhar et al. 2023; FUNIT,
Liu et al. 2019; pix2pixHD, Wang et al. 2018; SPADE/GauGAN, Park et al.
2019; SSD, Liu et al. 2016; TOOD, Feng et al. 2021; YOLOX, Ge et al. 2021;
BasicVSR++, Chan et al. 2022; CRAFT, Baek et al. 2019; Implicit Behavioral
Cloning, Florence et al. 2021.

These are random-init compact reconstructions that preserve the distinctive
architecture primitives needed for TorchLens rendering, not pretrained models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Small two-convolution residual block."""

    def __init__(self, channels: int) -> None:
        """Initialize the residual block.

        Parameters
        ----------
        channels:
            Feature width.
        """

        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a residual convolution update.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Updated feature map.
        """

        return x + self.conv2(F.relu(self.conv1(x)))


class MLP(nn.Module):
    """Layer-normalized feed-forward network."""

    def __init__(self, dim: int, hidden: int, out: int) -> None:
        """Initialize the MLP.

        Parameters
        ----------
        dim:
            Input width.
        hidden:
            Hidden width.
        out:
            Output width.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the feed-forward network.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        return self.net(x)


class TinySelfAttention(nn.Module):
    """Compact multi-head self-attention."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize attention projections.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scaled dot-product self-attention.

        Parameters
        ----------
        x:
            Token tensor shaped ``(batch, tokens, dim)``.

        Returns
        -------
        torch.Tensor
            Attended tokens.
        """

        batch, tokens, dim = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * (self.head_dim**-0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: int = 2) -> None:
        """Initialize attention and MLP sublayers.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Number of attention heads.
        mlp_ratio:
            Hidden multiplier.
        """

        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TinySelfAttention(dim, heads)
        self.mlp = MLP(dim, dim * mlp_ratio, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one transformer block.

        Parameters
        ----------
        x:
            Token tensor.

        Returns
        -------
        torch.Tensor
            Updated token tensor.
        """

        x = x + self.attn(self.norm1(x))
        return x + self.mlp(x)


class IgFoldCompact(nn.Module):
    """IgFold-like antibody folder with AntiBERTy tokens, pair trunk, and IPA-style points."""

    def __init__(self, vocab: int = 32, dim: int = 48, pair_dim: int = 24) -> None:
        """Initialize the compact IgFold reconstruction.

        Parameters
        ----------
        vocab:
            Amino-acid vocabulary size.
        dim:
            Residue embedding width.
        pair_dim:
            Pair representation width.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.seq_blocks = nn.ModuleList([TransformerBlock(dim, heads=4) for _ in range(2)])
        self.pair_left = nn.Linear(dim, pair_dim)
        self.pair_right = nn.Linear(dim, pair_dim)
        self.triangle = nn.Conv2d(pair_dim, pair_dim, 1)
        self.point_q = nn.Linear(dim, 3)
        self.point_k = nn.Linear(dim, 3)
        self.update = nn.Linear(dim + pair_dim, dim)
        self.coord = nn.Linear(dim, 4 * 3)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict per-residue backbone atom coordinates.

        Parameters
        ----------
        tokens:
            Antibody residue ids shaped ``(batch, residues)``.

        Returns
        -------
        torch.Tensor
            Coordinates shaped ``(batch, residues, 4, 3)``.
        """

        single = self.embed(tokens)
        for block in self.seq_blocks:
            single = block(single)
        left = self.pair_left(single).unsqueeze(2)
        right = self.pair_right(single).unsqueeze(1)
        pair = torch.tanh(left + right).permute(0, 3, 1, 2)
        pair = pair + self.triangle(pair)
        dist = torch.cdist(self.point_q(single), self.point_k(single)).unsqueeze(1)
        ipa_weights = torch.softmax(pair.mean(1, keepdim=True) - dist, dim=-1)
        pair_context = (ipa_weights * pair).sum(-1).transpose(1, 2)
        single = single + torch.tanh(self.update(torch.cat([single, pair_context], dim=-1)))
        return self.coord(single).view(tokens.shape[0], tokens.shape[1], 4, 3)


def build_igfold() -> nn.Module:
    """Build compact IgFold."""

    return IgFoldCompact()


def example_igfold() -> torch.Tensor:
    """Return antibody token ids."""

    return torch.randint(0, 32, (1, 14))


class ImageBindCompact(nn.Module):
    """ImageBind-like six-modality joint embedding model."""

    def __init__(self, dim: int = 48, out_dim: int = 32) -> None:
        """Initialize modality trunks and shared projection.

        Parameters
        ----------
        dim:
            Internal token width.
        out_dim:
            Joint embedding width.
        """

        super().__init__()
        self.image = nn.Conv2d(3, dim, 4, stride=4)
        self.audio = nn.Conv2d(1, dim, (4, 4), stride=(4, 4))
        self.depth = nn.Conv2d(1, dim, 4, stride=4)
        self.thermal = nn.Conv2d(1, dim, 4, stride=4)
        self.imu = nn.Linear(6, dim)
        self.text = nn.Embedding(64, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads=4) for _ in range(2)])
        self.modality = nn.Parameter(torch.randn(6, dim) * 0.02)
        self.proj = nn.Linear(dim, out_dim)

    def _pool_visual(self, x: torch.Tensor, conv: nn.Conv2d, index: int) -> torch.Tensor:
        """Encode grid-like modalities into one embedding.

        Parameters
        ----------
        x:
            Image-like tensor.
        conv:
            Patch projection.
        index:
            Modality embedding index.

        Returns
        -------
        torch.Tensor
            Normalized joint embedding.
        """

        tokens = conv(x).flatten(2).transpose(1, 2) + self.modality[index]
        for block in self.blocks:
            tokens = block(tokens)
        return F.normalize(self.proj(tokens.mean(1)), dim=-1)

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
        audio: torch.Tensor,
        depth: torch.Tensor,
        thermal: torch.Tensor,
        imu: torch.Tensor,
    ) -> torch.Tensor:
        """Return six aligned modality embeddings.

        Parameters
        ----------
        image:
            RGB image.
        text:
            Token ids.
        audio:
            Audio spectrogram.
        depth:
            Depth image.
        thermal:
            Thermal image.
        imu:
            IMU sequence with six channels.

        Returns
        -------
        torch.Tensor
            Embeddings shaped ``(batch, 6, out_dim)``.
        """

        image_z = self._pool_visual(image, self.image, 0)
        audio_z = self._pool_visual(audio, self.audio, 2)
        depth_z = self._pool_visual(depth, self.depth, 3)
        thermal_z = self._pool_visual(thermal, self.thermal, 4)
        text_tokens = self.text(text) + self.modality[1]
        imu_tokens = self.imu(imu) + self.modality[5]
        for block in self.blocks:
            text_tokens = block(text_tokens)
            imu_tokens = block(imu_tokens)
        text_z = F.normalize(self.proj(text_tokens.mean(1)), dim=-1)
        imu_z = F.normalize(self.proj(imu_tokens.mean(1)), dim=-1)
        return torch.stack([image_z, text_z, audio_z, depth_z, thermal_z, imu_z], dim=1)


def build_imagebind_jointembedding() -> nn.Module:
    """Build compact ImageBind joint embedding model."""

    return ImageBindCompact()


def example_imagebind() -> tuple[torch.Tensor, ...]:
    """Return six modality inputs."""

    return (
        torch.randn(1, 3, 32, 32),
        torch.randint(0, 64, (1, 8)),
        torch.randn(1, 1, 32, 32),
        torch.randn(1, 1, 32, 32),
        torch.randn(1, 1, 32, 32),
        torch.randn(1, 10, 6),
    )


class AdaIN(nn.Module):
    """Adaptive instance normalization from a style code."""

    def __init__(self, channels: int, style_dim: int) -> None:
        """Initialize style projections.

        Parameters
        ----------
        channels:
            Feature channels.
        style_dim:
            Style vector width.
        """

        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.gamma = nn.Linear(style_dim, channels)
        self.beta = nn.Linear(style_dim, channels)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Apply spatially uniform style modulation.

        Parameters
        ----------
        x:
            Feature map.
        style:
            Style code.

        Returns
        -------
        torch.Tensor
            Modulated feature map.
        """

        gamma = self.gamma(style).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(style).unsqueeze(-1).unsqueeze(-1)
        return self.norm(x) * (1 + gamma) + beta


class FUNITBlock(nn.Module):
    """Residual block with AdaIN target-class style injection."""

    def __init__(self, channels: int, style_dim: int) -> None:
        """Initialize a FUNIT generator block.

        Parameters
        ----------
        channels:
            Feature width.
        style_dim:
            Target-class style width.
        """

        super().__init__()
        self.adain1 = AdaIN(channels, style_dim)
        self.adain2 = AdaIN(channels, style_dim)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Run a style-modulated residual block.

        Parameters
        ----------
        x:
            Content feature map.
        style:
            Target-class style code.

        Returns
        -------
        torch.Tensor
            Updated feature map.
        """

        y = self.conv1(F.relu(self.adain1(x, style)))
        y = self.conv2(F.relu(self.adain2(y, style)))
        return x + y


class FUNITGenerator(nn.Module):
    """FUNIT/COCO-FUNIT few-shot generator with content and class encoders."""

    def __init__(self, style_dim: int = 32, channels: int = 32) -> None:
        """Initialize content, class, and decoder networks.

        Parameters
        ----------
        style_dim:
            Target style code width.
        channels:
            Feature width.
        """

        super().__init__()
        self.content = nn.Sequential(
            nn.Conv2d(3, channels, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.class_encoder = nn.Sequential(
            nn.Conv2d(3, channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, style_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.blocks = nn.ModuleList([FUNITBlock(channels, style_dim) for _ in range(2)])
        self.to_rgb = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, content: torch.Tensor, exemplars: torch.Tensor) -> torch.Tensor:
        """Translate content toward a target class represented by examples.

        Parameters
        ----------
        content:
            Source content image.
        exemplars:
            Target-class examples shaped ``(batch, shots, 3, height, width)``.

        Returns
        -------
        torch.Tensor
            Generated RGB image.
        """

        batch, shots, channels, height, width = exemplars.shape
        style = self.class_encoder(exemplars.view(batch * shots, channels, height, width)).view(
            batch, shots, -1
        )
        style = style.mean(1)
        x = self.content(content)
        for block in self.blocks:
            x = block(x, style)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return torch.tanh(self.to_rgb(x))


def build_funit_generator() -> nn.Module:
    """Build compact FUNIT generator."""

    return FUNITGenerator()


def example_funit() -> tuple[torch.Tensor, torch.Tensor]:
    """Return content image and target exemplars."""

    return torch.randn(1, 3, 32, 32), torch.randn(1, 2, 3, 32, 32)


class Pix2PixHDGenerator(nn.Module):
    """pix2pixHD generator with global path, local enhancer, and residual trunk."""

    def __init__(self, in_channels: int = 8, channels: int = 24) -> None:
        """Initialize compact pix2pixHD generator.

        Parameters
        ----------
        in_channels:
            Label-map channels.
        channels:
            Base feature width.
        """

        super().__init__()
        self.global_down = nn.Sequential(
            nn.Conv2d(in_channels, channels, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(channels * 2),
            ResidualBlock(channels * 2),
        )
        self.global_up = nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1)
        self.local = nn.Sequential(nn.Conv2d(in_channels, channels, 3, padding=1), nn.ReLU())
        self.fuse = nn.Sequential(ResidualBlock(channels), nn.Conv2d(channels, 3, 7, padding=3))

    def forward(self, label: torch.Tensor) -> torch.Tensor:
        """Synthesize an image from semantic labels.

        Parameters
        ----------
        label:
            Semantic label tensor.

        Returns
        -------
        torch.Tensor
            RGB image.
        """

        return torch.tanh(
            self.fuse(F.relu(self.global_up(self.global_down(label))) + self.local(label))
        )


def build_pix2pixhd_generator() -> nn.Module:
    """Build compact pix2pixHD generator."""

    return Pix2PixHDGenerator()


def example_label() -> torch.Tensor:
    """Return a semantic label map."""

    return torch.randn(1, 8, 32, 32)


class SPADELayer(nn.Module):
    """Spatially-adaptive denormalization layer."""

    def __init__(self, channels: int, seg_channels: int) -> None:
        """Initialize SPADE modulation.

        Parameters
        ----------
        channels:
            Feature channels.
        seg_channels:
            Segmentation-map channels.
        """

        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.shared = nn.Conv2d(seg_channels, 16, 3, padding=1)
        self.gamma = nn.Conv2d(16, channels, 3, padding=1)
        self.beta = nn.Conv2d(16, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """Apply per-pixel semantic modulation.

        Parameters
        ----------
        x:
            Feature map.
        seg:
            Semantic map.

        Returns
        -------
        torch.Tensor
            Modulated feature map.
        """

        seg = F.interpolate(seg, size=x.shape[-2:], mode="nearest")
        h = F.relu(self.shared(seg))
        return self.norm(x) * (1 + self.gamma(h)) + self.beta(h)


class SPADEResBlock(nn.Module):
    """SPADE residual block."""

    def __init__(self, channels: int, seg_channels: int) -> None:
        """Initialize the SPADE block.

        Parameters
        ----------
        channels:
            Feature channels.
        seg_channels:
            Segmentation channels.
        """

        super().__init__()
        self.s1 = SPADELayer(channels, seg_channels)
        self.s2 = SPADELayer(channels, seg_channels)
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """Run a SPADE residual update.

        Parameters
        ----------
        x:
            Feature map.
        seg:
            Semantic map.

        Returns
        -------
        torch.Tensor
            Updated feature map.
        """

        y = self.c1(F.relu(self.s1(x, seg)))
        y = self.c2(F.relu(self.s2(y, seg)))
        return x + y


class ImaginaireSPADEGenerator(nn.Module):
    """Imaginaire SPADE-style generator with semantic gamma/beta at each block."""

    def __init__(self, seg_channels: int = 8, channels: int = 24) -> None:
        """Initialize compact SPADE generator.

        Parameters
        ----------
        seg_channels:
            Semantic channels.
        channels:
            Base feature width.
        """

        super().__init__()
        self.stem = nn.Conv2d(seg_channels, channels, 3, padding=1)
        self.block1 = SPADEResBlock(channels, seg_channels)
        self.block2 = SPADEResBlock(channels, seg_channels)
        self.to_rgb = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, seg: torch.Tensor) -> torch.Tensor:
        """Generate an RGB image from segmentation.

        Parameters
        ----------
        seg:
            Semantic tensor.

        Returns
        -------
        torch.Tensor
            RGB image.
        """

        x = F.interpolate(F.relu(self.stem(seg)), scale_factor=2, mode="nearest")
        x = self.block1(x, seg)
        x = self.block2(x, seg)
        return torch.tanh(self.to_rgb(x))


def build_spade_generator() -> nn.Module:
    """Build compact Imaginaire SPADE generator."""

    return ImaginaireSPADEGenerator()


class SsdHead(nn.Module):
    """SSD multibox detector head over multiple feature maps."""

    def __init__(self, channels: list[int], anchors: int = 3, classes: int = 5) -> None:
        """Initialize classification and box towers.

        Parameters
        ----------
        channels:
            Feature-map channel widths.
        anchors:
            Anchors per location.
        classes:
            Number of object classes.
        """

        super().__init__()
        self.cls = nn.ModuleList(
            [nn.Conv2d(ch, anchors * classes, 3, padding=1) for ch in channels]
        )
        self.box = nn.ModuleList([nn.Conv2d(ch, anchors * 4, 3, padding=1) for ch in channels])

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Predict multiscale class and box maps.

        Parameters
        ----------
        features:
            Pyramid feature maps.

        Returns
        -------
        torch.Tensor
            Concatenated SSD predictions.
        """

        outs = []
        for feat, cls, box in zip(features, self.cls, self.box, strict=True):
            outs.extend([cls(feat).flatten(1), box(feat).flatten(1)])
        return torch.cat(outs, dim=1)


class SSDDetector(nn.Module):
    """SSD-style detector with progressively smaller feature maps."""

    def __init__(self) -> None:
        """Initialize compact SSD."""

        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(32, 48, 3, stride=2, padding=1)
        self.extra = nn.Conv2d(48, 48, 3, stride=2, padding=1)
        self.head = SsdHead([32, 48, 48])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict SSD multibox outputs.

        Parameters
        ----------
        image:
            Input image.

        Returns
        -------
        torch.Tensor
            Flat prediction vector.
        """

        x1 = F.relu(self.c1(image))
        x2 = F.relu(self.c2(x1))
        x3 = F.relu(self.c3(x2))
        x4 = F.relu(self.extra(x3))
        return self.head([x2, x3, x4])


def build_ssd() -> nn.Module:
    """Build compact SSD detector."""

    return SSDDetector()


def example_image() -> torch.Tensor:
    """Return a small RGB image."""

    return torch.randn(1, 3, 32, 32)


class TOODDetector(nn.Module):
    """TOOD detector with task-aligned head and alignment predictor."""

    def __init__(self, classes: int = 5, channels: int = 32) -> None:
        """Initialize TOOD compact head.

        Parameters
        ----------
        classes:
            Number of object classes.
        channels:
            Feature width.
        """

        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.interactive = nn.Sequential(ResidualBlock(channels), ResidualBlock(channels))
        self.cls_prob = nn.Conv2d(channels, classes, 3, padding=1)
        self.reg = nn.Conv2d(channels, 4, 3, padding=1)
        self.align = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict task-aligned classification and localization maps.

        Parameters
        ----------
        image:
            Input image.

        Returns
        -------
        torch.Tensor
            Concatenated aligned predictions.
        """

        feat = self.interactive(self.backbone(image))
        alignment = torch.sigmoid(self.align(feat))
        cls = torch.sigmoid(self.cls_prob(feat)) * alignment
        box = F.relu(self.reg(feat)) * alignment
        return torch.cat([cls, box, alignment], dim=1)


def build_tood() -> nn.Module:
    """Build compact TOOD detector."""

    return TOODDetector()


class YOLOXDetector(nn.Module):
    """YOLOX decoupled-head anchor-free detector."""

    def __init__(self, classes: int = 5, channels: int = 32) -> None:
        """Initialize YOLOX compact network.

        Parameters
        ----------
        classes:
            Number of classes.
        channels:
            Base channels.
        """

        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, stride=2, padding=1)
        self.dark = nn.Sequential(
            ResidualBlock(channels), nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)
        )
        self.cls_tower = nn.Sequential(
            nn.ReLU(), nn.Conv2d(channels * 2, channels, 3, padding=1), nn.ReLU()
        )
        self.reg_tower = nn.Sequential(
            nn.ReLU(), nn.Conv2d(channels * 2, channels, 3, padding=1), nn.ReLU()
        )
        self.cls = nn.Conv2d(channels, classes, 1)
        self.obj = nn.Conv2d(channels, 1, 1)
        self.box = nn.Conv2d(channels, 4, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict YOLOX class, objectness, and box maps.

        Parameters
        ----------
        image:
            Input image.

        Returns
        -------
        torch.Tensor
            Anchor-free prediction maps.
        """

        feat = F.relu(self.dark(F.relu(self.stem(image))))
        cls_feat = self.cls_tower(feat)
        reg_feat = self.reg_tower(feat)
        return torch.cat(
            [self.cls(cls_feat), self.obj(reg_feat), F.relu(self.box(reg_feat))], dim=1
        )


def build_yolox() -> nn.Module:
    """Build compact YOLOX detector."""

    return YOLOXDetector()


class BasicVSRPlusPlus(nn.Module):
    """BasicVSR++ with bidirectional second-order propagation and deformable-style alignment."""

    def __init__(self, channels: int = 16) -> None:
        """Initialize compact BasicVSR++.

        Parameters
        ----------
        channels:
            Feature width.
        """

        super().__init__()
        self.feat = nn.Conv2d(3, channels, 3, padding=1)
        self.offset = nn.Conv2d(channels * 3 + 3, 2, 3, padding=1)
        self.trunk = ResidualBlock(channels)
        self.fuse = nn.Conv2d(channels * 2, channels, 1)
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    @staticmethod
    def _grid(x: torch.Tensor) -> torch.Tensor:
        """Create a normalized sampling grid.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Grid for ``grid_sample``.
        """

        batch, _, height, width = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=x.device),
            torch.linspace(-1.0, 1.0, width, device=x.device),
            indexing="ij",
        )
        return torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch, height, width, 2)

    def _align(
        self, now: torch.Tensor, one: torch.Tensor, two: torch.Tensor, frame: torch.Tensor
    ) -> torch.Tensor:
        """Align first- and second-order propagated states.

        Parameters
        ----------
        now:
            Current features.
        one:
            First-order state.
        two:
            Second-order state.
        frame:
            Current image frame.

        Returns
        -------
        torch.Tensor
            Aligned propagated feature.
        """

        flow = torch.tanh(self.offset(torch.cat([now, one, two, frame], dim=1))) * 0.2
        warped = F.grid_sample(
            one + 0.5 * two, self._grid(now) + flow.permute(0, 2, 3, 1), align_corners=True
        )
        return warped

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Super-resolve a short video.

        Parameters
        ----------
        video:
            Low-resolution video shaped ``(batch, time, 3, height, width)``.

        Returns
        -------
        torch.Tensor
            Super-resolved frames.
        """

        batch, time, channels, height, width = video.shape
        feats = self.feat(video.reshape(batch * time, channels, height, width)).view(
            batch, time, -1, height, width
        )
        backward: list[torch.Tensor] = []
        one = torch.zeros_like(feats[:, 0])
        two = torch.zeros_like(one)
        for idx in range(time - 1, -1, -1):
            aligned = self._align(feats[:, idx], one, two, video[:, idx])
            two = one
            one = self.trunk(feats[:, idx] + aligned)
            backward.append(one)
        backward = list(reversed(backward))
        one = torch.zeros_like(feats[:, 0])
        two = torch.zeros_like(one)
        outs = []
        for idx in range(time):
            aligned = self._align(feats[:, idx], one, two, video[:, idx])
            two = one
            one = self.trunk(feats[:, idx] + aligned)
            outs.append(self.up(self.fuse(torch.cat([one, backward[idx]], dim=1))))
        return torch.stack(outs, dim=1)


def build_basicvsrplusplus_x4() -> nn.Module:
    """Build compact BasicVSR++."""

    return BasicVSRPlusPlus()


def example_video() -> torch.Tensor:
    """Return a short low-resolution video."""

    return torch.randn(1, 3, 3, 16, 16)


class CRNNRecognizer(nn.Module):
    """CRNN text recognizer with CNN sequence features, BiLSTM, and CTC logits."""

    def __init__(self, alphabet: int = 32, channels: int = 24) -> None:
        """Initialize CRNN.

        Parameters
        ----------
        alphabet:
            Output alphabet size including blank.
        channels:
            Base channels.
        """

        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.rnn = nn.LSTM(
            channels * 2, channels, num_layers=1, bidirectional=True, batch_first=True
        )
        self.ctc = nn.Linear(channels * 2, alphabet)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return per-column CTC logits.

        Parameters
        ----------
        image:
            Text-line image.

        Returns
        -------
        torch.Tensor
            CTC logits shaped ``(batch, width, alphabet)``.
        """

        feat = self.cnn(image).squeeze(2).transpose(1, 2)
        seq, _ = self.rnn(feat)
        return self.ctc(seq)


def build_crnn() -> nn.Module:
    """Build compact CRNN recognizer."""

    return CRNNRecognizer()


def example_text_image() -> torch.Tensor:
    """Return a grayscale word image."""

    return torch.randn(1, 1, 32, 96)


class IbcEnergy(nn.Module):
    """Implicit Behavioral Cloning energy model."""

    def __init__(self, state_dim: int = 10, action_dim: int = 4, conv: bool = False) -> None:
        """Initialize an IBC EBM.

        Parameters
        ----------
        state_dim:
            State vector width.
        action_dim:
            Action width.
        conv:
            Whether to use image-conditioned ConvMLP features.
        """

        super().__init__()
        self.conv = conv
        self.encoder = (
            nn.Sequential(nn.Conv2d(3, 12, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
            if conv
            else None
        )
        in_dim = (12 if conv else state_dim) + action_dim
        self.energy = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Score candidate actions with negative energies.

        Parameters
        ----------
        state:
            State vector or image.
        actions:
            Candidate action tensor shaped ``(batch, candidates, action_dim)``.

        Returns
        -------
        torch.Tensor
            Energy scores per candidate.
        """

        if self.conv:
            if self.encoder is None:
                raise ValueError("Conv encoder is missing.")
            state_vec = self.encoder(state).flatten(1)
        else:
            state_vec = state
        tiled = state_vec.unsqueeze(1).expand(-1, actions.shape[1], -1)
        return self.energy(torch.cat([tiled, actions], dim=-1)).squeeze(-1)


def build_ibc_mlp_ebm() -> nn.Module:
    """Build vector-state IBC EBM."""

    return IbcEnergy(conv=False)


def example_ibc_mlp() -> tuple[torch.Tensor, torch.Tensor]:
    """Return vector state and candidate actions."""

    return torch.randn(1, 10), torch.randn(1, 5, 4)


def build_ibc_convmlp_ebm() -> nn.Module:
    """Build image-state IBC ConvMLP EBM."""

    return IbcEnergy(conv=True)


def example_ibc_conv() -> tuple[torch.Tensor, torch.Tensor]:
    """Return image state and candidate actions."""

    return torch.randn(1, 3, 24, 24), torch.randn(1, 5, 4)


def build_spikformer_imagenet() -> nn.Module:
    """Build a compact ImageNet-style Spikformer with SSA spiking attention."""

    from menagerie.classics.spikformer import Spikformer

    return Spikformer(
        embed_dim=48, depth=2, num_heads=4, mlp_ratio=4, num_classes=1000, in_ch=3, timesteps=2
    )


def example_spikformer_imagenet() -> torch.Tensor:
    """Return a compact ImageNet-like RGB crop."""

    return torch.randn(1, 3, 32, 32)


def build_mmdit_layer() -> nn.Module:
    """Build a compact MMDiT layer stack with separate text/image weights and joint attention."""

    from menagerie.classics.multimodal_dit import CompactMMDiT

    return CompactMMDiT(dim=64, depth=1, heads=4)


def example_mmdit_layer() -> torch.Tensor:
    """Return text-token conditioning for MMDiT."""

    return torch.randn(1, 8, 64)


def build_siamese_rpn() -> nn.Module:
    """Build a compact SiamRPN tracker with depthwise cross-correlation."""

    from menagerie.classics.siamese_trackers import build_siamrpn

    return build_siamrpn()


def example_siamese_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Return template and search crops for Siamese tracking."""

    return torch.randn(1, 3, 31, 31), torch.randn(1, 3, 63, 63)


class DiscriminativeFilterTracker(nn.Module):
    """ATOM/DiMP/PrDiMP-style tracker using learned target filters and IoU refinement."""

    def __init__(self, probabilistic: bool = False) -> None:
        """Initialize the discriminative-filter tracker.

        Parameters
        ----------
        probabilistic:
            Whether to return variance-like uncertainty as in PrDiMP.
        """

        super().__init__()
        self.probabilistic = probabilistic
        self.template = nn.Sequential(nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(), ResidualBlock(24))
        self.search = nn.Sequential(nn.Conv2d(3, 24, 3, padding=1), nn.ReLU(), ResidualBlock(24))
        self.filter_gen = nn.Conv2d(24, 24, 1)
        self.iou_head = nn.Sequential(
            nn.Conv2d(48, 24, 3, padding=1), nn.ReLU(), nn.Conv2d(24, 5, 1)
        )

    def forward(self, template: torch.Tensor, search: torch.Tensor) -> torch.Tensor:
        """Score target location and refine boxes.

        Parameters
        ----------
        template:
            Target template crop.
        search:
            Search-region crop.

        Returns
        -------
        torch.Tensor
            Response, box refinement, and optional uncertainty maps.
        """

        z = self.template(template)
        x = self.search(search)
        filt = self.filter_gen(F.adaptive_avg_pool2d(z, 1))
        response = (x * filt).sum(dim=1, keepdim=True)
        z_up = F.interpolate(z, size=x.shape[-2:], mode="bilinear", align_corners=False)
        refine = self.iou_head(torch.cat([x, z_up], dim=1))
        if self.probabilistic:
            variance = F.softplus(refine[:, :1])
            return torch.cat([response, refine, variance], dim=1)
        return torch.cat([response, refine], dim=1)


def build_pytracking_atom() -> nn.Module:
    """Build compact ATOM tracker with discriminative classification and IoU head."""

    return DiscriminativeFilterTracker(probabilistic=False)


def build_pytracking_dimp() -> nn.Module:
    """Build compact DiMP tracker with learned discriminative target filter."""

    return DiscriminativeFilterTracker(probabilistic=False)


def build_pytracking_prdimp() -> nn.Module:
    """Build compact PrDiMP tracker with probabilistic uncertainty output."""

    return DiscriminativeFilterTracker(probabilistic=True)


class ByteTrackCompact(nn.Module):
    """ByteTrack-style MOT module using high/low confidence detection association."""

    def __init__(self) -> None:
        """Initialize detector and association heads."""

        super().__init__()
        self.detector = YOLOXDetector(classes=4, channels=24)
        self.embed = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=4, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.track_proj = nn.Linear(16, 8)

    def forward(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
        """Return detections and a two-stage association score.

        Parameters
        ----------
        frame_a:
            Previous frame.
        frame_b:
            Current frame.

        Returns
        -------
        torch.Tensor
            Detector predictions and high/low confidence association scores.
        """

        det = self.detector(frame_b).flatten(1)
        ea = F.normalize(self.track_proj(self.embed(frame_a).flatten(1)), dim=-1)
        eb = F.normalize(self.track_proj(self.embed(frame_b).flatten(1)), dim=-1)
        high_assoc = (ea * eb).sum(-1, keepdim=True)
        low_assoc = torch.sigmoid(det.mean(1, keepdim=True)) * high_assoc
        return torch.cat([det, high_assoc, low_assoc], dim=1)


def build_bytetrack() -> nn.Module:
    """Build compact ByteTrack detector-association model."""

    return ByteTrackCompact()


def example_two_frames() -> tuple[torch.Tensor, torch.Tensor]:
    """Return two RGB video frames."""

    return torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)


class TemporalFeatureTracker(nn.Module):
    """FGFA/DFF/SELSA-style temporal feature aggregation tracker."""

    def __init__(self, mode: str = "fgfa") -> None:
        """Initialize temporal aggregation.

        Parameters
        ----------
        mode:
            Aggregation mode: flow-guided, flow-warped, or sequence attention.
        """

        super().__init__()
        self.mode = mode
        self.feat = nn.Conv2d(3, 24, 3, padding=1)
        self.flow = nn.Conv2d(48, 2, 3, padding=1)
        self.attn = nn.Conv2d(48, 1, 1)
        self.head = nn.Conv2d(24, 6, 1)

    @staticmethod
    def _grid(x: torch.Tensor) -> torch.Tensor:
        """Create a normalized grid.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Sampling grid.
        """

        batch, _, height, width = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=x.device),
            torch.linspace(-1.0, 1.0, width, device=x.device),
            indexing="ij",
        )
        return torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch, height, width, 2)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Aggregate adjacent-frame features for detection/tracking.

        Parameters
        ----------
        frames:
            Tensor shaped ``(batch, time, 3, height, width)``.

        Returns
        -------
        torch.Tensor
            Detection logits on aggregated features.
        """

        batch, time, channels, height, width = frames.shape
        feats = F.relu(self.feat(frames.reshape(batch * time, channels, height, width))).view(
            batch, time, 24, height, width
        )
        ref = feats[:, time // 2]
        aggs = []
        weights = []
        for idx in range(time):
            pair = torch.cat([ref, feats[:, idx]], dim=1)
            flow = torch.tanh(self.flow(pair)) * 0.2
            warped = F.grid_sample(
                feats[:, idx], self._grid(ref) + flow.permute(0, 2, 3, 1), align_corners=True
            )
            aggs.append(warped)
            weights.append(self.attn(torch.cat([ref, warped], dim=1)))
        stacked = torch.stack(aggs, dim=1)
        if self.mode == "selsa":
            alpha = torch.softmax(torch.stack(weights, dim=1), dim=1)
            agg = (alpha * stacked).sum(1)
        elif self.mode == "dff":
            agg = 0.7 * ref + 0.3 * stacked.mean(1)
        else:
            agg = stacked.mean(1)
        return self.head(agg)


def build_temporal_fgfa() -> nn.Module:
    """Build compact FGFA tracker with flow-guided feature aggregation."""

    return TemporalFeatureTracker("fgfa")


def build_temporal_dff() -> nn.Module:
    """Build compact DFF tracker with flow-warped feature propagation."""

    return TemporalFeatureTracker("dff")


def build_temporal_selsa() -> nn.Module:
    """Build compact SELSA tracker with sequence-level semantic attention."""

    return TemporalFeatureTracker("selsa")


def example_clip() -> torch.Tensor:
    """Return a short RGB clip."""

    return torch.randn(1, 3, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("IgFold", "build_igfold", "example_igfold", "2023", "DC"),
    (
        "ImageBind-JointEmbedding",
        "build_imagebind_jointembedding",
        "example_imagebind",
        "2023",
        "DC",
    ),
    (
        "spikingformer_imagenet",
        "build_spikformer_imagenet",
        "example_spikformer_imagenet",
        "2023",
        "DC",
    ),
    (
        "spikformer_imagenet",
        "build_spikformer_imagenet",
        "example_spikformer_imagenet",
        "2023",
        "DC",
    ),
    ("coco_funit_generator", "build_funit_generator", "example_funit", "2020", "DC"),
    ("funit_generator", "build_funit_generator", "example_funit", "2019", "DC"),
    ("imaginaire_pix2pixhd_generator", "build_pix2pixhd_generator", "example_label", "2018", "DC"),
    ("imaginaire_spade_generator", "build_spade_generator", "example_label", "2019", "DC"),
    ("paddledet_ssd", "build_ssd", "example_image", "2016", "DC"),
    ("paddledet_tood", "build_tood", "example_image", "2021", "DC"),
    ("paddledet_yolox", "build_yolox", "example_image", "2021", "DC"),
    ("basicvsrplusplus_x4", "build_basicvsrplusplus_x4", "example_video", "2022", "DC"),
    ("CRNN-VGG-BiLSTM-CTC", "build_crnn", "example_text_image", "2015", "DC"),
    ("CRNN-ResNet31-CTC-MMOCR", "build_crnn", "example_text_image", "2015", "DC"),
    ("IBC_MLP_EBM", "build_ibc_mlp_ebm", "example_ibc_mlp", "2021", "DC"),
    ("IBC_ConvMLP_EBM", "build_ibc_convmlp_ebm", "example_ibc_conv", "2021", "DC"),
    ("mmdit_layer", "build_mmdit_layer", "example_mmdit_layer", "2024", "DC"),
    ("mmtrack:siamese_rpn", "build_siamese_rpn", "example_siamese_pair", "2018", "DC"),
    ("PyTracking-ATOM", "build_pytracking_atom", "example_siamese_pair", "2019", "DC"),
    ("PyTracking-DiMP", "build_pytracking_dimp", "example_siamese_pair", "2019", "DC"),
    ("PyTracking-PrDiMP", "build_pytracking_prdimp", "example_siamese_pair", "2020", "DC"),
    ("mmtrack:bytetrack", "build_bytetrack", "example_two_frames", "2021", "DC"),
    ("mmtrack:fgfa", "build_temporal_fgfa", "example_clip", "2017", "DC"),
    ("mmtrack:dff", "build_temporal_dff", "example_clip", "2017", "DC"),
    ("mmtrack:selsa", "build_temporal_selsa", "example_clip", "2019", "DC"),
]
