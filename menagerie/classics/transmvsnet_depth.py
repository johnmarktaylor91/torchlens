"""TransMVSNet: Global Context-Aware Multi-View Stereo with Transformers.

Ding et al., CVPR 2022.  TransMVSNet reframes MVS as feature matching and adds
a Feature Matching Transformer (FMT): intra-image self-attention and inter-image
cross-attention aggregate long-range context before pair-wise feature correlation
and depth regression.  The paper also uses an adaptive receptive field bridge
between CNN features and transformer matching.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Convolutional feature block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize the feature block.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.
        stride:
            Convolution stride.
        """

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and ReLU.

        Parameters
        ----------
        x:
            Image feature tensor.

        Returns
        -------
        torch.Tensor
            Transformed feature tensor.
        """

        return F.relu(self.norm(self.conv(x)), inplace=True)


class _AdaptiveReceptiveField(nn.Module):
    """Dilated multi-branch bridge used before transformer matching."""

    def __init__(self, channels: int) -> None:
        """Initialize the ARF module.

        Parameters
        ----------
        channels:
            Feature width.
        """

        super().__init__()
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.wide = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels)
        self.mix = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Blend local and wider receptive fields.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        torch.Tensor
            Receptive-field adapted feature map.
        """

        return self.mix(torch.cat([self.local(x), self.wide(x)], dim=1))


class _FeatureNet(nn.Module):
    """Small CNN feature pyramid stem with ARF bridge."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize the feature net.

        Parameters
        ----------
        channels:
            Output channel count.
        """

        super().__init__()
        self.net = nn.Sequential(
            _ConvBlock(3, channels // 2),
            _ConvBlock(channels // 2, channels, stride=2),
            _ConvBlock(channels, channels),
            _AdaptiveReceptiveField(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract adapted image features.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        torch.Tensor
            Feature map at half resolution.
        """

        return self.net(x)


class _FMTBlock(nn.Module):
    """Feature Matching Transformer self-attention plus cross-attention."""

    def __init__(self, channels: int, num_heads: int = 3) -> None:
        """Initialize the FMT block.

        Parameters
        ----------
        channels:
            Token width.
        num_heads:
            Attention head count.
        """

        super().__init__()
        self.self_norm = nn.LayerNorm(channels)
        self.self_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(channels)
        self.cross_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels)
        )

    def forward(self, ref: torch.Tensor, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run bidirectional self/cross feature matching.

        Parameters
        ----------
        ref:
            Reference-view tokens.
        src:
            Source-view tokens.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated reference and source tokens.
        """

        ref = ref + self.self_attn(self.self_norm(ref), self.self_norm(ref), self.self_norm(ref))[0]
        src = src + self.self_attn(self.self_norm(src), self.self_norm(src), self.self_norm(src))[0]
        ref = (
            ref
            + self.cross_attn(self.cross_norm(ref), self.cross_norm(src), self.cross_norm(src))[0]
        )
        src = (
            src
            + self.cross_attn(self.cross_norm(src), self.cross_norm(ref), self.cross_norm(ref))[0]
        )
        ref = ref + self.ff(self.ff_norm(ref))
        src = src + self.ff(self.ff_norm(src))
        return ref, src


class TransMVSNetDepth(nn.Module):
    """Compact TransMVSNet depth estimator."""

    def __init__(self, channels: int = 24, num_depths: int = 8) -> None:
        """Initialize TransMVSNet compact core.

        Parameters
        ----------
        channels:
            Feature and token width.
        num_depths:
            Number of plane-sweep depth hypotheses.
        """

        super().__init__()
        self.num_depths = num_depths
        self.feature = _FeatureNet(channels)
        self.fmt = nn.ModuleList([_FMTBlock(channels) for _ in range(2)])
        self.regularizer = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, 3, padding=1),
        )

    def _tokens_to_map(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Reshape tokens back to a feature map.

        Parameters
        ----------
        tokens:
            Token tensor ``(B, H*W, C)``.
        height:
            Feature-map height.
        width:
            Feature-map width.

        Returns
        -------
        torch.Tensor
            Feature map ``(B, C, H, W)``.
        """

        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)

    def _correlation_volume(self, ref: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """Build a shift-proxy pair-wise correlation volume.

        Parameters
        ----------
        ref:
            Reference features.
        src:
            Source features.

        Returns
        -------
        torch.Tensor
            Correlation cost volume ``(B, D, H, W)``.
        """

        costs = []
        for depth_index in range(self.num_depths):
            if depth_index == 0:
                shifted = src
            else:
                shifted = torch.zeros_like(src)
                shifted[:, :, :, depth_index:] = src[:, :, :, :-depth_index]
            costs.append((ref * shifted).mean(dim=1))
        return torch.stack(costs, dim=1)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Estimate depth from reference plus two source views.

        Parameters
        ----------
        imgs:
            Concatenated views ``(B, 9, H, W)``.

        Returns
        -------
        torch.Tensor
            Depth map ``(B, 1, H, W)``.
        """

        height, width = imgs.shape[2], imgs.shape[3]
        ref = self.feature(imgs[:, :3])
        src_a = self.feature(imgs[:, 3:6])
        src_b = self.feature(imgs[:, 6:9])
        feat_h, feat_w = ref.shape[2], ref.shape[3]
        ref_tokens = ref.flatten(2).transpose(1, 2)
        source_tokens = ((src_a + src_b) * 0.5).flatten(2).transpose(1, 2)
        for block in self.fmt:
            ref_tokens, source_tokens = block(ref_tokens, source_tokens)
        ref_matched = self._tokens_to_map(ref_tokens, feat_h, feat_w)
        src_matched = self._tokens_to_map(source_tokens, feat_h, feat_w)
        cost = self._correlation_volume(ref_matched, src_matched).unsqueeze(1)
        logits = self.regularizer(cost).squeeze(1)
        weights = torch.softmax(logits, dim=1)
        hypotheses = torch.arange(1, self.num_depths + 1, dtype=imgs.dtype, device=imgs.device)
        depth = (weights * hypotheses.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        return F.interpolate(depth, (height, width), mode="bilinear", align_corners=False)


def build_transmvsnet_depth() -> nn.Module:
    """Build compact TransMVSNet.

    Returns
    -------
    nn.Module
        Random-init TransMVSNet depth estimator.
    """

    return TransMVSNetDepth()


def example_input() -> torch.Tensor:
    """Return a small three-view MVS input.

    Returns
    -------
    torch.Tensor
        Tensor ``(1, 9, 32, 32)``.
    """

    return torch.randn(1, 9, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "TransMVSNet (FMT self/cross-attention MVS depth)",
        "build_transmvsnet_depth",
        "example_input",
        "2022",
        "DC",
    ),
]
