"""Dependency-gated OpenMMLab/Paddle-style classics for batch 6.

Paper: OpenMMLab model zoos, PaddleDetection/PaddleSeg/PaddleOCR model zoos, and
the cited architecture papers for each target family.

This module reconstructs install-hostile catalog targets in plain PyTorch.  It
uses compact random-initialized networks but keeps the distinctive architecture:
ST-GCN graph-temporal skeleton blocks, StyleGAN modulated synthesis, Swin/TimeSformer
video attention, text-detector FPN heads, CTC/attention recognizers, top-down pose
heatmaps/SimCC heads, and alignment-based video super-resolution.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from menagerie.classics._mmocr_shared import (
    CRNNSTNResNet31,
    MASTERRecognizer,
    NRTRRecognizer,
    RobustScannerRecognizer,
    SARRecognizer,
    SATRNRecognizer,
    ResNet31Backbone,
    SequenceAttentionDecoder,
    ThinPlateSplineLite,
    text_image,
)
from menagerie.classics.dbnet_mobilenetv3 import (
    build_abinet_vision_language,
    build_dbnet_resnet50,
    build_textsnake_unet,
    example_det_input,
    example_rec_input,
)
from menagerie.classics.stgcn_skeleton import STGCN, STGCNBlock, example_input_skeleton
from menagerie.classics.stylegan2 import StyleGAN2Generator
from menagerie.classics.stylegan3 import StyleGAN3Generator


class ConvBNAct(nn.Module):
    """Convolution, batch normalization, and ReLU helper."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, groups: int = 1) -> None:
        """Initialize a convolutional block.

        Parameters
        ----------
        in_ch:
            Input channels.
        out_ch:
            Output channels.
        stride:
            Spatial stride.
        groups:
            Convolution groups.
        """

        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        """Apply convolution, normalization, and activation.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Activated feature map.
        """

        return F.relu(self.bn(self.conv(x)))


class TinyFPNBackbone(nn.Module):
    """Compact CNN backbone with a top-down FPN neck."""

    def __init__(self, width: int = 24) -> None:
        """Initialize the backbone and lateral FPN projections.

        Parameters
        ----------
        width:
            Base channel width.
        """

        super().__init__()
        self.s1 = ConvBNAct(3, width)
        self.s2 = ConvBNAct(width, width * 2, stride=2)
        self.s3 = ConvBNAct(width * 2, width * 4, stride=2)
        self.s4 = ConvBNAct(width * 4, width * 4, stride=2)
        self.lat2 = nn.Conv2d(width * 2, width, 1)
        self.lat3 = nn.Conv2d(width * 4, width, 1)
        self.lat4 = nn.Conv2d(width * 4, width, 1)
        self.out2 = ConvBNAct(width, width)
        self.out3 = ConvBNAct(width, width)
        self.out4 = ConvBNAct(width, width)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Extract three FPN feature levels.

        Parameters
        ----------
        x:
            Input RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            FPN maps at progressively lower resolution.
        """

        c1 = self.s1(x)
        c2 = self.s2(c1)
        c3 = self.s3(c2)
        c4 = self.s4(c3)
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        return self.out2(p2), self.out3(p3), self.out4(p4)


class FourierContourHead(nn.Module):
    """FCENet-style Fourier contour text detector."""

    def __init__(self, channels: int = 24, order: int = 5) -> None:
        """Initialize classification and Fourier coefficient heads.

        Parameters
        ----------
        channels:
            FPN feature width.
        order:
            Number of Fourier harmonics on either side of zero.
        """

        super().__init__()
        self.backbone = TinyFPNBackbone(channels)
        coeffs = (2 * order + 1) * 2
        self.text_region = nn.Conv2d(channels, 1, 1)
        self.center_region = nn.Conv2d(channels, 1, 1)
        self.coeffs = nn.Conv2d(channels, coeffs, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict FCENet text maps and Fourier contour coefficients.

        Parameters
        ----------
        image:
            RGB text image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Text score, center score, and Fourier coefficients.
        """

        p2, _, _ = self.backbone(image)
        return (
            torch.sigmoid(self.text_region(p2)),
            torch.sigmoid(self.center_region(p2)),
            self.coeffs(p2),
        )


class PSEHead(nn.Module):
    """PSENet progressive-scale-expansion text detector."""

    def __init__(self, channels: int = 24, kernels: int = 6) -> None:
        """Initialize the FPN and multi-kernel segmentation head.

        Parameters
        ----------
        channels:
            FPN feature width.
        kernels:
            Number of nested text kernels.
        """

        super().__init__()
        self.backbone = TinyFPNBackbone(channels)
        self.head = nn.Conv2d(channels * 3, kernels, 1)

    def forward(self, image: Tensor) -> Tensor:
        """Predict nested segmentation kernels for progressive expansion.

        Parameters
        ----------
        image:
            RGB text image.

        Returns
        -------
        Tensor
            Kernel logits.
        """

        feats = self.backbone(image)
        target = feats[0].shape[-2:]
        aligned = [F.interpolate(feat, size=target, mode="nearest") for feat in feats]
        return torch.sigmoid(self.head(torch.cat(aligned, dim=1)))


class PANTextHead(nn.Module):
    """PANet text detector with FPEM-like repeated feature enhancement."""

    def __init__(self, channels: int = 24, repeats: int = 2) -> None:
        """Initialize the PANet feature enhancement and fusion modules.

        Parameters
        ----------
        channels:
            FPN feature width.
        repeats:
            Number of FPEM enhancement passes.
        """

        super().__init__()
        self.backbone = TinyFPNBackbone(channels)
        self.enhancers = nn.ModuleList(
            [ConvBNAct(channels, channels, groups=channels) for _ in range(repeats)]
        )
        self.fuse = nn.Conv2d(channels * 3, channels, 1)
        self.head = nn.Conv2d(channels, 6, 1)

    def forward(self, image: Tensor) -> Tensor:
        """Predict text/kernel/similarity maps after feature enhancement.

        Parameters
        ----------
        image:
            RGB text image.

        Returns
        -------
        Tensor
            PANet text maps.
        """

        feats = list(self.backbone(image))
        for enhancer in self.enhancers:
            feats = [feat + enhancer(feat) for feat in feats]
        target = feats[0].shape[-2:]
        aligned = [F.interpolate(feat, size=target, mode="nearest") for feat in feats]
        return self.head(F.relu(self.fuse(torch.cat(aligned, dim=1))))


class DRRGHead(nn.Module):
    """DRRG text detector with graph reasoning over local text components."""

    def __init__(self, channels: int = 24, nodes: int = 12) -> None:
        """Initialize local geometry prediction and graph reasoning.

        Parameters
        ----------
        channels:
            FPN feature width.
        nodes:
            Number of pooled component nodes.
        """

        super().__init__()
        self.nodes = nodes
        self.backbone = TinyFPNBackbone(channels)
        self.geom = nn.Conv2d(channels, 6, 1)
        self.node_proj = nn.Linear(channels, channels)
        self.edge_proj = nn.Linear(channels * 2, 1)
        self.classifier = nn.Linear(channels, 2)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Predict component geometry and graph edge-aware labels.

        Parameters
        ----------
        image:
            RGB text image.

        Returns
        -------
        tuple[Tensor, Tensor]
            Dense geometry map and per-node graph logits.
        """

        feat, _, _ = self.backbone(image)
        geom = self.geom(feat)
        pooled = F.adaptive_avg_pool2d(feat, (3, 4)).flatten(2).transpose(1, 2)
        nodes = torch.tanh(self.node_proj(pooled[:, : self.nodes]))
        a = nodes.unsqueeze(2).expand(-1, -1, self.nodes, -1)
        b = nodes.unsqueeze(1).expand(-1, self.nodes, -1, -1)
        edge = torch.softmax(self.edge_proj(torch.cat([a, b], dim=-1)).squeeze(-1), dim=-1)
        reasoned = torch.bmm(edge, nodes)
        return geom, self.classifier(reasoned)


class TextMaskRCNN(nn.Module):
    """Mask R-CNN-style text detector with RPN, box, and mask heads."""

    def __init__(self, channels: int = 24, classes: int = 2, proposals: int = 6) -> None:
        """Initialize the two-stage detector.

        Parameters
        ----------
        channels:
            FPN feature width.
        classes:
            Number of detection classes.
        proposals:
            Number of RPN proposals to RoIAlign.
        """

        super().__init__()
        self.proposals = proposals
        self.backbone = TinyFPNBackbone(channels)
        self.rpn_obj = nn.Conv2d(channels, 3, 1)
        self.rpn_box = nn.Conv2d(channels, 12, 1)
        self.box = nn.Linear(channels * 16, classes + 4)
        self.mask = nn.Sequential(
            ConvBNAct(channels, channels),
            nn.ConvTranspose2d(channels, channels, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(channels, classes, 1),
        )

    def _roi_align(self, feature: Tensor, objectness: Tensor, deltas: Tensor) -> Tensor:
        """Sample proposal-aligned RoI crops from RPN-selected locations.

        Parameters
        ----------
        feature:
            FPN feature map.
        objectness:
            RPN objectness logits.
        deltas:
            RPN box deltas.

        Returns
        -------
        Tensor
            RoI-aligned crops shaped ``(B * proposals, C, 4, 4)``.
        """

        batch, _, height, width = feature.shape
        scores = objectness.flatten(2).max(dim=1).values
        top = torch.topk(scores, self.proposals, dim=1).indices
        ys = torch.div(top, width, rounding_mode="floor").to(feature.dtype)
        xs = (top % width).to(feature.dtype)
        delta = deltas.flatten(2).transpose(1, 2)
        gathered = torch.gather(delta, 1, top.unsqueeze(-1).expand(-1, -1, delta.shape[-1]))
        shifts = torch.tanh(gathered[..., :2]) * 0.5
        scales = torch.sigmoid(gathered[..., 2:4]) + 0.5
        centers_x = (xs + 0.5 + shifts[..., 0]) / max(width, 1) * 2.0 - 1.0
        centers_y = (ys + 0.5 + shifts[..., 1]) / max(height, 1) * 2.0 - 1.0
        base_y, base_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 4, device=feature.device, dtype=feature.dtype),
            torch.linspace(-1.0, 1.0, 4, device=feature.device, dtype=feature.dtype),
            indexing="ij",
        )
        half_x = scales[..., 0] / max(width, 1)
        half_y = scales[..., 1] / max(height, 1)
        grid_x = centers_x[..., None, None] + base_x * half_x[..., None, None]
        grid_y = centers_y[..., None, None] + base_y * half_y[..., None, None]
        grid = torch.stack((grid_x, grid_y), dim=-1)
        expanded_feature = feature.unsqueeze(1).expand(-1, self.proposals, -1, -1, -1)
        expanded_feature = expanded_feature.reshape(
            batch * self.proposals, feature.shape[1], height, width
        )
        return F.grid_sample(
            expanded_feature,
            grid.reshape(batch * self.proposals, 4, 4, 2),
            align_corners=False,
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run compact Mask R-CNN text detection.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            RPN objectness, RPN boxes, ROI box logits, and mask logits.
        """

        p2, _, _ = self.backbone(image)
        objectness = self.rpn_obj(p2)
        deltas = self.rpn_box(p2)
        pooled = self._roi_align(p2, objectness, deltas)
        box_logits = self.box(pooled.flatten(1)).reshape(image.shape[0], self.proposals, -1)
        masks = self.mask(pooled).reshape(image.shape[0], self.proposals, -1, 8, 8)
        return objectness, deltas, box_logits, masks


class VideoSwinRecognizer(nn.Module):
    """Video Swin Transformer with 3D patch embedding and shifted-window attention."""

    def __init__(self, dim: int = 48, classes: int = 20) -> None:
        """Initialize compact Video Swin.

        Parameters
        ----------
        dim:
            Token dimension.
        classes:
            Number of action classes.
        """

        super().__init__()
        self.patch = nn.Conv3d(3, dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.local = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.shifted = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.merge = nn.Linear(dim * 4, dim * 2)
        self.norm = nn.LayerNorm(dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim * 2)
        )
        self.head = nn.Linear(dim * 2, classes)

    @staticmethod
    def _window_partition(tokens: Tensor, window: tuple[int, int, int]) -> Tensor:
        """Partition ``(B, T, H, W, C)`` tokens into local 3D windows.

        Parameters
        ----------
        tokens:
            Video tokens.
        window:
            Window shape ``(T, H, W)``.

        Returns
        -------
        Tensor
            Window tokens shaped ``(B * num_windows, window_volume, C)``.
        """

        batch, frames, height, width, channels = tokens.shape
        wt, wh, ww = window
        view = tokens.view(batch, frames // wt, wt, height // wh, wh, width // ww, ww, channels)
        return view.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, wt * wh * ww, channels)

    @staticmethod
    def _window_reverse(
        windows: Tensor, shape: tuple[int, int, int, int, int], window: tuple[int, int, int]
    ) -> Tensor:
        """Reverse local 3D windows back to a video token grid.

        Parameters
        ----------
        windows:
            Window tokens.
        shape:
            Target ``(B, T, H, W, C)`` shape.
        window:
            Window shape ``(T, H, W)``.

        Returns
        -------
        Tensor
            Reconstructed token grid.
        """

        batch, frames, height, width, channels = shape
        wt, wh, ww = window
        view = windows.view(batch, frames // wt, height // wh, width // ww, wt, wh, ww, channels)
        return view.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(batch, frames, height, width, channels)

    def _window_block(self, tokens: Tensor, attn: nn.MultiheadAttention, shift: bool) -> Tensor:
        """Apply one Video Swin local or shifted-window block.

        Parameters
        ----------
        tokens:
            Video token grid.
        attn:
            Attention module for the windows.
        shift:
            Whether to roll the grid by half a spatial window.

        Returns
        -------
        Tensor
            Updated token grid.
        """

        window = (1, 2, 2)
        work = torch.roll(tokens, shifts=(0, -1, -1), dims=(1, 2, 3)) if shift else tokens
        windows = self._window_partition(work, window)
        attended, _ = attn(windows, windows, windows)
        work = self._window_reverse(windows + attended, work.shape, window)
        if shift:
            work = torch.roll(work, shifts=(0, 1, 1), dims=(1, 2, 3))
        return work

    def forward(self, video: Tensor) -> Tensor:
        """Classify a video clip with window and shifted-window attention.

        Parameters
        ----------
        video:
            Video tensor ``(B, C, T, H, W)``.

        Returns
        -------
        Tensor
            Action logits.
        """

        feat = self.patch(video).permute(0, 2, 3, 4, 1)
        tokens = self._window_block(feat, self.local, shift=False)
        tokens = self._window_block(tokens, self.shifted, shift=True)
        batch, frames, height, width, channels = tokens.shape
        merged = tokens.view(batch, frames, height // 2, 2, width // 2, 2, channels)
        merged = merged.permute(0, 1, 2, 4, 3, 5, 6).reshape(
            batch, frames * height * width // 4, channels * 4
        )
        tokens = self.norm(self.merge(merged))
        tokens = tokens + self.mlp(tokens)
        return self.head(tokens.mean(dim=1))


class TimeSformerRecognizer(nn.Module):
    """TimeSformer with selectable divided, joint, or space-only attention."""

    def __init__(self, dim: int = 48, classes: int = 20, mode: str = "divided") -> None:
        """Initialize TimeSformer.

        Parameters
        ----------
        dim:
            Token dimension.
        classes:
            Number of action classes.
        mode:
            ``divided``, ``joint``, or ``space`` attention.
        """

        super().__init__()
        self.mode = mode
        self.patch = nn.Conv2d(3, dim, kernel_size=8, stride=8)
        self.temporal = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.spatial = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.joint = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)

    def forward(self, video: Tensor) -> Tensor:
        """Classify video with temporal attention followed by spatial attention.

        Parameters
        ----------
        video:
            Video tensor ``(B, C, T, H, W)``.

        Returns
        -------
        Tensor
            Action logits.
        """

        batch, channels, frames, height, width = video.shape
        flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        patches = self.patch(flat).flatten(2).transpose(1, 2)
        patches = patches.reshape(batch, frames, patches.shape[1], -1)
        if self.mode == "joint":
            joint_tokens = patches.reshape(batch, frames * patches.shape[2], -1)
            joint, _ = self.joint(joint_tokens, joint_tokens, joint_tokens)
            out = self.norm(joint_tokens + joint)
            return self.head(out.mean(dim=1))
        if self.mode == "space":
            spatial_tokens = patches.reshape(batch * frames, patches.shape[2], -1)
            spatial, _ = self.spatial(spatial_tokens, spatial_tokens, spatial_tokens)
            out = self.norm(spatial.reshape(batch, frames * patches.shape[2], -1))
            return self.head(out.mean(dim=1))
        temporal_tokens = patches.permute(0, 2, 1, 3).reshape(batch * patches.shape[2], frames, -1)
        temporal, _ = self.temporal(temporal_tokens, temporal_tokens, temporal_tokens)
        temporal = temporal.reshape(batch, patches.shape[2], frames, -1).permute(0, 2, 1, 3)
        spatial_tokens = temporal.reshape(batch * frames, patches.shape[2], -1)
        spatial, _ = self.spatial(spatial_tokens, spatial_tokens, spatial_tokens)
        out = self.norm(spatial.reshape(batch, frames * patches.shape[2], -1))
        return self.head(out.mean(dim=1))


class TemporalAdaptiveNet(nn.Module):
    """TANet-style temporal adaptive aggregation over frame features."""

    def __init__(self, classes: int = 20, channels: int = 32) -> None:
        """Initialize the temporal adaptive module.

        Parameters
        ----------
        classes:
            Number of action classes.
        channels:
            Backbone feature width.
        """

        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, channels, stride=2), ConvBNAct(channels, channels, stride=2)
        )
        self.gate = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())
        self.temporal = nn.Conv1d(channels, channels, 3, padding=1, groups=channels)
        self.head = nn.Linear(channels, classes)

    def forward(self, video: Tensor) -> Tensor:
        """Classify video with temporal adaptive frame weighting.

        Parameters
        ----------
        video:
            Video tensor ``(B, C, T, H, W)``.

        Returns
        -------
        Tensor
            Action logits.
        """

        batch, channels, frames, height, width = video.shape
        flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        feat = self.stem(flat).mean(dim=(2, 3)).reshape(batch, frames, -1)
        gate = self.gate(feat.mean(dim=1)).unsqueeze(1)
        feat = feat * gate
        temporal = self.temporal(feat.transpose(1, 2)).transpose(1, 2)
        return self.head(temporal.mean(dim=1))


class TINRecognizer(nn.Module):
    """TIN action recognizer with learned temporal interpolation."""

    def __init__(self, classes: int = 20, channels: int = 32) -> None:
        """Initialize compact TIN.

        Parameters
        ----------
        classes:
            Number of action classes.
        channels:
            Backbone feature width.
        """

        super().__init__()
        self.stem = ConvBNAct(3, channels, stride=2)
        self.mix = ConvBNAct(channels, channels)
        self.offset = nn.Linear(channels, 3)
        self.weight = nn.Linear(channels, 3)
        self.head = nn.Linear(channels, classes)

    def forward(self, video: Tensor) -> Tensor:
        """Classify video using interlaced temporal channel shifts.

        Parameters
        ----------
        video:
            Video tensor ``(B, C, T, H, W)``.

        Returns
        -------
        Tensor
            Action logits.
        """

        batch, channels, frames, height, width = video.shape
        flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        feat = self.stem(flat).reshape(batch, frames, -1, height // 2, width // 2)
        pooled = feat.mean(dim=(3, 4))
        offsets = torch.tanh(self.offset(pooled))
        weights = torch.softmax(self.weight(pooled), dim=-1)
        backward = torch.roll(feat, 1, 1)
        forward = torch.roll(feat, -1, 1)
        interp = (
            backward * weights[..., 0, None, None, None] * (1.0 + offsets[..., 0, None, None, None])
            + feat * weights[..., 1, None, None, None] * (1.0 + offsets[..., 1, None, None, None])
            + forward
            * weights[..., 2, None, None, None]
            * (1.0 + offsets[..., 2, None, None, None])
        )
        mixed = self.mix(interp.reshape(batch * frames, -1, height // 2, width // 2))
        return self.head(mixed.mean(dim=(2, 3)).reshape(batch, frames, -1).mean(dim=1))


class TCANetLocalizer(nn.Module):
    """TCANet localizer with LGTE and progressive boundary regression."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize temporal convolution and boundary heads.

        Parameters
        ----------
        channels:
            Hidden feature width.
        """

        super().__init__()
        self.proj = nn.Conv1d(16, channels, 1)
        self.local = nn.Conv1d(channels, channels, 3, padding=1)
        self.global_ctx = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.regressors = nn.ModuleList([nn.Linear(channels * 2, 2) for _ in range(2)])
        self.start = nn.Conv1d(channels, 1, 1)
        self.end = nn.Conv1d(channels, 1, 1)
        self.score = nn.Linear(channels * 2, 1)

    def forward(self, sequence: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict temporal action boundaries and proposal scores.

        Parameters
        ----------
        sequence:
            Feature sequence ``(B, T, C)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Start, end, proposal scores, and progressive boundary deltas.
        """

        feat = self.proj(sequence.transpose(1, 2))
        local = self.local(feat).transpose(1, 2)
        global_tokens, _ = self.global_ctx(local, local, local)
        ctx = local + global_tokens
        start_feat = ctx.unsqueeze(2).expand(-1, -1, ctx.shape[1], -1)
        end_feat = ctx.unsqueeze(1).expand(-1, ctx.shape[1], -1, -1)
        pair = torch.cat((start_feat, end_feat), dim=-1)
        deltas = torch.zeros(
            pair.shape[0], pair.shape[1], pair.shape[2], 2, device=pair.device, dtype=pair.dtype
        )
        refined = pair
        for regressor in self.regressors:
            delta = regressor(refined)
            deltas = deltas + delta
            refined = refined + delta.mean(dim=-1, keepdim=True).expand_as(refined) * 0.01
        return (
            torch.sigmoid(self.start(ctx.transpose(1, 2))),
            torch.sigmoid(self.end(ctx.transpose(1, 2))),
            torch.sigmoid(self.score(refined).squeeze(-1)),
            deltas,
        )


class TopDownPose(nn.Module):
    """Compact top-down pose estimator with configurable heads."""

    def __init__(self, mode: str = "heatmap", joints: int = 17, channels: int = 32) -> None:
        """Initialize backbone and pose head.

        Parameters
        ----------
        mode:
            ``heatmap``, ``simcc``, ``regression``, ``cpm``, or ``rtm``.
        joints:
            Number of keypoints.
        channels:
            Feature width.
        """

        super().__init__()
        self.mode = mode
        self.stem = nn.Sequential(
            ConvBNAct(3, channels, stride=2), ConvBNAct(channels, channels, stride=2)
        )
        self.context = nn.Sequential(ConvBNAct(channels, channels), ConvBNAct(channels, channels))
        self.deconv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)
        self.heatmap = nn.Conv2d(channels, joints, 1)
        self.regress = nn.Linear(channels, joints * 2)
        self.simcc_x = nn.Linear(channels, joints * 48)
        self.simcc_y = nn.Linear(channels, joints * 64)
        self.gau = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())

    def forward(self, image: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Predict pose heatmaps, SimCC distributions, or coordinates.

        Parameters
        ----------
        image:
            Person crop.

        Returns
        -------
        Tensor | tuple[Tensor, Tensor]
            Pose output for the configured head.
        """

        feat = self.context(self.stem(image))
        up = F.relu(self.deconv(feat))
        if self.mode == "regression":
            return self.regress(feat.mean(dim=(2, 3))).view(image.shape[0], -1, 2)
        if self.mode == "simcc":
            pooled = feat.mean(dim=(2, 3))
            return self.simcc_x(pooled).view(image.shape[0], -1, 48), self.simcc_y(pooled).view(
                image.shape[0], -1, 64
            )
        if self.mode == "rtm":
            gate = self.gau(feat.mean(dim=(2, 3))).view(image.shape[0], -1, 1, 1)
            pooled = (feat * gate).mean(dim=(2, 3))
            return self.simcc_x(pooled).view(image.shape[0], -1, 48), self.simcc_y(pooled).view(
                image.shape[0], -1, 64
            )
        return self.heatmap(up)


class CSPNeXtBlock(nn.Module):
    """CSPNeXt-style cross-stage partial convolution block."""

    def __init__(self, channels: int) -> None:
        """Initialize the partial depthwise block.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.left = ConvBNAct(channels // 2, channels // 2)
        self.right = nn.Sequential(
            ConvBNAct(channels // 2, channels // 2, groups=channels // 2),
            nn.Conv2d(channels // 2, channels // 2, 1),
            nn.ReLU(),
        )
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply CSPNeXt partial mixing.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Mixed feature map.
        """

        left, right = x.chunk(2, dim=1)
        return F.relu(self.fuse(torch.cat((self.left(left), self.right(right)), dim=1)))


class CSPNeXtUDPPose(nn.Module):
    """CSPNeXt top-down heatmap pose with UDP coordinate decoding."""

    def __init__(self, joints: int = 17, channels: int = 32) -> None:
        """Initialize CSPNeXt backbone and UDP heatmap head.

        Parameters
        ----------
        joints:
            Number of keypoints.
        channels:
            Feature width.
        """

        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, channels, stride=2), CSPNeXtBlock(channels), CSPNeXtBlock(channels)
        )
        self.deconv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)
        self.heatmap = nn.Conv2d(channels, joints, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Predict heatmaps and UDP unbiased coordinates.

        Parameters
        ----------
        image:
            Pose crop.

        Returns
        -------
        tuple[Tensor, Tensor]
            Heatmap logits and soft-argmax UDP coordinates.
        """

        heat = self.heatmap(F.relu(self.deconv(self.stem(image))))
        prob = torch.softmax(heat.flatten(2), dim=-1).reshape_as(heat)
        height, width = heat.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=image.device),
            torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=image.device),
            indexing="ij",
        )
        coords = torch.stack(((prob * xx).sum(dim=(2, 3)), (prob * yy).sum(dim=(2, 3))), dim=-1)
        return heat, coords


class IPRPose(nn.Module):
    """Integral pose regression with ResNet-style crop encoder."""

    def __init__(self, joints: int = 17, channels: int = 32) -> None:
        """Initialize IPR heatmap and integral heads.

        Parameters
        ----------
        joints:
            Number of keypoints.
        channels:
            Feature width.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNAct(3, channels, stride=2),
            ConvBNAct(channels, channels, stride=2),
            ConvBNAct(channels, channels),
        )
        self.heatmap = nn.Conv2d(channels, joints, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Return heatmaps and integral coordinates.

        Parameters
        ----------
        image:
            Pose crop.

        Returns
        -------
        tuple[Tensor, Tensor]
            Heatmap logits and integral coordinates.
        """

        heat = self.heatmap(self.encoder(image))
        prob = torch.softmax(heat.flatten(2), dim=-1).reshape_as(heat)
        height, width = heat.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=image.device),
            torch.linspace(0, 1, width, device=image.device),
            indexing="ij",
        )
        return heat, torch.stack(((prob * xx).sum(dim=(2, 3)), (prob * yy).sum(dim=(2, 3))), dim=-1)


class SwinPoseNet(nn.Module):
    """Swin top-down pose estimator with shifted-window backbone."""

    def __init__(self, joints: int = 17, dim: int = 32) -> None:
        """Initialize compact SwinPose.

        Parameters
        ----------
        joints:
            Number of keypoints.
        dim:
            Token dimension.
        """

        super().__init__()
        self.patch = nn.Conv2d(3, dim, 4, stride=4)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.shifted = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.head = nn.ConvTranspose2d(dim, joints, 4, stride=4)

    def _window_attend(self, feat: Tensor, attn: nn.MultiheadAttention, shift: bool) -> Tensor:
        """Apply Swin-style 2D window attention.

        Parameters
        ----------
        feat:
            Feature map.
        attn:
            Attention module.
        shift:
            Whether to use shifted windows.

        Returns
        -------
        Tensor
            Updated feature map.
        """

        if shift:
            feat = torch.roll(feat, shifts=(-1, -1), dims=(2, 3))
        batch, channels, height, width = feat.shape
        tokens = feat.reshape(batch, channels, height // 2, 2, width // 2, 2)
        tokens = tokens.permute(0, 2, 4, 3, 5, 1).reshape(-1, 4, channels)
        attended, _ = attn(tokens, tokens, tokens)
        out = (tokens + attended).reshape(batch, height // 2, width // 2, 2, 2, channels)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, height, width)
        if shift:
            out = torch.roll(out, shifts=(1, 1), dims=(2, 3))
        return out

    def forward(self, image: Tensor) -> Tensor:
        """Predict heatmaps with a shifted-window Swin backbone.

        Parameters
        ----------
        image:
            Pose crop.

        Returns
        -------
        Tensor
            Heatmap logits.
        """

        feat = self.patch(image)
        feat = self._window_attend(feat, self.attn, shift=False)
        feat = self._window_attend(feat, self.shifted, shift=True)
        return self.head(feat)


class MobileNetV2SimCCPose(nn.Module):
    """SimCC pose model with MobileNetV2 inverted residual backbone and no deconv."""

    def __init__(self, joints: int = 17, width: int = 24) -> None:
        """Initialize MobileNetV2 SimCC pose estimator.

        Parameters
        ----------
        joints:
            Number of keypoints.
        width:
            Backbone width.
        """

        super().__init__()
        self.stem = ConvBNAct(3, width, stride=2)
        self.expand = ConvBNAct(width, width * 4)
        self.depthwise = nn.Conv2d(width * 4, width * 4, 3, stride=2, padding=1, groups=width * 4)
        self.project = nn.Conv2d(width * 4, width, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.simcc_x = nn.Linear(width, joints * 48)
        self.simcc_y = nn.Linear(width, joints * 64)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Predict separate x/y SimCC coordinate distributions.

        Parameters
        ----------
        image:
            Person crop.

        Returns
        -------
        tuple[Tensor, Tensor]
            X-axis and y-axis SimCC logits.
        """

        feat = self.stem(image)
        feat = F.relu(self.project(F.relu(self.depthwise(self.expand(feat)))))
        pooled = self.pool(feat).flatten(1)
        return self.simcc_x(pooled).view(image.shape[0], -1, 48), self.simcc_y(pooled).view(
            image.shape[0], -1, 64
        )


class ResNetSimCCPose(nn.Module):
    """SimCC pose model with a compact ResNet bottleneck backbone."""

    def __init__(self, joints: int = 17, width: int = 32) -> None:
        """Initialize ResNet SimCC pose estimator.

        Parameters
        ----------
        joints:
            Number of keypoints.
        width:
            Backbone width.
        """

        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(3, width, stride=2), ConvBNAct(width, width, stride=2))
        self.res1 = ConvBNAct(width, width)
        self.res2 = ConvBNAct(width, width)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.simcc_x = nn.Linear(width, joints * 48)
        self.simcc_y = nn.Linear(width, joints * 64)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Predict separate x/y SimCC coordinate distributions.

        Parameters
        ----------
        image:
            Person crop.

        Returns
        -------
        tuple[Tensor, Tensor]
            X-axis and y-axis SimCC logits.
        """

        feat = self.stem(image)
        feat = F.relu(feat + self.res2(self.res1(feat)))
        pooled = self.pool(feat).flatten(1)
        return self.simcc_x(pooled).view(image.shape[0], -1, 48), self.simcc_y(pooled).view(
            image.shape[0], -1, 64
        )


class ViPNASSimCCPose(nn.Module):
    """ViPNAS-MobileNetV3 SimCC pose model with searched mixed mobile cells."""

    def __init__(self, joints: int = 17, width: int = 24) -> None:
        """Initialize ViPNAS SimCC pose estimator.

        Parameters
        ----------
        joints:
            Number of keypoints.
        width:
            Backbone width.
        """

        super().__init__()
        self.stem = ConvBNAct(3, width, stride=2)
        self.nas_depthwise = nn.ModuleList(
            [
                nn.Conv2d(width, width, 3, stride=2, padding=1, groups=width),
                nn.Conv2d(width, width, 5, padding=2, groups=width),
                nn.Conv2d(width, width, 3, padding=2, dilation=2, groups=width),
            ]
        )
        self.nas_mix = nn.Parameter(torch.tensor([0.45, 0.35, 0.20]))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(width, width, 1), nn.Hardsigmoid()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.simcc_x = nn.Linear(width, joints * 48)
        self.simcc_y = nn.Linear(width, joints * 64)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Predict SimCC distributions after a ViPNAS searched cell.

        Parameters
        ----------
        image:
            Person crop.

        Returns
        -------
        tuple[Tensor, Tensor]
            X-axis and y-axis SimCC logits.
        """

        feat = self.stem(image)
        mixed = 0.0
        weights = torch.softmax(self.nas_mix, dim=0)
        for weight, branch in zip(weights, self.nas_depthwise, strict=True):
            branch_feat = branch(feat)
            if branch_feat.shape[-2:] != feat.shape[-2:]:
                branch_feat = F.interpolate(branch_feat, size=feat.shape[-2:], mode="nearest")
            mixed = mixed + weight * branch_feat
        feat = F.hardswish(feat + mixed * self.se(mixed))
        pooled = self.pool(feat).flatten(1)
        return self.simcc_x(pooled).view(image.shape[0], -1, 48), self.simcc_y(pooled).view(
            image.shape[0], -1, 64
        )


class RTMWPose(nn.Module):
    """RTMW whole-body RTMPose model with GAU and whole-body refinement."""

    def __init__(self, joints: int = 133, width: int = 32) -> None:
        """Initialize RTMW whole-body pose estimator.

        Parameters
        ----------
        joints:
            Whole-body keypoint count.
        width:
            Backbone width.
        """

        super().__init__()
        self.backbone = nn.Sequential(
            ConvBNAct(3, width, stride=2), ConvBNAct(width, width * 2, stride=2)
        )
        self.gau_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(width * 2, width * 2, 1), nn.Sigmoid()
        )
        self.heatmap = nn.Conv2d(width * 2, joints, 1)
        self.simcc_x = nn.Linear(width * 2, joints * 48)
        self.simcc_y = nn.Linear(width * 2, joints * 64)
        self.wholebody_refine = nn.Linear(joints * 112, joints * 2)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict RTMW heatmaps, SimCC logits, and whole-body refinement offsets.

        Parameters
        ----------
        image:
            Person crop.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Heatmaps, x logits, y logits, and whole-body offsets.
        """

        feat = self.backbone(image)
        feat = feat * self.gau_gate(feat)
        pooled = feat.mean(dim=(2, 3))
        sx = self.simcc_x(pooled).view(image.shape[0], -1, 48)
        sy = self.simcc_y(pooled).view(image.shape[0], -1, 64)
        return (
            self.heatmap(feat),
            sx,
            sy,
            self.wholebody_refine(torch.cat([sx.flatten(1), sy.flatten(1)], dim=1)),
        )


class MSPNPose(nn.Module):
    """MSPN pose model with cross-stage feature aggregation and supervision."""

    def __init__(self, joints: int = 17, width: int = 32, stages: int = 2) -> None:
        """Initialize Multi-Stage Pose Network.

        Parameters
        ----------
        joints:
            Number of keypoints.
        width:
            Backbone width.
        stages:
            Number of MSPN stages.
        """

        super().__init__()
        self.stem = ConvBNAct(3, width, stride=2)
        self.down = nn.ModuleList([ConvBNAct(width, width, stride=2) for _ in range(stages)])
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(width, width, 4, stride=2, padding=1) for _ in range(stages)]
        )
        self.cross_stage_feature_aggregation = nn.ModuleList(
            [nn.Conv2d(width, width, 1) for _ in range(stages)]
        )
        self.heads = nn.ModuleList([nn.Conv2d(width, joints, 1) for _ in range(stages)])

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        """Return coarse-to-fine MSPN stage heatmaps.

        Parameters
        ----------
        image:
            Person crop.

        Returns
        -------
        tuple[Tensor, ...]
            Per-stage heatmap predictions.
        """

        feat = self.stem(image)
        carry = torch.zeros_like(feat)
        outputs = []
        for down, up, fuse, head in zip(
            self.down, self.up, self.cross_stage_feature_aggregation, self.heads, strict=True
        ):
            stage = down(feat + carry)
            refined = F.relu(up(stage))
            carry = fuse(refined)
            outputs.append(head(refined))
        return tuple(outputs)


class RSNPose(nn.Module):
    """RSN pose model with residual-step blocks and Pose Refine Machine."""

    def __init__(self, joints: int = 17, width: int = 32) -> None:
        """Initialize residual steps network.

        Parameters
        ----------
        joints:
            Number of keypoints.
        width:
            Backbone width.
        """

        super().__init__()
        self.stem = ConvBNAct(3, width, stride=2)
        self.residual_step_blocks = nn.ModuleList(
            [ConvBNAct(width, width), ConvBNAct(width, width), ConvBNAct(width, width)]
        )
        self.pose_refine_machine = nn.Sequential(
            ConvBNAct(width * 3, width), nn.Conv2d(width, joints, 1)
        )

    def forward(self, image: Tensor) -> Tensor:
        """Predict heatmaps through RSN residual-step aggregation.

        Parameters
        ----------
        image:
            Person crop.

        Returns
        -------
        Tensor
            Refined keypoint heatmaps.
        """

        state = self.stem(image)
        step_features = []
        for block in self.residual_step_blocks:
            state = F.relu(state + block(state))
            step_features.append(state)
        return self.pose_refine_machine(torch.cat(step_features, dim=1))


class AlexNetPose(nn.Module):
    """AlexNet top-down heatmap pose estimator."""

    def __init__(self, joints: int = 17) -> None:
        """Initialize AlexNet-style pose backbone.

        Parameters
        ----------
        joints:
            Number of keypoints.
        """

        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 11, stride=4, padding=2),
            nn.ReLU(inplace=False),
            nn.LocalResponseNorm(5),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(24, 32, 5, padding=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.pose_head = nn.ConvTranspose2d(32, joints, 4, stride=4)

    def forward(self, image: Tensor) -> Tensor:
        """Predict keypoint heatmaps with AlexNet features.

        Parameters
        ----------
        image:
            Person crop.

        Returns
        -------
        Tensor
            Keypoint heatmaps.
        """

        return self.pose_head(self.features(image))


class HRFormerPose(nn.Module):
    """HRFormer pose model with high-resolution branches and window attention."""

    def __init__(self, joints: int = 17, width: int = 24) -> None:
        """Initialize HRFormer pose estimator.

        Parameters
        ----------
        joints:
            Number of keypoints.
        width:
            Branch width.
        """

        super().__init__()
        self.high = ConvBNAct(3, width, stride=2)
        self.low = ConvBNAct(width, width, stride=2)
        self.hrformer_window_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.fuse = ConvBNAct(width * 2, width)
        self.head = nn.Conv2d(width, joints, 1)

    def forward(self, image: Tensor) -> Tensor:
        """Predict heatmaps with HRFormer local-window token mixing.

        Parameters
        ----------
        image:
            Person crop.

        Returns
        -------
        Tensor
            Keypoint heatmaps.
        """

        high = self.high(image)
        low = self.low(high)
        batch, channels, height, width = high.shape
        windows = (
            high.unfold(2, 4, 4)
            .unfold(3, 4, 4)
            .reshape(batch, channels, -1, 16)
            .permute(0, 2, 3, 1)
        )
        tokens = windows.reshape(-1, 16, channels)
        tokens = tokens + self.hrformer_window_attention(tokens, tokens, tokens)[0]
        mixed = tokens.mean(dim=1).reshape(batch, -1, channels).transpose(1, 2)
        mixed = F.interpolate(mixed.unsqueeze(-1), size=high.shape[-2:], mode="nearest")
        low_up = F.interpolate(low, size=high.shape[-2:], mode="nearest")
        return self.head(self.fuse(torch.cat([high + mixed, low_up], dim=1)))


class HRNetPose(nn.Module):
    """HRNet pose model with persistent multi-resolution parallel branches."""

    def __init__(self, joints: int = 17, width: int = 24) -> None:
        """Initialize HRNet pose estimator.

        Parameters
        ----------
        joints:
            Number of keypoints.
        width:
            Branch width.
        """

        super().__init__()
        self.stem = ConvBNAct(3, width, stride=2)
        self.hr_branch = ConvBNAct(width, width)
        self.low_branch = ConvBNAct(width, width, stride=2)
        self.fuse = ConvBNAct(width * 2, width)
        self.head = nn.Conv2d(width, joints, 1)

    def forward(self, image: Tensor) -> Tensor:
        """Predict heatmaps from fused high-resolution HRNet branches.

        Parameters
        ----------
        image:
            Person crop.

        Returns
        -------
        Tensor
            Keypoint heatmaps.
        """

        high = self.stem(image)
        high = self.hr_branch(high)
        low = F.interpolate(self.low_branch(high), size=high.shape[-2:], mode="nearest")
        return self.head(self.fuse(torch.cat([high, low], dim=1)))


class CPMPose(nn.Module):
    """Convolutional Pose Machine with belief-map feedback stages."""

    def __init__(self, joints: int = 17, width: int = 32, stages: int = 3) -> None:
        """Initialize CPM pose estimator.

        Parameters
        ----------
        joints:
            Number of keypoints.
        width:
            Feature width.
        stages:
            Number of belief refinement stages.
        """

        super().__init__()
        self.image_features = nn.Sequential(ConvBNAct(3, width, stride=2), ConvBNAct(width, width))
        self.stage0 = nn.Conv2d(width, joints, 1)
        self.belief_refinement_stages = nn.ModuleList(
            [
                nn.Sequential(ConvBNAct(width + joints, width), nn.Conv2d(width, joints, 1))
                for _ in range(stages - 1)
            ]
        )

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        """Return CPM intermediate belief maps.

        Parameters
        ----------
        image:
            Person crop.

        Returns
        -------
        tuple[Tensor, ...]
            Belief maps from each stage.
        """

        feat = self.image_features(image)
        belief = self.stage0(feat)
        outputs = [belief]
        for stage in self.belief_refinement_stages:
            belief = stage(torch.cat([feat, belief], dim=1))
            outputs.append(belief)
        return tuple(outputs)


class SwinIR(nn.Module):
    """SwinIR super-resolution with residual Swin Transformer blocks."""

    def __init__(self, dim: int = 48, scale: int = 2) -> None:
        """Initialize compact SwinIR.

        Parameters
        ----------
        dim:
            Feature width.
        scale:
            Upsampling scale.
        """

        super().__init__()
        self.head = nn.Conv2d(3, dim, 3, padding=1)
        self.window_attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.shift_attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.conv_after = nn.Conv2d(dim, dim, 3, padding=1)
        self.up = nn.Sequential(
            nn.Conv2d(dim, 3 * scale * scale, 3, padding=1), nn.PixelShuffle(scale)
        )

    def _swin_block(self, feat: Tensor, attn: nn.MultiheadAttention, shift: bool) -> Tensor:
        """Apply a 2D Swin window attention block.

        Parameters
        ----------
        feat:
            Feature map.
        attn:
            Window attention module.
        shift:
            Whether to use shifted windows.

        Returns
        -------
        Tensor
            Updated feature map.
        """

        if shift:
            feat = torch.roll(feat, shifts=(-2, -2), dims=(2, 3))
        batch, channels, height, width = feat.shape
        tokens = feat.reshape(batch, channels, height // 4, 4, width // 4, 4)
        tokens = tokens.permute(0, 2, 4, 3, 5, 1).reshape(-1, 16, channels)
        attended, _ = attn(tokens, tokens, tokens)
        tokens = tokens + attended + self.mlp(tokens + attended)
        out = tokens.reshape(batch, height // 4, width // 4, 4, 4, channels)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, height, width)
        if shift:
            out = torch.roll(out, shifts=(2, 2), dims=(2, 3))
        return out

    def forward(self, image: Tensor) -> Tensor:
        """Upsample an image through residual SwinIR attention blocks.

        Parameters
        ----------
        image:
            Low-resolution image.

        Returns
        -------
        Tensor
            Super-resolved image.
        """

        feat = self.head(image)
        body = self._swin_block(feat, self.window_attn, shift=False)
        body = self._swin_block(body, self.shift_attn, shift=True)
        return self.up(feat + self.conv_after(body))


class AlignmentVideoSR(nn.Module):
    """TDAN/TOF-style video SR with learned feature alignment."""

    def __init__(self, frames: int = 5, channels: int = 32, scale: int = 2) -> None:
        """Initialize alignment and SR reconstruction modules.

        Parameters
        ----------
        frames:
            Number of video frames.
        channels:
            Feature width.
        scale:
            Upsampling scale.
        """

        super().__init__()
        self.frames = frames
        self.feat = ConvBNAct(3, channels)
        self.offset = nn.Conv2d(channels * 2, 18, 3, padding=1)
        self.offset_weight = nn.Conv2d(channels * 2, 9, 3, padding=1)
        self.fuse = ConvBNAct(channels * frames, channels)
        self.up = nn.Sequential(
            nn.Conv2d(channels, 3 * scale * scale, 3, padding=1), nn.PixelShuffle(scale)
        )

    def _deform_align(self, feat: Tensor, center: Tensor) -> Tensor:
        """Align features with learned offset-bin weights over shifted samples.

        Parameters
        ----------
        feat:
            Neighbor-frame feature map.
        center:
            Center-frame feature map.

        Returns
        -------
        Tensor
            Feature-aligned neighbor map.
        """

        pair = torch.cat([feat, center], dim=1)
        offsets = torch.tanh(self.offset(pair)).view(
            feat.shape[0], 9, 2, feat.shape[2], feat.shape[3]
        )
        weights = torch.softmax(self.offset_weight(pair), dim=1)
        shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        aligned = torch.zeros_like(feat)
        for idx, shift in enumerate(shifts):
            shifted = torch.roll(feat, shifts=shift, dims=(2, 3))
            offset_gate = 1.0 + 0.1 * offsets[:, idx].mean(dim=1, keepdim=True)
            aligned = aligned + shifted * weights[:, idx : idx + 1] * offset_gate
        return aligned

    def forward(self, video: Tensor) -> Tensor:
        """Align neighboring frames to the center and super-resolve.

        Parameters
        ----------
        video:
            Clip tensor ``(B, T, C, H, W)``.

        Returns
        -------
        Tensor
            Super-resolved center frame.
        """

        start = max((video.shape[1] - self.frames) // 2, 0)
        clip = video[:, start : start + self.frames]
        feats = [self.feat(clip[:, idx]) for idx in range(clip.shape[1])]
        center = feats[clip.shape[1] // 2]
        aligned = []
        for feat in feats:
            aligned.append(self._deform_align(feat, center))
        return self.up(self.fuse(torch.cat(aligned, dim=1)))


class TextualInversionEncoder(nn.Module):
    """Textual inversion prompt encoder with a learned concept token."""

    def __init__(self, vocab: int = 128, dim: int = 48, concept_id: int = 7) -> None:
        """Initialize token embeddings, learned concept vector, and text encoder.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        dim:
            Text embedding width.
        concept_id:
            Token id replaced by the learned textual-inversion embedding.
        """

        super().__init__()
        self.concept_id = concept_id
        self.token = nn.Embedding(vocab, dim)
        self.concept = nn.Parameter(torch.randn(dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.proj = nn.Linear(dim, dim)

    def forward(self, ids: Tensor) -> Tensor:
        """Encode a prompt after replacing the placeholder with the learned concept.

        Parameters
        ----------
        ids:
            Token ids containing the placeholder id.

        Returns
        -------
        Tensor
            Prompt conditioning vector.
        """

        emb = self.token(ids)
        mask = (ids == self.concept_id).unsqueeze(-1)
        concept = self.concept.view(1, 1, -1).expand_as(emb)
        emb = torch.where(mask, concept, emb)
        return self.proj(self.encoder(emb).mean(dim=1))


class STGCNPlusPlus(nn.Module):
    """ST-GCN++-style skeleton recognizer with adaptive graph-temporal blocks."""

    def __init__(
        self, n_joints: int = 18, in_ch: int = 2, classes: int = 60, base_ch: int = 24
    ) -> None:
        """Initialize compact ST-GCN++.

        Parameters
        ----------
        n_joints:
            Number of skeleton joints.
        in_ch:
            Input coordinate channels.
        classes:
            Number of action classes.
        base_ch:
            Base channel width.
        """

        super().__init__()
        self.bn = nn.BatchNorm1d(in_ch * n_joints)
        self.gcn1 = STGCNBlock(in_ch, base_ch, n_joints=n_joints)
        self.gcn2 = STGCNBlock(base_ch, base_ch * 2, n_joints=n_joints)
        self.temporal_branches = nn.ModuleList(
            [
                nn.Conv2d(base_ch * 2, base_ch * 2, (3, 1), padding=(1, 0), dilation=(1, 1)),
                nn.Conv2d(base_ch * 2, base_ch * 2, (3, 1), padding=(2, 0), dilation=(2, 1)),
                nn.Conv2d(base_ch * 2, base_ch * 2, 1),
            ]
        )
        self.adaptive_edge = nn.Parameter(torch.zeros(1, base_ch * 2, 1, n_joints))
        self.head = nn.Linear(base_ch * 2, classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a skeleton sequence.

        Parameters
        ----------
        x:
            Skeleton tensor ``(B, C, T, V)``.

        Returns
        -------
        Tensor
            Action logits.
        """

        batch, channels, frames, joints = x.shape
        x = self.bn(x.permute(0, 3, 1, 2).reshape(batch, joints * channels, frames))
        x = x.reshape(batch, joints, channels, frames).permute(0, 2, 3, 1)
        x = self.gcn2(self.gcn1(x))
        x = x + torch.tanh(self.adaptive_edge)
        branches = [branch(x) for branch in self.temporal_branches]
        x = torch.stack(branches, dim=0).mean(dim=0)
        return self.head(x.mean(dim=(2, 3)))


class AdaIN(nn.Module):
    """Adaptive instance normalization used by StyleGAN1 synthesis blocks."""

    def __init__(self, channels: int, style_dim: int) -> None:
        """Initialize style affine projections.

        Parameters
        ----------
        channels:
            Feature channels.
        style_dim:
            Style vector dimension.
        """

        super().__init__()
        self.scale = nn.Linear(style_dim, channels)
        self.bias = nn.Linear(style_dim, channels)

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        """Apply AdaIN style modulation.

        Parameters
        ----------
        x:
            Feature map.
        style:
            Style vector.

        Returns
        -------
        Tensor
            Styled feature map.
        """

        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        normalized = (x - mean) / std
        scale = self.scale(style).view(style.shape[0], -1, 1, 1)
        bias = self.bias(style).view(style.shape[0], -1, 1, 1)
        return normalized * (scale + 1.0) + bias


class StyleGAN1Block(nn.Module):
    """StyleGAN1 progressive synthesis block with noise and AdaIN."""

    def __init__(self, in_ch: int, out_ch: int, style_dim: int, upsample: bool) -> None:
        """Initialize one StyleGAN1 synthesis block.

        Parameters
        ----------
        in_ch:
            Input channels.
        out_ch:
            Output channels.
        style_dim:
            Style vector dimension.
        upsample:
            Whether to upsample before convolution.
        """

        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.noise_gain = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.adain = AdaIN(out_ch, style_dim)

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        """Apply progressive convolution, noise injection, and AdaIN.

        Parameters
        ----------
        x:
            Feature map.
        style:
            Style vector.

        Returns
        -------
        Tensor
            Styled feature map.
        """

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        x = F.leaky_relu(x + self.noise_gain * noise, negative_slope=0.2)
        return self.adain(x, style)


class StyleGAN1Generator(nn.Module):
    """Compact StyleGAN1 generator with AdaIN and noise injection."""

    def __init__(self, z_dim: int = 64, style_dim: int = 64, base_ch: int = 24) -> None:
        """Initialize mapping network and progressive synthesis.

        Parameters
        ----------
        z_dim:
            Latent vector dimension.
        style_dim:
            Intermediate style dimension.
        base_ch:
            Base synthesis width.
        """

        super().__init__()
        self.mapping = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
        )
        self.const = nn.Parameter(torch.randn(1, base_ch * 4, 4, 4))
        self.blocks = nn.ModuleList(
            [
                StyleGAN1Block(base_ch * 4, base_ch * 2, style_dim, upsample=True),
                StyleGAN1Block(base_ch * 2, base_ch, style_dim, upsample=True),
                StyleGAN1Block(base_ch, base_ch, style_dim, upsample=True),
            ]
        )
        self.to_rgb = nn.Conv2d(base_ch, 3, 1)

    def forward(self, z: Tensor) -> Tensor:
        """Generate an image from a latent vector.

        Parameters
        ----------
        z:
            Latent vector.

        Returns
        -------
        Tensor
            Generated RGB image.
        """

        style = self.mapping(z)
        x = self.const.expand(z.shape[0], -1, -1, -1)
        for block in self.blocks:
            x = block(x, style)
        return torch.tanh(self.to_rgb(x))


class ASTERRecognizer(nn.Module):
    """ASTER with TPS rectification and attention recognition."""

    def __init__(self, vocab: int = 38, dim: int = 48) -> None:
        """Initialize ASTER.

        Parameters
        ----------
        vocab:
            Output vocabulary size.
        dim:
            Backbone feature width.
        """

        super().__init__()
        self.rectifier = ThinPlateSplineLite()
        self.backbone = ResNet31Backbone(channels=dim)
        self.sequence = nn.LSTM(dim * 2, dim, batch_first=True, bidirectional=True)
        self.decoder = SequenceAttentionDecoder(dim * 2, vocab=vocab)

    def forward(self, image: Tensor) -> Tensor:
        """Recognize scene text with TPS rectification and attention decoding.

        Parameters
        ----------
        image:
            Grayscale text image.

        Returns
        -------
        Tensor
            Character logits.
        """

        rectified = self.rectifier(image)
        features = self.backbone(rectified).flatten(2).transpose(1, 2)
        sequence, _ = self.sequence(features)
        return self.decoder(sequence)


class NRTRResNet31Recognizer(nn.Module):
    """NRTR with a ResNet31 image feature extractor."""

    def __init__(self, vocab: int = 38, dim: int = 64) -> None:
        """Initialize ResNet31-backed NRTR.

        Parameters
        ----------
        vocab:
            Output vocabulary size.
        dim:
            Transformer width.
        """

        super().__init__()
        self.backbone = ResNet31Backbone(channels=dim // 2)
        enc_layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(dim, 4, dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, 1)
        self.decoder = nn.TransformerDecoder(dec_layer, 1)
        self.query = nn.Parameter(torch.randn(1, 8, dim) * 0.02)
        self.head = nn.Linear(dim, vocab)

    def forward(self, image: Tensor) -> Tensor:
        """Recognize text from ResNet31 sequence features.

        Parameters
        ----------
        image:
            Grayscale word image.

        Returns
        -------
        Tensor
            Character logits.
        """

        tokens = self.backbone(image).mean(dim=2).transpose(1, 2)
        memory = self.encoder(tokens)
        query = self.query.expand(image.shape[0], -1, -1)
        return self.head(self.decoder(query, memory))


def build_stgcn() -> nn.Module:
    """Build ST-GCN skeleton action recognizer."""

    return STGCN(n_joints=18, in_ch=2, n_classes=60, n_blocks=3, base_ch=32).eval()


def build_stgcnpp() -> nn.Module:
    """Build ST-GCN++-style skeleton recognizer with wider graph-temporal blocks."""

    return STGCNPlusPlus().eval()


def build_styleganv1() -> nn.Module:
    """Build StyleGAN1/FFHQ-style synthesis using mapping plus styled convolution."""

    return StyleGAN1Generator(z_dim=64, style_dim=64, base_ch=24).eval()


def build_styleganv2() -> nn.Module:
    """Build StyleGAN2 generator with modulated/demodulated convolutions."""

    return StyleGAN2Generator(z_dim=64, w_dim=64, base_ch=24).eval()


def build_styleganv3() -> nn.Module:
    """Build StyleGAN3 alias-free generator."""

    return StyleGAN3Generator(z_dim=64, w_dim=64, base_ch=24).eval()


def example_latent() -> Tensor:
    """Return a latent vector for GAN generators.

    Returns
    -------
    Tensor
        Latent tensor.
    """

    return torch.randn(1, 64)


def build_video_swin() -> nn.Module:
    """Build compact Video Swin action recognizer."""

    return VideoSwinRecognizer().eval()


def build_timesformer() -> nn.Module:
    """Build compact divided-space-time TimeSformer recognizer."""

    return TimeSformerRecognizer().eval()


def build_timesformer_joint() -> nn.Module:
    """Build compact joint space-time TimeSformer recognizer."""

    return TimeSformerRecognizer(mode="joint").eval()


def build_timesformer_space() -> nn.Module:
    """Build compact space-only TimeSformer recognizer."""

    return TimeSformerRecognizer(mode="space").eval()


def build_tanet() -> nn.Module:
    """Build compact TANet action recognizer."""

    return TemporalAdaptiveNet().eval()


def build_tin() -> nn.Module:
    """Build compact TIN action recognizer."""

    return TINRecognizer().eval()


def example_video() -> Tensor:
    """Return a compact video clip.

    Returns
    -------
    Tensor
        Video tensor ``(B, C, T, H, W)``.
    """

    return torch.randn(1, 3, 4, 32, 32)


def build_tcanet() -> nn.Module:
    """Build compact TCANet temporal action localizer."""

    return TCANetLocalizer().eval()


def example_temporal_features() -> Tensor:
    """Return temporal features for action localization.

    Returns
    -------
    Tensor
        Sequence tensor ``(B, T, C)``.
    """

    return torch.randn(1, 16, 16)


def build_fcenet() -> nn.Module:
    """Build compact FCENet Fourier-contour text detector."""

    return FourierContourHead().eval()


def build_psenet() -> nn.Module:
    """Build compact PSENet progressive-scale-expansion text detector."""

    return PSEHead().eval()


def build_panet_text() -> nn.Module:
    """Build compact PANet/FPEM text detector."""

    return PANTextHead().eval()


def build_drrg() -> nn.Module:
    """Build compact DRRG graph-reasoning text detector."""

    return DRRGHead().eval()


def build_text_mask_rcnn() -> nn.Module:
    """Build compact Mask R-CNN text detector."""

    return TextMaskRCNN().eval()


def build_svtr() -> nn.Module:
    """Build compact SVTR CTC text recognizer."""

    from menagerie.classics.svtrv2_base import SVTRv2Base

    return SVTRv2Base(classes=38, dim=48).eval()


def build_crnn() -> nn.Module:
    """Build compact CRNN-STN-ResNet31 CTC recognizer."""

    return CRNNSTNResNet31().eval()


def build_aster() -> nn.Module:
    """Build compact ASTER-like rectified attention text recognizer."""

    return ASTERRecognizer().eval()


def build_master() -> nn.Module:
    """Build compact MASTER recognizer."""

    return MASTERRecognizer(extra=True).eval()


def build_nrtr() -> nn.Module:
    """Build compact NRTR Transformer recognizer."""

    return NRTRRecognizer(with_modality_transform=True).eval()


def build_nrtr_resnet31() -> nn.Module:
    """Build compact NRTR with ResNet31 feature extractor."""

    return NRTRResNet31Recognizer().eval()


def build_robustscanner() -> nn.Module:
    """Build compact RobustScanner recognizer."""

    return RobustScannerRecognizer(attention_backbone=True).eval()


def build_sar() -> nn.Module:
    """Build compact SAR recognizer."""

    return SARRecognizer().eval()


def build_satrn() -> nn.Module:
    """Build compact SATRN recognizer."""

    return SATRNRecognizer().eval()


def example_gray_text() -> Tensor:
    """Return grayscale text-line input.

    Returns
    -------
    Tensor
        Text image.
    """

    return text_image()


def build_pose_heatmap() -> nn.Module:
    """Build compact top-down heatmap pose estimator."""

    return TopDownPose("heatmap").eval()


def build_pose_cspnext_udp() -> nn.Module:
    """Build compact CSPNeXt+UDP top-down pose estimator."""

    return CSPNeXtUDPPose().eval()


def build_pose_ipr() -> nn.Module:
    """Build compact integral pose-regression top-down estimator."""

    return IPRPose().eval()


def build_swin_pose() -> nn.Module:
    """Build compact SwinPose top-down heatmap estimator."""

    return SwinPoseNet().eval()


def build_pose_internet() -> nn.Module:
    """Build compact InterNet InterHand3D top-down pose estimator."""

    from menagerie.classics.dreimpl_3_openmmlab import build_interhand3d

    return build_interhand3d()


def build_pose_regression() -> nn.Module:
    """Build compact top-down coordinate-regression pose estimator."""

    return TopDownPose("regression").eval()


def build_pose_simcc() -> nn.Module:
    """Build compact SimCC top-down pose estimator."""

    return TopDownPose("simcc").eval()


def build_pose_simcc_mobilenetv2() -> nn.Module:
    """Build SimCC pose estimator with MobileNetV2 no-deconv backbone."""

    return MobileNetV2SimCCPose().eval()


def build_pose_simcc_res50() -> nn.Module:
    """Build SimCC pose estimator with ResNet-style backbone."""

    return ResNetSimCCPose().eval()


def build_pose_simcc_vipnas_mbv3() -> nn.Module:
    """Build SimCC pose estimator with ViPNAS-MobileNetV3 searched cells."""

    return ViPNASSimCCPose().eval()


def build_pose_cpm() -> nn.Module:
    """Build compact CPM-style top-down heatmap pose estimator."""

    return CPMPose().eval()


def build_pose_rtm() -> nn.Module:
    """Build compact RTMPose/RTMW top-down pose estimator with gated attention unit."""

    return TopDownPose("rtm").eval()


def build_pose_rtmw() -> nn.Module:
    """Build RTMW whole-body RTMPose model with RTMCC and GAU refinement."""

    return RTMWPose().eval()


def build_mspn_pose() -> nn.Module:
    """Build MSPN pose model with cross-stage feature aggregation."""

    return MSPNPose().eval()


def build_rsn_pose() -> nn.Module:
    """Build RSN pose model with residual-step blocks and PRM."""

    return RSNPose().eval()


def build_alexnet_pose() -> nn.Module:
    """Build AlexNet top-down heatmap pose estimator."""

    return AlexNetPose().eval()


def build_hrformer_pose() -> nn.Module:
    """Build HRFormer high-resolution window-attention pose estimator."""

    return HRFormerPose().eval()


def build_hrnet_pose() -> nn.Module:
    """Build HRNet high-resolution multi-branch pose estimator."""

    return HRNetPose().eval()


def example_pose_image() -> Tensor:
    """Return person-crop pose input.

    Returns
    -------
    Tensor
        RGB pose crop.
    """

    return torch.randn(1, 3, 64, 48)


def build_hourglass_pose() -> nn.Module:
    """Build compact stacked-hourglass pose estimator."""

    from menagerie.classics.stackedhourglass_8stack import StackedHourglassNet

    return StackedHourglassNet(stacks=2, channels=16, joints=17).eval()


def build_swinir() -> nn.Module:
    """Build compact SwinIR super-resolution model."""

    return SwinIR().eval()


def example_sr_image() -> Tensor:
    """Return low-resolution super-resolution input.

    Returns
    -------
    Tensor
        RGB low-resolution image.
    """

    return torch.randn(1, 3, 32, 32)


def build_tdan() -> nn.Module:
    """Build compact TDAN alignment-based video SR model."""

    return AlignmentVideoSR(frames=5).eval()


def build_tof() -> nn.Module:
    """Build compact TOF/optical-flow-alignment video SR model."""

    return AlignmentVideoSR(frames=3).eval()


def build_textual_inversion() -> nn.Module:
    """Build compact textual-inversion prompt encoder."""

    return TextualInversionEncoder().eval()


def example_prompt_ids() -> Tensor:
    """Return prompt ids containing a learned placeholder token.

    Returns
    -------
    Tensor
        Token id sequence.
    """

    return torch.tensor([[1, 5, 7, 9, 2, 0, 0, 0]], dtype=torch.long)


def example_vsr_clip() -> Tensor:
    """Return video SR clip.

    Returns
    -------
    Tensor
        Clip tensor ``(B, T, C, H, W)``.
    """

    return torch.randn(1, 5, 3, 24, 24)


def _entry(
    original: str,
    builder: str,
    example: str,
    year: str,
    code: str = "DC",
) -> tuple[str, str, str, str, str]:
    """Create a MENAGERIE_ENTRIES tuple.

    Parameters
    ----------
    original:
        Original catalog model name.
    builder:
        Build function attribute.
    example:
        Example-input function attribute.
    year:
        Publication or system year.
    code:
        Menagerie era/category code.

    Returns
    -------
    tuple[str, str, str, str, str]
        Registry entry tuple.
    """

    return (original, builder, example, year, code)


_TARGETS: Final[Sequence[tuple[str, str, str, str, str]]] = [
    _entry("mmpose_hourglass", "build_hourglass_pose", "example_pose_image", "2016"),
    _entry("mmaction2_skeleton_stgcn", "build_stgcn", "example_input_skeleton", "2018"),
    _entry("mmaction:stgcn", "build_stgcn", "example_input_skeleton", "2018"),
    _entry("mmaction2_skeleton_stgcnpp", "build_stgcnpp", "example_input_skeleton", "2018"),
    _entry("mmaction:stgcnpp", "build_stgcnpp", "example_input_skeleton", "2018"),
    _entry("mmagic:styleganv1", "build_styleganv1", "example_latent", "2019"),
    _entry("mmagic:styleganv2", "build_styleganv2", "example_latent", "2020"),
    _entry("mmagic:styleganv3", "build_styleganv3", "example_latent", "2021"),
    _entry("mmagic_styleganv1", "build_styleganv1", "example_latent", "2019"),
    _entry(
        "mmagic_styleganv1_styleganv1_ffhq_1024x1024_8xb4_25mimgs",
        "build_styleganv1",
        "example_latent",
        "2019",
    ),
    _entry("mmagic_styleganv2", "build_styleganv2", "example_latent", "2020"),
    _entry("mmagic_styleganv3", "build_styleganv3", "example_latent", "2021"),
    _entry(
        "mmagic_styleganv3_stylegan3_r_ada_gamma3_3_8xb4_fp16_metfaces_1024x1024",
        "build_styleganv3",
        "example_latent",
        "2021",
    ),
    _entry("mmocr:svtr", "build_svtr", "example_gray_text", "2022"),
    _entry("mmocr_textrecog_svtr", "build_svtr", "example_gray_text", "2022"),
    _entry("mmaction2_recognition_swin", "build_video_swin", "example_video", "2021"),
    _entry("mmaction2_swin_b_32x2_k400", "build_video_swin", "example_video", "2021"),
    _entry("mmaction2_swin_l_32x2_k400", "build_video_swin", "example_video", "2021"),
    _entry("mmaction2_swin_l_32x2_k700", "build_video_swin", "example_video", "2021"),
    _entry("mmaction2_swin_s_32x2_k400", "build_video_swin", "example_video", "2021"),
    _entry("mmaction2_swin_s_32x2_k710", "build_video_swin", "example_video", "2021"),
    _entry("mmaction2_swin_t_32x2_k400", "build_video_swin", "example_video", "2021"),
    _entry("mmaction:swin", "build_video_swin", "example_video", "2021"),
    _entry("mmaction_videoswin_base", "build_video_swin", "example_video", "2021"),
    _entry("mmaction_videoswin_large", "build_video_swin", "example_video", "2021"),
    _entry("mmaction_videoswin_small", "build_video_swin", "example_video", "2021"),
    _entry("mmaction_videoswin_tiny", "build_video_swin", "example_video", "2021"),
    _entry("mmagic_swinir", "build_swinir", "example_sr_image", "2021"),
    _entry("SwinPose-Tiny", "build_swin_pose", "example_pose_image", "2022"),
    _entry("mmaction2_recognition_tanet", "build_tanet", "example_video", "2020"),
    _entry("mmaction2_tanet_r50_1x1x8_sthv1", "build_tanet", "example_video", "2020"),
    _entry("mmaction:tanet", "build_tanet", "example_video", "2020"),
    _entry("mmaction_tanet_r50", "build_tanet", "example_video", "2020"),
    _entry("mmaction2_localization_tcanet", "build_tcanet", "example_temporal_features", "2021"),
    _entry("mmaction:tcanet", "build_tcanet", "example_temporal_features", "2021"),
    _entry("mmagic:tdan", "build_tdan", "example_vsr_clip", "2020"),
    _entry("mmagic_tdan", "build_tdan", "example_vsr_clip", "2020"),
    _entry(
        "mmagic_tdan_tdan_x4_8xb16_lr1e_4_400k_vimeo90k_bd",
        "build_tdan",
        "example_vsr_clip",
        "2020",
    ),
    _entry(
        "mmocr_textdet_dbnet_resnet18_fpnc_100k_synthtext",
        "build_dbnet_resnet50",
        "example_det_input",
        "2020",
    ),
    _entry(
        "mmocr_textdet_dbnet_resnet50_oclip_1200e_icdar2015",
        "build_dbnet_resnet50",
        "example_det_input",
        "2020",
    ),
    _entry(
        "mmocr_textdet_drrg_resnet50_fpn_unet_1200e_ctw1500",
        "build_drrg",
        "example_det_input",
        "2020",
    ),
    _entry(
        "mmocr_textdet_drrg_resnet50_oclip_fpn_unet_1200e_ctw1500",
        "build_drrg",
        "example_det_input",
        "2020",
    ),
    _entry(
        "mmocr_textdet_fcenet_resnet50_dcnv2_fpn_1500e_ctw1500",
        "build_fcenet",
        "example_det_input",
        "2021",
    ),
    _entry(
        "mmocr_textdet_fcenet_resnet50_oclip_fpn_1500e_ctw1500",
        "build_fcenet",
        "example_det_input",
        "2021",
    ),
    _entry(
        "mmocr_textdet_mask_rcnn_resnet50_oclip_fpn_160e_ctw1500",
        "build_text_mask_rcnn",
        "example_det_input",
        "2017",
    ),
    _entry(
        "mmocr_textdet_panet_resnet18_fpem_ffm_600e_ctw1500",
        "build_panet_text",
        "example_det_input",
        "2019",
    ),
    _entry(
        "mmocr_textdet_psenet_resnet50_fpnf_600e_ctw1500",
        "build_psenet",
        "example_det_input",
        "2019",
    ),
    _entry(
        "mmocr_textdet_psenet_resnet50_oclip_fpnf_600e_ctw1500",
        "build_psenet",
        "example_det_input",
        "2019",
    ),
    _entry(
        "mmocr_textdet_textsnake_resnet50_fpn_unet_1200e_ctw1500",
        "build_textsnake_unet",
        "example_det_input",
        "2018",
    ),
    _entry(
        "mmocr_textdet_textsnake_resnet50_oclip_fpn_unet_1200e_ctw1500",
        "build_textsnake_unet",
        "example_det_input",
        "2018",
    ),
    _entry(
        "mmocr_textrecog_abinet_vision_20e_st_an_mj",
        "build_abinet_vision_language",
        "example_rec_input",
        "2021",
    ),
    _entry("mmocr_textrecog_aster_resnet45_6e_st_mj", "build_aster", "example_gray_text", "2018"),
    _entry("mmocr_textrecog_crnn_mini_vgg_5e_mj", "build_crnn", "example_gray_text", "2015"),
    _entry(
        "mmocr_textrecog_master_resnet31_12e_st_mj_sa", "build_master", "example_gray_text", "2019"
    ),
    _entry(
        "mmocr_textrecog_nrtr_modality_transform_6e_st_mj",
        "build_nrtr",
        "example_gray_text",
        "2019",
    ),
    _entry(
        "mmocr_textrecog_nrtr_resnet31_1by16_1by8_6e_st_mj",
        "build_nrtr_resnet31",
        "example_gray_text",
        "2019",
    ),
    _entry(
        "mmocr_textrecog_robustscanner_resnet31_5e_st_sub_mj_sub_sa_real",
        "build_robustscanner",
        "example_gray_text",
        "2020",
    ),
    _entry(
        "mmocr_textrecog_sar_resnet31_parallel_decoder_5e_st_sub_mj_sub_sa_real",
        "build_sar",
        "example_gray_text",
        "2019",
    ),
    _entry(
        "mmocr_textrecog_satrn_shallow_small_5e_st_mj", "build_satrn", "example_gray_text", "2020"
    ),
    _entry("mmocr_textrecog_svtr_base_20e_st_mj", "build_svtr", "example_gray_text", "2022"),
    _entry("mmocr:textsnake", "build_textsnake_unet", "example_det_input", "2018"),
    _entry("mmocr_textdet_textsnake", "build_textsnake_unet", "example_det_input", "2018"),
    _entry("mmagic:textual_inversion", "build_textual_inversion", "example_prompt_ids", "2022"),
    _entry(
        "mmagic_textual_inversion_textual_inversion",
        "build_textual_inversion",
        "example_prompt_ids",
        "2022",
    ),
    _entry("mmaction2_recognition_timesformer", "build_timesformer", "example_video", "2021"),
    _entry("mmaction:timesformer", "build_timesformer", "example_video", "2021"),
    _entry("mmaction_timesformer_divst", "build_timesformer", "example_video", "2021"),
    _entry("mmaction2_timesformer_divst_8x32_k400", "build_timesformer", "example_video", "2021"),
    _entry(
        "mmaction2_timesformer_jointst_8x32_k400",
        "build_timesformer_joint",
        "example_video",
        "2021",
    ),
    _entry(
        "mmaction2_timesformer_spaceonly_8x32_k400",
        "build_timesformer_space",
        "example_video",
        "2021",
    ),
    _entry("mmaction2_recognition_tin", "build_tin", "example_video", "2020"),
    _entry("mmaction2_tin_r50_1x1x8_sthv1", "build_tin", "example_video", "2020"),
    _entry("mmaction:tin", "build_tin", "example_video", "2020"),
    _entry("mmagic:tof", "build_tof", "example_vsr_clip", "2019"),
    _entry("mmagic_tof", "build_tof", "example_vsr_clip", "2019"),
    _entry(
        "mmagic_tof_tof_spynet_chair_wobn_1xb1_vimeo90k_triplet",
        "build_tof",
        "example_vsr_clip",
        "2019",
    ),
    _entry("mmagic_tof_tof_x4_official_vimeo90k", "build_tof", "example_vsr_clip", "2019"),
    _entry("mmpose_topdown_heatmap", "build_pose_heatmap", "example_pose_image", "2019"),
    _entry("mmpose_topdown_regression", "build_pose_regression", "example_pose_image", "2019"),
    _entry(
        "mmpose_topdown_pose_estimator_cspnext_m_udp_8xb64_210e_ap10k_256x256",
        "build_pose_cspnext_udp",
        "example_pose_image",
        "2022",
    ),
    _entry(
        "mmpose_topdown_pose_estimator_internet_res50_4xb16_20e_interhand3d_256x256",
        "build_pose_internet",
        "example_pose_image",
        "2020",
    ),
    _entry(
        "mmpose_topdown_pose_estimator_ipr_res50_8xb64_210e_coco_256x256",
        "build_pose_ipr",
        "example_pose_image",
        "2019",
    ),
    _entry(
        "mmpose_topdown_pose_estimator_rtmpose_m_8xb64_210e_ap10k_256x256",
        "build_pose_rtm",
        "example_pose_image",
        "2023",
    ),
    _entry(
        "mmpose_topdown_pose_estimator_rtmw_l_8xb1024_270e_cocktail14_256x192",
        "build_pose_rtmw",
        "example_pose_image",
        "2023",
    ),
    _entry(
        "mmpose_topdown_pose_estimator_simcc_mobilenetv2_wo_deconv_8xb64_210e_coco_256x192",
        "build_pose_simcc_mobilenetv2",
        "example_pose_image",
        "2022",
    ),
    _entry(
        "mmpose_topdown_pose_estimator_simcc_res50_8xb32_140e_coco_384x288",
        "build_pose_simcc_res50",
        "example_pose_image",
        "2022",
    ),
    _entry(
        "mmpose_topdown_pose_estimator_simcc_vipnas_mbv3_8xb64_210e_coco_256x192",
        "build_pose_simcc_vipnas_mbv3",
        "example_pose_image",
        "2022",
    ),
    _entry(
        "mmpose_topdown_pose_estimator_td_hm_2xmspn50_8xb32_210e_coco_256x192",
        "build_mspn_pose",
        "example_pose_image",
        "2019",
    ),
    _entry(
        "mmpose_topdown_pose_estimator_td_hm_2xrsn50_8xb32_210e_coco_256x192",
        "build_rsn_pose",
        "example_pose_image",
        "2020",
    ),
    _entry(
        "mmpose_topdown_pose_estimator_td_hm_alexnet_8xb64_210e_coco_256x192",
        "build_alexnet_pose",
        "example_pose_image",
        "2012",
    ),
    _entry(
        "mmpose_topdown_pose_estimator_td_hm_cpm_8xb32_210e_coco_384x288",
        "build_pose_cpm",
        "example_pose_image",
        "2016",
    ),
    _entry(
        "mmpose_topdown_pose_estimator_td_hm_hourglass52_8xb32_210e_coco_256x256",
        "build_hourglass_pose",
        "example_pose_image",
        "2016",
    ),
    _entry(
        "mmpose_topdown_pose_estimator_td_hm_hrformer_base_8xb32_210e_coco_256x192",
        "build_hrformer_pose",
        "example_pose_image",
        "2021",
    ),
    _entry(
        "mmpose_topdown_pose_estimator_td_hm_hrnet_w32_8xb32_300e_animalkingdom_p1_256x256",
        "build_hrnet_pose",
        "example_pose_image",
        "2019",
    ),
]


MENAGERIE_ENTRIES: Sequence[tuple[str, str, str, str, str]] = _TARGETS

__all__ = [
    "MENAGERIE_ENTRIES",
]
