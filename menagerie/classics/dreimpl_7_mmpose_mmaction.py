"""Compact faithful MMPose, MMAction, MMagic, and 3D-perception classics.

Paper: OpenMMLab model zoos with architecture-specific papers cited per entry.
The models are random-initialized PyTorch reconstructions for dependency-gated
catalog rows whose native OpenMMLab/Paddle stacks are install-hostile here.  They
retain the audited load-bearing primitives: top-down heatmap/regression pose
heads, HRNet/LiteHRNet/ViTPose/PVT/Swin-style backbones, temporal video
recognition modules (TPN/TRN/TSM/TSN/UniFormer/VideoMAE/X3D), video pose lifting,
TPV/TR3D/transfusion/VoteNet point-cloud heads, TTSR/VICO/WGAN-GP, and MinkUNet.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution, batch normalization, and ReLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        """Initialize a convolutional block.

        Parameters
        ----------
        in_channels:
            Number of input channels.
        out_channels:
            Number of output channels.
        kernel_size:
            Spatial kernel size.
        stride:
            Spatial stride.
        groups:
            Group count for grouped or depthwise convolutions.
        """

        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the convolutional block.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Output feature map.
        """

        return self.net(x)


class ResidualUnit(nn.Module):
    """Small residual unit with optional grouped and squeeze-excitation paths."""

    def __init__(self, channels: int, groups: int = 1, se: bool = False) -> None:
        """Initialize a residual unit.

        Parameters
        ----------
        channels:
            Feature width.
        groups:
            Group count for the middle convolution.
        se:
            Whether to include a squeeze-excitation gate.
        """

        super().__init__()
        self.conv1 = ConvBlock(channels, channels, 1)
        self.conv2 = ConvBlock(channels, channels, 3, groups=groups)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.se = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, max(4, channels // 4), 1),
                nn.ReLU(inplace=False),
                nn.Conv2d(max(4, channels // 4), channels, 1),
                nn.Sigmoid(),
            )
            if se
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual bottleneck-style processing.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Residual output.
        """

        y = self.conv3(self.conv2(self.conv1(x)))
        if self.se is not None:
            y = y * self.se(y)
        return F.relu(x + y)


class SplitAttentionUnit(nn.Module):
    """ResNeSt-style split-attention residual block."""

    def __init__(self, channels: int) -> None:
        """Initialize split-attention branches.

        Parameters
        ----------
        channels:
            Feature width.
        """

        super().__init__()
        self.a = ConvBlock(channels, channels, 3, groups=2)
        self.b = ConvBlock(channels, channels, 3, groups=2)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, 2, 1), nn.Softmax(dim=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Fuse split cardinal branches with learned attention.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Split-attended output.
        """

        a = self.a(x)
        b = self.b(x)
        gate = self.gate(a + b)
        return F.relu(x + gate[:, 0:1] * a + gate[:, 1:2] * b)


class ViPNASCell(nn.Module):
    """ViPNAS searched mobile cell with mixed depthwise kernels and SE gating."""

    def __init__(self, channels: int) -> None:
        """Initialize searched operation branches.

        Parameters
        ----------
        channels:
            Feature width.
        """

        super().__init__()
        self.branches = nn.ModuleList(
            [
                ConvBlock(channels, channels, 3, groups=channels),
                ConvBlock(channels, channels, 5, groups=channels),
                ConvBlock(channels, channels, 1),
            ]
        )
        self.arch_weights = nn.Parameter(torch.tensor([0.40, 0.35, 0.25]))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // 4), 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(max(4, channels // 4), channels, 1),
            nn.Hardsigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the searched ViPNAS mixed operation.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            NAS-cell feature map.
        """

        weights = torch.softmax(self.arch_weights, dim=0)
        mixed = sum(
            weight * branch(x) for weight, branch in zip(weights, self.branches, strict=True)
        )
        return F.hardswish(x + mixed * self.se(mixed))


class PoseBackbone(nn.Module):
    """Configurable compact pose backbone preserving family-specific primitives."""

    def __init__(self, kind: str, width: int = 24) -> None:
        """Initialize a named pose backbone.

        Parameters
        ----------
        kind:
            Backbone family name.
        width:
            Base channel width.
        """

        super().__init__()
        self.kind = kind
        self.stem = nn.Sequential(ConvBlock(3, width, 3, 2), ConvBlock(width, width, 3, 2))
        self.hr_high = ConvBlock(width, width, 3)
        self.hr_low = ConvBlock(width, width, 3, 2)
        self.lite_dw = ConvBlock(width, width, 3, groups=width)
        self.lite_channel_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(width, width, 1), nn.Sigmoid()
        )
        self.shuffle_group_pw = nn.Conv2d(width, width, 1, groups=2)
        self.mobile_exp = ConvBlock(width, width * 2, 1)
        self.mobile_dw = ConvBlock(width * 2, width * 2, 3, groups=width * 2)
        self.mobile_proj = nn.Conv2d(width * 2, width, 1)
        self.resnetv1d_stem = nn.Sequential(
            ConvBlock(width, width, 3), ConvBlock(width, width, 3), nn.AvgPool2d(2, stride=1)
        )
        self.csp_a = ConvBlock(width // 2, width // 2, 3)
        self.csp_b = ConvBlock(width // 2, width // 2, 5)
        self.csp_merge = ConvBlock(width, width, 1)
        self.res = ResidualUnit(width)
        self.resnext = ResidualUnit(width, groups=4)
        self.se = ResidualUnit(width, se=True)
        self.resnest = SplitAttentionUnit(width)
        self.vipnas_cell = ViPNASCell(width)
        self.sc_context = nn.Sequential(
            nn.AvgPool2d(2), ConvBlock(width, width, 3), nn.Upsample(scale_factor=2)
        )
        self.vgg = nn.Sequential(
            ConvBlock(width, width, 3), ConvBlock(width, width, 3), ConvBlock(width, width, 3)
        )
        self.token_proj = nn.Conv2d(width, width, 1)
        self.token_attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )
        self.pvt_spatial_reduction = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.swin_shifted_window_attention = nn.MultiheadAttention(width, 4, batch_first=True)

    def _token_mix(self, x: Tensor, shift: bool = False) -> Tensor:
        """Apply compact transformer token mixing.

        Parameters
        ----------
        x:
            Feature map.
        shift:
            Whether to roll tokens before attention, approximating Swin windows.

        Returns
        -------
        Tensor
            Token-mixed feature map.
        """

        if shift:
            x = torch.roll(x, shifts=(1, 1), dims=(2, 3))
        b, c, h, w = x.shape
        tokens = self.token_proj(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.token_attn(tokens, tokens, tokens)[0]
        tokens = tokens + self.ffn(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)

    def _pvt_mix(self, x: Tensor) -> Tensor:
        """Apply pyramid transformer spatial-reduction attention.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            PVT-style token-mixed map.
        """

        b, c, h, w = x.shape
        q = x.flatten(2).transpose(1, 2)
        kv_map = self.pvt_spatial_reduction(x)
        kv = kv_map.flatten(2).transpose(1, 2)
        mixed = q + self.token_attn(q, kv, kv)[0]
        return mixed.transpose(1, 2).reshape(b, c, h, w)

    def _swin_mix(self, x: Tensor) -> Tensor:
        """Apply shifted-window attention over local 4x4 windows.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Swin-style shifted-window feature map.
        """

        shifted = torch.roll(x, shifts=(2, 2), dims=(2, 3))
        b, c, h, w = shifted.shape
        windows = shifted.unfold(2, 4, 4).unfold(3, 4, 4).reshape(b, c, -1, 16).permute(0, 2, 3, 1)
        tokens = windows.reshape(-1, 16, c)
        tokens = tokens + self.swin_shifted_window_attention(tokens, tokens, tokens)[0]
        window_mean = tokens.mean(dim=1).reshape(b, -1, c).transpose(1, 2).unsqueeze(-1)
        mixed = F.interpolate(window_mean, size=(h, w), mode="nearest")
        return torch.roll(shifted + mixed, shifts=(-2, -2), dims=(2, 3))

    def forward(self, x: Tensor) -> Tensor:
        """Encode an image into top-down pose features.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        Tensor
            Pose feature map.
        """

        x = self.stem(x)
        if "litehrnet" in self.kind or "shuffle" in self.kind:
            shuffled = torch.cat(x.chunk(2, dim=1)[::-1], dim=1)
            weighted = shuffled * self.lite_channel_weight(shuffled)
            return self.lite_dw(F.relu(self.shuffle_group_pw(weighted)))
        if "cspnext" in self.kind:
            left, right = x.chunk(2, dim=1)
            crossed = torch.cat([self.csp_a(left), self.csp_b(right)], dim=1)
            return self.csp_merge(crossed)
        if "hrformer" in self.kind:
            high = self.hr_high(x)
            low = F.interpolate(self.hr_low(x), size=x.shape[2:], mode="nearest")
            return self._token_mix(high + low)
        if "hrnet" in self.kind or "udp" in self.kind:
            low = F.interpolate(self.hr_low(x), size=x.shape[2:], mode="nearest")
            return self.hr_high(x) + low
        if "vipnas" in self.kind:
            return self.vipnas_cell(x)
        if "mobilenet" in self.kind or "mbv3" in self.kind:
            return F.relu(x + self.mobile_proj(self.mobile_dw(self.mobile_exp(x))))
        if "pvt" in self.kind or "vitpose" in self.kind:
            y = self._pvt_mix(x)
            return self._token_mix(y, shift="pvtv2" in self.kind) if "vitpose" in self.kind else y
        if "swin" in self.kind:
            return self._swin_mix(x)
        if "resnetv1d" in self.kind:
            return self.res(self.resnetv1d_stem(x))
        if "resnext" in self.kind:
            return self.resnext(x)
        if "seresnet" in self.kind:
            return self.se(x)
        if "resnest" in self.kind:
            return self.resnest(x)
        if "scnet" in self.kind:
            return F.relu(self.res(x) + self.sc_context(x))
        if "vgg" in self.kind:
            return self.vgg(x)
        if "alexnet" in self.kind:
            return self.vgg(F.max_pool2d(x, 3, stride=1, padding=1))
        return self.res(x)


class TopDownPose(nn.Module):
    """Top-down pose estimator with heatmap or coordinate-regression head."""

    def __init__(self, backbone: str, head: str = "heatmap", keypoints: int = 17) -> None:
        """Initialize a compact top-down pose estimator.

        Parameters
        ----------
        backbone:
            Backbone family name.
        head:
            Either ``"heatmap"`` or ``"regression"``.
        keypoints:
            Number of predicted keypoints.
        """

        super().__init__()
        self.head = head
        width = 24
        self.backbone = PoseBackbone(backbone, width)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(width, width, 4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(width, width, 4, stride=2, padding=1),
            nn.ReLU(inplace=False),
        )
        self.heatmap = nn.Conv2d(width, keypoints, 1)
        self.reg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width, keypoints * 4)
        )
        self.wholebody_distill = nn.Conv2d(width, keypoints, 1) if keypoints >= 100 else None

    def forward(self, image: Tensor) -> Tensor:
        """Predict keypoint heatmaps or RLE-style coordinates.

        Parameters
        ----------
        image:
            Cropped person image.

        Returns
        -------
        Tensor
            Heatmap logits or keypoint coordinate/distribution parameters.
        """

        feat = self.backbone(image)
        if self.head == "regression":
            return self.reg(feat).view(image.shape[0], -1, 4)
        up = self.deconv(feat)
        heat = self.heatmap(up)
        if self.wholebody_distill is not None:
            return heat + 0.1 * self.wholebody_distill(up)
        return heat


class SimCCPose(nn.Module):
    """Top-down pose estimator with separate SimCC x/y coordinate classifiers."""

    def __init__(
        self, backbone: str, keypoints: int = 17, width: int = 24, wholebody: bool = False
    ) -> None:
        """Initialize a SimCC pose model.

        Parameters
        ----------
        backbone:
            Backbone family name.
        keypoints:
            Number of keypoints.
        width:
            Feature width.
        wholebody:
            Whether to add an RTMW-style whole-body refinement branch.
        """

        super().__init__()
        self.wholebody = wholebody
        self.backbone = PoseBackbone(backbone, width)
        self.gated_attention_unit = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width, width, 1),
            nn.Sigmoid(),
        )
        self.heatmap_probe = nn.Conv2d(width, keypoints, 1)
        self.simcc_x = nn.Linear(width, keypoints * 32)
        self.simcc_y = nn.Linear(width, keypoints * 32)
        self.wholebody_refine = nn.Linear(keypoints * 64, keypoints * 2)

    def forward(
        self, image: Tensor
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict heatmaps and SimCC coordinate distributions.

        Parameters
        ----------
        image:
            Cropped person image.

        Returns
        -------
        tuple[Tensor, ...]
            Heatmap logits, x-axis SimCC logits, y-axis SimCC logits, and optional offsets.
        """

        feat = self.backbone(image)
        feat = feat * self.gated_attention_unit(feat)
        pooled = feat.mean(dim=(2, 3))
        heat = self.heatmap_probe(feat)
        sx = self.simcc_x(pooled).view(image.shape[0], -1, 32)
        sy = self.simcc_y(pooled).view(image.shape[0], -1, 32)
        if self.wholebody:
            return (
                heat,
                sx,
                sy,
                self.wholebody_refine(torch.cat([sx.flatten(1), sy.flatten(1)], dim=1)),
            )
        return heat, sx, sy


class IterativePoseRefinement(nn.Module):
    """IPR-style heatmap estimator with repeated residual heatmap refinement."""

    def __init__(self, keypoints: int = 17, width: int = 24, steps: int = 3) -> None:
        """Initialize iterative pose refinement.

        Parameters
        ----------
        keypoints:
            Number of keypoints.
        width:
            Feature width.
        steps:
            Number of refinement iterations.
        """

        super().__init__()
        self.backbone = PoseBackbone("res50", width)
        self.initial_heatmap = nn.Conv2d(width, keypoints, 1)
        self.refiners = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(width + keypoints, width, 3), nn.Conv2d(width, keypoints, 1)
                )
                for _ in range(steps)
            ]
        )

    def forward(self, image: Tensor) -> Tensor:
        """Predict refined keypoint heatmaps.

        Parameters
        ----------
        image:
            Cropped person image.

        Returns
        -------
        Tensor
            Refined keypoint heatmaps.
        """

        feat = self.backbone(image)
        heat = self.initial_heatmap(feat)
        for refiner in self.refiners:
            heat = heat + refiner(torch.cat([feat, heat], dim=1))
        return F.interpolate(heat, size=image.shape[2:], mode="bilinear")


class MultiStagePoseNetwork(nn.Module):
    """MSPN-style multi-stage bottom-up/top-down pose network."""

    def __init__(self, keypoints: int = 17, width: int = 24, stages: int = 2) -> None:
        """Initialize a compact multi-stage pose network.

        Parameters
        ----------
        keypoints:
            Number of keypoints.
        width:
            Feature width.
        stages:
            Number of hourglass-like stages.
        """

        super().__init__()
        self.stem = nn.Sequential(ConvBlock(3, width, 3, 2), ConvBlock(width, width, 3))
        self.down = nn.ModuleList([ConvBlock(width, width, 3, 2) for _ in range(stages)])
        self.up = nn.ModuleList([ConvBlock(width, width, 3) for _ in range(stages)])
        self.stage_heads = nn.ModuleList([nn.Conv2d(width, keypoints, 1) for _ in range(stages)])

    def forward(self, image: Tensor) -> Tensor:
        """Predict keypoint heatmaps after multi-stage refinement.

        Parameters
        ----------
        image:
            Cropped person image.

        Returns
        -------
        Tensor
            Final-stage heatmap logits.
        """

        feat = self.stem(image)
        heat = torch.zeros(
            image.shape[0],
            self.stage_heads[0].out_channels,
            feat.shape[2],
            feat.shape[3],
            device=image.device,
        )
        for down, up, head in zip(self.down, self.up, self.stage_heads, strict=True):
            coarse = down(feat)
            feat = feat + F.interpolate(up(coarse), size=feat.shape[2:], mode="nearest")
            heat = head(feat) + heat
        return F.interpolate(heat, size=image.shape[2:], mode="bilinear")


class ResidualStepPose(nn.Module):
    """RSN pose estimator with residual-step feature refinement."""

    def __init__(self, keypoints: int = 17, width: int = 24, steps: int = 3) -> None:
        """Initialize residual-step pose refinement.

        Parameters
        ----------
        keypoints:
            Number of keypoints.
        width:
            Feature width.
        steps:
            Number of residual steps.
        """

        super().__init__()
        self.stem = PoseBackbone("res50", width)
        self.residual_step_blocks = nn.ModuleList([ResidualUnit(width) for _ in range(steps)])
        self.step_adapters = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(steps)])
        self.pose_refine_machine = ConvBlock(width * steps, width, 3)
        self.head = nn.Conv2d(width, keypoints, 1)

    def forward(self, image: Tensor) -> Tensor:
        """Predict RSN keypoint heatmaps.

        Parameters
        ----------
        image:
            Cropped person image.

        Returns
        -------
        Tensor
            Heatmap logits.
        """

        state = self.stem(image)
        states = []
        for block, adapter in zip(self.residual_step_blocks, self.step_adapters, strict=True):
            state = state + adapter(block(state))
            states.append(state)
        heat = self.head(self.pose_refine_machine(torch.cat(states, dim=1)))
        return F.interpolate(heat, size=image.shape[2:], mode="bilinear")


class CPMPose(nn.Module):
    """Convolutional Pose Machines with sequential context heatmap stages."""

    def __init__(self, keypoints: int = 17, width: int = 24, stages: int = 3) -> None:
        """Initialize CPM context stages.

        Parameters
        ----------
        keypoints:
            Number of keypoints.
        width:
            Feature width.
        stages:
            Number of context refinement stages.
        """

        super().__init__()
        self.feature = PoseBackbone("vgg", width)
        self.first = nn.Conv2d(width, keypoints, 1)
        self.context_stages = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(width + keypoints, width, 7), nn.Conv2d(width, keypoints, 1)
                )
                for _ in range(stages)
            ]
        )

    def forward(self, image: Tensor) -> Tensor:
        """Predict CPM refined heatmaps.

        Parameters
        ----------
        image:
            Cropped person image.

        Returns
        -------
        Tensor
            Heatmap logits.
        """

        feat = self.feature(image)
        heat = self.first(feat)
        for stage in self.context_stages:
            heat = heat + stage(torch.cat([feat, heat], dim=1))
        return F.interpolate(heat, size=image.shape[2:], mode="bilinear")


class InterNetHandPose(nn.Module):
    """InterNet-style two-hand 3D pose estimator with left/right hand interaction."""

    def __init__(self, keypoints: int = 21, width: int = 24) -> None:
        """Initialize interacting hand pose heads.

        Parameters
        ----------
        keypoints:
            Per-hand keypoint count.
        width:
            Feature width.
        """

        super().__init__()
        self.backbone = PoseBackbone("res50", width)
        self.left_head = nn.Linear(width, keypoints * 3)
        self.right_head = nn.Linear(width, keypoints * 3)
        self.interaction_gate = nn.Sequential(nn.Linear(width * 2, width), nn.Sigmoid())
        self.relative_depth = nn.Linear(width, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict interacting left/right hand coordinates and relative depth.

        Parameters
        ----------
        image:
            Cropped interacting-hands image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Left hand joints, right hand joints, and relative hand depth.
        """

        feat = self.backbone(image).mean(dim=(2, 3))
        mirrored = torch.flip(feat, dims=(1,))
        gate = self.interaction_gate(torch.cat([feat, mirrored], dim=1))
        left_feat = feat * gate
        right_feat = mirrored * (1.0 - gate)
        left = self.left_head(left_feat).view(image.shape[0], -1, 3)
        right = self.right_head(right_feat).view(image.shape[0], -1, 3)
        return left, right, self.relative_depth(left_feat - right_feat)


class VideoStem(nn.Module):
    """Shared 3D-convolutional video stem."""

    def __init__(self, width: int = 24) -> None:
        """Initialize a compact video stem.

        Parameters
        ----------
        width:
            Output feature width.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, width, (3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=False),
            nn.Conv3d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=False),
        )

    def forward(self, video: Tensor) -> Tensor:
        """Encode a video tensor.

        Parameters
        ----------
        video:
            Video tensor ``(B, 3, T, H, W)``.

        Returns
        -------
        Tensor
            Video feature map.
        """

        return self.net(video)


class TemporalRelation(nn.Module):
    """TRN-style multi-scale temporal relation head."""

    def __init__(self, channels: int) -> None:
        """Initialize relation projections.

        Parameters
        ----------
        channels:
            Feature width.
        """

        super().__init__()
        self.pair = nn.Linear(channels * 2, channels)
        self.triplet = nn.Linear(channels * 3, channels)

    def forward(self, seq: Tensor) -> Tensor:
        """Aggregate pair and triplet frame relations.

        Parameters
        ----------
        seq:
            Frame descriptors ``(B, T, C)``.

        Returns
        -------
        Tensor
            Relation descriptor.
        """

        pair = torch.cat([seq[:, 0], seq[:, -1]], dim=-1)
        mid = seq[:, seq.shape[1] // 2]
        trip = torch.cat([seq[:, 0], mid, seq[:, -1]], dim=-1)
        return F.relu(self.pair(pair) + self.triplet(trip))


class CompactActionRecognizer(nn.Module):
    """Compact action recognizer for MMAction families."""

    def __init__(self, kind: str, classes: int = 12, width: int = 24) -> None:
        """Initialize an action recognizer.

        Parameters
        ----------
        kind:
            Recognition architecture family.
        classes:
            Number of classes.
        width:
            Feature width.
        """

        super().__init__()
        self.kind = kind
        self.stem = VideoStem(width)
        self.tpn_lateral = nn.Conv3d(width, width, 1)
        self.tpn_smooth = nn.Conv3d(width * 2, width, 3, padding=1)
        self.relation = TemporalRelation(width)
        self.temporal_conv = nn.Conv3d(width, width, (3, 1, 1), padding=(1, 0, 0), groups=3)
        self.tin_interlacing = nn.Conv3d(width, width, (3, 1, 1), padding=(1, 0, 0), groups=width)
        self.token_norm = nn.LayerNorm(width)
        self.token_attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.token_ffn = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )
        self.uniformerv2_local_mhra = nn.Conv3d(
            width, width, (3, 1, 1), padding=(1, 0, 0), groups=width
        )
        self.uniformerv2_clip_projection = nn.Linear(width, width)
        self.videomae_tubelet_embed = nn.Conv3d(3, width, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.videomae_mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.vindlu_text_tokens = nn.Parameter(torch.randn(1, 6, width) * 0.02)
        self.vindlu_cross_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.x3d_expand = nn.Conv3d(width, width * 2, 1)
        self.x3d_dw = nn.Conv3d(width * 2, width * 2, 3, padding=1, groups=width * 2)
        self.x3d_proj = nn.Conv3d(width * 2, width, 1)
        self.cls = nn.Linear(width, classes)

    def _tokens(self, feat: Tensor) -> Tensor:
        """Convert 3D features to transformer tokens.

        Parameters
        ----------
        feat:
            Video feature map.

        Returns
        -------
        Tensor
            Token sequence.
        """

        return feat.flatten(2).transpose(1, 2)

    def forward(self, video: Tensor) -> Tensor:
        """Classify a video clip.

        Parameters
        ----------
        video:
            Video tensor ``(B, 3, T, H, W)``.

        Returns
        -------
        Tensor
            Classification logits.
        """

        feat = self.stem(video)
        if "tpn" in self.kind:
            coarse = F.avg_pool3d(feat, (2, 2, 2), stride=(2, 2, 2))
            coarse = F.interpolate(self.tpn_lateral(coarse), size=feat.shape[2:], mode="nearest")
            feat = F.relu(self.tpn_smooth(torch.cat([feat, coarse], dim=1)))
            pooled = feat.mean(dim=(2, 3, 4))
        elif "trn" in self.kind:
            seq = feat.mean(dim=(3, 4)).transpose(1, 2)
            pooled = self.relation(seq)
        elif "tsm" in self.kind:
            shifted = torch.roll(feat, shifts=1, dims=2)
            pooled = (feat + shifted).mean(dim=(2, 3, 4))
        elif "tin" in self.kind:
            first, second, third = feat.chunk(3, dim=1)
            interlaced = torch.cat(
                [
                    torch.roll(first, shifts=-1, dims=2),
                    second,
                    torch.roll(third, shifts=1, dims=2),
                ],
                dim=1,
            )
            pooled = (feat + self.tin_interlacing(interlaced)).mean(dim=(2, 3, 4))
        elif "uniformerv2" in self.kind:
            feat = feat + self.uniformerv2_local_mhra(feat)
            tokens = self._tokens(feat)
            tokens = (
                tokens
                + self.token_attn(
                    self.token_norm(tokens), self.token_norm(tokens), self.token_norm(tokens)
                )[0]
            )
            pooled = self.uniformerv2_clip_projection(tokens.mean(dim=1))
        elif "videomae" in self.kind:
            tube = self.videomae_tubelet_embed(video)
            tokens = self._tokens(tube)
            mask = (
                (torch.arange(tokens.shape[1], device=tokens.device) % 2)
                .view(1, -1, 1)
                .to(tokens.dtype)
            )
            tokens = tokens * (1.0 - mask) + self.videomae_mask_token.to(tokens.dtype) * mask
            tokens = (
                tokens
                + self.token_attn(
                    self.token_norm(tokens), self.token_norm(tokens), self.token_norm(tokens)
                )[0]
            )
            pooled = tokens.mean(dim=1)
        elif "vindlu" in self.kind:
            video_tokens = self._tokens(feat)
            text_tokens = self.vindlu_text_tokens.expand(video.shape[0], -1, -1)
            fused = (
                video_tokens
                + self.vindlu_cross_attention(video_tokens, text_tokens, text_tokens)[0]
            )
            pooled = fused.mean(dim=1)
        elif "uniformer" in self.kind:
            tokens = self._tokens(feat)
            tokens = (
                tokens
                + self.token_attn(
                    self.token_norm(tokens), self.token_norm(tokens), self.token_norm(tokens)
                )[0]
            )
            tokens = tokens + self.token_ffn(self.token_norm(tokens))
            pooled = tokens.mean(dim=1)
        elif "x3d" in self.kind:
            feat = F.relu(feat + self.x3d_proj(F.relu(self.x3d_dw(F.relu(self.x3d_expand(feat))))))
            pooled = feat.mean(dim=(2, 3, 4))
        else:
            segments = feat.mean(dim=(3, 4)).transpose(1, 2)
            pooled = segments.mean(dim=1)
        return self.cls(pooled)


class VideoPoseLift(nn.Module):
    """Temporal convolutional 2D-to-3D pose lifting network."""

    def __init__(self, joints: int = 17, width: int = 48) -> None:
        """Initialize a VideoPose3D-style TCN.

        Parameters
        ----------
        joints:
            Number of joints.
        width:
            Hidden width.
        """

        super().__init__()
        self.joints = joints
        self.in_proj = nn.Conv1d(joints * 2, width, 1)
        self.blocks = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=1, dilation=1),
            nn.ReLU(inplace=False),
            nn.Conv1d(width, width, 3, padding=2, dilation=2),
            nn.ReLU(inplace=False),
            nn.Conv1d(width, width, 3, padding=3, dilation=3),
            nn.ReLU(inplace=False),
        )
        self.out = nn.Conv1d(width, joints * 3, 1)

    def forward(self, poses_2d: Tensor) -> Tensor:
        """Lift 2D joint tracks to 3D.

        Parameters
        ----------
        poses_2d:
            Tensor ``(B, T, J, 2)``.

        Returns
        -------
        Tensor
            Tensor ``(B, T, J, 3)``.
        """

        b, t, _, _ = poses_2d.shape
        x = poses_2d.reshape(b, t, -1).transpose(1, 2)
        y = self.out(self.blocks(self.in_proj(x))).transpose(1, 2)
        return y.reshape(b, t, self.joints, 3)


class PointCloudDetector(nn.Module):
    """Compact point-cloud detector for TPVFormer/TR3D/TransFusion/VoteNet/MinkUNet."""

    def __init__(self, kind: str, channels: int = 32) -> None:
        """Initialize a point-cloud perception model.

        Parameters
        ----------
        kind:
            3D architecture family.
        channels:
            Feature width.
        """

        super().__init__()
        self.kind = kind
        self.point_mlp = nn.Sequential(
            nn.Linear(6, channels), nn.ReLU(inplace=False), nn.Linear(channels, channels)
        )
        self.vote = nn.Linear(channels, 3)
        self.center_proj = nn.Linear(3, channels)
        self.query = nn.Parameter(torch.randn(12, channels) * 0.02)
        self.attn = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.tpv_xy_plane_encoder = nn.Conv2d(channels, channels, 3, padding=1)
        self.tpv_yz_plane_encoder = nn.Conv2d(channels, channels, 3, padding=1)
        self.tpv_zx_plane_encoder = nn.Conv2d(channels, channels, 3, padding=1)
        self.tr3d_sparse_conv = nn.Conv1d(channels, channels, 3, padding=1)
        self.mink_encoder = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.mink_decoder = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)
        self.transfusion_heatmap = nn.Linear(channels, 1)
        self.voxel_bev_rpn = nn.Conv2d(channels, channels, 3, padding=1)
        self.voxel_roi_refine = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.cls = nn.Linear(channels, 8)
        self.box = nn.Linear(channels, 7)

    def _plane_context(self, feat: Tensor, encoder: nn.Conv2d) -> Tensor:
        """Encode a compact orthogonal plane and broadcast it to points.

        Parameters
        ----------
        feat:
            Point features ``(B, N, C)``.
        encoder:
            Plane convolution.

        Returns
        -------
        Tensor
            Broadcast plane context.
        """

        side = 4
        plane = (
            feat[:, : side * side]
            .transpose(1, 2)
            .reshape(feat.shape[0], feat.shape[-1], side, side)
        )
        encoded = encoder(plane).flatten(2).mean(dim=-1, keepdim=True).transpose(1, 2)
        return encoded.expand(-1, feat.shape[1], -1)

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        """Detect objects or semantic labels from points.

        Parameters
        ----------
        points:
            Point tensor ``(B, N, 6)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Class logits and box/offset predictions.
        """

        feat = F.relu(self.point_mlp(points))
        if "votenet" in self.kind:
            centers = points[..., :3] + self.vote(feat)
            global_feat = feat + self.center_proj(centers.mean(dim=1, keepdim=True))
        elif "transfusion" in self.kind:
            heat = self.transfusion_heatmap(feat).squeeze(-1)
            top = torch.topk(heat, k=min(self.query.shape[0], heat.shape[1]), dim=1).indices
            gather = top.unsqueeze(-1).expand(-1, -1, feat.shape[-1])
            heatmap_queries = torch.gather(feat, 1, gather)
            global_feat = feat + self.attn(feat, heatmap_queries, heatmap_queries)[0]
        elif "voxel_rcnn" in self.kind:
            side = 4
            bev = (
                feat[:, : side * side]
                .transpose(1, 2)
                .reshape(points.shape[0], feat.shape[-1], side, side)
            )
            bev = self.voxel_bev_rpn(bev).flatten(2).transpose(1, 2)
            roi_queries = self.query.unsqueeze(0).expand(points.shape[0], -1, -1)
            roi = self.voxel_roi_refine(roi_queries, bev, bev)[0]
            global_feat = feat + roi.mean(dim=1, keepdim=True)
        elif "tpvformer" in self.kind:
            xy = self._plane_context(feat, self.tpv_xy_plane_encoder)
            yz = self._plane_context(torch.roll(feat, shifts=1, dims=1), self.tpv_yz_plane_encoder)
            zx = self._plane_context(torch.roll(feat, shifts=2, dims=1), self.tpv_zx_plane_encoder)
            global_feat = feat + xy + yz + zx
        elif "minkunet" in self.kind:
            sparse = feat.transpose(1, 2)
            encoded = self.mink_encoder(sparse)
            decoded = self.mink_decoder(encoded, output_size=sparse.shape).transpose(1, 2)
            global_feat = feat + decoded
        elif "tr3d" in self.kind:
            global_feat = feat + self.tr3d_sparse_conv(feat.transpose(1, 2)).transpose(1, 2)
        else:
            global_feat = feat
        queries = self.query.unsqueeze(0).expand(points.shape[0], -1, -1)
        decoded = queries + self.attn(queries, global_feat, global_feat)[0]
        return self.cls(decoded), self.box(decoded)


class TTSR(nn.Module):
    """Texture Transformer Super-Resolution with reference attention."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize TTSR modules.

        Parameters
        ----------
        channels:
            Feature width.
        """

        super().__init__()
        self.lr = ConvBlock(3, channels, 3)
        self.ref = ConvBlock(3, channels, 3)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.up = nn.Sequential(
            ConvBlock(channels, channels * 4, 3),
            nn.PixelShuffle(2),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply reference-style texture transfer to a low-resolution image.

        Parameters
        ----------
        x:
            Low-resolution image.

        Returns
        -------
        Tensor
            Super-resolved image.
        """

        ref = F.interpolate(x, scale_factor=2, mode="bilinear")
        low = self.lr(x)
        r = F.interpolate(self.ref(ref), size=low.shape[2:], mode="bilinear")
        q = self.q(low).flatten(2).transpose(1, 2)
        k = self.k(r).flatten(2)
        v = self.v(r).flatten(2).transpose(1, 2)
        tex = torch.matmul(torch.softmax(torch.matmul(q, k) / (low.shape[1] ** 0.5), dim=-1), v)
        tex = tex.transpose(1, 2).reshape_as(low)
        return self.up(low + tex)


class VicoAndWGAN(nn.Module):
    """VICO/WGAN-GP-style generator-discriminator compact graph."""

    def __init__(self, kind: str, z_dim: int = 32) -> None:
        """Initialize generative model.

        Parameters
        ----------
        kind:
            Generative family name.
        z_dim:
            Latent dimension.
        """

        super().__init__()
        self.kind = kind
        self.fc = nn.Linear(z_dim, 32 * 4 * 4)
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(32, 24, 4, stride=2, padding=1),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(24, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.disc = nn.Sequential(
            ConvBlock(3, 16, 3, 2),
            ConvBlock(16, 32, 3, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )
        self.prompt = nn.Linear(z_dim, z_dim)
        self.visual_condition_encoder = nn.Sequential(
            nn.Linear(z_dim, z_dim), nn.LayerNorm(z_dim), nn.Tanh()
        )
        self.vico_cross_attention = nn.MultiheadAttention(z_dim, 4, batch_first=True)
        self.gradient_penalty_probe = nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Generate an image and score it with optional conditioning/GP terms.

        Parameters
        ----------
        z:
            Latent vector.

        Returns
        -------
        tuple[Tensor, ...]
            Generated image, discriminator/contrastive score, and optional gradient penalty.
        """

        if "vico" in self.kind:
            visual = self.visual_condition_encoder(z).unsqueeze(1)
            prompt = torch.tanh(self.prompt(z)).unsqueeze(1)
            z = z + self.vico_cross_attention(prompt, visual, visual)[0].squeeze(1)
        image = self.gen(self.fc(z).view(z.shape[0], 32, 4, 4))
        score = self.disc(image)
        if "wgan-gp" in self.kind:
            finite_diff_grad = self.gradient_penalty_probe(image)
            penalty = (finite_diff_grad.flatten(1).norm(2, dim=1) - 1.0).pow(2).unsqueeze(1)
            return image, score, penalty
        return image, score


def _build_pose(backbone: str, head: str = "heatmap", keypoints: int = 17) -> nn.Module:
    """Build a top-down pose model.

    Parameters
    ----------
    backbone:
        Backbone family.
    head:
        Head kind.
    keypoints:
        Number of keypoints.

    Returns
    -------
    nn.Module
        Evaluation-mode model.
    """

    return TopDownPose(backbone, head, keypoints).eval()


def _build_special_pose(kind: str) -> nn.Module:
    """Build an architecture-specific top-down pose variant.

    Parameters
    ----------
    kind:
        Pose family selector.

    Returns
    -------
    nn.Module
        Evaluation-mode pose model.
    """

    if kind == "internet":
        return InterNetHandPose().eval()
    if kind == "ipr":
        return IterativePoseRefinement().eval()
    if kind == "rtmpose":
        return SimCCPose("cspnext").eval()
    if kind == "rtmw":
        return SimCCPose("cspnext", keypoints=33, wholebody=True).eval()
    if kind == "simcc_mobilenetv2":
        return SimCCPose("mobilenetv2").eval()
    if kind == "simcc_res50":
        return SimCCPose("res50").eval()
    if kind == "simcc_vipnas":
        return SimCCPose("vipnas_mbv3").eval()
    if kind == "mspn":
        return MultiStagePoseNetwork().eval()
    if kind == "rsn":
        return ResidualStepPose().eval()
    if kind == "cpm":
        return CPMPose().eval()
    return TopDownPose(kind).eval()


def example_pose() -> Tensor:
    """Return a compact top-down pose image.

    Returns
    -------
    Tensor
        Image tensor.
    """

    return torch.randn(1, 3, 64, 48)


def example_video() -> Tensor:
    """Return a compact video clip.

    Returns
    -------
    Tensor
        Video tensor.
    """

    return torch.randn(1, 3, 4, 32, 32)


def example_pose_sequence() -> Tensor:
    """Return a 2D pose sequence for lifting.

    Returns
    -------
    Tensor
        Pose sequence.
    """

    return torch.randn(1, 9, 17, 2)


def example_points() -> Tensor:
    """Return a compact point cloud.

    Returns
    -------
    Tensor
        Point tensor.
    """

    return torch.randn(1, 32, 6)


def example_image() -> Tensor:
    """Return a compact RGB image.

    Returns
    -------
    Tensor
        Image tensor.
    """

    return torch.randn(1, 3, 16, 16)


def example_latent() -> Tensor:
    """Return a latent vector.

    Returns
    -------
    Tensor
        Latent tensor.
    """

    return torch.randn(1, 32)


def _builder_name(name: str) -> str:
    """Create a valid builder name.

    Parameters
    ----------
    name:
        Catalog row name.

    Returns
    -------
    str
        Builder function name.
    """

    safe = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
    return f"build_{safe}"


def _register_builder(name: str, factory: Callable[[], nn.Module]) -> str:
    """Register a builder function in module globals.

    Parameters
    ----------
    name:
        Catalog row name.
    factory:
        Zero-argument model factory.

    Returns
    -------
    str
        Builder attribute name.
    """

    attr = _builder_name(name)

    def build_model() -> nn.Module:
        """Build a compact faithful dependency-gated classic.

        Returns
        -------
        nn.Module
            Evaluation-mode model.
        """

        return factory()

    build_model.__name__ = attr
    globals()[attr] = build_model
    return attr


def _pose_backbone_from_name(name: str) -> str:
    """Infer the pose backbone family from a catalog name.

    Parameters
    ----------
    name:
        Catalog row name.

    Returns
    -------
    str
        Backbone family.
    """

    lower = name.lower()
    for token in (
        "litehrnet",
        "mobilenetv2",
        "pvtv2",
        "pvt",
        "resnest",
        "resnetv1d",
        "resnext",
        "scnet",
        "seresnet",
        "shufflenetv1",
        "shufflenetv2",
        "swin",
        "vipnas",
        "vgg",
        "vipnas_mbv3",
        "vipnas_res50",
        "vitpose",
        "hrnet",
        "res101",
        "res50",
        "udp",
    ):
        if token in lower:
            return token
    return "resnet"


POSE_NAMES: Final[tuple[str, ...]] = (
    "mmpose_topdown_pose_estimator_td_hm_hrnetv2_w18_8xb64_60e_300w_256x256",
    "mmpose_topdown_pose_estimator_td_hm_litehrnet_18_8xb32_210e_coco_384x288",
    "mmpose_topdown_pose_estimator_td_hm_mobilenetv2_8xb64_210e_coco_256x192",
    "mmpose_topdown_pose_estimator_td_hm_pvt_s_8xb64_210e_coco_256x192",
    "mmpose_topdown_pose_estimator_td_hm_pvtv2_b2_8xb64_210e_coco_256x192",
    "mmpose_topdown_pose_estimator_td_hm_res101_8xb64_210e_animalpose_256x256",
    "mmpose_topdown_pose_estimator_td_hm_resnest101_8xb32_210e_coco_384x288",
    "mmpose_topdown_pose_estimator_td_hm_resnetv1d101_8xb32_210e_coco_384x288",
    "mmpose_topdown_pose_estimator_td_hm_resnext101_8xb32_210e_coco_384x288",
    "mmpose_topdown_pose_estimator_td_hm_scnet101_8xb32_210e_coco_256x192",
    "mmpose_topdown_pose_estimator_td_hm_seresnet101_8xb32_210e_coco_384x288",
    "mmpose_topdown_pose_estimator_td_hm_shufflenetv1_8xb64_210e_coco_256x192",
    "mmpose_topdown_pose_estimator_td_hm_shufflenetv2_8xb64_210e_coco_256x192",
    "mmpose_topdown_pose_estimator_td_hm_swin_b_p4_w7_8xb32_210e_coco_256x192",
    "mmpose_topdown_pose_estimator_td_hm_vgg16_bn_8xb64_210e_coco_256x192",
    "mmpose_topdown_pose_estimator_td_hm_vipnas_mbv3_8xb64_210e_coco_256x192",
    "mmpose_topdown_pose_estimator_td_hm_vipnas_res50_8xb64_210e_coco_256x192",
    "mmpose_topdown_pose_estimator_td_hm_vis_res50_8xb64_210e_coco_aic_256x192_merge",
    "mmpose_topdown_pose_estimator_td_hm_vitpose_base_8xb64_210e_coco_256x192",
    "mmpose_topdown_pose_estimator_td_hm_vitpose_base_simple_8xb64_210e_coco_256x192",
    "mmpose_topdown_pose_estimator_td_hm_vitpose_huge_8xb64_210e_humanart_256x192",
    "mmpose_udp",
    "UDP-HRNet-W48",
    "mmpose_wholebody_topdown_heatmap",
    "mmpose_wholebody_2d_keypoint_dwpose_l_dis_m_coco_256x192",
    "mmpose_vipnas",
    "ViPNAS-MobileNetV3",
    "ViTPose-B-MMPose",
    "ViTPose-H-MMPose",
    "ViTPose-L-MMPose",
    "ViTPose-S-MMPose",
)

REG_NAMES: Final[tuple[str, ...]] = (
    "mmpose_topdown_pose_estimator_td_reg_mobilenetv2_rle_pretrained_8xb64_210e_coco_256x192",
    "mmpose_topdown_pose_estimator_td_reg_res101_8xb64_210e_coco_256x192",
    "mmpose_topdown_pose_estimator_td_reg_res101_rle_8xb64_210e_coco_256x192",
    "mmpose:topdown_regression",
)

SPECIAL_POSE_KINDS: Final[dict[str, tuple[str, str]]] = {
    "mmpose_topdown_pose_estimator_cspnext_m_udp_8xb64_210e_ap10k_256x256": ("cspnext_udp", "2022"),
    "mmpose_topdown_pose_estimator_internet_res50_4xb16_20e_interhand3d_256x256": (
        "internet",
        "2020",
    ),
    "mmpose_topdown_pose_estimator_ipr_res50_8xb64_210e_coco_256x256": ("ipr", "2019"),
    "mmpose_topdown_pose_estimator_rtmpose_m_8xb64_210e_ap10k_256x256": ("rtmpose", "2023"),
    "mmpose_topdown_pose_estimator_rtmw_l_8xb1024_270e_cocktail14_256x192": ("rtmw", "2023"),
    "mmpose_topdown_pose_estimator_simcc_mobilenetv2_wo_deconv_8xb64_210e_coco_256x192": (
        "simcc_mobilenetv2",
        "2022",
    ),
    "mmpose_topdown_pose_estimator_simcc_res50_8xb32_140e_coco_384x288": ("simcc_res50", "2022"),
    "mmpose_topdown_pose_estimator_simcc_vipnas_mbv3_8xb64_210e_coco_256x192": (
        "simcc_vipnas",
        "2022",
    ),
    "mmpose_topdown_pose_estimator_td_hm_2xmspn50_8xb32_210e_coco_256x192": ("mspn", "2019"),
    "mmpose_topdown_pose_estimator_td_hm_2xrsn50_8xb32_210e_coco_256x192": ("rsn", "2020"),
    "mmpose_topdown_pose_estimator_td_hm_alexnet_8xb64_210e_coco_256x192": ("alexnet", "2012"),
    "mmpose_topdown_pose_estimator_td_hm_cpm_8xb32_210e_coco_384x288": ("cpm", "2016"),
    "mmpose_topdown_pose_estimator_td_hm_hrformer_base_8xb32_210e_coco_256x192": (
        "hrformer",
        "2021",
    ),
    "mmpose_topdown_pose_estimator_td_hm_hrnet_w32_8xb32_300e_animalkingdom_p1_256x256": (
        "hrnet",
        "2019",
    ),
}

VIDEO_KINDS: Final[dict[str, str]] = {
    "mmaction:tin": "tin",
    "mmaction2_recognition_tpn": "tpn",
    "mmaction2_tpn_r50_1x1x8_k400": "tpn",
    "mmaction:tpn": "tpn",
    "mmaction_tpn_r50": "tpn",
    "mmaction2_recognition_trn": "trn",
    "mmaction2_trn_r50_1x1x8_sthv2": "trn",
    "mmaction:trn": "trn",
    "mmaction_trn_r50": "trn",
    "mmaction2_recognition_tsm": "tsm",
    "mmaction2_tsm_r50_1x1x8_k400": "tsm",
    "mmaction:tsm": "tsm",
    "mmaction_tsm_r50": "tsm",
    "mmaction2_recognition_tsn": "tsn",
    "mmaction2_tsn_r50_1x1x8_k400": "tsn",
    "mmaction_tsn_r50": "tsn",
    "mmaction2_recognition_uniformer": "uniformer",
    "mmaction:uniformer": "uniformer",
    "mmaction2_recognition_uniformerv2": "uniformerv2",
    "mmaction2_uniformerv2_l14_32_k400": "uniformerv2",
    "mmaction2_uniformerv2_l14_32_k600": "uniformerv2",
    "mmaction2_uniformerv2_l14_32_k700": "uniformerv2",
    "mmaction2_uniformerv2_l14_8_k710": "uniformerv2",
    "mmaction2_uniformerv2_l16_8_mitv1": "uniformerv2",
    "mmaction:uniformerv2": "uniformerv2",
    "mmaction2_detection_videomae": "videomae",
    "mmaction2_recognition_videomae": "videomae",
    "mmaction2_videomae_vit_b_16x4_k400": "videomae",
    "mmaction2_videomae_vit_l_16x4_k400": "videomae",
    "mmaction:videomae": "videomae",
    "mmaction2_recognition_videomaev2": "videomaev2",
    "mmaction:videomaev2": "videomaev2",
    "mmaction2_multimodal_vindlu": "vindlu",
    "mmaction:vindlu": "vindlu",
    "mmaction2_recognition_x3d": "x3d",
    "mmaction2_x3d_m_16x5_k400": "x3d",
    "mmaction2_x3d_s_13x6_k400": "x3d",
    "mmaction:x3d": "x3d",
    "mmaction_x3d_m": "x3d",
    "mmaction_x3d_s": "x3d",
}

POINT_KINDS: Final[dict[str, str]] = {
    "mmdet3d_tpvformer": "tpvformer",
    "mmdet3d_tr3d": "tr3d",
    "transfusion_lidar": "transfusion",
    "mmdet3d:minkunet": "minkunet",
    "mmdet3d_minkunet_minkunet18_w16_torchsparse_8xb2_amp_15e_semantickitti": "minkunet",
    "mmdet3d_minkunet_minkunet34v2_w32_torchsparse_8xb2_amp_laser_polar_mix_3x_semantickitti": "minkunet",
    "mmdet3d:votenet": "votenet",
    "mmdet3d_votenet_votenet_8xb16_sunrgbd_3d": "votenet",
    "openpcdet_voxel_rcnn": "voxel_rcnn",
}

MENAGERIE_ENTRIES: list[tuple[str, str, str, str, str]] = []

for _name, (_kind, _year) in SPECIAL_POSE_KINDS.items():
    _build = _register_builder(_name, lambda k=_kind: _build_special_pose(k))
    MENAGERIE_ENTRIES.append((_name, _build, "example_pose", _year, "CV"))

for _name in POSE_NAMES:
    if _name in SPECIAL_POSE_KINDS:
        continue
    _backbone = _pose_backbone_from_name(_name)
    _keys = 133 if "wholebody" in _name.lower() or "dwpose" in _name.lower() else 17
    _build = _register_builder(_name, lambda b=_backbone, k=_keys: _build_pose(b, "heatmap", k))
    MENAGERIE_ENTRIES.append((_name, _build, "example_pose", "2019", "CV"))

for _name in REG_NAMES:
    _backbone = _pose_backbone_from_name(_name)
    _build = _register_builder(_name, lambda b=_backbone: _build_pose(b, "regression", 17))
    MENAGERIE_ENTRIES.append((_name, _build, "example_pose", "2021", "CV"))

for _name, _kind in VIDEO_KINDS.items():
    _build = _register_builder(_name, lambda k=_kind: CompactActionRecognizer(k).eval())
    MENAGERIE_ENTRIES.append((_name, _build, "example_video", "2018", "CV"))

for _name, _kind in POINT_KINDS.items():
    _build = _register_builder(_name, lambda k=_kind: PointCloudDetector(k).eval())
    MENAGERIE_ENTRIES.append((_name, _build, "example_points", "2020", "CV"))

for _name in (
    "mmpose:video_pose_lift",
    "mmpose_video_pose_lift",
    "mmpose_videopose3d",
    "VideoPose3D-TCN",
):
    _build = _register_builder(_name, lambda: VideoPoseLift().eval())
    MENAGERIE_ENTRIES.append((_name, _build, "example_pose_sequence", "2019", "CV"))

for _name in ("mmagic:ttsr", "mmagic_ttsr_ttsr_gan_x4c64b16_1xb9_500k_cufed"):
    _build = _register_builder(_name, lambda: TTSR().eval())
    MENAGERIE_ENTRIES.append((_name, _build, "example_image", "2020", "CV"))

for _name, _kind in {
    "mmagic:vico": "vico",
    "mmagic_vico_vico": "vico",
    "mmagic:wgan-gp": "wgan-gp",
    "mmagic_wgan-gp": "wgan-gp",
    "mmagic_wgan_gp_wgangp_gn_gp_50_1xb64_160kiters_lsun_bedroom_128x128": "wgan-gp",
}.items():
    _build = _register_builder(_name, lambda k=_kind: VicoAndWGAN(k).eval())
    MENAGERIE_ENTRIES.append((_name, _build, "example_latent", "2017", "CV"))
