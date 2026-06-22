"""Dependency-free classics for OpenMMLab/Paddle-style gated targets.

Paper/System: OpenMMLab model zoos plus the cited architecture papers for
HRNet, DWPose, ED-Pose, Faster R-CNN/ACRN, RetinaNet/GWD/H2RBox/G-RepPoints,
FCOS3D, FCAF3D, Group-Free-3D, H3DNet, ImVoteNet, NAFNet, SwinIR, EDVR, EG3D,
FastComposer, FLAVR, FlowNet, LiteFlowNet/MaskFlowNet/GMA, GCA, GLEAN,
Guided Diffusion, I3D, IconVSR, and Flamingo.

These are compact random-initialized PyTorch reconstructions for install-hostile
menagerie capture.  They keep the distinctive graph primitives rather than
pretending to load the original dependency stacks.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _conv(in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1) -> nn.Sequential:
    """Build a convolution, normalization, and activation block.

    Parameters
    ----------
    in_ch:
        Input channels.
    out_ch:
        Output channels.
    stride:
        Convolution stride.
    dilation:
        Dilation rate.

    Returns
    -------
    nn.Sequential
        Conv-BN-ReLU block.
    """

    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=dilation, dilation=dilation),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )


def _conv3d(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """Build a 3D convolution, normalization, and activation block.

    Parameters
    ----------
    in_ch:
        Input channels.
    out_ch:
        Output channels.
    stride:
        Spatial and depth stride.

    Returns
    -------
    nn.Sequential
        Conv3d-BN-ReLU block.
    """

    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(),
    )


def _image_grid(batch: int, height: int, width: int, device: torch.device) -> Tensor:
    """Create a normalized image grid for differentiable sampling.

    Parameters
    ----------
    batch:
        Batch size.
    height:
        Grid height.
    width:
        Grid width.
    device:
        Output device.

    Returns
    -------
    Tensor
        Grid tensor of shape ``(batch, height, width, 2)``.
    """

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)


def _local_correlation(left: Tensor, right: Tensor) -> Tensor:
    """Compute a compact nine-offset correlation volume.

    Parameters
    ----------
    left:
        First feature map.
    right:
        Second feature map.

    Returns
    -------
    Tensor
        Correlation features for a 3x3 displacement window.
    """

    corrs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = torch.roll(right, shifts=(dy, dx), dims=(-2, -1))
            corrs.append((left * shifted).mean(1, keepdim=True))
    return torch.cat(corrs, dim=1)


class TinyFPN(nn.Module):
    """Backbone plus top-down feature pyramid."""

    def __init__(self, width: int = 24) -> None:
        """Initialize the pyramid backbone.

        Parameters
        ----------
        width:
            Base feature width.
        """

        super().__init__()
        self.c3 = _conv(3, width, 2)
        self.c4 = _conv(width, width * 2, 2)
        self.c5 = _conv(width * 2, width * 4, 2)
        self.lat3 = nn.Conv2d(width, width, 1)
        self.lat4 = nn.Conv2d(width * 2, width, 1)
        self.lat5 = nn.Conv2d(width * 4, width, 1)

    def forward(self, image: Tensor) -> list[Tensor]:
        """Extract three FPN levels.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        list[Tensor]
            Feature pyramid levels from high to low resolution.
        """

        c3 = self.c3(image)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return [p3, p4, p5]


class RotatedTwoStage(nn.Module):
    """Faster R-CNN style detector with rotated ROI boxes."""

    def __init__(self, classes: int = 10, proposals: int = 6) -> None:
        """Initialize FPN, RPN, and rotated ROI heads.

        Parameters
        ----------
        classes:
            Number of object classes.
        proposals:
            Number of proposal features.
        """

        super().__init__()
        self.proposals = proposals
        self.fpn = TinyFPN()
        self.rpn = nn.Conv2d(24, 24, 3, padding=1)
        self.rpn_obj = nn.Conv2d(24, 3, 1)
        self.rpn_box = nn.Conv2d(24, 15, 1)
        self.roi = nn.Sequential(nn.Linear(24, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.cls = nn.Linear(32, classes)
        self.box5 = nn.Linear(32, classes * 5)

    def _roi_features(self, feat: Tensor, obj: Tensor) -> Tensor:
        """Gather top-RPN locations as differentiable ROI summaries.

        Parameters
        ----------
        feat:
            FPN feature map.
        obj:
            RPN objectness logits.

        Returns
        -------
        Tensor
            Proposal features.
        """

        scores = obj.flatten(2).max(dim=1).values
        idx = torch.topk(scores, self.proposals, dim=1).indices
        flat = feat.flatten(2).transpose(1, 2)
        return torch.gather(flat, 1, idx.unsqueeze(-1).expand(-1, -1, flat.shape[-1]))

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run rotated two-stage detection.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            RPN rotated boxes, ROI class logits, and five-parameter box deltas.
        """

        p3 = self.fpn(image)[0]
        rpn_feat = F.relu(self.rpn(p3))
        obj = self.rpn_obj(rpn_feat)
        anchors = self.rpn_box(rpn_feat).flatten(2).transpose(1, 2)
        roi = self.roi(self._roi_features(p3, obj))
        return anchors, self.cls(roi), self.box5(roi)


class RotatedDenseDetector(nn.Module):
    """Rotated RetinaNet, Gliding Vertex, GWD, G-RepPoints, and H2RBox detector."""

    def __init__(self, mode: str = "retina", classes: int = 10) -> None:
        """Initialize dense rotated detector heads.

        Parameters
        ----------
        mode:
            Family variant: ``retina``, ``gliding``, ``gwd``, ``points``, or ``h2rbox``.
        classes:
            Number of object classes.
        """

        super().__init__()
        self.mode = mode
        self.fpn = TinyFPN()
        self.cls_head = nn.Conv2d(24, classes * 3, 3, padding=1)
        self.box_head = nn.Conv2d(24, 15, 3, padding=1)
        self.point_head = nn.Conv2d(24, 18, 3, padding=1)
        self.vertex_head = nn.Conv2d(24, 12, 3, padding=1)
        self.angle_head = nn.Conv2d(24, 3, 3, padding=1)
        self.quality = nn.Conv2d(24, 3, 3, padding=1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict rotated dense detections.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class logits, geometry parameters, and quality/uncertainty outputs.
        """

        p3 = self.fpn(image)[0]
        cls = self.cls_head(p3).flatten(2).transpose(1, 2)
        if self.mode == "points":
            geom = self.point_head(p3).flatten(2).transpose(1, 2)
        elif self.mode == "gliding":
            base = self.box_head(p3).flatten(2).transpose(1, 2)[..., :12]
            glide = torch.tanh(self.vertex_head(p3).flatten(2).transpose(1, 2))
            angle = self.angle_head(p3).flatten(2).transpose(1, 2)
            geom = torch.cat([base + glide, torch.atan2(angle.sin(), angle.cos())], dim=-1)
        else:
            geom = self.box_head(p3).flatten(2).transpose(1, 2)
        quality = self.quality(p3).flatten(2).transpose(1, 2)
        if self.mode == "gwd":
            cov = F.softplus(geom[..., :3]) + 1.0e-3
            target_cov = torch.ones_like(cov)
            wasserstein = (
                geom[..., :3].pow(2) + cov + target_cov - 2.0 * torch.sqrt(cov * target_cov)
            ).sum(-1, keepdim=True)
            quality = torch.exp(-wasserstein) * torch.sigmoid(quality)
        if self.mode == "h2rbox":
            hbox = geom[..., :4]
            center = (hbox[..., :2] + hbox[..., 2:4]) * 0.5
            size = (hbox[..., 2:4] - hbox[..., :2]).abs()
            theta = torch.atan2(geom[..., 4:5], geom[..., 5:6])
            rotated = torch.cat([center, size, theta], dim=-1)
            consistency = rotated - rotated.roll(1, dims=-1)
            geom = torch.cat([rotated, consistency, geom[..., 10:15]], dim=-1)
        return cls, geom, quality


class FCENetTextDetector(nn.Module):
    """FCENet Fourier-contour text detector over FPN features."""

    def __init__(self, order: int = 5) -> None:
        """Initialize FCENet heads.

        Parameters
        ----------
        order:
            Fourier contour order on either side of zero.
        """

        super().__init__()
        self.fpn = TinyFPN()
        coeffs = (2 * order + 1) * 2
        self.text_region = nn.Conv2d(24, 1, 1)
        self.center_region = nn.Conv2d(24, 1, 1)
        self.fourier = nn.Conv2d(24, coeffs, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict text regions, centers, and Fourier boundary coefficients.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Text score map, center score map, Fourier coefficients, and reconstructed contours.
        """

        p3 = self.fpn(image)[0]
        text = torch.sigmoid(self.text_region(p3))
        center = torch.sigmoid(self.center_region(p3))
        coeffs = self.fourier(p3)
        harmonics = coeffs.flatten(2).transpose(1, 2)
        basis = torch.linspace(0.0, 6.283185307179586, 8, device=image.device)
        waves = torch.stack([basis.cos(), basis.sin()], dim=0)
        contour = harmonics[..., :2] @ waves
        return text, center, harmonics, contour


class PoseHRNet(nn.Module):
    """HRNet/LiteHRNet pose estimator with multi-resolution fusion."""

    def __init__(self, keypoints: int = 17, regression: bool = False, lite: bool = False) -> None:
        """Initialize HRNet branches and pose heads.

        Parameters
        ----------
        keypoints:
            Number of landmarks.
        regression:
            Whether to return coordinate regression instead of heatmaps.
        lite:
            Whether to use shuffle-style LiteHRNet fusion.
        """

        super().__init__()
        self.regression = regression
        self.lite = lite
        width = 16 if lite else 24
        self.stem = _conv(3, width, 2)
        self.high = _conv(width, width)
        self.mid = _conv(width, width, 2)
        self.low = _conv(width, width, 2)
        self.fuse = nn.Conv2d(width * 3, width, 1)
        self.heatmap = nn.Conv2d(width, keypoints, 1)
        self.coord = nn.Linear(width, keypoints * 2)

    def forward(self, image: Tensor) -> Tensor:
        """Run top-down pose estimation.

        Parameters
        ----------
        image:
            Person or face crop.

        Returns
        -------
        Tensor
            Heatmaps or keypoint coordinates.
        """

        x = self.stem(image)
        h = self.high(x)
        m = F.interpolate(self.mid(x), size=h.shape[-2:], mode="nearest")
        low = F.interpolate(self.low(self.mid(x)), size=h.shape[-2:], mode="nearest")
        fused = F.relu(self.fuse(torch.cat([h, m, low], dim=1)))
        if self.lite:
            fused = fused + fused.flip(1)
        if self.regression:
            return self.coord(F.adaptive_avg_pool2d(fused, 1).flatten(1)).view(
                image.shape[0], -1, 2
            )
        return self.heatmap(fused)


class PoseLiftTCN(nn.Module):
    """2D-to-3D image-pose lifting with temporal convolutional regression."""

    def __init__(self, joints: int = 17, channels: int = 48) -> None:
        """Initialize the temporal pose-lifting network.

        Parameters
        ----------
        joints:
            Number of 2D keypoints.
        channels:
            Hidden temporal width.
        """

        super().__init__()
        self.joints = joints
        self.in_proj = nn.Conv1d(joints * 2, channels, 1)
        self.temporal = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 3, padding=4, dilation=4),
            nn.ReLU(),
        )
        self.out = nn.Conv1d(channels, joints * 3, 1)

    def forward(self, poses_2d: Tensor) -> Tensor:
        """Lift a sequence of 2D keypoints to 3D coordinates.

        Parameters
        ----------
        poses_2d:
            2D pose sequence ``(B, T, J, 2)``.

        Returns
        -------
        Tensor
            3D pose sequence ``(B, T, J, 3)``.
        """

        batch, steps, _, _ = poses_2d.shape
        x = poses_2d.reshape(batch, steps, -1).transpose(1, 2)
        y = self.out(self.temporal(self.in_proj(x))).transpose(1, 2)
        return y.reshape(batch, steps, self.joints, 3)


class EDpose(nn.Module):
    """ED-Pose style end-to-end transformer keypoint detector."""

    def __init__(self, queries: int = 12, keypoints: int = 17) -> None:
        """Initialize query transformer and box/keypoint heads.

        Parameters
        ----------
        queries:
            Object query count.
        keypoints:
            Keypoints per query.
        """

        super().__init__()
        self.backbone = TinyFPN(16)
        self.query = nn.Parameter(torch.randn(queries, 32))
        layer = nn.TransformerDecoderLayer(32, 4, 64, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.proj = nn.Linear(16, 32)
        self.box = nn.Linear(32, 4)
        self.kpt = nn.Linear(32, keypoints * 2)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Decode object and keypoint queries.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        tuple[Tensor, Tensor]
            Query boxes and keypoint coordinates.
        """

        feat = self.backbone(image)[0].flatten(2).transpose(1, 2)
        mem = self.proj(feat)
        query = self.query.unsqueeze(0).expand(image.shape[0], -1, -1)
        out = self.decoder(query, mem)
        return self.box(out).sigmoid(), self.kpt(out).view(image.shape[0], out.shape[1], -1, 2)


class WindowPoseAttention(nn.Module):
    """Local-window self-attention block used by HRFormer-style pose models."""

    def __init__(self, channels: int, heads: int = 4, window: int = 4) -> None:
        """Initialize local-window attention.

        Parameters
        ----------
        channels:
            Feature channels.
        heads:
            Attention head count.
        window:
            Window size.
        """

        super().__init__()
        self.window = window
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply windowed attention to a feature map.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Attended feature map.
        """

        b, c, h, w = x.shape
        ws = self.window
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x_pad.shape[-2:]
        tokens = x_pad.permute(0, 2, 3, 1).view(b, hp // ws, ws, wp // ws, ws, c)
        tokens = tokens.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)
        tokens = self.norm(tokens)
        tokens, _ = self.attn(tokens, tokens, tokens)
        out = tokens.view(b, hp // ws, wp // ws, ws, ws, c)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, c)
        out = out[:, :h, :w].permute(0, 3, 1, 2)
        return x + self.proj(out)


class DWPoseRTMWholeBody(nn.Module):
    """DWPose/RTMPose whole-body estimator with SimCC and distillation paths."""

    def __init__(self, keypoints: int = 133) -> None:
        """Initialize RTMPose-style whole-body heads.

        Parameters
        ----------
        keypoints:
            Whole-body keypoint count.
        """

        super().__init__()
        self.stem = _conv(3, 24, 2)
        self.depthwise = nn.Conv2d(24, 24, 5, padding=2, groups=24)
        self.mix = nn.Conv2d(24, 32, 1)
        self.teacher = nn.Sequential(_conv(32, 32), nn.Conv2d(32, keypoints, 1))
        self.student_heatmap = nn.Conv2d(32, keypoints, 1)
        self.simcc_x = nn.Linear(32, keypoints * 32)
        self.simcc_y = nn.Linear(32, keypoints * 32)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict whole-body SimCC logits plus a teacher heatmap signal.

        Parameters
        ----------
        image:
            RGB person crop.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Student heatmaps, SimCC logits, and teacher distillation heatmaps.
        """

        feat = F.relu(self.mix(self.depthwise(self.stem(image))))
        heatmap = self.student_heatmap(feat)
        pooled_x = feat.mean(-2).transpose(1, 2)
        pooled_y = feat.mean(-1).transpose(1, 2)
        simcc = torch.stack(
            [
                self.simcc_x(pooled_x).mean(1).view(image.shape[0], -1, 32),
                self.simcc_y(pooled_y).mean(1).view(image.shape[0], -1, 32),
            ],
            dim=2,
        )
        teacher = self.teacher(feat)
        return heatmap, simcc, teacher


class HRFormerPose(nn.Module):
    """HRFormer pose model with high-resolution local-window attention."""

    def __init__(self, keypoints: int = 17) -> None:
        """Initialize HRFormer branches.

        Parameters
        ----------
        keypoints:
            Number of output keypoints.
        """

        super().__init__()
        self.stem = _conv(3, 24, 2)
        self.high_attn = WindowPoseAttention(24)
        self.low = nn.Sequential(_conv(24, 24, 2), WindowPoseAttention(24))
        self.fuse = nn.Conv2d(48, 24, 1)
        self.head = nn.Conv2d(24, keypoints, 1)

    def forward(self, image: Tensor) -> Tensor:
        """Predict HRFormer heatmaps.

        Parameters
        ----------
        image:
            RGB person crop.

        Returns
        -------
        Tensor
            Keypoint heatmaps.
        """

        high = self.high_attn(self.stem(image))
        low = F.interpolate(self.low(high), size=high.shape[-2:], mode="nearest")
        return self.head(F.relu(self.fuse(torch.cat([high, low], dim=1))))


class HigherHRNetPose(nn.Module):
    """Bottom-up HigherHRNet with multi-resolution heatmaps and grouping tags."""

    def __init__(self, keypoints: int = 17) -> None:
        """Initialize HigherHRNet heads.

        Parameters
        ----------
        keypoints:
            Number of output keypoints.
        """

        super().__init__()
        self.stem = _conv(3, 24, 2)
        self.high = _conv(24, 24)
        self.low = _conv(24, 24, 2)
        self.heat_high = nn.Conv2d(24, keypoints, 1)
        self.heat_low = nn.Conv2d(24, keypoints, 1)
        self.tag_high = nn.Conv2d(24, keypoints, 1)
        self.tag_low = nn.Conv2d(24, keypoints, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict bottom-up heatmaps and associative embedding tags.

        Parameters
        ----------
        image:
            RGB scene image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            High/low-resolution heatmaps and grouping tags.
        """

        high = self.high(self.stem(image))
        low = self.low(high)
        return self.heat_high(high), self.heat_low(low), self.tag_high(high), self.tag_low(low)


class LiteHRNetPose(nn.Module):
    """LiteHRNet with shuffle blocks and conditional channel weighting."""

    def __init__(self, keypoints: int = 17) -> None:
        """Initialize LiteHRNet pose layers.

        Parameters
        ----------
        keypoints:
            Number of output keypoints.
        """

        super().__init__()
        self.stem = _conv(3, 16, 2)
        self.branch1 = _conv(8, 8)
        self.branch2 = _conv(8, 8)
        self.weight = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(16, 16, 1), nn.Sigmoid())
        self.head = nn.Conv2d(16, keypoints, 1)

    def forward(self, image: Tensor) -> Tensor:
        """Predict LiteHRNet heatmaps.

        Parameters
        ----------
        image:
            RGB person crop.

        Returns
        -------
        Tensor
            Keypoint heatmaps.
        """

        x = self.stem(image)
        even, odd = x[:, 0::2], x[:, 1::2]
        mixed = torch.cat([self.branch1(even), self.branch2(odd)], dim=1)
        shuffled = mixed.view(mixed.shape[0], 2, mixed.shape[1] // 2, *mixed.shape[2:])
        shuffled = shuffled.transpose(1, 2).reshape_as(mixed)
        return self.head(shuffled * self.weight(shuffled))


class HandPoseInterNet(nn.Module):
    """Interacting-hand 3D pose model with relation attention between hands."""

    def __init__(self, joints: int = 21) -> None:
        """Initialize two-hand relation model.

        Parameters
        ----------
        joints:
            Joints per hand.
        """

        super().__init__()
        self.stem = _conv(3, 24, 2)
        self.left = nn.Conv2d(24, joints, 1)
        self.right = nn.Conv2d(24, joints, 1)
        self.relation = nn.MultiheadAttention(24, 4, batch_first=True)
        self.depth = nn.Linear(24, joints * 2)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Estimate interacting left/right hand heatmaps and relative depth.

        Parameters
        ----------
        image:
            RGB crop containing hands.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Left heatmaps, right heatmaps, and hand-depth logits.
        """

        feat = self.stem(image)
        tokens = feat.flatten(2).transpose(1, 2)
        related, _ = self.relation(tokens, tokens, tokens)
        pooled = related.mean(1)
        return self.left(feat), self.right(feat), self.depth(pooled).view(image.shape[0], 2, -1)


class VideoRoIAction(nn.Module):
    """SlowFast/I3D action model with RoI/ACRN-style actor context."""

    def __init__(self, vit: bool = False, nonlocal_block: bool = False) -> None:
        """Initialize spatiotemporal action modules.

        Parameters
        ----------
        vit:
            Whether to use ViT patch tokens before RoI fusion.
        nonlocal_block:
            Whether to add a non-local affinity block.
        """

        super().__init__()
        self.vit = vit
        self.nonlocal_block = nonlocal_block
        self.fast = nn.Conv3d(3, 12, 3, padding=1)
        self.slow = nn.Conv3d(3, 12, 3, stride=(2, 1, 1), padding=1)
        self.roi_proj = nn.Linear(24 * 2 * 2, 32)
        self.context_proj = nn.Linear(24, 32)
        self.actor_context = nn.MultiheadAttention(32, 4, batch_first=True)
        self.lfb_bank = nn.Parameter(torch.randn(6, 32))
        self.temporal_patch = nn.Conv3d(3, 24, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.vit_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(24, 4, 48, batch_first=True),
            1,
        )
        self.patch = nn.Linear(24, 32)
        self.cls = nn.Linear(32, 8)

    def forward(self, video: Tensor) -> Tensor:
        """Classify actions from a short clip.

        Parameters
        ----------
        video:
            Video tensor ``(B, 3, T, H, W)``.

        Returns
        -------
        Tensor
            Action logits.
        """

        if self.vit:
            patches = self.temporal_patch(video).flatten(2).transpose(1, 2)
            tokens = self.vit_encoder(patches)
            return self.cls(self.patch(tokens).mean(1))

        fast = F.relu(self.fast(video))
        slow = F.relu(self.slow(video))
        slow = F.interpolate(slow, size=fast.shape[-3:], mode="trilinear", align_corners=False)
        fused = torch.cat([fast, slow], dim=1)
        mid = fused[:, :, fused.shape[2] // 2]
        base_grid = _image_grid(video.shape[0], 2, 2, video.device).unsqueeze(1) * 0.45
        shifts = torch.tensor([-0.35, 0.35], device=video.device).view(1, 2, 1, 1, 1)
        roi_grids = base_grid + torch.cat([shifts, torch.zeros_like(shifts)], dim=-1)
        roi_grids = roi_grids.reshape(video.shape[0] * 2, 2, 2, 2)
        roi_feat = (
            mid.unsqueeze(1)
            .expand(-1, 2, -1, -1, -1)
            .reshape(video.shape[0] * 2, 24, *mid.shape[-2:])
        )
        actor = (
            F.grid_sample(roi_feat, roi_grids, align_corners=False)
            .flatten(1)
            .view(video.shape[0], 2, -1)
        )
        actor = self.roi_proj(actor)
        context = self.context_proj(fused.mean(dim=(-2, -1)).transpose(1, 2))
        feat, _ = self.actor_context(actor, context, context)
        if self.nonlocal_block:
            attn = torch.softmax(feat @ feat.transpose(1, 2) / feat.shape[-1] ** 0.5, dim=-1)
            feat = attn @ feat
        lfb = torch.softmax(feat @ self.lfb_bank.t(), dim=-1) @ self.lfb_bank
        pooled = (feat + lfb).mean(1)
        return self.cls(pooled)


class VoxelPointDetector(nn.Module):
    """3D detectors with pillars, voxel grids, queries, and primitive branches."""

    def __init__(self, mode: str = "voxel", classes: int = 6) -> None:
        """Initialize point detector variant.

        Parameters
        ----------
        mode:
            ``voxel``, ``fcaf3d``, ``fcos3d``, ``free_anchor``, ``groupfree``,
            ``h3dnet``, or ``imvotenet``.
        classes:
            Number of classes.
        """

        super().__init__()
        self.mode = mode
        self.point = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.vote = nn.Linear(32, 3)
        self.pillar_conv = nn.Sequential(_conv(32, 32), _conv(32, 32))
        self.bev_fpn = TinyFPN(16)
        self.anchor = nn.Parameter(torch.randn(9, 7))
        self.anchor_cls = nn.Conv2d(32, classes * 3, 1)
        self.anchor_box = nn.Conv2d(32, 21, 1)
        self.voxel3d = nn.Sequential(_conv3d(32, 32), _conv3d(32, 32))
        self.fcaf_cls = nn.Conv3d(32, classes, 1)
        self.fcaf_box = nn.Conv3d(32, 7, 1)
        self.fcos_image = TinyFPN(24)
        self.fcos_cls = nn.Conv2d(24, classes, 1)
        self.fcos_box2d = nn.Conv2d(24, 4, 1)
        self.fcos_box3d = nn.Conv2d(24, 7, 1)
        self.centerness = nn.Conv2d(24, 1, 1)
        self.query = nn.Parameter(torch.randn(12, 32))
        decoder_layer = nn.TransformerDecoderLayer(32, 4, 64, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, 2)
        self.cls = nn.Linear(32, classes)
        self.box = nn.Linear(32, 7)
        self.plane = nn.Linear(32, 4)
        self.line = nn.Linear(32, 6)
        self.image = TinyFPN(12)
        self.fuse = nn.Linear(44, 32)

    def _pillar_bev(self, points: Tensor, feat: Tensor) -> Tensor:
        """Scatter dynamic pillar features to a BEV pseudo-image.

        Parameters
        ----------
        points:
            Point tensor.
        feat:
            Per-point features.

        Returns
        -------
        Tensor
            BEV feature map.
        """

        b, n, c = feat.shape
        xy = torch.sigmoid(points[..., :2])
        grid = (xy * 7.0).long().clamp(0, 7)
        linear = grid[..., 1] * 8 + grid[..., 0]
        bev = feat.new_zeros(b, c, 64)
        counts = feat.new_zeros(b, 1, 64)
        bev.scatter_add_(2, linear.unsqueeze(1).expand(-1, c, -1), feat.transpose(1, 2))
        counts.scatter_add_(
            2, linear.unsqueeze(1), torch.ones_like(linear, dtype=feat.dtype).unsqueeze(1)
        )
        return (bev / counts.clamp_min(1.0)).view(b, c, 8, 8)

    def _voxel_grid(self, points: Tensor, feat: Tensor) -> Tensor:
        """Scatter point features to a small 3D voxel grid.

        Parameters
        ----------
        points:
            Point tensor.
        feat:
            Per-point features.

        Returns
        -------
        Tensor
            3D voxel feature grid.
        """

        b, n, c = feat.shape
        xyz = (torch.sigmoid(points[..., :3]) * 3.0).long().clamp(0, 3)
        linear = xyz[..., 2] * 16 + xyz[..., 1] * 4 + xyz[..., 0]
        vox = feat.new_zeros(b, c, 64)
        vox.scatter_add_(2, linear.unsqueeze(1).expand(-1, c, -1), feat.transpose(1, 2))
        return vox.view(b, c, 4, 4, 4)

    def forward(self, inputs: Tensor | tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Run compact 3D detection.

        Parameters
        ----------
        inputs:
            Point tensor ``(B, N, 6)`` or, for ImVoteNet, ``(points, image)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Class logits and 3D boxes.
        """

        if self.mode == "imvotenet":
            points, image = inputs if isinstance(inputs, tuple) else (inputs, None)
        else:
            points = inputs if isinstance(inputs, Tensor) else inputs[0]
            image = None

        if self.mode == "fcos3d":
            pseudo = points.new_zeros(points.shape[0], 3, 48, 48)
            p3 = self.fcos_image(pseudo)[0]
            cls = self.fcos_cls(p3).flatten(2).transpose(1, 2)
            box2d = self.fcos_box2d(p3).flatten(2).transpose(1, 2)
            box3d = self.fcos_box3d(p3).flatten(2).transpose(1, 2)
            center = torch.sigmoid(self.centerness(p3)).flatten(2).transpose(1, 2)
            return cls * center, torch.cat([box2d, box3d], dim=-1)

        feat = self.point(points)
        centers = points[..., :3] + self.vote(feat)
        if self.mode in {"voxel", "free_anchor"}:
            bev = self.pillar_conv(self._pillar_bev(points, feat))
            cls = self.anchor_cls(bev).flatten(2).transpose(1, 2)
            box = self.anchor_box(bev).flatten(2).transpose(1, 2)
            anchors = self.anchor.mean(0).view(1, 1, -1)
            if self.mode == "free_anchor":
                likelihood = torch.sigmoid(cls.mean(-1, keepdim=True))
                box = box[..., :7] + likelihood * anchors
            else:
                box = box[..., :7] + anchors
            return cls, box
        if self.mode == "fcaf3d":
            vox = self.voxel3d(self._voxel_grid(points, feat))
            cls = self.fcaf_cls(vox).flatten(2).transpose(1, 2)
            box = self.fcaf_box(vox).flatten(2).transpose(1, 2)
            return cls, box
        if self.mode in {"groupfree", "h3dnet"}:
            query = self.query.unsqueeze(0).expand(points.shape[0], -1, -1)
            feat = self.decoder(query, feat)
        else:
            feat = feat[:, : self.query.shape[0]]
        if self.mode == "imvotenet":
            if image is None:
                raise ValueError("ImVoteNet requires paired point and image inputs.")
            img = F.adaptive_avg_pool2d(self.image(image)[0], 1).flatten(1)
            fused = torch.cat([feat, img.unsqueeze(1).expand(-1, feat.shape[1], -1)], -1)
            feat = self.fuse(fused)
            image_vote = self.vote(feat)
            centers = centers[:, : feat.shape[1]] + image_vote
        boxes = self.box(feat)
        if self.mode == "h3dnet":
            primitives = torch.cat([self.plane(feat), self.line(feat)], dim=-1)
            boxes = torch.cat(
                [centers[:, : boxes.shape[1]], boxes[..., 3:], primitives[..., :3]], dim=-1
            )
        elif self.mode in {"fcaf3d", "fcos3d"}:
            boxes = torch.cat([centers[:, : boxes.shape[1]], boxes[..., 3:]], dim=-1)
        return self.cls(feat), boxes


class FCOS3DDetector(nn.Module):
    """Monocular image FCOS3D detector with FPN dense 2D/3D heads."""

    def __init__(self, classes: int = 6) -> None:
        """Initialize FCOS3D image heads.

        Parameters
        ----------
        classes:
            Number of classes.
        """

        super().__init__()
        self.fpn = TinyFPN(24)
        self.cls = nn.Conv2d(24, classes, 3, padding=1)
        self.centerness = nn.Conv2d(24, 1, 3, padding=1)
        self.box2d = nn.Conv2d(24, 4, 3, padding=1)
        self.depth = nn.Conv2d(24, 1, 3, padding=1)
        self.size3d = nn.Conv2d(24, 3, 3, padding=1)
        self.yaw = nn.Conv2d(24, 2, 3, padding=1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Predict dense monocular 2D boxes and 3D attributes.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        tuple[Tensor, Tensor]
            Class logits and FCOS3D box attributes.
        """

        p3 = self.fpn(image)[0]
        center = torch.sigmoid(self.centerness(p3))
        cls = (self.cls(p3) * center).flatten(2).transpose(1, 2)
        attrs = torch.cat(
            [
                self.box2d(p3),
                F.softplus(self.depth(p3)),
                F.softplus(self.size3d(p3)),
                self.yaw(p3),
            ],
            dim=1,
        )
        return cls, attrs.flatten(2).transpose(1, 2)


class NAFBlock(nn.Module):
    """NAFNet simple-gate residual block without nonlinear activations."""

    def __init__(self, channels: int) -> None:
        """Initialize NAF block.

        Parameters
        ----------
        channels:
            Channel width.
        """

        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.depth = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.sca = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply NAFNet simple-gate residual mixing.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Residual feature map.
        """

        a, b = self.expand(self.norm(x)).chunk(2, 1)
        y = self.depth(a * b)
        y = y * self.sca(F.adaptive_avg_pool2d(y, 1))
        return x + self.proj(y)


class SwinWindowBlock(nn.Module):
    """SwinIR residual window-attention block with optional cyclic shift."""

    def __init__(self, channels: int, window: int = 4, shift: int = 0) -> None:
        """Initialize window attention.

        Parameters
        ----------
        channels:
            Feature channels.
        window:
            Attention window size.
        shift:
            Cyclic shift amount.
        """

        super().__init__()
        self.window = window
        self.shift = shift
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels)
        )
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply W-MSA/SW-MSA and a residual convolution.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Restored feature map.
        """

        shortcut = x
        if self.shift:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(-2, -1))
        b, c, h, w = x.shape
        ws = self.window
        windows = x.permute(0, 2, 3, 1).view(b, h // ws, ws, w // ws, ws, c)
        windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)
        attn_in = self.norm(windows)
        tokens, _ = self.attn(attn_in, attn_in, attn_in)
        tokens = tokens + self.mlp(self.norm(tokens))
        out = tokens.view(b, h // ws, w // ws, ws, ws, c)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c).permute(0, 3, 1, 2)
        if self.shift:
            out = torch.roll(out, shifts=(self.shift, self.shift), dims=(-2, -1))
        return shortcut + self.conv(out)


class ImageRestoration(nn.Module):
    """NAFNet/SwinIR/EDVR/GLEAN/IconVSR/GCA image restoration family."""

    def __init__(self, mode: str = "nafnet") -> None:
        """Initialize restoration variant.

        Parameters
        ----------
        mode:
            Restoration family name.
        """

        super().__init__()
        self.mode = mode
        self.inp = nn.Conv2d(3, 24, 3, padding=1)
        self.blocks = nn.ModuleList([NAFBlock(24), NAFBlock(24)])
        self.swin_blocks = nn.ModuleList(
            [SwinWindowBlock(24, shift=0), SwinWindowBlock(24, shift=2)]
        )
        self.offset = nn.Conv2d(48, 18, 3, padding=1)
        self.tsa = nn.MultiheadAttention(24, 4, batch_first=True)
        self.recurrent = NAFBlock(48)
        self.gan_latent = nn.Linear(24, 24)
        self.gan_constant = nn.Parameter(torch.randn(1, 24, 8, 8))
        self.gan_to_rgb = nn.Conv2d(24, 3, 1)
        self.out = nn.Conv2d(24, 3, 3, padding=1)
        self.alpha = nn.Conv2d(24, 1, 1)

    def _edvr(self, video: Tensor) -> Tensor:
        """Run EDVR-style PCD alignment and TSA fusion.

        Parameters
        ----------
        video:
            Video tensor ``(B, T, 3, H, W)``.

        Returns
        -------
        Tensor
            Restored reference frame.
        """

        b, t, _, h, w = video.shape
        feats = [self.inp(video[:, idx]) for idx in range(t)]
        ref = feats[t // 2]
        aligned = []
        base_grid = _image_grid(b, h, w, video.device)
        for feat in feats:
            offsets = torch.tanh(self.offset(torch.cat([ref, feat], dim=1))[:, :2]).permute(
                0, 2, 3, 1
            )
            aligned.append(F.grid_sample(feat, base_grid + offsets * 0.1, align_corners=False))
        stack = torch.stack(aligned, dim=1)
        tokens = stack.mean((-2, -1))
        _, weights = self.tsa(tokens[:, t // 2 : t // 2 + 1], tokens, tokens)
        fused = (stack * weights.view(b, t, 1, 1, 1)).sum(1)
        return self.out(fused)

    def _iconvsr(self, video: Tensor) -> Tensor:
        """Run IconVSR-style refill and coupled propagation.

        Parameters
        ----------
        video:
            Video tensor ``(B, T, 3, H, W)``.

        Returns
        -------
        Tensor
            Super-resolved video.
        """

        b, t, _, h, w = video.shape
        feats = [self.inp(video[:, idx]) for idx in range(t)]
        backward: list[Tensor] = []
        state = video.new_zeros(b, 24, h, w)
        for idx in reversed(range(t)):
            refill = feats[idx - 1] if idx > 0 and idx % 2 == 0 else torch.zeros_like(state)
            state = self.recurrent(torch.cat([feats[idx] + refill, state], dim=1))[:, :24]
            backward.insert(0, state)
        outputs = []
        state = torch.zeros_like(backward[0])
        for idx in range(t):
            state = self.recurrent(torch.cat([feats[idx] + backward[idx], state], dim=1))[:, :24]
            outputs.append(F.pixel_shuffle(torch.cat([self.out(state)] * 4, dim=1), 2))
        return torch.stack(outputs, dim=1)

    def forward(self, image: Tensor) -> Tensor:
        """Restore, super-resolve, align, or matte an image.

        Parameters
        ----------
        image:
            Image tensor.

        Returns
        -------
        Tensor
            Restored image or alpha matte.
        """

        if image.ndim == 5 and self.mode == "edvr":
            return self._edvr(image)
        if image.ndim == 5 and self.mode == "iconvsr":
            return self._iconvsr(image)

        if image.shape[1] == 4 and self.mode == "gca":
            trimap = image[:, 3:4]
            image = image[:, :3]
        else:
            trimap = torch.sigmoid(image[:, :1])

        x = self.inp(image)
        for block in self.blocks:
            x = block(x)
        if self.mode == "swinir":
            for block in self.swin_blocks:
                x = block(x)
        if self.mode == "glean":
            latent = self.gan_latent(F.adaptive_avg_pool2d(x, 1).flatten(1)).view(
                x.shape[0], 24, 1, 1
            )
            prior = self.gan_constant * latent
            prior = F.interpolate(prior, size=x.shape[-2:], mode="nearest")
            x = x + prior + self.inp(self.gan_to_rgb(prior))
        if self.mode == "gca":
            guide = F.interpolate(trimap, size=x.shape[-2:], mode="nearest")
            q = (x * guide).flatten(2).transpose(1, 2)
            k = (x * (1.0 - guide)).flatten(2)
            v = x.flatten(2).transpose(1, 2)
            attn = torch.softmax(torch.bmm(q, k) / x.shape[1] ** 0.5, dim=-1)
            x = x + torch.bmm(attn, v).transpose(1, 2).view_as(x)
            return torch.sigmoid(self.alpha(x))
        y = self.out(x)
        if self.mode in {"swinir", "glean"}:
            y = F.pixel_shuffle(torch.cat([y, y, y, y], 1), 2)
        return y


class Generative2D(nn.Module):
    """EG3D/FastComposer/GAN/Guided-Diffusion/global-local generator."""

    def __init__(self, mode: str = "eg3d") -> None:
        """Initialize generative image model.

        Parameters
        ----------
        mode:
            Generative family.
        """

        super().__init__()
        self.mode = mode
        self.fc = nn.Linear(64, 64 * 4 * 4)
        self.camera = nn.Linear(16, 32)
        self.text = nn.Linear(16, 64)
        self.subject = nn.Linear(16, 32)
        self.time_embed = nn.Linear(1, 32)
        self.unet_down = _conv(3, 32, 2)
        self.unet_mid = nn.MultiheadAttention(32, 4, batch_first=True)
        self.unet_up = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.unet_rgb = nn.Conv2d(32, 3, 3, padding=1)
        self.mask_head = nn.Conv2d(3, 1, 1)
        self.local_disc = nn.Sequential(_conv(3, 16, 2), nn.Conv2d(16, 1, 1))
        self.global_disc = nn.Sequential(
            _conv(3, 16, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 1)
        )
        self.triplane = nn.Conv2d(64, 96, 3, padding=1)
        self.decoder = nn.Sequential(nn.Linear(32 + 32, 64), nn.ReLU(), nn.Linear(64, 4))
        self.conv = nn.Sequential(_conv(64, 32), _conv(32, 32), nn.Conv2d(32, 3, 3, padding=1))
        self.discriminator = nn.Sequential(
            _conv(3, 16, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 1)
        )
        self.denoise = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        """Generate or denoise a compact image.

        Parameters
        ----------
        z:
            Latent vector.

        Returns
        -------
        Tensor
            Image tensor.
        """

        x = self.fc(z).view(z.shape[0], 64, 4, 4)
        if self.mode == "fastcomposer":
            x = x + self.text(z[:, :16]).view(z.shape[0], 64, 1, 1)
        if self.mode == "eg3d":
            planes = self.triplane(x).view(z.shape[0], 3, 32, 4, 4)
            coords = _image_grid(z.shape[0], 8, 8, z.device)
            xy = F.grid_sample(planes[:, 0], coords, align_corners=False)
            yz = F.grid_sample(planes[:, 1], coords.flip(-1), align_corners=False)
            xz = F.grid_sample(planes[:, 2], -coords, align_corners=False)
            camera = self.camera(z[:, :16]).view(z.shape[0], 32, 1, 1).expand_as(xy)
            decoded = self.decoder(torch.cat([xy + yz + xz, camera], dim=1).permute(0, 2, 3, 1))
            rgb = torch.sigmoid(decoded[..., :3]).permute(0, 3, 1, 2)
            sigma = torch.sigmoid(decoded[..., 3:4]).permute(0, 3, 1, 2)
            return rgb * sigma
        x = F.interpolate(x, scale_factor=4, mode="nearest")
        image = self.conv(x)
        if self.mode == "fastcomposer":
            subject = self.subject(z[:, :16]).view(z.shape[0], 32, 1, 1)
            h = self.unet_down(image) + subject
            tokens = h.flatten(2).transpose(1, 2)
            delayed, _ = self.unet_mid(tokens[:, 4:], tokens, tokens)
            tokens = torch.cat([tokens[:, :4], delayed], dim=1)
            h = tokens.transpose(1, 2).view_as(h)
            return image + self.unet_rgb(self.unet_up(h))
        if self.mode == "guided_diffusion":
            timestep = self.time_embed(torch.ones(z.shape[0], 1, device=z.device))
            h = self.unet_down(image) + timestep.view(z.shape[0], 32, 1, 1)
            tokens = h.flatten(2).transpose(1, 2)
            tokens, _ = self.unet_mid(tokens, tokens, tokens)
            h = tokens.transpose(1, 2).view_as(h)
            image = image - self.denoise(self.unet_rgb(self.unet_up(h)))
        if self.mode == "global_local":
            mask = torch.sigmoid(self.mask_head(image))
            completed = image * (1.0 - mask) + self.denoise(image) * mask
            local = F.adaptive_avg_pool2d(completed * mask, 8)
            score = self.global_disc(completed).view(z.shape[0], 1, 1, 1)
            image = completed + F.interpolate(self.local_disc(local), completed.shape[-2:]) * score
        if self.mode == "gan":
            lsgan_score = self.discriminator(image).view(z.shape[0], 1, 1, 1)
            image = image - (lsgan_score - 1.0).pow(2)
        return image


class FlowEstimator(nn.Module):
    """FlowNet/LiteFlowNet/MaskFlowNet/GMA optical-flow estimator."""

    def __init__(self, mode: str = "flownet") -> None:
        """Initialize flow model.

        Parameters
        ----------
        mode:
            Flow architecture variant.
        """

        super().__init__()
        self.mode = mode
        self.enc = nn.Sequential(_conv(6, 24, 2), _conv(24, 32, 2), _conv(32, 32))
        self.enc1 = nn.Sequential(_conv(3, 16, 2), _conv(16, 32, 2))
        self.enc2 = nn.Sequential(_conv(3, 16, 2), _conv(16, 32, 2))
        self.corr_proj = nn.Conv2d(9, 32, 1)
        self.refine = nn.ModuleList([_conv(34, 32), _conv(34, 32), _conv(34, 32)])
        self.decoder = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.update = nn.GRUCell(34, 32)
        self.mask = nn.Conv2d(32, 1, 3, padding=1)
        self.flow = nn.Conv2d(32, 2, 3, padding=1)

    def _warp(self, feat: Tensor, flow: Tensor) -> Tensor:
        """Warp features with a normalized flow field.

        Parameters
        ----------
        feat:
            Feature map to sample.
        flow:
            Flow tensor.

        Returns
        -------
        Tensor
            Warped feature map.
        """

        grid = _image_grid(feat.shape[0], feat.shape[-2], feat.shape[-1], feat.device)
        norm = torch.stack(
            [
                flow[:, 0] / max(feat.shape[-1] - 1, 1),
                flow[:, 1] / max(feat.shape[-2] - 1, 1),
            ],
            dim=-1,
        )
        return F.grid_sample(feat, grid + norm, align_corners=False)

    def forward(self, pair: Tensor) -> tuple[Tensor, Tensor]:
        """Estimate optical flow from two stacked frames.

        Parameters
        ----------
        pair:
            Two RGB frames stacked on channels.

        Returns
        -------
        tuple[Tensor, Tensor]
            Flow and optional occlusion/mask logits.
        """

        if self.mode == "flownet":
            feat = self.enc(pair)
            coarse = self.flow(feat)
            feat = self.decoder(feat)
            flow = F.interpolate(coarse, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            return self.flow(feat) + flow, self.mask(feat)

        left = self.enc1(pair[:, :3])
        right = self.enc2(pair[:, 3:])
        corr = self.corr_proj(_local_correlation(left, right))
        feat = left + corr
        flow = self.flow(feat)
        if self.mode == "flownet2":
            for block in self.refine[:2]:
                warped = self._warp(right, flow)
                feat = block(torch.cat([left - warped, flow], dim=1))
                flow = flow + self.flow(feat)
        if self.mode in {"liteflownet", "maskflownet"}:
            for block in self.refine[:2]:
                warped = self._warp(right, flow)
                match = self.corr_proj(_local_correlation(left, warped))
                feat = block(torch.cat([match, flow], dim=1))
                flow = flow + self.flow(feat)
        if self.mode == "gma":
            hidden = feat.flatten(2).transpose(1, 2)
            context, _ = self.attn(hidden, hidden, hidden)
            state = hidden.mean(1)
            for _ in range(3):
                corr_state = torch.cat([context.mean(1), flow.mean((-2, -1))], dim=1)
                state = self.update(corr_state, state)
            feat = feat + state.view(state.shape[0], state.shape[1], 1, 1)
        mask = self.mask(feat)
        if self.mode == "maskflownet":
            feat = feat * torch.sigmoid(mask)
            flow = flow + self.flow(feat)
        if self.mode == "flownetc":
            flow = self.flow(feat)
        return flow, mask


class FLAVR(nn.Module):
    """FLAVR 3D-UNet frame interpolation network."""

    def __init__(self) -> None:
        """Initialize 3D encoder-decoder."""

        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 24, 3, stride=(1, 2, 2), padding=1),
            nn.ReLU(),
        )
        self.dec = nn.ConvTranspose3d(24, 16, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
        self.out = nn.Conv3d(16, 3, 3, padding=1)

    def forward(self, video: Tensor) -> Tensor:
        """Interpolate a middle video frame.

        Parameters
        ----------
        video:
            Input frame stack ``(B, 3, T, H, W)``.

        Returns
        -------
        Tensor
            Interpolated RGB frame.
        """

        x = self.dec(self.enc(video))
        return self.out(x).mean(2)


class TextVisionModel(nn.Module):
    """Flamingo/GLIP vision-language fusion model."""

    def __init__(self, mode: str = "flamingo") -> None:
        """Initialize cross-modal model.

        Parameters
        ----------
        mode:
            ``flamingo`` for gated cross-attention or ``glip`` for grounded detection.
        """

        super().__init__()
        self.mode = mode
        self.vision = TinyFPN(16)
        self.token = nn.Embedding(64, 32)
        self.vis_proj = nn.Linear(16, 32)
        self.media_query = nn.Parameter(torch.randn(6, 32))
        self.perceiver = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(32, 4, 64, batch_first=True),
            1,
        )
        self.cross = nn.MultiheadAttention(32, 4, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))
        self.lang_self = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(32, 4, 64, batch_first=True),
            1,
        )
        self.dyhead = nn.Conv2d(16, 16, 3, padding=1)
        self.out = nn.Linear(32, 64 if mode == "flamingo" else 4)

    def forward(self, image: Tensor) -> Tensor:
        """Run image-conditioned text or grounding prediction.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        Tensor
            Language logits or grounded boxes.
        """

        feats = self.vision(image)
        visual = self.vis_proj(feats[0].flatten(2).transpose(1, 2))
        ids = torch.arange(8, device=image.device).unsqueeze(0).expand(image.shape[0], -1)
        text = self.token(ids)
        if self.mode == "flamingo":
            media = self.media_query.unsqueeze(0).expand(image.shape[0], -1, -1)
            media = self.perceiver(media, visual)
            fused, _ = self.cross(text, media, media)
            text = self.lang_self(text + torch.tanh(self.gate) * fused)
            return self.out(text)
        language = self.lang_self(text)
        scale = language.mean(1).unsqueeze(-1).unsqueeze(-1)
        dy = self.dyhead(feats[0] * scale[:, :16])
        grounded = self.vis_proj(dy.flatten(2).transpose(1, 2))
        fused, _ = self.cross(grounded, language, language)
        return self.out(fused)


Build = Callable[[], nn.Module]


def _image() -> Tensor:
    """Return a compact RGB image input.

    Returns
    -------
    Tensor
        Image tensor.
    """

    return torch.randn(1, 3, 48, 48)


def _small_image() -> Tensor:
    """Return a smaller RGB image input.

    Returns
    -------
    Tensor
        Image tensor.
    """

    return torch.randn(1, 3, 32, 32)


def _matting_image() -> Tensor:
    """Return RGB plus trimap input for matting.

    Returns
    -------
    Tensor
        Four-channel matting tensor.
    """

    return torch.randn(1, 4, 32, 32)


def _sr_video() -> Tensor:
    """Return a compact video restoration input.

    Returns
    -------
    Tensor
        Video tensor ``(B, T, C, H, W)``.
    """

    return torch.randn(1, 4, 3, 16, 16)


def _video() -> Tensor:
    """Return a compact video input.

    Returns
    -------
    Tensor
        Video tensor.
    """

    return torch.randn(1, 3, 4, 32, 32)


def _flow_pair() -> Tensor:
    """Return two stacked RGB frames.

    Returns
    -------
    Tensor
        Six-channel frame pair.
    """

    return torch.randn(1, 6, 48, 48)


def _points() -> Tensor:
    """Return point-cloud input.

    Returns
    -------
    Tensor
        Point tensor.
    """

    return torch.randn(1, 32, 6)


def _pose2d_sequence() -> Tensor:
    """Return 2D keypoint tracks for pose lifting.

    Returns
    -------
    Tensor
        Pose sequence tensor.
    """

    return torch.randn(1, 9, 17, 2)


def _imvotenet_input() -> tuple[Tensor, Tensor]:
    """Return paired point cloud and RGB image inputs for ImVoteNet.

    Returns
    -------
    tuple[Tensor, Tensor]
        Point tensor and paired image tensor.
    """

    return torch.randn(1, 32, 6), torch.randn(1, 3, 32, 32)


def _latent() -> Tensor:
    """Return latent-vector input.

    Returns
    -------
    Tensor
        Latent tensor.
    """

    return torch.randn(1, 64)


def build_rotated_faster_rcnn() -> nn.Module:
    """Build rotated Faster R-CNN.

    Returns
    -------
    nn.Module
        Rotated two-stage detector.
    """

    return RotatedTwoStage().eval()


def build_rotated_retinanet() -> nn.Module:
    """Build rotated RetinaNet.

    Returns
    -------
    nn.Module
        Rotated dense detector.
    """

    return RotatedDenseDetector("retina").eval()


def build_gliding_vertex() -> nn.Module:
    """Build Gliding Vertex rotated detector.

    Returns
    -------
    nn.Module
        Dense detector with gliding vertex offsets and angle conversion.
    """

    return RotatedDenseDetector("gliding").eval()


def build_gwd() -> nn.Module:
    """Build GWD rotated RetinaNet.

    Returns
    -------
    nn.Module
        Dense detector with Gaussian-distance quality.
    """

    return RotatedDenseDetector("gwd").eval()


def build_points() -> nn.Module:
    """Build G-RepPoints detector.

    Returns
    -------
    nn.Module
        Dense point-set detector.
    """

    return RotatedDenseDetector("points").eval()


def build_fcenet() -> nn.Module:
    """Build FCENet Fourier-contour text detector.

    Returns
    -------
    nn.Module
        Text detector with Fourier boundary coefficients.
    """

    return FCENetTextDetector().eval()


def build_h2rbox() -> nn.Module:
    """Build H2RBox detector.

    Returns
    -------
    nn.Module
        Horizontal-to-rotated box detector.
    """

    return RotatedDenseDetector("h2rbox").eval()


def build_pose_heatmap() -> nn.Module:
    """Build heatmap pose HRNet.

    Returns
    -------
    nn.Module
        Pose heatmap model.
    """

    return PoseHRNet().eval()


def build_pose_regression() -> nn.Module:
    """Build regression pose HRNet.

    Returns
    -------
    nn.Module
        Pose regression model.
    """

    return PoseHRNet(regression=True).eval()


def build_pose_lift_tcn() -> nn.Module:
    """Build image-pose-lift 2D-to-3D temporal regression model.

    Returns
    -------
    nn.Module
        Temporal pose lifting model.
    """

    return PoseLiftTCN().eval()


def build_wholebody() -> nn.Module:
    """Build whole-body DWPose-style model.

    Returns
    -------
    nn.Module
        Whole-body pose model.
    """

    return DWPoseRTMWholeBody(keypoints=133).eval()


def build_hand() -> nn.Module:
    """Build hand pose model.

    Returns
    -------
    nn.Module
        Hand pose model.
    """

    return HandPoseInterNet(joints=21).eval()


def build_litehrnet() -> nn.Module:
    """Build LiteHRNet pose model.

    Returns
    -------
    nn.Module
        LiteHRNet model.
    """

    return LiteHRNetPose().eval()


def build_hrformer() -> nn.Module:
    """Build HRFormer pose model.

    Returns
    -------
    nn.Module
        HRFormer with local-window attention.
    """

    return HRFormerPose().eval()


def build_higherhrnet() -> nn.Module:
    """Build HigherHRNet bottom-up pose model.

    Returns
    -------
    nn.Module
        HigherHRNet with associative embeddings.
    """

    return HigherHRNetPose().eval()


def build_edpose() -> nn.Module:
    """Build ED-Pose.

    Returns
    -------
    nn.Module
        Query-based keypoint detector.
    """

    return EDpose().eval()


def build_action() -> nn.Module:
    """Build SlowFast ACRN action detector.

    Returns
    -------
    nn.Module
        Action model.
    """

    return VideoRoIAction().eval()


def build_action_vit() -> nn.Module:
    """Build ViT action detector.

    Returns
    -------
    nn.Module
        ViT-style action model.
    """

    return VideoRoIAction(vit=True).eval()


def build_i3d() -> nn.Module:
    """Build I3D action recognizer.

    Returns
    -------
    nn.Module
        I3D-style model.
    """

    return VideoRoIAction(nonlocal_block=False).eval()


def build_i3d_nl() -> nn.Module:
    """Build non-local I3D action recognizer.

    Returns
    -------
    nn.Module
        I3D with non-local block.
    """

    return VideoRoIAction(nonlocal_block=True).eval()


def build_voxel() -> nn.Module:
    """Build dynamic voxelization model.

    Returns
    -------
    nn.Module
        Voxel point detector.
    """

    return VoxelPointDetector("voxel").eval()


def build_fcaf3d() -> nn.Module:
    """Build FCAF3D.

    Returns
    -------
    nn.Module
        Fully convolutional 3D detector.
    """

    return VoxelPointDetector("fcaf3d").eval()


def build_fcos3d() -> nn.Module:
    """Build FCOS3D.

    Returns
    -------
    nn.Module
        Monocular/point FCOS3D-style detector.
    """

    return FCOS3DDetector().eval()


def build_free_anchor() -> nn.Module:
    """Build FreeAnchor PointPillars.

    Returns
    -------
    nn.Module
        FreeAnchor detector.
    """

    return VoxelPointDetector("free_anchor").eval()


def build_groupfree() -> nn.Module:
    """Build Group-Free-3D.

    Returns
    -------
    nn.Module
        Transformer-query 3D detector.
    """

    return VoxelPointDetector("groupfree").eval()


def build_h3dnet() -> nn.Module:
    """Build H3DNet.

    Returns
    -------
    nn.Module
        Hybrid primitive 3D detector.
    """

    return VoxelPointDetector("h3dnet").eval()


def build_imvotenet() -> nn.Module:
    """Build ImVoteNet.

    Returns
    -------
    nn.Module
        Image-and-point vote detector.
    """

    return VoxelPointDetector("imvotenet").eval()


def build_nafnet() -> nn.Module:
    """Build NAFNet restoration model.

    Returns
    -------
    nn.Module
        NAFNet.
    """

    return ImageRestoration("nafnet").eval()


def build_swinir() -> nn.Module:
    """Build SwinIR model.

    Returns
    -------
    nn.Module
        Window-attention super-resolution model.
    """

    return ImageRestoration("swinir").eval()


def build_edvr() -> nn.Module:
    """Build EDVR model.

    Returns
    -------
    nn.Module
        Alignment-and-fusion video restorer.
    """

    return ImageRestoration("edvr").eval()


def build_gca() -> nn.Module:
    """Build GCA matting model.

    Returns
    -------
    nn.Module
        Guided-context alpha matting model.
    """

    return ImageRestoration("gca").eval()


def build_glean() -> nn.Module:
    """Build GLEAN restoration model.

    Returns
    -------
    nn.Module
        GAN-prior super-resolution model.
    """

    return ImageRestoration("glean").eval()


def build_iconvsr() -> nn.Module:
    """Build IconVSR model.

    Returns
    -------
    nn.Module
        Recurrent video super-resolution model.
    """

    return ImageRestoration("iconvsr").eval()


def build_eg3d() -> nn.Module:
    """Build EG3D generator.

    Returns
    -------
    nn.Module
        Tri-plane generator.
    """

    return Generative2D("eg3d").eval()


def build_fastcomposer() -> nn.Module:
    """Build FastComposer-style model.

    Returns
    -------
    nn.Module
        Text-conditioned generator.
    """

    return Generative2D("fastcomposer").eval()


def build_gan() -> nn.Module:
    """Build LSGAN-style generator.

    Returns
    -------
    nn.Module
        GAN generator.
    """

    return Generative2D("gan").eval()


def build_global_local() -> nn.Module:
    """Build global-local inpainting model.

    Returns
    -------
    nn.Module
        Global-local generator.
    """

    return Generative2D("global_local").eval()


def build_guided_diffusion() -> nn.Module:
    """Build guided diffusion denoiser.

    Returns
    -------
    nn.Module
        Denoising U-Net proxy.
    """

    return Generative2D("guided_diffusion").eval()


def build_flavr() -> nn.Module:
    """Build FLAVR frame interpolation model.

    Returns
    -------
    nn.Module
        3D-UNet interpolator.
    """

    return FLAVR().eval()


def _build_flow(mode: str) -> nn.Module:
    """Build a flow-estimation variant.

    Parameters
    ----------
    mode:
        Flow variant.

    Returns
    -------
    nn.Module
        Flow estimator.
    """

    return FlowEstimator(mode).eval()


def build_flownet() -> nn.Module:
    """Build FlowNetS.

    Returns
    -------
    nn.Module
        FlowNet-style estimator.
    """

    return _build_flow("flownet")


def build_flownetc() -> nn.Module:
    """Build FlowNetC/FlowNet2 correlation model.

    Returns
    -------
    nn.Module
        Correlation flow estimator.
    """

    return _build_flow("flownetc")


def build_flownet2() -> nn.Module:
    """Build FlowNet2 stacked correlation model.

    Returns
    -------
    nn.Module
        FlowNet2-style estimator.
    """

    return _build_flow("flownet2")


def build_liteflownet() -> nn.Module:
    """Build LiteFlowNet.

    Returns
    -------
    nn.Module
        Pyramid flow estimator.
    """

    return _build_flow("liteflownet")


def build_maskflownet() -> nn.Module:
    """Build MaskFlowNet.

    Returns
    -------
    nn.Module
        Occlusion-mask flow estimator.
    """

    return _build_flow("maskflownet")


def build_gma() -> nn.Module:
    """Build GMA optical-flow model.

    Returns
    -------
    nn.Module
        Global-motion-aggregation flow model.
    """

    return _build_flow("gma")


def build_flamingo() -> nn.Module:
    """Build Flamingo.

    Returns
    -------
    nn.Module
        Gated vision-language model.
    """

    return TextVisionModel("flamingo").eval()


def build_glip() -> nn.Module:
    """Build GLIP.

    Returns
    -------
    nn.Module
        Grounded vision-language detector.
    """

    return TextVisionModel("glip").eval()


MENAGERIE_ENTRIES = [
    ("mmpose_wholebody_dwpose", "build_wholebody", "_image", "2023", "DC"),
    ("mmdet3d:dynamic_voxelization", "build_voxel", "_points", "2019", "DC"),
    (
        "mmagic_base_edit_model_nafnet_c64eb11128mb1db1111_8xb8_lr1e_3_400k_gopro",
        "build_nafnet",
        "_small_image",
        "2022",
        "DC",
    ),
    (
        "mmagic_base_edit_model_nafnet_c64eb2248mb12db2222_8xb8_lr1e_3_400k_sidd",
        "build_nafnet",
        "_small_image",
        "2022",
        "DC",
    ),
    (
        "mmagic_base_edit_model_swinir_gan_x2s64w8d6e180_8xb4_lr1e_4_600k_df2k_ost",
        "build_swinir",
        "_small_image",
        "2021",
        "DC",
    ),
    ("mmpose_edpose", "build_edpose", "_image", "2022", "DC"),
    ("mmagic:edvr", "build_edvr", "_sr_video", "2019", "DC"),
    ("mmagic_edvr", "build_edvr", "_sr_video", "2019", "DC"),
    ("mmagic_edvr_edvrl_c128b40_8xb8_lr2e_4_600k_reds4", "build_edvr", "_sr_video", "2019", "DC"),
    ("mmagic:eg3d", "build_eg3d", "_latent", "2022", "DC"),
    ("mmagic_eg3d", "build_eg3d", "_latent", "2022", "DC"),
    ("mmagic_eg3d_eg3d_cvt_official_rgb_afhq_512x512", "build_eg3d", "_latent", "2022", "DC"),
    ("mmpose_face_topdown_heatmap", "build_pose_heatmap", "_image", "2019", "DC"),
    ("mmpose_face_topdown_regression", "build_pose_regression", "_image", "2019", "DC"),
    (
        "mmaction2_fast_rcnn_slowfast_acrn_kinetics400_pretrained_r50_8xb8_8x8x1_cosine_10e_ava21_rgb",
        "build_action",
        "_video",
        "2018",
        "DC",
    ),
    (
        "mmaction2_fast_rcnn_slowonly_lfb_infer_r50_ava21_rgb",
        "build_action",
        "_video",
        "2019",
        "DC",
    ),
    (
        "mmaction2_fast_rcnn_vit_base_p16_videomae_k400_pre_8xb8_16x4x1_20e_adamw_ava_kinetics_rgb",
        "build_action_vit",
        "_video",
        "2022",
        "DC",
    ),
    ("mmagic:fastcomposer", "build_fastcomposer", "_latent", "2023", "DC"),
    ("mmagic_fastcomposer_fastcomposer_8xb16_ffhq", "build_fastcomposer", "_latent", "2023", "DC"),
    ("mmrotate:rotated_faster_rcnn", "build_rotated_faster_rcnn", "_image", "2017", "DC"),
    (
        "mmrotate_rotated_faster_rcnn_rotated_faster_rcnn_r50_fpn_1x_dota_le90",
        "build_rotated_faster_rcnn",
        "_image",
        "2017",
        "DC",
    ),
    ("mmdet3d:fcaf3d", "build_fcaf3d", "_points", "2021", "DC"),
    ("mmdet3d_fcaf3d", "build_fcaf3d", "_points", "2021", "DC"),
    ("mmdet3d_fcaf3d_fcaf3d_2xb8_s3dis_3d_5class", "build_fcaf3d", "_points", "2021", "DC"),
    ("mmocr:fcenet", "build_fcenet", "_image", "2021", "DC"),
    ("mmocr_textdet_fcenet", "build_fcenet", "_image", "2021", "DC"),
    ("mmdet3d:fcos3d", "build_fcos3d", "_image", "2021", "DC"),
    (
        "mmdet3d_fcos3d_fcos3d_r101_caffe_dcn_fpn_head_gn_8xb2_1x_nus_mono3d",
        "build_fcos3d",
        "_image",
        "2021",
        "DC",
    ),
    ("mmpretrain:flamingo", "build_flamingo", "_image", "2022", "DC"),
    ("mmagic:flavr", "build_flavr", "_video", "2021", "DC"),
    ("mmagic_flavr", "build_flavr", "_video", "2021", "DC"),
    ("mmagic_flavr_flavr_in4out1_8xb4_vimeo90k_septuplet", "build_flavr", "_video", "2021", "DC"),
    (
        "mmflow_flownet_flownet2_8x1_sfine_flyingthings3d_subset_384x768",
        "build_flownet2",
        "_flow_pair",
        "2017",
        "DC",
    ),
    (
        "mmflow_flownet_flownet2cs_8x1_sfine_flyingthings3d_subset_384x768",
        "build_flownetc",
        "_flow_pair",
        "2017",
        "DC",
    ),
    (
        "mmflow_flownet_flownetc_8x1_sfine_flyingthings3d_subset_384x768",
        "build_flownetc",
        "_flow_pair",
        "2015",
        "DC",
    ),
    (
        "mmflow_flownet_flownets_8x1_sfine_sintel_384x448",
        "build_flownet",
        "_flow_pair",
        "2015",
        "DC",
    ),
    (
        "mmflow_flownet_liteflownet_8x1_500k_flyingthings3d_subset_384x768",
        "build_liteflownet",
        "_flow_pair",
        "2018",
        "DC",
    ),
    (
        "mmflow_flownet_maskflownet_8x1_500k_flyingthings3d_subset_384x768",  # pragma: allowlist secret
        "build_maskflownet",
        "_flow_pair",
        "2020",
        "DC",
    ),
    (
        "mmflow_flownet_maskflownets_8x1_sfine_flyingthings3d_subset_384x768",
        "build_maskflownet",
        "_flow_pair",
        "2020",
        "DC",
    ),
    ("mmflow_flownets", "build_flownet", "_flow_pair", "2015", "DC"),
    ("mmflow_flownet2", "build_flownet2", "_flow_pair", "2017", "DC"),
    ("mmdet3d:free_anchor", "build_free_anchor", "_points", "2019", "DC"),
    (
        "mmdet3d_free_anchor_pointpillars_hv_fpn_head_free_anchor_sbn_all_8xb4_2x_nus_3d",
        "build_free_anchor",
        "_points",
        "2019",
        "DC",
    ),
    (
        "mmrotate_g_reppoints_g_reppoints_r50_fpn_1x_dota_le135",
        "build_points",
        "_image",
        "2019",
        "DC",
    ),
    ("mmagic:gca", "build_gca", "_matting_image", "2020", "DC"),
    ("mmagic_gca", "build_gca", "_matting_image", "2020", "DC"),
    ("mmagic_gca_baseline_r34_4xb10_200k_comp1k", "build_gca", "_matting_image", "2020", "DC"),
    ("mmagic_gca_matting", "build_gca", "_matting_image", "2020", "DC"),
    ("mmagic:ggan", "build_gan", "_latent", "2017", "DC"),
    ("mmagic_ggan", "build_gan", "_latent", "2017", "DC"),
    (
        "mmagic_ggan_ggan_lsgan_archi_lr1e_4_1xb128_20mimgs_lsun_bedroom_64x64",
        "build_gan",
        "_latent",
        "2017",
        "DC",
    ),
    ("mmagic:glean", "build_glean", "_small_image", "2021", "DC"),
    (
        "mmagic_glean_glean_in128out1024_fp16_4xb2_300k_ffhq_celeba_hq",
        "build_glean",
        "_small_image",
        "2021",
        "DC",
    ),
    ("mmrotate:gliding_vertex", "build_gliding_vertex", "_image", "2020", "DC"),
    (
        "mmrotate_gliding_vertex_gliding_vertex_r50_fpn_1x_dota_le90",
        "build_gliding_vertex",
        "_image",
        "2020",
        "DC",
    ),
    ("mmrotate_gliding_vertex_r50_fpn", "build_gliding_vertex", "_image", "2020", "DC"),
    ("mmpretrain:glip", "build_glip", "_image", "2021", "DC"),
    ("mmagic:global_local", "build_global_local", "_latent", "2017", "DC"),
    ("mmagic_global_local_gl_8xb12_celeba_256x256", "build_global_local", "_latent", "2017", "DC"),
    ("mmflow_gma", "build_gma", "_flow_pair", "2021", "DC"),
    ("mmdet3d:groupfree3d", "build_groupfree", "_points", "2021", "DC"),
    ("mmdet3d_groupfree3d", "build_groupfree", "_points", "2021", "DC"),
    (
        "mmdet3d_groupfree3d_groupfree3d_head_l12_o256_4xb8_scannet_seg",
        "build_groupfree",
        "_points",
        "2021",
        "DC",
    ),
    ("mmagic:guided_diffusion", "build_guided_diffusion", "_latent", "2021", "DC"),
    ("mmagic_guided_diffusion", "build_guided_diffusion", "_latent", "2021", "DC"),
    (
        "mmagic_guided_diffusion_adm_g_ddim25_8xb32_imagenet_256x256",
        "build_guided_diffusion",
        "_latent",
        "2021",
        "DC",
    ),
    ("mmrotate_g_reppoints_r50_fpn", "build_points", "_image", "2019", "DC"),
    ("mmrotate:gwd", "build_gwd", "_image", "2021", "DC"),
    ("mmrotate_gwd_rotated_retinanet", "build_gwd", "_image", "2021", "DC"),
    ("H2RBox", "build_h2rbox", "_image", "2022", "DC"),
    ("mmdet3d:h3dnet", "build_h3dnet", "_points", "2020", "DC"),
    ("mmdet3d_h3dnet_h3dnet_8xb3_scannet_seg", "build_h3dnet", "_points", "2020", "DC"),
    ("mmpose_hand_topdown_heatmap", "build_hand", "_image", "2019", "DC"),
    ("mmpose_hand_topdown_regression", "build_pose_regression", "_image", "2019", "DC"),
    ("HandPose-InterNet", "build_hand", "_image", "2019", "DC"),
    ("mmpose_hrformer", "build_hrformer", "_image", "2021", "DC"),
    ("FaceAlignment-HRNetV2-W18-AWing", "build_pose_heatmap", "_image", "2019", "DC"),
    ("HigherHRNet-W32", "build_higherhrnet", "_image", "2019", "DC"),
    ("HRNetPose-W32", "build_pose_heatmap", "_image", "2019", "DC"),
    ("HRNetPose-W48", "build_pose_heatmap", "_image", "2019", "DC"),
    ("LiteHRNet-18", "build_litehrnet", "_image", "2021", "DC"),
    ("mmaction2_i3d_r50_32x2_k400", "build_i3d", "_video", "2017", "DC"),
    ("mmaction2_recognition_i3d", "build_i3d", "_video", "2017", "DC"),
    ("mmaction:i3d", "build_i3d", "_video", "2017", "DC"),
    ("mmaction_i3d_r50", "build_i3d", "_video", "2017", "DC"),
    ("mmaction_i3d_nl_r50", "build_i3d_nl", "_video", "2018", "DC"),
    ("mmaction2_i3d_r50_nl_32x2_k400", "build_i3d_nl", "_video", "2018", "DC"),
    ("mmagic:iconvsr", "build_iconvsr", "_sr_video", "2021", "DC"),
    ("mmagic_iconvsr", "build_iconvsr", "_sr_video", "2021", "DC"),
    ("mmagic_iconvsr_iconvsr_2xb4_reds4", "build_iconvsr", "_sr_video", "2021", "DC"),
    ("mmpose:image_pose_lift", "build_pose_lift_tcn", "_pose2d_sequence", "2019", "DC"),
    ("mmpose_image_pose_lift", "build_pose_lift_tcn", "_pose2d_sequence", "2019", "DC"),
    ("mmdet3d:imvotenet", "build_imvotenet", "_imvotenet_input", "2020", "DC"),
    (
        "mmdet3d_imvotenet_imvotenet_faster_rcnn_r50_fpn_4xb2_sunrgbd_3d",
        "build_imvotenet",
        "_imvotenet_input",
        "2020",
        "DC",
    ),
]
