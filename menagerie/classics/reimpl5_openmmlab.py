"""Dependency-gated OpenMMLab/Paddle-style classics reconstructed in base torch.

Paper: OpenMMLab toolkits and the cited model-family papers.

This module covers install-hostile detector, pose, OCR, action, 3D, and generative
targets from ``dreimpl_5.txt`` with compact random-initialized PyTorch models.  The
models are intentionally small for TorchLens atlas rendering, but each keeps the
distinctive inference primitive of its family: rotated FPN heads, two-stage
rotated ROI transformation, RSN/RTMPose pose heads, SlowFast/action skeleton
streams, robust-scanner/SAR/SATRN OCR decoders, point-pillar/SECOND 3D heads, and
SAGAN/Stable-Diffusion-style generator or denoising U-Net blocks.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Convolution, batch normalization, and SiLU activation."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dims: int = 2) -> None:
        """Initialize the block.

        Parameters
        ----------
        in_ch:
            Input channel count.
        out_ch:
            Output channel count.
        stride:
            Convolution stride.
        dims:
            Spatial dimensionality, either 2 or 3.
        """

        super().__init__()
        conv: type[nn.Conv2d] | type[nn.Conv3d] = nn.Conv3d if dims == 3 else nn.Conv2d
        bn: type[nn.BatchNorm2d] | type[nn.BatchNorm3d] = (
            nn.BatchNorm3d if dims == 3 else nn.BatchNorm2d
        )
        self.conv = conv(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = bn(out_ch)

    def forward(self, x: Tensor) -> Tensor:
        """Apply convolutional feature extraction.

        Parameters
        ----------
        x:
            Input feature tensor.

        Returns
        -------
        Tensor
            Activated feature tensor.
        """

        return F.silu(self.bn(self.conv(x)))


class TinyFPN(nn.Module):
    """Small ResNet/FPN-style hierarchy shared by detector and segmentation heads."""

    def __init__(self, width: int = 24) -> None:
        """Initialize bottom-up stages and lateral FPN projections.

        Parameters
        ----------
        width:
            Base channel count.
        """

        super().__init__()
        self.stem = ConvBNAct(3, width, stride=2)
        self.c3 = ConvBNAct(width, width * 2, stride=2)
        self.c4 = ConvBNAct(width * 2, width * 4, stride=2)
        self.c5 = ConvBNAct(width * 4, width * 4, stride=2)
        self.l3 = nn.Conv2d(width * 2, width, 1)
        self.l4 = nn.Conv2d(width * 4, width, 1)
        self.l5 = nn.Conv2d(width * 4, width, 1)
        self.o3 = ConvBNAct(width, width)
        self.o4 = ConvBNAct(width, width)
        self.o5 = ConvBNAct(width, width)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Build a three-level feature pyramid.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            P3, P4, and P5 feature maps.
        """

        c2 = self.stem(x)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return self.o3(p3), self.o4(p4), self.o5(p5)


class RotatedDenseDetector(nn.Module):
    """Rotated RetinaNet/FCOS/ATSS/RepPoints/RTMDet detector over FPN features."""

    def __init__(self, mode: str, classes: int = 6, width: int = 24) -> None:
        """Initialize dense rotated detector heads.

        Parameters
        ----------
        mode:
            Head family: ``retina``, ``fcos``, ``atss``, ``reppoints``, ``rtmdet``,
            ``s2anet``, or ``sasm``.
        classes:
            Number of object classes.
        width:
            FPN channel count.
        """

        super().__init__()
        self.mode = mode
        self.fpn = TinyFPN(width)
        self.cls_tower = nn.Sequential(ConvBNAct(width, width), ConvBNAct(width, width))
        self.box_tower = nn.Sequential(ConvBNAct(width, width), ConvBNAct(width, width))
        anchors = 3 if mode in {"retina", "atss"} else 1
        self.cls = nn.Conv2d(width, anchors * classes, 3, padding=1)
        self.box = nn.Conv2d(width, anchors * 5, 3, padding=1)
        self.center = nn.Conv2d(width, anchors, 3, padding=1)
        self.points = nn.Conv2d(width, 18, 3, padding=1)
        self.refine = nn.Conv2d(width + 18, 5, 3, padding=1)
        self.atss_adaptive_sample_selection = nn.Conv2d(width, anchors, 3, padding=1)
        self.rtmdet_cspnext = nn.Sequential(ConvBNAct(width, width), ConvBNAct(width, width))
        self.rtmdet_large_kernel_dw = nn.Conv2d(width, width, 7, padding=3, groups=width)
        self.s2anet_anchor_refinement = nn.Conv2d(width, 5, 3, padding=1)
        self.s2anet_align_conv = nn.Conv2d(width, width, 3, padding=1)
        self.sasm_sample_assignment = nn.Conv2d(18, 1, 1)

    def _head_one(self, feat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run one pyramid-level dense head.

        Parameters
        ----------
        feat:
            FPN feature map.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class logits, rotated boxes, and centerness/quality.
        """

        if self.mode == "rtmdet":
            feat = F.silu(self.rtmdet_large_kernel_dw(self.rtmdet_cspnext(feat)))
        cls_feat = self.cls_tower(feat)
        box_feat = self.box_tower(feat)
        cls = self.cls(cls_feat)
        if self.mode == "s2anet":
            anchors = self.s2anet_anchor_refinement(box_feat)
            aligned = self.s2anet_align_conv(box_feat * torch.sigmoid(anchors[:, :1]))
            box = (
                self.box(aligned)
                + anchors.repeat(1, max(1, self.box.out_channels // 5), 1, 1)[
                    :, : self.box.out_channels
                ]
            )
            quality = torch.sigmoid(self.center(aligned))
        elif self.mode in {"reppoints", "sasm"}:
            pts = self.points(box_feat)
            box = self.refine(torch.cat([box_feat, pts], dim=1))
            if self.mode == "sasm":
                quality = torch.sigmoid(self.sasm_sample_assignment(pts))
            else:
                quality = torch.sigmoid(pts.pow(2).mean(dim=1, keepdim=True))
        else:
            box = self.box(box_feat)
            quality = torch.sigmoid(self.center(box_feat))
            if self.mode == "atss":
                quality = quality * torch.sigmoid(self.atss_adaptive_sample_selection(box_feat))
        if self.mode in {"s2anet", "sasm"}:
            angle = torch.tanh(box[:, 4::5]) * 1.5708
            box = torch.cat([F.softplus(box[:, :4]), angle], dim=1)
        else:
            box = torch.cat([F.softplus(box[:, :4]), torch.tanh(box[:, 4:5])], dim=1)
        return cls, box, quality

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Predict rotated boxes from all FPN levels.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, ...]
            Per-level class, rotated-box, and quality tensors.
        """

        outputs: list[Tensor] = []
        for feat in self.fpn(x):
            outputs.extend(self._head_one(feat))
        return tuple(outputs)  # type: ignore[return-value]


class RotatedRoIDetector(nn.Module):
    """ROI Transformer: RPN proposals, rotated ROI transform, and oriented ROI head."""

    def __init__(self, classes: int = 6, width: int = 24, proposals: int = 8) -> None:
        """Initialize two-stage rotated detector.

        Parameters
        ----------
        classes:
            Number of object classes.
        width:
            Backbone/FPN width.
        proposals:
            Number of top proposal summaries.
        """

        super().__init__()
        self.proposals = proposals
        self.fpn = TinyFPN(width)
        self.rpn = nn.Conv2d(width, width, 3, padding=1)
        self.rpn_obj = nn.Conv2d(width, 3, 1)
        self.rpn_box = nn.Conv2d(width, 15, 1)
        self.roi_align = nn.AdaptiveAvgPool2d((2, 2))
        self.theta = nn.Linear(width * 4, 6)
        self.roi_fc = nn.Sequential(nn.Linear(width * 4, width), nn.ReLU(), nn.Linear(width, width))
        self.cls = nn.Linear(width, classes)
        self.obox = nn.Linear(width, classes * 5)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run RPN, ROI transform, and rotated ROI head.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            RPN rotated boxes, class logits, and ROI oriented boxes.
        """

        p3, _, _ = self.fpn(x)
        rpn_feat = F.relu(self.rpn(p3))
        obj = self.rpn_obj(rpn_feat).flatten(2).max(dim=1).values
        idx = torch.topk(obj, self.proposals, dim=1).indices
        flat = p3.flatten(2).transpose(1, 2)
        gather = idx.unsqueeze(-1).expand(-1, -1, flat.shape[-1])
        roi = torch.gather(flat, 1, gather)
        roi_map = roi.transpose(1, 2).reshape(x.shape[0], -1, 2, self.proposals // 2)
        pooled = self.roi_align(roi_map).flatten(1)
        theta = self.theta(pooled).view(x.shape[0], 2, 3)
        grid = F.affine_grid(theta, roi_map.shape, align_corners=False)
        rotated = self.roi_align(F.grid_sample(roi_map, grid, align_corners=False)).flatten(1)
        feat = self.roi_fc(rotated).unsqueeze(1).expand(-1, self.proposals, -1)
        return self.rpn_box(rpn_feat), self.cls(feat), self.obox(feat)


class ResidualStepsNetwork(nn.Module):
    """RSN pose estimator with repeated residual-step refinement."""

    def __init__(self, keypoints: int = 17, width: int = 24, steps: int = 3) -> None:
        """Initialize RSN heatmap network.

        Parameters
        ----------
        keypoints:
            Number of keypoint heatmaps.
        width:
            Feature width.
        steps:
            Number of residual refinement steps.
        """

        super().__init__()
        self.stem = ConvBNAct(3, width, stride=2)
        self.residual_step_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "intra_level": ConvBNAct(width, width),
                        "inter_level": ConvBNAct(width, width),
                        "adapter": nn.Conv2d(width, width, 1),
                    }
                )
                for _ in range(steps)
            ]
        )
        self.pose_refine_machine = nn.Sequential(
            ConvBNAct(width * steps, width), ConvBNAct(width, width)
        )
        self.head = nn.Conv2d(width, keypoints, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict refined keypoint heatmaps.

        Parameters
        ----------
        x:
            RGB person crop.

        Returns
        -------
        Tensor
            Keypoint heatmaps.
        """

        feat = self.stem(x)
        state = feat
        states = []
        for block in self.residual_step_blocks:
            intra = block["intra_level"](state)
            inter = block["inter_level"](torch.roll(state, shifts=1, dims=3))
            state = state + block["adapter"](intra + inter)
            states.append(state)
        refined = self.pose_refine_machine(torch.cat(states, dim=1))
        return self.head(refined)


class RTMPoseNet(nn.Module):
    """RTMPose/RTMO pose network with CSP backbone and SimCC coordinate heads."""

    def __init__(self, keypoints: int = 17, width: int = 24, wholebody: bool = False) -> None:
        """Initialize compact RTMPose.

        Parameters
        ----------
        keypoints:
            Number of keypoints.
        width:
            Base channel width.
        wholebody:
            Whether to include the RTMW whole-body refinement branch.
        """

        super().__init__()
        self.wholebody = wholebody
        self.backbone = nn.Sequential(
            ConvBNAct(3, width, stride=2),
            ConvBNAct(width, width),
            ConvBNAct(width, width * 2, stride=2),
            ConvBNAct(width * 2, width * 2),
        )
        self.gau = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(width * 2, width * 2, 1), nn.Sigmoid()
        )
        self.heatmap = nn.Conv2d(width * 2, keypoints, 1)
        self.simcc_x = nn.Linear(width * 2, keypoints * 32)
        self.simcc_y = nn.Linear(width * 2, keypoints * 32)
        self.refine = nn.Linear(keypoints * 64, keypoints * 2)

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict heatmap and SimCC coordinates.

        Parameters
        ----------
        x:
            RGB pose crop.

        Returns
        -------
        tuple[Tensor, ...]
            Heatmap, x-axis SimCC logits, y-axis SimCC logits, and optional whole-body offsets.
        """

        feat = self.backbone(x)
        feat = feat * self.gau(feat)
        pooled = feat.mean(dim=(2, 3))
        heat = self.heatmap(feat)
        sx = self.simcc_x(pooled).view(x.shape[0], -1, 32)
        sy = self.simcc_y(pooled).view(x.shape[0], -1, 32)
        if self.wholebody:
            return heat, sx, sy, self.refine(torch.cat([sx.flatten(1), sy.flatten(1)], dim=1))
        return heat, sx, sy


class RTMOOneStagePose(nn.Module):
    """RTMO one-stage multi-person detector with pose vectors per location."""

    def __init__(self, keypoints: int = 17, width: int = 24) -> None:
        """Initialize RTMO dense pose detector.

        Parameters
        ----------
        keypoints:
            Number of body keypoints.
        width:
            Feature width.
        """

        super().__init__()
        self.cspnext = nn.Sequential(
            ConvBNAct(3, width, stride=2),
            ConvBNAct(width, width),
            ConvBNAct(width, width * 2, stride=2),
        )
        self.one_stage_center_heatmap = nn.Conv2d(width * 2, 1, 1)
        self.person_bbox = nn.Conv2d(width * 2, 4, 1)
        self.keypoint_offsets = nn.Conv2d(width * 2, keypoints * 2, 1)
        self.simcc_refine_x = nn.Conv2d(width * 2, keypoints, 1)
        self.simcc_refine_y = nn.Conv2d(width * 2, keypoints, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Predict dense person centers, boxes, and pose offsets.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            Center heatmap, boxes, keypoint offsets, and SimCC-style x/y refinements.
        """

        feat = self.cspnext(x)
        return (
            self.one_stage_center_heatmap(feat),
            F.softplus(self.person_bbox(feat)),
            self.keypoint_offsets(feat),
            self.simcc_refine_x(feat),
            self.simcc_refine_y(feat),
        )


class RegressionPoseRLE(nn.Module):
    """RegressionPose model with residual log-likelihood estimation outputs."""

    def __init__(self, keypoints: int = 17, width: int = 24) -> None:
        """Initialize RLE direct-regression pose model.

        Parameters
        ----------
        keypoints:
            Number of keypoints.
        width:
            Feature width.
        """

        super().__init__()
        self.backbone = nn.Sequential(
            ConvBNAct(3, width, stride=2),
            ConvBNAct(width, width, stride=2),
            ConvBNAct(width, width),
        )
        self.mean = nn.Linear(width, keypoints * 2)
        self.log_sigma = nn.Linear(width, keypoints * 2)
        self.residual_log_likelihood = nn.Sequential(
            nn.Linear(keypoints * 4, width),
            nn.ReLU(),
            nn.Linear(width, keypoints),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict coordinates, uncertainty, and residual likelihood.

        Parameters
        ----------
        x:
            RGB pose crop.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Coordinates, log sigma, and residual log-likelihood scores.
        """

        feat = self.backbone(x).mean(dim=(2, 3))
        mean = self.mean(feat).view(x.shape[0], -1, 2)
        log_sigma = self.log_sigma(feat).view(x.shape[0], -1, 2)
        rle = self.residual_log_likelihood(
            torch.cat([mean.flatten(1), log_sigma.flatten(1)], dim=1)
        )
        return mean, log_sigma, rle


class SimpleBaselinePose(nn.Module):
    """SimpleBaseline pose estimator with ResNet-style body and deconv head."""

    def __init__(self, keypoints: int = 17, width: int = 24, depth: int = 3) -> None:
        """Initialize the ResNet plus deconvolutional heatmap head.

        Parameters
        ----------
        keypoints:
            Number of keypoint heatmaps.
        width:
            Base channel width.
        depth:
            Number of residual body blocks.
        """

        super().__init__()
        self.stem = ConvBNAct(3, width, stride=2)
        self.blocks = nn.ModuleList([ConvBNAct(width, width) for _ in range(depth)])
        self.short = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(width, width, 4, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(width, width, 4, stride=2, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=False),
        )
        self.heatmap = nn.Conv2d(width, keypoints, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict keypoint heatmaps.

        Parameters
        ----------
        x:
            RGB person crop.

        Returns
        -------
        Tensor
            Keypoint heatmaps.
        """

        feat = self.stem(x)
        for block, short in zip(self.blocks, self.short, strict=True):
            feat = F.relu(short(feat) + block(feat))
        return self.heatmap(self.deconv(feat))


class RGBPoseConv3D(nn.Module):
    """RGBPose action recognizer fusing 3D RGB video and pose heatmap streams."""

    def __init__(self, classes: int = 8, width: int = 12) -> None:
        """Initialize two-stream 3D ConvNet.

        Parameters
        ----------
        classes:
            Number of action classes.
        width:
            Stream channel width.
        """

        super().__init__()
        self.rgb = nn.Sequential(
            ConvBNAct(3, width, stride=1, dims=3), ConvBNAct(width, width * 2, stride=2, dims=3)
        )
        self.pose = nn.Sequential(
            ConvBNAct(17, width, stride=1, dims=3), ConvBNAct(width, width * 2, stride=2, dims=3)
        )
        self.fuse = nn.Conv3d(width * 4, width * 2, 1)
        self.fc = nn.Linear(width * 2, classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a video with appended pose heatmaps.

        Parameters
        ----------
        x:
            Tensor with RGB and pose channels, shape ``(B, 20, T, H, W)``.

        Returns
        -------
        Tensor
            Action logits.
        """

        rgb = self.rgb(x[:, :3])
        pose = self.pose(x[:, 3:])
        fused = F.silu(self.fuse(torch.cat([rgb, pose], dim=1)))
        return self.fc(fused.mean(dim=(2, 3, 4)))


class SlowFastNet(nn.Module):
    """SlowFast/SlowOnly video recognizer with lateral fast-to-slow fusion."""

    def __init__(self, fast: bool = True, classes: int = 8, width: int = 12) -> None:
        """Initialize video recognition model.

        Parameters
        ----------
        fast:
            Whether to include the fast pathway.
        classes:
            Number of action classes.
        width:
            Base channel width.
        """

        super().__init__()
        self.fast = fast
        self.slow_path = nn.Sequential(
            ConvBNAct(3, width, dims=3), ConvBNAct(width, width * 2, stride=2, dims=3)
        )
        self.fast_path = nn.Sequential(
            ConvBNAct(3, width // 2, dims=3), ConvBNAct(width // 2, width, stride=2, dims=3)
        )
        self.lateral = nn.Conv3d(width, width * 2, 1)
        self.fc = nn.Linear(width * 2, classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify video clips.

        Parameters
        ----------
        x:
            RGB video tensor, shape ``(B, 3, T, H, W)``.

        Returns
        -------
        Tensor
            Action logits.
        """

        slow = self.slow_path(x[:, :, ::2])
        if self.fast:
            fast = self.fast_path(x)
            slow = slow + F.interpolate(self.lateral(fast), size=slow.shape[-3:], mode="nearest")
        return self.fc(slow.mean(dim=(2, 3, 4)))


class SkeletonGCN(nn.Module):
    """2s-AGCN skeleton action model with joint/bone adaptive graph streams."""

    def __init__(self, classes: int = 8, joints: int = 12, channels: int = 3) -> None:
        """Initialize graph convolutional skeleton recognizer.

        Parameters
        ----------
        classes:
            Number of action classes.
        joints:
            Number of skeleton joints.
        channels:
            Coordinate channels per joint.
        """

        super().__init__()
        self.adj_joint = nn.Parameter(torch.eye(joints))
        self.adj_bone = nn.Parameter(torch.eye(joints))
        self.edge_importance = nn.Parameter(torch.ones(2, joints, joints))
        self.joint_proj = nn.Linear(channels, 32)
        self.bone_proj = nn.Linear(channels, 32)
        self.motion_proj = nn.Linear(channels, 32)
        self.temporal_joint = nn.Conv1d(32, 32, 3, padding=1)
        self.temporal_bone = nn.Conv1d(32, 32, 3, padding=1)
        self.fc = nn.Linear(64, classes)

    def _stream(self, x: Tensor, adj: Tensor, proj: nn.Linear, temporal: nn.Conv1d) -> Tensor:
        """Run one adaptive graph stream.

        Parameters
        ----------
        x:
            Skeleton stream tensor.
        adj:
            Adaptive adjacency matrix.
        proj:
            Stream input projection.
        temporal:
            Temporal convolution.

        Returns
        -------
        Tensor
            Pooled stream feature.
        """

        h = proj(x)
        h = torch.einsum("btjc,jk->btkc", h, adj)
        return temporal(h.mean(dim=2).transpose(1, 2)).mean(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """Classify a skeleton sequence.

        Parameters
        ----------
        x:
            Skeleton tensor ``(B, T, J, C)``.

        Returns
        -------
        Tensor
            Action logits.
        """

        bone = x - torch.roll(x, shifts=1, dims=2)
        motion = x - torch.roll(x, shifts=1, dims=1)
        aj = torch.softmax(self.adj_joint * self.edge_importance[0], dim=-1)
        ab = torch.softmax(self.adj_bone * self.edge_importance[1], dim=-1)
        joint_feat = self._stream(x + 0.25 * motion, aj, self.joint_proj, self.temporal_joint)
        bone_feat = self._stream(bone, ab, self.bone_proj, self.temporal_bone)
        motion_feat = self.motion_proj(motion).mean(dim=(1, 2))
        return self.fc(torch.cat([joint_feat + motion_feat, bone_feat], dim=-1))


class RobustScannerOCR(nn.Module):
    """RobustScanner OCR with position-aware and glimpse attention decoders."""

    def __init__(self, vocab: int = 38, width: int = 32, steps: int = 5) -> None:
        """Initialize robust scanner recognizer.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        width:
            CNN/LSTM width.
        steps:
            Decoder steps.
        """

        super().__init__()
        self.steps = steps
        self.cnn = nn.Sequential(ConvBNAct(1, width, stride=2), ConvBNAct(width, width, stride=2))
        self.encoder = nn.LSTM(width, width, batch_first=True, bidirectional=True)
        self.pos = nn.Embedding(steps, width * 2)
        self.query = nn.Linear(width * 2, width * 2)
        self.conventional_glimpse_decoder = nn.GRUCell(width * 2, width * 2)
        self.dynamic_fusion_gate = nn.Linear(width * 4, width * 2)
        self.char = nn.Linear(width * 4, vocab)

    def forward(self, x: Tensor) -> Tensor:
        """Recognize scene text with hybrid attention.

        Parameters
        ----------
        x:
            Grayscale text image.

        Returns
        -------
        Tensor
            Character logits over decoding steps.
        """

        feat = self.cnn(x).mean(dim=2).transpose(1, 2)
        memory, _ = self.encoder(feat)
        hidden = memory.mean(dim=1)
        outs = []
        for step in range(self.steps):
            q = self.query(self.pos.weight[step]).view(1, 1, -1)
            attn = torch.softmax((memory * q).sum(dim=-1), dim=-1)
            positional_ctx = (attn.unsqueeze(-1) * memory).sum(dim=1)
            hidden = self.conventional_glimpse_decoder(positional_ctx, hidden)
            gate = torch.sigmoid(
                self.dynamic_fusion_gate(torch.cat([positional_ctx, hidden], dim=-1))
            )
            ctx = gate * positional_ctx + (1.0 - gate) * hidden
            pos = self.pos.weight[step].unsqueeze(0).expand(x.shape[0], -1)
            outs.append(self.char(torch.cat([ctx, pos], dim=-1)).unsqueeze(1))
        return torch.cat(outs, dim=1)


class SAROCR(nn.Module):
    """SAR OCR with holistic 2D encoder and recurrent attention decoder."""

    def __init__(self, vocab: int = 38, width: int = 32, steps: int = 5) -> None:
        """Initialize SAR recognizer.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        width:
            Feature width.
        steps:
            Decoder steps.
        """

        super().__init__()
        self.steps = steps
        self.cnn = nn.Sequential(ConvBNAct(1, width, stride=2), ConvBNAct(width, width, stride=2))
        self.holistic = nn.Linear(width, width)
        self.decoder = nn.LSTMCell(width * 2, width)
        self.spatial_attention = nn.Conv2d(width, width, 1)
        self.query = nn.Linear(width, width)
        self.out = nn.Linear(width, vocab)

    def forward(self, x: Tensor) -> Tensor:
        """Recognize text with SAR recurrent attention.

        Parameters
        ----------
        x:
            Grayscale text image.

        Returns
        -------
        Tensor
            Character logits.
        """

        fmap = self.cnn(x)
        seq = self.spatial_attention(fmap).flatten(2).transpose(1, 2)
        holistic = self.holistic(fmap.mean(dim=(2, 3)))
        h = holistic
        c = torch.zeros_like(h)
        outs = []
        for _ in range(self.steps):
            scores = torch.softmax((seq * self.query(h).unsqueeze(1)).sum(dim=-1), dim=-1)
            ctx = (scores.unsqueeze(-1) * seq).sum(dim=1)
            h, c = self.decoder(torch.cat([ctx, holistic], dim=-1), (h, c))
            outs.append(self.out(h).unsqueeze(1))
        return torch.cat(outs, dim=1)


class SATRNOCR(nn.Module):
    """SATRN OCR with 2D positional encoding and transformer encoder-decoder."""

    def __init__(self, vocab: int = 38, width: int = 32, steps: int = 5) -> None:
        """Initialize SATRN recognizer.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        width:
            Transformer width.
        steps:
            Output sequence length.
        """

        super().__init__()
        self.steps = steps
        self.cnn = nn.Sequential(ConvBNAct(1, width, stride=2), ConvBNAct(width, width, stride=2))
        self.row_pos = nn.Embedding(16, width)
        self.col_pos = nn.Embedding(32, width)
        layer = nn.TransformerEncoderLayer(
            width, nhead=4, dim_feedforward=width * 2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        dec_layer = nn.TransformerDecoderLayer(
            width, nhead=4, dim_feedforward=width * 2, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=1)
        self.query = nn.Embedding(steps, width)
        self.out = nn.Linear(width, vocab)

    def forward(self, x: Tensor) -> Tensor:
        """Recognize text with transformer attention.

        Parameters
        ----------
        x:
            Grayscale text image.

        Returns
        -------
        Tensor
            Character logits.
        """

        feat = self.cnn(x)
        h, w = feat.shape[-2:]
        rows = torch.arange(h, device=x.device).clamp_max(self.row_pos.num_embeddings - 1)
        cols = torch.arange(w, device=x.device).clamp_max(self.col_pos.num_embeddings - 1)
        pos = self.row_pos(rows).view(1, h, 1, -1) + self.col_pos(cols).view(1, 1, w, -1)
        tokens = (feat.permute(0, 2, 3, 1) + pos).flatten(1, 2)
        memory = self.encoder(tokens)
        q = self.query.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        return self.out(self.decoder(q, memory))


class SDMGRKIE(nn.Module):
    """SDMGR KIE with visual/text node streams and iterative edge reasoning."""

    def __init__(self, nodes: int = 6, classes: int = 5, width: int = 24) -> None:
        """Initialize KIE graph model.

        Parameters
        ----------
        nodes:
            Number of text boxes.
        classes:
            Entity classes.
        width:
            Node width.
        """

        super().__init__()
        self.nodes = nodes
        self.visual = nn.Linear(8, width)
        self.text = nn.Linear(8, width)
        self.geom = nn.Linear(4, width)
        self.edge = nn.Linear(width * 2 + 4, 1)
        self.edge_update = nn.GRUCell(width, width)
        self.node_update = nn.GRUCell(width, width)
        self.cls = nn.Linear(width, classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify document text nodes.

        Parameters
        ----------
        x:
            Node tensor containing visual features, text features, and box coordinates.

        Returns
        -------
        Tensor
            Per-node KIE logits.
        """

        visual = x[..., :8]
        text = x[..., 8:16]
        box = x[..., 16:20]
        h = F.relu(self.visual(visual) + self.text(text) + self.geom(box))
        edge_state = h.unsqueeze(2).expand(-1, -1, self.nodes, -1).contiguous()
        for _ in range(2):
            bi = box.unsqueeze(2).expand(-1, -1, self.nodes, -1)
            bj = box.unsqueeze(1).expand(-1, self.nodes, -1, -1)
            hi = h.unsqueeze(2).expand(-1, -1, self.nodes, -1)
            hj = h.unsqueeze(1).expand(-1, self.nodes, -1, -1)
            edge_input = torch.cat([hi, hj, bj - bi], dim=-1)
            adj = torch.softmax(self.edge(edge_input).squeeze(-1), dim=-1)
            msg = torch.matmul(adj, h)
            edge_state = self.edge_update(
                hj.reshape(-1, hj.shape[-1]),
                edge_state.reshape(-1, edge_state.shape[-1]),
            ).view_as(edge_state)
            h = self.node_update(
                msg.reshape(-1, msg.shape[-1]), h.reshape(-1, h.shape[-1])
            ).view_as(h)
        return self.cls(h)


class PointPillarsSECOND(nn.Module):
    """SECOND-family voxel detector with sparse-middle and variant heads."""

    def __init__(self, mode: str = "second", classes: int = 3, width: int = 32) -> None:
        """Initialize 3D detector.

        Parameters
        ----------
        mode:
            3D family variant.
        classes:
            Number of object classes.
        width:
            Feature width.
        """

        super().__init__()
        self.mode = mode
        self.vfe = nn.Sequential(nn.Linear(8, width), nn.ReLU(), nn.Linear(width, width))
        self.middle3d = nn.Sequential(
            ConvBNAct(width, width, dims=3),
            ConvBNAct(width, width, stride=1, dims=3),
        )
        self.bev = nn.Sequential(
            ConvBNAct(width, width), ConvBNAct(width, width, stride=2), ConvBNAct(width, width)
        )
        self.cls = nn.Conv2d(width, classes, 1)
        self.box = nn.Conv2d(width, 7, 1)
        self.point_structure = nn.Linear(width + 3, 4)
        self.shape_signature = nn.Linear(width + 3, 6)
        self.roi_grid = nn.Linear(width, width)
        self.roi_refine = nn.Linear(width * 2, 7)

    def _voxelize(self, points: Tensor) -> tuple[Tensor, Tensor]:
        """Encode points into a compact fixed voxel grid.

        Parameters
        ----------
        points:
            Point tensor ``(B, N, 4)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Voxel grid and per-point encoded features.
        """

        xyz = points[..., :3]
        centered = xyz - xyz.mean(dim=1, keepdim=True)
        point_input = torch.cat([points, centered, points[..., 3:4]], dim=-1)
        point_feat = self.vfe(point_input)
        voxel_feat = point_feat[:, :64].reshape(points.shape[0], 4, 4, 4, -1)
        voxel_grid = voxel_feat.permute(0, 4, 1, 2, 3).contiguous()
        return voxel_grid, point_feat

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Detect 3D boxes from voxelized point features.

        Parameters
        ----------
        points:
            Point tensor ``(B, N, 4)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            BEV class map, 3D box map, and family-specific auxiliary output.
        """

        voxel_grid, point_feat = self._voxelize(points)
        sparse = self.middle3d(voxel_grid)
        bev = self.bev(sparse.mean(dim=2))
        if self.mode == "sassd":
            aux = self.point_structure(torch.cat([point_feat, points[..., :3]], dim=-1))
        elif self.mode == "pvrcnn":
            keypoints = point_feat[:, ::8]
            roi_context = self.roi_grid(keypoints).mean(dim=1)
            bev_context = bev.mean(dim=(2, 3))
            aux = self.roi_refine(torch.cat([roi_context, bev_context], dim=-1))
        elif self.mode == "ssn":
            grouped = point_feat.reshape(points.shape[0], 8, 8, -1).mean(dim=2)
            centers = points[..., :3].reshape(points.shape[0], 8, 8, 3).mean(dim=2)
            aux = self.shape_signature(torch.cat([grouped, centers], dim=-1))
        else:
            aux = sparse.mean(dim=(2, 3, 4))
        return self.cls(bev), self.box(bev), aux


class SMOKEMono3D(nn.Module):
    """SMOKE monocular 3D detector with keypoint heatmap and 3D box regression."""

    def __init__(self, classes: int = 3, width: int = 24) -> None:
        """Initialize monocular 3D model.

        Parameters
        ----------
        classes:
            Number of object classes.
        width:
            Feature width.
        """

        super().__init__()
        self.backbone = nn.Sequential(
            ConvBNAct(3, width, stride=2), ConvBNAct(width, width, stride=2)
        )
        self.heatmap = nn.Conv2d(width, classes, 1)
        self.depth = nn.Conv2d(width, 1, 1)
        self.dim = nn.Conv2d(width, 3, 1)
        self.orient = nn.Conv2d(width, 2, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict monocular 3D boxes.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Heatmap, depth, dimensions, and orientation.
        """

        feat = self.backbone(x)
        return (
            self.heatmap(feat),
            F.softplus(self.depth(feat)),
            F.softplus(self.dim(feat)),
            self.orient(feat),
        )


class SPVCNN(nn.Module):
    """SPVCNN semantic segmentation with sparse-point and voxel branches."""

    def __init__(self, classes: int = 8, width: int = 24) -> None:
        """Initialize point-voxel segmentation model.

        Parameters
        ----------
        classes:
            Semantic classes.
        width:
            Feature width.
        """

        super().__init__()
        self.point = nn.Sequential(nn.Linear(4, width), nn.ReLU(), nn.Linear(width, width))
        self.voxel = nn.Sequential(ConvBNAct(width, width), ConvBNAct(width, width))
        self.fuse = nn.Linear(width * 2, width)
        self.cls = nn.Linear(width, classes)

    def forward(self, points: Tensor) -> Tensor:
        """Segment point-cloud points.

        Parameters
        ----------
        points:
            Point tensor ``(B, N, 4)``.

        Returns
        -------
        Tensor
            Per-point semantic logits.
        """

        point_feat = self.point(points)
        voxel = point_feat[:, :64].transpose(1, 2).reshape(points.shape[0], -1, 8, 8)
        voxel_feat = self.voxel(voxel).flatten(2).transpose(1, 2)
        voxel_feat = F.interpolate(
            voxel_feat.transpose(1, 2), size=points.shape[1], mode="nearest"
        ).transpose(1, 2)
        return self.cls(F.relu(self.fuse(torch.cat([point_feat, voxel_feat], dim=-1))))


class SelfAttention2d(nn.Module):
    """Non-local 2D self-attention used by SAGAN/SNGAN projection variants."""

    def __init__(self, channels: int) -> None:
        """Initialize attention projections.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.q = nn.Conv2d(channels, channels // 8, 1)
        self.k = nn.Conv2d(channels, channels // 8, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        """Apply image self-attention.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Attention-enhanced feature map.
        """

        b, c, h, w = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)
        k = self.k(x).flatten(2)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        v = self.v(x).flatten(2)
        return x + self.gamma * torch.bmm(v, attn.transpose(1, 2)).view(b, c, h, w)


class SAGANGenerator(nn.Module):
    """SAGAN/SNGAN projection generator with spectral-normalized residual upsampling."""

    def __init__(
        self, attention: bool = True, z_dim: int = 32, classes: int = 10, width: int = 32
    ) -> None:
        """Initialize generator.

        Parameters
        ----------
        attention:
            Whether to include SAGAN self-attention.
        z_dim:
            Latent dimension.
        classes:
            Class count for projection conditioning.
        width:
            Base channel width.
        """

        super().__init__()
        self.embed = nn.Embedding(classes, z_dim)
        self.project = nn.utils.spectral_norm(nn.Linear(z_dim, width * 4 * 4 * 4))
        self.up1 = nn.utils.spectral_norm(nn.ConvTranspose2d(width * 4, width * 2, 4, 2, 1))
        self.up2 = nn.utils.spectral_norm(nn.ConvTranspose2d(width * 2, width, 4, 2, 1))
        self.attn = SelfAttention2d(width) if attention else nn.Identity()
        self.out = nn.utils.spectral_norm(nn.Conv2d(width, 3, 3, padding=1))

    def forward(self, z: Tensor) -> Tensor:
        """Generate an image from latent plus class id.

        Parameters
        ----------
        z:
            Latent tensor with final column interpreted as class id proxy.

        Returns
        -------
        Tensor
            Generated RGB image.
        """

        y = z[:, -1].abs().long() % self.embed.num_embeddings
        latent = z + self.embed(y)
        x = self.project(latent).view(z.shape[0], -1, 4, 4)
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        return torch.tanh(self.out(self.attn(x)))


class ProjectionDiscriminator(nn.Module):
    """Projection SNGAN discriminator with class-conditional inner product."""

    def __init__(self, classes: int = 10, width: int = 32) -> None:
        """Initialize spectral-normalized discriminator.

        Parameters
        ----------
        classes:
            Class count.
        width:
            Base channel width.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, width, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=False),
            nn.utils.spectral_norm(nn.Conv2d(width, width * 2, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.score = nn.utils.spectral_norm(nn.Linear(width * 2, 1))
        self.embed = nn.utils.spectral_norm(nn.Embedding(classes, width * 2))

    def forward(self, image: Tensor, labels: Tensor) -> Tensor:
        """Score an image with the projection discriminator term.

        Parameters
        ----------
        image:
            Generated or real RGB image.
        labels:
            Class labels.

        Returns
        -------
        Tensor
            Conditional discriminator scores.
        """

        feat = self.conv(image).mean(dim=(2, 3))
        projection = (feat * self.embed(labels)).sum(dim=-1, keepdim=True)
        return self.score(feat) + projection


class ProjectionSNGAN(nn.Module):
    """Projection SNGAN wrapper tracing generator and conditional discriminator."""

    def __init__(self, z_dim: int = 32, classes: int = 10, width: int = 32) -> None:
        """Initialize generator and projection discriminator.

        Parameters
        ----------
        z_dim:
            Latent dimension.
        classes:
            Class count.
        width:
            Base channel width.
        """

        super().__init__()
        self.generator = SAGANGenerator(False, z_dim, classes, width)
        self.discriminator = ProjectionDiscriminator(classes, width)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Generate an image and score it conditionally.

        Parameters
        ----------
        z:
            Latent tensor with final column interpreted as class id proxy.

        Returns
        -------
        tuple[Tensor, Tensor]
            Generated image and projection discriminator score.
        """

        labels = z[:, -1].abs().long() % self.discriminator.embed.num_embeddings
        image = self.generator(z)
        return image, self.discriminator(image, labels)


class SinGANPyramid(nn.Module):
    """SinGAN multi-scale fully convolutional image pyramid generator."""

    def __init__(self, z_dim: int = 32, width: int = 24, stages: int = 3) -> None:
        """Initialize a pyramid of convolutional generators.

        Parameters
        ----------
        z_dim:
            Latent dimension.
        width:
            Hidden channel width.
        stages:
            Number of image-pyramid stages.
        """

        super().__init__()
        self.seed = nn.Linear(z_dim, width * 4 * 4)
        self.noise = nn.ModuleList([nn.Linear(z_dim, width) for _ in range(stages)])
        self.generators = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct(width + 3, width), ConvBNAct(width, width), nn.Conv2d(width, 3, 1)
                )
                for _ in range(stages)
            ]
        )

    def forward(self, z: Tensor) -> Tensor:
        """Synthesize an image through coarse-to-fine SinGAN stages.

        Parameters
        ----------
        z:
            Latent vector.

        Returns
        -------
        Tensor
            Generated RGB image.
        """

        feat = self.seed(z).view(z.shape[0], -1, 4, 4)
        image = torch.zeros(z.shape[0], 3, 4, 4, device=z.device, dtype=z.dtype)
        for noise_fc, generator in zip(self.noise, self.generators, strict=True):
            if image.shape[-1] < 16:
                image = F.interpolate(image, scale_factor=2, mode="nearest")
                feat = F.interpolate(feat, size=image.shape[-2:], mode="nearest")
            noise = noise_fc(z).view(z.shape[0], -1, 1, 1).expand_as(feat)
            residual = generator(torch.cat([feat + noise, image], dim=1))
            image = torch.tanh(image + residual)
        return image


class CrossAttention2d(nn.Module):
    """Cross-attention from latent image features to text-conditioning tokens."""

    def __init__(self, width: int) -> None:
        """Initialize image/text attention.

        Parameters
        ----------
        width:
            Feature width.
        """

        super().__init__()
        self.attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, image: Tensor, text: Tensor) -> Tensor:
        """Apply text cross-attention to a feature map.

        Parameters
        ----------
        image:
            Latent image feature map.
        text:
            Text conditioning tokens.

        Returns
        -------
        Tensor
            Cross-attended feature map.
        """

        b, c, h, w = image.shape
        tokens = image.flatten(2).transpose(1, 2)
        attended = self.attn(tokens, text, text, need_weights=False)[0]
        return self.norm(tokens + attended).transpose(1, 2).reshape(b, c, h, w)


class DiffusionUNet(nn.Module):
    """Stable-Diffusion-style denoising U-Net with time/text conditioning."""

    def __init__(
        self, control: bool = False, xl: bool = False, inpaint: bool = False, width: int = 24
    ) -> None:
        """Initialize compact denoising U-Net.

        Parameters
        ----------
        control:
            Whether to include a ControlNet residual branch.
        xl:
            Whether to use the SDXL dual-context branch.
        inpaint:
            Whether to consume mask/paint channels.
        width:
            Feature width.
        """

        super().__init__()
        self.control = control
        self.xl = xl
        in_ch = 9 if inpaint else 4
        self.time = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.text = nn.Linear(16, width * 4)
        self.text2 = nn.Linear(16, width * 4)
        self.down = nn.Sequential(ConvBNAct(in_ch, width), ConvBNAct(width, width, stride=2))
        self.cross1 = CrossAttention2d(width)
        self.mid_attn = SelfAttention2d(width)
        self.cross2 = CrossAttention2d(width)
        self.locked_down = nn.Sequential(ConvBNAct(in_ch, width), ConvBNAct(width, width, stride=2))
        self.trainable_down = nn.Sequential(
            ConvBNAct(in_ch, width), ConvBNAct(width, width, stride=2)
        )
        self.zero_conv = nn.Conv2d(width, width, 1)
        self.up = nn.ConvTranspose2d(width, width, 4, 2, 1)
        self.refiner = ConvBNAct(width, width) if xl else nn.Identity()
        self.out = nn.Conv2d(width, 4, 3, padding=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
        for param in self.locked_down.parameters():
            param.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        """Denoise latent image features.

        Parameters
        ----------
        x:
            Latent tensor. Extra channels are used for inpainting/control variants.

        Returns
        -------
        Tensor
            Predicted denoising residual.
        """

        latent = x
        cond = self.time(x.mean(dim=(1, 2, 3), keepdim=False).view(-1, 1))
        text_seed = torch.stack([x.flatten(1).mean(dim=1) for _ in range(16)], dim=1)
        text = self.text(text_seed).view(x.shape[0], 4, -1)
        if self.xl:
            text = text + self.text2(torch.flip(text_seed, dims=[1])).view(x.shape[0], 4, -1)
        h = self.down(latent)
        if self.control:
            locked = self.locked_down(latent).detach()
            trainable = self.trainable_down(latent)
            h = h + self.zero_conv(locked + trainable)
        h = self.cross1(h + cond.unsqueeze(-1).unsqueeze(-1), text)
        h = self.mid_attn(h)
        h = self.cross2(h, text)
        return self.out(self.refiner(F.relu(self.up(h))))


def example_image() -> Tensor:
    """Return a small RGB image input.

    Returns
    -------
    Tensor
        RGB image tensor.
    """

    return torch.randn(1, 3, 64, 64)


def example_pose_image() -> Tensor:
    """Return a small pose image input.

    Returns
    -------
    Tensor
        RGB pose crop.
    """

    return torch.randn(1, 3, 48, 48)


def example_video() -> Tensor:
    """Return a compact RGB video.

    Returns
    -------
    Tensor
        RGB video tensor.
    """

    return torch.randn(1, 3, 4, 32, 32)


def example_rgbpose_video() -> Tensor:
    """Return compact RGB plus pose-heatmap video.

    Returns
    -------
    Tensor
        Video tensor with 20 channels.
    """

    return torch.randn(1, 20, 4, 32, 32)


def example_skeleton() -> Tensor:
    """Return a compact skeleton sequence.

    Returns
    -------
    Tensor
        Skeleton tensor.
    """

    return torch.randn(1, 5, 12, 3)


def example_text_image() -> Tensor:
    """Return a small grayscale text image.

    Returns
    -------
    Tensor
        Text image tensor.
    """

    return torch.randn(1, 1, 32, 64)


def example_kie() -> Tensor:
    """Return compact document node features.

    Returns
    -------
    Tensor
        Node text features plus boxes.
    """

    return torch.randn(1, 6, 20)


def example_points() -> Tensor:
    """Return compact point-cloud features.

    Returns
    -------
    Tensor
        Point tensor.
    """

    return torch.randn(1, 64, 4)


def example_latent() -> Tensor:
    """Return a compact GAN latent vector.

    Returns
    -------
    Tensor
        Latent tensor.
    """

    return torch.randn(1, 32)


def example_latent_image() -> Tensor:
    """Return a compact latent diffusion input.

    Returns
    -------
    Tensor
        Latent image tensor.
    """

    return torch.randn(1, 4, 16, 16)


def example_inpaint_latent() -> Tensor:
    """Return compact inpainting latent with mask channels.

    Returns
    -------
    Tensor
        Inpainting latent tensor.
    """

    return torch.randn(1, 9, 16, 16)


def _build_rotated(mode: str) -> nn.Module:
    """Build a rotated dense detector.

    Parameters
    ----------
    mode:
        Rotated detector family.

    Returns
    -------
    nn.Module
        Evaluation-mode detector.
    """

    return RotatedDenseDetector(mode=mode).eval()


def build_rotated_retinanet() -> nn.Module:
    """Build rotated RetinaNet with FPN and focal-style dense heads."""

    return _build_rotated("retina")


def build_roi_transformer() -> nn.Module:
    """Build ROI Transformer with RPN and rotated ROI transformation."""

    return RotatedRoIDetector().eval()


def build_rotated_atss() -> nn.Module:
    """Build rotated ATSS with FPN, adaptive-quality head, and oriented boxes."""

    return _build_rotated("atss")


def build_rotated_fcos() -> nn.Module:
    """Build rotated FCOS with anchor-free centerness and oriented boxes."""

    return _build_rotated("fcos")


def build_rotated_reppoints() -> nn.Module:
    """Build rotated RepPoints with point-set refinement to oriented boxes."""

    return _build_rotated("reppoints")


def build_rotated_rtmdet() -> nn.Module:
    """Build rotated RTMDet with shared FPN and decoupled rotated dense head."""

    return _build_rotated("rtmdet")


def build_s2anet() -> nn.Module:
    """Build S2ANet with feature alignment and oriented dense boxes."""

    return _build_rotated("s2anet")


def build_sasm_reppoints() -> nn.Module:
    """Build SASM RepPoints with sampling-aware point quality and oriented boxes."""

    return _build_rotated("sasm")


def build_rsn18() -> nn.Module:
    """Build compact RSN-18 pose model."""

    return ResidualStepsNetwork(width=16, steps=2).eval()


def build_rsn50() -> nn.Module:
    """Build compact RSN-50 pose model."""

    return ResidualStepsNetwork(width=24, steps=3).eval()


def build_rtmo() -> nn.Module:
    """Build RTMO one-stage multi-person pose model with RTMPose head."""

    return RTMOOneStagePose(keypoints=17, width=24).eval()


def build_rtmpose_tiny() -> nn.Module:
    """Build RTMPose-Tiny with SimCC heads."""

    return RTMPoseNet(keypoints=17, width=16).eval()


def build_rtmpose_s() -> nn.Module:
    """Build RTMPose-S with SimCC heads."""

    return RTMPoseNet(keypoints=17, width=20).eval()


def build_rtmpose_m() -> nn.Module:
    """Build RTMPose-M with SimCC heads."""

    return RTMPoseNet(keypoints=17, width=24).eval()


def build_rtmpose_l() -> nn.Module:
    """Build RTMPose-L with SimCC heads."""

    return RTMPoseNet(keypoints=17, width=28).eval()


def build_rtmpose_x() -> nn.Module:
    """Build RTMPose-X with SimCC heads."""

    return RTMPoseNet(keypoints=17, width=32).eval()


def build_rtmpose_face() -> nn.Module:
    """Build RTMPose face model with dense facial landmarks."""

    return RTMPoseNet(keypoints=68, width=20).eval()


def build_rtmpose_hand() -> nn.Module:
    """Build RTMPose hand model with hand landmarks."""

    return RTMPoseNet(keypoints=21, width=20).eval()


def build_rtmpose_wholebody() -> nn.Module:
    """Build RTMPose whole-body model."""

    return RTMPoseNet(keypoints=33, width=24, wholebody=True).eval()


def build_rtmw() -> nn.Module:
    """Build RTMW whole-body pose model with refinement offsets."""

    return RTMPoseNet(keypoints=33, width=28, wholebody=True).eval()


def build_simcc_mobilenet() -> nn.Module:
    """Build SimCC MobileNet-style coordinate classifier."""

    return RTMPoseNet(keypoints=17, width=16).eval()


def build_simcc_resnet() -> nn.Module:
    """Build SimCC ResNet-style coordinate classifier."""

    return RTMPoseNet(keypoints=17, width=24).eval()


def build_simple3d_baseline() -> nn.Module:
    """Build SimpleBaseline 3D pose heatmap model."""

    return SimpleBaselinePose(keypoints=17, width=24, depth=3).eval()


def build_regression_pose_rle() -> nn.Module:
    """Build RegressionPose with residual log-likelihood estimation."""

    return RegressionPoseRLE().eval()


def build_rgbpose_conv3d() -> nn.Module:
    """Build RGBPose Conv3D action recognizer."""

    return RGBPoseConv3D().eval()


def build_slowfast() -> nn.Module:
    """Build SlowFast action recognizer."""

    return SlowFastNet(fast=True).eval()


def build_slowonly() -> nn.Module:
    """Build SlowOnly action recognizer."""

    return SlowFastNet(fast=False).eval()


def build_skeleton_gcn() -> nn.Module:
    """Build 2s-AGCN/ST-GCN skeleton recognizer."""

    return SkeletonGCN().eval()


def build_robust_scanner() -> nn.Module:
    """Build RobustScanner OCR recognizer."""

    return RobustScannerOCR().eval()


def build_sar() -> nn.Module:
    """Build SAR OCR recognizer."""

    return SAROCR().eval()


def build_satrn() -> nn.Module:
    """Build SATRN OCR recognizer."""

    return SATRNOCR().eval()


def build_sdmgr() -> nn.Module:
    """Build SDMGR KIE graph model."""

    return SDMGRKIE().eval()


def build_second() -> nn.Module:
    """Build SECOND/PointPillars 3D detector."""

    return PointPillarsSECOND(mode="second").eval()


def build_sassd() -> nn.Module:
    """Build SASSD 3D detector with auxiliary point supervision branch."""

    return PointPillarsSECOND(mode="sassd").eval()


def build_pvrcnn() -> nn.Module:
    """Build PV-RCNN-style point-voxel ROI detector."""

    return PointPillarsSECOND(mode="pvrcnn").eval()


def build_smoke() -> nn.Module:
    """Build SMOKE monocular 3D detector."""

    return SMOKEMono3D().eval()


def build_spvcnn() -> nn.Module:
    """Build SPVCNN point-voxel semantic segmentor."""

    return SPVCNN().eval()


def build_ssn() -> nn.Module:
    """Build SSN shape-signature 3D detector using SECOND-style BEV heads."""

    return PointPillarsSECOND(mode="ssn").eval()


def build_sagan() -> nn.Module:
    """Build SAGAN generator with self-attention and spectral normalization."""

    return SAGANGenerator(attention=True).eval()


def build_sngan_proj() -> nn.Module:
    """Build projection SNGAN with generator and conditional discriminator."""

    return ProjectionSNGAN().eval()


def build_singan() -> nn.Module:
    """Build SinGAN-style multi-scale fully convolutional pyramid generator."""

    return SinGANPyramid(z_dim=32, width=24).eval()


def build_stable_diffusion() -> nn.Module:
    """Build Stable-Diffusion-style latent denoising U-Net."""

    return DiffusionUNet().eval()


def build_stable_diffusion_controlnet() -> nn.Module:
    """Build Stable Diffusion with ControlNet residual conditioning."""

    return DiffusionUNet(control=True).eval()


def build_stable_diffusion_inpaint() -> nn.Module:
    """Build Stable Diffusion inpainting U-Net."""

    return DiffusionUNet(inpaint=True).eval()


def build_stable_diffusion_xl() -> nn.Module:
    """Build SDXL-style denoising U-Net with dual text context."""

    return DiffusionUNet(xl=True).eval()


Entry = tuple[str, str, str, str, str]

_ALIASES: dict[str, tuple[str, str, str, str, str]] = {
    "mmrotate_rotated_retinanet_rotated_retinanet_obb_csl_gaussian_r50_fpn_fp16_1x_dota_le90": (
        "build_rotated_retinanet",
        "example_image",
        "2017",
        "DET",
        "rotated RetinaNet FPN focal head with oriented box angle branch",
    ),
    "mmrotate:roi_trans": (
        "build_roi_transformer",
        "example_image",
        "2019",
        "DET",
        "ROI Transformer RPN plus rotated ROI transform head",
    ),
    "mmrotate_roi_transformer_r50_fpn": (
        "build_roi_transformer",
        "example_image",
        "2019",
        "DET",
        "ROI Transformer RPN plus rotated ROI transform head",
    ),
    "mmrotate:rotated_atss": (
        "build_rotated_atss",
        "example_image",
        "2020",
        "DET",
        "rotated ATSS FPN dense quality head",
    ),
    "mmrotate:rotated_fcos": (
        "build_rotated_fcos",
        "example_image",
        "2019",
        "DET",
        "rotated FCOS anchor-free centerness head",
    ),
    "mmrotate_rotated_fcos_rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90": (
        "build_rotated_fcos",
        "example_image",
        "2019",
        "DET",
        "rotated FCOS anchor-free centerness head",
    ),
    "mmrotate_rotated_fcos_rotated_fcos_kld_r50_fpn_1x_dota_le90": (
        "build_rotated_fcos",
        "example_image",
        "2019",
        "DET",
        "rotated FCOS anchor-free centerness head",
    ),
    "mmrotate:rotated_reppoints": (
        "build_rotated_reppoints",
        "example_image",
        "2019",
        "DET",
        "rotated RepPoints point-set refinement head",
    ),
    "mmrotate:rotated_rtmdet": (
        "build_rotated_rtmdet",
        "example_image",
        "2022",
        "DET",
        "RTMDet decoupled rotated dense head",
    ),
    "mmrotate:s2anet": (
        "build_s2anet",
        "example_image",
        "2020",
        "DET",
        "S2ANet aligned convolution style oriented dense head",
    ),
    "mmrotate:sasm_reppoints": (
        "build_sasm_reppoints",
        "example_image",
        "2020",
        "DET",
        "SASM RepPoints sampling-aware point refinement",
    ),
    "mmrotate_sasm_reppoints_r50_fpn": (
        "build_sasm_reppoints",
        "example_image",
        "2020",
        "DET",
        "SASM RepPoints sampling-aware point refinement",
    ),
    "mmrotate_sasm_reppoints_sasm_reppoints_r50_fpn_1x_dota_oc": (
        "build_sasm_reppoints",
        "example_image",
        "2020",
        "DET",
        "SASM RepPoints sampling-aware point refinement",
    ),
    "mmpose_rsn": (
        "build_rsn18",
        "example_pose_image",
        "2020",
        "POSE",
        "RSN residual-step pose refinement",
    ),
    "RSN-18": (
        "build_rsn18",
        "example_pose_image",
        "2020",
        "POSE",
        "RSN residual-step pose refinement",
    ),
    "RSN-50": (
        "build_rsn50",
        "example_pose_image",
        "2020",
        "POSE",
        "deeper RSN residual-step pose refinement",
    ),
    "mmpose:rtmo": (
        "build_rtmo",
        "example_pose_image",
        "2023",
        "POSE",
        "RTMO one-stage dense person center, box, and pose head",
    ),
    "mmpose_rtmo": (
        "build_rtmo",
        "example_pose_image",
        "2023",
        "POSE",
        "RTMO one-stage dense person center, box, and pose head",
    ),
    "mmpose:rtmpose": (
        "build_rtmpose_m",
        "example_pose_image",
        "2023",
        "POSE",
        "RTMPose CSP backbone plus GAU and SimCC heads",
    ),
    "mmpose_face_rtmpose": (
        "build_rtmpose_face",
        "example_pose_image",
        "2023",
        "POSE",
        "RTMPose face SimCC landmark head",
    ),
    "mmpose_hand_rtmpose": (
        "build_rtmpose_hand",
        "example_pose_image",
        "2023",
        "POSE",
        "RTMPose hand SimCC landmark head",
    ),
    "mmpose_rtmpose": (
        "build_rtmpose_m",
        "example_pose_image",
        "2023",
        "POSE",
        "RTMPose CSP backbone plus GAU and SimCC heads",
    ),
    "mmpose_wholebody_rtmpose": (
        "build_rtmpose_wholebody",
        "example_pose_image",
        "2023",
        "POSE",
        "RTMPose whole-body SimCC head",
    ),
    "RTMPose-L": (
        "build_rtmpose_l",
        "example_pose_image",
        "2023",
        "POSE",
        "RTMPose-L SimCC coordinate classifier",
    ),
    "RTMPose-M": (
        "build_rtmpose_m",
        "example_pose_image",
        "2023",
        "POSE",
        "RTMPose-M SimCC coordinate classifier",
    ),
    "RTMPose-S": (
        "build_rtmpose_s",
        "example_pose_image",
        "2023",
        "POSE",
        "RTMPose-S SimCC coordinate classifier",
    ),
    "RTMPose-Tiny": (
        "build_rtmpose_tiny",
        "example_pose_image",
        "2023",
        "POSE",
        "RTMPose-Tiny SimCC coordinate classifier",
    ),
    "RTMPose-X": (
        "build_rtmpose_x",
        "example_pose_image",
        "2023",
        "POSE",
        "RTMPose-X SimCC coordinate classifier",
    ),
    "mmpose_rtmw": (
        "build_rtmw",
        "example_pose_image",
        "2024",
        "POSE",
        "RTMW whole-body RTMPose with refinement offsets",
    ),
    "RTMW-X-CocoWholeBody": (
        "build_rtmw",
        "example_pose_image",
        "2024",
        "POSE",
        "RTMW whole-body RTMPose with refinement offsets",
    ),
    "mmpose:simcc": (
        "build_simcc_resnet",
        "example_pose_image",
        "2022",
        "POSE",
        "SimCC coordinate classification pose head",
    ),
    "mmpose_simcc": (
        "build_simcc_resnet",
        "example_pose_image",
        "2022",
        "POSE",
        "SimCC coordinate classification pose head",
    ),
    "SimCC-MobileNetV2": (
        "build_simcc_mobilenet",
        "example_pose_image",
        "2022",
        "POSE",
        "SimCC MobileNet-style coordinate classifier",
    ),
    "SimCC-ResNet50": (
        "build_simcc_resnet",
        "example_pose_image",
        "2022",
        "POSE",
        "SimCC ResNet-style coordinate classifier",
    ),
    "mmpose_simple3dbaseline": (
        "build_simple3d_baseline",
        "example_pose_image",
        "2017",
        "POSE",
        "SimpleBaseline ResNet plus deconvolutional heatmap head",
    ),
    "SimpleBaselinePose-ResNet101": (
        "build_simple3d_baseline",
        "example_pose_image",
        "2017",
        "POSE",
        "SimpleBaseline ResNet plus deconvolutional heatmap head",
    ),
    "SimpleBaselinePose-ResNet50": (
        "build_simple3d_baseline",
        "example_pose_image",
        "2017",
        "POSE",
        "SimpleBaseline ResNet plus deconvolutional heatmap head",
    ),
    "mmaction:rgbpose_conv3d": (
        "build_rgbpose_conv3d",
        "example_rgbpose_video",
        "2022",
        "VID",
        "RGBPose two-stream 3D ConvNet fusion",
    ),
    "mmaction2_skeleton_rgbpose_conv3d": (
        "build_rgbpose_conv3d",
        "example_rgbpose_video",
        "2022",
        "VID",
        "RGBPose two-stream 3D ConvNet fusion",
    ),
    "mmaction2_skeleton_2s_agcn_8xb16_bone_motion_u100_80e_ntu60_xsub_keypoint_2d": (
        "build_skeleton_gcn",
        "example_skeleton",
        "2019",
        "VID",
        "2s-AGCN joint and bone adaptive graph streams with motion",
    ),
    "mmaction2_skeleton_stgcn_8xb16_bone_motion_u100_80e_ntu120_xsub_keypoint_2d": (
        "build_skeleton_gcn",
        "example_skeleton",
        "2018",
        "VID",
        "ST-GCN skeleton graph convolution",
    ),
    "mmaction2_detection_slowfast": (
        "build_slowfast",
        "example_video",
        "2019",
        "VID",
        "SlowFast dual pathway lateral fusion",
    ),
    "mmaction2_recognition_slowfast": (
        "build_slowfast",
        "example_video",
        "2019",
        "VID",
        "SlowFast dual pathway lateral fusion",
    ),
    "mmaction2_slowfast_r101_8x8_k400": (
        "build_slowfast",
        "example_video",
        "2019",
        "VID",
        "SlowFast dual pathway lateral fusion",
    ),
    "mmaction2_slowfast_r50_4x16_k400": (
        "build_slowfast",
        "example_video",
        "2019",
        "VID",
        "SlowFast dual pathway lateral fusion",
    ),
    "mmaction2_slowfast_r50_8x8_k400": (
        "build_slowfast",
        "example_video",
        "2019",
        "VID",
        "SlowFast dual pathway lateral fusion",
    ),
    "mmaction:slowfast": (
        "build_slowfast",
        "example_video",
        "2019",
        "VID",
        "SlowFast dual pathway lateral fusion",
    ),
    "mmaction_slowfast_r50": (
        "build_slowfast",
        "example_video",
        "2019",
        "VID",
        "SlowFast dual pathway lateral fusion",
    ),
    "mmaction_slowonly_r50": (
        "build_slowonly",
        "example_video",
        "2019",
        "VID",
        "SlowOnly 3D ResNet pathway",
    ),
    "mmaction2_recognition_slowonly": (
        "build_slowonly",
        "example_video",
        "2019",
        "VID",
        "SlowOnly 3D ResNet pathway",
    ),
    "mmaction2_slowonly_r101_8x8_k400": (
        "build_slowonly",
        "example_video",
        "2019",
        "VID",
        "SlowOnly 3D ResNet pathway",
    ),
    "mmaction2_slowonly_r50_8x8_k400": (
        "build_slowonly",
        "example_video",
        "2019",
        "VID",
        "SlowOnly 3D ResNet pathway",
    ),
    "mmaction:slowonly": (
        "build_slowonly",
        "example_video",
        "2019",
        "VID",
        "SlowOnly 3D ResNet pathway",
    ),
    "mmocr:robust_scanner": (
        "build_robust_scanner",
        "example_text_image",
        "2020",
        "OCR",
        "RobustScanner hybrid position-aware and glimpse attention OCR",
    ),
    "mmocr_textrecog_robust_scanner": (
        "build_robust_scanner",
        "example_text_image",
        "2020",
        "OCR",
        "RobustScanner hybrid position-aware and glimpse attention OCR",
    ),
    "mmocr:sar": (
        "build_sar",
        "example_text_image",
        "2019",
        "OCR",
        "SAR holistic CNN encoder plus recurrent attention decoder",
    ),
    "mmocr_textrecog_sar": (
        "build_sar",
        "example_text_image",
        "2019",
        "OCR",
        "SAR holistic CNN encoder plus recurrent attention decoder",
    ),
    "mmocr:satrn": (
        "build_satrn",
        "example_text_image",
        "2020",
        "OCR",
        "SATRN 2D positional encoding plus transformer encoder-decoder",
    ),
    "mmocr_textrecog_satrn": (
        "build_satrn",
        "example_text_image",
        "2020",
        "OCR",
        "SATRN 2D positional encoding plus transformer encoder-decoder",
    ),
    "mmocr:sdmgr": (
        "build_sdmgr",
        "example_kie",
        "2021",
        "OCR",
        "SDMGR dual visual/text node streams with iterative edge message reasoning",
    ),
    "mmocr_kie_sdmgr": (
        "build_sdmgr",
        "example_kie",
        "2021",
        "OCR",
        "SDMGR dual visual/text node streams with iterative edge message reasoning",
    ),
    "mmdet3d:sassd": (
        "build_sassd",
        "example_points",
        "2020",
        "3D",
        "SA-SSD SECOND voxel sparse-middle detector with structure-aware point branch",
    ),
    "mmdet3d_sassd": (
        "build_sassd",
        "example_points",
        "2020",
        "3D",
        "SA-SSD SECOND voxel sparse-middle detector with structure-aware point branch",
    ),
    "mmdet3d_second_hv_parta2_secfpn_4x8_cyclic_80e_pcdet_kitti_3d_3class": (
        "build_second",
        "example_points",
        "2018",
        "3D",
        "SECOND voxel feature encoder plus sparse 3D middle encoder and BEV head",
    ),
    "mmdet3d_second_hv_pointpillars_secfpn_3x8_100e_det3d_kitti_3d_car": (
        "build_second",
        "example_points",
        "2019",
        "3D",
        "PointPillars/SECOND BEV detector",
    ),
    "mmdet3d_second_pointpillars_dv_secfpn_8xb6_160e_kitti_3d_car": (
        "build_second",
        "example_points",
        "2019",
        "3D",
        "PointPillars/SECOND BEV detector",
    ),
    "mmdet3d_second_pv_rcnn_8xb2_80e_kitti_3d_3class": (
        "build_pvrcnn",
        "example_points",
        "2020",
        "3D",
        "PV-RCNN voxel set abstraction and RoI-grid pooling refinement",
    ),
    "mmdet3d_second_sassd_8xb6_80e_kitti_3d_3class": (
        "build_sassd",
        "example_points",
        "2020",
        "3D",
        "SA-SSD SECOND voxel sparse-middle detector with structure-aware point branch",
    ),
    "openpcdet_second": (
        "build_second",
        "example_points",
        "2018",
        "3D",
        "SECOND voxel feature encoder plus sparse 3D middle encoder and BEV head",
    ),
    "mmdet3d:second": (
        "build_second",
        "example_points",
        "2018",
        "3D",
        "SECOND voxel feature encoder plus sparse 3D middle encoder and BEV head",
    ),
    "mmdet3d:smoke": (
        "build_smoke",
        "example_image",
        "2020",
        "3D",
        "SMOKE monocular keypoint heatmap and 3D box head",
    ),
    "mmdet3d_smoke_smoke_dla34_dlaneck_gn_all_4xb8_6x_kitti_mono3d": (
        "build_smoke",
        "example_image",
        "2020",
        "3D",
        "SMOKE monocular keypoint heatmap and 3D box head",
    ),
    "smoke_kitti": (
        "build_smoke",
        "example_image",
        "2020",
        "3D",
        "SMOKE monocular keypoint heatmap and 3D box head",
    ),
    "mmdet3d:spvcnn": (
        "build_spvcnn",
        "example_points",
        "2020",
        "3D",
        "SPVCNN point-voxel semantic segmentation",
    ),
    "mmdet3d_spvcnn_spvcnn_w16_8xb2_amp_15e_semantickitti": (
        "build_spvcnn",
        "example_points",
        "2020",
        "3D",
        "SPVCNN point-voxel semantic segmentation",
    ),
    "mmdet3d:ssn": (
        "build_ssn",
        "example_points",
        "2020",
        "3D",
        "SSN SECOND sparse-middle detector with shape-signature grouping head",
    ),
    "mmdet3d_ssn": (
        "build_ssn",
        "example_points",
        "2020",
        "3D",
        "SSN SECOND sparse-middle detector with shape-signature grouping head",
    ),
    "mmagic:sagan": (
        "build_sagan",
        "example_latent",
        "2019",
        "GAN",
        "SAGAN spectral-normalized generator with non-local self-attention",
    ),
    "mmagic_sagan": (
        "build_sagan",
        "example_latent",
        "2019",
        "GAN",
        "SAGAN spectral-normalized generator with non-local self-attention",
    ),
    "mmagic_sagan_sagan_128_cvt_studiogan": (
        "build_sagan",
        "example_latent",
        "2019",
        "GAN",
        "SAGAN spectral-normalized generator with non-local self-attention",
    ),
    "mmagic:sngan_proj": (
        "build_sngan_proj",
        "example_latent",
        "2018",
        "GAN",
        "Projection SNGAN generator plus class-conditional projection discriminator",
    ),
    "mmagic_sngan_proj": (
        "build_sngan_proj",
        "example_latent",
        "2018",
        "GAN",
        "Projection SNGAN generator plus class-conditional projection discriminator",
    ),
    "mmagic_sngan_proj_sngan_proj_cvt_studiogan_cifar10_32x32": (
        "build_sngan_proj",
        "example_latent",
        "2018",
        "GAN",
        "Projection SNGAN generator plus class-conditional projection discriminator",
    ),
    "mmagic:singan": (
        "build_singan",
        "example_latent",
        "2019",
        "GAN",
        "SinGAN multi-scale fully convolutional image-pyramid generator",
    ),
    "mmagic_singan": (
        "build_singan",
        "example_latent",
        "2019",
        "GAN",
        "SinGAN multi-scale fully convolutional image-pyramid generator",
    ),
    "mmagic_singan_singan_balloons": (
        "build_singan",
        "example_latent",
        "2019",
        "GAN",
        "SinGAN multi-scale fully convolutional image-pyramid generator",
    ),
    "mmagic:stable_diffusion": (
        "build_stable_diffusion",
        "example_latent_image",
        "2022",
        "DIFF",
        "Stable Diffusion latent U-Net with time embedding and text cross-attention",
    ),
    "mmagic:stable_diffusion_xl": (
        "build_stable_diffusion_xl",
        "example_latent_image",
        "2023",
        "DIFF",
        "SDXL latent U-Net with dual text encoders and refiner block",
    ),
    "mmagic_stable_diffusion_anythingv3_config": (
        "build_stable_diffusion",
        "example_latent_image",
        "2022",
        "DIFF",
        "Stable Diffusion latent U-Net with time embedding and text cross-attention",
    ),
    "mmagic_stable_diffusion_controlnet_1xb1_fill50k": (
        "build_stable_diffusion_controlnet",
        "example_latent_image",
        "2023",
        "DIFF",
        "ControlNet locked/trainable branches with zero-conv residual into cross-attention U-Net",
    ),
    "mmagic_stable_diffusion_stable_diffusion_ddim_denoisingunet_inpaint": (
        "build_stable_diffusion_inpaint",
        "example_inpaint_latent",
        "2022",
        "DIFF",
        "Stable Diffusion inpainting U-Net with mask channels",
    ),
    "mmagic_stable_diffusion_stable_diffusion_ddim_denoisingunet_tomesd_5e_1": (
        "build_stable_diffusion",
        "example_latent_image",
        "2023",
        "DIFF",
        "Stable Diffusion latent U-Net with time embedding and text cross-attention; ToMeSD is a sampling optimization",
    ),
    "mmagic_stable_diffusion_stable_diffusion_xl_ddim_denoisingunet": (
        "build_stable_diffusion_xl",
        "example_latent_image",
        "2023",
        "DIFF",
        "SDXL latent U-Net with dual text encoders and refiner block",
    ),
    "RegressionPose-ResNet50-RLE": (
        "build_regression_pose_rle",
        "example_pose_image",
        "2021",
        "POSE",
        "RegressionPose direct coordinate regression with residual log-likelihood estimation",
    ),
}


def _make_entry(name: str, spec: tuple[str, str, str, str, str]) -> Entry:
    """Convert an alias specification to a MENAGERIE entry.

    Parameters
    ----------
    name:
        Catalog target name.
    spec:
        Build/example/year/code/notes tuple.

    Returns
    -------
    Entry
        Registry entry tuple.
    """

    build_attr, example_attr, year, code, _notes = spec
    return name, build_attr, example_attr, year, code


def notes_for_catalog(name: str) -> str:
    """Return the compact architecture note for a target.

    Parameters
    ----------
    name:
        Target catalog name.

    Returns
    -------
    str
        Distinctive primitive note.
    """

    return _ALIASES[name][4]


def recipe_for_catalog(name: str) -> str:
    """Return an executable menagerie recipe for a target.

    Parameters
    ----------
    name:
        Target catalog name.

    Returns
    -------
    str
        Python code snippet that builds ``model``.
    """

    build_attr = _ALIASES[name][0]
    return f"from menagerie.classics.reimpl5_openmmlab import {build_attr}; model={build_attr}()"


def category_for_catalog(name: str) -> str:
    """Return a broad catalog category.

    Parameters
    ----------
    name:
        Target catalog name.

    Returns
    -------
    str
        Category string.
    """

    code = _ALIASES[name][3]
    mapping = {
        "DET": "vision/detection",
        "POSE": "vision/pose",
        "VID": "video/action",
        "OCR": "vision/text",
        "3D": "vision/3d",
        "GAN": "generative/diffusion-GAN-flow",
        "DIFF": "generative/diffusion-GAN-flow",
    }
    return mapping.get(code, "exotic/other")


def shape_for_catalog(example_attr: str) -> str:
    """Return a stable input-shape string for TSV rows.

    Parameters
    ----------
    example_attr:
        Name of the example function.

    Returns
    -------
    str
        Shape representation.
    """

    example_fn: Callable[[], Tensor] = globals()[example_attr]
    return str(tuple(example_fn().shape))


MENAGERIE_ENTRIES: list[Entry] = [_make_entry(name, spec) for name, spec in _ALIASES.items()]
