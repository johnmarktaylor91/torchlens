"""Dependency-free OpenMMLab/Paddle-style menagerie reconstructions.

This shard replaces install-gated detector, segmentor, optical-flow, pose,
OCR/KIE, matting, restoration, action-localization, and multimodal rows with
small random-init PyTorch modules.  The models are compact, but each keeps the
load-bearing architecture primitive named by its source family: FPN/RPN/ROI
heads for two-stage detectors, Retina/FCOS dense heads, rotated-box refinement
for mmrotate, image-to-voxel lifting for ImVoxelNet, point-image fusion for
MVXNet, PWC/LiteFlow pyramids for flow, IntegralPose soft-argmax, MSPN
multi-stage pose refinement, BMN/BSN temporal proposal heads, MViT multiscale
video attention, IndexNet-guided matting, LIIF implicit coordinate decoding,
NAFNet gated restoration blocks, and small VLM connector stacks for LLaVA,
MiniGPT-4, and OFA.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from menagerie.classics._mmocr_shared import (
    MASTERRecognizer,
    NRTRRecognizer,
    SDMGRKIE,
    kie_input,
    text_image,
)


def _cba(in_ch: int, out_ch: int, stride: int = 1, groups: int = 1) -> nn.Sequential:
    """Build a convolution, batch norm, activation block.

    Parameters
    ----------
    in_ch:
        Input channel count.
    out_ch:
        Output channel count.
    stride:
        Spatial stride.
    groups:
        Group count for grouped or depthwise convolution.

    Returns
    -------
    nn.Sequential
        Convolutional block.
    """

    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(),
    )


def _flat(x: Tensor) -> Tensor:
    """Flatten a feature map to ``(B, HW, C)``.

    Parameters
    ----------
    x:
        Feature map.

    Returns
    -------
    Tensor
        Flattened sequence.
    """

    return x.flatten(2).transpose(1, 2)


class TinyFPN(nn.Module):
    """Compact convolutional backbone with top-down FPN fusion."""

    def __init__(self, in_ch: int = 3, width: int = 24) -> None:
        """Initialize staged CNN and lateral FPN projections.

        Parameters
        ----------
        in_ch:
            Input channel count.
        width:
            Base channel width.
        """

        super().__init__()
        self.c2 = _cba(in_ch, width, 2)
        self.c3 = _cba(width, width * 2, 2)
        self.c4 = _cba(width * 2, width * 4, 2)
        self.c5 = _cba(width * 4, width * 4, 2)
        self.lat3 = nn.Conv2d(width * 2, width, 1)
        self.lat4 = nn.Conv2d(width * 4, width, 1)
        self.lat5 = nn.Conv2d(width * 4, width, 1)
        self.out3 = _cba(width, width)
        self.out4 = _cba(width, width)
        self.out5 = _cba(width, width)

    def forward(self, image: Tensor) -> list[Tensor]:
        """Return P3-P5 feature pyramid maps.

        Parameters
        ----------
        image:
            Image tensor.

        Returns
        -------
        list[Tensor]
            Feature maps ordered fine-to-coarse.
        """

        c2 = self.c2(image)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return [self.out3(p3), self.out4(p4), self.out5(p5)]


class RPNRoIHeads(nn.Module):
    """RPN plus deterministic top-location ROI classifier/regressor heads."""

    def __init__(self, channels: int = 24, classes: int = 5, proposals: int = 6) -> None:
        """Initialize RPN and ROI heads.

        Parameters
        ----------
        channels:
            FPN channel width.
        classes:
            Number of classes.
        proposals:
            Number of proposal features to gather.
        """

        super().__init__()
        self.proposals = proposals
        self.classes = classes
        self.rpn_conv = _cba(channels, channels)
        self.rpn_obj = nn.Conv2d(channels, 3, 1)
        self.rpn_box = nn.Conv2d(channels, 12, 1)
        self.roi_fc = nn.Sequential(
            nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, channels)
        )
        self.cls = nn.Linear(channels, classes)
        self.box = nn.Linear(channels, classes * 4)

    def roi_features(self, feat: Tensor, objectness: Tensor) -> Tensor:
        """Gather top objectness locations as ROI-aligned surrogates.

        Parameters
        ----------
        feat:
            FPN feature map.
        objectness:
            RPN objectness logits.

        Returns
        -------
        Tensor
            Proposal features.
        """

        scores = objectness.flatten(2).max(dim=1).values
        idx = torch.topk(scores, self.proposals, dim=1).indices
        flat = _flat(feat)
        gather = idx.unsqueeze(-1).expand(-1, -1, flat.shape[-1])
        return torch.gather(flat, 1, gather)

    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict RPN maps and ROI classification/regression.

        Parameters
        ----------
        feat:
            Fine FPN feature map.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            RPN boxes, ROI class logits, ROI box deltas, and ROI features.
        """

        rpn = self.rpn_conv(feat)
        obj = self.rpn_obj(rpn)
        rpn_box = _flat(self.rpn_box(rpn))
        roi = self.roi_fc(self.roi_features(feat, obj))
        return rpn_box, self.cls(roi), self.box(roi), roi


class OrientedRPNRoIHeads(RPNRoIHeads):
    """Oriented RPN with midpoint offsets and rotated RoI refinement."""

    def __init__(self, channels: int = 24, classes: int = 5, proposals: int = 6) -> None:
        """Initialize oriented proposal and rotated-RoI heads.

        Parameters
        ----------
        channels:
            Feature width.
        classes:
            Number of classes.
        proposals:
            Number of RoIs.
        """

        super().__init__(channels, classes, proposals)
        self.midpoint = nn.Conv2d(channels, 6, 1)
        self.rotated_refine = nn.Linear(channels, classes * 5)

    def forward(self, feat: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Predict oriented proposals and rotated RoI refinements.

        Parameters
        ----------
        feat:
            FPN feature map.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
            RPN boxes, midpoint offsets, class logits, rotated boxes, and RoI features.
        """

        rpn = self.rpn_conv(feat)
        obj = self.rpn_obj(rpn)
        midpoint = _flat(self.midpoint(rpn))
        rpn_box = _flat(self.rpn_box(rpn))
        roi = self.roi_fc(self.roi_features(feat, obj))
        return rpn_box, midpoint, self.cls(roi), self.rotated_refine(roi), roi


class TwoStageDetector(nn.Module):
    """FPN/RPN/RoI detector with optional mask and rotated-box branches."""

    def __init__(self, mask: bool = False, rotated: bool = False, cascade: bool = False) -> None:
        """Initialize two-stage detector.

        Parameters
        ----------
        mask:
            Whether to add a Mask R-CNN FCN mask head.
        rotated:
            Whether to predict rotated-box angle deltas.
        cascade:
            Whether to add a cascade refinement head.
        """

        super().__init__()
        self.backbone = TinyFPN()
        self.roi: RPNRoIHeads | OrientedRPNRoIHeads = (
            OrientedRPNRoIHeads() if rotated else RPNRoIHeads()
        )
        self.mask = nn.Linear(24, 5 * 8 * 8) if mask else None
        self.angle = nn.Linear(24, 5) if rotated else None
        self.cascade = nn.ModuleList([nn.Linear(24, 5 * 4) for _ in range(2)]) if cascade else None

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        """Run two-stage detection.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        tuple[Tensor, ...]
            Detector predictions.
        """

        feat = self.backbone(image)[0]
        roi_out = self.roi(feat)
        if self.angle is not None:
            rpn_box, midpoint, cls, box, roi = roi_out
            outputs: list[Tensor] = [rpn_box, midpoint, cls, box, self.angle(roi)]
        else:
            rpn_box, cls, box, roi = roi_out
            outputs = [rpn_box, cls, box]
        if self.mask is not None:
            outputs.append(self.mask(roi).reshape(image.shape[0], self.roi.proposals, 5, 8, 8))
        if self.cascade is not None:
            refined = box
            for head in self.cascade:
                refined = refined + 0.1 * head(roi)
                outputs.append(refined)
        return tuple(outputs)


class RetinaStyleDetector(nn.Module):
    """FPN detector with RetinaNet/FCOS-style dense prediction towers."""

    def __init__(self, rotated: bool = False, fcos3d: bool = False) -> None:
        """Initialize dense detector.

        Parameters
        ----------
        rotated:
            Whether to emit rotated box angles.
        fcos3d:
            Whether to emit depth, dimensions, and yaw for 3D FCOS.
        """

        super().__init__()
        self.backbone = TinyFPN()
        self.cls_tower = nn.Sequential(_cba(24, 24), _cba(24, 24), nn.Conv2d(24, 15, 3, padding=1))
        box_dim = 10 if fcos3d else (5 if rotated else 4)
        self.box_tower = nn.Sequential(
            _cba(24, 24), _cba(24, 24), nn.Conv2d(24, box_dim, 3, padding=1)
        )
        self.ctr = nn.Conv2d(24, 1, 3, padding=1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict dense classes, boxes, and quality logits.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Flattened dense predictions.
        """

        cls_outs: list[Tensor] = []
        box_outs: list[Tensor] = []
        ctr_outs: list[Tensor] = []
        for feat in self.backbone(image):
            cls_outs.append(_flat(self.cls_tower(feat)))
            box_outs.append(_flat(F.softplus(self.box_tower(feat))))
            ctr_outs.append(_flat(self.ctr(feat)))
        return torch.cat(cls_outs, 1), torch.cat(box_outs, 1), torch.cat(ctr_outs, 1)


class MonoFlexDetector(nn.Module):
    """MonoFlex with FCOS3D heads, edge fusion, and uncertainty depth ensemble."""

    def __init__(self) -> None:
        """Initialize monocular dense 3D detection heads."""

        super().__init__()
        self.backbone = TinyFPN()
        self.cls = nn.Conv2d(24, 5, 3, padding=1)
        self.box3d = nn.Conv2d(24, 10, 3, padding=1)
        self.edge = nn.Conv2d(24, 10, 3, padding=1)
        self.depth_direct = nn.Conv2d(24, 1, 3, padding=1)
        self.depth_keypoint = nn.Conv2d(24, 1, 3, padding=1)
        self.depth_uncertainty = nn.Conv2d(24, 2, 3, padding=1)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        """Predict MonoFlex dense 3D boxes with edge and uncertainty branches.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Classes, fused boxes, depth ensemble, and uncertainties.
        """

        feat = self.backbone(image)[0]
        cls = _flat(self.cls(feat))
        edge_fused_box = _flat(self.box3d(feat) + self.edge(feat))
        depth_stack = torch.cat((self.depth_direct(feat), self.depth_keypoint(feat)), dim=1)
        uncertainty = F.softplus(self.depth_uncertainty(feat))
        weights = torch.softmax(-uncertainty, dim=1)
        depth = (depth_stack * weights).sum(dim=1, keepdim=True)
        return cls, edge_fused_box, _flat(depth), _flat(uncertainty)


class S2ANetDetector(nn.Module):
    """S2ANet-style rotated detector with anchor refinement and feature alignment."""

    def __init__(self) -> None:
        """Initialize FPN, anchor-refine, alignment, and final ODM heads."""

        super().__init__()
        self.backbone = TinyFPN()
        self.arm = nn.Conv2d(24, 6, 3, padding=1)
        self.align = _cba(24 + 6, 24)
        self.odm_cls = nn.Conv2d(24, 5, 3, padding=1)
        self.odm_box = nn.Conv2d(24, 5, 3, padding=1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Run two-step rotated detection.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        tuple[Tensor, Tensor]
            Oriented class and rotated-box predictions.
        """

        feat = self.backbone(image)[0]
        refined_anchor = self.arm(feat)
        aligned = self.align(torch.cat((feat, refined_anchor), dim=1))
        return _flat(self.odm_cls(aligned)), _flat(self.odm_box(aligned))


class OrientedRepPointsDetector(nn.Module):
    """Oriented RepPoints detector with adaptive point sets and box conversion."""

    def __init__(self, points: int = 9) -> None:
        """Initialize point-set prediction and conversion heads.

        Parameters
        ----------
        points:
            Number of adaptive points per location.
        """

        super().__init__()
        self.points = points
        self.backbone = TinyFPN()
        self.point_init = nn.Conv2d(24, points * 2, 3, padding=1)
        self.point_refine = nn.Conv2d(24 + points * 2, points * 2, 3, padding=1)
        self.cls = nn.Conv2d(24, 5, 3, padding=1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict adaptive points and convert them to oriented boxes.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class logits, point sets, and oriented boxes.
        """

        feat = self.backbone(image)[0]
        init_points = self.point_init(feat)
        points = init_points + self.point_refine(torch.cat((feat, init_points), dim=1))
        pts = points.reshape(image.shape[0], self.points, 2, feat.shape[-2], feat.shape[-1])
        center = pts.mean(dim=1)
        spread = pts.std(dim=1)
        angle = torch.atan2(pts[:, -1, 1] - pts[:, 0, 1], pts[:, -1, 0] - pts[:, 0, 0]).unsqueeze(1)
        obb = torch.cat((center, spread, angle), dim=1)
        return _flat(self.cls(feat)), _flat(points), _flat(obb)


class GaussianRotatedRetinaNet(nn.Module):
    """Rotated RetinaNet with Gaussian KLD/KFIoU overlap primitive."""

    def __init__(self, mode: str = "kld") -> None:
        """Initialize dense rotated detector and Gaussian overlap mode.

        Parameters
        ----------
        mode:
            ``"kld"`` or ``"kfiou"``.
        """

        super().__init__()
        self.mode = mode
        self.base = RetinaStyleDetector(rotated=True)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict rotated boxes and Gaussian overlap quality.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        dict[str, Tensor]
            Class logits, boxes, centerness, and named Gaussian overlap quality.
        """

        cls, box, quality = self.base(image)
        mean = box[..., :2]
        cov = F.softplus(box[..., 2:4]) + 1.0e-3
        target_cov = torch.ones_like(cov)
        if self.mode == "kfiou":
            kalman_gain = cov / (cov + target_cov)
            innovation = mean * (1.0 - kalman_gain)
            overlap = torch.exp(
                -(innovation.pow(2) + kalman_gain * target_cov).sum(-1, keepdim=True)
            )
            overlap_key = "kalman_iou_overlap"
        else:
            kld = 0.5 * (
                cov / target_cov + mean.pow(2) / target_cov - 1.0 - torch.log(cov / target_cov)
            )
            overlap = torch.exp(-kld.sum(-1, keepdim=True))
            overlap_key = "gaussian_kld_overlap"
        return {"cls": cls, "rbox": box, "quality": quality, overlap_key: overlap}


class R3DetDetector(nn.Module):
    """R3Det-style refined single-stage rotated detector."""

    def __init__(self) -> None:
        """Initialize initial and refined rotated dense heads."""

        super().__init__()
        self.base = RetinaStyleDetector(rotated=True)
        self.refine = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 5))

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run rotated detection with box refinement.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class logits, initial boxes, and refined rotated boxes.
        """

        cls, box, quality = self.base(image)
        return cls, box, box + 0.1 * self.refine(box) * torch.sigmoid(quality)


class VoxelDetector(nn.Module):
    """Image/point-cloud 3D detector with BEV prediction heads."""

    def __init__(self, mode: str = "imvoxel") -> None:
        """Initialize 3D detector.

        Parameters
        ----------
        mode:
            ``imvoxel``, ``mvxnet``, or ``multiview``.
        """

        super().__init__()
        self.mode = mode
        self.image_backbone = TinyFPN(in_ch=3, width=16)
        self.point_mlp = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 16))
        self.voxel = nn.Sequential(
            nn.Conv3d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(16, 12, 1)
        self.point_vote = nn.Linear(16, 3)
        self.point_fuse = nn.Linear(32, 16)
        self.fcos_cls = nn.Conv2d(16, 5, 1)
        self.fcos_center = nn.Conv2d(16, 1, 1)
        self.fcos_attr3d = nn.Conv2d(16, 10, 1)

    def _lift_image_to_voxels(self, image: Tensor) -> Tensor:
        """Project image features into a compact 3D voxel volume.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        Tensor
            Voxel feature volume.
        """

        feat2d = self.image_backbone(image)[0]
        batch, channels, height, width = feat2d.shape
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, 4, device=image.device),
            torch.linspace(-1, 1, height, device=image.device),
            torch.linspace(-1, 1, width, device=image.device),
            indexing="ij",
        )
        grid = torch.stack((xx + 0.15 * zz, yy - 0.15 * zz), dim=-1)
        flat_grid = grid.reshape(1, 4 * height, width, 2).expand(batch, -1, -1, -1)
        sampled = F.grid_sample(feat2d, flat_grid, align_corners=False)
        return sampled.reshape(batch, channels, 4, height, width)

    def forward(self, x: Tensor | tuple[Tensor, Tensor]) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """Run lifted voxel or point-image fusion detection.

        Parameters
        ----------
        x:
            Image, multiview image batch, point tensor, or ``(points, image)``.

        Returns
        -------
        Tensor
            BEV 3D detection predictions.
        """

        if self.mode == "mvxnet":
            points, image = x if isinstance(x, tuple) else (x, None)
            if image is None:
                raise ValueError("MVXNet requires paired point and image inputs.")
            point_feat = self.point_mlp(points)
            votes = points[..., :3] + self.point_vote(point_feat)
            image_feat = F.adaptive_avg_pool2d(self.image_backbone(image)[0], (4, 4))
            point_grid = point_feat.transpose(1, 2).reshape(points.shape[0], 16, 4, 4)
            fused = torch.cat((point_grid, image_feat), dim=1).permute(0, 2, 3, 1)
            feat2d = self.point_fuse(fused).permute(0, 3, 1, 2)
            feat2d = feat2d + votes.mean(dim=1).view(points.shape[0], 3, 1, 1).mean(1, keepdim=True)
        elif self.mode == "multiview":
            bsz, views, channels, height, width = x.shape
            view_feat = self.image_backbone(x.reshape(bsz * views, channels, height, width))[0]
            feat2d = view_feat.reshape(
                bsz, views, 16, view_feat.shape[-2], view_feat.shape[-1]
            ).mean(1)
            cls = _flat(self.fcos_cls(feat2d))
            center = _flat(torch.sigmoid(self.fcos_center(feat2d)))
            attr3d = _flat(self.fcos_attr3d(feat2d))
            return cls * center, attr3d, center
        else:
            volume = self._lift_image_to_voxels(x)
            bev = self.voxel(volume).mean(2)
            return _flat(self.head(bev))
        volume = feat2d.unsqueeze(2).expand(-1, -1, 4, -1, -1)
        bev = self.voxel(volume).mean(2)
        return _flat(self.head(bev))


class PWCFlow(nn.Module):
    """PWC/LiteFlow-style pyramid flow estimator."""

    def __init__(self, mask: bool = False, recurrent: bool = False) -> None:
        """Initialize flow feature pyramid and decoders.

        Parameters
        ----------
        mask:
            Whether to predict occlusion masks as in MaskFlowNet.
        recurrent:
            Whether to apply recurrent residual refinement as in IRR-PWC.
        """

        super().__init__()
        self.mask = mask
        self.recurrent = recurrent
        self.pyr = TinyFPN(in_ch=3, width=16)
        self.hidden = _cba(16 * 3 + 2, 32)
        self.decode = nn.Sequential(_cba(16 * 3 + 2, 32), nn.Conv2d(32, 2, 3, padding=1))
        self.mask_head = nn.Conv2d(32, 1, 3, padding=1)
        self.subpixel = nn.Sequential(_cba(16 * 2 + 2, 32), nn.Conv2d(32, 2, 3, padding=1))
        self.regularizer = nn.Sequential(_cba(16 + 2, 16), nn.Conv2d(16, 2, 3, padding=1))

    def _warp(self, image: Tensor, flow: Tensor) -> Tensor:
        """Warp an image with a normalized flow field.

        Parameters
        ----------
        image:
            Feature map to sample.
        flow:
            Pixel-space flow.

        Returns
        -------
        Tensor
            Warped feature map.
        """

        bsz, _, height, width = image.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=image.device),
            torch.linspace(-1, 1, width, device=image.device),
            indexing="ij",
        )
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(bsz, 1, 1, 1)
        norm = torch.stack(
            (flow[:, 0] / max(width - 1, 1) * 2, flow[:, 1] / max(height - 1, 1) * 2), dim=-1
        )
        return F.grid_sample(image, grid + norm, align_corners=True)

    def forward(self, frames: Tensor) -> tuple[Tensor, ...]:
        """Estimate optical flow between two frames.

        Parameters
        ----------
        frames:
            Tensor with shape ``(B, 2, 3, H, W)``.

        Returns
        -------
        tuple[Tensor, ...]
            Flow, and optionally occlusion mask.
        """

        first, second = frames[:, 0], frames[:, 1]
        pyr1 = self.pyr(first)
        pyr2 = self.pyr(second)
        coarse = pyr1[-1]
        flow = torch.zeros(
            first.shape[0], 2, coarse.shape[-2], coarse.shape[-1], device=first.device
        )
        hidden = torch.zeros(
            first.shape[0], 32, coarse.shape[-2], coarse.shape[-1], device=first.device
        )
        steps = 3 if self.recurrent else 1
        occlusion: Tensor | None = None
        for feat1, feat2 in zip(reversed(pyr1), reversed(pyr2), strict=True):
            flow = F.interpolate(flow, size=feat1.shape[-2:], mode="bilinear", align_corners=False)
            hidden = F.interpolate(
                hidden, size=feat1.shape[-2:], mode="bilinear", align_corners=False
            )
            warped = self._warp(feat2, flow)
            if self.mask and occlusion is not None:
                warped = warped * F.interpolate(occlusion, size=warped.shape[-2:], mode="nearest")
            corr = feat1 * warped
            for _ in range(steps):
                inp = torch.cat((feat1, warped, corr, flow), dim=1)
                hidden = hidden + self.hidden(inp)
                delta = self.decode(inp)
                flow = flow + delta
            subpixel = self.subpixel(torch.cat((feat1, warped, flow), dim=1))
            flow = flow + subpixel + self.regularizer(torch.cat((feat1, flow), dim=1))
            if self.mask:
                occlusion = torch.sigmoid(self.mask_head(hidden))
        outputs: list[Tensor] = [
            F.interpolate(flow, size=first.shape[-2:], mode="bilinear", align_corners=False)
        ]
        if self.mask:
            outputs.append(
                occlusion if occlusion is not None else torch.sigmoid(self.mask_head(hidden))
            )
        return tuple(outputs)


class IntegralPoseNet(nn.Module):
    """Heatmap pose model with integral soft-argmax coordinates."""

    def __init__(self, joints: int = 17, three_d: bool = False) -> None:
        """Initialize pose backbone and heatmap head.

        Parameters
        ----------
        joints:
            Joint count.
        three_d:
            Whether to produce depth logits for 3D hand pose.
        """

        super().__init__()
        self.joints = joints
        self.three_d = three_d
        self.backbone = nn.Sequential(_cba(3, 24, 2), _cba(24, 32, 2), _cba(32, 32))
        self.heatmap = nn.Conv2d(32, joints, 1)
        self.depth = nn.Linear(32, joints) if three_d else None

    def forward(self, image: Tensor) -> Tensor:
        """Predict integral joint coordinates.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        Tensor
            Joint coordinates.
        """

        feat = self.backbone(image)
        heat = self.heatmap(feat)
        prob = torch.softmax(heat.flatten(2), dim=-1).reshape_as(heat)
        height, width = heat.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=image.device),
            torch.linspace(0, 1, width, device=image.device),
            indexing="ij",
        )
        x_coord = (prob * xx).sum(dim=(2, 3))
        y_coord = (prob * yy).sum(dim=(2, 3))
        coords = torch.stack((x_coord, y_coord), dim=-1)
        if self.depth is not None:
            depth = self.depth(feat.mean(dim=(2, 3))).unsqueeze(-1)
            coords = torch.cat((coords, depth), dim=-1)
        return coords


class TopDownHeatmapNet(nn.Module):
    """Top-down crop pose estimator that returns heatmaps rather than coordinates."""

    def __init__(self, joints: int = 17) -> None:
        """Initialize a compact deconvolutional heatmap head.

        Parameters
        ----------
        joints:
            Number of keypoint heatmaps.
        """

        super().__init__()
        self.backbone = nn.Sequential(_cba(3, 24, 2), _cba(24, 32, 2), _cba(32, 32))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 24, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.heatmap = nn.Conv2d(24, joints, 1)

    def forward(self, image: Tensor) -> Tensor:
        """Predict per-joint heatmap logits.

        Parameters
        ----------
        image:
            Top-down person crop.

        Returns
        -------
        Tensor
            Heatmap logits.
        """

        return self.heatmap(self.deconv(self.backbone(image)))


class InterNetHand3D(nn.Module):
    """InterNet/InterHand3D with joint, hand-type, and relative-root heads."""

    def __init__(self, joints: int = 42) -> None:
        """Initialize the interacting-hand 3D pose heads.

        Parameters
        ----------
        joints:
            Number of hand joints across both hands.
        """

        super().__init__()
        self.joints = joints
        self.backbone = nn.Sequential(
            _cba(3, 24, 2), _cba(24, 32, 2), _cba(32, 48, 2), _cba(48, 48)
        )
        self.heatmap = nn.Conv2d(48, joints, 1)
        self.joint_depth = nn.Linear(48, joints)
        self.hand_type = nn.Linear(48, 2)
        self.root_depth = nn.Linear(48, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict 3D joint coordinates, hand type, and root-relative depth.

        Parameters
        ----------
        image:
            Interacting-hand image crop.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Joint coordinates, right/left hand-type logits, and relative root depth.
        """

        feat = self.backbone(image)
        heat = self.heatmap(feat)
        prob = torch.softmax(heat.flatten(2), dim=-1).reshape_as(heat)
        height, width = heat.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=image.device),
            torch.linspace(0, 1, width, device=image.device),
            indexing="ij",
        )
        x_coord = (prob * xx).sum(dim=(2, 3))
        y_coord = (prob * yy).sum(dim=(2, 3))
        pooled = feat.mean(dim=(2, 3))
        depth = self.joint_depth(pooled).unsqueeze(-1)
        coords = torch.cat((torch.stack((x_coord, y_coord), dim=-1), depth), dim=-1)
        return coords, self.hand_type(pooled), self.root_depth(pooled)


class MSPN(nn.Module):
    """MSPN with U-shaped multi-scale single-stage modules."""

    def __init__(self, stages: int = 2, joints: int = 17) -> None:
        """Initialize MSPN stages.

        Parameters
        ----------
        stages:
            Number of refinement stages.
        joints:
            Joint count.
        """

        super().__init__()
        self.stem = _cba(3, 24, 2)
        self.downs = nn.ModuleList(
            [nn.Sequential(_cba(24 + joints, 32, 2), _cba(32, 32)) for _ in range(stages)]
        )
        self.ups = nn.ModuleList(
            [
                nn.Sequential(_cba(32 + 24 + joints, 24), nn.Conv2d(24, joints, 1))
                for _ in range(stages)
            ]
        )

    def forward(self, image: Tensor) -> Tensor:
        """Predict refined heatmaps.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        Tensor
            Final-stage heatmaps.
        """

        feat = self.stem(image)
        heat = torch.zeros(image.shape[0], 17, feat.shape[-2], feat.shape[-1], device=image.device)
        for down, up in zip(self.downs, self.ups, strict=True):
            low = down(torch.cat((feat, heat), dim=1))
            low = F.interpolate(low, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            heat = up(torch.cat((low, feat, heat), dim=1))
        return heat


class MotionBERTPose(nn.Module):
    """MotionBERT DSTformer with dual spatial/temporal streams."""

    def __init__(self, joints: int = 17, dim: int = 48) -> None:
        """Initialize joint embedding and temporal transformer.

        Parameters
        ----------
        joints:
            Joint count.
        dim:
            Transformer width.
        """

        super().__init__()
        self.embed = nn.Linear(3, dim)
        self.joint_pos = nn.Parameter(torch.zeros(1, joints, dim))
        self.time_pos = nn.Parameter(torch.zeros(1, 8, dim))
        spatial_layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        temporal_layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        self.spatial = nn.TransformerEncoder(spatial_layer, 2)
        self.temporal = nn.TransformerEncoder(temporal_layer, 2)
        self.fuse = nn.Linear(dim * 2, dim)
        self.head = nn.Linear(dim, 3)
        self.joints = joints

    def forward(self, pose: Tensor) -> Tensor:
        """Refine a sequence of 3D poses.

        Parameters
        ----------
        pose:
            Pose tensor ``(B, T, J, 3)``.

        Returns
        -------
        Tensor
            Refined poses.
        """

        bsz, steps, joints, _ = pose.shape
        tokens = self.embed(pose)
        spatial_tokens = tokens + self.joint_pos[:, :joints].unsqueeze(1)
        spatial_tokens = spatial_tokens.reshape(bsz * steps, joints, -1)
        spatial = self.spatial(spatial_tokens).reshape(bsz, steps, joints, -1)
        temporal_tokens = tokens + self.time_pos[:, :steps].unsqueeze(2)
        temporal_tokens = temporal_tokens.permute(0, 2, 1, 3).reshape(bsz * joints, steps, -1)
        temporal = (
            self.temporal(temporal_tokens).reshape(bsz, joints, steps, -1).permute(0, 2, 1, 3)
        )
        fused = self.fuse(torch.cat((spatial, temporal), dim=-1))
        out = self.head(fused).reshape(bsz, steps, joints, 3)
        return out


class MViTVideo(nn.Module):
    """MViT/MViTv2-style video transformer with token pooling hierarchy."""

    def __init__(self, classes: int = 10, dim: int = 48) -> None:
        """Initialize 3D patch embedding and transformer.

        Parameters
        ----------
        classes:
            Number of classes.
        dim:
            Token width.
        """

        super().__init__()
        self.patch = nn.Conv3d(3, dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.attn1 = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.attn2 = nn.MultiheadAttention(dim * 2, 4, batch_first=True)
        self.proj = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim * 2)
        self.pool = nn.Linear(dim * 2, classes)

    def forward(self, video: Tensor) -> Tensor:
        """Classify a video clip.

        Parameters
        ----------
        video:
            Video tensor ``(B, C, T, H, W)``.

        Returns
        -------
        Tensor
            Class logits.
        """

        feat = self.patch(video)
        batch, channels, frames, height, width = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        attended, _ = self.attn1(tokens, tokens, tokens)
        tokens = tokens + attended
        feat = tokens.transpose(1, 2).reshape(batch, channels, frames, height, width)
        pooled = F.avg_pool3d(feat, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        pooled_tokens = self.proj(pooled.flatten(2).transpose(1, 2))
        attended, _ = self.attn2(pooled_tokens, pooled_tokens, pooled_tokens)
        tokens = self.norm(pooled_tokens + attended)
        return self.pool(tokens.mean(1))


class TemporalProposalNet(nn.Module):
    """BMN/BSN temporal action proposal network."""

    def __init__(self, mode: str = "bmn") -> None:
        """Initialize temporal proposal model.

        Parameters
        ----------
        mode:
            Proposal-family mode.
        """

        super().__init__()
        self.mode = mode
        self.tem = nn.Sequential(nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 2, 1))
        self.boundary = nn.Sequential(
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 16, 1)
        )
        self.pem = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, feats: Tensor) -> tuple[Tensor, Tensor]:
        """Predict temporal boundaries and proposal scores.

        Parameters
        ----------
        feats:
            Temporal features ``(B, T, C)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Boundary/actionness logits and proposal confidence matrix.
        """

        x = feats.transpose(1, 2)
        boundary = self.tem(x)
        enc = self.boundary(x).transpose(1, 2)
        start = enc.unsqueeze(2).expand(-1, -1, enc.shape[1], -1)
        end = enc.unsqueeze(1).expand(-1, enc.shape[1], -1, -1)
        pair = torch.cat((start, end), dim=-1)
        score = self.pem(pair).squeeze(-1)
        if self.mode == "tcanet":
            score = score + score.transpose(1, 2)
        return boundary, score


class DRNProposalRefiner(nn.Module):
    """DRN-style dense proposal refinement network."""

    def __init__(self) -> None:
        """Initialize DRN local/global proposal refinement heads."""

        super().__init__()
        self.base = TemporalProposalNet("bsn")
        self.local = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1)
        )
        self.global_ctx = nn.MultiheadAttention(16, 4, batch_first=True)
        self.refine = nn.Conv2d(16, 1, 1)

    def forward(self, feats: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Refine BSN proposal scores with DRN proposal context.

        Parameters
        ----------
        feats:
            Temporal features ``(B, T, C)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Boundary logits, coarse proposal scores, and refined proposal scores.
        """

        boundary, score = self.base(feats)
        local = self.local(score.unsqueeze(1))
        tokens = local.flatten(2).transpose(1, 2)
        global_tokens, _ = self.global_ctx(tokens, tokens, tokens)
        refined = local + global_tokens.transpose(1, 2).reshape_as(local)
        return boundary, score, score + self.refine(refined).squeeze(1)


class TCANetProposalRefiner(nn.Module):
    """TCANet with LGTE context encoder and progressive boundary regression."""

    def __init__(self, stages: int = 2) -> None:
        """Initialize TCANet proposal refiners.

        Parameters
        ----------
        stages:
            Number of progressive boundary-regression stages.
        """

        super().__init__()
        self.tem = nn.Sequential(nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 2, 1))
        self.local = nn.Conv1d(16, 32, 3, padding=1)
        self.global_attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.boundary_regressors = nn.ModuleList([nn.Linear(64, 2) for _ in range(stages)])
        self.score = nn.Linear(64, 1)

    def forward(self, feats: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict and progressively refine temporal proposals.

        Parameters
        ----------
        feats:
            Temporal features ``(B, T, C)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Boundary logits, proposal scores, and refined start/end deltas.
        """

        x = feats.transpose(1, 2)
        boundary = self.tem(x)
        local = self.local(x).transpose(1, 2)
        global_ctx, _ = self.global_attn(local, local, local)
        ctx = local + global_ctx
        start = ctx.unsqueeze(2).expand(-1, -1, ctx.shape[1], -1)
        end = ctx.unsqueeze(1).expand(-1, ctx.shape[1], -1, -1)
        pair = torch.cat((start, end), dim=-1)
        deltas = torch.zeros(
            pair.shape[0], pair.shape[1], pair.shape[2], 2, device=pair.device, dtype=pair.dtype
        )
        refined_pair = pair
        for regressor in self.boundary_regressors:
            delta = regressor(refined_pair)
            deltas = deltas + delta
            update = delta.mean(dim=-1, keepdim=True).expand_as(refined_pair)
            refined_pair = refined_pair + update * 0.01
        score = self.score(refined_pair).squeeze(-1)
        return boundary, score, deltas


class LFBAction(nn.Module):
    """Long-term Feature Bank action detector."""

    def __init__(self) -> None:
        """Initialize short-term and long-term feature fusion heads."""

        super().__init__()
        self.short = nn.Linear(16, 32)
        self.long_attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.cls = nn.Linear(32, 6)

    def forward(self, feats: Tensor) -> Tensor:
        """Classify actions with long-term feature-bank attention.

        Parameters
        ----------
        feats:
            Temporal features.

        Returns
        -------
        Tensor
            Action logits.
        """

        tokens = self.short(feats)
        ctx, _ = self.long_attn(tokens[:, :1], tokens, tokens)
        return self.cls(ctx.squeeze(1))


class IndexNetMatting(nn.Module):
    """IndexNet-guided encoder-decoder image matting."""

    def __init__(self) -> None:
        """Initialize encoder, learned index maps, and decoder."""

        super().__init__()
        self.enc1 = _cba(4, 16, 2)
        self.enc2 = _cba(16, 24, 2)
        self.index = nn.Conv2d(24, 24, 1)
        self.index_pool = nn.Conv2d(24, 24, 1)
        self.index_up = nn.Conv2d(24, 24, 1)
        self.dec = nn.Sequential(_cba(24, 16), nn.Conv2d(16, 1, 1))

    def forward(self, image_trimap: Tensor) -> Tensor:
        """Predict alpha matte.

        Parameters
        ----------
        image_trimap:
            RGB image plus trimap channel.

        Returns
        -------
        Tensor
            Alpha matte.
        """

        enc = self.enc2(self.enc1(image_trimap))
        pooled, indices = F.max_pool2d(enc, 2, stride=2, return_indices=True)
        pool_index = torch.sigmoid(self.index_pool(pooled))
        guided_pool = pooled * pool_index
        unpooled = F.max_unpool2d(guided_pool, indices, 2, stride=2, output_size=enc.shape)
        guided = unpooled * torch.sigmoid(self.index(enc)) + self.index_up(enc)
        up = F.interpolate(guided, scale_factor=4, mode="bilinear", align_corners=False)
        return torch.sigmoid(self.dec(up))


class InstanceColorization(nn.Module):
    """Instance-aware colorization with global and local branches."""

    def __init__(self) -> None:
        """Initialize colorization branches."""

        super().__init__()
        self.global_branch = nn.Sequential(_cba(1, 16, 2), _cba(16, 32, 2), nn.AdaptiveAvgPool2d(1))
        self.local_branch = nn.Sequential(_cba(2, 16), _cba(16, 16))
        self.head = nn.Conv2d(48, 2, 1)

    def forward(self, gray_mask: Tensor) -> Tensor:
        """Colorize an image with an instance mask.

        Parameters
        ----------
        gray_mask:
            Luminance and instance-mask tensor.

        Returns
        -------
        Tensor
            Predicted ``ab`` color channels.
        """

        gray = gray_mask[:, :1]
        local = self.local_branch(gray_mask)
        global_feat = self.global_branch(gray).expand(-1, -1, local.shape[-2], local.shape[-1])
        return torch.tanh(self.head(torch.cat((local, global_feat), dim=1)))


class LIIFSR(nn.Module):
    """LIIF implicit image function super-resolution model."""

    def __init__(self) -> None:
        """Initialize EDSR-style encoder and coordinate MLP."""

        super().__init__()
        self.enc = nn.Sequential(_cba(3, 32), _cba(32, 32), _cba(32, 32))
        self.mlp = nn.Sequential(nn.Linear(34, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, image: Tensor) -> Tensor:
        """Decode a 2x-resolution image from features and coordinates.

        Parameters
        ----------
        image:
            Low-resolution image.

        Returns
        -------
        Tensor
            Super-resolved image.
        """

        feat = F.interpolate(self.enc(image), scale_factor=2, mode="bilinear", align_corners=False)
        height, width = feat.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=image.device),
            torch.linspace(-1, 1, width, device=image.device),
            indexing="ij",
        )
        coord = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(image.shape[0], -1, -1, -1)
        rgb = self.mlp(torch.cat((feat, coord), dim=1).permute(0, 2, 3, 1))
        return rgb.permute(0, 3, 1, 2)


class NAFBlock(nn.Module):
    """NAFNet block with SimpleGate and channel attention."""

    def __init__(self, channels: int) -> None:
        """Initialize NAF block.

        Parameters
        ----------
        channels:
            Channel count.
        """

        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, channels * 2, 1)
        self.dw = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.pw2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply NAFNet gated residual block.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Refined feature map.
        """

        y1, y2 = self.dw(self.pw1(self.norm(x))).chunk(2, dim=1)
        y = y1 * y2
        y = self.pw2(y * self.ca(y))
        return x + y


class NAFNet(nn.Module):
    """Compact NAFNet image-restoration network."""

    def __init__(self) -> None:
        """Initialize encoder, NAF blocks, and decoder."""

        super().__init__()
        self.inp = nn.Conv2d(3, 24, 3, padding=1)
        self.blocks = nn.Sequential(NAFBlock(24), NAFBlock(24), NAFBlock(24))
        self.out = nn.Conv2d(24, 3, 3, padding=1)

    def forward(self, image: Tensor) -> Tensor:
        """Restore an image.

        Parameters
        ----------
        image:
            Degraded image.

        Returns
        -------
        Tensor
            Restored image.
        """

        return image + self.out(self.blocks(self.inp(image)))


class LSGAN(nn.Module):
    """Least-squares GAN generator/discriminator pair."""

    def __init__(self, latent: int = 32) -> None:
        """Initialize DCGAN-style generator and discriminator.

        Parameters
        ----------
        latent:
            Latent vector size.
        """

        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(latent, 64 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )
        self.disc = nn.Sequential(
            _cba(3, 16, 2), _cba(16, 32, 2), nn.Flatten(), nn.Linear(32 * 4 * 4, 1)
        )

    def forward(self, z: Tensor) -> dict[str, Tensor]:
        """Generate an image and expose the least-squares generator objective.

        Parameters
        ----------
        z:
            Latent vector.

        Returns
        -------
        dict[str, Tensor]
            Fake image, discriminator score, fake-target L2 loss, and real-target L2 loss.
        """

        fake = self.gen(z)
        score = self.disc(fake)
        fake_loss = score.square().mean(dim=1, keepdim=True)
        real_loss = (score - 1.0).square().mean(dim=1, keepdim=True)
        return {
            "fake": fake,
            "score": score,
            "least_squares_fake_loss": fake_loss,
            "least_squares_real_loss": real_loss,
        }


class VLMConnector(nn.Module):
    """Vision-language connector used by LLaVA/MiniGPT/OFA-style rows."""

    def __init__(self, mode: str = "llava") -> None:
        """Initialize vision encoder, connector, and tiny decoder.

        Parameters
        ----------
        mode:
            Connector family.
        """

        super().__init__()
        self.mode = mode
        self.vision = TinyFPN(width=16)
        self.query = nn.Parameter(torch.zeros(1, 4, 48))
        self.proj = nn.Linear(16, 48)
        self.text = nn.Embedding(64, 48)
        layer = nn.TransformerDecoderLayer(48, 4, 96, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.head = nn.Linear(48, 64)

    def forward(self, sample: Tensor | tuple[Tensor, Tensor] | list[Tensor]) -> Tensor:
        """Decode text conditioned on visual tokens.

        Parameters
        ----------
        sample:
            Either token ids for backward compatibility or ``(image, token_ids)``.

        Returns
        -------
        Tensor
            Vocabulary logits.
        """

        if isinstance(sample, (tuple, list)):
            image, tokens = sample
        else:
            tokens = sample
            image = torch.zeros(tokens.shape[0], 3, 64, 64, device=tokens.device)
        visual = self.proj(_flat(self.vision(image)[0]))
        if self.mode == "minigpt4":
            query = self.query.expand(tokens.shape[0], -1, -1)
            visual = torch.cat((query, visual[:, :4]), dim=1)
        text = self.text(tokens)
        if self.mode == "ofa":
            text = text + visual.mean(1, keepdim=True)
        return self.head(self.decoder(text, visual))


class OCLIPBackbone(nn.Module):
    """OCR CLIP-style visual/text contrastive backbone."""

    def __init__(self) -> None:
        """Initialize OCR visual encoder and text encoder."""

        super().__init__()
        self.visual = TinyFPN(width=16)
        self.text = nn.Embedding(64, 16)
        self.proj_img = nn.Linear(16, 32)
        self.proj_txt = nn.Linear(16, 32)

    def forward(self, sample: Tensor | tuple[Tensor, Tensor]) -> Tensor:
        """Compute image-text similarity logits.

        Parameters
        ----------
        sample:
            Either text token ids for backward compatibility or ``(image, token_ids)``.

        Returns
        -------
        Tensor
            CLIP-style similarity matrix.
        """

        if isinstance(sample, tuple):
            image, tokens = sample
        else:
            tokens = sample
            image = torch.zeros(tokens.shape[0], 3, 64, 64, device=tokens.device)
        image_feat = self.proj_img(_flat(self.visual(image)[0]).mean(1))
        text_feat = self.proj_txt(self.text(tokens).mean(1))
        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        return image_feat @ text_feat.T


class VinDLU(nn.Module):
    """Video-language dual encoder/fusion transformer used by VindLU."""

    def __init__(self) -> None:
        """Initialize video, text, and fusion heads."""

        super().__init__()
        self.video = MViTVideo(classes=48, dim=48)
        self.text = nn.Embedding(64, 48)
        layer = nn.TransformerEncoderLayer(48, 4, 96, batch_first=True)
        self.fusion = nn.TransformerEncoder(layer, 1)
        self.retrieval = nn.Linear(48, 1)
        self.vqa = nn.Linear(48, 8)

    def forward(self, sample: Tensor | tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Run video-language retrieval and VQA heads.

        Parameters
        ----------
        sample:
            Either text token ids or ``(video, token_ids)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Retrieval score and answer logits.
        """

        if isinstance(sample, tuple):
            video, tokens = sample
        else:
            tokens = sample
            video = torch.zeros(tokens.shape[0], 3, 4, 32, 32, device=tokens.device)
        visual = self.video(video).unsqueeze(1)
        text = self.text(tokens)
        fused = self.fusion(torch.cat((visual, text), dim=1)).mean(1)
        return self.retrieval(fused), self.vqa(fused)


class OmniSourceAction(nn.Module):
    """OmniSource-style multi-source action classifier."""

    def __init__(self) -> None:
        """Initialize RGB, flow, and auxiliary source branches."""

        super().__init__()
        self.rgb = MViTVideo(classes=24, dim=48)
        self.flow = MViTVideo(classes=24, dim=48)
        self.aux = nn.Linear(24, 24)
        self.cls = nn.Linear(48, 10)

    def forward(self, video: Tensor) -> Tensor:
        """Classify action from multiple source branches.

        Parameters
        ----------
        video:
            RGB video clip.

        Returns
        -------
        Tensor
            Class logits.
        """

        rgb = self.rgb(video)
        flow_like = video[:, :, 1:] - video[:, :, :-1]
        flow_like = F.pad(flow_like, (0, 0, 0, 0, 0, 1))
        flow = self.flow(flow_like)
        return self.cls(torch.cat((rgb, self.aux(flow)), dim=-1))


class LeNet(nn.Module):
    """Classic LeNet convolutional classifier."""

    def __init__(self) -> None:
        """Initialize LeNet layers."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Classify an MNIST-like image.

        Parameters
        ----------
        image:
            Grayscale image.

        Returns
        -------
        Tensor
            Class logits.
        """

        return self.net(image)


def image_input() -> Tensor:
    """Return a compact RGB image.

    Returns
    -------
    Tensor
        Image tensor.
    """

    return torch.randn(1, 3, 64, 64)


def gray_input() -> Tensor:
    """Return a compact grayscale image.

    Returns
    -------
    Tensor
        Image tensor.
    """

    return torch.randn(1, 1, 32, 32)


def flow_input() -> Tensor:
    """Return two RGB frames for optical flow.

    Returns
    -------
    Tensor
        Frame-pair tensor.
    """

    return torch.randn(1, 2, 3, 48, 48)


def matting_input() -> Tensor:
    """Return RGB plus trimap input.

    Returns
    -------
    Tensor
        Matting tensor.
    """

    return torch.randn(1, 4, 48, 48)


def color_input() -> Tensor:
    """Return luminance plus instance mask input.

    Returns
    -------
    Tensor
        Colorization tensor.
    """

    return torch.randn(1, 2, 48, 48)


def latent_input() -> Tensor:
    """Return GAN latent input.

    Returns
    -------
    Tensor
        Latent tensor.
    """

    return torch.randn(1, 32)


def points_input() -> Tensor:
    """Return compact point cloud input.

    Returns
    -------
    Tensor
        Point tensor.
    """

    return torch.randn(1, 16, 4)


def multiview_input() -> Tensor:
    """Return compact multiview image input.

    Returns
    -------
    Tensor
        Multiview tensor.
    """

    return torch.randn(1, 2, 3, 48, 48)


def point_image_input() -> tuple[Tensor, Tensor]:
    """Return paired point cloud and image input.

    Returns
    -------
    tuple[Tensor, Tensor]
        Point tensor and RGB image tensor.
    """

    return torch.randn(1, 16, 4), torch.randn(1, 3, 64, 64)


def pose3d_input() -> Tensor:
    """Return a short 3D pose sequence.

    Returns
    -------
    Tensor
        Pose tensor.
    """

    return torch.randn(1, 8, 17, 3)


def video_input() -> Tensor:
    """Return a compact video clip.

    Returns
    -------
    Tensor
        Video tensor.
    """

    return torch.randn(1, 3, 4, 32, 32)


def temporal_input() -> Tensor:
    """Return temporal proposal features.

    Returns
    -------
    Tensor
        Feature tensor.
    """

    return torch.randn(1, 16, 16)


def token_input() -> Tensor:
    """Return text token ids.

    Returns
    -------
    Tensor
        Token tensor.
    """

    return torch.randint(0, 64, (1, 8))


def image_text_input() -> tuple[Tensor, Tensor]:
    """Return an image/text pair for vision-language rows.

    Returns
    -------
    tuple[Tensor, Tensor]
        RGB image and token tensor.
    """

    return torch.randn(1, 3, 64, 64), torch.randint(0, 64, (1, 8))


def video_text_input() -> tuple[Tensor, Tensor]:
    """Return a video/text pair for video-language rows.

    Returns
    -------
    tuple[Tensor, Tensor]
        Video clip and token tensor.
    """

    return torch.randn(1, 3, 4, 32, 32), torch.randint(0, 64, (1, 8))


def kie_example_input() -> tuple[Tensor, Tensor, Tensor]:
    """Return SDMGR KIE document graph inputs.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Text ids, visual features, and boxes.
    """

    return kie_input()


def text_recognition_input() -> Tensor:
    """Return scene-text recognition image input.

    Returns
    -------
    Tensor
        Text image tensor.
    """

    return text_image()


def build_imvoxelnet() -> nn.Module:
    """Build ImVoxelNet with image-to-voxel lifting and BEV head."""

    return VoxelDetector("imvoxel").eval()


def build_mvxnet() -> nn.Module:
    """Build MVXNet with point-image fusion before BEV detection."""

    return VoxelDetector("mvxnet").eval()


def build_mvfcos3d() -> nn.Module:
    """Build multiview FCOS3D-style BEV detector."""

    return VoxelDetector("multiview").eval()


def build_monoflex() -> nn.Module:
    """Build MonoFlex-style monocular FCOS3D detector."""

    return MonoFlexDetector().eval()


def build_mask_rcnn() -> nn.Module:
    """Build Mask R-CNN with FPN, RPN, RoI, and mask head."""

    return TwoStageDetector(mask=True).eval()


def build_oriented_rcnn() -> nn.Module:
    """Build Oriented R-CNN with rotated RoI angle branch."""

    return TwoStageDetector(rotated=True).eval()


def build_htc() -> nn.Module:
    """Build Hybrid Task Cascade-style detector with cascaded box and mask heads."""

    return TwoStageDetector(mask=True, cascade=True).eval()


def build_retinanet_rotated() -> nn.Module:
    """Build rotated RetinaNet with FPN dense focal-loss heads."""

    return RetinaStyleDetector(rotated=True).eval()


def build_kld_retinanet() -> nn.Module:
    """Build KLD rotated RetinaNet with Gaussian KLD overlap."""

    return GaussianRotatedRetinaNet("kld").eval()


def build_kfiou_retinanet() -> nn.Module:
    """Build KFIoU rotated RetinaNet with Kalman Gaussian overlap."""

    return GaussianRotatedRetinaNet("kfiou").eval()


def build_s2anet() -> nn.Module:
    """Build S2ANet with anchor refinement and alignment."""

    return S2ANetDetector().eval()


def build_oriented_reppoints() -> nn.Module:
    """Build Oriented RepPoints with adaptive point-set conversion."""

    return OrientedRepPointsDetector().eval()


def build_r3det() -> nn.Module:
    """Build R3Det with refined rotated boxes."""

    return R3DetDetector().eval()


def build_liteflownet() -> nn.Module:
    """Build LiteFlowNet/PWC-style optical flow estimator."""

    return PWCFlow().eval()


def build_irrpwc() -> nn.Module:
    """Build IRR-PWC with recurrent residual flow refinement."""

    return PWCFlow(recurrent=True).eval()


def build_maskflownet() -> nn.Module:
    """Build MaskFlowNet with occlusion-mask prediction."""

    return PWCFlow(mask=True).eval()


def build_integral_pose() -> nn.Module:
    """Build IntegralPose with heatmap soft-argmax coordinates."""

    return IntegralPoseNet().eval()


def build_heatmap_pose() -> nn.Module:
    """Build top-down heatmap pose estimator."""

    return TopDownHeatmapNet().eval()


def build_interhand3d() -> nn.Module:
    """Build InterHand3D-style integral 3D hand pose estimator."""

    return InterNetHand3D().eval()


def build_motionbert() -> nn.Module:
    """Build MotionBERT spatial-temporal pose transformer."""

    return MotionBERTPose().eval()


def build_mspn2() -> nn.Module:
    """Build two-stage MSPN pose network."""

    return MSPN(stages=2).eval()


def build_mspn4() -> nn.Module:
    """Build four-stage MSPN pose network."""

    return MSPN(stages=4).eval()


def build_bmn() -> nn.Module:
    """Build BMN boundary-matching temporal proposal network."""

    return TemporalProposalNet("bmn").eval()


def build_bsn() -> nn.Module:
    """Build BSN temporal evaluation/proposal network."""

    return TemporalProposalNet("bsn").eval()


def build_drn() -> nn.Module:
    """Build DRN temporal proposal refinement network."""

    return DRNProposalRefiner().eval()


def build_tcanet() -> nn.Module:
    """Build TCANet with temporal context aggregation."""

    return TCANetProposalRefiner().eval()


def build_lfb() -> nn.Module:
    """Build Long-term Feature Bank action detector."""

    return LFBAction().eval()


def build_mvit() -> nn.Module:
    """Build MViT/MViTv2 multiscale video transformer."""

    return MViTVideo().eval()


def build_omnisource() -> nn.Module:
    """Build OmniSource multi-source action classifier."""

    return OmniSourceAction().eval()


def build_indexnet() -> nn.Module:
    """Build IndexNet-guided image matting model."""

    return IndexNetMatting().eval()


def build_inst_colorization() -> nn.Module:
    """Build instance-aware colorization model."""

    return InstanceColorization().eval()


def build_liif() -> nn.Module:
    """Build LIIF implicit image function super-resolution model."""

    return LIIFSR().eval()


def build_nafnet() -> nn.Module:
    """Build NAFNet image-restoration model."""

    return NAFNet().eval()


def build_lsgan() -> nn.Module:
    """Build LSGAN generator/discriminator pair."""

    return LSGAN().eval()


def build_llava() -> nn.Module:
    """Build LLaVA-style vision projector plus language decoder."""

    return VLMConnector("llava").eval()


def build_minigpt4() -> nn.Module:
    """Build MiniGPT-4-style Q-Former visual connector."""

    return VLMConnector("minigpt4").eval()


def build_ofa() -> nn.Module:
    """Build OFA-style unified multimodal encoder-decoder."""

    return VLMConnector("ofa").eval()


def build_vindlu() -> nn.Module:
    """Build VindLU video-language retrieval/VQA model."""

    return VinDLU().eval()


def build_oclip() -> nn.Module:
    """Build OCR CLIP-style contrastive backbone."""

    return OCLIPBackbone().eval()


def build_sdmgr() -> nn.Module:
    """Build SDMGR KIE graph reasoning model."""

    return SDMGRKIE().eval()


def build_master() -> nn.Module:
    """Build MASTER text recognizer with multi-aspect non-local attention."""

    return MASTERRecognizer(extra=False).eval()


def build_nrtr() -> nn.Module:
    """Build NRTR Transformer text recognizer."""

    return NRTRRecognizer(with_modality_transform=False).eval()


def build_lenet() -> nn.Module:
    """Build LeNet classifier."""

    return LeNet().eval()


Entry = tuple[str, str, str, str, str]


def _rows(names: list[str], build: str, example: str, year: str, code: str) -> list[Entry]:
    """Create MENAGERIE_ENTRIES rows for alias-heavy target families.

    Parameters
    ----------
    names:
        Entry names.
    build:
        Builder function attribute.
    example:
        Example-input function attribute.
    year:
        Publication year.
    code:
        Domain code.

    Returns
    -------
    list[Entry]
        Registry entries.
    """

    return [(name, build, example, year, code) for name in names]


MENAGERIE_ENTRIES: list[Entry] = []
MENAGERIE_ENTRIES += _rows(
    [
        "mmdet3d:imvoxelnet",
        "mmdet3d_imvoxelnet",
        "mmdet3d_imvoxelnet_imvoxelnet_2xb4_sunrgbd_3d_10class",
        "mmdet3d_imvoxelnet_imvoxelnet_8xb4_kitti_3d_car",
    ],
    "build_imvoxelnet",
    "image_input",
    "2021",
    "3D",
)
MENAGERIE_ENTRIES += _rows(
    [
        "mmdet3d:mvxnet",
        "mmdet3d_mvxnet",
        "mmdet3d_mvxnet_mvxnet_fpn_dv_second_secfpn_8xb2_80e_kitti_3d_3class",
    ],
    "build_mvxnet",
    "point_image_input",
    "2019",
    "3D",
)
MENAGERIE_ENTRIES += _rows(
    [
        "mmdet3d:mvfcos3d",
        "mmdet3d_mvfcos3d",
        "mmdet3d_mvfcos3d_multiview_fcos3d_r101_dcn_8xb2_waymod5_3d_3class",
    ],
    "build_mvfcos3d",
    "multiview_input",
    "2021",
    "3D",
)
MENAGERIE_ENTRIES += _rows(["mmdet3d_monoflex"], "build_monoflex", "image_input", "2021", "3D")
MENAGERIE_ENTRIES += _rows(
    [
        "mmdet3d_mask_rcnn_mask_rcnn_r101_fpn_1x_nuim",
        "mmdet3d_mask_rcnn_mask_rcnn_x101_32x4d_fpn_1x_nuim",
        "mmocr_textdet_maskrcnn",
    ],
    "build_mask_rcnn",
    "image_input",
    "2017",
    "DET",
)
MENAGERIE_ENTRIES += _rows(
    [
        "mmdet3d_nuimages_htc_r50_fpn_1x_nuim",
        "mmdet3d_nuimages_htc_x101_64x4d_fpn_dconv_c3_c5_coco_20e_1xb16_nuim",
    ],
    "build_htc",
    "image_input",
    "2019",
    "DET",
)
MENAGERIE_ENTRIES += _rows(
    [
        "mmrotate:oriented_rcnn",
        "mmrotate_oriented_rcnn_oriented_rcnn_r50_fpn_1x_dota_le90",
        "mmrotate_oriented_rcnn_oriented_rcnn_swin_tiny_fpn_1x_dota_le90",
        "mmrotate_oriented_rcnn_r50_fpn",
    ],
    "build_oriented_rcnn",
    "image_input",
    "2021",
    "DET",
)
MENAGERIE_ENTRIES += _rows(
    ["mmrotate_oriented_rcnn_oriented_reppoints_r50_fpn_1x_dota_le135"],
    "build_oriented_reppoints",
    "image_input",
    "2019",
    "DET",
)
MENAGERIE_ENTRIES += _rows(
    ["mmrotate:kld", "mmrotate_kld_rotated_retinanet"],
    "build_kld_retinanet",
    "image_input",
    "2021",
    "DET",
)
MENAGERIE_ENTRIES += _rows(
    [
        "mmrotate:kfiou",
        "mmrotate_kfiou_roi_trans_kfiou_ln_r50_fpn_1x_dota_le90",
        "mmrotate_kfiou_roi_trans_kfiou_ln_swin_tiny_fpn_1x_dota_le90",
    ],
    "build_kfiou_retinanet",
    "image_input",
    "2022",
    "DET",
)
MENAGERIE_ENTRIES += _rows(
    ["mmrotate_kfiou_s2anet_kfiou_ln_r50_fpn_1x_dota_le135"],
    "build_kfiou_retinanet",
    "image_input",
    "2022",
    "DET",
)
MENAGERIE_ENTRIES += _rows(
    ["mmrotate_kfiou_r3det"], "build_kfiou_retinanet", "image_input", "2022", "DET"
)
MENAGERIE_ENTRIES += _rows(
    ["mmflow_liteflownet", "mmflow_liteflownet2"], "build_liteflownet", "flow_input", "2018", "FLOW"
)
MENAGERIE_ENTRIES += _rows(["mmflow_irrpwc"], "build_irrpwc", "flow_input", "2019", "FLOW")
MENAGERIE_ENTRIES += _rows(
    ["mmflow_maskflownet"], "build_maskflownet", "flow_input", "2020", "FLOW"
)
MENAGERIE_ENTRIES += _rows(
    ["mmagic:indexnet", "mmagic_indexnet_indexnet_mobv2_dimaug_1xb16_78k_comp1k"],
    "build_indexnet",
    "matting_input",
    "2019",
    "MAT",
)
MENAGERIE_ENTRIES += _rows(
    [
        "mmagic:inst_colorization",
        "mmagic_inst_colorization",
        "mmagic_inst_colorization_inst_colorizatioon_full_official_cocostuff_256x256",
    ],
    "build_inst_colorization",
    "color_input",
    "2020",
    "GEN",
)
MENAGERIE_ENTRIES += _rows(
    [
        "mmagic:liif",
        "mmagic_liif",
        "mmagic_liif_liif_edsr_norm_c64b16_1xb16_1000k_div2k",
        "mmagic_liif_liif_rdn_norm_c64b16_1xb16_1000k_div2k",
    ],
    "build_liif",
    "image_input",
    "2021",
    "SR",
)
MENAGERIE_ENTRIES += _rows(["mmagic:nafnet"], "build_nafnet", "image_input", "2022", "SR")
MENAGERIE_ENTRIES += _rows(
    [
        "mmagic:lsgan",
        "mmagic_lsgan",
        "mmagic_lsgan_lsgan_dcgan_archi_lr1e_3_1xb128_12mimgs_celeba_cropped_64x64",
        "mmagic_lsgan_lsgan_lsgan_archi_lr1e_4_1xb64_10mimgs_lsun_bedroom_128x128",
    ],
    "build_lsgan",
    "latent_input",
    "2017",
    "GAN",
)
MENAGERIE_ENTRIES += _rows(
    [
        "IntegralPose-ResNet50",
        "mmpose:integral_regression",
        "mmpose_integral_regression",
        "mmpose_ipr",
    ],
    "build_integral_pose",
    "image_input",
    "2018",
    "POSE",
)
MENAGERIE_ENTRIES += _rows(
    ["mmpose_interhand3d", "mmpose:internet", "InterNet-InterHand3D"],
    "build_interhand3d",
    "image_input",
    "2020",
    "POSE",
)
MENAGERIE_ENTRIES += _rows(
    ["mmpose:motionbert", "mmpose_motionbert", "MotionBERT-H36M", "mmpose_motionbert_3d"],
    "build_motionbert",
    "pose3d_input",
    "2023",
    "POSE",
)
MENAGERIE_ENTRIES += _rows(
    ["mmpose:topdown_heatmap"], "build_heatmap_pose", "image_input", "2014", "POSE"
)
MENAGERIE_ENTRIES += _rows(["MSPN-2stage"], "build_mspn2", "image_input", "2019", "POSE")
MENAGERIE_ENTRIES += _rows(["MSPN-4stage"], "build_mspn4", "image_input", "2019", "POSE")
MENAGERIE_ENTRIES += _rows(
    ["mmaction2_detection_lfb", "mmaction:lfb"], "build_lfb", "temporal_input", "2019", "ACTION"
)
MENAGERIE_ENTRIES += _rows(
    ["mmaction2_localization_bmn_2xb8_2048x100_9e_activitynet_slowonly_k700_feature"],
    "build_bmn",
    "temporal_input",
    "2018",
    "ACTION",
)
MENAGERIE_ENTRIES += _rows(
    [
        "mmaction2_localization_bsn_pem_1xb16_2048x100_20e_activitynet_slowonly_k700_feature",
        "mmaction2_localization_bsn_tem_1xb16_2048x100_20e_activitynet_k700_feature",
    ],
    "build_bsn",
    "temporal_input",
    "2018",
    "ACTION",
)
MENAGERIE_ENTRIES += _rows(
    ["mmaction2_localization_drn_2xb16_4096_10e_c3d_feature_first"],
    "build_drn",
    "temporal_input",
    "2018",
    "ACTION",
)
MENAGERIE_ENTRIES += _rows(
    ["mmaction2_localization_tcanet_2xb8_700x100_9e_hacs_feature"],
    "build_tcanet",
    "temporal_input",
    "2021",
    "ACTION",
)
MENAGERIE_ENTRIES += _rows(
    [
        "mmaction2_recognition_mvit",
        "mmaction:mvit",
        "mmaction_mvit_base",
        "mmaction_mvit_large",
        "mmaction_mvit_small",
        "mmaction2_mvitv2_b_32x3_k400",
        "mmaction2_mvitv2_b_u32_sthv2",
        "mmaction2_mvitv2_l_40x3_k400",
        "mmaction2_mvitv2_l_u40_sthv2",
        "mmaction2_mvitv2_s_16x4_k400",
        "mmaction2_mvitv2_s_u16_sthv2",
        "mmaction2_mvitv2_s_maskfeat_16x4_k400",
    ],
    "build_mvit",
    "video_input",
    "2021",
    "ACTION",
)
MENAGERIE_ENTRIES += _rows(
    ["mmaction:omnisource"], "build_omnisource", "video_input", "2020", "ACTION"
)
MENAGERIE_ENTRIES += _rows(
    [
        "mmaction2_multimodal_vindlu_beit_base_8x16_retrieval_msrvtt_9k",
        "mmaction2_multimodal_vindlu_beit_base_8x8_vqa_msrvtt_qa",
        "mmaction2_multimodal_vindlu_beit_base_vqa_mc_msrvtt_mc",
    ],
    "build_vindlu",
    "video_text_input",
    "2023",
    "VLM",
)
MENAGERIE_ENTRIES += _rows(["mmpretrain:lenet"], "build_lenet", "gray_input", "1998", "CLS")
MENAGERIE_ENTRIES += _rows(["mmpretrain:llava"], "build_llava", "image_text_input", "2023", "VLM")
MENAGERIE_ENTRIES += _rows(
    ["mmpretrain:minigpt4"], "build_minigpt4", "image_text_input", "2023", "VLM"
)
MENAGERIE_ENTRIES += _rows(["mmpretrain:ofa"], "build_ofa", "image_text_input", "2022", "VLM")
MENAGERIE_ENTRIES += _rows(
    ["mmocr_kie_sdmgr_novisual_60e_wildreceipt_openset", "mmocr_kie_sdmgr_unet16_60e_wildreceipt"],
    "build_sdmgr",
    "kie_example_input",
    "2021",
    "KIE",
)
MENAGERIE_ENTRIES += _rows(
    ["mmocr:master", "mmocr_textrecog_master"],
    "build_master",
    "text_recognition_input",
    "2019",
    "OCR",
)
MENAGERIE_ENTRIES += _rows(
    ["mmocr:nrtr", "mmocr_textrecog_nrtr"], "build_nrtr", "text_recognition_input", "2019", "OCR"
)
MENAGERIE_ENTRIES += _rows(["mmocr_backbone_oclip"], "build_oclip", "token_input", "2022", "OCR")
