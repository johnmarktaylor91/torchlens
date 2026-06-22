"""Compact faithful OpenMMLab/Paddle dependency-gated model reconstructions.

The entries in this module replace install-hostile OpenMMLab, Paddle, and
adjacent zoo recipes with small random-initialized PyTorch models.  Each model
keeps the load-bearing architecture primitive of the named family: two-stage
detectors include backbone/FPN/RPN/ROI heads, dense detectors include their
family-specific heads, optical-flow models include correlation/warping/update
machinery, and segmentation/video/OCR/generation entries expose the named
paper mechanisms rather than a generic CNN.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import torch
from torch import nn
import torch.nn.functional as F


TensorDict = dict[str, torch.Tensor]


class ConvAct(nn.Module):
    """Convolution followed by normalization and SiLU activation."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, groups: int = 1) -> None:
        """Initialize the convolutional block.

        Parameters
        ----------
        in_ch:
            Input channel count.
        out_ch:
            Output channel count.
        stride:
            Spatial stride.
        groups:
            Group count for the convolution.
        """

        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, batch normalization, and activation.

        Parameters
        ----------
        x:
            Input feature tensor.

        Returns
        -------
        torch.Tensor
            Activated output tensor.
        """

        return F.silu(self.bn(self.conv(x)))


class TinyBackbone(nn.Module):
    """Small multi-stage image backbone returning a feature pyramid stem."""

    def __init__(self, width: int = 24) -> None:
        """Initialize the backbone.

        Parameters
        ----------
        width:
            Base feature width.
        """

        super().__init__()
        self.stem = ConvAct(3, width, stride=2)
        self.c3 = ConvAct(width, width * 2, stride=2)
        self.c4 = ConvAct(width * 2, width * 4, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return three backbone stages.

        Parameters
        ----------
        x:
            Input image batch.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            C2, C3, and C4 feature maps.
        """

        c2 = self.stem(x)
        c3 = self.c3(c2)
        return c2, c3, self.c4(c3)


class TinyFPN(nn.Module):
    """Top-down feature pyramid network with lateral projections."""

    def __init__(self, width: int = 24, out_ch: int = 32) -> None:
        """Initialize FPN lateral and smoothing layers.

        Parameters
        ----------
        width:
            Backbone base width.
        out_ch:
            Pyramid output width.
        """

        super().__init__()
        self.backbone = TinyBackbone(width)
        self.lat2 = nn.Conv2d(width, out_ch, 1)
        self.lat3 = nn.Conv2d(width * 2, out_ch, 1)
        self.lat4 = nn.Conv2d(width * 4, out_ch, 1)
        self.smooth2 = ConvAct(out_ch, out_ch)
        self.smooth3 = ConvAct(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a compact feature pyramid.

        Parameters
        ----------
        x:
            Input image batch.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            P2, P3, and P4 pyramid maps.
        """

        c2, c3, c4 = self.backbone(x)
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        return self.smooth2(p2), self.smooth3(p3), p4


class RPNHead(nn.Module):
    """Region proposal network head with objectness and box deltas."""

    def __init__(self, channels: int = 32, anchors: int = 3) -> None:
        """Initialize the RPN head.

        Parameters
        ----------
        channels:
            Feature channel count.
        anchors:
            Anchor count per location.
        """

        super().__init__()
        self.conv = ConvAct(channels, channels)
        self.obj = nn.Conv2d(channels, anchors, 1)
        self.box = nn.Conv2d(channels, anchors * 4, 1)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict proposal objectness and box deltas.

        Parameters
        ----------
        feat:
            Pyramid feature map.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Objectness logits and box deltas.
        """

        hidden = self.conv(feat)
        return self.obj(hidden), self.box(hidden)


class ROIBoxHead(nn.Module):
    """Second-stage ROI classifier and regressor."""

    def __init__(self, channels: int = 32, classes: int = 5) -> None:
        """Initialize the ROI head.

        Parameters
        ----------
        channels:
            FPN channel count.
        classes:
            Number of object classes.
        """

        super().__init__()
        self.fc1 = nn.Linear(channels * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.cls = nn.Linear(64, classes)
        self.box = nn.Linear(64, classes * 4)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool ROI features and predict classes and boxes.

        Parameters
        ----------
        feat:
            FPN feature map.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Hidden ROI descriptor, class logits, and box deltas.
        """

        roi = F.adaptive_avg_pool2d(feat, (4, 4)).flatten(1)
        hidden = F.relu(self.fc1(roi))
        hidden = F.relu(self.fc2(hidden))
        return hidden, self.cls(hidden), self.box(hidden)


class MaskHead(nn.Module):
    """Mask R-CNN style deconvolutional instance mask head."""

    def __init__(self, channels: int = 32, classes: int = 5) -> None:
        """Initialize the mask head.

        Parameters
        ----------
        channels:
            FPN channel count.
        classes:
            Number of mask classes.
        """

        super().__init__()
        self.conv = ConvAct(channels, channels)
        self.up = nn.ConvTranspose2d(channels, channels, 2, stride=2)
        self.mask = nn.Conv2d(channels, classes, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Predict per-class instance masks from ROI-aligned features.

        Parameters
        ----------
        feat:
            FPN feature map.

        Returns
        -------
        torch.Tensor
            Mask logits.
        """

        roi = F.adaptive_avg_pool2d(feat, (7, 7))
        return self.mask(F.relu(self.up(self.conv(roi))))


class TwoStageDetector(nn.Module):
    """FPN/RPN/ROI detector family including Faster, Mask, HTC, and QDTrack."""

    def __init__(self, kind: str = "faster", classes: int = 5) -> None:
        """Initialize the two-stage detector variant.

        Parameters
        ----------
        kind:
            Variant name controlling extra mask/cascade/embedding heads.
        classes:
            Number of object classes.
        """

        super().__init__()
        self.kind = kind
        self.fpn = TinyFPN()
        self.rpn = RPNHead()
        self.roi = ROIBoxHead(classes=classes)
        self.mask = MaskHead(classes=classes) if "mask" in kind or "htc" in kind else None
        self.cascade = nn.ModuleList([ROIBoxHead(classes=classes) for _ in range(2)])
        self.track_embed = nn.Linear(64, 16) if "track" in kind or "qd" in kind else None
        self.teacher = nn.Linear(64, classes + classes * 4) if "distilling" in kind else None

    def forward(self, x: torch.Tensor) -> TensorDict:
        """Run the detector variant.

        Parameters
        ----------
        x:
            Input image batch.

        Returns
        -------
        TensorDict
            Detector outputs.
        """

        p2, p3, _ = self.fpn(x)
        rpn_obj, rpn_box = self.rpn(p3)
        hidden, cls, box = self.roi(p2)
        out: TensorDict = {"rpn_obj": rpn_obj, "rpn_box": rpn_box, "roi_cls": cls, "roi_box": box}
        if self.mask is not None:
            out["mask_logits"] = self.mask(p2)
        if "cascade" in self.kind or "htc" in self.kind:
            cascade_state = p2
            for idx, stage in enumerate(self.cascade):
                _, stage_cls, stage_box = stage(cascade_state)
                out[f"cascade_cls_{idx}"] = stage_cls
                out[f"cascade_box_{idx}"] = stage_box
                cascade_state = cascade_state + 0.05 * torch.tanh(cascade_state)
        if self.track_embed is not None:
            out["track_embedding"] = F.normalize(self.track_embed(hidden), dim=-1)
        if self.teacher is not None:
            teacher_logits = self.teacher(hidden.detach())
            student_logits = torch.cat((cls, box), dim=-1)
            out["dod_teacher_logits"] = teacher_logits
            out["dod_distillation"] = F.mse_loss(student_logits, teacher_logits, reduction="none")
        return out


class DenseDetector(nn.Module):
    """Single-stage FPN detector with variant-specific dense heads."""

    def __init__(self, kind: str, classes: int = 5, bins: int = 8) -> None:
        """Initialize dense detector.

        Parameters
        ----------
        kind:
            Dense detector family name.
        classes:
            Class count.
        bins:
            Distributional box bins for GFL-style heads.
        """

        super().__init__()
        self.kind = kind
        self.bins = bins
        self.register_buffer("project", torch.arange(bins, dtype=torch.float32))
        self.fpn = TinyFPN()
        self.tower = nn.ModuleList([ConvAct(32, 32), ConvAct(32, 32)])
        self.cls = nn.Conv2d(32, classes, 3, padding=1)
        self.box = nn.Conv2d(32, 4 * (bins if "gfl" in kind else 1), 3, padding=1)
        self.quality = nn.Conv2d(32, 1, 3, padding=1)
        self.points = nn.Conv2d(32, 18, 3, padding=1)
        self.dy_scale = nn.Conv2d(32, 32, 1)
        self.dy_spatial = nn.Conv2d(32, 1, 3, padding=1)
        self.dy_task = nn.Linear(32, 32)
        self.anchor_score = nn.Conv2d(32, 1, 1)
        self.center_heatmap = nn.Conv2d(32, classes, 3, padding=1)
        self.wh = nn.Conv2d(32, 2, 3, padding=1)
        self.offset = nn.Conv2d(32, 2, 3, padding=1)
        self.assignment = nn.Conv2d(32, 4, 1)
        self.dgqp = nn.Sequential(nn.Conv2d(8, 16, 1), nn.ReLU(), nn.Conv2d(16, 1, 1))
        self.teacher = nn.Conv2d(32, classes + 4, 1)
        self.text = nn.Embedding(8, 32)
        self.ground = nn.MultiheadAttention(32, 4, batch_first=True)

    def _head_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute dense head features.

        Parameters
        ----------
        x:
            Input image batch.

        Returns
        -------
        torch.Tensor
            Dense feature map.
        """

        feats = self.fpn(x)
        feat = feats[1]
        for layer in self.tower:
            feat = layer(feat)
        if "dyhead" in self.kind:
            p2, p3, p4 = feats
            scale = torch.softmax(
                torch.stack(
                    [
                        p2.mean(dim=(1, 2, 3)),
                        p3.mean(dim=(1, 2, 3)),
                        p4.mean(dim=(1, 2, 3)),
                    ],
                    dim=1,
                ),
                dim=1,
            )
            fused = (
                scale[:, 0].view(-1, 1, 1, 1)
                * F.interpolate(p2, size=feat.shape[-2:], mode="nearest")
                + scale[:, 1].view(-1, 1, 1, 1) * p3
                + scale[:, 2].view(-1, 1, 1, 1)
                * F.interpolate(p4, size=feat.shape[-2:], mode="nearest")
            )
            spatial = torch.sigmoid(self.dy_spatial(fused))
            task = torch.sigmoid(self.dy_task(fused.mean(dim=(2, 3)))).unsqueeze(-1).unsqueeze(-1)
            feat = self.dy_scale(fused) * spatial * task
        return feat

    def forward(self, x: torch.Tensor) -> TensorDict:
        """Predict dense detector outputs.

        Parameters
        ----------
        x:
            Input image batch.

        Returns
        -------
        TensorDict
            Dense class, box, and variant-specific outputs.
        """

        feat = self._head_features(x)
        out: TensorDict = {"cls": self.cls(feat), "quality": torch.sigmoid(self.quality(feat))}
        if "ttfnet" in self.kind:
            out["center_heatmap"] = torch.sigmoid(self.center_heatmap(feat))
            out["wh"] = F.softplus(self.wh(feat))
            out["offset"] = self.offset(feat)
        if "gfl" in self.kind:
            raw = self.box(feat)
            bsz, _, height, width = raw.shape
            dist = raw.view(bsz, 4, self.bins, height, width).softmax(dim=2)
            out["box_distribution"] = dist
            out["box"] = (dist * self.project.view(1, 1, self.bins, 1, 1)).sum(dim=2)
            if "v2" in self.kind:
                top2 = dist.topk(k=2, dim=2).values.flatten(1, 2)
                out["dgqp"] = torch.sigmoid(self.dgqp(top2))
        elif "rep" in self.kind:
            pts = self.points(feat).view(feat.shape[0], 9, 2, feat.shape[2], feat.shape[3])
            out["points"] = pts
            out["box"] = pts.mean(dim=1).repeat(1, 2, 1, 1)
        else:
            out["box"] = self.box(feat)
        if "atss" in self.kind:
            assign_logits = self.assignment(feat)
            mean = assign_logits.mean(dim=(2, 3), keepdim=True)
            std = assign_logits.std(dim=(2, 3), keepdim=True).clamp_min(1.0e-4)
            out["adaptive_sample_score"] = (
                torch.sigmoid((assign_logits - mean) / std).topk(k=2, dim=1).values
            )
        if "fsaf" in self.kind:
            level_logits = self.assignment(feat)
            out["feature_select"] = torch.softmax(level_logits, dim=1)
            out["anchor_free_box"] = F.relu(out["box"])
        if "lad" in self.kind:
            teacher = self.teacher(feat)
            out["label_assignment_distillation"] = F.mse_loss(
                out["cls"].sigmoid(), teacher[:, : out["cls"].shape[1]].sigmoid(), reduction="none"
            )
        if "ld" in self.kind:
            out["localization_distillation"] = torch.softmax(
                self.box(feat).view(feat.shape[0], 4, self.bins, *feat.shape[-2:]), dim=2
            )
        if "grounding" in self.kind or "glip" in self.kind:
            visual = feat.flatten(2).transpose(1, 2)
            text_ids = (
                torch.arange(self.text.num_embeddings, device=x.device)
                .unsqueeze(0)
                .expand(x.shape[0], -1)
            )
            text = self.text(text_ids)
            grounded = self.ground(visual, text, text, need_weights=False)[0]
            out["grounding_logits"] = torch.matmul(grounded, text.transpose(1, 2))
        if "autoassign" in self.kind or "freeanchor" in self.kind:
            out["anchor_matching"] = torch.sigmoid(self.anchor_score(feat))
        if "tood" in self.kind or "paa" in self.kind or "vfnet" in self.kind:
            out["task_alignment"] = out["cls"].sigmoid().mean(dim=1, keepdim=True) * out["quality"]
        if "paa" in self.kind:
            error = (
                (out["box"] - out["box"].mean(dim=(2, 3), keepdim=True))
                .pow(2)
                .mean(dim=1, keepdim=True)
            )
            out["paa_probability"] = torch.exp(
                -error / (error.mean(dim=(2, 3), keepdim=True).clamp_min(1.0e-4))
            )
        if "fovea" in self.kind:
            out["fovea_heatmap"] = out["quality"].pow(2.0)
        return out


class CARAFEUpsampler(nn.Module):
    """CARAFE content-aware feature reassembly operator."""

    def __init__(self, channels: int = 32, scale: int = 2, kernel: int = 3) -> None:
        """Initialize CARAFE.

        Parameters
        ----------
        channels:
            Input feature channels.
        scale:
            Upsampling factor.
        kernel:
            Reassembly kernel width.
        """

        super().__init__()
        self.scale = scale
        self.kernel = kernel
        self.encoder = nn.Conv2d(channels, scale * scale * kernel * kernel, 3, padding=1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply content-aware upsampling.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Reassembled higher-resolution features.
        """

        weights = F.pixel_shuffle(self.encoder(x), self.scale)
        weights = weights.view(
            x.shape[0], self.kernel * self.kernel, x.shape[2] * 2, x.shape[3] * 2
        )
        weights = weights.softmax(dim=1)
        up = F.interpolate(self.proj(x), scale_factor=self.scale, mode="nearest")
        patches = F.unfold(up, self.kernel, padding=self.kernel // 2)
        patches = patches.view(x.shape[0], up.shape[1], self.kernel * self.kernel, *up.shape[-2:])
        return (patches * weights.unsqueeze(1)).sum(dim=2)


class YOLOFDetector(nn.Module):
    """YOLOF-style C5 detector with dilated encoder and no FPN."""

    def __init__(self, classes: int = 5) -> None:
        """Initialize YOLOF.

        Parameters
        ----------
        classes:
            Number of object classes.
        """

        super().__init__()
        self.backbone = TinyBackbone()
        self.dilated = nn.Sequential(
            nn.Conv2d(96, 32, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=4, dilation=4),
            nn.ReLU(),
        )
        self.cls = nn.Conv2d(32, classes, 1)
        self.box = nn.Conv2d(32, 4, 1)

    def forward(self, x: torch.Tensor) -> TensorDict:
        """Predict YOLOF dense outputs from one C5 level.

        Parameters
        ----------
        x:
            Image batch.

        Returns
        -------
        TensorDict
            Class and box maps.
        """

        _, _, c5 = self.backbone(x)
        feat = self.dilated(c5)
        return {"cls": self.cls(feat), "box": self.box(feat)}


class SOLOv2Like(nn.Module):
    """SOLOv2 grid-conditioned dynamic-kernel instance segmenter."""

    def __init__(self, channels: int = 32, grid: int = 4, classes: int = 5) -> None:
        """Initialize SOLOv2-like model.

        Parameters
        ----------
        channels:
            Feature width.
        grid:
            Kernel prediction grid size.
        classes:
            Class count.
        """

        super().__init__()
        self.grid = grid
        self.fpn = TinyFPN(out_ch=channels)
        self.mask_feat = ConvAct(channels, channels)
        self.kernel = nn.Conv2d(channels + 2, channels, 1)
        self.cls = nn.Conv2d(channels + 2, classes, 1)

    def _coords(self, feat: torch.Tensor) -> torch.Tensor:
        """Create normalized SOLO coordinate channels.

        Parameters
        ----------
        feat:
            Feature map that determines the grid.

        Returns
        -------
        torch.Tensor
            Coordinate tensor.
        """

        bsz, _, height, width = feat.shape
        yy = torch.linspace(-1.0, 1.0, height, device=feat.device, dtype=feat.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=feat.device, dtype=feat.dtype)
        yv, xv = torch.meshgrid(yy, xx, indexing="ij")
        return torch.stack((xv, yv), dim=0).unsqueeze(0).expand(bsz, -1, -1, -1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict SOLOv2 dynamic masks and grid classes.

        Parameters
        ----------
        x:
            Image batch.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Class logits and dynamic masks.
        """

        p2, _, p4 = self.fpn(x)
        masks = self.mask_feat(p2)
        grid_feat = F.adaptive_avg_pool2d(p4, (self.grid, self.grid))
        grid_feat = torch.cat((grid_feat, self._coords(grid_feat)), dim=1)
        kernels = self.kernel(grid_feat).flatten(2).transpose(1, 2)
        cls = self.cls(grid_feat).flatten(2).transpose(1, 2)
        return cls, torch.einsum("bnc,bchw->bnhw", kernels, masks)


class TextGraphDetector(nn.Module):
    """DRRG/PSE-style text detector with region maps and graph reasoning."""

    def __init__(self, kind: str = "drrg", classes: int = 4) -> None:
        """Initialize text detector.

        Parameters
        ----------
        kind:
            Text detector family.
        classes:
            Pixel output channels.
        """

        super().__init__()
        self.kind = kind
        self.backbone = TinyFPN(out_ch=24)
        self.seg = nn.Conv2d(24, classes, 1)
        self.node_proj = nn.Linear(24, 24)
        self.edge = nn.Linear(48, 1)

    def forward(self, x: torch.Tensor) -> TensorDict:
        """Predict text regions and graph links.

        Parameters
        ----------
        x:
            Image batch.

        Returns
        -------
        TensorDict
            Segmentation maps, graph nodes, and edge logits.
        """

        p2, _, _ = self.backbone(x)
        maps = self.seg(p2)
        pooled = F.adaptive_avg_pool2d(p2, (3, 3)).flatten(2).transpose(1, 2)
        nodes = self.node_proj(pooled)
        pair = torch.cat(
            (nodes.unsqueeze(2).expand(-1, -1, 9, -1), nodes.unsqueeze(1).expand(-1, 9, -1, -1)),
            dim=-1,
        )
        edge = self.edge(pair).squeeze(-1)
        if "pse" in self.kind:
            maps = torch.cumsum(torch.sigmoid(maps), dim=1)
        return {"region_maps": maps, "graph_nodes": nodes, "edge_logits": edge}


class CorrelationFlow(nn.Module):
    """Optical-flow family with correlation volume, warping, and updates."""

    def __init__(self, kind: str = "pwc", iters: int = 3) -> None:
        """Initialize optical-flow model.

        Parameters
        ----------
        kind:
            Flow architecture family.
        iters:
            Number of recurrent/refinement steps.
        """

        super().__init__()
        self.kind = kind
        self.iters = iters
        self.enc = nn.Sequential(ConvAct(3, 16, stride=2), ConvAct(16, 24, stride=2))
        self.update = nn.GRUCell(24 * 2 + 9 + 2, 32)
        self.delta = nn.Linear(32, 2)
        self.context = nn.Conv2d(24, 32, 3, padding=1)
        self.mask = nn.Conv2d(32, 9, 1)

    def _warp(self, feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp a feature map with a dense flow field.

        Parameters
        ----------
        feat:
            Feature map to sample.
        flow:
            Flow tensor in feature pixels.

        Returns
        -------
        torch.Tensor
            Warped feature map.
        """

        batch, _, height, width = feat.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=feat.device, dtype=feat.dtype),
            torch.linspace(-1.0, 1.0, width, device=feat.device, dtype=feat.dtype),
            indexing="ij",
        )
        base = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, height, width, 2)
        norm = torch.stack(
            (
                flow[:, 0] / max(width - 1, 1) * 2.0,
                flow[:, 1] / max(height - 1, 1) * 2.0,
            ),
            dim=-1,
        )
        return F.grid_sample(feat, base + norm, align_corners=True)

    def _corr(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute a local 3x3 correlation volume.

        Parameters
        ----------
        a:
            First feature map.
        b:
            Second feature map.

        Returns
        -------
        torch.Tensor
            Local correlation tensor.
        """

        vals = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                shifted = torch.roll(b, shifts=(dy, dx), dims=(-2, -1))
                vals.append((a * shifted).mean(dim=1, keepdim=True))
        return torch.cat(vals, dim=1)

    def _all_pairs_corr(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute RAFT-style all-pairs correlation.

        Parameters
        ----------
        a:
            First feature map.
        b:
            Second feature map.

        Returns
        -------
        torch.Tensor
            Correlation tensor ``(B, HW, HW)``.
        """

        query = F.normalize(a.flatten(2).transpose(1, 2), dim=-1)
        key = F.normalize(b.flatten(2), dim=1)
        return torch.bmm(query, key)

    def forward(self, frames: torch.Tensor) -> TensorDict:
        """Estimate flow between two frames.

        Parameters
        ----------
        frames:
            Tensor shaped ``(B, 2, 3, H, W)``.

        Returns
        -------
        TensorDict
            Flow field and optional refinement masks.
        """

        img1 = frames[:, 0]
        img2 = frames[:, 1]
        f1 = self.enc(img1)
        f2 = self.enc(img2)
        flow = torch.zeros(
            f1.shape[0], 2, f1.shape[2], f1.shape[3], device=frames.device, dtype=frames.dtype
        )
        hidden = self.context(f1).flatten(2).transpose(1, 2).reshape(-1, 32)
        if "raft" in self.kind:
            all_pairs = self._all_pairs_corr(f1, f2)
            corr = (
                all_pairs.mean(dim=-1)
                .view(f1.shape[0], 1, f1.shape[2], f1.shape[3])
                .repeat(1, 9, 1, 1)
            )
        elif "pwc" in self.kind or "lite" in self.kind or "mask" in self.kind:
            coarse1 = F.avg_pool2d(f1, 2)
            coarse2 = F.avg_pool2d(f2, 2)
            coarse_corr = self._corr(coarse1, coarse2)
            coarse_flow = coarse_corr[:, :2]
            flow = F.interpolate(
                coarse_flow, size=f1.shape[-2:], mode="bilinear", align_corners=False
            )
            f2 = self._warp(f2, flow)
            corr = self._corr(f1, f2)
        else:
            corr = self._corr(f1, f2)
        loops = self.iters + (1 if "flownet2" in self.kind else 0)
        for idx in range(loops):
            if "flownet2" in self.kind and idx == self.iters:
                f2 = self._warp(f2, flow)
                corr = self._corr(f1, f2)
            step = torch.cat((f1, f2, corr, flow), dim=1).flatten(2).transpose(1, 2).reshape(-1, 59)
            hidden = self.update(step, hidden)
            delta = (
                self.delta(hidden)
                .view(f1.shape[0], f1.shape[2], f1.shape[3], 2)
                .permute(0, 3, 1, 2)
            )
            flow = flow + delta
            if "gma" in self.kind or "raft" in self.kind:
                attn = torch.softmax(corr.flatten(2), dim=-1).view_as(corr)
                corr = corr + attn * corr
        out: TensorDict = {"flow": flow, "correlation": corr}
        if "mask" in self.kind or "lite" in self.kind:
            out["upsample_mask"] = self.mask(
                hidden.view(f1.shape[0], f1.shape[2], f1.shape[3], 32).permute(0, 3, 1, 2)
            )
        return out


class VideoActionModel(nn.Module):
    """Action model variants for CLIP4Clip, ACRN, skeleton GCN, and audio."""

    def __init__(self, kind: str = "clip4clip", classes: int = 8) -> None:
        """Initialize video/audio action model.

        Parameters
        ----------
        kind:
            Variant family.
        classes:
            Output class count.
        """

        super().__init__()
        self.kind = kind
        if "skeleton" in kind:
            self.joint_embed = nn.Linear(2, 32)
            self.gcn = nn.Linear(32, 32)
        elif "audio" in kind:
            self.audio_stem = nn.Conv1d(1, 24, 7, stride=2, padding=3)
            self.audio_res1 = nn.Conv1d(24, 24, 3, padding=1)
            self.audio_res2 = nn.Conv1d(24, 32, 3, stride=2, padding=1)
            self.audio_skip = nn.Conv1d(24, 32, 1, stride=2)
        else:
            self.frame = nn.Conv3d(3, 32, (3, 3, 3), padding=1)
            self.actor = nn.Conv3d(3, 32, (1, 3, 3), padding=(0, 1, 1))
            self.context = nn.Conv3d(3, 32, (3, 5, 5), padding=(1, 2, 2))
            self.relation = nn.MultiheadAttention(32, 4, batch_first=True)
            self.text = nn.Embedding(64, 32)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(32, 4, 64, batch_first=True), 1
            )
        self.head = nn.Linear(32, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run action/retrieval inference.

        Parameters
        ----------
        x:
            Video, skeleton, or audio tensor.

        Returns
        -------
        torch.Tensor
            Logits or retrieval scores.
        """

        if "skeleton" in self.kind:
            nodes = self.joint_embed(x)
            adj = torch.softmax(torch.matmul(nodes, nodes.transpose(-1, -2)) / 32.0**0.5, dim=-1)
            feat = self.gcn(torch.matmul(adj, nodes)).mean(dim=(1, 2))
            return self.head(feat)
        if "audio" in self.kind:
            stem = F.relu(self.audio_stem(x))
            res = F.relu(self.audio_res1(stem)) + stem
            feat = F.relu(self.audio_res2(res) + self.audio_skip(res)).mean(dim=-1)
            return self.head(feat)
        if "acrn" in self.kind:
            actor = (
                F.adaptive_avg_pool3d(F.relu(self.actor(x)), (x.shape[2], 2, 2))
                .flatten(2)
                .transpose(1, 2)
            )
            context = (
                F.adaptive_avg_pool3d(F.relu(self.context(x)), (x.shape[2], 2, 2))
                .flatten(2)
                .transpose(1, 2)
            )
            tokens = actor + context
            related = self.relation(tokens, tokens, tokens, need_weights=False)[0]
            feat = related.mean(dim=1)
            return self.head(feat)
        vid = self.frame(x).mean(dim=(3, 4)).transpose(1, 2)
        text_ids = torch.arange(6, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        txt = self.text(text_ids)
        fused = self.transformer(torch.cat((vid, txt), dim=1)).mean(dim=1)
        return self.head(fused)


class MultiObjectTracker(nn.Module):
    """Tracking-by-detection model with motion and association heads."""

    def __init__(self, kind: str = "deepsort") -> None:
        """Initialize tracker.

        Parameters
        ----------
        kind:
            Tracker family.
        """

        super().__init__()
        self.kind = kind
        self.detector = DenseDetector("yolox" if "ocsort" in kind or "byte" in kind else "gfl")
        self.reid = nn.Linear(32, 16)
        self.motion = nn.GRUCell(4, 16)
        self.box_head = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> TensorDict:
        """Detect objects and update track states.

        Parameters
        ----------
        x:
            Two-frame tensor ``(B, 2, 3, H, W)``.

        Returns
        -------
        TensorDict
            Detection maps, motion boxes, and association embeddings.
        """

        det = self.detector(x[:, -1])
        pooled = F.adaptive_avg_pool2d(det["cls"], (2, 2)).flatten(2).transpose(1, 2)
        embed = F.normalize(
            self.reid(F.pad(pooled, (0, max(0, 32 - pooled.shape[-1])))[:, :, :32]), dim=-1
        )
        if "ocsort" in self.kind or "byte" in self.kind:
            prev = self.detector(x[:, 0])
            curr_score = F.adaptive_avg_pool2d(det["quality"], (2, 2)).flatten(2).transpose(1, 2)
            curr_obs = F.adaptive_avg_pool2d(det["box"][:, :4], (2, 2)).flatten(2).transpose(1, 2)
            prev_obs = F.adaptive_avg_pool2d(prev["box"][:, :4], (2, 2)).flatten(2).transpose(1, 2)
            velocity = curr_obs - prev_obs
            kalman_gain = torch.sigmoid(curr_score.mean(dim=1, keepdim=True))
            predicted = curr_obs + kalman_gain * velocity
            association = torch.matmul(
                F.normalize(predicted, dim=-1), F.normalize(prev_obs, dim=-1).transpose(1, 2)
            )
            out: TensorDict = {
                "det_cls": det["cls"],
                "observations": curr_obs,
                "predicted_boxes": predicted,
                "ocsort_association": association,
                "reid": embed,
            }
            if "byte" in self.kind:
                high = (curr_score > curr_score.mean(dim=1, keepdim=True)).to(x.dtype)
                low = 1.0 - high
                out["high_conf_assoc"] = association * high
                out["low_conf_assoc"] = association * low
            return out
        state = torch.zeros(x.shape[0] * 4, 16, device=x.device, dtype=x.dtype)
        boxes = torch.zeros(x.shape[0] * 4, 4, device=x.device, dtype=x.dtype)
        for _ in range(2):
            state = self.motion(boxes, state)
            boxes = boxes + self.box_head(state)
        return {"det_cls": det["cls"], "track_boxes": boxes.view(x.shape[0], 4, 4), "reid": embed}


class STARKTracker(nn.Module):
    """STARK-style template-search transformer tracker."""

    def __init__(self, queries: int = 4) -> None:
        """Initialize compact STARK tracker.

        Parameters
        ----------
        queries:
            Number of target queries.
        """

        super().__init__()
        self.template = nn.Conv2d(3, 32, 4, stride=4)
        self.search = nn.Conv2d(3, 32, 4, stride=4)
        self.query = nn.Embedding(queries, 32)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(32, 4, 64, batch_first=True), 1
        )
        self.box = nn.Linear(32, 4)
        self.score = nn.Linear(32, 1)

    def forward(self, clip: torch.Tensor) -> TensorDict:
        """Track a template object in the search frame.

        Parameters
        ----------
        clip:
            Video tensor ``(B, 3, T, H, W)``.

        Returns
        -------
        TensorDict
            Tracking boxes and confidence logits.
        """

        template = self.template(clip[:, :, 0]).flatten(2).transpose(1, 2)
        search = self.search(clip[:, :, -1]).flatten(2).transpose(1, 2)
        memory = torch.cat((template, search), dim=1)
        query = self.query.weight.unsqueeze(0).expand(clip.shape[0], -1, -1)
        decoded = self.decoder(query, memory)
        return {"track_box": self.box(decoded).sigmoid(), "track_score": self.score(decoded)}


class TemporalRoIAlignTracker(nn.Module):
    """Temporal RoIAlign tracker over two-frame FPN features."""

    def __init__(self, proposals: int = 4) -> None:
        """Initialize temporal RoIAlign tracker.

        Parameters
        ----------
        proposals:
            Number of compact proposal features.
        """

        super().__init__()
        self.proposals = proposals
        self.fpn = TinyFPN()
        self.rpn = RPNHead()
        self.temporal = nn.MultiheadAttention(32, 4, batch_first=True)
        self.head = nn.Linear(32, 4)

    def _roi_tokens(self, feat: torch.Tensor) -> torch.Tensor:
        """Extract top proposal RoI tokens.

        Parameters
        ----------
        feat:
            FPN feature map.

        Returns
        -------
        torch.Tensor
            RoI tokens.
        """

        obj, _ = self.rpn(feat)
        idx = torch.topk(obj.flatten(2).mean(dim=1), self.proposals, dim=-1).indices
        flat = feat.flatten(2).transpose(1, 2)
        gathered = torch.gather(flat, 1, idx.unsqueeze(-1).expand(-1, -1, flat.shape[-1]))
        return F.adaptive_avg_pool1d(gathered.transpose(1, 2), 4).transpose(1, 2)

    def forward(self, frames: torch.Tensor) -> TensorDict:
        """Align proposal features over two frames.

        Parameters
        ----------
        frames:
            Two-frame tensor ``(B, 2, 3, H, W)``.

        Returns
        -------
        TensorDict
            Temporally aligned boxes and tokens.
        """

        feat0 = self.fpn(frames[:, 0])[1]
        feat1 = self.fpn(frames[:, 1])[1]
        tokens = torch.cat((self._roi_tokens(feat0), self._roi_tokens(feat1)), dim=1)
        aligned = self.temporal(tokens, tokens, tokens, need_weights=False)[0]
        return {"temporal_roi_tokens": aligned, "track_box": self.head(aligned.mean(dim=1))}


class QueryInstCompact(nn.Module):
    """QueryInst-style query detector with dynamic mask heads."""

    def __init__(self, queries: int = 6, classes: int = 5) -> None:
        """Initialize compact QueryInst.

        Parameters
        ----------
        queries:
            Number of instance queries.
        classes:
            Number of object classes.
        """

        super().__init__()
        self.fpn = TinyFPN()
        self.query = nn.Embedding(queries, 32)
        self.cross = nn.MultiheadAttention(32, 4, batch_first=True)
        self.cls = nn.Linear(32, classes)
        self.box = nn.Linear(32, 4)
        self.kernel = nn.Linear(32, 32)
        self.mask_feat = ConvAct(32, 32)

    def forward(self, image: torch.Tensor) -> TensorDict:
        """Predict query-conditioned instance boxes and masks.

        Parameters
        ----------
        image:
            RGB image batch.

        Returns
        -------
        TensorDict
            Query classes, boxes, and dynamic masks.
        """

        p2, _, p4 = self.fpn(image)
        memory = p4.flatten(2).transpose(1, 2)
        query = self.query.weight.unsqueeze(0).expand(image.shape[0], -1, -1)
        inst = self.cross(query, memory, memory, need_weights=False)[0]
        masks = torch.einsum("bqc,bchw->bqhw", self.kernel(inst), self.mask_feat(p2))
        return {
            "query_cls": self.cls(inst),
            "query_box": self.box(inst).sigmoid(),
            "dynamic_masks": masks,
        }


class BoxInstCompact(nn.Module):
    """BoxInst-style box-supervised instance segmentation with pairwise constraints."""

    def __init__(self, classes: int = 5) -> None:
        """Initialize compact BoxInst.

        Parameters
        ----------
        classes:
            Number of classes.
        """

        super().__init__()
        self.detector = DenseDetector("fcos", classes=classes)
        self.mask_branch = SOLOv2Like(classes=classes)

    def forward(self, image: torch.Tensor) -> TensorDict:
        """Predict masks with projection and pairwise consistency terms.

        Parameters
        ----------
        image:
            RGB image batch.

        Returns
        -------
        TensorDict
            BoxInst detection, masks, projection, and pairwise maps.
        """

        det = self.detector(image)
        cls, masks = self.mask_branch(image)
        projection = torch.stack((masks.amax(dim=-1), masks.amax(dim=-2)), dim=-1)
        pairwise = (masks[:, :, :, 1:] - masks[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
        return {
            "det_cls": det["cls"],
            "mask_cls": cls,
            "masks": masks,
            "projection": projection,
            "pairwise": pairwise,
        }


class Mask2FormerVISCompact(nn.Module):
    """Mask2Former VIS model with masked-attention transformer decoding."""

    def __init__(self, queries: int = 6, classes: int = 5) -> None:
        """Initialize compact Mask2Former VIS.

        Parameters
        ----------
        queries:
            Number of mask queries.
        classes:
            Number of classes.
        """

        super().__init__()
        self.fpn = TinyFPN()
        self.query = nn.Embedding(queries, 32)
        self.self_attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.mask_embed = nn.Linear(32, 32)
        self.cls = nn.Linear(32, classes)

    def forward(self, image: torch.Tensor) -> TensorDict:
        """Predict class logits and masks through masked attention.

        Parameters
        ----------
        image:
            RGB image batch.

        Returns
        -------
        TensorDict
            Mask2Former query logits, masks, and association embeddings.
        """

        p2, _, p4 = self.fpn(image)
        memory = p4.flatten(2).transpose(1, 2)
        query = self.query.weight.unsqueeze(0).expand(image.shape[0], -1, -1)
        query = self.self_attn(query, query, query, need_weights=False)[0]
        coarse_mask = torch.matmul(query, memory.transpose(1, 2)).sigmoid()
        masked_memory = memory * coarse_mask.mean(dim=1, keepdim=True).transpose(1, 2)
        query = self.cross_attn(query, masked_memory, masked_memory, need_weights=False)[0]
        masks = torch.einsum("bqc,bchw->bqhw", self.mask_embed(query), p2)
        assoc = F.normalize(query[:, 1:] - query[:, :-1], dim=-1)
        return {"mask_cls": self.cls(query), "masks": masks, "vis_association": assoc}


class Point3DSSD(nn.Module):
    """3DSSD point detector with sampling-free set abstraction and center head."""

    def __init__(self, classes: int = 3) -> None:
        """Initialize 3DSSD.

        Parameters
        ----------
        classes:
            Number of 3D classes.
        """

        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 32))
        self.sa = nn.Sequential(nn.Linear(4 + 3, 32), nn.ReLU(), nn.Linear(32, 32))
        self.fusion = nn.Linear(64, 32)
        self.center = nn.Linear(32, 3)
        self.box = nn.Linear(32, 7)
        self.cls = nn.Linear(32, classes)

    def forward(self, points: torch.Tensor) -> TensorDict:
        """Predict point-wise 3D centers and boxes.

        Parameters
        ----------
        points:
            Point tensor ``(B, N, 4)``.

        Returns
        -------
        TensorDict
            Center, box, and class predictions.
        """

        xyz = points[..., :3]
        dist = torch.cdist(xyz, xyz)
        idx = torch.topk(dist, k=min(8, points.shape[1]), dim=-1, largest=False).indices
        batch = torch.arange(points.shape[0], device=points.device).view(-1, 1, 1).expand_as(idx)
        grouped = points[batch, idx]
        rel = grouped[..., :3] - xyz.unsqueeze(2)
        set_feat = self.sa(torch.cat((grouped, rel), dim=-1)).amax(dim=2)
        point_feat = self.mlp(points)
        fused = F.relu(self.fusion(torch.cat((point_feat, set_feat), dim=-1)))
        center_feat = fused + fused.mean(dim=1, keepdim=True)
        return {
            "set_abstraction": set_feat,
            "center": self.center(center_feat),
            "box": self.box(center_feat),
            "cls": self.cls(center_feat),
        }


class OCRRecognizer(nn.Module):
    """ABINet/ASTER-style scene text recognizer."""

    def __init__(self, kind: str = "abinet", vocab: int = 32) -> None:
        """Initialize OCR recognizer.

        Parameters
        ----------
        kind:
            OCR family.
        vocab:
            Character vocabulary size.
        """

        super().__init__()
        self.kind = kind
        self.cnn = nn.Sequential(ConvAct(3, 24, stride=2), ConvAct(24, 32, stride=2))
        self.rectifier = nn.Linear(32, 6)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(32, 4, 64, batch_first=True), 1
        )
        self.language = nn.GRU(32, 32, batch_first=True)
        self.head = nn.Linear(32, vocab)

    def forward(self, x: torch.Tensor) -> TensorDict:
        """Recognize text from an image.

        Parameters
        ----------
        x:
            Text-line image batch.

        Returns
        -------
        TensorDict
            Visual logits and optional language-refined logits.
        """

        feat = self.cnn(x)
        theta = self.rectifier(feat.mean(dim=(2, 3)))
        seq = feat.mean(dim=2).transpose(1, 2)
        visual = self.encoder(seq)
        out: TensorDict = {"visual_logits": self.head(visual), "rectifier_theta": theta}
        if "abinet" in self.kind:
            lang, _ = self.language(visual)
            out["language_logits"] = self.head(lang + visual)
        return out


class HRNetPose(nn.Module):
    """HRNet pose model with parallel high-resolution branches."""

    def __init__(self, joints: int = 17) -> None:
        """Initialize HRNet pose estimator.

        Parameters
        ----------
        joints:
            Number of output keypoints.
        """

        super().__init__()
        self.stem = ConvAct(3, 24, stride=2)
        self.high = ConvAct(24, 24)
        self.low = ConvAct(24, 48, stride=2)
        self.fuse = nn.Conv2d(72, 32, 1)
        self.heatmap = nn.Conv2d(32, joints, 1)
        self.tags = nn.Conv2d(32, joints, 1)

    def forward(self, x: torch.Tensor) -> TensorDict:
        """Predict keypoint heatmaps.

        Parameters
        ----------
        x:
            Input image batch.

        Returns
        -------
        TensorDict
            Joint heatmaps and associative embedding tags.
        """

        base = self.stem(x)
        high = self.high(base)
        low = F.interpolate(self.low(base), size=high.shape[-2:], mode="nearest")
        feat = F.silu(self.fuse(torch.cat((high, low), dim=1)))
        return {"heatmap": self.heatmap(feat), "associative_tags": self.tags(feat)}


class AnimatedDiffTiny(nn.Module):
    """AnimateDiff-style temporal adapter around a latent U-Net block."""

    def __init__(self, channels: int = 4) -> None:
        """Initialize AnimateDiff model.

        Parameters
        ----------
        channels:
            Latent channel count.
        """

        super().__init__()
        self.spatial = nn.Conv3d(channels, 32, (1, 3, 3), padding=(0, 1, 1))
        self.temporal = nn.Conv3d(32, 32, (3, 1, 1), padding=(1, 0, 0), groups=32)
        self.out = nn.Conv3d(32, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Denoise a latent video clip.

        Parameters
        ----------
        x:
            Latent video tensor.

        Returns
        -------
        torch.Tensor
            Predicted noise tensor.
        """

        return self.out(F.silu(self.temporal(F.silu(self.spatial(x)))))


class AOTGANInpaint(nn.Module):
    """AOT-GAN inpainting generator with aggregated contextual dilations."""

    def __init__(self) -> None:
        """Initialize AOT-GAN compact generator."""

        super().__init__()
        self.enc = ConvAct(4, 32, stride=2)
        self.dilations = nn.ModuleList(
            [nn.Conv2d(32, 32, 3, padding=d, dilation=d) for d in (1, 2, 4)]
        )
        self.gate = nn.Conv2d(96, 3, 1)
        self.dec = nn.ConvTranspose2d(32, 3, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inpaint a masked RGB image.

        Parameters
        ----------
        x:
            Four-channel RGB plus mask tensor.

        Returns
        -------
        torch.Tensor
            Inpainted RGB image.
        """

        feat = self.enc(x)
        branches = [F.silu(layer(feat)) for layer in self.dilations]
        gates = torch.softmax(self.gate(torch.cat(branches, dim=1)), dim=1)
        fused = sum(gates[:, i : i + 1] * branches[i] for i in range(3))
        return torch.tanh(self.dec(fused))


class BasicVSRFamily(nn.Module):
    """BasicVSR/BasicVSR++ bidirectional recurrent video super-resolution."""

    def __init__(self, plus: bool = False) -> None:
        """Initialize BasicVSR variant.

        Parameters
        ----------
        plus:
            Whether to include second-order alignment as in BasicVSR++.
        """

        super().__init__()
        self.plus = plus
        self.feat = ConvAct(3, 24)
        self.prop = ConvAct(48, 24)
        self.align = nn.Conv2d(24 * 3 + 3, 2 if plus else 2, 3, padding=1)
        self.up = nn.Sequential(
            nn.Conv2d(24, 48, 3, padding=1), nn.PixelShuffle(2), nn.Conv2d(12, 3, 1)
        )

    def _grid(self, feat: torch.Tensor) -> torch.Tensor:
        """Create a normalized sampling grid.

        Parameters
        ----------
        feat:
            Feature map.

        Returns
        -------
        torch.Tensor
            Sampling grid.
        """

        batch, _, height, width = feat.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=feat.device, dtype=feat.dtype),
            torch.linspace(-1.0, 1.0, width, device=feat.device, dtype=feat.dtype),
            indexing="ij",
        )
        return torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, height, width, 2)

    def _align(
        self, feat: torch.Tensor, one: torch.Tensor, two: torch.Tensor, frame: torch.Tensor
    ) -> torch.Tensor:
        """Align propagated states with first- or second-order flow.

        Parameters
        ----------
        feat:
            Current frame feature.
        one:
            First-order propagated state.
        two:
            Second-order propagated state.
        frame:
            Current RGB frame.

        Returns
        -------
        torch.Tensor
            Flow-aligned state.
        """

        flow = torch.tanh(self.align(torch.cat((feat, one, two, frame), dim=1))) * 0.25
        state = one + (0.5 * two if self.plus else 0.0)
        return F.grid_sample(state, self._grid(feat) + flow.permute(0, 2, 3, 1), align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Super-resolve a video sequence.

        Parameters
        ----------
        x:
            Low-resolution video ``(B, T, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Super-resolved center frame.
        """

        feats = [self.feat(x[:, i]) for i in range(x.shape[1])]
        state = torch.zeros_like(feats[0])
        prev_state = torch.zeros_like(state)
        for feat in reversed(feats):
            aligned = self._align(feat, state, prev_state, x[:, 0])
            prev_state = state
            state = self.prop(torch.cat((feat, aligned), dim=1))
        prev_state = torch.zeros_like(state)
        for feat in feats:
            aligned = self._align(feat, state, prev_state, x[:, -1])
            prev_state = state
            state = self.prop(torch.cat((feat, aligned), dim=1))
        return self.up(state)


class RealBasicVSRFamily(nn.Module):
    """RealBasicVSR with recurrent cleaning followed by BasicVSR++ restoration."""

    def __init__(self) -> None:
        """Initialize compact RealBasicVSR."""

        super().__init__()
        self.clean = nn.Sequential(ConvAct(3, 16), ConvAct(16, 3))
        self.restore = BasicVSRFamily(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clean and restore a low-quality video.

        Parameters
        ----------
        x:
            Low-quality video ``(B, T, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Restored center frame.
        """

        cleaned = []
        residual = torch.zeros_like(x[:, 0])
        for idx in range(x.shape[1]):
            residual = self.clean(x[:, idx] + 0.1 * residual)
            cleaned.append(x[:, idx] + residual)
        return self.restore(torch.stack(cleaned, dim=1))


class BigGANSmall(nn.Module):
    """BigGAN generator with class conditioning, conditional BN, and self-attention."""

    def __init__(self, z_dim: int = 16, classes: int = 10) -> None:
        """Initialize BigGAN compact generator.

        Parameters
        ----------
        z_dim:
            Latent width.
        classes:
            Class embedding count.
        """

        super().__init__()
        self.embed = nn.Embedding(classes, z_dim)
        self.fc = nn.Linear(z_dim, 32 * 4 * 4)
        self.to_gamma = nn.Linear(z_dim, 32)
        self.to_beta = nn.Linear(z_dim, 32)
        self.qkv = nn.Conv2d(32, 96, 1)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate an image from latent and implicit class ids.

        Parameters
        ----------
        z:
            Latent tensor.

        Returns
        -------
        torch.Tensor
            Generated image.
        """

        labels = torch.arange(z.shape[0], device=z.device) % self.embed.num_embeddings
        cond = z + self.embed(labels)
        feat = self.fc(cond).view(z.shape[0], 32, 4, 4)
        gamma = self.to_gamma(cond).view(z.shape[0], 32, 1, 1)
        beta = self.to_beta(cond).view(z.shape[0], 32, 1, 1)
        feat = F.relu(F.batch_norm(feat, None, None, training=True) * (1.0 + gamma) + beta)
        q, k, v = self.qkv(feat).chunk(3, dim=1)
        attn = torch.softmax(
            torch.matmul(q.flatten(2).transpose(1, 2), k.flatten(2)) / 32.0**0.5, dim=-1
        )
        feat = feat + torch.matmul(attn, v.flatten(2).transpose(1, 2)).transpose(1, 2).view_as(feat)
        return torch.tanh(self.out(feat))


class BigGANDeepBlock(nn.Module):
    """BigGAN-deep residual upsampling block with conditional affine normalization."""

    def __init__(self, in_ch: int, out_ch: int, z_dim: int) -> None:
        """Initialize a deep BigGAN residual block.

        Parameters
        ----------
        in_ch:
            Input channels.
        out_ch:
            Output channels.
        z_dim:
            Conditioning latent width.
        """

        super().__init__()
        self.gamma1 = nn.Linear(z_dim, in_ch)
        self.beta1 = nn.Linear(z_dim, in_ch)
        self.gamma2 = nn.Linear(z_dim, out_ch)
        self.beta2 = nn.Linear(z_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def _cond_norm(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Apply conditional affine normalization.

        Parameters
        ----------
        x:
            Feature map.
        gamma:
            Scale vector.
        beta:
            Shift vector.

        Returns
        -------
        torch.Tensor
            Normalized feature map.
        """

        normed = F.batch_norm(x, None, None, training=True)
        return normed * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply residual upsampling synthesis.

        Parameters
        ----------
        x:
            Input feature map.
        cond:
            Conditioning vector.

        Returns
        -------
        torch.Tensor
            Upsampled feature map.
        """

        h = F.interpolate(
            F.relu(self._cond_norm(x, self.gamma1(cond), self.beta1(cond))),
            scale_factor=2,
            mode="nearest",
        )
        h = self.conv1(h)
        h = self.conv2(F.relu(self._cond_norm(h, self.gamma2(cond), self.beta2(cond))))
        return h + self.skip(F.interpolate(x, scale_factor=2, mode="nearest"))


class BigGANDeep(nn.Module):
    """BigGAN-deep generator with hierarchical latent chunks and self-attention."""

    def __init__(self, z_dim: int = 16, classes: int = 10) -> None:
        """Initialize compact BigGAN-deep.

        Parameters
        ----------
        z_dim:
            Latent width.
        classes:
            Class embedding count.
        """

        super().__init__()
        self.embed = nn.Embedding(classes, z_dim)
        self.fc = nn.Linear(z_dim, 64 * 4 * 4)
        self.block1 = BigGANDeepBlock(64, 48, z_dim)
        self.block2 = BigGANDeepBlock(48, 32, z_dim)
        self.qkv = nn.Conv2d(32, 96, 1)
        self.block3 = BigGANDeepBlock(32, 16, z_dim)
        self.out = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate an image from latent and implicit class conditioning.

        Parameters
        ----------
        z:
            Latent tensor.

        Returns
        -------
        torch.Tensor
            Generated image.
        """

        labels = z[:, -1].abs().long() % self.embed.num_embeddings
        cond = z + self.embed(labels)
        feat = self.fc(cond).view(z.shape[0], 64, 4, 4)
        feat = self.block1(feat, cond)
        feat = self.block2(feat, cond)
        q, k, v = self.qkv(feat).chunk(3, dim=1)
        attn = torch.softmax(
            torch.bmm(q.flatten(2).transpose(1, 2), k.flatten(2)) / 32.0**0.5, dim=-1
        )
        feat = feat + torch.bmm(attn, v.flatten(2).transpose(1, 2)).transpose(1, 2).view_as(feat)
        feat = self.block3(feat, cond)
        return torch.tanh(self.out(F.relu(feat)))


def image_input() -> torch.Tensor:
    """Return a small RGB image input.

    Returns
    -------
    torch.Tensor
        Image tensor.
    """

    return torch.randn(1, 3, 64, 64)


def video_pair_input() -> torch.Tensor:
    """Return a two-frame video input.

    Returns
    -------
    torch.Tensor
        Video pair tensor.
    """

    return torch.randn(1, 2, 3, 64, 64)


def flow_input() -> torch.Tensor:
    """Return a two-frame optical-flow input.

    Returns
    -------
    torch.Tensor
        Flow input tensor.
    """

    return torch.randn(1, 2, 3, 48, 48)


def clip_input() -> torch.Tensor:
    """Return a compact video clip input.

    Returns
    -------
    torch.Tensor
        Video tensor.
    """

    return torch.randn(1, 3, 4, 32, 32)


def skeleton_input() -> torch.Tensor:
    """Return a two-stream skeleton tensor.

    Returns
    -------
    torch.Tensor
        Skeleton tensor ``(B, streams, joints, xy)``.
    """

    return torch.randn(1, 2, 17, 2)


def audio_input() -> torch.Tensor:
    """Return a waveform-like audio input.

    Returns
    -------
    torch.Tensor
        Audio tensor.
    """

    return torch.randn(1, 1, 128)


def points_input() -> torch.Tensor:
    """Return a point-cloud input.

    Returns
    -------
    torch.Tensor
        Point tensor.
    """

    return torch.randn(1, 32, 4)


def text_image_input() -> torch.Tensor:
    """Return a scene-text image input.

    Returns
    -------
    torch.Tensor
        Text image tensor.
    """

    return torch.randn(1, 3, 32, 96)


def latent_video_input() -> torch.Tensor:
    """Return a latent video diffusion input.

    Returns
    -------
    torch.Tensor
        Latent video tensor.
    """

    return torch.randn(1, 4, 4, 16, 16)


def masked_image_input() -> torch.Tensor:
    """Return an RGB-plus-mask inpainting input.

    Returns
    -------
    torch.Tensor
        Masked image tensor.
    """

    return torch.randn(1, 4, 64, 64)


def vsr_input() -> torch.Tensor:
    """Return a low-resolution video super-resolution input.

    Returns
    -------
    torch.Tensor
        Video tensor.
    """

    return torch.randn(1, 4, 3, 24, 24)


def latent_input() -> torch.Tensor:
    """Return a generator latent input.

    Returns
    -------
    torch.Tensor
        Latent tensor.
    """

    return torch.randn(1, 16)


def _build_two_stage(kind: str) -> nn.Module:
    """Build a two-stage detector variant.

    Parameters
    ----------
    kind:
        Variant name.

    Returns
    -------
    nn.Module
        Detector in eval mode.
    """

    return TwoStageDetector(kind).eval()


def _build_dense(kind: str) -> nn.Module:
    """Build a dense detector variant.

    Parameters
    ----------
    kind:
        Dense detector family.

    Returns
    -------
    nn.Module
        Detector in eval mode.
    """

    return DenseDetector(kind).eval()


def _make_builder(factory: Callable[[], nn.Module]) -> Callable[[], nn.Module]:
    """Create a documented build function for a MENAGERIE entry.

    Parameters
    ----------
    factory:
        Zero-argument model factory.

    Returns
    -------
    Callable[[], nn.Module]
        Build function.
    """

    def build_model() -> nn.Module:
        """Build the compact random-initialized classic.

        Returns
        -------
        nn.Module
            Model in evaluation mode.
        """

        return factory().eval()

    return build_model


build_clip4clip = _make_builder(lambda: VideoActionModel("clip4clip"))
build_deepsort = _make_builder(lambda: MultiObjectTracker("deepsort"))
build_flownet = _make_builder(lambda: CorrelationFlow("flownet", iters=1))
build_flownet2 = _make_builder(lambda: CorrelationFlow("flownet2", iters=2))
build_gma = _make_builder(lambda: CorrelationFlow("gma", iters=3))
build_irr = _make_builder(lambda: CorrelationFlow("irr", iters=2))
build_liteflownet = _make_builder(lambda: CorrelationFlow("liteflownet", iters=2))
build_liteflownet2 = _make_builder(lambda: CorrelationFlow("liteflownet2", iters=3))
build_maskflownet = _make_builder(lambda: CorrelationFlow("maskflownet", iters=2))
build_masktrack = _make_builder(lambda: TwoStageDetector("masktrack"))
build_ocsort = _make_builder(lambda: MultiObjectTracker("ocsort"))
build_pwcnet = _make_builder(lambda: CorrelationFlow("pwcnet", iters=2))
build_qdtrack = _make_builder(lambda: TwoStageDetector("qdtrack"))
build_raft = _make_builder(lambda: CorrelationFlow("raft", iters=4))
build_stark = _make_builder(lambda: STARKTracker())
build_temporal_roi_align = _make_builder(lambda: TemporalRoIAlignTracker())
build_tracktor = _make_builder(lambda: MultiObjectTracker("tracktor"))
build_drrg = _make_builder(lambda: TextGraphDetector("drrg"))
build_pse = _make_builder(lambda: TextGraphDetector("pse"))
build_queryinst = _make_builder(lambda: QueryInstCompact())
build_centerpoint2d_ttfnet = _make_builder(lambda: DenseDetector("ttfnet"))
build_atss = _make_builder(lambda: DenseDetector("atss"))
build_autoassign = _make_builder(lambda: DenseDetector("autoassign"))
build_boxinst = _make_builder(lambda: BoxInstCompact())
build_bytetrack = _make_builder(lambda: MultiObjectTracker("bytetrack"))
build_carafe = _make_builder(lambda: CARAFEUpsampler())
build_htc = _make_builder(lambda: TwoStageDetector("htc-cascade-mask"))
build_dod = _make_builder(lambda: TwoStageDetector("distilling-object-detectors"))
build_dyhead_atss = _make_builder(lambda: DenseDetector("dyhead-atss"))
build_faster_rcnn = _make_builder(lambda: TwoStageDetector("faster"))
build_foveabox = _make_builder(lambda: DenseDetector("foveabox"))
build_freeanchor = _make_builder(lambda: DenseDetector("freeanchor"))
build_fsaf = _make_builder(lambda: DenseDetector("fsaf"))
build_gfl = _make_builder(lambda: DenseDetector("gfl"))
build_gflv2 = _make_builder(lambda: DenseDetector("gflv2"))
build_glip = _make_builder(lambda: DenseDetector("glip-dyhead-atss"))
build_grounding_dino = _make_builder(lambda: DenseDetector("grounding-dino"))
build_lad = _make_builder(lambda: DenseDetector("lad"))
build_ld = _make_builder(lambda: DenseDetector("gfl-ld"))
build_mask_rcnn = _make_builder(lambda: TwoStageDetector("mask"))
build_mask2former_vis = _make_builder(lambda: Mask2FormerVISCompact())
build_paa = _make_builder(lambda: DenseDetector("paa"))
build_reppoints = _make_builder(lambda: DenseDetector("reppoints"))
build_solov2 = _make_builder(lambda: SOLOv2Like())
build_tood = _make_builder(lambda: DenseDetector("tood"))
build_vfnet = _make_builder(lambda: DenseDetector("vfnet"))
build_yolof = _make_builder(lambda: YOLOFDetector())
build_2s_agcn = _make_builder(lambda: VideoActionModel("skeleton"))
build_3dssd = _make_builder(lambda: Point3DSSD())
build_abinet = _make_builder(lambda: OCRRecognizer("abinet"))
build_acrn = _make_builder(lambda: VideoActionModel("acrn"))
build_hrnet_pose = _make_builder(lambda: HRNetPose())
build_animatediff = _make_builder(lambda: AnimatedDiffTiny())
build_aot_gan = _make_builder(lambda: AOTGANInpaint())
build_aster = _make_builder(lambda: OCRRecognizer("aster"))
build_audioonly = _make_builder(lambda: VideoActionModel("audio"))
build_basicvsr = _make_builder(lambda: BasicVSRFamily(False))
build_basicvsr_pp = _make_builder(lambda: BasicVSRFamily(True))
build_real_basicvsr = _make_builder(lambda: RealBasicVSRFamily())
build_biggan = _make_builder(lambda: BigGANDeep())


ENTRY_SPECS: Final[list[tuple[str, str, str, str, str, str, str, str]]] = [
    (
        "mmaction:clip4clip",
        "build_clip4clip",
        "clip_input",
        "clip4clip",
        "video/action-retrieval",
        "2021",
        "CLIP4Clip text-video retrieval with video/text Transformer fusion",
    ),
    (
        "mmaction2_retrieval_clip4clip_vit_base_p32_res224_clip_pre_8xb16_u12_5e_msrvtt_9k_rgb",
        "build_clip4clip",
        "clip_input",
        "clip4clip",
        "video/action-retrieval",
        "2021",
        "CLIP4Clip text-video retrieval with video/text Transformer fusion",
    ),
    (
        "mmtrack:deepsort",
        "build_deepsort",
        "video_pair_input",
        "deepsort",
        "vision/detection-tracking",
        "2017",
        "DeepSORT detector plus motion state and ReID association embeddings",
    ),
    (
        "mmflow:flownet",
        "build_flownet",
        "flow_input",
        "flownet",
        "vision/optical-flow",
        "2015",
        "FlowNet-style encoder with correlation and direct flow regression",
    ),
    (
        "mmflow:flownet2",
        "build_flownet2",
        "flow_input",
        "flownet2",
        "vision/optical-flow",
        "2017",
        "FlowNet2 stacked refinement over correlation flow",
    ),
    (
        "mmflow:gma",
        "build_gma",
        "flow_input",
        "gma",
        "vision/optical-flow",
        "2021",
        "GMA global-motion aggregation over correlation features",
    ),
    (
        "mmflow:irr",
        "build_irr",
        "flow_input",
        "irr",
        "vision/optical-flow",
        "2019",
        "IRR iterative residual flow refinement",
    ),
    (
        "mmflow:liteflownet",
        "build_liteflownet",
        "flow_input",
        "liteflownet",
        "vision/optical-flow",
        "2018",
        "LiteFlowNet compact correlation/refinement with learned upsample mask",
    ),
    (
        "mmflow:liteflownet2",
        "build_liteflownet2",
        "flow_input",
        "liteflownet2",
        "vision/optical-flow",
        "2019",
        "LiteFlowNet2 cascaded lightweight flow refinement",
    ),
    (
        "mmflow:maskflownet",
        "build_maskflownet",
        "flow_input",
        "maskflownet",
        "vision/optical-flow",
        "2020",
        "MaskFlowNet flow with learned occlusion/upsampling mask",
    ),
    (
        "mmtrack:masktrack_rcnn",
        "build_masktrack",
        "image_input",
        "masktrack-rcnn",
        "vision/detection-tracking",
        "2019",
        "MaskTrack R-CNN backbone/FPN/RPN/ROI/mask heads plus track embedding",
    ),
    (
        "mmtrack:ocsort",
        "build_ocsort",
        "video_pair_input",
        "ocsort",
        "vision/detection-tracking",
        "2023",
        "OC-SORT detection with observation-centric motion association",
    ),
    (
        "mmflow:pwcnet",
        "build_pwcnet",
        "flow_input",
        "pwcnet",
        "vision/optical-flow",
        "2018",
        "PWC-Net pyramid warping/correlation compact flow estimator",
    ),
    (
        "mmtrack:qdtrack",
        "build_qdtrack",
        "image_input",
        "qdtrack",
        "vision/detection-tracking",
        "2021",
        "QDTrack two-stage detector with quasi-dense track embeddings",
    ),
    (
        "mmflow:raft",
        "build_raft",
        "flow_input",
        "raft",
        "vision/optical-flow",
        "2020",
        "RAFT recurrent all-pairs/local correlation update loop",
    ),
    (
        "mmtrack:stark",
        "build_stark",
        "clip_input",
        "stark",
        "vision/detection-tracking",
        "2021",
        "STARK-style spatiotemporal Transformer tracker surrogate",
    ),
    (
        "mmtrack:temporal_roi_align",
        "build_temporal_roi_align",
        "video_pair_input",
        "temporal-roi-align",
        "vision/detection-tracking",
        "2020",
        "Temporal RoI feature alignment over two-frame features inside track head",
    ),
    (
        "mmtrack:tracktor",
        "build_tracktor",
        "video_pair_input",
        "tracktor",
        "vision/detection-tracking",
        "2019",
        "Tracktor tracking-by-regression with detector and motion association",
    ),
    (
        "DRRG-ResNet50",
        "build_drrg",
        "image_input",
        "drrg",
        "vision/text-detection",
        "2020",
        "DRRG text components with graph reasoning over region nodes",
    ),
    (
        "DRRG-ResNet50-Graph",
        "build_drrg",
        "image_input",
        "drrg",
        "vision/text-detection",
        "2020",
        "DRRG graph edge reasoning over detected text components",
    ),
    (
        "PSE-MobileNetV3",
        "build_pse",
        "image_input",
        "pse-net",
        "vision/text-detection",
        "2019",
        "PSE progressive scale expansion text segmentation maps",
    ),
    (
        "queryinst_r50_fpn",
        "build_queryinst",
        "image_input",
        "queryinst",
        "vision/instance-segmentation",
        "2021",
        "QueryInst query-conditioned boxes and dynamic instance masks over FPN features",
    ),
    (
        "centerpoint2d_ttfnet",
        "build_centerpoint2d_ttfnet",
        "image_input",
        "ttfnet",
        "vision/detection",
        "2019",
        "TTFNet center heatmap dense detector head",
    ),
    (
        "atss_r50_fpn",
        "build_atss",
        "image_input",
        "atss",
        "vision/detection",
        "2020",
        "ATSS FPN dense head with adaptive quality scoring",
    ),
    (
        "autoassign_r50_fpn",
        "build_autoassign",
        "image_input",
        "autoassign",
        "vision/detection",
        "2020",
        "AutoAssign dense detector with learned anchor/location assignment",
    ),
    (
        "mmdet_boxinst_boxinst_r101_fpn_ms_90k_coco",
        "build_boxinst",
        "image_input",
        "boxinst",
        "vision/instance-segmentation",
        "2021",
        "BoxInst box-supervised masks with projection and pairwise consistency constraints",
    ),
    (
        "mmdet_bytetrack_bytetrack_yolox_x_8xb4_80e_crowdhuman_mot17halftrain_test_mot17halfval",
        "build_bytetrack",
        "video_pair_input",
        "bytetrack",
        "vision/detection-tracking",
        "2021",
        "ByteTrack detector plus high/low confidence association embeddings",
    ),
    (
        "mmdet:carafe",
        "build_carafe",
        "carafe_input",
        "carafe",
        "vision/feature-upsampling",
        "2019",
        "CARAFE content-aware feature reassembly kernels",
    ),
    (
        "mmdet_carafe",
        "build_carafe",
        "carafe_input",
        "carafe",
        "vision/feature-upsampling",
        "2019",
        "CARAFE content-aware feature reassembly kernels",
    ),
    (
        "detectors_htc_r50",
        "build_htc",
        "image_input",
        "htc",
        "vision/instance-segmentation",
        "2019",
        "Hybrid Task Cascade with FPN/RPN/cascade ROI and mask heads",
    ),
    (
        "mmdet_detectors_detectors_htc_r101_20e_coco",
        "build_htc",
        "image_input",
        "htc",
        "vision/instance-segmentation",
        "2019",
        "Hybrid Task Cascade with interleaved cascade box and mask heads",
    ),
    (
        "mmdet_detectors_htc",
        "build_htc",
        "image_input",
        "htc",
        "vision/instance-segmentation",
        "2019",
        "Hybrid Task Cascade detector primitive",
    ),
    (
        "mmdet:dod",
        "build_dod",
        "image_input",
        "dod",
        "vision/detection",
        "2021",
        "Distilling Object Detectors two-stage detector teacher-style heads",
    ),
    (
        "dyhead_atss_r50",
        "build_dyhead_atss",
        "image_input",
        "dyhead-atss",
        "vision/detection",
        "2021",
        "DyHead dynamic scale/channel gating on ATSS dense head",
    ),
    (
        "mmdet_faster_rcnn_faster_rcnn_r50_fpn_carafe_1x_coco",
        "build_faster_rcnn",
        "image_input",
        "faster-rcnn",
        "vision/detection",
        "2015",
        "Faster R-CNN backbone/FPN/RPN/RoI box head",
    ),
    (
        "foveabox_r50_fpn",
        "build_foveabox",
        "image_input",
        "foveabox",
        "vision/detection",
        "2020",
        "FoveaBox anchor-free fovea heatmap and box head",
    ),
    (
        "freeanchor_r50_fpn",
        "build_freeanchor",
        "image_input",
        "freeanchor",
        "vision/detection",
        "2019",
        "FreeAnchor dense detector with learned anchor matching likelihood",
    ),
    (
        "fsaf_r50_fpn",
        "build_fsaf",
        "image_input",
        "fsaf",
        "vision/detection",
        "2019",
        "FSAF feature-selective anchor-free dense head",
    ),
    (
        "gfl_r50_fpn",
        "build_gfl",
        "image_input",
        "gfl",
        "vision/detection",
        "2020",
        "GFL quality-class logits with distributional box bins",
    ),
    (
        "gflv2_r50_fpn",
        "build_gflv2",
        "image_input",
        "gflv2",
        "vision/detection",
        "2021",
        "GFLv2 quality-guided distributional box regression",
    ),
    (
        "mmdet_glip",
        "build_glip",
        "image_input",
        "glip",
        "vision/language-detection",
        "2022",
        "GLIP grounded detector using DyHead-style dense visual grounding",
    ),
    (
        "mmdet_glip_glip_atss_swin_l_fpn_dyhead_pretrain_mixeddata",
        "build_glip",
        "image_input",
        "glip",
        "vision/language-detection",
        "2022",
        "GLIP ATSS/FPN/DyHead grounded detector primitive",
    ),
    (
        "mmdet_grounding_dino",
        "build_grounding_dino",
        "image_input",
        "grounding-dino",
        "vision/language-detection",
        "2023",
        "Grounding-DINO dense detection surrogate with grounding logits",
    ),
    (
        "mmdet_lad",
        "build_lad",
        "image_input",
        "lad",
        "vision/detection",
        "2021",
        "LAD label-assignment distillation dense detector",
    ),
    (
        "mmdet:ld",
        "build_ld",
        "image_input",
        "ld",
        "vision/detection",
        "2022",
        "Localization distillation on GFL distributional boxes",
    ),
    (
        "mmdet_ld",
        "build_ld",
        "image_input",
        "ld",
        "vision/detection",
        "2022",
        "Localization distillation on GFL distributional boxes",
    ),
    (
        "mmdet_ld_ld_r101_gflv1_r101_dcn_fpn_2x_coco",
        "build_ld",
        "image_input",
        "ld",
        "vision/detection",
        "2022",
        "LD GFL-style distributional localization distillation",
    ),
    (
        "mmdet_mask_rcnn_mask_rcnn_r50_fpn_carafe_1x_coco",
        "build_mask_rcnn",
        "image_input",
        "mask-rcnn",
        "vision/instance-segmentation",
        "2017",
        "Mask R-CNN adds mask head to Faster R-CNN FPN/RPN/RoI detector",
    ),
    (
        "mmdet:mask2former_vis",
        "build_mask2former_vis",
        "image_input",
        "mask2former-vis",
        "vision/video-instance-segmentation",
        "2022",
        "Mask2Former masked-attention Transformer decoder with VIS association embeddings",
    ),
    (
        "mmdet_mask2former_vis",
        "build_mask2former_vis",
        "image_input",
        "mask2former-vis",
        "vision/video-instance-segmentation",
        "2022",
        "Mask2Former masked-attention Transformer decoder with VIS association embeddings",
    ),
    (
        "mmdet_misc_d2_faster_rcnn_r50_caffe_fpn_ms_90k_coco",
        "build_faster_rcnn",
        "image_input",
        "faster-rcnn",
        "vision/detection",
        "2015",
        "Detectron2 Faster R-CNN backbone/FPN/RPN/RoI box head",
    ),
    (
        "mmdet:ocsort",
        "build_ocsort",
        "video_pair_input",
        "ocsort",
        "vision/detection-tracking",
        "2023",
        "OC-SORT observation-centric motion association",
    ),
    (
        "mmdet_ocsort",
        "build_ocsort",
        "video_pair_input",
        "ocsort",
        "vision/detection-tracking",
        "2023",
        "OC-SORT observation-centric motion association",
    ),
    (
        "mmdet_ocsort_ocsort_yolox_x_8xb4_amp_80e_crowdhuman_mot17halftrain_test_mot17halfval",
        "build_ocsort",
        "video_pair_input",
        "ocsort",
        "vision/detection-tracking",
        "2023",
        "OC-SORT YOLOX detector plus motion association",
    ),
    (
        "paa_r50_fpn",
        "build_paa",
        "image_input",
        "paa",
        "vision/detection",
        "2020",
        "PAA dense detector with probabilistic assignment/alignment score",
    ),
    (
        "QDTrack-FasterRCNN-R50",
        "build_qdtrack",
        "image_input",
        "qdtrack",
        "vision/detection-tracking",
        "2021",
        "QDTrack Faster R-CNN with quasi-dense track embeddings",
    ),
    (
        "reppoints_r50_fpn",
        "build_reppoints",
        "image_input",
        "reppoints",
        "vision/detection",
        "2019",
        "RepPoints point-set object representation head",
    ),
    (
        "solov2_r50_fpn",
        "build_solov2",
        "image_input",
        "solov2",
        "vision/instance-segmentation",
        "2020",
        "SOLOv2 dynamic mask kernels and mask feature branch",
    ),
    (
        "tood_r50_fpn",
        "build_tood",
        "image_input",
        "tood",
        "vision/detection",
        "2021",
        "TOOD task-aligned one-stage dense detector head",
    ),
    (
        "vfnet_r50_fpn",
        "build_vfnet",
        "image_input",
        "vfnet",
        "vision/detection",
        "2021",
        "VarifocalNet dense quality-aware classification head",
    ),
    (
        "yolof_r50_c5",
        "build_yolof",
        "image_input",
        "yolof",
        "vision/detection",
        "2021",
        "YOLOF single C5 level with dilated encoder, no FPN",
    ),
    (
        "mmaction2_skeleton_2s-agcn",
        "build_2s_agcn",
        "skeleton_input",
        "2s-agcn",
        "video/skeleton-action",
        "2019",
        "2s-AGCN adaptive graph convolution over joint/bone streams",
    ),
    (
        "mmaction:2s-agcn",
        "build_2s_agcn",
        "skeleton_input",
        "2s-agcn",
        "video/skeleton-action",
        "2019",
        "2s-AGCN adaptive graph convolution over skeleton streams",
    ),
    (
        "mmdet3d:3dssd",
        "build_3dssd",
        "points_input",
        "3dssd",
        "vision/3d-detection",
        "2020",
        "3DSSD point set abstraction with center and 3D box heads",
    ),
    (
        "mmdet3d_3dssd_3dssd_4xb4_kitti_3d_car",
        "build_3dssd",
        "points_input",
        "3dssd",
        "vision/3d-detection",
        "2020",
        "3DSSD point set abstraction with center and 3D box heads",
    ),
    (
        "openpcdet_3dssd",
        "build_3dssd",
        "points_input",
        "3dssd",
        "vision/3d-detection",
        "2020",
        "3DSSD point set abstraction with center and 3D box heads",
    ),
    (
        "mmocr:abinet",
        "build_abinet",
        "text_image_input",
        "abinet",
        "vision/ocr",
        "2021",
        "ABINet visual recognizer plus bidirectional language refinement",
    ),
    (
        "mmocr_textrecog_abinet",
        "build_abinet",
        "text_image_input",
        "abinet",
        "vision/ocr",
        "2021",
        "ABINet visual recognizer plus language refinement branch",
    ),
    (
        "mmaction2_detection_acrn",
        "build_acrn",
        "clip_input",
        "acrn",
        "video/action-detection",
        "2018",
        "ACRN-style actor-context video feature fusion",
    ),
    (
        "mmaction:acrn",
        "build_acrn",
        "clip_input",
        "acrn",
        "video/action-detection",
        "2018",
        "ACRN-style actor-context video feature fusion",
    ),
    (
        "mmpose_animalpose_hrnet",
        "build_hrnet_pose",
        "image_input",
        "hrnet-pose",
        "vision/pose",
        "2019",
        "HRNet parallel high/low-resolution branches fused for keypoint heatmaps",
    ),
    (
        "mmagic:animatediff",
        "build_animatediff",
        "latent_video_input",
        "animatediff",
        "generative/video-diffusion",
        "2023",
        "AnimateDiff temporal adapter over latent U-Net features",
    ),
    (
        "mmagic_animatediff_animatediff_lyriel",
        "build_animatediff",
        "latent_video_input",
        "animatediff",
        "generative/video-diffusion",
        "2023",
        "AnimateDiff temporal adapter over latent U-Net features",
    ),
    (
        "mmagic:aot_gan",
        "build_aot_gan",
        "masked_image_input",
        "aot-gan",
        "generative/inpainting",
        "2021",
        "AOT-GAN aggregated contextual transformation with dilated branches",
    ),
    (
        "mmagic_aot_gan_aot_gan_smpgan_4xb4_places_512x512",
        "build_aot_gan",
        "masked_image_input",
        "aot-gan",
        "generative/inpainting",
        "2021",
        "AOT-GAN aggregated contextual transformation with dilated branches",
    ),
    (
        "AssociativeEmbedding-HRNet-W32",
        "build_hrnet_pose",
        "image_input",
        "associative-embedding-hrnet",
        "vision/pose",
        "2017",
        "HRNet heatmaps plus associative tag embeddings for pose grouping",
    ),
    (
        "mmpose:associative_embedding",
        "build_hrnet_pose",
        "image_input",
        "associative-embedding-hrnet",
        "vision/pose",
        "2017",
        "HRNet heatmaps plus associative tag embeddings for pose grouping",
    ),
    (
        "mmpose_associative_embedding",
        "build_hrnet_pose",
        "image_input",
        "associative-embedding-hrnet",
        "vision/pose",
        "2017",
        "HRNet heatmaps plus associative tag embeddings for pose grouping",
    ),
    (
        "mmocr:aster",
        "build_aster",
        "text_image_input",
        "aster",
        "vision/ocr",
        "2018",
        "ASTER rectification plus attention/Transformer text recognition",
    ),
    (
        "mmocr_textrecog_aster",
        "build_aster",
        "text_image_input",
        "aster",
        "vision/ocr",
        "2018",
        "ASTER rectification plus sequence recognition",
    ),
    (
        "mmaction2_recognition_audio_audioonly",
        "build_audioonly",
        "audio_input",
        "audioonly-action",
        "audio/action",
        "2020",
        "Audio-only temporal convolution action recognizer",
    ),
    (
        "mmaction2_recognition_audio_resnet",
        "build_audioonly",
        "audio_input",
        "audio-resnet-action",
        "audio/action",
        "2020",
        "Audio temporal convolution action recognizer",
    ),
    (
        "mmaction:audioonly",
        "build_audioonly",
        "audio_input",
        "audioonly-action",
        "audio/action",
        "2020",
        "Audio-only temporal convolution action recognizer",
    ),
    (
        "mmagic:basicvsr",
        "build_basicvsr",
        "vsr_input",
        "basicvsr",
        "vision/video-super-resolution",
        "2021",
        "BasicVSR bidirectional recurrent propagation and pixel-shuffle SR",
    ),
    (
        "mmagic:basicvsr_pp",
        "build_basicvsr_pp",
        "vsr_input",
        "basicvsr-plus-plus",
        "vision/video-super-resolution",
        "2022",
        "BasicVSR++ second-order alignment and bidirectional propagation",
    ),
    (
        "mmagic:real_basicvsr",
        "build_real_basicvsr",
        "vsr_input",
        "real-basicvsr",
        "vision/video-super-resolution",
        "2022",
        "RealBasicVSR recurrent cleaning plus BasicVSR++ propagation/alignment restoration",
    ),
    (
        "mmagic_basicvsr",
        "build_basicvsr",
        "vsr_input",
        "basicvsr",
        "vision/video-super-resolution",
        "2021",
        "BasicVSR bidirectional recurrent propagation and pixel-shuffle SR",
    ),
    (
        "mmagic_basicvsr_basicvsr_2xb4_reds4",
        "build_basicvsr",
        "vsr_input",
        "basicvsr",
        "vision/video-super-resolution",
        "2021",
        "BasicVSR bidirectional recurrent propagation and pixel-shuffle SR",
    ),
    (
        "mmagic_basicvsr_pp_basicvsr_pp_c128n25_600k_ntire_decompress_track1",
        "build_basicvsr_pp",
        "vsr_input",
        "basicvsr-plus-plus",
        "vision/video-super-resolution",
        "2022",
        "BasicVSR++ second-order alignment and bidirectional propagation",
    ),
    (
        "mmagic_real_basicvsr",
        "build_real_basicvsr",
        "vsr_input",
        "real-basicvsr",
        "vision/video-super-resolution",
        "2022",
        "RealBasicVSR recurrent cleaning plus BasicVSR++ propagation/alignment restoration",
    ),
    (
        "mmagic_real_basicvsr_realbasicvsr_c64b20_1x30x8_8xb1_lr5e_5_150k_reds",
        "build_real_basicvsr",
        "vsr_input",
        "real-basicvsr",
        "vision/video-super-resolution",
        "2022",
        "RealBasicVSR recurrent cleaning plus BasicVSR++ propagation/alignment restoration",
    ),
    (
        "mmagic:biggan",
        "build_biggan",
        "latent_input",
        "biggan",
        "generative/gan",
        "2019",
        "BigGAN class-conditional BN and self-attention generator",
    ),
    (
        "mmagic_biggan_biggan_2xb25_500kiters_cifar10_32x32",
        "build_biggan",
        "latent_input",
        "biggan",
        "generative/gan",
        "2019",
        "BigGAN class-conditional BN and self-attention generator",
    ),
    (
        "mmagic_biggan_biggan_deep_cvt_hugging_face_rgb_imagenet1k_128x128",
        "build_biggan",
        "latent_input",
        "biggan-deep",
        "generative/gan",
        "2019",
        "BigGAN-deep hierarchical residual upsampling with class-conditional normalization and self-attention",
    ),
]


def carafe_input() -> torch.Tensor:
    """Return a CARAFE feature-map input.

    Returns
    -------
    torch.Tensor
        Feature tensor.
    """

    return torch.randn(1, 32, 16, 16)


MENAGERIE_ENTRIES = [
    (name, build_attr, input_attr, year, "E7")
    for name, build_attr, input_attr, _slug, _area, year, _notes in ENTRY_SPECS
]
