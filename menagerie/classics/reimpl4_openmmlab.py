"""Compact faithful classics for dependency-gated OpenMMLab/Paddle targets.

Paper: OpenMMLab model zoos plus cited architecture papers: Faster/Mask R-CNN,
RetinaNet, R3Det/ReDet/RepPoints for oriented detection, PANet/PSENet text
detection, PointNet++/PointRCNN/PV-RCNN/PointPillars/Part-A2/PETR for 3D,
PWC-Net/RAFT for optical flow, R(2+1)D/SlowFast/TSN/TSM/TimeSformer/MViT for
video recognition, Partial Convolution, PGGAN, Pix2Pix, and positional GANs.

The implementations are intentionally small and random-initialized, but each
keeps the distinctive data path of its original family rather than collapsing to
a generic CNN.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _conv(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """Create a convolution, batch-normalization, activation block.

    Parameters
    ----------
    in_ch:
        Input channel count.
    out_ch:
        Output channel count.
    stride:
        Convolution stride.

    Returns
    -------
    nn.Sequential
        Compact Conv-BN-ReLU block.
    """

    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=False),
    )


class TinyFPN(nn.Module):
    """Backbone plus feature pyramid used by detector and segmentor classics."""

    def __init__(self, width: int = 24) -> None:
        """Initialize a three-level convolutional pyramid.

        Parameters
        ----------
        width:
            Base feature width.
        """

        super().__init__()
        self.c2 = _conv(3, width, 2)
        self.c3 = _conv(width, width * 2, 2)
        self.c4 = _conv(width * 2, width * 4, 2)
        self.l2 = nn.Conv2d(width, width, 1)
        self.l3 = nn.Conv2d(width * 2, width, 1)
        self.l4 = nn.Conv2d(width * 4, width, 1)
        self.p2 = _conv(width, width)
        self.p3 = _conv(width, width)
        self.p4 = _conv(width, width)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Extract top-down FPN features.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            P2, P3, and P4 feature maps.
        """

        c2 = self.c2(image)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        p4 = self.l4(c4)
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        return self.p2(p2), self.p3(p3), self.p4(p4)


class RotatedFPNDetector(nn.Module):
    """FPN detector with rotated boxes, refinement, and optional point sets."""

    def __init__(self, mode: str = "retinanet", classes: int = 6, width: int = 24) -> None:
        """Initialize rotated detector heads.

        Parameters
        ----------
        mode:
            Detector family: ``retinanet``, ``r3det``, ``redet``, ``reppoints``, or ``psc``.
        classes:
            Number of classes.
        width:
            Shared feature width.
        """

        super().__init__()
        self.mode = mode
        self.fpn = TinyFPN(width)
        self.cls = nn.Conv2d(width, classes, 3, padding=1)
        self.box = nn.Conv2d(width, 5, 3, padding=1)
        self.refine = nn.Conv2d(width + 5, 5, 3, padding=1)
        self.points = nn.Conv2d(width, 18, 3, padding=1)
        self.align = nn.Conv2d(width, width, 3, padding=1)
        self.polar = nn.Conv2d(width, 8, 1)
        self.refine_cls = nn.Conv2d(width, classes, 3, padding=1)

    def _aligned_refine(self, feat: Tensor, angle: Tensor) -> Tensor:
        """Apply R3Det-style feature refinement with an angle-derived grid.

        Parameters
        ----------
        feat:
            FPN feature map.
        angle:
            Predicted orientation map.

        Returns
        -------
        Tensor
            Aligned feature map.
        """

        theta = torch.zeros(feat.shape[0], 2, 3, device=feat.device, dtype=feat.dtype)
        pooled_angle = angle.mean(dim=(2, 3)).squeeze(1)
        theta[:, 0, 0] = torch.cos(pooled_angle)
        theta[:, 0, 1] = -torch.sin(pooled_angle)
        theta[:, 1, 0] = torch.sin(pooled_angle)
        theta[:, 1, 1] = torch.cos(pooled_angle)
        grid = F.affine_grid(theta, feat.shape, align_corners=False)
        return self.align(F.grid_sample(feat, grid, align_corners=False))

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict oriented classes, boxes, and family-specific auxiliary output.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class logits, oriented box map, and auxiliary point/polar/refined output.
        """

        p2, p3, p4 = self.fpn(image)
        feat = p2 + F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
        feat = feat + F.interpolate(p4, size=p2.shape[-2:], mode="nearest")
        base_box = torch.cat(
            [torch.sigmoid(self.box(feat)[:, :4]), torch.tanh(self.box(feat)[:, 4:5])], dim=1
        )
        if self.mode == "reppoints":
            pts = self.points(feat).view(image.shape[0], 9, 2, *feat.shape[-2:])
            aux = pts.mean(dim=1)
        elif self.mode == "redet":
            steered = self.align(feat) * torch.cos(base_box[:, 4:5])
            aux = self.refine(torch.cat([steered, base_box], dim=1))
        elif self.mode == "r3det":
            aligned = self._aligned_refine(feat, base_box[:, 4:5])
            aux = self.refine(torch.cat([aligned, base_box], dim=1))
            return self.refine_cls(aligned), base_box, aux
        elif self.mode == "atss":
            scores = self.refine(torch.cat([feat, base_box], dim=1))
            level_mean = scores.mean(dim=(2, 3), keepdim=True)
            level_std = scores.std(dim=(2, 3), keepdim=True).clamp_min(1.0e-4)
            aux = torch.topk(torch.sigmoid((scores - level_mean) / level_std), k=2, dim=1).values
        elif self.mode == "psc":
            phase = self.polar(feat)
            shifts = torch.linspace(0.0, 3.14159, phase.shape[1], device=phase.device).view(
                1, -1, 1, 1
            )
            aux = torch.cat([torch.sin(phase + shifts), torch.cos(phase + shifts)], dim=1)
        else:
            aux = self.refine(torch.cat([feat, base_box], dim=1))
        return self.cls(feat), base_box, aux


class ReDetCompact(nn.Module):
    """ReDet-style rotated detector with equivariant features and rotated ROI sampling."""

    def __init__(self, classes: int = 6, width: int = 24, proposals: int = 6) -> None:
        """Initialize compact ReDet.

        Parameters
        ----------
        classes:
            Number of detection classes.
        width:
            FPN feature width.
        proposals:
            Number of compact ROI summaries.
        """

        super().__init__()
        self.proposals = proposals
        self.fpn = TinyFPN(width)
        self.reconv = nn.Conv2d(width, width, 3, padding=1)
        self.refpn = nn.Conv2d(width * 4, width, 1)
        self.rpn_obj = nn.Conv2d(width, 1, 1)
        self.rpn_box = nn.Conv2d(width, 5, 1)
        self.roi_fc = nn.Sequential(
            nn.Linear(width * 3 * 3, width), nn.ReLU(), nn.Linear(width, width)
        )
        self.cls = nn.Linear(width, classes)
        self.obox = nn.Linear(width, 5)

    def _equivariant_feature(self, feat: Tensor) -> Tensor:
        """Apply a shared convolution over four right-angle feature orientations.

        Parameters
        ----------
        feat:
            Input FPN feature map.

        Returns
        -------
        Tensor
            Rotation-pooled equivariant feature map.
        """

        oriented = []
        for k in range(4):
            rotated = torch.rot90(feat, k=k, dims=(-2, -1))
            filtered = self.reconv(rotated)
            oriented.append(torch.rot90(filtered, k=-k, dims=(-2, -1)))
        return self.refpn(torch.cat(oriented, dim=1))

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict oriented boxes through ReFPN and rotated ROI alignment.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            RPN boxes, ROI class logits, and oriented ROI boxes.
        """

        p2, p3, p4 = self.fpn(image)
        feat = self._equivariant_feature(
            p2
            + F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
            + F.interpolate(p4, size=p2.shape[-2:], mode="nearest")
        )
        rpn_scores = self.rpn_obj(feat).flatten(2)
        proposal_idx = torch.topk(rpn_scores, k=self.proposals, dim=-1).indices.squeeze(1)
        flat_feat = feat.flatten(2).transpose(1, 2)
        gather_idx = proposal_idx.unsqueeze(-1).expand(-1, -1, flat_feat.shape[-1])
        roi_tokens = torch.gather(flat_feat, 1, gather_idx)
        roi_map = roi_tokens.transpose(1, 2).reshape(image.shape[0], -1, 2, 3)
        angle = torch.tanh(self.rpn_box(feat)[:, 4:5].mean(dim=(2, 3)))
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        zeros = torch.zeros_like(cos)
        theta = torch.stack((cos, -sin, zeros, sin, cos, zeros), dim=-1).view(image.shape[0], 2, 3)
        grid = F.affine_grid(theta, roi_map.shape, align_corners=False)
        aligned = F.grid_sample(roi_map, grid, align_corners=False)
        pooled = F.adaptive_avg_pool2d(aligned, (3, 3)).flatten(1)
        roi = self.roi_fc(pooled).unsqueeze(1).expand(-1, self.proposals, -1)
        return self.rpn_box(feat), self.cls(roi), self.obox(roi)


class RegNetCompact(nn.Module):
    """RegNet-style image backbone with quantized stage widths and bottlenecks."""

    def __init__(self, classes: int = 8, widths: tuple[int, ...] = (16, 32, 48, 80)) -> None:
        """Initialize compact RegNet.

        Parameters
        ----------
        classes:
            Output class count.
        widths:
            Per-stage quantized channel widths.
        """

        super().__init__()
        self.stem = _conv(3, widths[0], stride=2)
        stages = []
        in_ch = widths[0]
        for width in widths:
            stride = 1 if width == widths[0] else 2
            stages.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, width, 1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(
                        width,
                        width,
                        3,
                        stride=stride,
                        padding=1,
                        groups=max(1, width // 16),
                        bias=False,
                    ),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(width, width, 1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=False),
                )
            )
            in_ch = width
        self.stages = nn.ModuleList(stages)
        self.head = nn.Linear(widths[-1], classes)

    def forward(self, image: Tensor) -> Tensor:
        """Classify an image through RegNet bottleneck stages.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        Tensor
            Class logits.
        """

        feat = self.stem(image)
        for stage in self.stages:
            feat = stage(feat)
        return self.head(feat.mean(dim=(2, 3)))


class TextKernelDetector(nn.Module):
    """PANet/PSENet text detector with FPN fusion and kernel embeddings."""

    def __init__(self, mode: str = "panet", width: int = 24) -> None:
        """Initialize text detector.

        Parameters
        ----------
        mode:
            ``panet`` for pixel aggregation embeddings or ``psenet`` for progressive kernels.
        width:
            Feature width.
        """

        super().__init__()
        self.mode = mode
        self.fpn = TinyFPN(width)
        self.fuse = _conv(width * 3, width)
        self.text = nn.Conv2d(width, 1, 1)
        self.kernels = nn.Conv2d(width, 6 if mode == "psenet" else 4, 1)
        self.embed = nn.Conv2d(width, 4, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict text regions, kernels, and pixel aggregation embeddings.

        Parameters
        ----------
        image:
            RGB document image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Text score, kernel maps, and embedding map.
        """

        p2, p3, p4 = self.fpn(image)
        feat = self.fuse(
            torch.cat(
                [
                    p2,
                    F.interpolate(p3, size=p2.shape[-2:], mode="nearest"),
                    F.interpolate(p4, size=p2.shape[-2:], mode="nearest"),
                ],
                dim=1,
            )
        )
        kernels = torch.sigmoid(self.kernels(feat))
        if self.mode == "psenet":
            kernels = torch.cummax(kernels, dim=1).values
        return torch.sigmoid(self.text(feat)), kernels, self.embed(feat)


class PartialConv2d(nn.Module):
    """Partial convolution layer with mask renormalization for image inpainting."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        """Initialize image and mask convolutions.

        Parameters
        ----------
        in_ch:
            Input image channels.
        out_ch:
            Output channels.
        """

        super().__init__()
        self.image = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.mask = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        nn.init.constant_(self.mask.weight, 1.0)
        for param in self.mask.parameters():
            param.requires_grad_(False)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Apply masked convolution and update the binary support mask.

        Parameters
        ----------
        x:
            Masked image/features.
        mask:
            Binary valid-pixel mask.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated features and mask.
        """

        support = self.mask(mask).clamp_min(1.0)
        out = self.image(x * mask) * (9.0 / support)
        new_mask = (support > 0).to(x.dtype)
        return F.relu(out), new_mask


class PartialConvUNet(nn.Module):
    """Partial-convolution U-Net for free-form image inpainting."""

    def __init__(self, width: int = 16) -> None:
        """Initialize encoder-decoder partial convolution network.

        Parameters
        ----------
        width:
            Base channel width.
        """

        super().__init__()
        self.e1 = PartialConv2d(3, width)
        self.e2 = PartialConv2d(width, width * 2)
        self.d1 = PartialConv2d(width * 2 + width, width)
        self.out = nn.Conv2d(width, 3, 1)

    def forward(self, image_mask: Tensor) -> Tensor:
        """Inpaint an RGB image concatenated with a single-channel mask.

        Parameters
        ----------
        image_mask:
            Tensor with RGB channels followed by a valid-pixel mask.

        Returns
        -------
        Tensor
            Reconstructed RGB image.
        """

        image = image_mask[:, :3]
        mask = image_mask[:, 3:4].expand_as(image)
        x1, m1 = self.e1(image, mask)
        x2, m2 = self.e2(F.avg_pool2d(x1, 2), F.avg_pool2d(m1, 2))
        up = F.interpolate(x2, size=x1.shape[-2:], mode="nearest")
        um = F.interpolate(m2, size=m1.shape[-2:], mode="nearest")
        x3, _ = self.d1(torch.cat([up, x1], dim=1), torch.cat([um, m1], dim=1))
        return torch.tanh(self.out(x3))


class ProgressiveGAN(nn.Module):
    """Progressive-growing GAN generator with fade-in RGB outputs."""

    def __init__(self, latent: int = 32, width: int = 32, stages: int = 3) -> None:
        """Initialize learned constant and progressive upsampling blocks.

        Parameters
        ----------
        latent:
            Latent vector size.
        width:
            Feature width.
        stages:
            Number of progressive stages.
        """

        super().__init__()
        self.fc = nn.Linear(latent, width * 4 * 4)
        self.blocks = nn.ModuleList([_conv(width, width) for _ in range(stages)])
        self.to_rgb = nn.ModuleList([nn.Conv2d(width, 3, 1) for _ in range(stages)])

    def forward(self, z: Tensor) -> Tensor:
        """Synthesize an image through progressive upsample blocks.

        Parameters
        ----------
        z:
            Latent tensor.

        Returns
        -------
        Tensor
            Generated RGB image.
        """

        x = self.fc(z).view(z.shape[0], -1, 4, 4)
        rgb = torch.tanh(self.to_rgb[0](x))
        for block, to_rgb in zip(self.blocks, self.to_rgb, strict=False):
            x = block(F.interpolate(x, scale_factor=2, mode="nearest"))
            rgb = F.interpolate(rgb, size=x.shape[-2:], mode="nearest") * 0.5
            rgb = rgb + 0.5 * torch.tanh(to_rgb(x))
        return rgb


class Pix2PixUNet(nn.Module):
    """Pix2Pix conditional U-Net generator with skip connections."""

    def __init__(self, width: int = 24) -> None:
        """Initialize encoder, bottleneck, and decoder.

        Parameters
        ----------
        width:
            Base channel width.
        """

        super().__init__()
        self.e1 = _conv(3, width, 2)
        self.e2 = _conv(width, width * 2, 2)
        self.mid = _conv(width * 2, width * 2)
        self.d1 = _conv(width * 3, width)
        self.out = nn.Conv2d(width + 3, 3, 1)

    def forward(self, image: Tensor) -> Tensor:
        """Translate an input image to an output image.

        Parameters
        ----------
        image:
            Source-domain RGB image.

        Returns
        -------
        Tensor
            Target-domain RGB image.
        """

        e1 = self.e1(image)
        e2 = self.e2(e1)
        mid = self.mid(e2)
        d1 = self.d1(torch.cat([F.interpolate(mid, size=e1.shape[-2:], mode="nearest"), e1], dim=1))
        return torch.tanh(
            self.out(
                torch.cat([F.interpolate(d1, size=image.shape[-2:], mode="nearest"), image], dim=1)
            )
        )


class PositionalGAN(nn.Module):
    """Coordinate-conditioned SinGAN/CSG multi-scale generator."""

    def __init__(self, latent: int = 32, width: int = 32) -> None:
        """Initialize coordinate-conditioned synthesis network.

        Parameters
        ----------
        latent:
            Latent vector size.
        width:
            Hidden channel width.
        """

        super().__init__()
        self.fc = nn.Linear(latent, width)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(_conv(width + 7, width), _conv(width, width), nn.Conv2d(width, 3, 1))
                for _ in range(3)
            ]
        )

    def _coord(self, z: Tensor, size: int) -> Tensor:
        """Create Fourier coordinate channels for one scale.

        Parameters
        ----------
        z:
            Latent tensor.
        size:
            Spatial size.

        Returns
        -------
        Tensor
            Coordinate channels.
        """

        grid = torch.linspace(-1.0, 1.0, size, device=z.device)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        coord = torch.stack([xx, yy, torch.sin(3.14159 * xx), torch.cos(3.14159 * yy)], dim=0)
        return coord.unsqueeze(0).expand(z.shape[0], -1, -1, -1)

    def forward(self, z: Tensor) -> Tensor:
        """Generate an image from latent and sinusoidal coordinates.

        Parameters
        ----------
        z:
            Latent tensor.

        Returns
        -------
        Tensor
            Generated RGB image.
        """

        style_seed = self.fc(z).view(z.shape[0], -1, 1, 1)
        image = torch.zeros(z.shape[0], 3, 4, 4, device=z.device, dtype=z.dtype)
        for block in self.blocks:
            if image.shape[-1] < 16:
                image = F.interpolate(image, scale_factor=2, mode="nearest")
            coord = self._coord(z, image.shape[-1])
            style = style_seed.expand(-1, -1, image.shape[-2], image.shape[-1])
            residual = block(torch.cat([style, coord, image], dim=1))
            image = torch.tanh(image + residual)
        return image


class PointBackbone(nn.Module):
    """PointNet++-style local grouping backbone for point-cloud tasks."""

    def __init__(self, in_dim: int = 3, width: int = 32) -> None:
        """Initialize point MLP and neighborhood aggregation.

        Parameters
        ----------
        in_dim:
            Input point feature size.
        width:
            Feature width.
        """

        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + 3, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU()
        )

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        """Encode points with nearest-neighbor relative coordinates.

        Parameters
        ----------
        points:
            Point tensor ``(B, N, C)`` with xyz in the first three channels.

        Returns
        -------
        tuple[Tensor, Tensor]
            Coordinates and per-point features.
        """

        xyz = points[..., :3]
        dist = torch.cdist(xyz, xyz)
        idx = torch.topk(dist, k=8, dim=-1, largest=False).indices
        batch = torch.arange(points.shape[0], device=points.device).view(-1, 1, 1).expand_as(idx)
        nbr_xyz = xyz[batch, idx]
        nbr_feat = points[batch, idx]
        rel = nbr_xyz - xyz.unsqueeze(2)
        feat = self.mlp(torch.cat([nbr_feat, rel], dim=-1)).amax(dim=2)
        return xyz, feat


class PAConvBackbone(nn.Module):
    """Point backbone with PAConv dynamic kernel-bank assembly."""

    def __init__(self, in_dim: int = 4, width: int = 32, kernels: int = 4) -> None:
        """Initialize PAConv score network and kernel bank.

        Parameters
        ----------
        in_dim:
            Input point feature size.
        width:
            Output feature width.
        kernels:
            Number of basis kernels in the weight bank.
        """

        super().__init__()
        self.kernels = kernels
        self.score = nn.Sequential(nn.Linear(3, width), nn.ReLU(), nn.Linear(width, kernels))
        self.kernel_bank = nn.Parameter(torch.randn(kernels, in_dim + 3, width) * 0.02)
        self.mix = nn.Linear(width, width)

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode local neighborhoods with dynamic PAConv kernels.

        Parameters
        ----------
        points:
            Point tensor with xyz and features.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Coordinates, point features, and per-neighbor kernel weights.
        """

        xyz = points[..., :3]
        dist = torch.cdist(xyz, xyz)
        idx = torch.topk(dist, k=8, dim=-1, largest=False).indices
        batch = torch.arange(points.shape[0], device=points.device).view(-1, 1, 1).expand_as(idx)
        nbr_xyz = xyz[batch, idx]
        nbr_feat = points[batch, idx]
        rel = nbr_xyz - xyz.unsqueeze(2)
        weights = torch.softmax(self.score(rel), dim=-1)
        assembled = torch.einsum(
            "bnmk,kcw,bnmc->bnmw", weights, self.kernel_bank, torch.cat([nbr_feat, rel], dim=-1)
        )
        return xyz, F.relu(self.mix(assembled.amax(dim=2))), weights


class Point3DModel(nn.Module):
    """Compact 3D detector/segmentor for OpenPCDet and MMDetection3D families."""

    def __init__(self, mode: str = "pointnet2", classes: int = 5, width: int = 32) -> None:
        """Initialize point-cloud model.

        Parameters
        ----------
        mode:
            3D family name.
        classes:
            Number of semantic or detection classes.
        width:
            Feature width.
        """

        super().__init__()
        self.mode = mode
        self.point = PAConvBackbone(4, width) if mode == "paconv" else PointBackbone(3, width)
        self.vfe = nn.Sequential(nn.Linear(width + 3, width), nn.ReLU(), nn.Linear(width, width))
        self.middle = nn.Sequential(
            nn.Conv3d(width, width, 3, padding=1),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=False),
        )
        self.vote = nn.Linear(width, 3)
        self.seg = nn.Linear(width, 1)
        self.cls = nn.Linear(width, classes)
        self.box = nn.Linear(width, 7)
        self.canonical_box = nn.Linear(width + 3, 7)
        self.pillar = nn.Conv2d(width, width, 3, padding=1)
        self.part = nn.Linear(width, 8)
        self.roi_grid = nn.Linear(width * 2, width)

    def _voxel_middle(self, xyz: Tensor, feat: Tensor) -> tuple[Tensor, Tensor]:
        """Run a compact voxel middle encoder.

        Parameters
        ----------
        xyz:
            Point coordinates.
        feat:
            Point features.

        Returns
        -------
        tuple[Tensor, Tensor]
            Sparse-style voxel tensor and BEV tokens.
        """

        voxel_feat = self.vfe(torch.cat([feat, xyz], dim=-1))
        grid = voxel_feat[:, :64].reshape(xyz.shape[0], 4, 4, 4, -1).permute(0, 4, 1, 2, 3)
        sparse = self.middle(grid.contiguous())
        bev = F.relu(self.pillar(sparse.mean(dim=2))).flatten(2).transpose(1, 2)
        return sparse, bev

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run a point-cloud 3D architecture.

        Parameters
        ----------
        points:
            Point cloud tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class logits, 3D boxes/segmentation, and auxiliary votes/parts.
        """

        if self.mode == "paconv":
            xyz, feat, weights = self.point(points)  # type: ignore[misc]
            return self.cls(feat), self.box(feat), weights.mean(dim=2)
        xyz, feat = self.point(points)  # type: ignore[misc]
        if self.mode == "pointpillars":
            _, bev = self._voxel_middle(xyz, feat)
            return self.cls(bev), self.box(bev), bev
        foreground = torch.sigmoid(self.seg(feat))
        votes = xyz + self.vote(feat) * foreground
        pooled = feat + feat.mean(dim=1, keepdim=True)
        if self.mode == "pointrcnn":
            canonical = votes - votes.mean(dim=1, keepdim=True)
            boxes = self.canonical_box(torch.cat([pooled, canonical], dim=-1))
            return self.cls(pooled), boxes, torch.cat([foreground, votes], dim=-1)
        if self.mode == "pvrcnn":
            sparse, bev = self._voxel_middle(xyz, feat)
            keypoints = pooled[:, ::8]
            voxel_context = sparse.mean(dim=(2, 3, 4)).unsqueeze(1).expand_as(keypoints)
            roi = self.roi_grid(torch.cat([keypoints, voxel_context], dim=-1))
            return self.cls(roi), self.box(roi), bev[:, : roi.shape[1]]
        if self.mode == "parta2":
            part_logits = self.part(pooled)
            object_feat = pooled * torch.softmax(part_logits, dim=-1).mean(dim=-1, keepdim=True)
            return self.cls(object_feat), self.box(object_feat), part_logits
        return self.cls(pooled), self.box(pooled), votes


class Monocular3D(nn.Module):
    """PGD/PETR-style monocular 3D detector with depth and query geometry."""

    def __init__(self, mode: str = "pgd", width: int = 32, queries: int = 16) -> None:
        """Initialize monocular 3D detector.

        Parameters
        ----------
        mode:
            ``pgd`` or ``petr``.
        width:
            Feature width.
        queries:
            Number of PETR object queries.
        """

        super().__init__()
        self.mode = mode
        self.fpn = TinyFPN(width)
        self.depth = nn.Conv2d(width, 8, 1)
        self.box3d = nn.Conv2d(width, 7, 1)
        self.pos3d = nn.Linear(4, width)
        self.camera = nn.Embedding(4, width)
        self.query = nn.Embedding(queries, width)
        dec_layer = nn.TransformerDecoderLayer(
            width, 4, dim_feedforward=width * 2, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=1)
        self.head = nn.Linear(width, 10)

    def _petr_memory(self, image: Tensor, feat: Tensor, depth: Tensor) -> Tensor:
        """Build PETR memory with multi-view camera and 3D position embeddings.

        Parameters
        ----------
        image:
            Input image tensor.
        feat:
            FPN feature map.
        depth:
            Per-pixel depth distribution.

        Returns
        -------
        Tensor
            Camera-aware memory tokens.
        """

        views = torch.stack([image, image.flip(-1), image.flip(-2), image.transpose(-1, -2)], dim=1)
        cam_bias = self.camera.weight.view(1, 4, 1, -1)
        h, w = feat.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=image.device),
            torch.linspace(-1.0, 1.0, w, device=image.device),
            indexing="ij",
        )
        z = depth.mul(
            torch.linspace(1.0, 8.0, depth.shape[1], device=image.device).view(1, -1, 1, 1)
        ).sum(dim=1)
        pos = self.pos3d(
            torch.stack([xx.expand_as(z), yy.expand_as(z), z, torch.ones_like(z)], dim=-1)
        )
        base = feat.flatten(2).transpose(1, 2).unsqueeze(1).expand(-1, 4, -1, -1)
        view_scale = views.mean(dim=(2, 3, 4), keepdim=False).view(image.shape[0], 4, 1, 1)
        return (base * (1.0 + view_scale) + pos.flatten(1, 2).unsqueeze(1) + cam_bias).flatten(1, 2)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict monocular 3D boxes or PETR query logits.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Depth distribution, box/query logits, and geometry features.
        """

        p2, _, _ = self.fpn(image)
        depth = torch.softmax(self.depth(p2), dim=1)
        if self.mode == "petr":
            mem = self._petr_memory(image, p2, depth)
            query = self.query.weight.unsqueeze(0).expand(image.shape[0], -1, -1)
            out = self.decoder(query, mem)
            return depth, self.head(out), out
        return depth, self.box3d(p2), depth.cumsum(dim=1)


class FlowNet(nn.Module):
    """PWC-Net/RAFT optical-flow estimator with cost volume and refinement."""

    def __init__(self, mode: str = "pwc", width: int = 24) -> None:
        """Initialize flow network.

        Parameters
        ----------
        mode:
            ``pwc`` for pyramid warping or ``raft`` for recurrent updates.
        width:
            Feature width.
        """

        super().__init__()
        self.mode = mode
        self.enc1 = _conv(3, width, 2)
        self.enc2 = _conv(width, width, 2)
        self.pwc_decode = nn.Sequential(
            _conv(width * 2 + 9 + 2, width), nn.Conv2d(width, 2, 3, padding=1)
        )
        self.pwc_refine = nn.Sequential(
            _conv(width + 9 + 2, width), nn.Conv2d(width, 2, 3, padding=1)
        )
        self.context = nn.Conv2d(width + 1, width, 3, padding=1)
        self.gru_z = nn.Conv2d(width + width + 2, width, 3, padding=1)
        self.gru_r = nn.Conv2d(width + width + 2, width, 3, padding=1)
        self.gru_q = nn.Conv2d(width + width + 2, width, 3, padding=1)
        self.delta = nn.Conv2d(width, 2, 3, padding=1)

    def _cost_volume(self, f1: Tensor, f2: Tensor) -> Tensor:
        """Build a nine-channel local correlation cost volume.

        Parameters
        ----------
        f1:
            First feature map.
        f2:
            Second feature map.

        Returns
        -------
        Tensor
            Local correlation volume.
        """

        costs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                shifted = torch.roll(f2, shifts=(dy, dx), dims=(2, 3))
                costs.append((f1 * shifted).mean(dim=1, keepdim=True))
        return torch.cat(costs, dim=1)

    def _warp(self, feat: Tensor, flow: Tensor) -> Tensor:
        """Warp a feature map by a flow field.

        Parameters
        ----------
        feat:
            Feature map to sample.
        flow:
            Flow tensor in feature-pixel units.

        Returns
        -------
        Tensor
            Warped feature map.
        """

        b, _, h, w = feat.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=feat.device),
            torch.linspace(-1.0, 1.0, w, device=feat.device),
            indexing="ij",
        )
        norm = torch.stack([flow[:, 0] / max(w - 1, 1), flow[:, 1] / max(h - 1, 1)], dim=-1) * 2.0
        grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(b, -1, -1, -1) + norm
        return F.grid_sample(feat, grid, align_corners=False)

    def _conv_gru(self, hidden: Tensor, context: Tensor, flow: Tensor) -> Tensor:
        """Run one convolutional GRU flow-update step.

        Parameters
        ----------
        hidden:
            Hidden state.
        context:
            Correlation/context features.
        flow:
            Current flow.

        Returns
        -------
        Tensor
            Updated hidden state.
        """

        inp = torch.cat([context, flow], dim=1)
        z = torch.sigmoid(self.gru_z(torch.cat([hidden, inp], dim=1)))
        r = torch.sigmoid(self.gru_r(torch.cat([hidden, inp], dim=1)))
        q = torch.tanh(self.gru_q(torch.cat([r * hidden, inp], dim=1)))
        return (1.0 - z) * hidden + z * q

    def forward(self, pair: Tensor) -> Tensor:
        """Estimate dense optical flow from an image pair.

        Parameters
        ----------
        pair:
            Tensor ``(B, 2, 3, H, W)``.

        Returns
        -------
        Tensor
            Flow field.
        """

        f1a = self.enc1(pair[:, 0])
        f2a = self.enc1(pair[:, 1])
        f1 = self.enc2(f1a)
        f2 = self.enc2(f2a)
        if self.mode == "raft":
            all_pairs = torch.bmm(f1.flatten(2).transpose(1, 2), f2.flatten(2))
            corr = all_pairs.mean(dim=-1).view(f1.shape[0], 1, *f1.shape[-2:])
            context = F.relu(self.context(torch.cat([f1, corr], dim=1)))
            hidden = torch.tanh(context)
            flow = torch.zeros(f1.shape[0], 2, *f1.shape[-2:], device=f1.device, dtype=f1.dtype)
            for _ in range(2):
                hidden = self._conv_gru(hidden, context, flow)
                flow = flow + self.delta(hidden)
        else:
            coarse_corr = self._cost_volume(f1, f2)
            zero_flow = torch.zeros(
                f1.shape[0], 2, *f1.shape[-2:], device=f1.device, dtype=f1.dtype
            )
            flow = self.pwc_decode(torch.cat([f1, f2, coarse_corr, zero_flow], dim=1))
            up_flow = (
                F.interpolate(flow, size=f1a.shape[-2:], mode="bilinear", align_corners=False) * 2.0
            )
            warped = self._warp(f2a, up_flow)
            fine_corr = self._cost_volume(f1a, warped)
            flow = up_flow + self.pwc_refine(torch.cat([f1a, fine_corr, up_flow], dim=1))
        return F.interpolate(flow, size=pair.shape[-2:], mode="bilinear", align_corners=False)


class VideoRecognizer(nn.Module):
    """Compact video recognizer for 2D, 3D, transformer, and SlowFast families."""

    def __init__(self, mode: str = "r2plus1d", classes: int = 12, width: int = 24) -> None:
        """Initialize video recognition modules.

        Parameters
        ----------
        mode:
            Video architecture family.
        classes:
            Number of action classes.
        width:
            Feature width.
        """

        super().__init__()
        self.mode = mode
        self.frame2d = nn.Sequential(_conv(3, width, stride=2), _conv(width, width, stride=2))
        self.swin_shift = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.c3d = nn.Sequential(
            nn.Conv3d(17, width, 3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=False),
            nn.Conv3d(width, width, 3, stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=False),
        )
        self.pose_heatmap = nn.Conv2d(3, 17, 3, padding=1)
        self.c2d_temporal = nn.Conv1d(width, width, 3, padding=1)
        self.tam = nn.Conv1d(width, width, 3, padding=1, groups=width)
        self.interlace = nn.Identity()
        self.i3d_b1 = nn.Conv3d(3, width // 2, 1)
        self.i3d_b3 = nn.Conv3d(3, width // 2, 3, stride=(1, 2, 2), padding=1)
        self.i3d_pool = nn.Conv3d(3, width // 2, 1)
        self.i3d_fuse = nn.Conv3d(width + width // 2, width, 1)
        self.csn_depthwise = nn.Conv3d(3, 3, 3, stride=(1, 2, 2), padding=1, groups=3)
        self.csn_pointwise = nn.Conv3d(3, width, 1)
        self.slow = nn.Conv3d(3, width, 3, stride=(2, 2, 2), padding=1)
        self.slow_res = nn.Conv3d(width, width, 3, padding=1)
        self.fast = nn.Conv3d(3, width // 2, 3, stride=(1, 2, 2), padding=1)
        self.patch = nn.Conv2d(3, width, 8, stride=8)
        self.video_patch = nn.Conv3d(3, width, (2, 8, 8), stride=(2, 8, 8))
        self.local_mhra = nn.Conv3d(width, width, 3, padding=1, groups=width)
        self.x3d_expand = nn.Conv3d(3, width * 2, 1)
        self.x3d_depthwise = nn.Conv3d(
            width * 2, width * 2, 3, stride=(1, 2, 2), padding=1, groups=width * 2
        )
        self.x3d_project = nn.Conv3d(width * 2, width, 1)
        self.proj = nn.Linear(width, width)
        self.temporal_attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.spatial_attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.pool_attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.cls = nn.Linear(width, classes)

    def _frame_tokens(self, video: Tensor, sparse: bool) -> Tensor:
        """Encode selected frames with a 2D backbone.

        Parameters
        ----------
        video:
            Video tensor ``(B, T, C, H, W)``.
        sparse:
            Whether to sample TSN-style sparse temporal segments.

        Returns
        -------
        Tensor
            Per-frame descriptors.
        """

        if sparse:
            idx = torch.linspace(
                0, video.shape[1] - 1, steps=min(3, video.shape[1]), device=video.device
            ).long()
            frames = video.index_select(1, idx)
        else:
            frames = video
        bsz, steps, channels, height, width = frames.shape
        feat = self.frame2d(frames.reshape(bsz * steps, channels, height, width))
        return feat.mean(dim=(2, 3)).view(bsz, steps, -1)

    def _temporal_shift(self, tokens: Tensor) -> Tensor:
        """Apply TSM channel-wise temporal shifting.

        Parameters
        ----------
        tokens:
            Per-frame descriptors.

        Returns
        -------
        Tensor
            Shifted descriptors.
        """

        channels = tokens.shape[-1]
        left = torch.roll(tokens[..., : channels // 4], shifts=1, dims=1)
        right = torch.roll(tokens[..., channels // 4 : channels // 2], shifts=-1, dims=1)
        return torch.cat([left, right, tokens[..., channels // 2 :]], dim=-1)

    def _temporal_interlace(self, tokens: Tensor) -> Tensor:
        """Apply TIN temporal interlacing over frame descriptors.

        Parameters
        ----------
        tokens:
            Per-frame descriptors.

        Returns
        -------
        Tensor
            Interlaced descriptors.
        """

        even = tokens[:, ::2]
        odd = tokens[:, 1::2]
        odd = F.pad(odd, (0, 0, 0, even.shape[1] - odd.shape[1]))
        return self.interlace(torch.stack([even, odd], dim=2).flatten(1, 2)[:, : tokens.shape[1]])

    def _timesformer(self, video: Tensor) -> Tensor:
        """Apply divided space-time attention over frame patch tokens.

        Parameters
        ----------
        video:
            Video tensor.

        Returns
        -------
        Tensor
            Video descriptor.
        """

        bsz, steps, channels, height, width = video.shape
        patches = self.patch(video.reshape(bsz * steps, channels, height, width))
        _, dim, ph, pw = patches.shape
        tokens = patches.flatten(2).transpose(1, 2).view(bsz, steps, ph * pw, dim)
        temporal = tokens.permute(0, 2, 1, 3).reshape(bsz * ph * pw, steps, dim)
        temporal = self.temporal_attn(temporal, temporal, temporal, need_weights=False)[0]
        tokens = temporal.view(bsz, ph * pw, steps, dim).permute(0, 2, 1, 3)
        spatial = tokens.reshape(bsz * steps, ph * pw, dim)
        spatial = self.spatial_attn(spatial, spatial, spatial, need_weights=False)[0]
        return spatial.view(bsz, steps, ph * pw, dim).mean(dim=(1, 2))

    def _swin3d(self, x: Tensor) -> Tensor:
        """Apply shifted-window-style local attention over 3D patch tokens.

        Parameters
        ----------
        x:
            Channel-first video tensor.

        Returns
        -------
        Tensor
            Video descriptor.
        """

        feat = self.video_patch(x)
        shifted = torch.roll(feat, shifts=(1, 1, 1), dims=(2, 3, 4))
        tokens = (feat + shifted).flatten(2).transpose(1, 2)
        tokens = self.attn(tokens, tokens, tokens, need_weights=False)[0]
        pooled = F.avg_pool3d(
            tokens.transpose(1, 2).reshape_as(feat), kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        return pooled.mean(dim=(2, 3, 4))

    def _mvit(self, x: Tensor) -> Tensor:
        """Apply multiscale pooling attention.

        Parameters
        ----------
        x:
            Channel-first video tensor.

        Returns
        -------
        Tensor
            Video descriptor.
        """

        feat = self.video_patch(x)
        fine = feat.flatten(2).transpose(1, 2)
        coarse_feat = F.avg_pool3d(feat, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        coarse = coarse_feat.flatten(2).transpose(1, 2)
        attended = self.pool_attn(coarse, fine, fine, need_weights=False)[0]
        return attended.mean(dim=1)

    def _uniformer(self, x: Tensor) -> Tensor:
        """Apply UniFormer local MHRA followed by global relation attention.

        Parameters
        ----------
        x:
            Channel-first video tensor.

        Returns
        -------
        Tensor
            Video descriptor.
        """

        feat = self.video_patch(x)
        feat = feat + self.local_mhra(feat)
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.attn(tokens, tokens, tokens, need_weights=False)[0]
        return tokens.mean(dim=1)

    def forward(self, video: Tensor) -> Tensor:
        """Classify a video clip.

        Parameters
        ----------
        video:
            Video tensor ``(B, T, C, H, W)``.

        Returns
        -------
        Tensor
            Action logits.
        """

        x = video.transpose(1, 2)
        if self.mode == "tsn":
            feat = self._frame_tokens(video, sparse=True).mean(dim=1)
        elif self.mode == "tsn_swin":
            tokens = self._frame_tokens(video, sparse=True)
            feat = self.swin_shift(tokens.transpose(1, 2).unsqueeze(-1)).squeeze(-1).mean(dim=-1)
        elif self.mode == "c2d":
            seq = self._frame_tokens(video, sparse=False).transpose(1, 2)
            feat = self.c2d_temporal(seq).mean(dim=-1)
        elif self.mode == "c3d":
            bsz, channels, steps, height, width = x.shape
            heat = self.pose_heatmap(
                x.transpose(1, 2).reshape(bsz * steps, channels, height, width)
            )
            heat = heat.view(bsz, steps, 17, height, width).transpose(1, 2)
            feat = self.c3d(heat).mean(dim=(2, 3, 4))
        elif self.mode == "i3d":
            branch1 = F.relu(self.i3d_b1(F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))))
            branch3 = F.relu(self.i3d_b3(x))
            branchp = F.relu(
                self.i3d_pool(F.avg_pool3d(x, kernel_size=3, stride=(1, 2, 2), padding=1))
            )
            feat = self.i3d_fuse(torch.cat([branch1, branch3, branchp], dim=1)).mean(dim=(2, 3, 4))
        elif self.mode == "ipcsn":
            feat = self.csn_pointwise(F.relu(self.csn_depthwise(x))).mean(dim=(2, 3, 4))
        elif self.mode in {"slowonly", "slowfast"}:
            slow = F.relu(self.slow(x))
            slow = F.relu(slow + self.slow_res(slow))
            if self.mode == "slowfast":
                fast = F.relu(self.fast(x))
                feat = slow.mean(dim=(2, 3, 4)) + F.pad(
                    fast.mean(dim=(2, 3, 4)), (0, slow.shape[1] - fast.shape[1])
                )
            else:
                feat = slow.mean(dim=(2, 3, 4))
        elif self.mode == "tpn":
            slow = F.relu(self.slow_res(F.relu(self.slow(x))))
            tempo1 = slow.mean(dim=(3, 4))
            tempo2 = F.avg_pool1d(tempo1, kernel_size=2, stride=1, padding=1)[
                ..., : tempo1.shape[-1]
            ]
            feat = (tempo1 + tempo2).mean(dim=-1)
        elif self.mode in {"timesformer", "videomae"}:
            feat = self._timesformer(video)
        elif self.mode == "mvit":
            feat = self._mvit(x)
        elif self.mode == "swin3d":
            feat = self._swin3d(x)
        elif self.mode == "uniformer":
            feat = self._uniformer(x)
        elif self.mode == "x3d":
            expanded = F.silu(self.x3d_expand(x))
            feat = self.x3d_project(F.silu(self.x3d_depthwise(expanded))).mean(dim=(2, 3, 4))
        else:
            tokens = self._frame_tokens(video, sparse=False)
            if self.mode == "tsm":
                tokens = tokens + self._temporal_shift(tokens)
            elif self.mode == "tin":
                tokens = tokens + self._temporal_interlace(tokens)
            elif self.mode == "tanet":
                gate = torch.sigmoid(self.tam(tokens.transpose(1, 2))).transpose(1, 2)
                tokens = tokens * gate
            feat = tokens.mean(dim=1)
        feat = self.proj(feat)
        return self.cls(feat)


class PoseLifter(nn.Module):
    """Temporal 2D-to-3D pose lifter with TCN, DSTFormer, or pose warping."""

    def __init__(self, mode: str = "tcn", joints: int = 17, width: int = 64) -> None:
        """Initialize pose lifter.

        Parameters
        ----------
        mode:
            ``tcn``, ``dstformer``, or ``posewarper``.
        joints:
            Number of body joints.
        width:
            Hidden width.
        """

        super().__init__()
        self.mode = mode
        self.inp = nn.Linear(joints * 2, width)
        self.joint_inp = nn.Linear(2, width)
        self.tcn = nn.Conv1d(width, width, 3, padding=1)
        self.spatial_attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.fuse = nn.Linear(width * 2, width)
        self.warp_offset = nn.Linear(width * 2, 2)
        self.out = nn.Linear(width, joints * 3)
        self.joint_out = nn.Linear(width, 3)

    def forward(self, pose2d: Tensor) -> Tensor:
        """Lift a sequence of 2D poses to 3D.

        Parameters
        ----------
        pose2d:
            Tensor ``(B, T, J, 2)``.

        Returns
        -------
        Tensor
            3D pose sequence.
        """

        if self.mode == "dstformer":
            joint_tokens = self.joint_inp(pose2d)
            bsz, steps, joints, width = joint_tokens.shape
            spatial = joint_tokens.reshape(bsz * steps, joints, width)
            spatial = self.spatial_attn(spatial, spatial, spatial, need_weights=False)[0]
            spatial = spatial.view(bsz, steps, joints, width)
            temporal = joint_tokens.transpose(1, 2).reshape(bsz * joints, steps, width)
            temporal = self.temporal_attn(temporal, temporal, temporal, need_weights=False)[0]
            temporal = temporal.view(bsz, joints, steps, width).transpose(1, 2)
            fused = self.fuse(torch.cat([spatial, temporal], dim=-1))
            return self.joint_out(F.relu(fused))
        if self.mode == "posewarper":
            joint_tokens = self.joint_inp(pose2d)
            support = torch.roll(joint_tokens, shifts=1, dims=1)
            offset = self.warp_offset(torch.cat([joint_tokens, support], dim=-1))
            aligned = pose2d + offset
            x = self.inp(aligned.flatten(2))
        else:
            x = self.inp(pose2d.flatten(2))
            x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.out(F.relu(x)).view(pose2d.shape[0], pose2d.shape[1], -1, 3)


class OtterVLM(nn.Module):
    """OTTER-style open-vocabulary vision-language model with gated cross-attention."""

    def __init__(self, vocab: int = 64, width: int = 32) -> None:
        """Initialize vision encoder, token encoder, and gated cross-attention.

        Parameters
        ----------
        vocab:
            Token vocabulary size.
        width:
            Shared embedding width.
        """

        super().__init__()
        self.vision = TinyFPN(width)
        self.embed = nn.Embedding(vocab, width)
        self.cross = nn.MultiheadAttention(width, 4, batch_first=True)
        self.gate = nn.Linear(width, width)
        self.head = nn.Linear(width, vocab)

    def forward(self, inputs: tuple[Tensor, Tensor]) -> Tensor:
        """Fuse image tokens into text tokens.

        Parameters
        ----------
        inputs:
            RGB image and token ids.

        Returns
        -------
        Tensor
            Token logits.
        """

        image, ids = inputs
        p2, _, _ = self.vision(image)
        visual = p2.flatten(2).transpose(1, 2)
        text = self.embed(ids)
        attended = self.cross(text, visual, visual, need_weights=False)[0]
        return self.head(text + torch.sigmoid(self.gate(text)) * attended)


class PVTv2Pose(nn.Module):
    """PVTv2-style pyramid transformer backbone with heatmap pose head."""

    def __init__(self, width: int = 32, joints: int = 17) -> None:
        """Initialize pyramid token stages and pose head.

        Parameters
        ----------
        width:
            Feature width.
        joints:
            Number of pose joints.
        """

        super().__init__()
        self.fpn = TinyFPN(width)
        self.sr2 = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.sr3 = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.sr4 = nn.Conv2d(width, width, 1)
        self.attn2 = nn.MultiheadAttention(width, 4, batch_first=True)
        self.attn3 = nn.MultiheadAttention(width, 4, batch_first=True)
        self.attn4 = nn.MultiheadAttention(width, 4, batch_first=True)
        self.head = nn.Conv2d(width, joints, 1)

    def _sra(self, feat: Tensor, reducer: nn.Conv2d, attn: nn.MultiheadAttention) -> Tensor:
        """Apply PVTv2 spatial-reduction attention.

        Parameters
        ----------
        feat:
            Feature map.
        reducer:
            Spatial-reduction convolution for keys and values.
        attn:
            Multi-head attention module.

        Returns
        -------
        Tensor
            Attention-updated feature map.
        """

        query = feat.flatten(2).transpose(1, 2)
        reduced = reducer(feat).flatten(2).transpose(1, 2)
        out = attn(query, reduced, reduced, need_weights=False)[0]
        return out.transpose(1, 2).reshape_as(feat)

    def forward(self, image: Tensor) -> Tensor:
        """Predict keypoint heatmaps from pyramid transformer features.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        Tensor
            Keypoint heatmaps.
        """

        p2, p3, p4 = self.fpn(image)
        p2 = self._sra(p2, self.sr2, self.attn2)
        p3 = self._sra(p3, self.sr3, self.attn3)
        p4 = self._sra(p4, self.sr4, self.attn4)
        feat = p2 + F.interpolate(p3, size=p2.shape[-2:], mode="nearest")
        feat = feat + F.interpolate(p4, size=p2.shape[-2:], mode="nearest")
        return self.head(feat)


def image_input() -> Tensor:
    """Return a small RGB image input.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


def image_mask_input() -> Tensor:
    """Return an RGB image plus mask tensor.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 4, 64, 64)``.
    """

    mask = (torch.rand(1, 1, 64, 64) > 0.25).float()
    return torch.cat([torch.randn(1, 3, 64, 64), mask], dim=1)


def latent_input() -> Tensor:
    """Return a latent vector.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 32)``.
    """

    return torch.randn(1, 32)


def points_input() -> Tensor:
    """Return a small xyz point cloud.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 64, 3)``.
    """

    return torch.randn(1, 64, 3)


def paconv_points_input() -> Tensor:
    """Return xyz plus PAConv score input.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 64, 4)``.
    """

    return torch.randn(1, 64, 4)


def flow_input() -> Tensor:
    """Return a small optical-flow image pair.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 2, 3, 48, 64)``.
    """

    return torch.randn(1, 2, 3, 48, 64)


def video_input() -> Tensor:
    """Return a compact video clip.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 4, 3, 48, 48)``.
    """

    return torch.randn(1, 4, 3, 48, 48)


def pose_input() -> Tensor:
    """Return a sequence of 2D keypoints.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 9, 17, 2)``.
    """

    return torch.randn(1, 9, 17, 2)


def otter_input() -> tuple[Tensor, Tensor]:
    """Return an image and short text prompt.

    Returns
    -------
    tuple[Tensor, Tensor]
        RGB image and token ids.
    """

    return torch.randn(1, 3, 64, 64), torch.randint(0, 64, (1, 8))


def _build_rotated(mode: str) -> nn.Module:
    """Build a compact rotated detector.

    Parameters
    ----------
    mode:
        Rotated detector family.

    Returns
    -------
    nn.Module
        Random-initialized detector.
    """

    return RotatedFPNDetector(mode).eval()


def _build_text(mode: str) -> nn.Module:
    """Build a compact OCR text detector.

    Parameters
    ----------
    mode:
        Text detector family.

    Returns
    -------
    nn.Module
        Random-initialized text detector.
    """

    return TextKernelDetector(mode).eval()


def _build_point(mode: str) -> nn.Module:
    """Build a compact 3D point model.

    Parameters
    ----------
    mode:
        Point-cloud architecture family.

    Returns
    -------
    nn.Module
        Random-initialized point model.
    """

    return Point3DModel(mode).eval()


def _build_mono(mode: str) -> nn.Module:
    """Build a compact monocular 3D detector.

    Parameters
    ----------
    mode:
        Monocular architecture family.

    Returns
    -------
    nn.Module
        Random-initialized detector.
    """

    return Monocular3D(mode).eval()


def _build_flow(mode: str) -> nn.Module:
    """Build a compact optical-flow model.

    Parameters
    ----------
    mode:
        Flow architecture family.

    Returns
    -------
    nn.Module
        Random-initialized flow model.
    """

    return FlowNet(mode).eval()


def _build_video(mode: str) -> nn.Module:
    """Build a compact video recognizer.

    Parameters
    ----------
    mode:
        Video recognition family.

    Returns
    -------
    nn.Module
        Random-initialized recognizer.
    """

    return VideoRecognizer(mode).eval()


def build_pconv() -> nn.Module:
    """Build Partial Convolution U-Net.

    Returns
    -------
    nn.Module
        Random-initialized inpainting model.
    """

    return PartialConvUNet().eval()


def build_pggan() -> nn.Module:
    """Build Progressive GAN generator.

    Returns
    -------
    nn.Module
        Random-initialized generator.
    """

    return ProgressiveGAN().eval()


def build_pix2pix() -> nn.Module:
    """Build Pix2Pix U-Net generator.

    Returns
    -------
    nn.Module
        Random-initialized generator.
    """

    return Pix2PixUNet().eval()


def build_posgan() -> nn.Module:
    """Build positional-encoding GAN generator.

    Returns
    -------
    nn.Module
        Random-initialized generator.
    """

    return PositionalGAN().eval()


def build_pose_tcn() -> nn.Module:
    """Build TCN pose lifter.

    Returns
    -------
    nn.Module
        Random-initialized pose lifter.
    """

    return PoseLifter("tcn").eval()


def build_posewarper() -> nn.Module:
    """Build PoseWarper temporal alignment pose model.

    Returns
    -------
    nn.Module
        Random-initialized pose lifter.
    """

    return PoseLifter("posewarper").eval()


def build_pose_dstformer() -> nn.Module:
    """Build DSTFormer pose lifter.

    Returns
    -------
    nn.Module
        Random-initialized pose lifter.
    """

    return PoseLifter("dstformer").eval()


def build_otter() -> nn.Module:
    """Build compact OTTER vision-language model.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return OtterVLM().eval()


def build_pvtv2pose() -> nn.Module:
    """Build compact PVTv2Pose model.

    Returns
    -------
    nn.Module
        Random-initialized model.
    """

    return PVTv2Pose().eval()


def _build_redet_compact() -> nn.Module:
    """Build compact ReDet with ReFPN and rotated ROI alignment.

    Returns
    -------
    nn.Module
        Random-initialized ReDet model.
    """

    return ReDetCompact().eval()


def _build_regnet_compact() -> nn.Module:
    """Build compact RegNet backbone/classifier.

    Returns
    -------
    nn.Module
        Random-initialized RegNet model.
    """

    return RegNetCompact().eval()


def _make_builder(factory: Callable[[str], nn.Module], mode: str) -> Callable[[], nn.Module]:
    """Create a named zero-argument builder.

    Parameters
    ----------
    factory:
        Factory that accepts a mode string.
    mode:
        Mode to bind.

    Returns
    -------
    Callable[[], nn.Module]
        Zero-argument builder.
    """

    def build_bound() -> nn.Module:
        """Build the bound compact model.

        Returns
        -------
        nn.Module
            Random-initialized model.
        """

        return factory(mode)

    build_bound.__name__ = f"build_{mode}"
    return build_bound


for _mode in ["retinanet", "r3det", "redet", "reppoints", "psc", "atss"]:
    globals()[f"build_{_mode}"] = _make_builder(_build_rotated, _mode)
for _mode in ["panet", "psenet"]:
    globals()[f"build_{_mode}"] = _make_builder(_build_text, _mode)
for _mode in ["pointnet2", "pointpillars", "pointrcnn", "pvrcnn", "parta2", "paconv"]:
    globals()[f"build_{_mode}"] = _make_builder(_build_point, _mode)
for _mode in ["pgd", "petr"]:
    globals()[f"build_{_mode}"] = _make_builder(_build_mono, _mode)
for _mode in ["pwc", "raft"]:
    globals()[f"build_{_mode}"] = _make_builder(_build_flow, _mode)
for _mode in [
    "r2plus1d",
    "slowonly",
    "slowfast",
    "c2d",
    "c3d",
    "i3d",
    "ipcsn",
    "tsn",
    "tsn_swin",
    "tsm",
    "tin",
    "tanet",
    "tpn",
    "timesformer",
    "videomae",
    "mvit",
    "uniformer",
    "swin3d",
    "x3d",
]:
    globals()[f"build_{_mode}"] = _make_builder(_build_video, _mode)

build_redet = _build_redet_compact
build_regnet = _build_regnet_compact


MENAGERIE_ENTRIES = [
    ("mmrotate_oriented_reppoints_r50_fpn", "build_reppoints", "image_input", "2019", "DET"),
    ("mmrotate:oriented_reppoints", "build_reppoints", "image_input", "2019", "DET"),
    ("mmrotate:psc", "build_psc", "image_input", "2022", "DET"),
    ("mmrotate:r3det", "build_r3det", "image_input", "2019", "DET"),
    ("mmrotate_r3det_r50_fpn", "build_r3det", "image_input", "2019", "DET"),
    ("mmrotate_r3det_r3det_atss_r50_fpn_1x_dota_oc", "build_r3det", "image_input", "2019", "DET"),
    (
        "mmrotate_r3det_r3det_kfiou_ln_r50_fpn_1x_dota_oc",
        "build_r3det",
        "image_input",
        "2022",
        "DET",
    ),
    (
        "mmrotate_r3det_r3det_kfiou_ln_swin_tiny_adamw_fpn_1x_dota_ms_rr_oc",
        "build_r3det",
        "image_input",
        "2022",
        "DET",
    ),
    ("mmrotate_r3det_r3det_kld_r50_fpn_1x_dota_oc", "build_r3det", "image_input", "2021", "DET"),
    ("mmrotate:redet", "build_redet", "image_input", "2021", "DET"),
    ("mmrotate_redet_re50_refpn", "build_redet", "image_input", "2021", "DET"),
    ("mmrotate_redet_redet_re50_refpn_1x_dota_le90", "build_redet", "image_input", "2021", "DET"),
    ("mmrotate:rotated_retinanet", "build_retinanet", "image_input", "2017", "DET"),
    (
        "mmrotate_rotated_retinanet_rotated_retinanet_hbb_gwd_r50_fpn_1x_dota_oc",
        "build_retinanet",
        "image_input",
        "2021",
        "DET",
    ),
    (
        "mmrotate_rotated_retinanet_rotated_retinanet_hbb_kfiou_r50_fpn_1x_dota_le135",
        "build_retinanet",
        "image_input",
        "2022",
        "DET",
    ),
    (
        "mmrotate_rotated_retinanet_rotated_atss_hbb_r50_fpn_1x_dota_oc",
        "build_atss",
        "image_input",
        "2020",
        "DET",
    ),
    ("mmocr_textdet_panet", "build_panet", "image_input", "2019", "OCR"),
    ("mmocr:panet", "build_panet", "image_input", "2019", "OCR"),
    ("mmocr_textdet_psenet", "build_psenet", "image_input", "2019", "OCR"),
    ("mmocr:psenet", "build_psenet", "image_input", "2019", "OCR"),
    (
        "mmagic_partial_conv_pconv_stage1_8xb12_places_256x256",
        "build_pconv",
        "image_mask_input",
        "2018",
        "GAN",
    ),
    ("mmagic:partial_conv", "build_pconv", "image_mask_input", "2018", "GAN"),
    ("mmagic_pggan", "build_pggan", "latent_input", "2018", "GAN"),
    ("mmagic:pggan", "build_pggan", "latent_input", "2018", "GAN"),
    (
        "mmagic_pggan_pggan_8xb4_12mimg_celeba_hq_1024x1024",
        "build_pggan",
        "latent_input",
        "2018",
        "GAN",
    ),
    ("mmagic_pix2pix", "build_pix2pix", "image_input", "2017", "GAN"),
    ("mmagic:pix2pix", "build_pix2pix", "image_input", "2017", "GAN"),
    (
        "mmagic_pix2pix_pix2pix_vanilla_unet_bn_1xb1_220kiters_aerial2maps",
        "build_pix2pix",
        "image_input",
        "2017",
        "GAN",
    ),
    ("mmagic_positional_encoding_gan", "build_posgan", "latent_input", "2021", "GAN"),
    ("mmagic:positional_encoding_in_gans", "build_posgan", "latent_input", "2021", "GAN"),
    (
        "mmagic_positional_encoding_in_gans_mspie_stylegan2_config_c_c2_8xb3_1100kiters_ffhq_256_512",
        "build_posgan",
        "latent_input",
        "2021",
        "GAN",
    ),
    (
        "mmagic_positional_encoding_in_gans_singan_csg_bohemian",
        "build_posgan",
        "latent_input",
        "2021",
        "GAN",
    ),
    (
        "mmdet3d_paconv_paconv_ssg_cuda_8xb8_cosine_200e_s3dis_seg",
        "build_paconv",
        "paconv_points_input",
        "2021",
        "3D",
    ),
    ("mmdet3d:paconv", "build_paconv", "paconv_points_input", "2021", "3D"),
    ("mmdet3d:regnet", "build_regnet", "image_input", "2017", "3D"),
    ("openpcdet_part_a2", "build_parta2", "points_input", "2020", "3D"),
    ("mmdet3d:parta2", "build_parta2", "points_input", "2020", "3D"),
    ("petr_nuscenes", "build_petr", "image_input", "2022", "3D"),
    ("mmdet3d_pgd", "build_pgd", "image_input", "2021", "3D"),
    ("mmdet3d:pgd", "build_pgd", "image_input", "2021", "3D"),
    (
        "mmdet3d_pgd_pgd_r101_caffe_fpn_head_gn_16xb2_1x_nus_mono3d",
        "build_pgd",
        "image_input",
        "2021",
        "3D",
    ),
    (
        "mmdet3d_pointnet2_pointnet2_msg_2xb16_cosine_250e_scannet_seg_xyz_only",
        "build_pointnet2",
        "points_input",
        "2017",
        "3D",
    ),
    (
        "mmdet3d_pointnet2_pointnet2_ssg_2xb16_cosine_200e_scannet_seg_xyz_only",
        "build_pointnet2",
        "points_input",
        "2017",
        "3D",
    ),
    ("mmdet3d:pointnet2", "build_pointnet2", "points_input", "2017", "3D"),
    ("openpcdet_pointpillars", "build_pointpillars", "points_input", "2019", "3D"),
    ("mmdet3d:pointpillars", "build_pointpillars", "points_input", "2019", "3D"),
    (
        "mmdet3d_point_rcnn_point_rcnn_8xb2_kitti_3d_3class",
        "build_pointrcnn",
        "points_input",
        "2019",
        "3D",
    ),
    ("openpcdet_pointrcnn", "build_pointrcnn", "points_input", "2019", "3D"),
    ("mmdet3d:point_rcnn", "build_pointrcnn", "points_input", "2019", "3D"),
    ("openpcdet_pvrcnn", "build_pvrcnn", "points_input", "2020", "3D"),
    ("mmdet3d:pv_rcnn", "build_pvrcnn", "points_input", "2020", "3D"),
    ("mmflow_pwcnet", "build_pwc", "flow_input", "2018", "FLOW"),
    (
        "mmflow_pwcnet_irrpwc_8x1_sfine_half_flyingthings3d_subset_384x768",  # pragma: allowlist secret
        "build_pwc",
        "flow_input",
        "2018",
        "FLOW",
    ),
    (
        "mmflow_pwcnet_pwcnet_8x1_sfine_flyingthings3d_subset_384x768",
        "build_pwc",
        "flow_input",
        "2018",
        "FLOW",
    ),
    ("mmflow_raft_gma_8x2_120k_flyingchairs_368x496", "build_raft", "flow_input", "2020", "FLOW"),
    (
        "mmpose_pose_lifter_image_pose_lift_tcn_8xb64_200e_h36m",
        "build_pose_tcn",
        "pose_input",
        "2019",
        "POSE",
    ),
    (
        "mmpose_pose_lifter_motionbert_dstformer_243frm_8xb32_240e_h36m_original",
        "build_pose_dstformer",
        "pose_input",
        "2022",
        "POSE",
    ),
    ("mmpose_posewarper", "build_posewarper", "pose_input", "2019", "POSE"),
    ("PVTv2Pose-B2", "build_pvtv2pose", "image_input", "2022", "POSE"),
    ("mmpretrain:otter", "build_otter", "otter_input", "2023", "VL"),
    ("mmaction_r2plus1d_r34", "build_r2plus1d", "video_input", "2018", "VID"),
    ("mmaction2_recognition_r2plus1d", "build_r2plus1d", "video_input", "2018", "VID"),
    ("mmaction:r2plus1d", "build_r2plus1d", "video_input", "2018", "VID"),
    ("mmaction:resnet", "build_c2d", "video_input", "2016", "VID"),
    ("mmaction2_skeleton_posec3d", "build_c3d", "video_input", "2022", "VID"),
    (
        "mmaction2_recognition_audio_audioonly_r50_8xb160_64x1x1_100e_kinetics400_audio_feature",
        "build_c2d",
        "video_input",
        "2020",
        "VID",
    ),
    (
        "mmaction2_recognition_audio_tsn_r18_8xb320_64x1x1_100e_kinetics400_audio_feature",
        "build_tsn",
        "video_input",
        "2016",
        "VID",
    ),
    (
        "mmaction2_recognizer2d_c2d_r101_in1k_pre_nopool_8xb32_8x8x1_100e_kinetics400_rgb",
        "build_c2d",
        "video_input",
        "2018",
        "VID",
    ),
    (
        "mmaction2_recognizer2d_tanet_imagenet_pretrained_r50_8xb6_1x1x16_50e_sthv1_rgb",
        "build_tanet",
        "video_input",
        "2020",
        "VID",
    ),
    (
        "mmaction2_recognizer2d_tin_imagenet_pretrained_r50_8xb6_1x1x8_40e_sthv1_rgb",
        "build_tin",
        "video_input",
        "2020",
        "VID",
    ),
    (
        "mmaction2_recognizer2d_tpn_tsm_imagenet_pretrained_r50_8xb8_1x1x8_150e_sthv1_rgb",
        "build_tpn",
        "video_input",
        "2020",
        "VID",
    ),
    (
        "mmaction2_recognizer2d_tsm_imagenet_pretrained_mobilenetv2_8xb16_1x1x8_100e_kinetics400_rgb",
        "build_tsm",
        "video_input",
        "2019",
        "VID",
    ),
    (
        "mmaction2_recognizer2d_tsm_imagenet_pretrained_mobileone_s4_8xb16_1x1x16_50e_kinetics400_rgb",
        "build_tsm",
        "video_input",
        "2019",
        "VID",
    ),
    (
        "mmaction2_recognizer2d_tsm_imagenet_pretrained_r101_8xb16_1x1x8_50e_sthv2_rgb",
        "build_tsm",
        "video_input",
        "2019",
        "VID",
    ),
    (
        "mmaction2_recognizer2d_tsn_imagenet_pretrained_dense161_8xb32_1x1x3_100e_kinetics400_rgb",
        "build_tsn",
        "video_input",
        "2016",
        "VID",
    ),
    (
        "mmaction2_recognizer2d_tsn_imagenet_pretrained_mobileone_s4_8xb32_1x1x8_100e_kinetics400_rgb",
        "build_tsn",
        "video_input",
        "2016",
        "VID",
    ),
    (
        "mmaction2_recognizer2d_tsn_imagenet_pretrained_rn101_32x4d_8xb32_1x1x3_100e_kinetics400_rgb",
        "build_tsn",
        "video_input",
        "2016",
        "VID",
    ),
    (
        "mmaction2_recognizer2d_tsn_imagenet_pretrained_swin_transformer_32xb8_1x1x8_50e_kinetics400_rgb",
        "build_tsn_swin",
        "video_input",
        "2021",
        "VID",
    ),
    (
        "mmaction2_recognizer3d_c2d_r50_in1k_pre_8xb32_16x4x1_100e_kinetics400_rgb",
        "build_c2d",
        "video_input",
        "2018",
        "VID",
    ),
    (
        "mmaction2_recognizer3d_c3d_sports1m_pretrained_8xb30_16x1x1_45e_ucf101_rgb",
        "build_c3d",
        "video_input",
        "2014",
        "VID",
    ),
    (
        "mmaction2_recognizer3d_i3d_imagenet_pretrained_r50_heavy_8xb8_32x2x1_100e_kinetics400_rgb",
        "build_i3d",
        "video_input",
        "2017",
        "VID",
    ),
    (
        "mmaction2_recognizer3d_ipcsn_ig65m_pretrained_r152_bnfrozen_32x2x1_58e_kinetics400_rgb",
        "build_ipcsn",
        "video_input",
        "2019",
        "VID",
    ),
    (
        "mmaction2_recognizer3d_r2plus1d_r34_8xb8_32x2x1_180e_kinetics400_rgb",
        "build_r2plus1d",
        "video_input",
        "2018",
        "VID",
    ),
    (
        "mmaction2_recognizer3d_slowonly_imagenet_pretrained_r50_16xb16_4x16x1_steplr_150e",
        "build_slowonly",
        "video_input",
        "2019",
        "VID",
    ),
    (
        "mmaction2_recognizer3d_tpn_slowonly_imagenet_pretrained_r50_8xb8_8x8x1_150e_kinetics400_rgb",
        "build_tpn",
        "video_input",
        "2020",
        "VID",
    ),
    (
        "mmaction2_recognition_slowonly_r50_8xb16_8x8x1_256e_imagenet_kinetics400_rgb",
        "build_slowonly",
        "video_input",
        "2019",
        "VID",
    ),
    (
        "mmaction2_recognizer3d_slowfast_r101_r50_32xb8_4x16x1_256e_kinetics400_rgb",  # pragma: allowlist secret
        "build_slowfast",
        "video_input",
        "2019",
        "VID",
    ),
    (
        "mmaction2_recognizer3d_timesformer_divst_8xb8_8x32x1_15e_kinetics400_rgb",
        "build_timesformer",
        "video_input",
        "2021",
        "VID",
    ),
    (
        "mmaction2_recognizer3d_mvit_base_p244_32x3x1_kinetics400_rgb",
        "build_mvit",
        "video_input",
        "2021",
        "VID",
    ),
    (
        "mmaction2_recognizer3d_uniformer_base_imagenet1k_pre_16x4x1_kinetics400_rgb",
        "build_uniformer",
        "video_input",
        "2022",
        "VID",
    ),
    (
        "mmaction2_recognizer3d_uniformerv2_base_p16_res224_clip_kinetics710_kinetics_k400",
        "build_uniformer",
        "video_input",
        "2022",
        "VID",
    ),
    (
        "mmaction2_recognizer3d_vit_base_p16_videomae_k400_pre_16x4x1_kinetics_400",
        "build_videomae",
        "video_input",
        "2021",
        "VID",
    ),
    (
        "mmaction2_recognizer3d_swin_base_p244_w877_in1k_pre_8xb8_amp_32x2x1_30e_kinetics400_rgb",
        "build_swin3d",
        "video_input",
        "2021",
        "VID",
    ),
    (
        "mmaction2_recognizer3d_x3d_m_16x5x1_facebook_kinetics400_rgb",
        "build_x3d",
        "video_input",
        "2020",
        "VID",
    ),
]
