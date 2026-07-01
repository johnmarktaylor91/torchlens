"""Menagerie batch w3a7: vector graphics, style transfer, and 3D perception architectures.

Sources checked (reference only; no cloning, no pip installs):
  - VectorFusion: https://github.com/ximinng/VectorFusion-pytorch (unofficial, CVPR 2023)
    + paper https://arxiv.org/abs/2211.11319 -- text-to-SVG by abstracting a pixel-based
    diffusion model. A population of vector paths is rasterized by a differentiable
    rasterizer (DiffVG) and the raster is scored by Score Distillation Sampling (SDS)
    against a frozen text-to-image diffusion UNet; gradients flow back through the
    rasterizer into the path control points. The traceable computational graph is
    "path params -> differentiable rasterizer -> noised raster -> UNet epsilon
    prediction", which is exactly what SDS backprops through every optimization step.
  - WCT (Universal Style Transfer via Feature Transforms): Li et al., NIPS 2017.
    Paper https://arxiv.org/abs/1705.08086, source
    https://github.com/Yijunmaverick/UniversalStyleTransfer. A VGG encoder feeds
    per-level auto-encoding decoders; at each of 5 encoder depths (relu1_1..relu5_1)
    a Whitening and Coloring Transform matches the covariance of the content feature
    to the style feature (no learned style parameters -- purely a per-instance linear
    transform via eigendecomposition-free approximation), then a matching decoder
    reconstructs to pixel space. Levels are chained coarse-to-fine (multi-level
    stylization pipeline).
  - 3D-CVF: Yoo et al., ECCV 2020. Paper https://arxiv.org/abs/2004.12636, official
    source https://github.com/rasd3/3D-CVF. Fuses camera-image features and LiDAR
    BEV features via (1) auto-calibrated projection: camera features are projected
    into the LiDAR's BEV grid via learned per-pixel calibration offsets (a
    depth-independent smooth spatial mapping), producing a camera-BEV feature map
    aligned with the LiDAR-BEV map, and (2) a gated fusion network that predicts a
    spatial attention gate per BEV cell to mix the two aligned BEV feature maps.
  - 3D-LaneNet: Garnett & Cohen et al., ICCV 2019. Paper
    https://arxiv.org/abs/1811.10203; no official repo (PyTorch community baseline
    lives inside the Gen-LaneNet codebase, already captured separately as
    GenLaneNet-3D in genlane_3d.py -- this file builds the distinct, earlier
    3D-LaneNet architecture faithfully from the paper). Dual-pathway backbone: an
    image-view pathway processes the raw image at full resolution, and at 4 stages
    a differentiable Inverse-Perspective-Mapping-style projective transform layer
    samples the image-view features onto a virtual top-view (BEV) grid and feeds a
    parallel top-view pathway. The top-view pathway predicts per-BEV-column anchors:
    lane existence, lateral (x) offset, and height (z) at fixed forward (y) samples.
  - Anchor3DLane: Huang et al., CVPR 2023. Paper https://arxiv.org/abs/2301.02371,
    official source https://github.com/tusen-ai/Anchor3DLane. BEV-free 3D lane
    detection: 3D lane anchors are defined directly in 3D space (not projected into
    a BEV grid first). Each 3D anchor is projected with the camera intrinsics/pose
    directly onto the 2D front-view (FV) feature map to sample per-anchor-point
    features (grid_sample along the anchor's 3D->2D projected polyline), which are
    pooled and regressed to 3D offsets (x, z) and an existence/visibility score --
    contrasted with prior IPM/BEV-based methods (including 3D-LaneNet above).

All models below are compact, faithfully-reimplemented-from-scratch nn.Modules with
random init and small dims for TorchLens architecture-catalog tracing (not a
trained-weights zoo).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Shared helpers
# ============================================================


def _cbr(in_ch: int, out_ch: int, k: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
    """Conv-BatchNorm-ReLU block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ============================================================
# VectorFusion -- SVG path rasterizer + SDS-scored diffusion UNet
# ============================================================


class DiffRasterizer(nn.Module):
    """Differentiable rasterizer standing in for DiffVG.

    Renders a small set of quadratic Bezier "paths" (each a control-point polygon
    with a per-path RGBA color) onto a raster canvas by soft-splatting each control
    point with a learned per-path spread (kept differentiable end-to-end, unlike
    DiffVG's analytic anti-aliased rasterizer, but faithful to "path params ->
    raster image" being the operation SDS backprops through).
    """

    def __init__(self, n_paths: int, n_ctrl: int, canvas: int) -> None:
        super().__init__()
        self.n_paths = n_paths
        self.n_ctrl = n_ctrl
        self.canvas = canvas
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, canvas), torch.linspace(0, 1, canvas), indexing="ij"
        )
        self.register_buffer("grid", torch.stack([xx, yy], dim=-1))  # (H, W, 2)
        self.log_spread = nn.Parameter(torch.zeros(n_paths))

    def forward(self, points: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        """Rasterize path control points + RGBA colors to an image.

        Parameters
        ----------
        points : torch.Tensor
            Control points, shape (B, n_paths, n_ctrl, 2) in [0, 1].
        colors : torch.Tensor
            Per-path RGBA, shape (B, n_paths, 4).

        Returns
        -------
        torch.Tensor
            Rasterized RGB canvas, shape (B, 3, canvas, canvas).
        """
        b = points.shape[0]
        spread = F.softplus(self.log_spread).view(1, self.n_paths, 1, 1, 1) + 1e-3
        grid = self.grid.view(1, 1, 1, self.canvas, self.canvas, 2)
        pts = points.view(b, self.n_paths, self.n_ctrl, 1, 1, 2)
        dist2 = ((grid - pts) ** 2).sum(-1)  # (B, n_paths, n_ctrl, H, W)
        weight = torch.exp(-dist2 / spread).sum(dim=2)  # (B, n_paths, H, W)
        alpha = torch.sigmoid(colors[..., 3]).view(b, self.n_paths, 1, 1)
        rgb = torch.sigmoid(colors[..., :3])  # (B, n_paths, 3)
        coverage = weight * alpha  # (B, n_paths, H, W)
        canvas = torch.ones(b, 3, self.canvas, self.canvas, device=points.device)
        for p in range(self.n_paths):
            cov = coverage[:, p : p + 1]
            col = rgb[:, p].view(b, 3, 1, 1)
            canvas = canvas * (1 - cov) + col * cov
        return canvas


class TinyScoreUNet(nn.Module):
    """Compact text-conditioned diffusion score UNet (epsilon predictor)."""

    def __init__(self, ch: int = 16, text_dim: int = 32) -> None:
        super().__init__()
        self.time_embed = nn.Linear(1, ch)
        self.text_proj = nn.Linear(text_dim, ch)
        self.down = _cbr(3, ch, stride=2)
        self.mid = _cbr(ch, ch)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(
        self, x_noisy: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor
    ) -> torch.Tensor:
        cond = (self.time_embed(t.view(-1, 1)) + self.text_proj(text_emb)).view(
            x_noisy.shape[0], -1, 1, 1
        )
        h = self.down(x_noisy)
        h = self.mid(h) + cond
        h = self.up(h)
        return self.out(h)


class VectorFusion(nn.Module):
    """VectorFusion: SDS-optimized vector paths scored by a frozen diffusion prior.

    One forward pass performs the operation SDS differentiates through: rasterize
    the current path parameters, add diffusion noise at timestep ``t``, and predict
    the noise with the (frozen at inference, trainable at optimization time) score
    UNet conditioned on a text embedding. Returns the predicted noise and the raster
    so an external SDS loss (``predicted_noise - true_noise``) can be attached.
    """

    def __init__(
        self, n_paths: int = 4, n_ctrl: int = 4, canvas: int = 32, text_dim: int = 32
    ) -> None:
        super().__init__()
        self.rasterizer = DiffRasterizer(n_paths, n_ctrl, canvas)
        self.unet = TinyScoreUNet(ch=16, text_dim=text_dim)

    def forward(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raster = self.rasterizer(points, colors)
        alpha_t = (1.0 - t.view(-1, 1, 1, 1) * 0.9).clamp(min=0.05)
        x_noisy = alpha_t.sqrt() * raster + (1 - alpha_t).sqrt() * noise
        eps_pred = self.unet(x_noisy, t, text_emb)
        return raster, eps_pred


def build_vectorfusion() -> nn.Module:
    """Build a small VectorFusion (SDS-over-DiffVG) module."""
    return VectorFusion(n_paths=4, n_ctrl=4, canvas=32, text_dim=32).eval()


def example_input_vectorfusion() -> tuple[torch.Tensor, ...]:
    """Path control points, colors, diffusion noise, timestep, text embedding."""
    b = 1
    points = torch.rand(b, 4, 4, 2)
    colors = torch.randn(b, 4, 4)
    noise = torch.randn(b, 3, 32, 32)
    t = torch.rand(b)
    text_emb = torch.randn(b, 32)
    return points, colors, noise, t, text_emb


# ============================================================
# WCT -- multi-level VGG encoder + Whitening-Coloring Transform + decoders
# ============================================================


class WCTEncoderStage(nn.Module):
    """One VGG-style downsampling stage (relu_X_1 tap point)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.relu(self.conv(x))
        return self.pool(feat)


class WCTDecoderStage(nn.Module):
    """Matching upsampling decoder stage that mirrors a WCTEncoderStage."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return F.relu(self.conv(x))


def whiten_color_transform(
    content_feat: torch.Tensor, style_feat: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Whitening-and-Coloring Transform: match content feature covariance to style.

    Whitens the (per-sample, per-channel) content feature to zero-correlation unit
    variance via its covariance's inverse-square-root, then "colors" it by the
    style feature's covariance square-root, matching second-order statistics
    exactly as in Li et al. 2017. Implemented per-batch-item with `torch.linalg`
    so the whole transform stays inside the traced graph (no learned parameters).
    """
    b, c, h, w = content_feat.shape
    cf = content_feat.view(b, c, h * w)
    sf = style_feat.view(b, c, style_feat.shape[2] * style_feat.shape[3])

    cf_mean = cf.mean(dim=2, keepdim=True)
    cf_centered = cf - cf_mean
    cf_cov = cf_centered @ cf_centered.transpose(1, 2) / (h * w - 1) + eps * torch.eye(
        c, device=content_feat.device
    )
    c_eigval, c_eigvec = torch.linalg.eigh(cf_cov)
    c_eigval = c_eigval.clamp(min=eps)
    whiten = c_eigvec @ torch.diag_embed(c_eigval.rsqrt()) @ c_eigvec.transpose(1, 2)
    whitened = whiten @ cf_centered

    sf_mean = sf.mean(dim=2, keepdim=True)
    sf_centered = sf - sf_mean
    sf_cov = sf_centered @ sf_centered.transpose(1, 2) / (sf.shape[2] - 1) + eps * torch.eye(
        c, device=content_feat.device
    )
    s_eigval, s_eigvec = torch.linalg.eigh(sf_cov)
    s_eigval = s_eigval.clamp(min=eps)
    color = s_eigvec @ torch.diag_embed(s_eigval.sqrt()) @ s_eigvec.transpose(1, 2)
    colored = color @ whitened + sf_mean

    return colored.view(b, c, h, w)


class WCT(nn.Module):
    """Universal Style Transfer via multi-level Whitening-and-Coloring Transform.

    Encodes content and style images through a shared 3-stage VGG-style encoder,
    applies the (learning-free) WCT at each level to blend the covariance
    statistics, decodes back to pixels with a matching decoder, and chains levels
    coarse-to-fine as in the paper's multi-level stylization pipeline.
    """

    def __init__(self, base: int = 8) -> None:
        super().__init__()
        self.enc1 = WCTEncoderStage(3, base)
        self.enc2 = WCTEncoderStage(base, base * 2)
        self.enc3 = WCTEncoderStage(base * 2, base * 4)
        self.dec3 = WCTDecoderStage(base * 4, base * 2)
        self.dec2 = WCTDecoderStage(base * 2, base)
        self.dec1 = WCTDecoderStage(base, 3)
        self.alpha = 0.6  # style-content blend weight (paper's user control knob)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        return f1, f2, f3

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        c1, c2, c3 = self._encode(content)
        _, _, s3 = self._encode(style)

        wct3 = whiten_color_transform(c3, s3)
        blended3 = self.alpha * wct3 + (1 - self.alpha) * c3
        d3 = self.dec3(blended3)
        d3 = F.interpolate(d3, size=c2.shape[-2:], mode="nearest")

        wct2 = whiten_color_transform(d3, c2)
        blended2 = self.alpha * wct2 + (1 - self.alpha) * d3
        d2 = self.dec2(blended2)
        d2 = F.interpolate(d2, size=c1.shape[-2:], mode="nearest")

        wct1 = whiten_color_transform(d2, c1)
        blended1 = self.alpha * wct1 + (1 - self.alpha) * d2
        out = self.dec1(blended1)
        return F.interpolate(out, size=content.shape[-2:], mode="nearest")


def build_wct() -> nn.Module:
    """Build a small multi-level WCT style-transfer module."""
    return WCT(base=8).eval()


def example_input_wct() -> tuple[torch.Tensor, torch.Tensor]:
    """(content image, style image), both (1, 3, 64, 64)."""
    return torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64)


# ============================================================
# 3D-CVF -- auto-calibrated camera-to-BEV projection + gated fusion
# ============================================================


class AutoCalibratedProjection(nn.Module):
    """Projects camera-view features onto the LiDAR BEV grid.

    Learns a per-BEV-cell sampling offset on top of a nominal (fixed) camera-to-BEV
    homography grid, approximating 3D-CVF's "auto-calibrated" projection that
    corrects for depth-independent misalignment between the camera and LiDAR views.
    """

    def __init__(self, in_ch: int, out_ch: int, bev_h: int, bev_w: int) -> None:
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, bev_h), torch.linspace(-1, 1, bev_w), indexing="ij"
        )
        self.register_buffer("base_grid", torch.stack([xx, yy], dim=-1))  # (bev_h, bev_w, 2)
        self.offset_net = nn.Conv2d(in_ch, 2, 3, padding=1)
        self.proj_conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, cam_feat: torch.Tensor) -> torch.Tensor:
        b = cam_feat.shape[0]
        offset = torch.tanh(self.offset_net(cam_feat))  # (B, 2, Hc, Wc)
        offset_bev = F.interpolate(
            offset, size=(self.bev_h, self.bev_w), mode="bilinear", align_corners=False
        )
        offset_bev = offset_bev.permute(0, 2, 3, 1)  # (B, bev_h, bev_w, 2)
        grid = self.base_grid.unsqueeze(0) + 0.1 * offset_bev
        cam_bev = F.grid_sample(cam_feat, grid, mode="bilinear", align_corners=False)
        return self.proj_conv(cam_bev)


class GatedFusion(nn.Module):
    """Spatial-attention gated fusion of two aligned BEV feature maps."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.gate = nn.Conv2d(ch * 2, ch, 1)

    def forward(self, cam_bev: torch.Tensor, lidar_bev: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([cam_bev, lidar_bev], dim=1)))
        return gate * cam_bev + (1 - gate) * lidar_bev


class ThreeDCVF(nn.Module):
    """3D-CVF: camera+LiDAR cross-view spatial feature fusion for 3D detection.

    Camera-image features are projected into the LiDAR's BEV grid via an
    auto-calibrated projection, then a gated fusion network mixes the aligned
    camera-BEV and LiDAR-BEV feature maps per spatial cell before a shared
    detection head regresses per-cell objectness + box offsets.
    """

    def __init__(self, ch: int = 16, bev_h: int = 24, bev_w: int = 24) -> None:
        super().__init__()
        self.cam_backbone = nn.Sequential(_cbr(3, ch, stride=2), _cbr(ch, ch, stride=2))
        self.lidar_backbone = nn.Sequential(_cbr(ch, ch), _cbr(ch, ch))
        self.projection = AutoCalibratedProjection(ch, ch, bev_h, bev_w)
        self.fusion = GatedFusion(ch)
        self.head = nn.Conv2d(ch, 7, 1)  # objectness + 6 box params per cell

    def forward(self, image: torch.Tensor, lidar_bev: torch.Tensor) -> torch.Tensor:
        cam_feat = self.cam_backbone(image)
        lidar_feat = self.lidar_backbone(lidar_bev)
        cam_bev = self.projection(cam_feat)
        fused = self.fusion(cam_bev, lidar_feat)
        return self.head(fused)


def build_3dcvf() -> nn.Module:
    """Build a small 3D-CVF camera+LiDAR fusion detector."""
    return ThreeDCVF(ch=16, bev_h=24, bev_w=24).eval()


def example_input_3dcvf() -> tuple[torch.Tensor, torch.Tensor]:
    """(camera image (1,3,96,96), LiDAR BEV pseudo-feature grid (1,16,24,24))."""
    return torch.randn(1, 3, 96, 96), torch.randn(1, 16, 24, 24)


# ============================================================
# 3D-LaneNet -- dual-pathway backbone with intra-network IPM
# ============================================================


class ProjectiveTransformLayer(nn.Module):
    """Differentiable projective sampling from image-view to a virtual top view.

    Samples the image-view feature map onto a fixed perspective-warped grid
    (standing in for the paper's homography derived from camera pose), producing
    a top-view feature map with matching channel count -- the "intra-network IPM"
    mechanism that feeds the top-view pathway at each of the 4 stages.
    """

    def __init__(self, out_h: int, out_w: int) -> None:
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        v, u = torch.meshgrid(
            torch.linspace(0.2, 1.0, out_h), torch.linspace(-1, 1, out_w), indexing="ij"
        )
        # simple perspective foreshortening: rows near the image bottom (v->1) map
        # to nearby top-view rows; rows further away compress toward the horizon.
        y_img = 1.0 - 2.0 * v
        x_img = u * v
        grid = torch.stack([x_img, y_img], dim=-1)
        self.register_buffer("grid", grid.unsqueeze(0))

    def forward(self, image_view_feat: torch.Tensor) -> torch.Tensor:
        b = image_view_feat.shape[0]
        grid = self.grid.expand(b, -1, -1, -1)
        return F.grid_sample(image_view_feat, grid, mode="bilinear", align_corners=False)


class DualPathwayStage(nn.Module):
    """One dual-pathway stage: image-view conv + projected fusion into top-view."""

    def __init__(
        self, img_ch: int, img_out: int, top_ch: int, top_out: int, top_h: int, top_w: int
    ) -> None:
        super().__init__()
        self.image_conv = _cbr(img_ch, img_out, stride=2)
        self.project = ProjectiveTransformLayer(top_h, top_w)
        self.proj_reduce = nn.Conv2d(img_out, top_ch, 1)
        self.top_conv = _cbr(top_ch * 2, top_out, stride=2)

    def forward(
        self, image_feat: torch.Tensor, top_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_feat = self.image_conv(image_feat)
        projected = self.proj_reduce(self.project(image_feat))
        projected = F.interpolate(
            projected, size=top_feat.shape[-2:], mode="bilinear", align_corners=False
        )
        top_feat = self.top_conv(torch.cat([top_feat, projected], dim=1))
        return image_feat, top_feat


class ThreeDLaneNet(nn.Module):
    """3D-LaneNet: dual-pathway backbone + anchor-based 3D lane prediction.

    An image-view pathway processes the raw image; at each stage a projective
    transform layer samples the image-view features onto a top-view grid and
    fuses them into a parallel top-view pathway. The top-view pathway's final
    feature map predicts, per BEV column anchor: lane existence, lateral (x)
    offset, and height (z) at fixed forward-direction (y) samples.
    """

    def __init__(self, base: int = 8, n_stages: int = 4, n_y_samples: int = 6) -> None:
        super().__init__()
        self.top_h, self.top_w = 32, 16
        self.top_stem = nn.Conv2d(1, base, 3, padding=1)
        stages = []
        img_ch, top_ch = 3, base
        for i in range(n_stages):
            img_out = base * (2**i)
            top_out = base * (2 ** (i + 1))
            stages.append(
                DualPathwayStage(img_ch, img_out, top_ch, top_out, self.top_h, self.top_w)
            )
            img_ch, top_ch = img_out, top_out
        self.stages = nn.ModuleList(stages)
        self.head = nn.Conv2d(top_ch, 1 + 2 * n_y_samples, 1)  # exist + (x,z) per y-sample
        self.n_y_samples = n_y_samples

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = image.shape[0]
        top_feat = self.top_stem(torch.zeros(b, 1, self.top_h, self.top_w, device=image.device))
        image_feat = image
        for stage in self.stages:
            image_feat, top_feat = stage(image_feat, top_feat)
        out = self.head(top_feat)
        exist = torch.sigmoid(out[:, :1])
        xz = out[:, 1:]
        return exist, xz


def build_3dlanenet() -> nn.Module:
    """Build a small 3D-LaneNet dual-pathway lane detector."""
    return ThreeDLaneNet(base=8, n_stages=4, n_y_samples=6).eval()


def example_input_3dlanenet() -> torch.Tensor:
    """RGB image (1, 3, 128, 128)."""
    return torch.randn(1, 3, 128, 128)


# ============================================================
# Anchor3DLane -- BEV-free 3D anchors projected onto front-view features
# ============================================================


class FVBackbone(nn.Module):
    """Compact front-view (FV) CNN feature extractor."""

    def __init__(self, base: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(_cbr(3, base, stride=2), _cbr(base, base * 2, stride=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Anchor3DProjection(nn.Module):
    """Projects 3D lane anchors directly onto the 2D front-view feature map.

    Each 3D anchor is a straight line in 3D space (fixed lateral x0, height z=0,
    forward range y in [y_min, y_max]) parameterized by a learnable per-anchor
    lateral offset field sampled along the anchor's forward extent. A simple
    pinhole projection maps each 3D anchor point to normalized FV image
    coordinates, and `grid_sample` pulls per-anchor-point features off the FV
    feature map -- avoiding any intermediate BEV grid, per the paper's BEV-free
    design.
    """

    def __init__(self, n_anchors: int, n_pts: int, feat_ch: int) -> None:
        super().__init__()
        self.n_anchors, self.n_pts = n_anchors, n_pts
        x0 = torch.linspace(-3.0, 3.0, n_anchors)
        y = torch.linspace(3.0, 30.0, n_pts)
        self.register_buffer("anchor_x0", x0)
        self.register_buffer("anchor_y", y)
        self.offset_mlp = nn.Linear(1, 1)  # tiny learnable lateral-offset predictor
        self.point_proj = nn.Linear(feat_ch, feat_ch)

    def forward(self, fv_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = fv_feat.shape
        y = self.anchor_y.view(1, self.n_pts, 1)
        offset = self.offset_mlp(y)  # (1, n_pts, 1)
        x = self.anchor_x0.view(1, self.n_anchors, 1, 1) + offset.view(1, 1, self.n_pts, 1)
        x = x.expand(b, self.n_anchors, self.n_pts, 1)
        y_b = self.anchor_y.view(1, 1, self.n_pts, 1).expand(b, self.n_anchors, self.n_pts, 1)

        z_cam = y_b.clamp(min=1e-3)
        u = (x / z_cam).clamp(-1, 1)
        v = 1.0 - 2.0 * (y_b - self.anchor_y.min()) / (
            self.anchor_y.max() - self.anchor_y.min() + 1e-6
        )
        grid = torch.cat([u, v], dim=-1).view(b, self.n_anchors * self.n_pts, 1, 2)

        sampled = F.grid_sample(
            fv_feat, grid, mode="bilinear", align_corners=False
        )  # (B,C,n_anchors*n_pts,1)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B, n_anchors*n_pts, C)
        sampled = self.point_proj(sampled)
        return sampled.view(b, self.n_anchors, self.n_pts, c)


class Anchor3DLane(nn.Module):
    """Anchor3DLane: 3D anchors projected onto front-view features, BEV-free.

    3D lane anchors are defined directly in 3D space and projected with a pinhole
    camera model onto the front-view (FV) feature map to sample per-anchor-point
    features (no intermediate BEV representation), which are pooled per anchor and
    regressed to lateral/height offsets plus an existence score.
    """

    def __init__(self, base: int = 16, n_anchors: int = 8, n_pts: int = 10) -> None:
        super().__init__()
        feat_ch = base * 2
        self.backbone = FVBackbone(base=base)
        self.anchor_proj = Anchor3DProjection(n_anchors, n_pts, feat_ch)
        self.regress = nn.Sequential(
            nn.Linear(feat_ch, feat_ch),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, 3),  # (exist_logit, dx, dz) per anchor point
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        fv_feat = self.backbone(image)
        anchor_feat = self.anchor_proj(fv_feat)  # (B, n_anchors, n_pts, C)
        return self.regress(anchor_feat)  # (B, n_anchors, n_pts, 3)


def build_anchor3dlane() -> nn.Module:
    """Build a small Anchor3DLane BEV-free 3D lane detector."""
    return Anchor3DLane(base=16, n_anchors=8, n_pts=10).eval()


def example_input_anchor3dlane() -> torch.Tensor:
    """RGB image (1, 3, 96, 128)."""
    return torch.randn(1, 3, 96, 128)


# ============================================================
# Registry
# ============================================================

MENAGERIE_ENTRIES = [
    ("VectorFusion", "build_vectorfusion", "example_input_vectorfusion", "2023", "GEN"),
    ("WCT", "build_wct", "example_input_wct", "2017", "GEN"),
    ("3D-CVF", "build_3dcvf", "example_input_3dcvf", "2020", "VIS"),
    ("3D-LaneNet", "build_3dlanenet", "example_input_3dlanenet", "2019", "VIS"),
    ("Anchor3DLane", "build_anchor3dlane", "example_input_anchor3dlane", "2023", "VIS"),
]
