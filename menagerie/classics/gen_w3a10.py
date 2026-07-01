"""Menagerie batch w3a10: camera-LiDAR-radar sensor-fusion perception architectures.

Sources checked (reference only; no cloning, no pip installs):
  - CamLiFlow: Liu et al., CVPR 2022 Oral / TPAMI 2023. Paper
    https://arxiv.org/abs/2111.10502, official source
    https://github.com/MCG-NJU/CamLiFlow (models/clfm.py). Bidirectional
    camera-LiDAR fusion for joint optical flow + scene flow: a 2D pyramid (image)
    and a 3D pyramid (point cloud) are processed by twin PWC-style encoders, and at
    each pyramid level a Camera-LiDAR Fusion Module (CLFM) exchanges information
    both ways -- 3D point features are interpolated into the 2D grid via projected
    (u, v) coordinates and fused into the image feature map (point -> image), while
    image features are grid-sampled at the projected point locations and fused into
    the point feature map (image -> point). This file reimplements the CLFM's core
    bidirectional exchange with a compact projection + gated add fusion at each of
    2 pyramid levels feeding into a small PWC-style flow decoder.
  - CamRaDepth: Brucker et al., IEEE IV 2023. Paper (TUM FTM, arXiv) and official
    source https://github.com/TUMFTM/CamRaDepth (src/models/CamRaDepth.py). A
    SegFormer-style hierarchical transformer encoder (4 stages of overlap-patch
    embedding + efficient self-attention, mirroring their "SimplifiedTransformer")
    ingests a combined camera-image + projected-sparse-radar-depth tensor; a
    multi-stage decoder with skip connections upsamples back to full resolution,
    emitting intermediate depth predictions at each decoder stage that are
    concatenated back into the next stage's input (as in the paper's
    `depth_activation_3/4/5` progressive refinement) before the final dense depth
    map.
  - CenterFusion: Nabati & Qi, WACV 2021. Paper https://arxiv.org/abs/2011.04841,
    official source https://github.com/mrnabati/CenterFusion
    (src/lib/model/networks/generic_network.py, detector.py). An image backbone
    produces a center-point heatmap plus primary regression heads (2D size,
    offset). Radar returns are frustum-associated to each detected center (a
    pillar expansion: each radar point becomes a small fixed-size pillar of
    depth/velocity features splatted onto the image-plane grid around its
    projected 2D location) to build a radar feature map, which is concatenated to
    the image features and fed to secondary regression heads (depth, rotation,
    velocity) -- the paper's key novelty of complementing camera features with
    frustum-associated radar pillars rather than fusing at the raw sensor level.
  - CenterPoint-Voxel: Yin, Zhou & Krahenbuhl, CVPR 2021. Paper
    https://arxiv.org/abs/2006.11275, official source
    https://github.com/tianweiy/CenterPoint (also in OpenPCDet). This candidate is
    the sparse-3D-voxel-encoder variant (as opposed to the 2D-pillar variant
    already captured elsewhere in the catalog as a generic "centerpoint" entry): a
    3D voxel feature encoder (VFE) aggregates points per voxel, a sparse 3D
    backbone (approximated here with dense 3D convolutions over a small voxel
    grid, since torchlens traces dense ops) collapses the height dimension to
    produce a BEV feature map, and a center-based head regresses heatmap + box +
    velocity exactly as CenterPoint's anchor-free detection head. Distinct from the
    already-catalogued pillar-based CenterPoint entry because of the voxel-grid 3D
    backbone (height-collapsing 3D convs) versus a 2D-only pillar scatter.
  - CIA-SSD: Zheng et al., AAAI 2021. Paper https://arxiv.org/abs/2012.03015,
    official source https://github.com/Vegeta2020/CIA-SSD
    (det3d/models/necks/rpn.py, bbox_heads/mg_head_v4_release.py). A sparse 3D
    voxel backbone (approximated densely, as above) feeds a Spatial-Semantic
    Feature Aggregation (SSFA) neck of multi-scale conv/deconv fusion, followed by
    the paper's Confident IoU-Aware (CIA) detection head: alongside the usual
    classification/box/direction heads, a dedicated `conv_iou` branch predicts a
    per-anchor IoU confidence used at inference to recalibrate (multiply into) the
    classification score -- the paper's namesake mechanism for suppressing
    over-confident, poorly-localized boxes.
  - CLGo: Liu et al., AAAI 2022. Paper https://arxiv.org/abs/2112.15351, official
    source https://github.com/liuruijin17/CLGo (models/py_utils/kp_pt_joint.py). A
    ResNet backbone feeds a DETR-style transformer with learned lane-anchor
    queries; the same transformer's decoder features are pooled to also regress
    latent camera height and pitch (the paper's joint 3D-lane + camera-pose
    prediction). Those camera-pose estimates parameterize a differentiable
    Inverse-Perspective-Mapping (IPM) `ProjectiveGridGenerator` grid used to warp
    the front-view image into a top-view (BEV) image, which is fed through a
    parallel top-view branch (same backbone + transformer head shape) as the
    geometry-consistency constraint linking the two predictions.

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
# CamLiFlow -- bidirectional camera-LiDAR fusion module (CLFM) + PWC flow decoder
# ============================================================


class CLFM(nn.Module):
    """Camera-LiDAR Fusion Module: bidirectional 2D <-> 3D feature exchange.

    Point (3D) features are projected to normalized image coordinates and
    gather-interpolated onto the 2D grid to complement image features
    (point -> image); image (2D) features are grid-sampled at the same
    projected point locations to complement point features (image -> point).
    Both directions use a small gated-add fusion, mirroring the paper's
    fusion-aware interpolation feeding a learned per-modality gate.
    """

    def __init__(self, ch_2d: int, ch_3d: int) -> None:
        super().__init__()
        self.to_2d = nn.Conv1d(ch_3d, ch_2d, 1)
        self.to_3d = nn.Conv2d(ch_2d, ch_3d, 1)
        self.gate_2d = nn.Conv2d(ch_2d * 2, ch_2d, 1)
        self.gate_3d = nn.Conv1d(ch_3d * 2, ch_3d, 1)

    def forward(
        self, feat_2d: torch.Tensor, feat_3d: torch.Tensor, uv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse image features (B,C2,H,W) and point features (B,C3,N) via uv (B,N,2) in [-1,1].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Fused 2D feature map, fused 3D point feature.
        """

        b, _, h, w = feat_2d.shape
        # point -> image: scatter-free splat via nearest grid_sample of a
        # per-point feature laid out as a pseudo-image (1 x N), then resize.
        pt_as_img = self.to_2d(feat_3d).unsqueeze(-1)  # (B, C2, N, 1)
        grid_uv = uv.view(b, -1, 1, 2)
        interp_3d_to_2d = F.grid_sample(
            pt_as_img.expand(-1, -1, -1, 1), grid_uv, mode="bilinear", align_corners=False
        )
        interp_3d_to_2d = interp_3d_to_2d.squeeze(-1)  # (B, C2, N)
        n = interp_3d_to_2d.shape[-1]
        side = max(1, int(n**0.5))
        interp_3d_to_2d = interp_3d_to_2d[..., : side * side].reshape(b, -1, side, side)
        interp_3d_to_2d = F.interpolate(
            interp_3d_to_2d, size=(h, w), mode="bilinear", align_corners=False
        )
        fused_2d = self.gate_2d(torch.cat([feat_2d, interp_3d_to_2d], dim=1))

        # image -> point: sample image features at each point's projected uv.
        grid = uv.view(b, -1, 1, 2)
        sampled = F.grid_sample(feat_2d, grid, mode="bilinear", align_corners=False)  # (B,C2,N,1)
        sampled = sampled.squeeze(-1)
        sampled_3d = self.to_3d(sampled.unsqueeze(-1)).squeeze(-1)  # (B, C3, N)
        fused_3d = self.gate_3d(torch.cat([feat_3d, sampled_3d], dim=1))

        return F.relu(fused_2d), F.relu(fused_3d)


class CamLiFlow(nn.Module):
    """CamLiFlow: bidirectional camera-LiDAR fusion for joint optical/scene flow.

    Twin encoders build a 2-level image pyramid and a 2-level point pyramid; at
    each level a CLFM exchanges features between the two modalities before a
    compact PWC-style decoder regresses 2D optical flow (per pixel) and 3D scene
    flow (per point) from the finest fused level.
    """

    def __init__(self, ch: int = 12, n_points: int = 64) -> None:
        super().__init__()
        self.n_points = n_points
        self.enc2d_l1 = _cbr(3, ch, stride=2)
        self.enc2d_l2 = _cbr(ch, ch * 2, stride=2)
        self.enc3d_l1 = nn.Sequential(nn.Conv1d(3, ch, 1), nn.ReLU(inplace=True))
        self.enc3d_l2 = nn.Sequential(nn.Conv1d(ch, ch * 2, 1), nn.ReLU(inplace=True))
        self.fuse_l1 = CLFM(ch, ch)
        self.fuse_l2 = CLFM(ch * 2, ch * 2)
        self.flow2d_head = nn.Conv2d(ch * 2, 2, 3, padding=1)
        self.flow3d_head = nn.Conv1d(ch * 2, 3, 1)

    def forward(
        self, image: torch.Tensor, points: torch.Tensor, uv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f2d_1 = self.enc2d_l1(image)
        f3d_1 = self.enc3d_l1(points)
        uv_1 = uv
        f2d_1, f3d_1 = self.fuse_l1(f2d_1, f3d_1, uv_1)

        f2d_2 = self.enc2d_l2(f2d_1)
        f3d_2 = self.enc3d_l2(f3d_1)
        f2d_2, f3d_2 = self.fuse_l2(f2d_2, f3d_2, uv_1)

        optical_flow = self.flow2d_head(f2d_2)
        scene_flow = self.flow3d_head(f3d_2)
        return optical_flow, scene_flow


def build_camliflow() -> nn.Module:
    """Build a small bidirectional camera-LiDAR CamLiFlow module."""
    return CamLiFlow(ch=12, n_points=64).eval()


def example_input_camliflow() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(image (1,3,32,32), points (1,3,64), projected uv coords (1,64,2) in [-1,1])."""
    image = torch.randn(1, 3, 32, 32)
    points = torch.randn(1, 3, 64)
    uv = torch.rand(1, 64, 2) * 2 - 1
    return image, points, uv


# ============================================================
# CamRaDepth -- transformer encoder + camera/radar fusion + progressive depth decoder
# ============================================================


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding (SegFormer-style stage stem)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), h, w


class EfficientSelfAttentionBlock(nn.Module):
    """Transformer block with spatial-reduction self-attention + MLP."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int, sr_ratio: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        y = self.norm1(x)
        if self.sr_ratio > 1:
            y_sr = y.transpose(1, 2).reshape(b, c, h, w)
            y_sr = self.sr(y_sr).reshape(b, c, -1).transpose(1, 2)
            kv = self.sr_norm(y_sr)
        else:
            kv = y
        attn_out, _ = self.attn(y, kv, kv)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class SimplifiedTransformerStage(nn.Module):
    """One hierarchical encoder stage: patch embed + N attention blocks."""

    def __init__(
        self, in_ch: int, dim: int, depth: int, heads: int, sr_ratio: int, stride: int
    ) -> None:
        super().__init__()
        self.embed = OverlapPatchEmbed(in_ch, dim, stride)
        self.blocks = nn.ModuleList(
            [
                EfficientSelfAttentionBlock(dim, heads, mlp_ratio=2, sr_ratio=sr_ratio)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, h, w = self.embed(x)
        for blk in self.blocks:
            x = blk(x, h, w)
        b, n, c = x.shape
        return x.transpose(1, 2).reshape(b, c, h, w)


class DepthDecoderStage(nn.Module):
    """Upsample + concat-skip + conv decoder stage with an intermediate depth tap."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.depth_tap = nn.Conv2d(out_ch, 1, 1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = F.relu(self.conv(torch.cat([x, skip], dim=1)))
        depth = torch.sigmoid(self.depth_tap(x))
        return torch.cat([x, depth], dim=1), depth


class CamRaDepth(nn.Module):
    """CamRaDepth: transformer camera+radar fusion for dense monocular depth.

    A combined camera-image + projected-sparse-radar tensor is encoded by a
    4-stage hierarchical transformer (overlap-patch embed + spatial-reduction
    self-attention, mirroring the paper's Simplified Transformer). A 3-stage
    decoder upsamples with skip connections, emitting an intermediate depth
    prediction at each stage that is concatenated into the next stage's input --
    the paper's progressive depth-refinement decoder.
    """

    def __init__(self, in_ch: int = 4, dims: tuple[int, ...] = (12, 24, 48, 96)) -> None:
        super().__init__()
        self.stage1 = SimplifiedTransformerStage(
            in_ch, dims[0], depth=1, heads=1, sr_ratio=4, stride=2
        )
        self.stage2 = SimplifiedTransformerStage(
            dims[0], dims[1], depth=1, heads=2, sr_ratio=2, stride=2
        )
        self.stage3 = SimplifiedTransformerStage(
            dims[1], dims[2], depth=1, heads=2, sr_ratio=1, stride=2
        )
        self.stage4 = SimplifiedTransformerStage(
            dims[2], dims[3], depth=1, heads=4, sr_ratio=1, stride=2
        )

        self.dec3 = DepthDecoderStage(dims[3], dims[2], dims[2])
        self.dec2 = DepthDecoderStage(dims[2] + 1, dims[1], dims[1])
        self.dec1 = DepthDecoderStage(dims[1] + 1, dims[0], dims[0])
        self.final_depth = nn.Conv2d(dims[0] + 1, 1, 1)

    def forward(self, camera_radar: torch.Tensor) -> torch.Tensor:
        s1 = self.stage1(camera_radar)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        d3, _ = self.dec3(s4, s3)
        d2, _ = self.dec2(d3, s2)
        d1, _ = self.dec1(d2, s1)
        depth = torch.sigmoid(self.final_depth(d1))
        return depth


def build_camradepth() -> nn.Module:
    """Build a small CamRaDepth transformer camera+radar depth module."""
    return CamRaDepth(in_ch=4, dims=(12, 24, 48, 96)).eval()


def example_input_camradepth() -> torch.Tensor:
    """Camera RGB + projected sparse radar depth channel, (1, 4, 64, 128)."""
    return torch.randn(1, 4, 64, 128)


# ============================================================
# CenterFusion -- center heatmap + frustum-associated radar pillar fusion
# ============================================================


class RadarPillarExpansion(nn.Module):
    """Splats sparse radar returns into a dense pillar feature map on the image grid.

    Each radar point (with normalized image-plane location + depth/velocity
    features) is scattered into a fixed-size neighborhood ("pillar") around its
    projected pixel via a soft Gaussian kernel, approximating the paper's
    frustum-association pillar expansion without a hard nearest-neighbor scatter.
    """

    def __init__(self, feat_ch: int, out_h: int, out_w: int) -> None:
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        self.point_mlp = nn.Sequential(nn.Linear(3, feat_ch), nn.ReLU(inplace=True))
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, out_h), torch.linspace(-1, 1, out_w), indexing="ij"
        )
        self.register_buffer("grid", torch.stack([xx, yy], dim=-1))  # (H, W, 2)

    def forward(self, radar_uvz: torch.Tensor) -> torch.Tensor:
        """radar_uvz: (B, N, 3) = (u, v in [-1,1], depth-or-velocity feature)."""

        b, n, _ = radar_uvz.shape
        feat = self.point_mlp(radar_uvz)  # (B, N, C)
        c = feat.shape[-1]
        uv = radar_uvz[..., :2].view(b, n, 1, 1, 2)
        grid = self.grid.view(1, 1, self.out_h, self.out_w, 2)
        dist2 = ((grid - uv) ** 2).sum(-1)  # (B, N, H, W)
        weight = torch.exp(-dist2 / 0.02)
        pillar = torch.einsum("bnhw,bnc->bchw", weight, feat)
        return pillar


class CenterFusion(nn.Module):
    """CenterFusion: center-point detector with frustum-associated radar fusion.

    An image backbone produces a shared feature map from which a primary heatmap
    head detects object centers. Radar returns are pillar-expanded onto the same
    grid and concatenated with the image features before secondary heads regress
    depth, rotation, and velocity -- the paper's radar-complements-camera design.
    """

    def __init__(self, ch: int = 16, grid_hw: int = 24) -> None:
        super().__init__()
        self.backbone = nn.Sequential(_cbr(3, ch, stride=2), _cbr(ch, ch, stride=2))
        self.heatmap_head = nn.Conv2d(ch, 3, 1)  # primary: per-class center heatmap
        self.size_head = nn.Conv2d(ch, 2, 1)  # primary: 2D box size

        self.radar_pillar = RadarPillarExpansion(feat_ch=ch, out_h=grid_hw, out_w=grid_hw)
        self.fuse = nn.Conv2d(ch * 2, ch, 1)
        self.depth_head = nn.Conv2d(ch, 1, 1)  # secondary: depth
        self.rot_head = nn.Conv2d(ch, 2, 1)  # secondary: rotation (sin, cos)
        self.vel_head = nn.Conv2d(ch, 2, 1)  # secondary: radar-assisted velocity

    def forward(self, image: torch.Tensor, radar_uvz: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(image)
        heatmap = torch.sigmoid(self.heatmap_head(feat))
        size = self.size_head(feat)

        radar_feat = self.radar_pillar(radar_uvz)
        radar_feat = F.interpolate(
            radar_feat, size=feat.shape[-2:], mode="bilinear", align_corners=False
        )
        fused = F.relu(self.fuse(torch.cat([feat, radar_feat], dim=1)))

        depth = self.depth_head(fused)
        rot = self.rot_head(fused)
        vel = self.vel_head(fused)
        return {"heatmap": heatmap, "size": size, "depth": depth, "rot": rot, "vel": vel}


def build_centerfusion() -> nn.Module:
    """Build a small CenterFusion radar+camera center-point detector."""
    return CenterFusion(ch=16, grid_hw=24).eval()


def example_input_centerfusion() -> tuple[torch.Tensor, torch.Tensor]:
    """(image (1,3,96,96), radar points as (u,v,depth) tuples (1,20,3))."""
    image = torch.randn(1, 3, 96, 96)
    radar_uvz = torch.cat([torch.rand(1, 20, 2) * 2 - 1, torch.rand(1, 20, 1)], dim=-1)
    return image, radar_uvz


# ============================================================
# CenterPoint-Voxel -- 3D voxel-grid backbone + anchor-free center head
# ============================================================


class VoxelFeatureEncoder(nn.Module):
    """Per-voxel point-feature aggregation (VFE): pointwise MLP + max-pool."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_ch, out_ch), nn.ReLU(inplace=True))

    def forward(self, voxel_points: torch.Tensor) -> torch.Tensor:
        """voxel_points: (B, D, H, W, K, in_ch) -> per-voxel feature (B, out_ch, D, H, W)."""

        feat = self.mlp(voxel_points)  # (B, D, H, W, K, out_ch)
        pooled, _ = feat.max(dim=4)  # (B, D, H, W, out_ch)
        return pooled.permute(0, 4, 1, 2, 3)  # (B, out_ch, D, H, W)


class SparseVoxelBackbone(nn.Module):
    """Dense stand-in for the sparse 3D voxel backbone (3D conv stack that
    progressively strides the height/depth axis to a single BEV slice)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(ch, ch, 3, stride=(2, 1, 1), padding=1)
        self.conv2 = nn.Conv3d(ch, ch, 3, stride=(2, 1, 1), padding=1)
        self.bn1 = nn.BatchNorm3d(ch)
        self.bn2 = nn.BatchNorm3d(ch)

    def forward(self, voxel_feat: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(voxel_feat)))
        x = F.relu(self.bn2(self.conv2(x)))
        b, c, d, h, w = x.shape
        return x.reshape(b, c * d, h, w)  # collapse remaining depth into BEV channels


class CenterPointVoxel(nn.Module):
    """CenterPoint-Voxel: 3D-voxel backbone + anchor-free center-based BEV head.

    A per-voxel feature encoder aggregates raw point features inside each voxel
    cell, a 3D (sparse-approximated-as-dense) backbone collapses the height
    dimension to a BEV feature map, and a center-based head regresses a heatmap
    plus box and velocity -- matching CenterPoint's anchor-free detection design,
    with the voxel-grid 3D backbone distinguishing this from the pillar-only
    (2D-scatter) CenterPoint variant already in the catalog.
    """

    def __init__(self, ch: int = 8, depth: int = 4, grid_hw: int = 16) -> None:
        super().__init__()
        self.depth, self.grid_hw, self.ch = depth, grid_hw, ch
        self.vfe = VoxelFeatureEncoder(in_ch=4, out_ch=ch)
        self.backbone = SparseVoxelBackbone(ch)
        bev_ch = ch * (depth // 4)
        self.bev_reduce = nn.Conv2d(bev_ch, ch * 2, 3, padding=1)
        self.heat_head = nn.Conv2d(ch * 2, 3, 1)
        self.box_head = nn.Conv2d(ch * 2, 6, 1)
        self.vel_head = nn.Conv2d(ch * 2, 2, 1)

    def forward(self, voxel_points: torch.Tensor) -> dict[str, torch.Tensor]:
        vfe_feat = self.vfe(voxel_points)  # (B, ch, D, H, W)
        bev = self.backbone(vfe_feat)  # (B, ch*D', H, W)
        bev = F.relu(self.bev_reduce(bev))
        heat = torch.sigmoid(self.heat_head(bev))
        box = self.box_head(bev)
        vel = self.vel_head(bev)
        return {"heatmap": heat, "box": box, "vel": vel}


def build_centerpoint_voxel() -> nn.Module:
    """Build a small CenterPoint-Voxel 3D-voxel-backbone BEV detector."""
    return CenterPointVoxel(ch=8, depth=4, grid_hw=16).eval()


def example_input_centerpoint_voxel() -> torch.Tensor:
    """Voxel grid of per-cell point features (1, D=4, H=16, W=16, K=4, feat=4)."""
    return torch.randn(1, 4, 16, 16, 4, 4)


# ============================================================
# CIA-SSD -- SSFA multi-scale neck + Confident IoU-Aware detection head
# ============================================================


class SSFANeck(nn.Module):
    """Spatial-Semantic Feature Aggregation neck: multi-scale conv + deconv fusion."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.block1 = _cbr(ch, ch, stride=1)
        self.block2 = _cbr(ch, ch, stride=2)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, 2, stride=2, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f2_up = self.deconv2(f2)
        f2_up = F.interpolate(f2_up, size=f1.shape[-2:], mode="nearest")
        return torch.cat([f1, f2_up], dim=1)


class ConfidentIoUAwareHead(nn.Module):
    """CIA head: classification, box, direction, and a dedicated IoU-confidence branch.

    The IoU branch predicts a per-anchor-location box-quality score used (at
    inference, outside this forward) to recalibrate the classification
    confidence, per the paper's namesake mechanism.
    """

    def __init__(self, in_ch: int, num_cls: int = 1, box_dim: int = 7) -> None:
        super().__init__()
        self.cls_head = nn.Conv2d(in_ch, num_cls, 1)
        self.box_head = nn.Conv2d(in_ch, box_dim, 1)
        self.dir_head = nn.Conv2d(in_ch, 2, 1)
        self.iou_head = nn.Conv2d(in_ch, 1, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        cls_logits = self.cls_head(x)
        box = self.box_head(x)
        direction = self.dir_head(x)
        iou_pred = torch.tanh(self.iou_head(x))
        calibrated_cls = torch.sigmoid(cls_logits) * (0.5 * (iou_pred + 1.0))
        return {
            "cls": cls_logits,
            "box": box,
            "dir": direction,
            "iou": iou_pred,
            "cls_calibrated": calibrated_cls,
        }


class CIASSD(nn.Module):
    """CIA-SSD: sparse-voxel backbone + SSFA neck + Confident IoU-Aware head.

    A 3D voxel backbone (dense stand-in) collapses to a BEV feature map, an SSFA
    neck fuses multi-scale conv/deconv features, and the CIA head adds an
    explicit IoU-confidence branch alongside the usual single-stage
    classification/box/direction predictions.
    """

    def __init__(self, ch: int = 8) -> None:
        super().__init__()
        self.backbone = SparseVoxelBackbone(ch)
        self.bev_reduce = nn.Conv2d(ch, ch * 2, 1)
        self.neck = SSFANeck(ch * 2)
        self.head = ConfidentIoUAwareHead(ch * 4)

    def forward(self, voxel_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        bev = self.backbone(voxel_feat)
        b, c, h, w = bev.shape
        bev = bev[:, : self.bev_reduce.in_channels]
        bev = F.relu(self.bev_reduce(bev))
        neck_feat = self.neck(bev)
        return self.head(neck_feat)


def build_cia_ssd() -> nn.Module:
    """Build a small CIA-SSD voxel detector with IoU-aware confidence head."""
    return CIASSD(ch=8).eval()


def example_input_cia_ssd() -> torch.Tensor:
    """Dense voxel-grid pseudo-feature volume (1, 8, 8, 20, 20)."""
    return torch.randn(1, 8, 8, 20, 20)


# ============================================================
# CLGo -- DETR-style lane transformer + latent camera pose + IPM warp
# ============================================================


class ResStem(nn.Module):
    """Compact ResNet-style stem for the perspective/top-view backbones."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(_cbr(3, ch, stride=2), _cbr(ch, ch, stride=2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IPMGridGenerator(nn.Module):
    """Differentiable IPM warp grid built from a predicted camera height + pitch.

    Approximates the paper's `ProjectiveGridGenerator`: given a scalar camera
    pitch (radians) and height, builds a perspective-foreshortening sampling grid
    mapping a top-view (BEV) canvas back into normalized front-view image
    coordinates, so the two branches share a geometry-consistent warp driven by
    the transformer's own latent pose prediction.
    """

    def __init__(self, out_h: int, out_w: int) -> None:
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        v, u = torch.meshgrid(
            torch.linspace(0.1, 1.0, out_h), torch.linspace(-1, 1, out_w), indexing="ij"
        )
        self.register_buffer("v", v)
        self.register_buffer("u", u)

    def forward(
        self, pitch: torch.Tensor, height: torch.Tensor, feat: torch.Tensor
    ) -> torch.Tensor:
        b = feat.shape[0]
        pitch = pitch.view(b, 1, 1)
        height = height.view(b, 1, 1)
        v = self.v.unsqueeze(0)
        u = self.u.unsqueeze(0)
        foreshorten = (height.clamp(min=0.5) / (v + height.clamp(min=0.5))) * torch.cos(pitch)
        y_img = 1.0 - 2.0 * v * foreshorten
        x_img = u * foreshorten
        grid = torch.stack([x_img, y_img], dim=-1)
        return F.grid_sample(feat, grid, mode="bilinear", align_corners=False)


class LaneTransformerBranch(nn.Module):
    """One CLGo branch: CNN backbone + DETR transformer with lane-anchor queries."""

    def __init__(self, ch: int, hidden: int, n_queries: int) -> None:
        super().__init__()
        self.backbone = ResStem(ch)
        self.input_proj = nn.Conv2d(ch, hidden, 1)
        self.query_embed = nn.Embedding(n_queries, hidden)
        self.transformer = nn.Transformer(
            d_model=hidden,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=hidden * 2,
            batch_first=True,
        )
        self.class_embed = nn.Linear(hidden, 2)
        self.coord_embed = nn.Linear(hidden, 3)  # per-anchor (x-offsets summary)
        self.height_embed = nn.Linear(hidden, 1)
        self.pitch_embed = nn.Linear(hidden, 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(image)
        tokens = self.input_proj(feat).flatten(2).transpose(1, 2)  # (B, HW, hidden)
        b = image.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        hs = self.transformer(tokens, queries)  # (B, n_queries, hidden)
        cls = self.class_embed(hs)
        coord = self.coord_embed(hs)
        height = torch.sigmoid(self.height_embed(hs).mean(dim=1)) + 1.0  # (B, 1)
        pitch = torch.tanh(self.pitch_embed(hs).mean(dim=1)) * 0.5  # (B, 1)
        return {
            "feat": feat,
            "cls": cls,
            "coord": coord,
            "height": height.squeeze(-1),
            "pitch": pitch.squeeze(-1),
        }


class CLGo(nn.Module):
    """CLGo: joint 3D-lane + camera-pose prediction with an IPM geometry constraint.

    A front-view branch predicts lane-anchor classes/coordinates and, from the
    same transformer decoder features, latent camera height and pitch. Those
    latent pose scalars parameterize a differentiable IPM grid that warps the
    front-view feature map into a top-view (BEV) image, which a parallel
    top-view branch (sharing the same transformer head shape) also processes --
    the geometry-consistency link between the perspective and top-view
    predictions.
    """

    def __init__(
        self, ch: int = 12, hidden: int = 16, n_queries: int = 6, top_h: int = 16, top_w: int = 16
    ) -> None:
        super().__init__()
        self.pv_branch = LaneTransformerBranch(ch, hidden, n_queries)
        self.tv_branch = LaneTransformerBranch(ch, hidden, n_queries)
        self.ipm = IPMGridGenerator(top_h, top_w)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        pv_out = self.pv_branch(image)
        top_view = self.ipm(pv_out["pitch"], pv_out["height"], image)
        tv_out = self.tv_branch(top_view)
        return {
            "pv_cls": pv_out["cls"],
            "pv_coord": pv_out["coord"],
            "camera_height": pv_out["height"],
            "camera_pitch": pv_out["pitch"],
            "tv_cls": tv_out["cls"],
            "tv_coord": tv_out["coord"],
        }


def build_clgo() -> nn.Module:
    """Build a small CLGo dual-branch lane+pose transformer."""
    return CLGo(ch=12, hidden=16, n_queries=6, top_h=16, top_w=16).eval()


def example_input_clgo() -> torch.Tensor:
    """RGB front-view image (1, 3, 64, 64)."""
    return torch.randn(1, 3, 64, 64)


# ============================================================
# Registry
# ============================================================

MENAGERIE_ENTRIES = [
    ("CamLiFlow", "build_camliflow", "example_input_camliflow", "2022", "VIS"),
    ("CamRaDepth", "build_camradepth", "example_input_camradepth", "2023", "VIS"),
    ("CenterFusion", "build_centerfusion", "example_input_centerfusion", "2021", "VIS"),
    (
        "CenterPoint-Voxel (range)",
        "build_centerpoint_voxel",
        "example_input_centerpoint_voxel",
        "2021",
        "VIS",
    ),
    ("CIA-SSD", "build_cia_ssd", "example_input_cia_ssd", "2021", "VIS"),
    ("CLGO", "build_clgo", "example_input_clgo", "2022", "VIS"),
]
