"""Menagerie batch w3a15: camera/LiDAR 3D perception and HD-map architectures for
autonomous driving.

Sources checked (reference only; no cloning, no pip installs):
  - LaRa (Latents and Rays): Bartoccioni et al., CoRL 2022. Paper
    https://arxiv.org/abs/2206.13294, official source https://github.com/valeoai/LaRa.
    Multi-camera BEV semantic segmentation via a two-stage cross-attention "latent
    bottleneck": a small fixed-size set of learned latent vectors cross-attends into
    the (flattened, position+ray-embedded) multi-camera image feature tokens to compress
    all cameras into a compact representation (avoiding the O(cameras x pixels x BEV
    cells) cost of a direct camera-to-BEV cross-attention); the latents are refined with
    self-attention blocks; finally BEV grid-cell queries (each with a ray/position
    embedding) cross-attend back into the refined latents to produce the per-cell BEV
    segmentation feature. This is the distinctive "latents + rays" bottleneck mechanism.
  - LaserNet: Meyer et al. (Uber ATG), CVPR 2019. Paper
    https://openaccess.thecvf.com/content_CVPR_2019/html/Meyer_LaserNet_An_Efficient_Probabilistic_3D_Object_Detector_for_Autonomous_Driving_CVPR_2019_paper.html,
    community PyTorch reference https://github.com/kareemalsawah/Modified_LaserNet_Pytorch
    (no official public code). Operates directly on the LiDAR sensor's native dense
    range-view image (2D grid indexed by azimuth x elevation/laser-ring, channels =
    range/height/intensity) via a fully convolutional encoder-decoder (no
    point-cloud-to-voxel or point-cloud-to-BEV conversion, unlike LiDAR detectors that
    project to a bird's-eye view first). Each range-image pixel predicts a per-point
    *probability distribution* over a 3D box (mean + log-variance of box parameters)
    rather than a single deterministic box -- the "probabilistic" detector -- and
    per-point predictions are later fused (mean-shift-style clustering of nearby modes,
    approximated here with a mean-shift clustering head that predicts a per-point 2D
    offset toward its instance's centroid).
  - Lift-Splat-Shoot (LSS): Philion & Fidler, ECCV 2020. Paper
    https://arxiv.org/abs/2008.05711, official source
    https://github.com/nv-tlabs/lift-splat-shoot. "Lift": for each camera-feature
    pixel, predicts a categorical distribution over discrete depth bins and takes the
    outer product with the pixel's context feature to "lift" the 2D pixel into a
    frustum of (depth x channel) points. "Splat": every frustum point is unprojected
    with the camera's known intrinsics/extrinsics into a fixed 3D "pillar" (BEV cell)
    index, then pooled into the unified BEV grid via the paper's *cumulative-sum
    pooling trick* -- points are sorted by pillar index, a running cumsum of features
    is taken along the sorted order, and per-pillar sums are recovered by differencing
    the cumsum at each pillar's last-point boundary (equivalent to, but much faster
    than, scatter-add via padding). "Shoot": a BEV CNN consumes the pooled grid (here
    reduced to a lightweight segmentation/detection head, since planning-trajectory
    "shooting" itself has no extra trainable parameters). This candidate reimplements
    the LSS view transformer standalone with the genuine cumsum-pooling mechanism
    (distinct from the scatter_add-based `LiftSplatShoot` helper already used inside
    the BEVDet classic in gen_w3a8.py).
  - M3D-RPN: Brazil & Liu, ICCV 2019 Oral. Paper https://arxiv.org/abs/1907.06038,
    official source https://github.com/garrickbrazil/M3D-RPN. Monocular 3D region
    proposal network. The distinctive mechanism is "depth-aware convolution": the
    feature map is split into `n_depth_bins` row-wise horizontal bands (a proxy for
    depth, since farther objects project to higher rows in a perspective image), and
    each band uses its OWN (non-shared) convolution kernel -- i.e. a single conv layer
    is really `n_depth_bins` independent kernels, each applied only to its row-band,
    then the bands are re-stacked. This location-specific (row-conditioned) feature
    extraction is fused via weighted sum with a normal spatially-invariant (shared
    kernel) global convolution branch, and both feed a shared-anchor RPN head that
    regresses both 2D and 3D box parameters per anchor.
  - MapVR: Zhang et al., NeurIPS 2023. Paper https://arxiv.org/abs/2306.10502,
    official source https://github.com/ZhangGongjie/MapVR. Online HD-map
    vectorization. A DETR-style query decoder predicts, per map instance query, a
    fixed-length *ordered sequence of 2D points* (polyline/polygon vertices) describing
    a map element (lane divider, road boundary, pedestrian crossing). The distinctive
    mechanism is a differentiable-rasterization "render loss": the predicted vectorized
    point sequence is converted back into a soft raster mask by placing a small
    Gaussian "stamp" at each predicted vertex and taking a smooth (soft-max-like)
    union of the per-vertex stamps and per-edge (consecutive-vertex-pair) line stamps,
    yielding a differentiable rasterized map that can be directly supervised against
    the ground-truth rasterized HD map -- geometry-aware fine-grained supervision that
    a vertex-coordinate-only loss cannot provide. This candidate implements the
    predict-then-rasterize forward pass (the render loss itself is a training-time
    loss term, so the traced forward returns the vectorized points AND the rasterized
    mask that the loss would consume).
  - MonoCon: Liu et al. (Learning Auxiliary Monocular Contexts Helps Monocular 3D
    Object Detection), AAAI 2022. Paper https://arxiv.org/abs/2112.04628, official
    source https://github.com/Xianpeng919/MonoCon. Monocular 3D detection with a
    single backbone and a *center-point* detection head design (CenterNet-style: a
    heatmap over object centers plus per-center regressed 3D box parameters), trained
    with extra "auxiliary monocular context" heads -- regressing the 8 projected 2D
    corner-keypoint offsets (from the 3D box's 8 corners projected into the image) and
    the 2D bounding-box quantization-error offset -- that are discarded at inference
    but sharpen the shared backbone features during training. This candidate keeps
    all heads (main + auxiliary) present in the forward pass, matching the paper's
    described architecture (the paper "discards" the auxiliary heads only as an
    inference-time speed optimization, not an architectural change).
"""

from __future__ import annotations

import torch
from torch import nn


def _cbr(in_ch: int, out_ch: int, k: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
    """Conv-BatchNorm-ReLU block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ============================================================
# LaRa -- Latents and Rays cross-attention bottleneck for multi-camera BEV segmentation
# ============================================================


class _CrossAttnBlock(nn.Module):
    """Single-head-pooled cross-attention: queries attend into a (fixed) key/value set."""

    def __init__(self, dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.norm_ff = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        k = self.norm_kv(kv)
        attn_out, _ = self.attn(q, k, k)
        query = query + attn_out
        query = query + self.ff(self.norm_ff(query))
        return query


class LaRa(nn.Module):
    """LaRa: multi-camera images -> compact latent bottleneck -> BEV segmentation.

    Stage 1 (encoder cross-attention): a small fixed set of learned latent vectors
    cross-attends into position+ray-embedded multi-camera image tokens, compressing
    all cameras into the latent set. Stage 2: latents are refined with self-attention.
    Stage 3 (decoder cross-attention): BEV grid-cell query embeddings cross-attend
    back into the refined latents to read out a per-cell segmentation feature.
    """

    def __init__(
        self,
        dim: int = 32,
        n_latents: int = 24,
        bev_size: int = 10,
        patch: int = 8,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.bev_size = bev_size
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.ray_embed = nn.Linear(4, dim)  # per-pixel-token ray direction (dx, dy, dz, cam_id)
        self.latents = nn.Parameter(torch.randn(1, n_latents, dim) * 0.02)
        self.encoder_xattn = _CrossAttnBlock(dim)
        self.self_attn = nn.TransformerEncoderLayer(
            dim, nhead=4, dim_feedforward=dim * 2, batch_first=True
        )
        self.bev_query_embed = nn.Linear(2, dim)  # per-BEV-cell (x, y) position embedding
        self.decoder_xattn = _CrossAttnBlock(dim)
        self.seg_head = nn.Linear(dim, 1)

    def forward(self, images: torch.Tensor, ray_dirs: torch.Tensor) -> torch.Tensor:
        n_cams, _, h, w = images.shape
        feat = self.patch_embed(images)  # (n_cams, dim, h', w')
        _, dim, hp, wp = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (n_cams, h'*w', dim)
        tokens = tokens + self.ray_embed(ray_dirs)
        tokens = tokens.reshape(1, n_cams * hp * wp, dim)  # merge all cameras into one KV set

        latents = self.latents.expand(1, -1, -1)
        latents = self.encoder_xattn(latents, tokens)
        latents = self.self_attn(latents)

        g = self.bev_size
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, g), torch.linspace(-1, 1, g), indexing="ij")
        bev_pos = torch.stack([xs, ys], dim=-1).reshape(1, g * g, 2)
        bev_query = self.bev_query_embed(bev_pos)
        bev_feat = self.decoder_xattn(bev_query, latents)
        seg = self.seg_head(bev_feat).reshape(1, g, g)
        return seg


def build_lara() -> nn.Module:
    """Build a small LaRa multi-camera BEV segmentation model."""
    return LaRa(dim=32, n_latents=24, bev_size=10, patch=8).eval()


def example_input_lara() -> tuple[torch.Tensor, torch.Tensor]:
    """(multi-camera images (n_cams=4, 3, 32, 32), per-pixel-token ray directions)."""
    n_cams, patch, h, w = 4, 8, 32, 32
    images = torch.randn(n_cams, 3, h, w)
    hp, wp = h // patch, w // patch
    ray_dirs = torch.randn(n_cams, hp * wp, 4)
    return images, ray_dirs


# ============================================================
# LaserNet -- dense range-image probabilistic 3D detector
# ============================================================


class LaserNet(nn.Module):
    """LaserNet: fully-convolutional dense range-image probabilistic 3D detector.

    Operates directly on the LiDAR sensor's native range-view image (no point-cloud
    voxelization or BEV projection). A fully convolutional encoder-decoder predicts,
    per range-image pixel, a probability distribution over 3D box parameters (mean +
    log-variance, the "probabilistic" detector) plus a 2D mean-shift offset used to
    cluster nearby per-point predictions into a single instance at inference.
    """

    def __init__(self, in_ch: int = 3, feat_ch: int = 16, n_box_params: int = 7) -> None:
        super().__init__()
        self.n_box_params = n_box_params
        self.encoder = nn.Sequential(
            _cbr(in_ch, feat_ch, stride=1),
            _cbr(feat_ch, feat_ch, stride=2),
            _cbr(feat_ch, feat_ch * 2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feat_ch * 2, feat_ch, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feat_ch, feat_ch, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.class_head = nn.Conv2d(feat_ch, 1, 1)
        self.box_mean_head = nn.Conv2d(feat_ch, n_box_params, 1)
        self.box_logvar_head = nn.Conv2d(feat_ch, n_box_params, 1)  # probabilistic distribution
        self.meanshift_offset_head = nn.Conv2d(feat_ch, 2, 1)  # instance-clustering offset

    def forward(
        self, range_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.encoder(range_image)
        feat = self.decoder(feat)
        cls = torch.sigmoid(self.class_head(feat))
        box_mean = self.box_mean_head(feat)
        box_logvar = self.box_logvar_head(feat)
        offset = self.meanshift_offset_head(feat)
        return cls, box_mean, box_logvar, offset


def build_lasernet() -> nn.Module:
    """Build a small LaserNet range-image probabilistic detector."""
    return LaserNet(in_ch=3, feat_ch=16, n_box_params=7).eval()


def example_input_lasernet() -> torch.Tensor:
    """Range image (1, 3, H=32 elevation rings, W=64 azimuth bins) of (range, height, intensity)."""
    return torch.randn(1, 3, 32, 64)


# ============================================================
# Lift-Splat-Shoot -- cumsum-pooled frustum lift + BEV splat (standalone view transformer)
# ============================================================


class LiftSplatShootCumsum(nn.Module):
    """LSS view transformer using the paper's genuine cumulative-sum pooling trick.

    "Lift": each image-feature pixel gets a categorical distribution over depth bins,
    outer-producted with its context feature into a frustum of (depth x channel)
    points. "Splat": every frustum point is unprojected to a fixed pillar (BEV cell)
    index; points are sorted by pillar index, a running cumsum of features is taken
    along the sort order, and per-pillar sums are recovered by differencing the
    cumsum at each pillar's final-point boundary -- the cumsum-pooling trick, which
    avoids scatter/padding.
    """

    def __init__(self, in_ch: int, out_ch: int, n_depth: int, bev_size: int) -> None:
        super().__init__()
        self.n_depth = n_depth
        self.bev_size = bev_size
        self.depth_head = nn.Conv2d(in_ch, n_depth, 1)
        self.feat_proj = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, img_feat: torch.Tensor, pillar_idx: torch.Tensor) -> torch.Tensor:
        b, _, h, w = img_feat.shape
        depth_prob = torch.softmax(self.depth_head(img_feat), dim=1)  # (B, D, H, W)
        feat = self.feat_proj(img_feat)  # (B, C, H, W)
        c = feat.shape[1]

        # lift: outer product of per-pixel context feature and depth distribution
        lifted = depth_prob.unsqueeze(2) * feat.unsqueeze(1)  # (B, D, C, H, W)
        lifted = lifted.permute(0, 1, 3, 4, 2).reshape(b, self.n_depth * h * w, c)
        pillar_flat = pillar_idx.reshape(b, -1)  # (B, D*H*W) fixed per-point pillar index

        g = self.bev_size
        n_pts = self.n_depth * h * w
        outs = []
        for bi in range(b):
            order = torch.argsort(pillar_flat[bi])  # sort points by pillar id
            sorted_pillars = pillar_flat[bi][order]
            sorted_feat = lifted[bi][order]
            cs = torch.cumsum(sorted_feat, dim=0)  # running cumsum along sort order
            # per-pillar sum = cumsum at each pillar's LAST point (boundary where the
            # next point belongs to a different pillar, or it's the final point)
            is_boundary = torch.ones(n_pts, dtype=torch.bool)
            is_boundary[:-1] = sorted_pillars[1:] != sorted_pillars[:-1]
            boundary_cs = cs[is_boundary]  # (n_boundaries, C) cumsum values at boundaries
            boundary_pillars = sorted_pillars[is_boundary]  # (n_boundaries,)
            # per-pillar sum = boundary cumsum minus the PRECEDING boundary's cumsum
            prev_cs = torch.zeros_like(boundary_cs)
            prev_cs[1:] = boundary_cs[:-1]
            pillar_sums = boundary_cs - prev_cs  # (n_boundaries, C)
            bev = feat.new_zeros(g * g, c)
            bev = bev.index_copy(0, boundary_pillars.clamp(0, g * g - 1), pillar_sums)
            outs.append(bev)
        bev_batch = torch.stack(outs, dim=0)  # (B, g*g, C)
        return bev_batch.transpose(1, 2).view(b, c, g, g)


class LiftSplatShoot(nn.Module):
    """Standalone LSS model: image backbone -> cumsum-pooled view transformer -> BEV head."""

    def __init__(self, ch: int = 12, n_depth: int = 6, bev_size: int = 12) -> None:
        super().__init__()
        self.img_backbone = nn.Sequential(_cbr(3, ch, stride=2), _cbr(ch, ch, stride=2))
        self.view_transform = LiftSplatShootCumsum(ch, ch, n_depth=n_depth, bev_size=bev_size)
        self.bev_backbone = nn.Sequential(_cbr(ch, ch), _cbr(ch, ch))
        self.seg_head = nn.Conv2d(ch, 1, 1)

    def forward(self, images: torch.Tensor, pillar_idx: torch.Tensor) -> torch.Tensor:
        img_feat = self.img_backbone(images)
        bev = self.view_transform(img_feat, pillar_idx)
        bev = bev.sum(dim=0, keepdim=True)  # merge per-camera BEV grids into one ego-frame grid
        bev = self.bev_backbone(bev)
        return self.seg_head(bev)


def build_lift_splat_shoot() -> nn.Module:
    """Build a small standalone Lift-Splat-Shoot BEV segmentation model."""
    return LiftSplatShoot(ch=12, n_depth=6, bev_size=12).eval()


def example_input_lift_splat_shoot() -> tuple[torch.Tensor, torch.Tensor]:
    """(multi-view camera images (n_cams=3, 3, 32, 32), fixed per-frustum-point pillar index)."""
    n_cams, ch, n_depth, h, w, bev = 3, 12, 6, 8, 8, 12
    images = torch.randn(n_cams, 3, 32, 32)
    pillar_idx = torch.randint(0, bev * bev, (n_cams, n_depth, h, w))
    return images, pillar_idx


# ============================================================
# M3D-RPN -- depth-aware convolution monocular 3D region proposal network
# ============================================================


class DepthAwareConv2d(nn.Module):
    """Depth-aware convolution: independent (non-shared) kernels per row-band.

    The input feature map is split into `n_depth_bins` horizontal row-bands (a proxy
    for depth, since farther objects project higher in a perspective camera image);
    each band is convolved with its OWN kernel, then bands are re-stacked -- giving
    location-(row-)specific features that a normal spatially-invariant conv cannot.
    """

    def __init__(self, in_ch: int, out_ch: int, n_depth_bins: int, k: int = 3) -> None:
        super().__init__()
        self.n_depth_bins = n_depth_bins
        self.band_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, out_ch, k, padding=k // 2) for _ in range(n_depth_bins)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, _ = x.shape
        band_h = h // self.n_depth_bins
        bands = []
        for i, conv in enumerate(self.band_convs):
            start = i * band_h
            end = h if i == self.n_depth_bins - 1 else start + band_h
            band_out = conv(x[:, :, start:end, :])
            bands.append(band_out)
        return torch.cat(bands, dim=2)


class M3DRPN(nn.Module):
    """M3D-RPN: fused depth-aware + global convolution feeding a shared-anchor 2D+3D RPN head."""

    def __init__(
        self, in_ch: int = 3, feat_ch: int = 16, n_depth_bins: int = 4, n_anchors: int = 6
    ) -> None:
        super().__init__()
        self.stem = _cbr(in_ch, feat_ch, stride=2)
        self.global_branch = _cbr(feat_ch, feat_ch, stride=1)
        self.depth_aware_branch = DepthAwareConv2d(feat_ch, feat_ch, n_depth_bins=n_depth_bins)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        # per-anchor: 4 (2D box) + 7 (3D box: depth, w/h/l, rotation, 2D projected center) = 11
        self.rpn_head = nn.Conv2d(feat_ch, n_anchors * 11, 1)
        self.n_anchors = n_anchors

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.stem(image)
        global_feat = self.global_branch(feat)
        depth_feat = torch.relu(self.depth_aware_branch(feat))
        w = torch.sigmoid(self.fusion_weight)
        fused = w * global_feat + (1.0 - w) * depth_feat
        return self.rpn_head(fused)


def build_m3d_rpn() -> nn.Module:
    """Build a small M3D-RPN monocular 3D region proposal network."""
    return M3DRPN(in_ch=3, feat_ch=16, n_depth_bins=4, n_anchors=6).eval()


def example_input_m3d_rpn() -> torch.Tensor:
    """Single monocular RGB image (1, 3, 32, 32)."""
    return torch.randn(1, 3, 32, 32)


# ============================================================
# MapVR -- vectorized map queries + differentiable-rasterization render loss
# ============================================================


class MapVR(nn.Module):
    """MapVR: DETR-style vectorized map-element queries + differentiable rasterization.

    A CNN backbone extracts BEV features; a set of learned instance queries
    cross-attends into the flattened BEV feature map (DETR-style) and each query
    regresses a fixed-length ordered sequence of 2D vertex coordinates describing one
    map element (lane divider, boundary, crossing). The distinctive "render loss"
    mechanism differentiably rasterizes the predicted vertex sequence back into a
    soft raster mask (Gaussian stamps at vertices, smooth-union over instances) so a
    geometry-aware raster-space loss can supervise the vectorized prediction directly.
    """

    def __init__(
        self,
        dim: int = 24,
        n_queries: int = 8,
        n_points: int = 10,
        raster_size: int = 24,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_points = n_points
        self.raster_size = raster_size
        self.bev_backbone = nn.Sequential(_cbr(dim, dim, stride=1), _cbr(dim, dim, stride=1))
        self.query_embed = nn.Parameter(torch.randn(1, n_queries, dim) * 0.02)
        self.decoder = nn.TransformerDecoderLayer(
            dim, nhead=4, dim_feedforward=dim * 2, batch_first=True
        )
        self.point_head = nn.Linear(dim, n_points * 2)  # ordered (x, y) vertex sequence per query
        self.register_buffer(
            "raster_grid",
            torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, raster_size),
                    torch.linspace(-1, 1, raster_size),
                    indexing="ij",
                ),
                dim=-1,
            ),  # (raster_size, raster_size, 2)
        )

    def forward(self, bev_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = bev_feat.shape
        feat = self.bev_backbone(bev_feat)
        memory = feat.flatten(2).transpose(1, 2)  # (B, h*w, C)
        queries = self.query_embed.expand(b, -1, -1)
        decoded = self.decoder(queries, memory)
        points = torch.tanh(self.point_head(decoded)).reshape(b, -1, self.n_points, 2)

        # differentiable rasterization: soft Gaussian stamp per vertex, smooth-union
        # (log-sum-exp) over all vertices of all instances -> a soft raster mask
        grid = self.raster_grid.reshape(1, 1, self.raster_size * self.raster_size, 2)
        pts_flat = points.reshape(b, -1, 1, 2)  # (B, n_queries*n_points, 1, 2)
        sq_dist = ((pts_flat - grid) ** 2).sum(-1)  # (B, n_queries*n_points, raster*raster)
        stamps = -sq_dist / (2 * 0.05**2)
        raster = torch.logsumexp(stamps, dim=1).reshape(b, self.raster_size, self.raster_size)
        raster = torch.sigmoid(raster)
        return points, raster


def build_mapvr() -> nn.Module:
    """Build a small MapVR vectorized map + render-loss rasterization model."""
    return MapVR(dim=24, n_queries=8, n_points=10, raster_size=24).eval()


def example_input_mapvr() -> torch.Tensor:
    """BEV feature map (1, 24, 12, 12) as would come from an upstream BEV encoder."""
    return torch.randn(1, 24, 12, 12)


# ============================================================
# MonoCon -- center-point monocular 3D detection + auxiliary 2D-corner-keypoint heads
# ============================================================


class MonoCon(nn.Module):
    """MonoCon: CenterNet-style monocular 3D detector with auxiliary monocular-context heads.

    A shared backbone feeds (a) the MAIN heads: an object-center heatmap and
    per-center regressed 3D box parameters (depth, dimensions, orientation), and
    (b) AUXILIARY "monocular context" heads: the 8 projected-3D-corner 2D-keypoint
    offsets and the 2D-bbox quantization offset, supervised only during training as
    extra geometric context that sharpens the shared backbone (discarded at
    inference in the paper as a speed optimization, not an architectural change --
    kept here for a faithful forward pass).
    """

    def __init__(self, in_ch: int = 3, feat_ch: int = 16, n_classes: int = 3) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            _cbr(in_ch, feat_ch, stride=2),
            _cbr(feat_ch, feat_ch, stride=1),
        )
        # main heads
        self.center_heatmap_head = nn.Conv2d(feat_ch, n_classes, 1)
        self.dim_head = nn.Conv2d(feat_ch, 3, 1)  # 3D box (w, h, l)
        self.depth_head = nn.Conv2d(feat_ch, 1, 1)
        self.orientation_head = nn.Conv2d(feat_ch, 2, 1)  # (sin, cos) of heading angle
        # auxiliary monocular-context heads (training-time only in the paper)
        self.corner_keypoint_head = nn.Conv2d(feat_ch, 16, 1)  # 8 corners x (x, y) offset
        self.bbox_quant_offset_head = nn.Conv2d(feat_ch, 2, 1)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(image)
        heatmap = torch.sigmoid(self.center_heatmap_head(feat))
        dims = self.dim_head(feat)
        depth = self.depth_head(feat)
        orientation = self.orientation_head(feat)
        corner_kpts = self.corner_keypoint_head(feat)
        bbox_quant_offset = self.bbox_quant_offset_head(feat)
        return heatmap, dims, depth, orientation, corner_kpts, bbox_quant_offset


def build_monocon() -> nn.Module:
    """Build a small MonoCon monocular 3D detector with auxiliary context heads."""
    return MonoCon(in_ch=3, feat_ch=16, n_classes=3).eval()


def example_input_monocon() -> torch.Tensor:
    """Single monocular RGB image (1, 3, 32, 32)."""
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("LaRa", "build_lara", "example_input_lara", "2022", "VIS"),
    ("LaserNet", "build_lasernet", "example_input_lasernet", "2019", "VIS"),
    ("Lift-Splat-Shoot", "build_lift_splat_shoot", "example_input_lift_splat_shoot", "2020", "VIS"),
    ("M3D-RPN", "build_m3d_rpn", "example_input_m3d_rpn", "2019", "VIS"),
    ("MapVR", "build_mapvr", "example_input_mapvr", "2023", "VIS"),
    ("MonoCon", "build_monocon", "example_input_monocon", "2022", "VIS"),
]
