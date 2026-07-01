"""Compact faithful reconstructions of six autonomous-driving 3D-perception architectures.

Sources checked (repo READMEs / source files / arXiv abstracts, no clone/install):
  - D4LCN: https://github.com/dingmyu/D4LCN (models/resnet_dilate.py:
    ``dynamic_local_filtering``; arXiv:1912.04799, CVPR 2020)
  - DAROD: https://github.com/colindecourt/darod (darod/models/model.py,
    darod/models/darod_blocks.py; IEEE IV 2022 -- TensorFlow original,
    reimplemented here as an equivalent torch nn.Module)
  - DD3D: https://github.com/TRI-ML/dd3d (tridet/modeling/dd3d/fcos3d.py;
    arXiv:2108.06417, ICCV 2021)
  - DeepInteraction: https://github.com/fudan-zvg/DeepInteraction
    (projects/mmdet3d_plugin/models/necks/deepinteraction_encoder.py;
    NeurIPS 2022)
  - DriveAdapter: https://github.com/OpenDriveLab/DriveAdapter
    (open_loop_training/code/model_code/dense_heads/driver_adapter_head.py:
    ``AdapterCNN``/``SEModule``; arXiv:2308.00398, ICCV 2023 Oral)
  - Ego3RT: https://github.com/fudan-zvg/Ego3RT (README methods figure +
    arXiv:2206.04042 abstract; ECCV 2022)

Every model below reimplements the paper's distinctive mechanism from scratch in
base-env torch with small random-init dimensions; none of these load pretrained
weights or third-party code.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# D4LCN: depth-guided dynamic local convolutions for monocular 3D detection
# (Ding et al., CVPR 2020).
#
# Distinctive mechanism: a dual-branch backbone (RGB image branch + a
# parallel depth-map branch of identical topology) where, at intermediate
# stages, the RGB features are re-filtered by ``dynamic_local_filtering``:
# a depth-MODULATED, per-pixel-adaptive local aggregation over a small
# dilated neighborhood (the depth values act as per-location dynamic conv
# weights), then the multiple dilation rates (1, 2, 3) are combined with a
# learned per-channel softmax weighting ("adaptive dilation").
# ---------------------------------------------------------------------------


def dynamic_local_filtering(x: Tensor, depth: Tensor, dilated: int = 1) -> Tensor:
    """Depth-modulated local aggregation over a dilated 3x3 neighborhood.

    Parameters
    ----------
    x : Tensor
        ``(B, C, H, W)`` RGB-branch feature map to be re-filtered.
    depth : Tensor
        ``(B, C, H, W)`` depth-branch feature map used as a per-location,
        per-channel dynamic filter weight (same shape as ``x``).
    dilated : int
        Dilation rate of the 3x3 neighborhood sampled around each pixel.

    Returns
    -------
    Tensor
        ``(B, C, H, W)`` depth-guided locally-filtered feature map.
    """

    pad = nn.ReflectionPad2d(dilated)
    pad_depth = pad(depth)
    _n, c, h, w = x.shape
    # Cheap 3-channel local mix along the channel axis (paper's simplified
    # low-rank channel context) before the spatial depth-weighted sum.
    shift1 = torch.cat((x[:, -1:, :, :], x[:, :-1, :, :]), dim=1)
    shift2 = torch.cat((x[:, -2:, :, :], x[:, :-2, :, :]), dim=1)
    x = (x + shift1 + shift2) / 3
    pad_x = pad(x)
    out = (
        pad_depth[:, :, dilated : dilated + h, dilated : dilated + w]
        * pad_x[:, :, dilated : dilated + h, dilated : dilated + w]
    ).clone()
    for di in (-dilated, 0, dilated):
        for dj in (-dilated, 0, dilated):
            if di != 0 or dj != 0:
                out = out + (
                    pad_depth[
                        :, :, dilated + di : dilated + di + h, dilated + dj : dilated + dj + w
                    ]
                    * pad_x[:, :, dilated + di : dilated + di + h, dilated + dj : dilated + dj + w]
                )
    return out / 9


class D4LCNBranch(nn.Module):
    """Small shared-topology conv branch used for both the RGB and depth streams."""

    def __init__(self, in_channels: int, width: int) -> None:
        """Initialize a 3-stage strided conv tower producing ``width`` channels."""

        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(width, width, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(width, width, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the three intermediate feature maps of the branch."""

        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return f1, f2, f3


class D4LCNDetector(nn.Module):
    """Compact D4LCN: dual RGB/depth branches fused via depth-guided local conv."""

    def __init__(self, width: int = 16, n_anchors: int = 6) -> None:
        """Initialize the RGB/depth twin branches, adaptive-dilation gate, and head."""

        super().__init__()
        self.rgb_branch = D4LCNBranch(3, width)
        self.depth_branch = D4LCNBranch(1, width)
        self.adaptive_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(width, width * 3, 1))
        self.bn = nn.BatchNorm2d(width)
        # RPN-style output: objectness + 2D box + 3D box (x,y,z,w,h,l,ry) per anchor.
        self.cls = nn.Conv2d(width, n_anchors, 1)
        self.bbox_2d = nn.Conv2d(width, n_anchors * 4, 1)
        self.bbox_3d = nn.Conv2d(width, n_anchors * 7, 1)

    def forward(self, rgb: Tensor, depth: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Detect 2D+3D boxes from an RGB image guided by a monocular depth map.

        Parameters
        ----------
        rgb : Tensor
            ``(B, 3, H, W)`` input RGB image.
        depth : Tensor
            ``(B, 1, H, W)`` monocular depth map (e.g. from DORN).

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Per-cell objectness logits, 2D box deltas, and 3D box parameters.
        """

        _, _, rgb3 = self.rgb_branch(rgb)
        _, _, depth3 = self.depth_branch(depth)

        weight = self.adaptive_gate(rgb3).reshape(rgb3.shape[0], rgb3.shape[1], 1, 3)
        weight = torch.softmax(weight, dim=3)

        fused = (
            dynamic_local_filtering(rgb3, depth3, dilated=1) * weight[:, :, :, 0:1]
            + dynamic_local_filtering(rgb3, depth3, dilated=2) * weight[:, :, :, 1:2]
            + dynamic_local_filtering(rgb3, depth3, dilated=3) * weight[:, :, :, 2:3]
        )
        fused = F.relu(self.bn(fused))
        return self.cls(fused), self.bbox_2d(fused), self.bbox_3d(fused)


def build_d4lcn() -> nn.Module:
    """Build a compact random-init D4LCN monocular 3D detector."""

    return D4LCNDetector().eval()


def example_input_d4lcn() -> tuple[Tensor, Tensor]:
    """Return a small RGB image and its aligned monocular depth map."""

    return torch.randn(1, 3, 64, 64), torch.randn(1, 1, 64, 64)


# ---------------------------------------------------------------------------
# DAROD: Faster R-CNN adapted to automotive radar range-Doppler maps
# (Decourt et al., IEEE IV 2022).
#
# Distinctive mechanism: a Faster-R-CNN-style RPN + RoI head run directly on
# range-Doppler (RD) spectra instead of camera images, with an ASYMMETRIC
# backbone (range axis is pooled more aggressively than the Doppler axis,
# reflecting the very different physical resolutions of the two axes) and a
# ``RadarFeatures`` side-branch that folds RoI box geometry + RPN objectness
# score directly into the Fast-R-CNN classifier input (radar boxes carry
# little visual texture, so raw geometry/confidence is an informative cue).
# ---------------------------------------------------------------------------


class DARODConvBlock(nn.Module):
    """Two/three-conv block with group norm, LeakyReLU, and asymmetric pooling."""

    def __init__(
        self, in_ch: int, out_ch: int, n_conv: int = 2, pool: tuple[int, int] = (2, 2)
    ) -> None:
        """Initialize ``n_conv`` stacked 3x3 convs followed by asymmetric pooling."""

        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_ch
        for _ in range(n_conv):
            layers += [
                nn.Conv2d(c_in, out_ch, 3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.LeakyReLU(0.1, inplace=True),
            ]
            c_in = out_ch
        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(pool, stride=pool)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the conv stack then the block's (range, Doppler)-asymmetric pool."""

        return self.pool(self.convs(x))


class DARODDetector(nn.Module):
    """Compact DAROD: RD-map backbone + RPN + RoI head with radar-geometry features."""

    def __init__(self, in_channels: int = 2, n_anchors: int = 9, n_classes: int = 4) -> None:
        """Initialize the asymmetric-pool backbone, RPN, and Fast-R-CNN head.

        ``in_channels=2`` models a complex RD map (real + imaginary channels).
        """

        super().__init__()
        self.block1 = DARODConvBlock(in_channels, 32, n_conv=2, pool=(2, 2))
        self.block2 = DARODConvBlock(32, 64, n_conv=2, pool=(2, 1))  # range >> Doppler
        self.block3 = DARODConvBlock(64, 96, n_conv=3, pool=(2, 1))

        self.rpn_conv = nn.Conv2d(96, 96, 3, padding=1)
        self.rpn_cls = nn.Conv2d(96, n_anchors, 1)
        self.rpn_reg = nn.Conv2d(96, n_anchors * 4, 1)

        roi_size = 5
        self.roi_pool = nn.AdaptiveAvgPool2d(roi_size)
        flat_dim = 96 * roi_size * roi_size
        # RadarFeatures side-branch: RoI box geometry (y1,x1,y2,x2) + objectness score.
        self.radar_feat_mlp = nn.Sequential(nn.Linear(5, 16), nn.ReLU(inplace=True))
        self.fc1 = nn.Linear(flat_dim + 16, 64)
        self.fc2 = nn.Linear(64, 64)
        self.frcnn_cls = nn.Linear(64, n_classes)
        self.frcnn_reg = nn.Linear(64, n_classes * 4)

    def forward(self, rd_map: Tensor, roi_boxes: Tensor, roi_scores: Tensor) -> tuple[Tensor, ...]:
        """Detect objects on a radar range-Doppler map given proposal RoIs.

        Parameters
        ----------
        rd_map : Tensor
            ``(B, 2, R, D)`` range-Doppler spectrum (real/imaginary channels).
        roi_boxes : Tensor
            ``(B, N, 4)`` proposal boxes ``(y1, x1, y2, x2)`` in ``[0, 1]``.
        roi_scores : Tensor
            ``(B, N, 1)`` proposal (RPN) objectness scores.

        Returns
        -------
        tuple[Tensor, ...]
            ``(rpn_cls, rpn_reg, frcnn_cls, frcnn_reg)``.
        """

        feats = self.block1(rd_map)
        feats = self.block2(feats)
        feats = self.block3(feats)

        rpn_hidden = F.relu(self.rpn_conv(feats))
        rpn_cls = self.rpn_cls(rpn_hidden)
        rpn_reg = self.rpn_reg(rpn_hidden)

        batch, n_roi = roi_boxes.shape[0], roi_boxes.shape[1]
        pooled = self.roi_pool(feats)  # shared pooled grid stands in for per-RoI align
        pooled_flat = pooled.flatten(1).unsqueeze(1).expand(-1, n_roi, -1)

        radar_feats = self.radar_feat_mlp(torch.cat([roi_boxes, roi_scores], dim=-1))
        merged = torch.cat([pooled_flat, radar_feats], dim=-1)

        hidden = F.relu(self.fc1(merged))
        hidden = F.relu(self.fc2(hidden))
        frcnn_cls = self.frcnn_cls(hidden)
        frcnn_reg = self.frcnn_reg(hidden)
        return rpn_cls, rpn_reg, frcnn_cls.view(batch, n_roi, -1), frcnn_reg.view(batch, n_roi, -1)


def build_darod() -> nn.Module:
    """Build a compact random-init DAROD radar detector."""

    return DARODDetector().eval()


def example_input_darod() -> tuple[Tensor, Tensor, Tensor]:
    """Return a small RD map plus a handful of proposal RoIs and their scores."""

    rd_map = torch.randn(1, 2, 64, 64)
    boxes = torch.rand(1, 5, 4).sort(dim=-1).values
    scores = torch.rand(1, 5, 1)
    return rd_map, boxes, scores


# ---------------------------------------------------------------------------
# DD3D: end-to-end single-stage monocular 3D detection with depth pretraining
# (Park et al., ICCV 2021).
#
# Distinctive mechanism: an FCOS-style dense per-pixel head extended with a
# DISENTANGLED 3D box parameterization -- an allocentric rotation quaternion,
# a 2D projected-center offset, a (focal-length-normalized) depth, a 3D size,
# and a confidence -- all regressed densely per feature-map location, so the
# same backbone can be pretrained on dense depth estimation before 3D-box
# fine-tuning (no RoI pooling / two-stage pipeline).
# ---------------------------------------------------------------------------


class FCOS3DHead(nn.Module):
    """Dense FCOS-style head with a disentangled 3D box parameterization."""

    def __init__(self, in_channels: int, n_classes: int = 3) -> None:
        """Initialize the shared tower and the five disentangled prediction branches."""

        super().__init__()
        self.tower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(8, in_channels),
            nn.ReLU(inplace=True),
        )
        self.cls_logits = nn.Conv2d(in_channels, n_classes, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.box2d = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.quat = nn.Conv2d(in_channels, 4, 3, padding=1)  # allocentric rotation
        self.proj_ctr = nn.Conv2d(in_channels, 2, 3, padding=1)  # 2D center offset
        self.depth = nn.Conv2d(in_channels, 1, 3, padding=1)  # focal-normalized depth
        self.size3d = nn.Conv2d(in_channels, 3, 3, padding=1)
        self.conf3d = nn.Conv2d(in_channels, 1, 3, padding=1)

    def forward(self, feats: Tensor) -> dict[str, Tensor]:
        """Predict per-location 2D class/centerness and disentangled 3D box terms."""

        hidden = self.tower(feats)
        quat = self.quat(hidden)
        quat = quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-7)
        return {
            "cls_logits": self.cls_logits(hidden),
            "centerness": self.centerness(hidden),
            "box2d": self.box2d(hidden),
            "quat": quat,
            "proj_ctr": self.proj_ctr(hidden),
            "depth": F.softplus(self.depth(hidden)),
            "size3d": self.size3d(hidden).tanh(),
            "conf3d": self.conf3d(hidden),
        }


class DD3DDetector(nn.Module):
    """Compact DD3D: FPN-lite backbone + dense disentangled FCOS3D head."""

    def __init__(self, width: int = 24) -> None:
        """Initialize a small strided conv backbone feeding the FCOS3D head."""

        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = FCOS3DHead(width)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        """Densely predict per-pixel 3D cuboids from a single RGB image."""

        feats = self.backbone(image)
        return self.head(feats)


def build_dd3d() -> nn.Module:
    """Build a compact random-init DD3D single-stage monocular 3D detector."""

    return DD3DDetector().eval()


def example_input_dd3d() -> Tensor:
    """Return a small RGB image."""

    return torch.randn(1, 3, 64, 64)


# ---------------------------------------------------------------------------
# DeepInteraction: multi-modal 3D detection via bidirectional modality
# interaction (Yang et al., NeurIPS 2022).
#
# Distinctive mechanism: instead of fusing camera + LiDAR into one shared
# representation, DeepInteraction keeps TWO separate modality-specific
# representations throughout the encoder. Each encoder layer runs an
# image-to-point (I2P) and a point-to-image (P2I) cross-modal interaction
# block IN PARALLEL with a same-modality local-context self-attention block,
# then fuses ("augments") each modality's own stream with the cross-modal
# read-out -- so both the image and the LiDAR/BEV representations are
# refined and preserved (not collapsed into one tensor) across layers.
# ---------------------------------------------------------------------------


class CrossModalInteraction(nn.Module):
    """Cross-attention read-out from one modality's tokens into another's grid."""

    def __init__(self, dim: int, n_heads: int = 4) -> None:
        """Initialize the cross-modal multihead attention block."""

        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query_tokens: Tensor, kv_tokens: Tensor) -> Tensor:
        """Read the other modality's tokens into ``query_tokens``'s frame."""

        out, _ = self.attn(query_tokens, kv_tokens, kv_tokens)
        return self.norm(query_tokens + out)


class DeepInteractionEncoderLayer(nn.Module):
    """One dual-stream layer: I2P + P_self, P2I + I_self, then per-modality fuse."""

    def __init__(self, dim: int) -> None:
        """Initialize the four interaction blocks and two per-modality fuse MLPs."""

        super().__init__()
        self.i2p = CrossModalInteraction(dim)
        self.p_self = CrossModalInteraction(dim)
        self.p_fuse = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(inplace=True))

        self.p2i = CrossModalInteraction(dim)
        self.i_self = CrossModalInteraction(dim)
        self.i_fuse = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(inplace=True))

    def forward(self, img_tokens: Tensor, pts_tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Refine both modality streams with cross-modal and self-modal context."""

        i2p_out = self.i2p(pts_tokens, img_tokens)
        p_self_out = self.p_self(pts_tokens, pts_tokens)
        new_pts = pts_tokens + self.p_fuse(torch.cat([i2p_out, p_self_out], dim=-1))

        p2i_out = self.p2i(img_tokens, pts_tokens)
        i_self_out = self.i_self(img_tokens, img_tokens)
        new_img = img_tokens + self.i_fuse(torch.cat([p2i_out, i_self_out], dim=-1))

        return new_img, new_pts


class DeepInteractionEncoder(nn.Module):
    """Compact DeepInteraction: dual-stream image/LiDAR encoder + fused query decode."""

    def __init__(self, dim: int = 32, n_layers: int = 2, n_queries: int = 6) -> None:
        """Initialize modality projections, encoder layers, and a query decode head."""

        super().__init__()
        self.img_proj = nn.Linear(16, dim)
        self.pts_proj = nn.Linear(8, dim)
        self.layers = nn.ModuleList([DeepInteractionEncoderLayer(dim) for _ in range(n_layers)])
        self.query_embed = nn.Parameter(torch.randn(n_queries, dim) * 0.02)
        self.decoder_attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.cls_head = nn.Linear(dim, 3)
        self.box_head = nn.Linear(dim, 7)

    def forward(self, img_feats: Tensor, pts_feats: Tensor) -> tuple[Tensor, Tensor]:
        """Fuse camera and LiDAR streams, then decode per-query 3D boxes.

        Parameters
        ----------
        img_feats : Tensor
            ``(B, N_img, 16)`` per-pixel/patch camera tokens.
        pts_feats : Tensor
            ``(B, N_pts, 8)`` per-voxel LiDAR/BEV tokens.

        Returns
        -------
        tuple[Tensor, Tensor]
            Per-query class logits ``(B, Q, 3)`` and box regression ``(B, Q, 7)``.
        """

        img = self.img_proj(img_feats)
        pts = self.pts_proj(pts_feats)
        for layer in self.layers:
            img, pts = layer(img, pts)

        memory = torch.cat([img, pts], dim=1)
        queries = self.query_embed[None].expand(img.shape[0], -1, -1)
        decoded, _ = self.decoder_attn(queries, memory, memory)
        return self.cls_head(decoded), self.box_head(decoded)


def build_deepinteraction() -> nn.Module:
    """Build a compact random-init DeepInteraction dual-stream detector."""

    return DeepInteractionEncoder().eval()


def example_input_deepinteraction() -> tuple[Tensor, Tensor]:
    """Return small camera-token and LiDAR/BEV-token feature sets."""

    return torch.randn(1, 10, 16), torch.randn(1, 14, 8)


# ---------------------------------------------------------------------------
# DriveAdapter: decoupling perception and planning via feature-aligning
# adapters (Jia et al., ICCV 2023 Oral).
#
# Distinctive mechanism: a FROZEN teacher planner (that expects privileged
# BEV-segmentation features) stays untouched; a student perception backbone
# predicts its own BEV features, and small SE-modulated residual "adapter"
# CNNs are inserted at each stage to align the student's features with the
# teacher's expected features (multi-stage feature-alignment loss target),
# rather than training an end-to-end student planner from scratch.
# ---------------------------------------------------------------------------


class SEModule(nn.Module):
    """Squeeze-excitation gate mixing average- and max-pooled channel statistics."""

    def __init__(self, channels: int) -> None:
        """Initialize the two 1x1-conv squeeze-excitation projections."""

        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels, 1)
        self.fc2 = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Gate ``x`` channel-wise by a blend of mean- and max-pooled statistics."""

        pooled = 0.5 * x.mean((2, 3), keepdim=True) + 0.5 * x.amax((2, 3), keepdim=True)
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(pooled))))
        return x * gate


class AdapterCNN(nn.Module):
    """Residual SE-gated adapter aligning a student feature map to a teacher's."""

    def __init__(self, channels: int, out_channels: int) -> None:
        """Initialize the bottleneck residual conv + SE gate + output projection."""

        super().__init__()
        mid = max(channels // 4, 4)
        self.conv1 = nn.Conv2d(channels, mid, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEModule(channels)
        self.out_conv = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Refine ``x`` through a residual SE bottleneck, then project to teacher width."""

        residual = x
        hidden = F.relu(self.bn1(self.conv1(x)))
        hidden = self.bn2(self.conv2(hidden))
        hidden = self.se(hidden)
        hidden = F.relu(hidden + residual)
        return F.relu(self.out_conv(hidden))


class DriveAdapterStudent(nn.Module):
    """Compact DriveAdapter: student perception backbone + multi-stage adapters.

    Emits both its own BEV feature prediction and the corresponding
    teacher-aligned features (what would be compared against a frozen
    teacher planner's intermediate activations via a feature-alignment loss).
    """

    def __init__(self, width: int = 20, teacher_channels: int = 20) -> None:
        """Initialize the two-stage student backbone and their aligning adapters."""

        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(6, width, 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(inplace=True)
        )
        self.adapter1 = AdapterCNN(width, teacher_channels)
        self.adapter2 = AdapterCNN(width, teacher_channels)
        self.bev_seg_head = nn.Conv2d(width, 5, 1)

    def forward(self, bev_input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict a BEV segmentation and teacher-aligned adapter features.

        Parameters
        ----------
        bev_input : Tensor
            ``(B, 6, H, W)`` rasterized multi-camera BEV feature stack.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            BEV segmentation logits, and the two stages' teacher-aligned
            adapter feature maps (targets for the feature-alignment loss).
        """

        f1 = self.stage1(bev_input)
        f2 = self.stage2(f1)
        aligned1 = self.adapter1(f1)
        aligned2 = self.adapter2(f2)
        bev_seg = self.bev_seg_head(f1)
        return bev_seg, aligned1, aligned2


def build_driveadapter() -> nn.Module:
    """Build a compact random-init DriveAdapter student perception module."""

    return DriveAdapterStudent().eval()


def example_input_driveadapter() -> Tensor:
    """Return a small rasterized multi-camera BEV feature stack."""

    return torch.randn(1, 6, 48, 48)


# ---------------------------------------------------------------------------
# Ego3RT: ego 3D representation learned via a ray-tracing-inspired polar
# grid of "imaginary eyes" (Lu et al., ECCV 2022).
#
# Distinctive mechanism: rather than estimating per-pixel depth or using an
# unstructured set of 3D queries, Ego3RT defines a POLARIZED grid of
# learnable 3D query points (r, theta, phi around the ego vehicle -- an
# "imaginary eye" per ray/cell) with built-in geometric structure. Each
# query is projected into every camera view via a 3D-to-2D projection and
# adaptively cross-attends to the corresponding image-plane features,
# lifting multi-camera 2D features into a single ego-centric 3D polar (then
# reshaped to BEV) representation without any depth supervision.
# ---------------------------------------------------------------------------


class PolarQueryProjection(nn.Module):
    """Project polar-grid 3D query points to camera-plane sampling embeddings."""

    def __init__(self, dim: int) -> None:
        """Initialize the small MLP mapping projected (u, v, valid) to an embedding."""

        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))

    def forward(self, uv_valid: Tensor) -> Tensor:
        """Encode per-camera projected coordinates (and validity) into query bias."""

        return self.mlp(uv_valid)


class Ego3RTLifter(nn.Module):
    """Compact Ego3RT: polar imaginary-eye grid cross-attending multi-camera features."""

    def __init__(self, dim: int = 24, n_range: int = 6, n_angle: int = 8, n_cams: int = 3) -> None:
        """Initialize the polar query grid, camera-projection encoder, and attention."""

        super().__init__()
        self.n_range = n_range
        self.n_angle = n_angle
        self.n_cams = n_cams
        n_rays = n_range * n_angle
        self.query_embed = nn.Parameter(torch.randn(n_rays, dim) * 0.02)
        self.proj_encoder = PolarQueryProjection(dim)
        self.img_proj = nn.Linear(32, dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.seg_head = nn.Linear(dim, 4)

    def forward(self, cam_feats: Tensor, cam_uv_valid: Tensor) -> Tensor:
        """Lift multi-camera 2D features into an ego-centric polar BEV grid.

        Parameters
        ----------
        cam_feats : Tensor
            ``(B, n_cams, N_pix, 32)`` per-camera flattened image-plane
            features (one token per sampled pixel/patch).
        cam_uv_valid : Tensor
            ``(B, n_rays, n_cams, 3)`` per-ray, per-camera projected
            ``(u, v, valid)`` coordinates for each polar query ray.

        Returns
        -------
        Tensor
            ``(B, n_range, n_angle, 4)`` per-cell BEV-segmentation logits.
        """

        batch = cam_feats.shape[0]
        img_tokens = self.img_proj(cam_feats).flatten(1, 2)  # (B, n_cams*N_pix, dim)

        proj_bias = self.proj_encoder(cam_uv_valid).mean(dim=2)  # (B, n_rays, dim)
        queries = self.query_embed[None].expand(batch, -1, -1) + proj_bias

        lifted, _ = self.attn(queries, img_tokens, img_tokens)
        lifted = self.norm(queries + lifted)

        logits = self.seg_head(lifted)
        return logits.view(batch, self.n_range, self.n_angle, -1)


def build_ego3rt() -> nn.Module:
    """Build a compact random-init Ego3RT polar ray-tracing BEV lifter."""

    return Ego3RTLifter().eval()


def example_input_ego3rt() -> tuple[Tensor, Tensor]:
    """Return small multi-camera feature tokens and their per-ray projections."""

    cam_feats = torch.randn(1, 3, 20, 32)
    cam_uv_valid = torch.rand(1, 6 * 8, 3, 3)
    return cam_feats, cam_uv_valid


MENAGERIE_ENTRIES = [
    ("D4LCN", "build_d4lcn", "example_input_d4lcn", "2020", "DET"),
    ("DAROD", "build_darod", "example_input_darod", "2022", "DET"),
    ("DD3D", "build_dd3d", "example_input_dd3d", "2021", "DET"),
    ("DeepInteraction", "build_deepinteraction", "example_input_deepinteraction", "2022", "DET"),
    ("DriveAdapter", "build_driveadapter", "example_input_driveadapter", "2023", "RL"),
    ("Ego3RT", "build_ego3rt", "example_input_ego3rt", "2022", "DET"),
]
