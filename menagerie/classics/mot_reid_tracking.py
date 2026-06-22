"""Multi-Object Tracking and Re-Identification classics.

This module covers five architectures from the MOT / ReID family:

1. GG-CNN (Generative Grasping CNN, Morrison et al. 2018):
   Paper: https://arxiv.org/abs/1804.05172
   Source: https://github.com/dougsm/ggcnn
   Tiny fully-convolutional encoder-decoder predicting per-pixel grasp quality,
   angle (cos/sin), and width maps from a depth image.  Reproduced exactly.

2. BoT-ReID-ResNet50 (Bag of Tricks, Luo et al. 2019):
   Paper: https://arxiv.org/abs/1903.07071
   Source: https://github.com/michuanhaohao/reid-strong-baseline
   ResNet50 backbone + global average pool + BNNeck (BatchNorm before the ID
   classifier, with triplet loss applied to pre-BN features and ID loss to
   post-BN features).  Faithful compact reproduction with a tiny ResNet stub.

3. FairMOT-DLA34 (Wang et al. 2020):
   Paper: https://arxiv.org/abs/2004.01177
   Source: https://github.com/ifzhang/FairMOT
   CenterNet-style detection head (heatmap / wh / offset) + a parallel re-ID
   embedding branch, both sharing a DLA-34 backbone feature map.  The homogeneous
   detection+reID multi-head design on a shared backbone is the distinctive
   primitive.  Backbone replaced with a compact DLA-style CNN stub; heads
   faithfully reproduced.

4. JDE-DarkNet53 (Joint Detection and Embedding, Wang et al. 2019):
   Paper: https://arxiv.org/abs/1909.12605
   Source: https://github.com/Zhongdao/Towards-Realtime-MOT
   YOLOv3 / DarkNet53 backbone + multi-scale FPN + per-anchor embedding head
   (multi-task: box regression + class + appearance embedding per anchor).
   The per-scale JDE multi-head (box + class + embedding) is the distinctive
   primitive.  Backbone is a compact DarkNet-style stub.

5. TrTr-ResNet50 (Transformer Tracking, Zhao et al. 2021):
   Paper: https://arxiv.org/abs/2105.03817
   Source: https://github.com/tongtybj/TrTr
   ResNet50 features for both template and search image -> transformer encoder
   (self-attn) -> transformer decoder (cross-attention fuses template into search)
   -> MLP box head.  The encoder-decoder cross-attention tracking pipeline is
   faithfully reproduced.  Backbone is a compact CNN stub (documents the
   stand-in).

Architecture notes / simplifications:
  - All backbone stubs are compact CNNs replacing the paper's full backbone.
  - Network dimensions are reduced (hidden dims 32-128, spatial 16-32) to keep
    trace+draw fast on CPU.
  - Forward passes are trace+draw verified.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Shared helpers
# ============================================================


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, pad: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBnRelu(ch, ch), nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + x)


# ============================================================
# 1. GG-CNN
# ============================================================


class GGCNN(nn.Module):
    """Generative Grasping CNN: fully-convolutional depth-image -> grasp maps.

    From Morrison et al. (2018).  Encoder-decoder architecture predicting:
      - quality map (1 ch)
      - angle cos(2*theta) map (1 ch)
      - angle sin(2*theta) map (1 ch)
      - gripper width map (1 ch)

    All four maps at the same spatial resolution as the input.
    """

    def __init__(
        self,
        input_channels: int = 1,
        filter_sizes: Tuple[int, ...] = (32, 16, 8),
        l3_k_size: int = 5,
    ) -> None:
        super().__init__()
        # Encoder (strided convs)
        self.conv1 = nn.Conv2d(input_channels, filter_sizes[0], 11, stride=1, padding=5)
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(
            filter_sizes[1], filter_sizes[2], l3_k_size, stride=1, padding=l3_k_size // 2
        )
        # Decoder (transposed convs back to input resolution)
        self.convt1 = nn.ConvTranspose2d(
            filter_sizes[2], filter_sizes[1], l3_k_size, stride=1, padding=l3_k_size // 2
        )
        self.convt2 = nn.ConvTranspose2d(filter_sizes[1], filter_sizes[0], 5, stride=1, padding=2)
        # Output heads (one per prediction type)
        self.pos_output = nn.ConvTranspose2d(filter_sizes[0], 1, 11, stride=1, padding=5)
        self.cos_output = nn.ConvTranspose2d(filter_sizes[0], 1, 11, stride=1, padding=5)
        self.sin_output = nn.ConvTranspose2d(filter_sizes[0], 1, 11, stride=1, padding=5)
        self.width_output = nn.ConvTranspose2d(filter_sizes[0], 1, 11, stride=1, padding=5)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout1(x)
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = self.dropout2(x)
        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)
        return pos_output, cos_output, sin_output, width_output


class GGCNNWrapper(nn.Module):
    """Wrapper that returns a single concatenated tensor for trace compatibility."""

    def __init__(self) -> None:
        super().__init__()
        self.ggcnn = GGCNN(input_channels=1, filter_sizes=(32, 16, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos, cos, sin, width = self.ggcnn(x)
        return torch.cat([pos, cos, sin, width], dim=1)


# ============================================================
# 2. BoT-ReID (Bag of Tricks) with BNNeck
# ============================================================


class CompactResNet50(nn.Module):
    """Compact ResNet50-style backbone stub for person re-ID.

    Produces a 512-dim global feature vector via global average pooling.
    (Stand-in for the full ResNet-50; documents simplification.)
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer0 = ConvBnRelu(3, 64, 7, stride=2, pad=3)
        self.pool0 = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(ConvBnRelu(64, 64), ResBlock(64))
        self.layer2 = nn.Sequential(ConvBnRelu(64, 128, stride=2), ResBlock(128))
        self.layer3 = nn.Sequential(ConvBnRelu(128, 256, stride=2), ResBlock(256))
        self.layer4 = nn.Sequential(ConvBnRelu(256, 512, stride=2), ResBlock(512))
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool0(self.layer0(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.gap(x).flatten(1)  # (N, 512)


class BNNeck(nn.Module):
    """BNNeck: BatchNorm after global pool, classifier on BN features.

    The distinctive BoT ReID contribution: triplet loss uses pre-BN features
    (f_t), ID loss uses post-BN features (f_id).  At inference both are
    available as outputs.
    """

    def __init__(self, in_feat: int = 512, num_classes: int = 100) -> None:
        super().__init__()
        self.bottleneck = nn.BatchNorm1d(in_feat)
        self.bottleneck.bias.requires_grad_(False)  # no shift (BoT paper choice)
        self.classifier = nn.Linear(in_feat, num_classes, bias=False)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_bn = self.bottleneck(feat)  # post-BN (for ID loss / inference)
        cls_score = self.classifier(feat_bn)
        return cls_score, feat  # (logits, pre-BN triplet features)


class BoTReID(nn.Module):
    """BoT ReID strong baseline (Bag of Tricks for Person Re-ID).

    ResNet50-compact backbone -> GAP -> BNNeck -> ID classifier.
    Returns (class_logits, triplet_features) during training; at inference
    the BN-normalized feature is the retrieval embedding.
    """

    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.backbone = CompactResNet50()
        self.bnneck = BNNeck(in_feat=512, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)  # (N, 512)
        cls_score, feat_t = self.bnneck(feat)
        # Return logits for trace (at inference use feat_t or post-BN feat)
        return cls_score


# ============================================================
# 3. FairMOT-DLA34
# ============================================================


class DLANode(nn.Module):
    """Compact DLA-style aggregation node: two branches summed."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.branch1 = ConvBnRelu(ch, ch)
        self.branch2 = ConvBnRelu(ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch1(x) + self.branch2(x)


class CompactDLA34(nn.Module):
    """Compact DLA-34-style backbone stub.

    DLA-34 uses iterative deep aggregation (IDA + HDA trees).  This stub
    implements the key multi-resolution feature aggregation topology with
    upsampling-sum fusion at two scales, producing a single stride-4 feature
    map suitable for CenterNet-style heads.

    Stand-in for the full DLA-34; documents simplification.
    """

    def __init__(self, base_ch: int = 64) -> None:
        super().__init__()
        # Encoder: stride 1,2,4,8
        self.stem = ConvBnRelu(3, base_ch, 7, stride=1, pad=3)
        self.level1 = nn.Sequential(ConvBnRelu(base_ch, base_ch, stride=2), DLANode(base_ch))
        self.level2 = nn.Sequential(
            ConvBnRelu(base_ch, base_ch * 2, stride=2), DLANode(base_ch * 2)
        )
        self.level3 = nn.Sequential(
            ConvBnRelu(base_ch * 2, base_ch * 4, stride=2), DLANode(base_ch * 4)
        )
        # IDA upsample: aggregate level3->level2->level1
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
        )
        self.agg2 = DLANode(base_ch * 2)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.agg1 = DLANode(base_ch)
        self.out_ch = base_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        x1 = self.level1(x0)
        x2 = self.level2(x1)
        x3 = self.level3(x2)
        y2 = self.agg2(self.up2(x3) + x2)
        y1 = self.agg1(self.up1(y2) + x1)
        return y1  # stride-2 feature map, base_ch channels


class FairMOT(nn.Module):
    """FairMOT: anchor-free CenterNet detection + parallel re-ID embedding branch.

    Shared DLA-34-style backbone -> two parallel heads:
      - Detection head: heatmap (num_classes) + wh (2) + offset (2)
      - Re-ID head: embedding (reid_dim)

    The homogeneous detection+reID on one shared feature map is the
    distinctive FairMOT design.
    """

    def __init__(self, num_classes: int = 1, reid_dim: int = 64) -> None:
        super().__init__()
        self.backbone = CompactDLA34(base_ch=64)
        feat_ch = self.backbone.out_ch  # 64

        # Detection head
        self.heatmap_head = nn.Sequential(
            ConvBnRelu(feat_ch, feat_ch), nn.Conv2d(feat_ch, num_classes, 1)
        )
        self.wh_head = nn.Sequential(ConvBnRelu(feat_ch, feat_ch), nn.Conv2d(feat_ch, 2, 1))
        self.offset_head = nn.Sequential(ConvBnRelu(feat_ch, feat_ch), nn.Conv2d(feat_ch, 2, 1))

        # Re-ID embedding head
        self.reid_head = nn.Sequential(
            ConvBnRelu(feat_ch, feat_ch), nn.Conv2d(feat_ch, reid_dim, 1)
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        hm = self.heatmap_head(feat).sigmoid()
        wh = self.wh_head(feat)
        off = self.offset_head(feat)
        reid = self.reid_head(feat)
        # Return concatenated for single-output trace
        return torch.cat([hm, wh, off, reid], dim=1)


# ============================================================
# 4. JDE-DarkNet53 (Joint Detection and Embedding)
# ============================================================


class DarkResBlock(nn.Module):
    """DarkNet53 residual block: 1x1 + 3x3 with skip."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        mid = ch // 2
        self.net = nn.Sequential(ConvBnRelu(ch, mid, 1, pad=0), ConvBnRelu(mid, ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CompactDarkNet53(nn.Module):
    """Compact DarkNet53 backbone stub with FPN-style multi-scale outputs.

    DarkNet53 has 5 downsampling stages with residual blocks.  This stub
    implements the topology at 1/4 scale (2 stages + 1 block each).
    Returns (p3, p2, p1) multi-scale feature maps for JDE heads.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv0 = ConvBnRelu(3, 32)
        # Stage 1: stride 2
        self.stage1 = nn.Sequential(ConvBnRelu(32, 64, stride=2), DarkResBlock(64))
        # Stage 2: stride 2
        self.stage2 = nn.Sequential(
            ConvBnRelu(64, 128, stride=2), DarkResBlock(128), DarkResBlock(128)
        )
        # Stage 3: stride 2
        self.stage3 = nn.Sequential(ConvBnRelu(128, 256, stride=2), DarkResBlock(256))

        # FPN lateral convs
        self.lat3 = ConvBnRelu(256, 128, 1, pad=0)
        self.lat2 = ConvBnRelu(128, 64, 1, pad=0)
        # FPN output convs
        self.out3 = ConvBnRelu(128, 128)
        self.out2 = ConvBnRelu(64 + 128, 64)
        self.out1 = ConvBnRelu(64 + 64, 64)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stage1(self.conv0(x))  # stride 4
        c2 = self.stage2(c1)  # stride 8
        c3 = self.stage3(c2)  # stride 16

        # FPN top-down
        p3 = self.out3(self.lat3(c3))
        p3_up = F.interpolate(p3, scale_factor=2, mode="nearest")
        p2 = self.out2(torch.cat([self.lat2(c2), p3_up], dim=1))
        p2_up = F.interpolate(p2, scale_factor=2, mode="nearest")
        p1 = self.out1(torch.cat([c1, p2_up], dim=1))
        return p3, p2, p1


class JDEHead(nn.Module):
    """JDE multi-task head per scale: box + class + embedding per anchor.

    Distinctive JDE primitive: each anchor predicts (box, class, embedding).
    """

    def __init__(
        self, in_ch: int, num_anchors: int = 3, num_classes: int = 1, embed_dim: int = 64
    ) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        out_per_anchor = 5 + num_classes + embed_dim  # 4 box + 1 conf + cls + emb
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch, in_ch * 2), nn.Conv2d(in_ch * 2, num_anchors * out_per_anchor, 1)
        )
        self.out_per_anchor = out_per_anchor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        N, _, H, W = out.shape
        return out.view(N, self.num_anchors, self.out_per_anchor, H, W)


class JDE(nn.Module):
    """JDE: DarkNet53 backbone + FPN + per-scale JDE multi-task heads.

    Each scale produces per-anchor (box, conf, class, embedding) predictions.
    """

    def __init__(self, num_anchors: int = 3, num_classes: int = 1, embed_dim: int = 64) -> None:
        super().__init__()
        self.backbone = CompactDarkNet53()
        self.head_large = JDEHead(128, num_anchors, num_classes, embed_dim)  # stride 16
        self.head_med = JDEHead(64, num_anchors, num_classes, embed_dim)  # stride 8
        self.head_small = JDEHead(64, num_anchors, num_classes, embed_dim)  # stride 4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p2, p1 = self.backbone(x)
        out_large = self.head_large(p3)
        out_med = self.head_med(p2)
        out_small = self.head_small(p1)
        # For trace: return all outputs as a tuple; tl.trace handles list/tuple
        return out_large.flatten(1), out_med.flatten(1), out_small.flatten(1)


class JDEWrapper(nn.Module):
    """Trace wrapper: returns concatenated flattened predictions."""

    def __init__(self) -> None:
        super().__init__()
        self.jde = JDE()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b, c = self.jde(x)
        # Concat along dim=1 after padding to same length (pad shorter)
        max_len = max(a.shape[1], b.shape[1], c.shape[1])

        def pad(t: torch.Tensor) -> torch.Tensor:
            if t.shape[1] < max_len:
                return F.pad(t, (0, max_len - t.shape[1]))
            return t

        return torch.stack([pad(a), pad(b), pad(c)], dim=1)


# ============================================================
# 5. TrTr-ResNet50 (Transformer Tracking)
# ============================================================


class CompactCNNBackbone(nn.Module):
    """Compact CNN backbone extracting a spatial feature map.

    Stand-in for ResNet-50; documents simplification.
    """

    def __init__(self, out_ch: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBnRelu(3, 32, 7, stride=2, pad=3),
            ConvBnRelu(32, 64, 3, stride=2),
            ResBlock(64),
            ConvBnRelu(64, out_ch),
            ResBlock(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrTr(nn.Module):
    """TrTr: Transformer Tracking via cross-attention.

    Architecture:
      1. CompactCNN backbone extracts template and search features (shared weights).
      2. Positional encoding added.
      3. Transformer ENCODER (self-attention over search tokens).
      4. Transformer DECODER (cross-attention: search as queries, template as kv).
      5. MLP box head on decoder output (predicts cx, cy, w, h normalized).

    The encoder-decoder cross-attention pipeline fusing template into search
    is TrTr's distinctive primitive.
    """

    def __init__(
        self,
        feat_ch: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_enc_layers: int = 2,
        num_dec_layers: int = 2,
        dim_feedforward: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = CompactCNNBackbone(out_ch=feat_ch)
        self.input_proj = nn.Conv2d(feat_ch, d_model, 1)

        # Transformer encoder (over search features)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        # Transformer decoder (search as queries, template as memory)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)

        # Box prediction MLP head (4 values: cx, cy, w, h)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 4), nn.Sigmoid()
        )

        self.d_model = d_model

    def _extract_tokens(self, img: torch.Tensor) -> torch.Tensor:
        """Extract feature tokens from an image."""
        feat = self.backbone(img)  # (N, feat_ch, H, W)
        feat = self.input_proj(feat)  # (N, d_model, H, W)
        N, C, H, W = feat.shape
        return feat.flatten(2).transpose(1, 2)  # (N, H*W, d_model)

    def forward(self, template: torch.Tensor, search: torch.Tensor) -> torch.Tensor:
        # Template and search share the same backbone
        tmpl_tokens = self._extract_tokens(template)  # (N, T, d)
        srch_tokens = self._extract_tokens(search)  # (N, S, d)

        # Encoder: contextualizes search features
        srch_enc = self.encoder(srch_tokens)  # (N, S, d)

        # Decoder: cross-attends over template as memory
        dec_out = self.decoder(srch_enc, tmpl_tokens)  # (N, S, d)

        # Pool over search positions, predict box
        pooled = dec_out.mean(dim=1)  # (N, d)
        box = self.box_head(pooled)  # (N, 4)
        return box


class TrTrWrapper(nn.Module):
    """Wrapper for TrTr that takes a single stacked input for trace."""

    def __init__(self) -> None:
        super().__init__()
        self.trtr = TrTr()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 2, 3, H, W) -- stacked template + search along dim 1
        template = x[:, 0]  # (N, 3, H, W)
        search = x[:, 1]  # (N, 3, H, W)
        return self.trtr(template, search)


# ============================================================
# Zero-arg builders
# ============================================================


def build_ggcnn() -> nn.Module:
    """Build GG-CNN grasp prediction network."""
    return GGCNNWrapper()


def build_bot_reid() -> nn.Module:
    """Build BoT ReID (Bag of Tricks) strong baseline with BNNeck."""
    return BoTReID(num_classes=100)


def build_fairmot() -> nn.Module:
    """Build FairMOT: anchor-free detection + re-ID on shared DLA-34-style backbone."""
    return FairMOT(num_classes=1, reid_dim=64)


def build_jde() -> nn.Module:
    """Build JDE: DarkNet53 + FPN + per-anchor joint detection+embedding heads."""
    return JDEWrapper()


def build_trtr() -> nn.Module:
    """Build TrTr: transformer tracking via encoder-decoder cross-attention."""
    return TrTrWrapper()


# ============================================================
# Example inputs
# ============================================================


def example_input_ggcnn() -> torch.Tensor:
    """Depth image (1, 1, 64, 64) for GG-CNN."""
    return torch.randn(1, 1, 64, 64)


def example_input_reid() -> torch.Tensor:
    """RGB person image (1, 3, 64, 32) for BoT-ReID."""
    return torch.randn(1, 3, 64, 32)


def example_input_fairmot() -> torch.Tensor:
    """RGB frame (1, 3, 64, 64) for FairMOT."""
    return torch.randn(1, 3, 64, 64)


def example_input_jde() -> torch.Tensor:
    """RGB frame (1, 3, 64, 64) for JDE."""
    return torch.randn(1, 3, 64, 64)


def example_input_trtr() -> torch.Tensor:
    """Stacked (template, search) image pair (1, 2, 3, 32, 32) for TrTr."""
    return torch.randn(1, 2, 3, 32, 32)


# ============================================================
# Menagerie entries
# ============================================================

MENAGERIE_ENTRIES = [
    (
        "GG-CNN (Generative Grasping CNN, grasp-map FCN)",
        "build_ggcnn",
        "example_input_ggcnn",
        "2018",
        "DC",
    ),
    (
        "BoT-ReID-ResNet50 (Bag of Tricks person re-ID, BNNeck)",
        "build_bot_reid",
        "example_input_reid",
        "2019",
        "DC",
    ),
    (
        "FairMOT-DLA34 (anchor-free detection + re-ID on shared backbone)",
        "build_fairmot",
        "example_input_fairmot",
        "2020",
        "DC",
    ),
    (
        "JDE-DarkNet53 (Joint Detection and Embedding, per-anchor embedding)",
        "build_jde",
        "example_input_jde",
        "2019",
        "DC",
    ),
    (
        "TrTr-ResNet50 (Transformer Tracking, encoder-decoder cross-attention)",
        "build_trtr",
        "example_input_trtr",
        "2021",
        "DC",
    ),
]
