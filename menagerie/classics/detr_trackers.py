"""DETR-based multi-object trackers: TrackFormer and TransTrack.

TrackFormer: Multi-Object Tracking with Transformers (Meinhardt et al., CVPR 2022).
  Paper: https://arxiv.org/abs/2101.02702
  Source: https://github.com/timmeinhardt/trackformer
TransTrack: Multiple Object Tracking with Transformer (Sun et al., 2020).
  Paper: https://arxiv.org/abs/2012.15460
  Source: https://github.com/PeizeSun/TransTrack

Both build on the DETR detection transformer (CNN backbone -> transformer
encoder over flattened image features -> transformer decoder reading a set of
object queries -> per-query class + box heads). The tracking-specific structure:

* **TrackFormer** augments the fixed object queries with *track queries*: the
  decoder output embeddings of objects detected in the previous frame are fed
  back in as additional queries this frame, so identity is propagated
  autoregressively through the query set. (The base detector is Deformable-DETR
  with a ResNet-50 backbone; here we use a compact ResNet stem + standard
  attention to keep the graph CUDA-free and traceable.)
* **TransTrack** runs the DETR decoder twice with a *shared* decoder: learned
  object queries produce detection boxes on the current frame, and the previous
  frame's decoder embeddings act as track queries producing tracking boxes; the
  two sets are matched by IoU. Architecturally this is one shared decoder reading
  two query sets over the current encoder memory.

These are faithful random-init reimplementations of the shared DETR tracking core.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStem(nn.Module):
    """Compact ResNet-style backbone stem producing a feature map (stand-in for ResNet-50)."""

    def __init__(self, out_ch: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, out_ch, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class TransformerEncoderLayer(nn.Module):
    """Standard pre-norm-free DETR encoder layer (self-attn + FFN)."""

    def __init__(self, dim: int = 256, n_heads: int = 8, ffn: int = 1024) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.ReLU(inplace=True), nn.Linear(ffn, dim))
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        attn, _ = self.self_attn(src, src, src, need_weights=False)
        src = self.ln1(src + attn)
        src = self.ln2(src + self.ffn(src))
        return src


class TransformerDecoderLayer(nn.Module):
    """DETR decoder layer: self-attn over queries + cross-attn to encoder memory + FFN."""

    def __init__(self, dim: int = 256, n_heads: int = 8, ffn: int = 1024) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.ReLU(inplace=True), nn.Linear(ffn, dim))
        self.ln3 = nn.LayerNorm(dim)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        sa, _ = self.self_attn(queries, queries, queries, need_weights=False)
        queries = self.ln1(queries + sa)
        ca, _ = self.cross_attn(queries, memory, memory, need_weights=False)
        queries = self.ln2(queries + ca)
        queries = self.ln3(queries + self.ffn(queries))
        return queries


class DETRCore(nn.Module):
    """Shared DETR core: backbone + encoder + decoder + class/box heads."""

    def __init__(
        self, dim: int = 256, n_enc: int = 2, n_dec: int = 2, num_classes: int = 91
    ) -> None:
        super().__init__()
        self.backbone = ConvStem(out_ch=dim)
        self.col_embed = nn.Parameter(torch.randn(50, dim // 2))
        self.row_embed = nn.Parameter(torch.randn(50, dim // 2))
        self.encoder = nn.ModuleList([TransformerEncoderLayer(dim) for _ in range(n_enc)])
        self.decoder = nn.ModuleList([TransformerDecoderLayer(dim) for _ in range(n_dec)])
        self.class_head = nn.Linear(dim, num_classes + 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, 4)
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(image)  # (B, dim, H, W)
        b, c, h, w = feat.shape
        pos = torch.cat(
            [
                self.col_embed[:w].unsqueeze(0).expand(h, -1, -1),
                self.row_embed[:h].unsqueeze(1).expand(-1, w, -1),
            ],
            dim=-1,
        ).reshape(h * w, c)
        src = feat.flatten(2).permute(0, 2, 1) + pos.unsqueeze(0)
        for layer in self.encoder:
            src = layer(src)
        return src

    def decode(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder:
            queries = layer(queries, memory)
        return queries

    def heads(self, embeds: torch.Tensor) -> torch.Tensor:
        cls = self.class_head(embeds)
        box = self.bbox_head(embeds).sigmoid()
        return torch.cat([cls, box], dim=-1)


class TrackFormer(nn.Module):
    """TrackFormer: DETR detector whose query set = object queries + track queries.

    The track queries are the previous frame's decoder output embeddings, fed
    back in as additional queries this frame to propagate identity.
    """

    def __init__(self, dim: int = 256, num_object_queries: int = 100, num_track_queries: int = 20):
        super().__init__()
        self.core = DETRCore(dim=dim)
        self.object_queries = nn.Parameter(torch.randn(num_object_queries, dim))
        self.num_track = num_track_queries

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        memory = self.core.encode(image)
        obj_q = self.object_queries.unsqueeze(0).expand(b, -1, -1)
        # Track queries: a learned-free initialization standing in for last frame's
        # decoder embeddings (random per forward; in deployment these are recurrent).
        track_q = torch.randn(b, self.num_track, obj_q.shape[-1], device=image.device)
        queries = torch.cat([obj_q, track_q], dim=1)
        embeds = self.core.decode(queries, memory)
        return self.core.heads(embeds)


class TransTrack(nn.Module):
    """TransTrack: shared DETR decoder reading object queries + track queries on one frame.

    Detection queries produce detection boxes; the previous frame's decoder
    embeddings (track queries) produce tracking boxes via the SAME decoder.
    """

    def __init__(self, dim: int = 256, num_queries: int = 100) -> None:
        super().__init__()
        self.core = DETRCore(dim=dim)
        self.det_queries = nn.Parameter(torch.randn(num_queries, dim))
        self.num_queries = num_queries

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        memory = self.core.encode(image)
        det_q = self.det_queries.unsqueeze(0).expand(b, -1, -1)
        # Track queries: previous-frame decoder embeddings (random stand-in here).
        track_q = torch.randn(b, self.num_queries, det_q.shape[-1], device=image.device)
        det_embeds = self.core.decode(det_q, memory)  # detection set
        track_embeds = self.core.decode(track_q, memory)  # shared-decoder tracking set
        det_out = self.core.heads(det_embeds)
        track_out = self.core.heads(track_embeds)
        return torch.cat([det_out, track_out], dim=1)


def build_trackformer() -> nn.Module:
    """Build the TrackFormer DETR tracker (object + track queries)."""
    return TrackFormer(dim=256, num_object_queries=100, num_track_queries=20)


def build_transtrack() -> nn.Module:
    """Build the TransTrack shared-decoder DETR tracker."""
    return TransTrack(dim=256, num_queries=100)


def example_input() -> torch.Tensor:
    """Example image tensor ``(1, 3, 256, 256)`` (small frame keeps the trace compact)."""
    return torch.randn(1, 3, 256, 256)


MENAGERIE_ENTRIES = [
    (
        "TrackFormer (DETR multi-object tracker with track queries)",
        "build_trackformer",
        "example_input",
        "2022",
        "DC",
    ),
    (
        "TransTrack (shared-decoder DETR multi-object tracker)",
        "build_transtrack",
        "example_input",
        "2020",
        "DC",
    ),
]
