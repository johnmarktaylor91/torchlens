"""SAM-DETR: Accelerating DETR Convergence via Semantic-Aligned Matching.

Zhang, Liu, Zhao, Zhang, Li, Wu & Liu (Huawei Noah), CVPR 2022, arXiv:2203.06883.
Source: https://github.com/ZhangGongjie/SAM-DETR

SAM-DETR addresses DETR's slow convergence by adding a Semantic-Aligned Matching
(SAM) auxiliary branch. The distinctive primitive is the SAM module:
  1. For each object query, extract the top-K most semantically salient positions
     from the encoder feature map (by scoring encoder tokens with the query).
  2. Sample (bilinear interpolation / scatter) the encoder features at these
     salient positions to get "resampled" reference features per query.
  3. Align query semantics to resampled encoder features: the decoder cross-
     attention is additionally guided by these semantically aligned references.
  4. An auxiliary matching loss trains the SAM module to align query embeddings
     with the best-matching object features early in training.

Architecture: compact CNN backbone -> flatten -> transformer encoder -> SAM
resampling -> transformer decoder with SAM alignment -> class + box heads.
Compact: 2 encoder layers, 2 decoder layers, 8 object queries, d_model=64.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Compact CNN backbone (ResNet-like stem)                                     #
# --------------------------------------------------------------------------- #


class CompactBackbone(nn.Module):
    """Minimal CNN backbone: 3 conv stages -> feature map."""

    def __init__(self, out_channels: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        return self.layer2(x)  # (B, C, H/4, W/4)


# --------------------------------------------------------------------------- #
#  Semantic-Aligned Matching (SAM) module                                      #
# --------------------------------------------------------------------------- #


class SAMModule(nn.Module):
    """SAM: For each query, find top-K salient encoder positions and resample.

    1. Score each encoder token by its cosine similarity to each query.
    2. Take top-K positions per query.
    3. Return the gathered encoder features as aligned references.
    """

    def __init__(self, d_model: int, top_k: int = 4) -> None:
        super().__init__()
        self.top_k = top_k
        # Project queries and encoder memory to same space for similarity scoring
        self.q_proj = nn.Linear(d_model, d_model)
        self.m_proj = nn.Linear(d_model, d_model)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        queries: (B, num_q, d_model) -- object queries
        memory: (B, HW, d_model) -- encoder output tokens
        Returns aligned_refs: (B, num_q, top_k, d_model)
        """
        B, num_q, D = queries.shape
        HW = memory.shape[1]

        q = F.normalize(self.q_proj(queries), dim=-1)  # (B, Q, D)
        m = F.normalize(self.m_proj(memory), dim=-1)  # (B, HW, D)

        # Similarity: (B, Q, HW)
        scores = torch.bmm(q, m.transpose(1, 2))

        # Top-K positions per query
        k = min(self.top_k, HW)
        topk_idx = torch.topk(scores, k, dim=-1)[1]  # (B, Q, k)

        # Gather encoder features at salient positions
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, Q, k, D)
        mem_exp = memory.unsqueeze(1).expand(-1, num_q, -1, -1)  # (B, Q, HW, D)
        aligned_refs = torch.gather(mem_exp, 2, topk_idx_exp)  # (B, Q, k, D)
        return aligned_refs


# --------------------------------------------------------------------------- #
#  SAM-DETR                                                                    #
# --------------------------------------------------------------------------- #


class SAMDETR(nn.Module):
    """SAM-DETR: DETR with Semantic-Aligned Matching auxiliary branch."""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_queries: int = 8,
        num_classes: int = 10,
        sam_top_k: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        # Backbone
        self.backbone = CompactBackbone(out_channels=d_model)

        # Input projection (1x1 conv to d_model, already set in backbone)
        self.input_proj = nn.Conv2d(d_model, d_model, 1)

        # Positional encoding (learned, max 100 spatial positions)
        self.pos_embed = nn.Embedding(100, d_model)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # SAM module: semantic-aligned matching
        self.sam = SAMModule(d_model, top_k=sam_top_k)
        # SAM projection: aggregate aligned refs -> per-query context
        self.sam_proj = nn.Linear(sam_top_k * d_model, d_model)

        # Object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, dropout=0.0, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # Prediction heads
        self.class_head = nn.Linear(d_model, num_classes + 1)  # +1 for no-object
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        B = x.shape[0]

        # Backbone -> feature map
        feat = self.backbone(x)  # (B, d_model, H', W')
        feat = self.input_proj(feat)  # (B, d_model, H', W')
        H, W = feat.shape[2], feat.shape[3]
        HW = H * W

        # Flatten spatial dims -> token sequence
        feat_flat = feat.flatten(2).transpose(1, 2)  # (B, HW, d_model)

        # Add positional embeddings
        pos_idx = torch.arange(min(HW, 100), device=x.device)
        pos = self.pos_embed(pos_idx).unsqueeze(0)  # (1, HW, d_model) [clipped to 100]
        n_tok = pos.shape[1]
        feat_flat_for_pos = feat_flat[:, :n_tok, :]
        memory = self.encoder(feat_flat_for_pos + pos)  # (B, HW, d_model)

        # Object queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, d_model)

        # SAM: find semantically aligned encoder features for each query
        aligned_refs = self.sam(queries, memory)  # (B, Q, k, d_model)
        # Flatten k references and project to query size
        B2, Q, k, D = aligned_refs.shape
        sam_context = self.sam_proj(aligned_refs.reshape(B2, Q, k * D))  # (B, Q, d_model)

        # Augment queries with SAM context (the alignment step)
        queries_aligned = queries + sam_context

        # Transformer decoder
        out = self.decoder(queries_aligned, memory)  # (B, Q, d_model)

        # Prediction heads -- return class logits averaged over queries for single output
        class_logits = self.class_head(out)  # (B, Q, num_classes+1)
        return class_logits.mean(dim=1)  # (B, num_classes+1)


# --------------------------------------------------------------------------- #
#  Wrapper & menagerie interface                                               #
# --------------------------------------------------------------------------- #


class _Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = SAMDETR()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_sam_detr_r50() -> nn.Module:
    return _Wrapper()


def example_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "SAM-DETR (semantic-aligned matching DETR with salient resampling)",
        "build_sam_detr_r50",
        "example_input",
        "2022",
        "DC",
    ),
]
