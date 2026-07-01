"""Menagerie batch w4a2: transformer-based lane-shape prediction.

Sources checked (reference only; no cloning, no pip installs):
  - LSTR / "E2ELSPTRs" (cand_00290, "LaneTransformer" in the catalog queue):
    Liu & Yuan, WACV 2021, "End-to-end Lane Shape Prediction with
    Transformers". Paper https://arxiv.org/abs/2011.04233, official source
    https://github.com/liuruijin17/E2ELSPTRs (models/py_utils/kp.py). A
    ResNet-style backbone extracts a low-resolution HxWxC feature map that is
    flattened into a length-HW token sequence and fed, with learned
    positional embeddings, into a 2-layer transformer encoder. A 2-layer
    transformer decoder attends to N learned "lane query" embeddings (zero
    input tokens + learned positional embeddings, one query slot per
    potential lane instance) against the encoded image tokens, mirroring
    DETR's set-prediction decoder. Each of the N decoder output tokens feeds
    three small FFN heads: (1) a 2-way classification head (lane vs.
    background), (2) a per-lane FFN regressing 4 lane-specific curve
    parameters, and (3) a shared-parameters FFN whose N per-query outputs are
    averaged into one global set of parameters shared by every lane in the
    image (the paper's approximation of a shared camera/road geometry term).
    The per-lane and shared parameters together parameterize the paper's
    projected lane-shape model (a cubic road curve X = kZ^3 + mZ^2 + nZ + b
    projected onto the image as u = k'/v^2 + m'/v + n' + b'*v), which this
    module reproduces structurally: N lane query slots, a shared-vs-per-lane
    parameter split, and the class head, exactly matching LSTR's namesake
    "transformers directly regress explicit lane-curve parameters" design.
    (See also patrick-llgc.github.io paper-notes summary of the same repo.)

All models below are compact, faithfully-reimplemented-from-scratch nn.Modules
with random init and small dims for TorchLens architecture-catalog tracing
(not a trained-weights zoo).

Skipped candidates from this batch (already faithfully present in the
catalog under a different family name -- see build report for verification):
  - cand_00181 "Gated Convolution Inpainting" == DeepFillv2, already present
    as ``DeepFillv2Generator`` / ``build_deepfillv2_gated_generator`` in
    ``menagerie/classics/ri_vision_b_gan.py``.
  - cand_00232 "3DSSD" already present as ``Point3DSSD`` /
    ``build_3dssd`` in ``menagerie/classics/openmmlab_reimpl0.py``.
  - cand_00265 "Cylinder3D" already present as ``Cylinder3DCompact`` /
    ``build_cylinder3d`` in ``menagerie/classics/dreimpl_1_openmmlab.py``.
  - cand_00283 "ImVoxelNet" already present via ``build_imvoxelnet`` in
    ``menagerie/classics/dreimpl_3_openmmlab.py``.
  - cand_00287 "LaneGAN" (catalog notes identify this as RESA, the Recurrent
    Feature-Shift Aggregator for lane detection) already present as
    ``RESA`` / ``build_resa`` in ``menagerie/classics/lanedet.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ============================================================
# LSTR / E2ELSPTRs -- transformer-based explicit lane-curve regression
# ============================================================


class _ResStem(nn.Module):
    """Compact convolutional backbone producing a low-resolution feature map."""

    def __init__(self, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, out_ch // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 2, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTR(nn.Module):
    """LSTR: transformer encoder-decoder that directly regresses lane curves.

    A CNN backbone yields a flattened token sequence that a small transformer
    encoder contextualizes; a transformer decoder cross-attends N learned
    lane-query embeddings against the encoded tokens (DETR-style set
    prediction). Each query slot predicts a lane-vs-background class plus
    per-lane curve parameters, and a separate shared-parameters FFN averages
    its N outputs into one set of parameters common to every lane -- the
    paper's split between lane-specific and shared (camera/road) curve terms.
    """

    def __init__(
        self,
        backbone_ch: int = 16,
        hidden: int = 16,
        n_queries: int = 6,
        n_shared_params: int = 4,
        n_lane_params: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = _ResStem(backbone_ch)
        self.input_proj = nn.Conv2d(backbone_ch, hidden, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, hidden) * 0.02)
        self.query_embed = nn.Embedding(n_queries, hidden)
        self.transformer = nn.Transformer(
            d_model=hidden,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=hidden * 2,
            batch_first=True,
        )
        self.class_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 2)
        )
        self.lane_param_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, n_lane_params)
        )
        self.shared_param_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, n_shared_params)
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(image)
        b, _c, h, w = feat.shape
        tokens = self.input_proj(feat).flatten(2).transpose(1, 2)  # (B, HW, hidden)
        n_tok = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :n_tok, :]
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        hs = self.transformer(tokens, queries)  # (B, n_queries, hidden)

        logits = self.class_head(hs)  # (B, n_queries, 2)
        lane_params = self.lane_param_head(hs)  # (B, n_queries, n_lane_params) per-lane terms
        shared_params = self.shared_param_head(hs).mean(
            dim=1
        )  # (B, n_shared_params) shared road/camera terms

        return {
            "logits": logits,
            "lane_params": lane_params,
            "shared_params": shared_params,
            "feat_hw": torch.tensor([h, w]),
        }


def build_lstr() -> nn.Module:
    """Build a small LSTR (E2ELSPTRs) transformer lane-shape predictor."""
    return LSTR(backbone_ch=16, hidden=16, n_queries=6, n_shared_params=4, n_lane_params=4).eval()


def example_input_lstr() -> torch.Tensor:
    """RGB road image ``(1, 3, 64, 64)`` (small, to keep the token count tiny)."""
    return torch.randn(1, 3, 64, 64)


# ============================================================
# Registry
# ============================================================

MENAGERIE_ENTRIES = [
    ("LaneTransformer", "build_lstr", "example_input_lstr", "2021", "VIS"),
]
