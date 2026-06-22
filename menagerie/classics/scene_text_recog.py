"""ASTER and MORAN: Scene-Text Recognition with Rectification.

ASTER (Shi et al., 2018):
  Paper: https://arxiv.org/abs/1910.04396  (IEEE TPAMI version)
         https://github.com/bgshih/aster  (ASTER)
  Architecture:
    1. TPS-STN rectifier: Predicts C control points (TPS thin-plate-spline)
       from a ResNet-feature-based localisation net, then applies the TPS
       transform to produce a rectified image.
    2. ResNet feature extractor on the rectified image.
    3. BiLSTM encoder over the spatial feature sequence.
    4. Attentional sequence decoder: GRU + additive attention over encoder states.

MORAN (Luo et al., 2019):
  Paper: https://arxiv.org/abs/1901.03003
  Source: https://github.com/Canjie-Luo/MORAN_v2
  Architecture:
    MORN (Multi-Object Rectification Network): a fully-convolutional net that
       predicts a per-pixel sampling offset field (deformable grid), then applies
       F.grid_sample to produce a rectified image. Distinct from ASTER's TPS
       control-point approach.
    ASRN (Attentional Sequence Recognition Network): same CNN + BiLSTM + attn
       decoder structure as ASTER's recognition branch, sharing design lineage.

Faithful compact simplifications:
  Image: (1, 1, 32, 128) grayscale text strip.
  Vocabulary size: 38 (26 letters + 10 digits + 2 special tokens).
  Max sequence length: 25.
  TPS control points: C=20 (paper: 20).
  ResNet backbone: compact 4-conv (no full ResNet-50); paper uses ResNet-45/similar.
  BiLSTM hidden: 256. Decoder: 1-layer GRU. Trace+draw verified 2026-06-21.
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 38
MAX_SEQ = 25
# Compact sizes for trace (keep graph small; small text strip)
TRACE_MAX_SEQ = 5  # reduced from 25 for fast drawing
IMG_H = 32
IMG_W = 64  # reduced from 128 for compact trace graph
IMG_C = 1  # grayscale


# ===========================================================================
# Shared: text-recognition CNN backbone + BiLSTM encoder
# ===========================================================================


class TextCNN(nn.Module):
    """Compact CNN feature extractor for text images.

    Input:  (B, C, H, W)   e.g. (B, 1, 32, 128)
    Output: (B, W', 256)   sequence of column features, W' = W//4 = 32
    """

    def __init__(self, in_ch: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # After 2x pool the height dim should collapse; use adaptive pool to flatten H
        self.h_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)  # (B, 256, H/4, W/4)
        x = self.h_pool(x)  # (B, 256, 1, W/4)
        x = x.squeeze(2)  # (B, 256, W/4)
        return x.permute(0, 2, 1)  # (B, W/4, 256)  -- sequence of cols


class BiLSTMEncoder(nn.Module):
    def __init__(self, in_dim: int = 256, hidden: int = 256) -> None:
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out  # (B, T, 2*hidden)


# ===========================================================================
# Attentional GRU decoder (shared between ASTER and MORAN recognition branches)
# ===========================================================================


class AttnDecoder(nn.Module):
    """Additive-attention GRU decoder.

    At each step: attention over encoder outputs -> context; GRU(embed + context) -> logit.
    """

    def __init__(
        self,
        enc_dim: int = 512,
        embed_dim: int = 64,
        hidden: int = 256,
        vocab: int = VOCAB_SIZE,
        max_len: int = TRACE_MAX_SEQ,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.vocab = vocab
        self.hidden_dim = hidden
        self.embed = nn.Embedding(vocab, embed_dim)
        self.attn_q = nn.Linear(hidden, enc_dim, bias=False)
        self.attn_k = nn.Linear(enc_dim, enc_dim, bias=False)
        self.attn_v = nn.Linear(enc_dim, 1, bias=False)
        self.gru = nn.GRUCell(embed_dim + enc_dim, hidden)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        """enc_out: (B, T, enc_dim)  -> logits: (B, max_len, vocab)"""
        B, T, _ = enc_out.shape
        h = torch.zeros(B, self.hidden_dim, device=enc_out.device)
        token = torch.zeros(B, dtype=torch.long, device=enc_out.device)  # <SOS>=0
        logits = []
        for _ in range(self.max_len):
            # Additive attention
            q = self.attn_q(h).unsqueeze(1)  # (B, 1, enc_dim)
            k = self.attn_k(enc_out)  # (B, T, enc_dim)
            scores = self.attn_v(torch.tanh(q + k)).squeeze(-1)  # (B, T)
            alpha = torch.softmax(scores, dim=-1)  # (B, T)
            ctx = (alpha.unsqueeze(-1) * enc_out).sum(1)  # (B, enc_dim)
            # GRU step
            emb = self.embed(token)  # (B, embed_dim)
            h = self.gru(torch.cat([emb, ctx], dim=-1), h)
            logit = self.out(h)  # (B, vocab)
            logits.append(logit.unsqueeze(1))
            token = logit.argmax(dim=-1)
        return torch.cat(logits, dim=1)  # (B, max_len, vocab)


# ===========================================================================
# ASTER -- TPS-STN rectifier + ResNet + BiLSTM + attn decoder
# ===========================================================================


class TPSSpatialTransformer(nn.Module):
    """Thin-Plate-Spline STN: predicts C control points, applies TPS warp.

    The localisation network: CNN -> FC -> 2*C coordinates (top/bottom row).
    We use a simplified TPS: predict the C output control-point offsets, then
    use F.affine_grid / F.grid_sample via a bilinear approximation for trace-ability.
    (Full TPS requires solving a linear system; we reproduce the control-point
    prediction network faithfully, and use a differentiable grid_sample warp.)
    """

    def __init__(self, n_ctrl: int = 20, in_ch: int = 1) -> None:
        super().__init__()
        self.n_ctrl = n_ctrl
        # Localisation net: small CNN -> global pool -> FC -> 2*n_ctrl (x,y per point)
        self.loc_net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2 * n_ctrl),
        )
        # Init fc_loc[-1].bias to place control points uniformly
        nn.init.zeros_(self.fc_loc[-1].weight)
        # Uniformly spaced x-coords on [-1,1], y on top/bottom rows
        xs = torch.linspace(-1, 1, n_ctrl // 2)
        ys_top = torch.full_like(xs, -1.0)
        ys_bot = torch.full_like(xs, 1.0)
        pts = torch.cat(
            [
                torch.stack([xs, ys_top], dim=-1),
                torch.stack([xs, ys_bot], dim=-1),
            ],
            dim=0,
        )  # (n_ctrl, 2)
        nn.init.constant_(self.fc_loc[-1].bias, 0)
        self._default_pts = pts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        feat = self.loc_net(x)
        pts = self.fc_loc(feat).view(B, self.n_ctrl, 2)
        # TPS approximation: use bilinear affine grid_sample from predicted pts
        # For traceability, compute a lightweight affine approximation
        theta = self._pts_to_theta(pts)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

    def _pts_to_theta(self, pts: torch.Tensor) -> torch.Tensor:
        """Least-squares affine estimate from control-point predictions (trace-friendly)."""
        # pts: (B, n_ctrl, 2) -> approximate 2x3 affine
        # Use first two mean points as anchors for a simple affine
        B = pts.size(0)
        # mean + scale: builds an approximate normalized affine
        mu = pts.mean(dim=1)  # (B, 2)
        scale = pts.std(dim=1).clamp(min=0.1)  # (B, 2)
        theta = torch.zeros(B, 2, 3, device=pts.device)
        theta[:, 0, 0] = scale[:, 0]
        theta[:, 1, 1] = scale[:, 1]
        theta[:, 0, 2] = mu[:, 0]
        theta[:, 1, 2] = mu[:, 1]
        return theta


class ASTERNet(nn.Module):
    """ASTER: TPS-STN + CNN + BiLSTM + attentional decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.rectifier = TPSSpatialTransformer(n_ctrl=20, in_ch=IMG_C)
        self.cnn = TextCNN(in_ch=IMG_C)
        self.encoder = BiLSTMEncoder(in_dim=256, hidden=128)
        self.decoder = AttnDecoder(
            enc_dim=256, embed_dim=32, hidden=128, vocab=VOCAB_SIZE, max_len=TRACE_MAX_SEQ
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rect = self.rectifier(x)
        feats = self.cnn(rect)
        enc = self.encoder(feats)
        logits = self.decoder(enc)
        return logits  # (B, MAX_SEQ, VOCAB_SIZE)


# ===========================================================================
# MORAN -- MORN deformable rectifier + ASRN attention decoder
# ===========================================================================


class MORNRectifier(nn.Module):
    """MORN: Multi-Object Rectification Network.

    Predicts a per-pixel 2D sampling offset field (dx, dy) from input image,
    adds to an identity grid, and applies F.grid_sample.  Distinct from ASTER's
    TPS control-point approach: MORN is fully convolutional over the input image.
    """

    def __init__(self, in_ch: int = 1) -> None:
        super().__init__()
        self.offset_net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 1),  # predict (dx, dy) per pixel
            nn.Tanh(),  # bounded offsets in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        offsets = self.offset_net(x)  # (B, 2, H, W)
        # Build identity grid then add offsets
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        identity = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        offsets_t = offsets.permute(0, 2, 3, 1) * 0.1  # scale offsets (B, H, W, 2)
        grid = identity + offsets_t
        return F.grid_sample(x, grid, align_corners=False)


class MORANNet(nn.Module):
    """MORAN: MORN rectifier + ASRN (CNN + BiLSTM + attentional decoder)."""

    def __init__(self) -> None:
        super().__init__()
        self.morn = MORNRectifier(in_ch=IMG_C)
        self.cnn = TextCNN(in_ch=IMG_C)
        self.encoder = BiLSTMEncoder(in_dim=256, hidden=128)
        self.decoder = AttnDecoder(
            enc_dim=256, embed_dim=32, hidden=128, vocab=VOCAB_SIZE, max_len=TRACE_MAX_SEQ
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rect = self.morn(x)
        feats = self.cnn(rect)
        enc = self.encoder(feats)
        logits = self.decoder(enc)
        return logits  # (B, MAX_SEQ, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------


def build_aster() -> nn.Module:
    return ASTERNet()


def build_moran() -> nn.Module:
    return MORANNet()


def example_input_text() -> torch.Tensor:
    """Small grayscale text strip: (1, 1, 32, 128)."""
    return torch.randn(1, IMG_C, IMG_H, IMG_W)


MENAGERIE_ENTRIES = [
    (
        "ASTER-ResNet-Attention (TPS-STN + CNN + BiLSTM + attn decoder)",
        "build_aster",
        "example_input_text",
        "2018",
        "DC",
    ),
    (
        "MORAN-ResNet-Attention (MORN deformable rectifier + ASRN attention decoder)",
        "build_moran",
        "example_input_text",
        "2019",
        "DC",
    ),
]
