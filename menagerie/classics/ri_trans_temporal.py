"""Temporal / sequence architectures: VIBE, VAC-CSLR, VIME, UniVLA.

Four distinct families that all turn a per-frame / per-row feature stream into a
task output via a sequence module + a specialized head.

============================================================================
VIBE: Video Inference for Human Body pose and shape Estimation.
  Kocabas et al., CVPR 2020. Paper: https://arxiv.org/abs/1912.05656
  Source: https://github.com/mkocabas/VIBE
  Distinctive primitive: per-frame CNN image features (2048-d) -> a BIDIRECTIONAL
  GRU temporal encoder (with a residual connection around it) -> an iterative
  SMPL-parameter regressor (pose theta + shape beta + camera) that refines an
  initial mean-pose estimate over a few feedback iterations. Adversarial motion
  prior is a training-time discriminator (omitted from the forward).

============================================================================
VAC-CSLR: Visual Alignment Constraint for continuous sign language recognition.
  Min et al., ICCV 2021. Paper: https://arxiv.org/abs/2104.02330
  Source: https://github.com/ycmin95/VAC_CSLR
  Distinctive primitive: a 2D-CNN per-frame visual extractor -> a 1D-temporal
  conv block -> a BiLSTM sequence model -> a CTC classifier over gloss
  vocabulary. The Visual Alignment Constraint adds an auxiliary CTC head ON the
  pre-LSTM visual features (visual enhancement / alignment), so two CTC logits
  streams (visual + sequential) are produced.

============================================================================
VIME: Value Imputation and Mask Estimation self-supervised learning for tabular.
  Yoon et al., NeurIPS 2020. Paper: https://arxiv.org/abs/2006.07733-ish (NeurIPS)
  Source: https://github.com/jsyoon0823/VIME
  Distinctive primitive: an encoder maps a (corrupted) tabular row to a latent;
  the self-supervised pretext has TWO heads off that latent -- a MASK ESTIMATOR
  (predict which features were masked, sigmoid per feature) and a FEATURE
  ESTIMATOR (reconstruct the original feature values). This dual-head pretext is
  the VIME signature.

============================================================================
UniVLA: a unified vision-language-action autoregressive model.
  BAAI, 2025. (Unified VLA token stream: vision + language + action.)
  Distinctive primitive: a single causal LLM (Llama-style decoder transformer)
  consumes an interleaved token stream of vision + language + action tokens and
  autoregressively predicts the next token; an action head reads the final
  hidden state to emit a continuous action vector. Reproduced with a compact
  Llama-style decoder + action head.

All four are faithful compact random-init reimplementations; small widths and
short sequences so the unrolled trace draws quickly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ===========================================================================
# VIBE
# ===========================================================================
class _SMPLRegressor(nn.Module):
    """Iterative SMPL parameter regressor (pose + shape + camera) with feedback."""

    def __init__(self, in_dim: int, n_iter: int = 3, out_dim: int = 85) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim + out_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(256, out_dim)
        self.register_buffer("init_params", torch.zeros(1, out_dim))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B*T, in_dim)
        params = self.init_params.expand(feat.shape[0], -1)
        for _ in range(self.n_iter):
            h = torch.cat([feat, params], dim=-1)
            h = self.drop(torch.relu(self.fc1(h)))
            h = self.drop(torch.relu(self.fc2(h)))
            params = params + self.head(h)  # residual refinement
        return params


class VIBE(nn.Module):
    """Per-frame feature -> bi-GRU temporal encoder -> iterative SMPL regressor."""

    def __init__(self, feat_dim: int = 2048, hidden: int = 256, n_layers: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(feat_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=n_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden * 2, hidden)
        self.regressor = _SMPLRegressor(hidden, n_iter=3, out_dim=85)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, feat_dim)
        b, t, _ = x.shape
        h = self.proj(x)
        g, _ = self.gru(h)
        g = self.linear(g) + h  # residual around the temporal encoder
        params = self.regressor(g.reshape(b * t, -1))
        return params.reshape(b, t, -1)


def build_vibe() -> nn.Module:
    """Build a compact VIBE (bi-GRU temporal SMPL pose/shape regressor)."""
    return VIBE(feat_dim=2048, hidden=256, n_layers=2)


def example_input_vibe() -> torch.Tensor:
    """Example per-frame CNN feature sequence ``(1, 16, 2048)`` for VIBE."""
    return torch.randn(1, 16, 2048)


# ===========================================================================
# VAC-CSLR
# ===========================================================================
class _TinyVisualExtractor(nn.Module):
    """Small 2D-CNN per-frame feature extractor."""

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class VAC_CSLR(nn.Module):
    """CNN -> 1D-temporal conv -> BiLSTM CTC with a visual-alignment aux CTC head."""

    def __init__(self, num_classes: int = 100, feat: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.visual = _TinyVisualExtractor(feat)
        self.temporal = nn.Sequential(
            nn.Conv1d(feat, hidden, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.visual_ctc = nn.Linear(hidden, num_classes)  # alignment-constraint aux head
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, bidirectional=True, batch_first=True)
        self.seq_ctc = nn.Linear(hidden * 2, num_classes)  # main sequential head

    def forward(self, frames: torch.Tensor):
        # frames: (B, T, 3, H, W)
        b, t = frames.shape[0], frames.shape[1]
        feats = self.visual(frames.reshape(b * t, *frames.shape[2:]))
        feats = feats.reshape(b, t, -1).transpose(1, 2)  # (B, feat, T)
        tconv = self.temporal(feats).transpose(1, 2)  # (B, T, hidden)
        visual_logits = self.visual_ctc(tconv)
        seq, _ = self.lstm(tconv)
        seq_logits = self.seq_ctc(seq)
        return visual_logits, seq_logits


class _VACWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, frames: torch.Tensor):
        return self.model(frames)


def build_vac_cslr() -> nn.Module:
    """Build a compact VAC-CSLR (CNN+1D-temporal+BiLSTM CTC sign-language recognizer)."""
    return _VACWrapper(VAC_CSLR(num_classes=100, feat=128, hidden=256))


def example_input_vac() -> torch.Tensor:
    """Example sign-language clip ``(1, 8, 3, 64, 64)`` (B, T, C, H, W)."""
    return torch.randn(1, 8, 3, 64, 64)


# ===========================================================================
# VIME
# ===========================================================================
class VIME(nn.Module):
    """Self-supervised tabular pretext: encoder + mask-estimator + feature-estimator."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mask_estimator = nn.Linear(hidden_dim, input_dim)  # which features were masked
        self.feature_estimator = nn.Linear(hidden_dim, input_dim)  # reconstruct values

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        mask_hat = torch.sigmoid(self.mask_estimator(z))
        feat_hat = self.feature_estimator(z)
        return mask_hat, feat_hat


def build_vime() -> nn.Module:
    """Build a compact VIME (self-supervised tabular mask + feature estimator)."""
    return VIME(input_dim=32, hidden_dim=128)


def example_input_vime() -> torch.Tensor:
    """Example tabular batch ``(16, 32)`` for VIME."""
    return torch.randn(16, 32)


# ===========================================================================
# UniVLA
# ===========================================================================
class _CausalDecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class UniVLA(nn.Module):
    """Unified vision-language-action causal LLM + action head."""

    def __init__(
        self, vocab: int = 32000, dim: int = 128, n_layers: int = 2, heads: int = 4, n_act: int = 7
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, 512, dim))
        self.blocks = nn.ModuleList([_CausalDecoderBlock(dim, heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab)
        self.act_head = nn.Linear(dim, n_act)

    def forward(self, input_ids: torch.Tensor):
        t = input_ids.shape[1]
        x = self.embed(input_ids) + self.pos[:, :t]
        mask = torch.triu(torch.full((t, t), float("-inf"), device=input_ids.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        action = self.act_head(x[:, -1, :])  # continuous action from last token
        return logits, action


class _UniVLAWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor):
        return self.model(input_ids)


def build_univla() -> nn.Module:
    """Build a compact UniVLA (unified vision-language-action causal transformer)."""
    return _UniVLAWrapper(UniVLA(vocab=32000, dim=128, n_layers=2, heads=4, n_act=7))


def example_input_univla() -> torch.Tensor:
    """Example interleaved token-id stream ``(1, 48)`` int64 for UniVLA."""
    return torch.randint(0, 32000, (1, 48), dtype=torch.int64)


MENAGERIE_ENTRIES = [
    (
        "VIBE (bi-GRU temporal encoder + iterative SMPL pose regressor)",
        "build_vibe",
        "example_input_vibe",
        "2020",
        "DC",
    ),
    (
        "VAC-CSLR (CNN + 1D-temporal + BiLSTM CTC sign-language, visual-alignment aux head)",
        "build_vac_cslr",
        "example_input_vac",
        "2021",
        "DC",
    ),
    (
        "VIME (self-supervised tabular: mask-estimator + feature-estimator pretext)",
        "build_vime",
        "example_input_vime",
        "2020",
        "DC",
    ),
    (
        "UniVLA (unified vision-language-action autoregressive transformer)",
        "build_univla",
        "example_input_univla",
        "2025",
        "DC",
    ),
]
