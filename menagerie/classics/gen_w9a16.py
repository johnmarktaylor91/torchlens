"""Compact faithful reimplementations for build_queue rows 97-102 (W9A16).

Sources checked (repo/paper browsed via web search / arXiv, no clone/pip-install):
  - ArchesWeather: Couairon, Singh, Charantonis, Lessig, Monteleoni,
    "ArchesWeather: An efficient AI weather forecasting model at 1.5deg
    resolution", arXiv:2405.14527 (INRIA/ECMWF; official repo
    ``gcouairon/ArchesWeather``, since moved to ``INRIA/geoarches``).
    Distinctive mechanism: unlike Pangu-Weather's full 3D local
    (window) attention over (level, height, width), ArchesWeather
    factorizes attention -- a 3D Swin-style U-Net stage applies 2D
    local *windowed* attention only within the horizontal (height,
    width) plane per pressure level, and a separate *column-wise*
    global attention module lets every pressure level at a given
    horizontal location attend to every other level (full vertical
    mixing at comparatively low cost since the column length is tiny
    next to the horizontal grid). Reproduced here as a compact block
    alternating (a) horizontal windowed multi-head self-attention
    applied independently per level and (b) column (vertical) global
    multi-head self-attention applied independently per horizontal
    position, inside a small encoder-decoder U-Net with a downsample/
    upsample stage (patch-merge / patch-expand), matching the
    reference's Swin-U-Net skeleton at a compact scale.
  - AtmoRep: Lessig et al., "AtmoRep: A stochastic model of atmosphere
    dynamics using large scale representation learning",
    arXiv:2308.13280. Official repo ``clessig/atmorep``. Distinctive
    mechanism: a "Multiformer" -- one encoder-decoder transformer *per
    physical field* (temperature, u-wind, v-wind, ...), with per-field
    encoders coupled to every other field's encoder via cross-
    attention (so fields interact), per-field decoders that cross-
    attend only back to their own field's encoder (via U-Net-style
    skip connections), and an *ensemble* of prediction heads (several
    independent linear heads) per field that each reconstruct the
    masked input tokens, giving a stochastic ensemble of
    reconstructions rather than one deterministic prediction.
    Reproduced here as a compact 2-field Multiformer: per-field token
    self-attention encoders, an explicit cross-field cross-attention
    coupling layer, per-field decoders that cross-attend to their own
    encoder's skip output, and a small ensemble (multiple heads) of
    linear reconstruction heads per field.
  - CACo: Mall, Hariharan, Bala, "Change-Aware Sampling and Contrastive
    Learning for Satellite Images", CVPR 2023, arXiv:2211.11594.
    Official repo ``utkarshmall13/CACo``. Distinctive mechanism: a
    ResNet encoder embeds pairs of satellite images sampled at
    different temporal gaps (a *short-term* pair, expected to look
    almost identical, and a *long-term* pair, which may have
    genuinely changed); the CACo loss modulates the standard
    contrastive (InfoNCE-style) attraction/repulsion target for each
    pair by a *change score* -- a small head predicts how "changed" a
    pair is directly from the encoder's own feature difference, and
    that predicted change score is fed back to *rescale the target
    positive-pair similarity* (heavily-changed long-term pairs are
    pulled less tightly together than unchanged short-term pairs),
    rather than treating all temporal positive pairs identically like
    plain SimCLR/SeCo. Reproduced here as a compact ResNet-style CNN
    encoder + projection head shared across both images of a pair,
    plus a change-magnitude head over the pooled feature difference
    that outputs the change-conditioned target-similarity scalar used
    to reweight the contrastive loss (the loss itself is computed
    downstream of this trainable module, mirroring how the reference
    keeps change-scoring as a small trainable head on top of the
    shared encoder).
  - CDNetV2: Guo, Yang, Cui, "CDnetV2: CNN-Based Cloud Detection for
    Remote Sensing Imagery With Cloud-Snow Coexistence", IEEE TGRS
    2021. Official repo ``nkszjx/CDnetV2-pytorch-master``. Distinctive
    mechanism: an encoder-decoder CNN with two named fusion modules --
    an Adaptive Feature Fusing Module (AFFM) that fuses adjacent
    -resolution encoder feature maps through a channel-attention gate
    (global-pooled per-channel weighting), a spatial-attention gate
    (per-pixel weighting from a 1x1 conv over the concatenated maps),
    and a channel-attention *refinement* pass on the fused result; and
    High-level Semantic Information Guidance Flows (HSIGF) that inject
    a once-computed high-level semantic feature (from the deepest
    encoder stage, upsampled) into every decoder stage so shallow
    decoder layers stay aware of *where* the cloud objects are.
    Reproduced here as a compact 3-stage encoder, an AFFM module
    (channel-gate + spatial-gate + channel-refine) fusing each pair of
    adjacent encoder stages, and an HSIGF broadcast of the deepest
    encoder feature (via a small high-level feature fusing conv) added
    into every decoder stage before the final per-pixel cloud/snow/
    background segmentation head.
  - ClimODE: Verma, Heinonen, Garg, "ClimODE: Climate and Weather
    Forecasting with Physics-informed Neural ODEs", ICLR 2024 (oral),
    arXiv:2404.10024. Official repo ``Aalto-QuML/ClimODE``.
    Distinctive mechanism: weather evolution is modeled as a
    *continuity-equation* advection PDE, ``d(u)/dt = -div(u * v)``,
    where a learned flow-velocity field ``v(x, y)`` (combining local
    convolutions with a long-range global-attention correction) is
    used to *semi-Lagrangian advect* the current quantity field by
    resampling it along the backward-warped grid ``x - v * dt``
    (bilinear ``grid_sample``, i.e. literally moving mass along the
    predicted flow rather than convolving it away), after which a
    separate Gaussian *emission* network predicts a local bias and
    variance correction (source/sink term) that is added post-
    advection. Reproduced here as one ODE-step module: a flow-velocity
    head (local conv features + a global-attention long-range term)
    predicting a per-pixel 2D velocity, a differentiable semi-
    Lagrangian advection step via ``F.grid_sample`` on a normalized
    backward-warped sampling grid, and a Gaussian emission head
    (bias + log-variance) added to the advected field -- matching the
    reference's velocity + advection + emission decomposition.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# ArchesWeather: 2D horizontal windowed attention + column-wise (vertical)
# global attention, alternated inside a small Swin-style U-Net
# ---------------------------------------------------------------------------


class _HorizontalWindowAttention(nn.Module):
    """2D local windowed multi-head self-attention within the horizontal plane.

    Applied identically (weight-shared) to every pressure level: each
    level's ``(H, W)`` feature map is split into non-overlapping
    windows and attention is computed only within a window, matching
    the horizontal half of ArchesWeather's factorized 3D attention.
    """

    def __init__(self, dim: int, window_size: int = 4, num_heads: int = 4) -> None:
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply windowed horizontal attention.

        Parameters
        ----------
        x : Tensor
            Level-stacked feature map, shape ``(batch, levels, H, W, dim)``.

        Returns
        -------
        Tensor
            Same shape as ``x``.
        """

        b, lev, h, w, dim = x.shape
        ws = self.window_size
        x_flat = x.reshape(b * lev, h, w, dim)
        x_win = x_flat.reshape(b * lev, h // ws, ws, w // ws, ws, dim)
        x_win = x_win.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, dim)
        normed = self.norm(x_win)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        out = x_win + attn_out
        out = out.reshape(b * lev, h // ws, w // ws, ws, ws, dim)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(b * lev, h, w, dim)
        return out.reshape(b, lev, h, w, dim)


class _ColumnAttention(nn.Module):
    """Column-wise (vertical) global multi-head self-attention.

    Every pressure level at a fixed horizontal location attends to
    every other level -- full vertical mixing, cheap because the
    column length (number of pressure levels) is small.
    """

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply global column (vertical) attention.

        Parameters
        ----------
        x : Tensor
            Level-stacked feature map, shape ``(batch, levels, H, W, dim)``.

        Returns
        -------
        Tensor
            Same shape as ``x``.
        """

        b, lev, h, w, dim = x.shape
        x_col = x.permute(0, 2, 3, 1, 4).reshape(b * h * w, lev, dim)
        normed = self.norm(x_col)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        out = x_col + attn_out
        return out.reshape(b, h, w, lev, dim).permute(0, 3, 1, 2, 4)


class _ArchesBlock(nn.Module):
    """One horizontal-window + column-attention pair with an MLP."""

    def __init__(self, dim: int, window_size: int = 4, num_heads: int = 4) -> None:
        super().__init__()
        self.horiz = _HorizontalWindowAttention(dim, window_size, num_heads)
        self.column = _ColumnAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run one horizontal-then-column attention block."""

        x = self.horiz(x)
        x = self.column(x)
        return x + self.mlp(x)


class ArchesWeather(nn.Module):
    """Compact 3D Swin U-Net with factorized horizontal/column attention.

    Reproduces ``gcouairon/ArchesWeather``'s core idea: rather than
    full 3D local attention (Pangu-Weather), attention is factorized
    into cheap 2D horizontal windows plus a global column (vertical)
    pass, inside a tiny U-Net with one patch-merge downsample and one
    patch-expand upsample stage.
    """

    def __init__(
        self,
        n_vars: int = 4,
        n_levels: int = 4,
        dim: int = 16,
        window_size: int = 4,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(n_vars, dim)
        self.encoder_block = _ArchesBlock(dim, window_size, num_heads=4)
        self.downsample = nn.Linear(4 * dim, 2 * dim)
        self.bottleneck_block = _ArchesBlock(2 * dim, window_size=2, num_heads=4)
        self.upsample = nn.Linear(2 * dim, 4 * dim)
        self.decoder_block = _ArchesBlock(dim, window_size, num_heads=4)
        self.head = nn.Linear(dim, n_vars)
        self.n_levels = n_levels

    def forward(self, x: Tensor) -> Tensor:
        """Forecast the next pressure-level field stack.

        Parameters
        ----------
        x : Tensor
            Input atmospheric fields, shape
            ``(batch, levels, H, W, n_vars)``.

        Returns
        -------
        Tensor
            Predicted next-step fields, same shape as ``x``.
        """

        b, lev, h, w, _ = x.shape
        feat = self.patch_embed(x)
        feat = self.encoder_block(feat)
        skip = feat

        # patch-merge 2x2 downsample in the horizontal plane
        merged = feat.reshape(b, lev, h // 2, 2, w // 2, 2, -1)
        merged = merged.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, lev, h // 2, w // 2, -1)
        merged = self.downsample(merged)
        merged = self.bottleneck_block(merged)

        expanded = self.upsample(merged)
        expanded = expanded.reshape(b, lev, h // 2, w // 2, 2, 2, -1)
        expanded = expanded.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, lev, h, w, -1)

        feat = self.decoder_block(expanded + skip)
        return self.head(feat)


def build_archesweather() -> nn.Module:
    """Build a compact ArchesWeather factorized-attention forecaster.

    Returns
    -------
    nn.Module
        ``ArchesWeather`` in eval mode.
    """

    return ArchesWeather().eval()


def example_input_archesweather() -> Tensor:
    """Create example input for :func:`build_archesweather`.

    Returns
    -------
    Tensor
        Atmospheric field stack, shape ``(1, 4, 8, 8, 4)`` (batch,
        pressure levels, height, width, variables).
    """

    torch.manual_seed(0)
    return torch.randn(1, 4, 8, 8, 4)


# ---------------------------------------------------------------------------
# AtmoRep: "Multiformer" -- one encoder-decoder transformer per physical
# field, coupled across fields via cross-attention, ensemble prediction heads
# ---------------------------------------------------------------------------


class _FieldEncoder(nn.Module):
    """Per-field token self-attention encoder."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: Tensor) -> Tensor:
        """Self-attend over one field's masked token sequence."""

        normed = self.norm(tokens)
        attn_out, _ = self.self_attn(normed, normed, normed, need_weights=False)
        return tokens + attn_out


class _CrossFieldCoupling(nn.Module):
    """Cross-attention coupling between two fields' encoders."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, query_field: Tensor, other_field: Tensor) -> Tensor:
        """Let ``query_field`` attend into ``other_field``'s tokens."""

        q = self.norm_q(query_field)
        kv = self.norm_kv(other_field)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        return query_field + attn_out


class _FieldDecoder(nn.Module):
    """Per-field decoder cross-attending only to its own encoder's skip output."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, decoder_tokens: Tensor, encoder_skip: Tensor) -> Tensor:
        """Cross-attend the decoder tokens to the same field's encoder output."""

        q = self.norm_q(decoder_tokens)
        kv = self.norm_kv(encoder_skip)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        return decoder_tokens + attn_out


class AtmoRep(nn.Module):
    """Compact 2-field Multiformer with cross-field coupling and head ensembles.

    Reproduces ``clessig/atmorep``'s Multiformer design at a compact
    2-field scale: per-field self-attention encoders coupled by
    cross-field cross-attention, per-field decoders that cross-attend
    to their own field's (U-Net-style skip) encoder output, and a
    small ensemble of independent linear reconstruction heads per
    field over the masked tokens.
    """

    def __init__(self, dim: int = 16, num_heads: int = 4, ensemble_size: int = 3) -> None:
        super().__init__()
        self.encoder_a = _FieldEncoder(dim, num_heads)
        self.encoder_b = _FieldEncoder(dim, num_heads)
        self.couple_a_from_b = _CrossFieldCoupling(dim, num_heads)
        self.couple_b_from_a = _CrossFieldCoupling(dim, num_heads)
        self.decoder_a = _FieldDecoder(dim, num_heads)
        self.decoder_b = _FieldDecoder(dim, num_heads)
        self.heads_a = nn.ModuleList([nn.Linear(dim, 1) for _ in range(ensemble_size)])
        self.heads_b = nn.ModuleList([nn.Linear(dim, 1) for _ in range(ensemble_size)])

    def forward(self, tokens_a: Tensor, tokens_b: Tensor) -> tuple[Tensor, Tensor]:
        """Reconstruct masked tokens for two coupled physical fields.

        Parameters
        ----------
        tokens_a : Tensor
            Masked token embeddings for field A, shape
            ``(batch, n_tokens, dim)``.
        tokens_b : Tensor
            Masked token embeddings for field B, shape
            ``(batch, n_tokens, dim)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Ensemble reconstructions for field A and field B, each
            shape ``(batch, n_tokens, ensemble_size)``.
        """

        enc_a = self.encoder_a(tokens_a)
        enc_b = self.encoder_b(tokens_b)

        coupled_a = self.couple_a_from_b(enc_a, enc_b)
        coupled_b = self.couple_b_from_a(enc_b, enc_a)

        dec_a = self.decoder_a(coupled_a, enc_a)
        dec_b = self.decoder_b(coupled_b, enc_b)

        recon_a = torch.cat([head(dec_a) for head in self.heads_a], dim=-1)
        recon_b = torch.cat([head(dec_b) for head in self.heads_b], dim=-1)
        return recon_a, recon_b


def build_atmorep() -> nn.Module:
    """Build a compact 2-field AtmoRep Multiformer.

    Returns
    -------
    nn.Module
        ``AtmoRep`` in eval mode.
    """

    return AtmoRep().eval()


def example_input_atmorep() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_atmorep`.

    Returns
    -------
    tuple[Tensor, Tensor]
        Masked token embeddings for two physical fields, each shape
        ``(2, 12, 16)``.
    """

    torch.manual_seed(0)
    return torch.randn(2, 12, 16), torch.randn(2, 12, 16)


# ---------------------------------------------------------------------------
# CACo: shared ResNet-style encoder + change-magnitude head that produces the
# change-conditioned target similarity for temporal contrastive pairs
# ---------------------------------------------------------------------------


class _ResBlock(nn.Module):
    """Small residual convolutional block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply a residual conv-bn-relu-conv-bn block."""

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(x + out, inplace=True)


class CACo(nn.Module):
    """Change-aware contrastive encoder with a change-conditioned similarity head.

    Reproduces ``utkarshmall13/CACo``: a shared ResNet-style CNN
    encoder embeds both images of a temporal pair (short-term or
    long-term), a projection head maps pooled features to the
    contrastive embedding space, and a change-magnitude head reads the
    *encoder feature difference* between the two images to predict a
    scalar change score -- this predicted change score rescales the
    target positive-pair similarity used by the CACo contrastive loss
    (heavily-changed pairs are pulled together less strongly than
    near-identical pairs), rather than every temporal pair sharing one
    fixed target similarity.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 16, embed_dim: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.res1 = _ResBlock(base_channels)
        self.downsample = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.res2 = _ResBlock(base_channels * 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection_head = nn.Sequential(
            nn.Linear(base_channels * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.change_head = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels, 1),
            nn.Sigmoid(),
        )

    def _encode(self, x: Tensor) -> Tensor:
        feat = self.stem(x)
        feat = self.res1(feat)
        feat = self.downsample(feat)
        feat = self.res2(feat)
        return self.pool(feat).flatten(1)

    def forward(self, image_t0: Tensor, image_t1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Embed a temporal image pair and predict its change-conditioned target.

        Parameters
        ----------
        image_t0 : Tensor
            Earlier image in the temporal pair, shape
            ``(batch, 3, H, W)``.
        image_t1 : Tensor
            Later image in the temporal pair (short- or long-term
            gap), shape ``(batch, 3, H, W)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Contrastive embeddings for ``image_t0`` and ``image_t1``
            (each ``(batch, embed_dim)``), and the predicted
            change-conditioned target similarity in ``[0, 1]`` shape
            ``(batch, 1)``.
        """

        feat0 = self._encode(image_t0)
        feat1 = self._encode(image_t1)
        embed0 = self.projection_head(feat0)
        embed1 = self.projection_head(feat1)
        change_score = self.change_head(feat1 - feat0)
        target_similarity = 1.0 - change_score
        return embed0, embed1, target_similarity


def build_caco() -> nn.Module:
    """Build a compact CACo change-aware contrastive encoder.

    Returns
    -------
    nn.Module
        ``CACo`` in eval mode.
    """

    return CACo().eval()


def example_input_caco() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_caco`.

    Returns
    -------
    tuple[Tensor, Tensor]
        A temporal satellite-image pair, each shape ``(2, 3, 32, 32)``.
    """

    torch.manual_seed(0)
    return torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)


# ---------------------------------------------------------------------------
# CDNetV2: encoder-decoder with AFFM (channel+spatial fusion gate +
# channel-refinement) and HSIGF (deep semantic feature broadcast to decoder)
# ---------------------------------------------------------------------------


class _AFFM(nn.Module):
    """Adaptive Feature Fusing Module: channel-gate + spatial-gate + refine.

    Fuses a shallow and a deep encoder feature map (deep map upsampled
    to the shallow map's resolution first) through a channel-attention
    gate, a spatial-attention gate, and a final channel-attention
    refinement pass on the fused result.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2 * channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2 * channels, 1, 1),
            nn.Sigmoid(),
        )
        self.fuse_conv = nn.Conv2d(2 * channels, channels, 1)
        self.channel_refine = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, shallow: Tensor, deep: Tensor) -> Tensor:
        """Adaptively fuse a shallow and a (upsampled) deep feature map.

        Parameters
        ----------
        shallow : Tensor
            Shallow encoder feature map, shape ``(batch, C, H, W)``.
        deep : Tensor
            Deeper encoder feature map already resized to ``(H, W)``,
            shape ``(batch, C, H, W)``.

        Returns
        -------
        Tensor
            Fused feature map, shape ``(batch, C, H, W)``.
        """

        concat = torch.cat([shallow, deep], dim=1)
        channel_weight = self.channel_gate(concat)
        gated = concat * channel_weight
        spatial_weight = self.spatial_gate(gated)
        gated = gated * spatial_weight
        fused = self.fuse_conv(gated)
        refine_weight = self.channel_refine(fused)
        return fused * refine_weight


class CDNetV2(nn.Module):
    """Compact cloud-detection network with AFFM fusion and HSIGF guidance.

    Reproduces ``nkszjx/CDnetV2-pytorch-master``: a 3-stage
    downsampling encoder, an AFFM module fusing each pair of adjacent
    -resolution encoder stages, and a High-level Semantic Information
    Guidance Flow that broadcasts a once-computed deepest-stage
    semantic feature into every decoder stage before the final
    per-pixel classification head.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 8, n_classes: int = 3) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1),
            nn.BatchNorm2d(2 * c),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1),
            nn.BatchNorm2d(4 * c),
            nn.ReLU(inplace=True),
        )
        self.proj2_to_1 = nn.Conv2d(2 * c, c, 1)
        self.proj3_to_2 = nn.Conv2d(4 * c, 2 * c, 1)
        self.affm_12 = _AFFM(c)
        self.affm_23 = _AFFM(2 * c)

        self.hffm = nn.Conv2d(4 * c, c, 1)  # high-level feature fusing model
        self.hsigf_to_stage2 = nn.Conv2d(c, 2 * c, 1)  # HSIGF broadcast, stage-2 width
        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(2 * c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )
        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )
        self.seg_head = nn.Conv2d(c, n_classes, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Segment cloud / snow / background classes.

        Parameters
        ----------
        x : Tensor
            Input remote-sensing image, shape ``(batch, 3, H, W)``.

        Returns
        -------
        Tensor
            Per-pixel class logits, shape
            ``(batch, n_classes, H, W)``.
        """

        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)

        f2_up = F.interpolate(f2, size=f1.shape[-2:], mode="nearest")
        f3_up = F.interpolate(f3, size=f2.shape[-2:], mode="nearest")

        fused_12 = self.affm_12(f1, self.proj2_to_1(f2_up))
        fused_23 = self.affm_23(f2, self.proj3_to_2(f3_up))

        high_level = F.interpolate(self.hffm(f3), size=f1.shape[-2:], mode="nearest")
        high_level_mid = F.interpolate(high_level, size=fused_23.shape[-2:], mode="nearest")
        high_level_mid = self.hsigf_to_stage2(high_level_mid)

        dec2 = self.decoder_stage2(fused_23 + high_level_mid)
        dec2_up = F.interpolate(dec2, size=fused_12.shape[-2:], mode="nearest")
        dec1 = self.decoder_stage1(fused_12 + dec2_up + high_level)

        return self.seg_head(dec1)


def build_cdnetv2() -> nn.Module:
    """Build a compact CDnetV2 cloud-detection network.

    Returns
    -------
    nn.Module
        ``CDNetV2`` in eval mode.
    """

    return CDNetV2().eval()


def example_input_cdnetv2() -> Tensor:
    """Create example input for :func:`build_cdnetv2`.

    Returns
    -------
    Tensor
        A remote-sensing image batch, shape ``(1, 3, 32, 32)``.
    """

    torch.manual_seed(0)
    return torch.randn(1, 3, 32, 32)


# ---------------------------------------------------------------------------
# ClimODE: velocity network (local conv + global attention) + semi
# -Lagrangian advection (grid_sample warp) + Gaussian emission network
# ---------------------------------------------------------------------------


class _FlowVelocityNet(nn.Module):
    """Predicts a 2D advective velocity field from local conv + global attention."""

    def __init__(self, channels: int = 1, hidden: int = 8, num_heads: int = 2) -> None:
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.global_attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.velocity_head = nn.Conv2d(hidden, 2, 1)

    def forward(self, field: Tensor) -> Tensor:
        """Predict the per-pixel advective velocity ``(vx, vy)``.

        Parameters
        ----------
        field : Tensor
            Current quantity field, shape ``(batch, channels, H, W)``.

        Returns
        -------
        Tensor
            Velocity field, shape ``(batch, 2, H, W)``.
        """

        local_feat = self.local_conv(field)
        b, c, h, w = local_feat.shape
        tokens = local_feat.flatten(2).transpose(1, 2)
        normed = self.norm(tokens)
        global_out, _ = self.global_attn(normed, normed, normed, need_weights=False)
        combined = (tokens + global_out).transpose(1, 2).reshape(b, c, h, w)
        return torch.tanh(self.velocity_head(combined)) * 0.1


class _GaussianEmissionNet(nn.Module):
    """Predicts a per-pixel Gaussian source/sink correction (bias + log-variance)."""

    def __init__(self, channels: int = 1, hidden: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 2 * channels, 3, padding=1),
        )
        self.channels = channels

    def forward(self, field: Tensor) -> tuple[Tensor, Tensor]:
        """Predict a Gaussian bias and log-variance correction.

        Returns
        -------
        tuple[Tensor, Tensor]
            Bias ``(batch, channels, H, W)`` and log-variance
            ``(batch, channels, H, W)``.
        """

        out = self.net(field)
        bias, log_var = out.chunk(2, dim=1)
        return bias, log_var


class ClimODE(nn.Module):
    """Physics-informed advection-PDE ODE step: velocity, advect, emit.

    Reproduces ``Aalto-QuML/ClimODE``'s one-step neural-ODE dynamics:
    a flow-velocity network (local convolutions plus a global-
    attention long-range correction) predicts a 2D advective velocity
    field, the current quantity field is *semi-Lagrangian advected* by
    backward-warping it along that velocity with a differentiable
    bilinear ``grid_sample`` (a continuity-equation transport step,
    not a learned convolutional blur), and a Gaussian emission network
    adds a local bias + variance source/sink correction on top of the
    advected field.
    """

    def __init__(self, channels: int = 1, hidden: int = 8) -> None:
        super().__init__()
        self.velocity_net = _FlowVelocityNet(channels, hidden)
        self.emission_net = _GaussianEmissionNet(channels, hidden)

    @staticmethod
    def _make_base_grid(h: int, w: int, device: torch.device) -> Tensor:
        ys = torch.linspace(-1, 1, h, device=device)
        xs = torch.linspace(-1, 1, w, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1)

    def forward(self, field: Tensor, dt: Tensor) -> tuple[Tensor, Tensor]:
        """Advect ``field`` forward by ``dt`` and apply the emission correction.

        Parameters
        ----------
        field : Tensor
            Current quantity field, shape ``(batch, channels, H, W)``.
        dt : Tensor
            Scalar integration step size, shape ``(batch, 1, 1, 1)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Advected-and-emitted next field ``(batch, channels, H, W)``
            and the emission log-variance ``(batch, channels, H, W)``.
        """

        b, _, h, w = field.shape
        velocity = self.velocity_net(field)

        base_grid = self._make_base_grid(h, w, field.device).unsqueeze(0).expand(b, -1, -1, -1)
        # normalized-grid displacement: 2 / (size - 1) per unit pixel velocity
        vel_norm_x = velocity[:, 0] * dt[:, 0, 0, 0].view(-1, 1, 1) * (2.0 / max(w - 1, 1))
        vel_norm_y = velocity[:, 1] * dt[:, 0, 0, 0].view(-1, 1, 1) * (2.0 / max(h - 1, 1))
        warp_grid = base_grid - torch.stack([vel_norm_x, vel_norm_y], dim=-1)

        advected = F.grid_sample(field, warp_grid, mode="bilinear", align_corners=True)
        bias, log_var = self.emission_net(advected)
        return advected + bias, log_var


def build_climode() -> nn.Module:
    """Build a compact ClimODE advection-PDE ODE-step model.

    Returns
    -------
    nn.Module
        ``ClimODE`` in eval mode.
    """

    return ClimODE().eval()


def example_input_climode() -> tuple[Tensor, Tensor]:
    """Create example input for :func:`build_climode`.

    Returns
    -------
    tuple[Tensor, Tensor]
        A quantity field, shape ``(2, 1, 16, 16)``, and a per-sample
        integration step size, shape ``(2, 1, 1, 1)``.
    """

    torch.manual_seed(0)
    field = torch.randn(2, 1, 16, 16)
    dt = torch.full((2, 1, 1, 1), 0.5)
    return field, dt


MENAGERIE_ENTRIES = [
    ("ArchesWeather", "build_archesweather", "example_input_archesweather", "2024", "VIS"),
    ("AtmoRep", "build_atmorep", "example_input_atmorep", "2023", "VIS"),
    ("CACo", "build_caco", "example_input_caco", "2023", "VIS"),
    ("CDNetV2", "build_cdnetv2", "example_input_cdnetv2", "2021", "VIS"),
    ("ClimODE", "build_climode", "example_input_climode", "2024", "VIS"),
]

if __name__ == "__main__":
    print(f"{len(MENAGERIE_ENTRIES)} entries defined; run the smoke-trace gate to verify.")
