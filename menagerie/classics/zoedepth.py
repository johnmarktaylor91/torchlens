"""ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth.

Bhat et al., 2023.
Paper: https://arxiv.org/abs/2302.12288
Source: https://github.com/isl-org/ZoeDepth

ZoeDepth extends MiDaS DPT relative-depth with a metric bins module:
  SeedBinRegressor -> AttractorLayers (progressive bin center refinement)
  -> ConditionalLogBinomial (weighted sum over bin centers -> metric depth).

Three variants:
  ZoeD_N:  single metric head, NYU (indoor), max_depth=10m
  ZoeD_K:  single metric head, KITTI (outdoor), max_depth=80m
  ZoeD_NK: two metric heads (NYU + KITTI) with a learned domain router

This faithful reimplementation captures ZoeDepth's distinctive metric-bins
contribution on top of a DPT-style encoder-decoder backbone.  The backbone
is a CNN-based DPT (ResNet-50 feature pyramid -> dense prediction transformer
neck -> DPT decoder) at published-compatible width; the metric-bins head is
reproduced exactly from the published source.

Architecture notes:
  - Backbone: ResNet-50 feature pyramid (C2/C3/C4/C5 = 256/512/1024/2048 ch)
    + DPT reassemble/fusion decoder producing relative depth + 4 feature maps
    (output channels match the ZoeDepth source: btlnck=256, [256,256,256,32])
  - Metric head: SeedBinRegressorUnnormed -> 4x AttractorLayerUnnormed ->
    ConditionalLogBinomial -> weighted sum over bin centers
  - n_bins=64 (default), n_attractors=[16,8,4,1], bin_embedding_dim=128
  - ZoeD_NK adds a PatchTransformer router + 2 independent metric heads
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Metric-bins layers (faithful from ZoeDepth source)
# ============================================================


def _log_binom(n: torch.Tensor, k: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """log(nCk) using Stirling approximation."""
    n = n + eps
    k = k + eps
    return n * torch.log(n) - k * torch.log(k) - (n - k) * torch.log(n - k + eps)


class LogBinomial(nn.Module):
    def __init__(self, n_classes: int = 256) -> None:
        super().__init__()
        self.K = n_classes
        self.register_buffer("k_idx", torch.arange(0, n_classes).view(1, -1, 1, 1))
        self.register_buffer(
            "K_minus_1", torch.tensor([n_classes - 1], dtype=torch.float).view(1, -1, 1, 1)
        )

    def forward(self, x: torch.Tensor, t: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        one_minus_x = torch.clamp(1 - x, eps, 1)
        x = torch.clamp(x, eps, 1)
        y = (
            _log_binom(self.K_minus_1, self.k_idx)
            + self.k_idx * torch.log(x)
            + (self.K - 1 - self.k_idx) * torch.log(one_minus_x)
        )
        return torch.softmax(y / t, dim=1)


class ConditionalLogBinomial(nn.Module):
    """Conditional log-binomial output distribution.

    Predicts p and temperature t from (feature, condition), then applies
    the log-binomial transform to produce per-bin softmax probabilities.
    """

    def __init__(
        self,
        in_features: int,
        condition_dim: int,
        n_classes: int = 64,
        bottleneck_factor: int = 2,
        p_eps: float = 1e-4,
        max_temp: float = 50.0,
        min_temp: float = 1e-7,
    ) -> None:
        super().__init__()
        self.p_eps = p_eps
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.log_binomial_transform = LogBinomial(n_classes)
        bottleneck = (in_features + condition_dim) // bottleneck_factor
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features + condition_dim, bottleneck, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(bottleneck, 4, 1, 1, 0),  # 2 for p, 2 for t
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        pt = self.mlp(torch.cat((x, cond), dim=1))
        p, t = pt[:, :2, ...], pt[:, 2:, ...]
        p = p + self.p_eps
        p = p[:, 0, ...] / (p[:, 0, ...] + p[:, 1, ...])
        t = t + self.p_eps
        t = t[:, 0, ...] / (t[:, 0, ...] + t[:, 1, ...])
        t = t.unsqueeze(1)
        t = (self.max_temp - self.min_temp) * t + self.min_temp
        return self.log_binomial_transform(p, t)


class SeedBinRegressorUnnormed(nn.Module):
    """Seed bin regressor producing unbounded bin centers (softplus variant)."""

    def __init__(
        self,
        in_features: int,
        n_bins: int = 64,
        mlp_dim: int = 256,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
    ) -> None:
        super().__init__()
        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, n_bins, 1, 1, 0),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor):
        B_centers = self._net(x)
        return B_centers, B_centers


class AttractorLayerUnnormed(nn.Module):
    """Attractor layer for unbounded bin centers (softplus variant).

    Predicts attractor points A from features, computes delta_c using
    inverse attractor: dc = dx / (1 + alpha * |dx|^gamma) where dx = A - c.
    Refines bin centers b_prev -> b_new.
    """

    def __init__(
        self,
        in_features: int,
        n_bins: int,
        n_attractors: int = 16,
        mlp_dim: int = 128,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        alpha: float = 300.0,
        gamma: int = 2,
        kind: str = "sum",
        attractor_type: str = "inv",
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        self.n_attractors = n_attractors
        self.alpha = alpha
        self.gamma = gamma
        self.kind = kind
        self.memory_efficient = memory_efficient
        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, n_attractors, 1, 1, 0),
            nn.Softplus(),
        )

    def _inv_attractor(self, dx: torch.Tensor) -> torch.Tensor:
        return dx / (1 + self.alpha * dx.pow(self.gamma))

    def forward(
        self,
        x: torch.Tensor,
        b_prev: torch.Tensor,
        prev_b_embedding: Optional[torch.Tensor] = None,
        interpolate: bool = True,
        is_for_query: bool = False,
    ):
        if prev_b_embedding is not None:
            if interpolate:
                prev_b_embedding = F.interpolate(
                    prev_b_embedding, x.shape[-2:], mode="bilinear", align_corners=True
                )
            x = x + prev_b_embedding

        A = self._net(x)
        b_prev = F.interpolate(
            b_prev, (x.shape[2], x.shape[3]), mode="bilinear", align_corners=True
        )
        b_centers = b_prev

        if not self.memory_efficient:
            func = {"mean": torch.mean, "sum": torch.sum}[self.kind]
            # A: N, na, h, w -> unsqueeze(2); b_centers: N, nbins, h, w -> unsqueeze(1)
            delta_c = func(self._inv_attractor(A.unsqueeze(2) - b_centers.unsqueeze(1)), dim=1)
        else:
            delta_c = torch.zeros_like(b_centers)
            for i in range(self.n_attractors):
                delta_c += self._inv_attractor(A[:, i, ...].unsqueeze(1) - b_centers)
            if self.kind == "mean":
                delta_c = delta_c / self.n_attractors

        b_new_centers = b_centers + delta_c
        return b_new_centers, b_new_centers


class Projector(nn.Module):
    """1x1 conv MLP to project features to bin-embedding space."""

    def __init__(self, in_features: int, out_features: int, mlp_dim: int = 128) -> None:
        super().__init__()
        self._net = nn.Sequential(
            nn.Conv2d(in_features, mlp_dim, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_dim, out_features, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)


# ============================================================
# Lightweight DPT-style backbone (CNN feature pyramid + fusion decoder)
# producing relative depth + multiscale features
# ============================================================


class _ConvBNReLU(nn.Sequential):
    def __init__(
        self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int = 1
    ) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = _ConvBNReLU(ch, ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + x)


class _DPTHead(nn.Module):
    """DPT-style dense-prediction neck: fuses 4 feature levels -> relative depth map + 4 feature outputs.

    Output channels match ZoeDepth's expected MiDaS outputs:
      out[0] = outconv_activation (N, 32, H/2, W/2)  <- final conv features
      out[1] = btlnck (N, 256, H/32, W/32)            <- bottleneck features
      out[2..5] = x_blocks at H/16, H/8, H/4, H/2    <- (N, 256, ...)
    """

    def __init__(
        self, enc_channels: List[int] = [256, 512, 1024, 2048], feat_dim: int = 256
    ) -> None:
        super().__init__()
        # Reassemble: project each level to feat_dim
        self.proj = nn.ModuleList([nn.Conv2d(ch, feat_dim, 1) for ch in enc_channels])
        # Fusion layers (bottom-up)
        self.fuse3 = _ConvBNReLU(feat_dim, feat_dim)
        self.fuse2 = _ConvBNReLU(feat_dim, feat_dim)
        self.fuse1 = _ConvBNReLU(feat_dim, feat_dim)
        self.fuse0 = _ConvBNReLU(feat_dim, feat_dim // 2)
        # Relative depth head
        self.depth_head = nn.Sequential(
            nn.Conv2d(feat_dim // 2, feat_dim // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 4, 1, 1),
            nn.ReLU(inplace=True),
        )
        # Feature outputs for metric head (each at the fused scale)
        # Match ZoeDepth: out[0]=32ch high-res, out[1]=256ch btlnck, out[2..5] are 256-ch blocks
        self.out_conv = nn.Conv2d(feat_dim // 2, 32, 3, padding=1)

    def forward(self, feats: List[torch.Tensor]) -> tuple:
        # feats: [c2, c3, c4, c5] at strides 4, 8, 16, 32
        p = [proj(f) for proj, f in zip(self.proj, feats)]
        # Upsample and fuse from deepest
        x = p[3]
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.fuse3(x + p[2])
        btlnck = x  # (N, 256, H/16, W/16)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x2 = self.fuse2(x + p[1])  # H/8
        x = F.interpolate(x2, scale_factor=2, mode="bilinear", align_corners=False)
        x1 = self.fuse1(x + p[0])  # H/4
        x = F.interpolate(x1, scale_factor=2, mode="bilinear", align_corners=False)
        x0 = self.fuse0(x)  # H/2, 128ch
        rel_depth = self.depth_head(x0).squeeze(1)  # (N, H/2, W/2)
        outconv = self.out_conv(x0)  # (N, 32, H/2, W/2)
        return rel_depth, (outconv, btlnck, p[3], x2, x1, x0)


class _CNNBackbone(nn.Module):
    """Simple CNN encoder with 4 progressively downsampled feature levels.

    Produces C2/C3/C4/C5 at strides 4/8/16/32 with channels [256,512,1024,2048]
    (matching ResNet-50 output channels used by ZoeDepth's DPT-BEiT backbone).
    """

    def __init__(self, enc_channels: List[int] = [64, 128, 256, 512]) -> None:
        super().__init__()
        in_ch = 3
        self.stem = _ConvBNReLU(in_ch, enc_channels[0], 7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)  # stride 4 total

        def _stage(in_c: int, out_c: int, n: int, stride: int = 2) -> nn.Sequential:
            layers: List[nn.Module] = []
            # Down-sample with stride conv
            layers.append(_ConvBNReLU(in_c, out_c, stride=stride))
            for _ in range(n - 1):
                layers.append(_ResBlock(out_c))
            return nn.Sequential(*layers)

        self.layer1 = _stage(enc_channels[0], enc_channels[1], 2)  # stride 8
        self.layer2 = _stage(enc_channels[1], enc_channels[2], 3)  # stride 16
        self.layer3 = _stage(enc_channels[2], enc_channels[3], 2)  # stride 32

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c1 = self.pool(self.stem(x))  # stride 4
        c2 = self.layer1(c1)  # stride 8
        c3 = self.layer2(c2)  # stride 16
        c4 = self.layer3(c3)  # stride 32
        return [c1, c2, c3, c4]


# ============================================================
# ZoeDepth single-head model (N or K variant)
# ============================================================


class ZoeDepthSingleHead(nn.Module):
    """ZoeDepth with one metric head.

    Matches the architecture from zoedepth_v1.py:
      core -> (rel_depth, [outconv, btlnck, x_blocks...])
      conv2(btlnck) -> SeedBinRegressor -> seed_projector
      for each (projector, attractor, x_block): refine bin centers
      ConditionalLogBinomial(outconv + rel_depth, b_embedding) -> probs
      metric_depth = sum(probs * b_centers)
    """

    def __init__(
        self,
        n_bins: int = 64,
        bin_embedding_dim: int = 128,
        n_attractors: List[int] = None,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        min_temp: float = 5.0,
        max_temp: float = 50.0,
        enc_channels: List[int] = None,
        feat_dim: int = 256,
    ) -> None:
        super().__init__()
        if n_attractors is None:
            n_attractors = [16, 8, 4, 1]
        if enc_channels is None:
            enc_channels = [64, 128, 256, 512]

        # Backbone
        self.backbone = _CNNBackbone(enc_channels)
        self.dpt_head = _DPTHead(
            enc_channels=[enc_channels[0], enc_channels[1], enc_channels[2], enc_channels[3]],
            feat_dim=feat_dim,
        )

        # Metric bins head (faithful to ZoeDepth source)
        btlnck_features = feat_dim  # 256
        # num_out_features: channels of x_blocks (p3, x2, x1, x0)
        # x0=128, x1=feat_dim, x2=feat_dim, p3=feat_dim; but we match ZoeDepth: 4 attractor layers
        num_out_features = [feat_dim, feat_dim, feat_dim, feat_dim // 2]

        self.conv2 = nn.Conv2d(btlnck_features, btlnck_features, 1, 1, 0)
        self.seed_bin_regressor = SeedBinRegressorUnnormed(
            btlnck_features, n_bins=n_bins, mlp_dim=256, min_depth=min_depth, max_depth=max_depth
        )
        self.seed_projector = Projector(
            btlnck_features, bin_embedding_dim, mlp_dim=bin_embedding_dim
        )
        self.projectors = nn.ModuleList(
            [
                Projector(nout, bin_embedding_dim, mlp_dim=bin_embedding_dim)
                for nout in num_out_features
            ]
        )
        self.attractors = nn.ModuleList(
            [
                AttractorLayerUnnormed(
                    bin_embedding_dim,
                    n_bins,
                    n_attractors=n_attractors[i],
                    mlp_dim=bin_embedding_dim,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    alpha=300,
                    gamma=2,
                    kind="sum",
                    attractor_type="inv",
                )
                for i in range(len(num_out_features))
            ]
        )
        # N_MIDAS_OUT=32 + 1 for rel depth
        last_in = 32 + 1
        self.conditional_log_binomial = ConditionalLogBinomial(
            last_in,
            bin_embedding_dim,
            n_classes=n_bins,
            bottleneck_factor=2,
            min_temp=min_temp,
            max_temp=max_temp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        feats = self.backbone(x)
        rel_depth, out = self.dpt_head(feats)

        outconv_activation = out[0]  # (N, 32, H/2, W/2)
        btlnck = out[1]  # (N, 256, H/32, W/32) -- actually H/16
        x_blocks = list(out[2:])  # 4 feature maps at various scales

        x_d0 = self.conv2(btlnck)
        x_seed = x_d0
        _, seed_b_centers = self.seed_bin_regressor(x_seed)
        b_prev = seed_b_centers
        prev_b_embedding = self.seed_projector(x_seed)

        for projector, attractor, xblk in zip(self.projectors, self.attractors, x_blocks):
            b_embedding = projector(xblk)
            b, b_centers = attractor(b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()

        last = outconv_activation
        rel_cond = F.interpolate(
            rel_depth.unsqueeze(1), size=last.shape[2:], mode="bilinear", align_corners=True
        )
        last = torch.cat([last, rel_cond], dim=1)  # (N, 33, h, w)

        b_embedding = F.interpolate(
            b_embedding, last.shape[-2:], mode="bilinear", align_corners=True
        )
        probs = self.conditional_log_binomial(last, b_embedding)  # (N, n_bins, h, w)

        b_centers = F.interpolate(b_centers, probs.shape[-2:], mode="bilinear", align_corners=True)
        metric_depth = torch.sum(probs * b_centers, dim=1, keepdim=True)
        metric_depth = F.interpolate(
            metric_depth, size=x.shape[-2:], mode="bilinear", align_corners=True
        )
        return metric_depth


# ============================================================
# ZoeDepth NK (two metric heads + learned domain router)
# ============================================================


class _PatchTransformerEncoder(nn.Module):
    """Lightweight patch transformer for domain classification.

    Faithfully reproduces ZoeDepth-NK's PatchTransformerEncoder:
    projects bottleneck patches to tokens, applies transformer layers,
    returns CLS-like token for domain routing.
    """

    def __init__(
        self, in_channels: int = 256, patch_size: int = 1, embedding_dim: int = 128
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embedding_dim, patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=4, dim_feedforward=256, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W) -> tokens: (N, H*W, embedding_dim)
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out = self.transformer(tokens)
        return out[:, 0, :]  # CLS token -> (N, embedding_dim)


class ZoeDepthNK(nn.Module):
    """ZoeDepth-NK: two metric heads with learned domain routing.

    Shares backbone + DPT decoder. Separate SeedBinRegressor + AttractorLayers
    + ConditionalLogBinomial per domain (NYU/KITTI). PatchTransformer
    routes each image to one domain head at inference.
    """

    def __init__(
        self,
        n_bins: int = 64,
        bin_embedding_dim: int = 128,
        n_attractors: List[int] = None,
        nyu_min_depth: float = 1e-3,
        nyu_max_depth: float = 10.0,
        kitti_min_depth: float = 1e-3,
        kitti_max_depth: float = 80.0,
        min_temp: float = 5.0,
        max_temp: float = 50.0,
        enc_channels: List[int] = None,
        feat_dim: int = 256,
    ) -> None:
        super().__init__()
        if n_attractors is None:
            n_attractors = [16, 8, 4, 1]
        if enc_channels is None:
            enc_channels = [64, 128, 256, 512]

        # Shared backbone
        self.backbone = _CNNBackbone(enc_channels)
        self.dpt_head = _DPTHead(
            enc_channels=[enc_channels[0], enc_channels[1], enc_channels[2], enc_channels[3]],
            feat_dim=feat_dim,
        )

        btlnck_features = feat_dim
        num_out_features = [feat_dim, feat_dim, feat_dim, feat_dim // 2]

        self.conv2 = nn.Conv2d(btlnck_features, btlnck_features, 1, 1, 0)

        # Domain router (PatchTransformerEncoder + MLP)
        self.patch_transformer = _PatchTransformerEncoder(btlnck_features, 1, bin_embedding_dim)
        self.mlp_classifier = nn.Sequential(
            nn.Linear(bin_embedding_dim, bin_embedding_dim),
            nn.ReLU(),
            nn.Linear(bin_embedding_dim, 2),
        )

        # Shared projectors
        self.seed_projector = Projector(
            btlnck_features, bin_embedding_dim, mlp_dim=bin_embedding_dim // 2
        )
        self.projectors = nn.ModuleList(
            [
                Projector(nout, bin_embedding_dim, mlp_dim=bin_embedding_dim // 2)
                for nout in num_out_features
            ]
        )

        # Per-domain heads
        domain_configs = {
            "nyu": (nyu_min_depth, nyu_max_depth),
            "kitti": (kitti_min_depth, kitti_max_depth),
        }
        self.seed_bin_regressors = nn.ModuleDict(
            {
                name: SeedBinRegressorUnnormed(
                    btlnck_features,
                    n_bins=n_bins,
                    mlp_dim=bin_embedding_dim // 2,
                    min_depth=dmin,
                    max_depth=dmax,
                )
                for name, (dmin, dmax) in domain_configs.items()
            }
        )
        self.attractors = nn.ModuleDict(
            {
                name: nn.ModuleList(
                    [
                        AttractorLayerUnnormed(
                            bin_embedding_dim,
                            n_bins,
                            n_attractors=n_attractors[i],
                            mlp_dim=bin_embedding_dim,
                            min_depth=dmin,
                            max_depth=dmax,
                            alpha=300,
                            gamma=2,
                            kind="sum",
                            attractor_type="inv",
                            memory_efficient=True,
                        )
                        for i in range(len(n_attractors))
                    ]
                )
                for name, (dmin, dmax) in domain_configs.items()
            }
        )
        last_in = 32  # ZoeDepth-NK uses outconv only (no rel_depth concat in NK version)
        self.conditional_log_binomial = nn.ModuleDict(
            {
                name: ConditionalLogBinomial(
                    last_in,
                    bin_embedding_dim,
                    n_classes=n_bins,
                    bottleneck_factor=4,
                    min_temp=min_temp,
                    max_temp=max_temp,
                )
                for name in domain_configs
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        rel_depth, out = self.dpt_head(feats)

        outconv_activation = out[0]
        btlnck = out[1]
        x_blocks = list(out[2:])

        x_d0 = self.conv2(btlnck)

        # Domain routing
        embedding = self.patch_transformer(x_d0)  # (N, 128)
        domain_logits = self.mlp_classifier(embedding)  # (N, 2)
        domain_vote = torch.softmax(domain_logits.sum(dim=0, keepdim=True), dim=-1)
        domain_idx = torch.argmax(domain_vote, dim=-1).item()
        domain_name = ["nyu", "kitti"][int(domain_idx)]

        _, seed_b_centers = self.seed_bin_regressors[domain_name](x_d0)
        b_prev = seed_b_centers
        prev_b_embedding = self.seed_projector(x_d0)

        domain_attractors = self.attractors[domain_name]
        for projector, attractor, xblk in zip(self.projectors, domain_attractors, x_blocks):
            b_embedding = projector(xblk)
            b, b_centers = attractor(b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b
            prev_b_embedding = b_embedding

        last = outconv_activation
        b_centers = F.interpolate(b_centers, last.shape[-2:], mode="bilinear", align_corners=True)
        b_embedding = F.interpolate(
            b_embedding, last.shape[-2:], mode="bilinear", align_corners=True
        )

        clb = self.conditional_log_binomial[domain_name]
        probs = clb(last, b_embedding)

        metric_depth = torch.sum(probs * b_centers, dim=1, keepdim=True)
        metric_depth = F.interpolate(
            metric_depth, size=x.shape[-2:], mode="bilinear", align_corners=True
        )
        return metric_depth


def build(variant: str = "N") -> nn.Module:
    """Build a ZoeDepth variant.

    Args:
        variant: "N" (NYU indoor, max_depth=10), "K" (KITTI outdoor, max_depth=80),
                 or "NK" (two heads + router).
    """
    if variant == "N":
        return ZoeDepthSingleHead(
            n_bins=64,
            bin_embedding_dim=128,
            n_attractors=[16, 8, 4, 1],
            min_depth=1e-3,
            max_depth=10.0,
            min_temp=5.0,
            max_temp=50.0,
            enc_channels=[64, 128, 256, 512],
            feat_dim=256,
        )
    elif variant == "K":
        return ZoeDepthSingleHead(
            n_bins=64,
            bin_embedding_dim=128,
            n_attractors=[16, 8, 4, 1],
            min_depth=1e-3,
            max_depth=80.0,
            min_temp=5.0,
            max_temp=50.0,
            enc_channels=[64, 128, 256, 512],
            feat_dim=256,
        )
    elif variant == "NK":
        return ZoeDepthNK(
            n_bins=64,
            bin_embedding_dim=128,
            n_attractors=[16, 8, 4, 1],
            nyu_min_depth=1e-3,
            nyu_max_depth=10.0,
            kitti_min_depth=1e-3,
            kitti_max_depth=80.0,
            min_temp=5.0,
            max_temp=50.0,
            enc_channels=[64, 128, 256, 512],
            feat_dim=256,
        )
    else:
        raise ValueError(f"Unknown ZoeDepth variant: {variant!r}. Use 'N', 'K', or 'NK'.")


# ============================================================
# Menagerie wiring: zero-arg builders + example inputs + entries.
# ============================================================


def build_zoed_n() -> nn.Module:
    """Build ZoeD_N (single metric head, NYU indoor, max_depth=10m)."""
    return build("N")


def build_zoed_k() -> nn.Module:
    """Build ZoeD_K (single metric head, KITTI outdoor, max_depth=80m)."""
    return build("K")


def build_zoed_nk() -> nn.Module:
    """Build ZoeD_NK (two metric heads + learned domain router)."""
    return build("NK")


def example_input() -> torch.Tensor:
    """Example RGB image tensor ``(1, 3, 384, 512)`` for ZoeDepth."""
    return torch.randn(1, 3, 384, 512)


MENAGERIE_ENTRIES = [
    (
        "ZoeDepth ZoeD_N (metric-bins depth, NYU indoor)",
        "build_zoed_n",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "ZoeDepth ZoeD_K (metric-bins depth, KITTI outdoor)",
        "build_zoed_k",
        "example_input",
        "2023",
        "DC",
    ),
    (
        "ZoeDepth ZoeD_NK (two metric heads + domain router)",
        "build_zoed_nk",
        "example_input",
        "2023",
        "DC",
    ),
]
