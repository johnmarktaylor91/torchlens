"""Missing notable vision architectures from proceedings sweeps.

Paper: DKM: Dense Kernelized Feature Matching for Geometry Estimation.
Edstedt et al., CVPR 2023.

Paper: CoTracker: It is Better to Track Together.
Karaev et al., ECCV 2024.

Paper: TAPIR: Tracking Any Point with per-frame Initialization and temporal Refinement.
Doersch et al., ICCV 2023.

Paper: XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model.
Cheng and Schwing, ECCV 2022.

Paper: D2-Net: A Trainable CNN for Joint Description and Detection of Local Features.
Dusmanu et al., CVPR 2019.

Paper: DenseASPP for Semantic Segmentation in Street Scenes.
Yang et al., CVPR 2018.

Paper: RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation.
Lin et al., CVPR 2017.

These are compact, faithful random-init reimplementations of the load-bearing
architecture ideas for tracing: dense GP-style matching with depthwise warp
refinement (DKM), joint point-token transformer tracking (CoTracker), per-frame
matching plus temporal local-correlation refinement (TAPIR), and sensory /
working / long-term memory reading for video object segmentation (XMem), joint
local-feature detection/description (D2-Net), densely connected atrous context
(DenseASPP), and cascaded multi-resolution refinement (RefineNet).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _grid_xy(batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Create a normalized ``(B, H*W, 2)`` coordinate grid.

    Parameters
    ----------
    batch:
        Batch size.
    height:
        Grid height.
    width:
        Grid width.
    device:
        Target torch device.

    Returns
    -------
    torch.Tensor
        Normalized ``x, y`` coordinates in ``[-1, 1]``.
    """

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    return torch.stack((xx, yy), dim=-1).view(1, height * width, 2).repeat(batch, 1, 1)


def _sample_points(features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample point features at normalized coordinates.

    Parameters
    ----------
    features:
        Feature map ``(B, C, H, W)``.
    coords:
        Normalized coordinates ``(B, Q, 2)``.

    Returns
    -------
    torch.Tensor
        Sampled point descriptors ``(B, Q, C)``.
    """

    grid = coords.view(coords.shape[0], coords.shape[1], 1, 2)
    sampled = F.grid_sample(features, grid, align_corners=True)
    return sampled.squeeze(-1).transpose(1, 2)


class ConvStem(nn.Module):
    """Small convolutional encoder shared by the compact vision classics."""

    def __init__(self, in_ch: int, dim: int) -> None:
        """Initialize the convolutional stem.

        Parameters
        ----------
        in_ch:
            Number of image channels.
        dim:
            Output feature dimension.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image tensor.

        Parameters
        ----------
        x:
            Image tensor ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Feature map ``(B, dim, H/2, W/2)``.
        """

        return self.net(x)


class DKMMatcher(nn.Module):
    """Dense Kernelized Matching with GP-style global matching and local refinement."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize the compact DKM matcher.

        Parameters
        ----------
        dim:
            Feature dimension.
        """

        super().__init__()
        self.encoder = ConvStem(3, dim)
        self.query = nn.Conv2d(dim, dim, 1)
        self.key = nn.Conv2d(dim, dim, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(4, 16, 7, padding=3, groups=4),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 2, 3, padding=1),
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """Estimate a dense warp and confidence from a packed image pair.

        Parameters
        ----------
        pair:
            Packed two-image tensor ``(B, 6, H, W)``.

        Returns
        -------
        torch.Tensor
            Dense warp plus confidence ``(B, 3, H/2, W/2)``.
        """

        image_a, image_b = pair[:, :3], pair[:, 3:]
        fa = F.normalize(self.query(self.encoder(image_a)), dim=1)
        fb = F.normalize(self.key(self.encoder(image_b)), dim=1)
        bsz, channels, height, width = fa.shape
        qa = fa.flatten(2).transpose(1, 2)
        kb = fb.flatten(2)
        corr = torch.bmm(qa, kb) / channels**0.5
        prob = torch.softmax(corr, dim=-1)
        coords = _grid_xy(bsz, height, width, pair.device)
        coarse = torch.bmm(prob, coords).transpose(1, 2).view(bsz, 2, height, width)
        base = coords.transpose(1, 2).view(bsz, 2, height, width)
        delta = self.refine(torch.cat((base, coarse), dim=1))
        warp = coarse + 0.1 * torch.tanh(delta)
        confidence = prob.max(dim=-1).values.view(bsz, 1, height, width)
        return torch.cat((warp, confidence), dim=1)


class CoTrackerTiny(nn.Module):
    """Joint point tracker with shared transformer updates across query points."""

    def __init__(self, dim: int = 48, n_points: int = 6, n_frames: int = 4) -> None:
        """Initialize the compact CoTracker model.

        Parameters
        ----------
        dim:
            Token dimension.
        n_points:
            Number of fixed point queries.
        n_frames:
            Number of video frames.
        """

        super().__init__()
        self.encoder = ConvStem(3, dim)
        coords = torch.linspace(-0.75, 0.75, n_points)
        self.register_buffer("query_xy", torch.stack((coords, coords.flip(0)), dim=-1))
        self.time_embed = nn.Parameter(torch.randn(n_frames, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=dim * 2, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.update = nn.Linear(dim, 3)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Track fixed point queries jointly across a video.

        Parameters
        ----------
        video:
            Video tensor ``(B, T, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Per-frame point coordinates and visibility ``(B, T, Q, 3)``.
        """

        bsz, n_frames, channels, height, width = video.shape
        flat = video.view(bsz * n_frames, channels, height, width)
        feats = self.encoder(flat)
        _, feat_ch, _, _ = feats.shape
        feats = feats.view(bsz, n_frames, feat_ch, feats.shape[-2], feats.shape[-1])
        coords = self.query_xy.view(1, 1, -1, 2).repeat(bsz, n_frames, 1, 1)
        tokens = []
        for frame_idx in range(n_frames):
            sampled = _sample_points(feats[:, frame_idx], coords[:, frame_idx])
            sampled = sampled + self.time_embed[frame_idx].view(1, 1, -1)
            tokens.append(sampled)
        joint = torch.cat(tokens, dim=1)
        updated = self.transformer(joint).view(bsz, n_frames, -1, feat_ch)
        delta_vis = self.update(updated)
        tracks = coords + 0.1 * torch.tanh(delta_vis[..., :2])
        visibility = torch.sigmoid(delta_vis[..., 2:3])
        return torch.cat((tracks, visibility), dim=-1)


class TAPIRTiny(nn.Module):
    """TAPIR two-stage tracker: per-frame matching followed by temporal refinement."""

    def __init__(self, dim: int = 40, n_points: int = 5) -> None:
        """Initialize the compact TAPIR model.

        Parameters
        ----------
        dim:
            Feature dimension.
        n_points:
            Number of fixed query points on the first frame.
        """

        super().__init__()
        self.encoder = ConvStem(3, dim)
        coords = torch.linspace(-0.8, 0.8, n_points)
        self.register_buffer("query_xy", torch.stack((coords, torch.zeros_like(coords)), dim=-1))
        self.refine = nn.GRU(input_size=dim + 3, hidden_size=dim, batch_first=True)
        self.head = nn.Linear(dim, 3)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Track points with per-frame initialization and temporal refinement.

        Parameters
        ----------
        video:
            Video tensor ``(B, T, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Refined coordinates and occlusion probabilities ``(B, T, Q, 3)``.
        """

        bsz, n_frames, channels, height, width = video.shape
        flat = video.view(bsz * n_frames, channels, height, width)
        feats = F.normalize(self.encoder(flat), dim=1)
        _, feat_ch, feat_h, feat_w = feats.shape
        feats = feats.view(bsz, n_frames, feat_ch, feat_h, feat_w)
        query = self.query_xy.view(1, -1, 2).repeat(bsz, 1, 1)
        query_desc = _sample_points(feats[:, 0], query)
        grid = _grid_xy(bsz, feat_h, feat_w, video.device)
        coarse_tracks = []
        coarse_scores = []
        for frame_idx in range(n_frames):
            fmap = feats[:, frame_idx].flatten(2)
            corr = torch.bmm(query_desc, fmap)
            prob = torch.softmax(corr, dim=-1)
            coarse_tracks.append(torch.bmm(prob, grid))
            coarse_scores.append(prob.max(dim=-1).values.unsqueeze(-1))
        tracks = torch.stack(coarse_tracks, dim=1)
        scores = torch.stack(coarse_scores, dim=1)
        sampled_over_time = []
        for frame_idx in range(n_frames):
            sampled_over_time.append(_sample_points(feats[:, frame_idx], tracks[:, frame_idx]))
        local_features = torch.stack(sampled_over_time, dim=1)
        refine_in = torch.cat((local_features, tracks, scores), dim=-1)
        refine_in = refine_in.transpose(1, 2).reshape(bsz * query.shape[1], n_frames, feat_ch + 3)
        refined, _ = self.refine(refine_in)
        delta_occ = self.head(refined).view(bsz, query.shape[1], n_frames, 3).transpose(1, 2)
        coords = tracks + 0.05 * torch.tanh(delta_occ[..., :2])
        occlusion = torch.sigmoid(delta_occ[..., 2:3])
        return torch.cat((coords, occlusion), dim=-1)


class XMemTiny(nn.Module):
    """XMem-style video object segmenter with three temporal memory stores."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize the compact XMem model.

        Parameters
        ----------
        dim:
            Memory key/value dimension.
        """

        super().__init__()
        self.encoder = ConvStem(4, dim)
        self.key = nn.Conv2d(dim, dim, 1)
        self.value = nn.Conv2d(dim, dim, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 4, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 1, 1),
        )

    def _read_memory(
        self, query: torch.Tensor, mem_key: torch.Tensor, mem_value: torch.Tensor
    ) -> torch.Tensor:
        """Read a key-value memory bank with dense attention.

        Parameters
        ----------
        query:
            Query key map ``(B, C, H, W)``.
        mem_key:
            Memory keys ``(B, M, C, H, W)``.
        mem_value:
            Memory values ``(B, M, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Retrieved value map ``(B, C, H, W)``.
        """

        bsz, channels, height, width = query.shape
        q = query.flatten(2).transpose(1, 2)
        mk = mem_key.flatten(3).permute(0, 2, 1, 3).reshape(bsz, channels, -1)
        mv = mem_value.flatten(3).permute(0, 2, 1, 3).reshape(bsz, channels, -1)
        attn = torch.softmax(torch.bmm(q, mk) / channels**0.5, dim=-1)
        read = torch.bmm(attn, mv.transpose(1, 2)).transpose(1, 2)
        return read.view(bsz, channels, height, width)

    def forward(self, video_mask: torch.Tensor) -> torch.Tensor:
        """Segment a short video with sensory, working, and long-term memory reads.

        Parameters
        ----------
        video_mask:
            Packed tensor ``(B, T, 4, H, W)`` with RGB plus initial-mask channel.

        Returns
        -------
        torch.Tensor
            Mask logits for all frames ``(B, T, 1, H/2, W/2)``.
        """

        bsz, n_frames, channels, height, width = video_mask.shape
        flat = video_mask.view(bsz * n_frames, channels, height, width)
        feat = self.encoder(flat)
        _, feat_ch, feat_h, feat_w = feat.shape
        feat = feat.view(bsz, n_frames, feat_ch, feat_h, feat_w)
        keys = self.key(feat.view(bsz * n_frames, feat_ch, feat_h, feat_w)).view(
            bsz, n_frames, feat_ch, feat_h, feat_w
        )
        vals = self.value(feat.view(bsz * n_frames, feat_ch, feat_h, feat_w)).view(
            bsz, n_frames, feat_ch, feat_h, feat_w
        )
        sensory_key = keys[:, :1]
        sensory_val = vals[:, :1]
        working_key = keys[:, :2]
        working_val = vals[:, :2]
        long_key = keys.mean(dim=1, keepdim=True)
        long_val = vals.mean(dim=1, keepdim=True)
        outputs = []
        for frame_idx in range(n_frames):
            query = keys[:, frame_idx]
            sensory = self._read_memory(query, sensory_key, sensory_val)
            working = self._read_memory(query, working_key, working_val)
            long_term = self._read_memory(query, long_key, long_val)
            fused = torch.cat((feat[:, frame_idx], sensory, working, long_term), dim=1)
            outputs.append(self.fuse(fused))
        return torch.stack(outputs, dim=1)


class D2NetTiny(nn.Module):
    """Joint local-feature detector and descriptor network."""

    def __init__(self, dim: int = 32, desc_dim: int = 64) -> None:
        """Initialize compact D2-Net.

        Parameters
        ----------
        dim:
            Intermediate feature width.
        desc_dim:
            Dense descriptor dimension.
        """

        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(dim, dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, desc_dim, 3, padding=1),
        )
        self.detector_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Produce dense descriptors and D2-Net detection scores.

        Parameters
        ----------
        image:
            Image tensor ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Descriptor channels plus score map ``(B, desc_dim + 1, H/2, W/2)``.
        """

        desc = F.normalize(self.backbone(image), dim=1)
        channel_peak = desc.max(dim=1, keepdim=True).values
        local_peak = F.max_pool2d(channel_peak, kernel_size=3, stride=1, padding=1)
        score = torch.sigmoid((channel_peak - local_peak) * self.detector_temperature)
        return torch.cat((desc, score), dim=1)


class DenseASPPBlock(nn.Module):
    """One DenseASPP atrous block that appends its output to prior features."""

    def __init__(self, in_ch: int, growth: int, dilation: int) -> None:
        """Initialize a densely connected atrous block.

        Parameters
        ----------
        in_ch:
            Input channel count.
        growth:
            Number of channels produced by the block.
        dilation:
            Atrous convolution dilation.
        """

        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, growth, 1),
            nn.BatchNorm2d(growth),
            nn.ReLU(inplace=True),
            nn.Conv2d(growth, growth, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(growth),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Append atrous context features to the input tensor.

        Parameters
        ----------
        x:
            Current dense feature stack.

        Returns
        -------
        torch.Tensor
            Concatenated dense feature stack.
        """

        return torch.cat((x, self.block(x)), dim=1)


class DenseASPPTiny(nn.Module):
    """DenseASPP segmentation network with chained dense atrous context."""

    def __init__(self, base: int = 24, classes: int = 6) -> None:
        """Initialize compact DenseASPP.

        Parameters
        ----------
        base:
            Backbone channel width.
        classes:
            Number of segmentation classes.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, stride=2, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
        )
        channels = base * 2
        blocks = []
        for dilation in (3, 6, 12, 18):
            blocks.append(DenseASPPBlock(channels, base, dilation))
            channels += base
        self.aspp = nn.Sequential(*blocks)
        self.head = nn.Conv2d(channels, classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Segment an image with dense atrous spatial pyramid features.

        Parameters
        ----------
        image:
            Image tensor ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Segmentation logits ``(B, classes, H, W)``.
        """

        features = self.aspp(self.stem(image))
        logits = self.head(features)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


class RefineBlock(nn.Module):
    """RefineNet block with residual convolution and chained residual pooling."""

    def __init__(self, channels: int) -> None:
        """Initialize a refinement block.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.pool_conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine features using residual convolution and pooled context.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Refined feature tensor.
        """

        residual = F.relu(x + self.residual(x))
        pooled = F.max_pool2d(residual, kernel_size=5, stride=1, padding=2)
        return F.relu(residual + self.pool_conv(pooled))


class RefineNetTiny(nn.Module):
    """Multi-path refinement segmentation network."""

    def __init__(self, channels: int = 32, classes: int = 6) -> None:
        """Initialize compact RefineNet.

        Parameters
        ----------
        channels:
            Unified lateral channel count.
        classes:
            Number of segmentation classes.
        """

        super().__init__()
        self.stage1 = ConvStem(3, channels)
        self.stage2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.refine3 = RefineBlock(channels)
        self.refine2 = RefineBlock(channels)
        self.refine1 = RefineBlock(channels)
        self.head = nn.Conv2d(channels, classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Segment an image with cascaded high-resolution refinement.

        Parameters
        ----------
        image:
            Image tensor ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Segmentation logits ``(B, classes, H, W)``.
        """

        low = self.stage1(image)
        mid = self.stage2(low)
        high = self.stage3(mid)
        refined = self.refine3(high)
        refined = F.interpolate(refined, size=mid.shape[-2:], mode="bilinear", align_corners=False)
        refined = self.refine2(refined + mid)
        refined = F.interpolate(refined, size=low.shape[-2:], mode="bilinear", align_corners=False)
        refined = self.refine1(refined + low)
        logits = self.head(refined)
        return F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)


def build_dkm() -> nn.Module:
    """Build compact DKM.

    Returns
    -------
    nn.Module
        Random-init compact DKM matcher.
    """

    return DKMMatcher()


def example_input_dkm() -> torch.Tensor:
    """Create a compact packed two-image input for DKM.

    Returns
    -------
    torch.Tensor
        Tensor ``(1, 6, 32, 32)``.
    """

    return torch.randn(1, 6, 32, 32)


def build_cotracker() -> nn.Module:
    """Build compact CoTracker.

    Returns
    -------
    nn.Module
        Random-init compact CoTracker.
    """

    return CoTrackerTiny()


def example_input_cotracker() -> torch.Tensor:
    """Create a compact video input for CoTracker.

    Returns
    -------
    torch.Tensor
        Tensor ``(1, 4, 3, 32, 32)``.
    """

    return torch.randn(1, 4, 3, 32, 32)


def build_tapir() -> nn.Module:
    """Build compact TAPIR.

    Returns
    -------
    nn.Module
        Random-init compact TAPIR.
    """

    return TAPIRTiny()


def example_input_tapir() -> torch.Tensor:
    """Create a compact video input for TAPIR.

    Returns
    -------
    torch.Tensor
        Tensor ``(1, 4, 3, 32, 32)``.
    """

    return torch.randn(1, 4, 3, 32, 32)


def build_xmem() -> nn.Module:
    """Build compact XMem.

    Returns
    -------
    nn.Module
        Random-init compact XMem.
    """

    return XMemTiny()


def example_input_xmem() -> torch.Tensor:
    """Create a compact RGB-plus-mask video input for XMem.

    Returns
    -------
    torch.Tensor
        Tensor ``(1, 4, 4, 32, 32)``.
    """

    x = torch.randn(1, 4, 4, 32, 32)
    x[:, 1:, 3] = 0.0
    return x


def build_d2net() -> nn.Module:
    """Build compact D2-Net.

    Returns
    -------
    nn.Module
        Random-init compact D2-Net.
    """

    return D2NetTiny()


def example_input_d2net() -> torch.Tensor:
    """Create a compact image input for D2-Net.

    Returns
    -------
    torch.Tensor
        Tensor ``(1, 3, 32, 32)``.
    """

    return torch.randn(1, 3, 32, 32)


def build_denseaspp() -> nn.Module:
    """Build compact DenseASPP.

    Returns
    -------
    nn.Module
        Random-init compact DenseASPP.
    """

    return DenseASPPTiny()


def example_input_denseaspp() -> torch.Tensor:
    """Create a compact image input for DenseASPP.

    Returns
    -------
    torch.Tensor
        Tensor ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


def build_refinenet() -> nn.Module:
    """Build compact RefineNet.

    Returns
    -------
    nn.Module
        Random-init compact RefineNet.
    """

    return RefineNetTiny()


def example_input_refinenet() -> torch.Tensor:
    """Create a compact image input for RefineNet.

    Returns
    -------
    torch.Tensor
        Tensor ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "DKM (Dense Kernelized Matching: GP global matcher + depthwise warp refinement)",
        "build_dkm",
        "example_input_dkm",
        "2023",
        "DC",
    ),
    (
        "CoTracker (joint transformer tracking for points in video)",
        "build_cotracker",
        "example_input_cotracker",
        "2024",
        "DC",
    ),
    (
        "TAPIR (per-frame point matching + temporal refinement tracker)",
        "build_tapir",
        "example_input_tapir",
        "2023",
        "DC",
    ),
    (
        "XMem (Atkinson-Shiffrin memory video object segmentation)",
        "build_xmem",
        "example_input_xmem",
        "2022",
        "DC",
    ),
    (
        "D2-Net (joint dense local-feature detection and description)",
        "build_d2net",
        "example_input_d2net",
        "2019",
        "DC",
    ),
    (
        "DenseASPP (densely connected atrous spatial pyramid segmentation)",
        "build_denseaspp",
        "example_input_denseaspp",
        "2018",
        "DC",
    ),
    (
        "RefineNet (multi-path high-resolution semantic segmentation)",
        "build_refinenet",
        "example_input_refinenet",
        "2017",
        "DC",
    ),
]
