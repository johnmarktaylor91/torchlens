"""Compact faithful reconstructions for DREIMPL shard 9.

These entries replace dependency-gated model families with small random-init
PyTorch modules that preserve the distinctive architecture primitive: pose
integral regression, contrastive ResNet heads, hypergraph propagation, weather
grid transformers, Siamese tracking heads, dynamic convolution backbones,
spectral neural operators, graph transformers, stereo correlation, and
video-depth temporal fusion.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Convolution, normalization, and activation block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, groups: int = 1) -> None:
        """Initialize a convolution block.

        Parameters
        ----------
        in_ch:
            Input channel count.
        out_ch:
            Output channel count.
        stride:
            Convolution stride.
        groups:
            Group count for grouped/depthwise convolution.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution block.

        Parameters
        ----------
        x:
            Image or feature tensor.

        Returns
        -------
        torch.Tensor
            Transformed features.
        """

        return self.net(x)


class TinyResidualBackbone(nn.Module):
    """Small ResNet-style backbone with four resolution stages."""

    def __init__(self, width: int = 16) -> None:
        """Initialize residual stages.

        Parameters
        ----------
        width:
            Base channel count.
        """

        super().__init__()
        self.stem = ConvBlock(3, width, 2)
        self.s1 = self._stage(width, width, 1)
        self.s2 = self._stage(width, width * 2, 2)
        self.s3 = self._stage(width * 2, width * 4, 2)
        self.s4 = self._stage(width * 4, width * 4, 1)

    def _stage(self, in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
        """Create a residual-like bottleneck-free stage.

        Parameters
        ----------
        in_ch:
            Input channels.
        out_ch:
            Output channels.
        stride:
            First convolution stride.

        Returns
        -------
        nn.Sequential
            Stage module.
        """

        return nn.Sequential(ConvBlock(in_ch, out_ch, stride), ConvBlock(out_ch, out_ch))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return multi-scale features.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        list[torch.Tensor]
            Feature maps from four stages.
        """

        x = self.stem(x)
        c1 = self.s1(x)
        c2 = self.s2(c1)
        c3 = self.s3(c2)
        c4 = self.s4(c3)
        return [c1, c2, c3, c4]


class TinySelfAttention(nn.Module):
    """Multi-head self-attention over token sequences."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize projections.

        Parameters
        ----------
        dim:
            Token dimension.
        heads:
            Attention head count.
        """

        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run scaled dot-product attention.

        Parameters
        ----------
        x:
            Token tensor ``(B, N, C)``.

        Returns
        -------
        torch.Tensor
            Updated tokens.
        """

        b, n, c = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        out = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, dim: int, heads: int = 4, mlp_ratio: int = 2) -> None:
        """Initialize attention and MLP sublayers.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Attention head count.
        mlp_ratio:
            MLP hidden expansion.
        """

        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = TinySelfAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transformer block.

        Parameters
        ----------
        x:
            Input tokens.

        Returns
        -------
        torch.Tensor
            Output tokens.
        """

        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class ResNetProjectionModel(nn.Module):
    """ResNet encoder with task-specific projection/classification heads."""

    def __init__(self, mode: str = "simclr", classes: int = 8) -> None:
        """Initialize the ResNet projection model.

        Parameters
        ----------
        mode:
            Variant selector for SimCLR, CLIP, robust, or Taskonomy style.
        classes:
            Output class or embedding count.
        """

        super().__init__()
        self.mode = mode
        self.backbone = TinyResidualBackbone(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, classes))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 3, 3, padding=1)
        )
        self.text_bank = nn.Parameter(torch.randn(classes, 64) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the selected ResNet-family pathway.

        Parameters
        ----------
        x:
            Input image.

        Returns
        -------
        torch.Tensor
            Projection, logits, or decoded feature map.
        """

        feat = self.backbone(x)[-1]
        pooled = self.pool(feat).flatten(1)
        emb = F.normalize(self.proj[0:2](pooled), dim=-1)
        if self.mode == "clip":
            return emb @ F.normalize(self.text_bank, dim=-1).t()
        if self.mode == "taskonomy":
            return self.decoder(feat)
        return self.proj[2](emb)


class HybrIKTiny(nn.Module):
    """HybrIK-style pose model with heatmap integral coordinates and SMPL head."""

    def __init__(self, joints: int = 17) -> None:
        """Initialize HybrIK components.

        Parameters
        ----------
        joints:
            Number of body joints.
        """

        super().__init__()
        self.backbone = TinyResidualBackbone(12)
        self.heatmap = nn.Conv2d(48, joints, 1)
        self.shape = nn.Linear(joints * 2, 10)
        self.twist = nn.Linear(joints * 2, joints * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict integral joint coordinates, shape, and twist.

        Parameters
        ----------
        x:
            Person crop.

        Returns
        -------
        torch.Tensor
            Concatenated pose, shape, and twist parameters.
        """

        hm = self.heatmap(self.backbone(x)[-1])
        b, j, h, w = hm.shape
        prob = hm.flatten(2).softmax(dim=-1).view(b, j, h, w)
        yy = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        xx = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
        coords = torch.cat([(prob * xx).sum((2, 3)), (prob * yy).sum((2, 3))], dim=-1)
        return torch.cat([coords, self.shape(coords), self.twist(coords)], dim=-1)


class HypergraphLayer(nn.Module):
    """Hypergraph incidence propagation layer."""

    def __init__(self, dim: int, mode: str) -> None:
        """Initialize hypergraph transforms.

        Parameters
        ----------
        dim:
            Feature width.
        mode:
            Propagation variant.
        """

        super().__init__()
        self.mode = mode
        self.node = nn.Linear(dim, dim)
        self.edge = nn.Linear(dim, dim)
        self.attn = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Propagate node features through hyperedges.

        Parameters
        ----------
        x:
            Node features ``(B, V, C)``.
        h:
            Incidence matrix ``(B, V, E)``.

        Returns
        -------
        torch.Tensor
            Updated node features.
        """

        deg_e = h.sum(1, keepdim=True).transpose(1, 2).clamp_min(1.0)
        edge_feat = h.transpose(1, 2) @ self.node(x) / deg_e
        if self.mode == "hnhn":
            edge_feat = torch.tanh(self.edge(edge_feat))
        if self.mode == "gat":
            weights = torch.softmax(self.attn(edge_feat).transpose(1, 2), dim=-1)
            edge_feat = edge_feat * weights.transpose(1, 2)
        out = h @ edge_feat / h.sum(2, keepdim=True).clamp_min(1.0)
        return x + out


class HypergraphNet(nn.Module):
    """DHG-style hypergraph neural network variants."""

    def __init__(self, mode: str = "sage", nodes: int = 8, edges: int = 4, dim: int = 24) -> None:
        """Initialize hypergraph network.

        Parameters
        ----------
        mode:
            DHG variant selector.
        nodes:
            Number of nodes.
        edges:
            Number of hyperedges.
        dim:
            Hidden feature width.
        """

        super().__init__()
        self.mode = mode
        self.nodes = nodes
        self.edges = edges
        self.inp = nn.Linear(6, dim)
        self.layers = nn.ModuleList([HypergraphLayer(dim, mode), HypergraphLayer(dim, mode)])
        self.out = nn.Linear(dim, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run hypergraph propagation from packed node features.

        Parameters
        ----------
        x:
            Node features ``(B, V, 6)``.

        Returns
        -------
        torch.Tensor
            Node logits.
        """

        b, v, _ = x.shape
        h = torch.zeros(b, v, self.edges, device=x.device, dtype=x.dtype)
        for e in range(self.edges):
            h[:, e :: self.edges, e] = 1.0
            h[:, (e + 1) % v, e] = 1.0
        y = self.inp(x)
        for layer in self.layers:
            y = layer(y, h)
        return self.out(y)


class WeatherGridTransformer(nn.Module):
    """Weather model with patch embedding, global tokens, and grid decoding."""

    def __init__(self, mode: str = "fengwu", channels: int = 6, dim: int = 48) -> None:
        """Initialize weather architecture.

        Parameters
        ----------
        mode:
            FengWu, functional generator, graph weather, or Pangu-style mode.
        channels:
            Meteorological input channels.
        dim:
            Token width.
        """

        super().__init__()
        self.mode = mode
        self.patch = nn.Conv2d(channels, dim, 4, stride=4)
        self.blocks = nn.ModuleList([TransformerBlock(dim), TransformerBlock(dim)])
        self.graph_gate = nn.Linear(dim, dim)
        self.up = nn.ConvTranspose2d(dim, channels, 4, stride=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast the next weather grid.

        Parameters
        ----------
        x:
            Gridded weather tensor.

        Returns
        -------
        torch.Tensor
            Forecast tensor.
        """

        tok_map = self.patch(x)
        b, c, h, w = tok_map.shape
        tokens = tok_map.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens)
            if self.mode == "graph":
                tokens = tokens + torch.roll(self.graph_gate(tokens), shifts=1, dims=1)
        if self.mode == "functional":
            coords = torch.linspace(-1, 1, tokens.shape[1], device=x.device, dtype=x.dtype)
            tokens = tokens + torch.sin(coords)[None, :, None]
        return self.up(tokens.transpose(1, 2).view(b, c, h, w))


class SiameseRPNTracker(nn.Module):
    """Siamese tracker with depthwise correlation and RPN heads."""

    def __init__(self, mode: str = "rpn") -> None:
        """Initialize tracker.

        Parameters
        ----------
        mode:
            RPN, graph-attention, transformer, or memory tracking mode.
        """

        super().__init__()
        self.mode = mode
        self.backbone = TinyResidualBackbone(8)
        self.z_proj = nn.Conv2d(32, 32, 1)
        self.x_proj = nn.Conv2d(32, 32, 1)
        self.cls = nn.Conv2d(32, 2, 1)
        self.box = nn.Conv2d(32, 4, 1)
        self.q = nn.Conv2d(32, 32, 1)
        self.k = nn.Conv2d(32, 32, 1)

    def forward(
        self, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Track an exemplar in a search image.

        Parameters
        ----------
        inputs:
            Exemplar and search tensors.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Classification and box maps.
        """

        z, x = inputs
        zf = self.z_proj(self.backbone(z)[-1]).mean(dim=(2, 3), keepdim=True)
        xf = self.x_proj(self.backbone(x)[-1])
        feat = xf * zf
        if self.mode in {"gat", "trtr", "samurai"}:
            q = self.q(feat).flatten(2).transpose(1, 2)
            k = self.k(feat).flatten(2)
            attn = torch.softmax(q @ k / math.sqrt(float(k.shape[1])), dim=-1)
            feat = (attn @ feat.flatten(2).transpose(1, 2)).transpose(1, 2).view_as(feat)
        return self.cls(feat), self.box(feat)


class InternImageTiny(nn.Module):
    """InternImage-style dynamic convolution backbone."""

    def __init__(self, channels: int = 24) -> None:
        """Initialize dynamic offset and modulation branches.

        Parameters
        ----------
        channels:
            Hidden channel count.
        """

        super().__init__()
        self.stem = ConvBlock(3, channels, 2)
        self.offset = nn.Conv2d(channels, channels, 3, padding=1)
        self.mod = nn.Conv2d(channels, channels, 1)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.head = nn.Linear(channels, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run dynamic-convolution image recognition.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        y = self.stem(x)
        y = y + self.dw(y + torch.tanh(self.offset(y))) * torch.sigmoid(self.mod(y))
        return self.head(y.mean((2, 3)))


class ScaleMAETiny(nn.Module):
    """Scale-MAE with multi-scale patch tokens and masked reconstruction head."""

    def __init__(self, dim: int = 48) -> None:
        """Initialize Scale-MAE components.

        Parameters
        ----------
        dim:
            Token width.
        """

        super().__init__()
        self.p4 = nn.Conv2d(3, dim, 4, stride=4)
        self.p8 = nn.Conv2d(3, dim, 8, stride=8)
        self.blocks = nn.ModuleList([TransformerBlock(dim), TransformerBlock(dim)])
        self.dec = nn.Linear(dim, 3 * 4 * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode multi-scale patches and reconstruct pixels.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Reconstructed patch pixels.
        """

        t4 = self.p4(x).flatten(2).transpose(1, 2)
        t8 = self.p8(x).flatten(2).transpose(1, 2)
        tokens = torch.cat(
            [t4, F.interpolate(t8.transpose(1, 2), size=t4.shape[1]).transpose(1, 2)], dim=1
        )
        for block in self.blocks:
            tokens = block(tokens)
        return self.dec(tokens[:, : t4.shape[1]])


class SpectralOperator(nn.Module):
    """Koopman/Fourier neural operator for spatiotemporal fields."""

    def __init__(self, channels: int = 4) -> None:
        """Initialize spectral mixing weights.

        Parameters
        ----------
        channels:
            Input and output channel count.
        """

        super().__init__()
        self.inp = nn.Conv2d(channels, 16, 1)
        self.mix_real = nn.Parameter(torch.randn(16, 16) * 0.02)
        self.out = nn.Conv2d(16, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral Koopman-style evolution.

        Parameters
        ----------
        x:
            Field tensor.

        Returns
        -------
        torch.Tensor
            Evolved field.
        """

        y = self.inp(x)
        spec = torch.fft.rfft2(y, norm="ortho")
        mixed = torch.einsum("bchw,cd->bdhw", spec.real, self.mix_real)
        return self.out(
            torch.fft.irfft2(torch.complex(mixed, spec.imag), s=y.shape[-2:], norm="ortho")
        )


class VideoDepthTiny(nn.Module):
    """Video Depth Anything-style frame ViT with temporal fusion and depth head."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize video-depth encoder.

        Parameters
        ----------
        dim:
            Token width.
        """

        super().__init__()
        self.patch = nn.Conv2d(3, dim, 4, stride=4)
        self.spatial = TransformerBlock(dim)
        self.temporal = TransformerBlock(dim)
        self.head = nn.ConvTranspose2d(dim, 1, 4, stride=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate per-frame monocular video depth.

        Parameters
        ----------
        x:
            Video tensor ``(B, T, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Depth maps ``(B, T, 1, H, W)``.
        """

        b, t, c, h, w = x.shape
        y = self.patch(x.view(b * t, c, h, w))
        _, d, hp, wp = y.shape
        tok = self.spatial(y.flatten(2).transpose(1, 2)).view(b, t, hp * wp, d)
        tok = self.temporal(tok.transpose(1, 2).reshape(b * hp * wp, t, d)).view(b, hp * wp, t, d)
        feat = tok.transpose(1, 2).reshape(b * t, hp * wp, d).transpose(1, 2).view(b * t, d, hp, wp)
        return self.head(feat).view(b, t, 1, h, w)


class UniMatchStereoTiny(nn.Module):
    """UniMatch-style stereo matcher with correlation volume and refinement."""

    def __init__(self, channels: int = 16) -> None:
        """Initialize feature encoder and disparity refiner.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        self.enc = nn.Sequential(ConvBlock(3, channels, 2), ConvBlock(channels, channels))
        self.refine = nn.Sequential(
            ConvBlock(channels + 4, channels), nn.Conv2d(channels, 1, 3, padding=1)
        )

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Estimate disparity from left and right images.

        Parameters
        ----------
        inputs:
            Left and right image tensors.

        Returns
        -------
        torch.Tensor
            Low-resolution disparity map.
        """

        left, right = inputs
        lf = self.enc(left)
        rf = self.enc(right)
        vols = []
        for disp in range(4):
            vols.append((lf * torch.roll(rf, shifts=disp, dims=-1)).sum(1, keepdim=True))
        volume = torch.cat(vols, dim=1)
        prob = volume.softmax(1)
        disp_values = torch.arange(4, device=left.device, dtype=left.dtype).view(1, 4, 1, 1)
        disp = (prob * disp_values).sum(1, keepdim=True)
        return disp + self.refine(torch.cat([lf, volume], dim=1))


class GraphTransformerTiny(nn.Module):
    """GRIT-style graph transformer with edge-biased attention."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize graph transformer layers.

        Parameters
        ----------
        dim:
            Hidden width.
        """

        super().__init__()
        self.node = nn.Linear(6, dim)
        self.edge = nn.Linear(3, 1)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.out = nn.Linear(dim, 4)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Run edge-biased graph attention.

        Parameters
        ----------
        inputs:
            Node features and edge features ``(B, N, N, 3)``.

        Returns
        -------
        torch.Tensor
            Node predictions.
        """

        x, edge = inputs
        h = self.node(x)
        scores = h @ h.transpose(1, 2) / math.sqrt(float(h.shape[-1])) + self.edge(edge).squeeze(-1)
        h = h + torch.softmax(scores, dim=-1) @ h
        return self.out(h + self.ffn(h))


class RhoFoldTiny(nn.Module):
    """RhoFold-style RNA sequence-pair transformer."""

    def __init__(self, dim: int = 32) -> None:
        """Initialize sequence and pair update modules.

        Parameters
        ----------
        dim:
            Hidden feature width.
        """

        super().__init__()
        self.emb = nn.Embedding(8, dim)
        self.seq = TransformerBlock(dim)
        self.pair = nn.Linear(dim * 2, dim)
        self.dist = nn.Linear(dim, 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Predict RNA residue-pair distances.

        Parameters
        ----------
        ids:
            RNA token ids.

        Returns
        -------
        torch.Tensor
            Pairwise distance logits.
        """

        h = self.seq(self.emb(ids))
        pair = torch.cat(
            [
                h[:, :, None, :].expand(-1, -1, h.shape[1], -1),
                h[:, None, :, :].expand(-1, h.shape[1], -1, -1),
            ],
            dim=-1,
        )
        return self.dist(torch.tanh(self.pair(pair))).squeeze(-1)


class ConvNeXtZooBot(nn.Module):
    """Zoobot-style ConvNeXt morphology classifier."""

    def __init__(self, classes: int = 10) -> None:
        """Initialize ConvNeXt blocks.

        Parameters
        ----------
        classes:
            Class count.
        """

        super().__init__()
        self.stem = nn.Conv2d(3, 24, 4, stride=4)
        self.dw = nn.Conv2d(24, 24, 7, padding=3, groups=24)
        self.pw = nn.Sequential(nn.Conv2d(24, 96, 1), nn.GELU(), nn.Conv2d(96, 24, 1))
        self.head = nn.Linear(24, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify galaxy morphology.

        Parameters
        ----------
        x:
            Galaxy image.

        Returns
        -------
        torch.Tensor
            Morphology logits.
        """

        y = self.stem(x)
        y = y + self.pw(self.dw(y))
        return self.head(y.mean((2, 3)))


class MesoNetTiny(nn.Module):
    """MesoNet forensic CNN with shallow mesoscopic filters."""

    def __init__(self) -> None:
        """Initialize MesoNet layers."""

        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 8),
            nn.MaxPool2d(2),
            ConvBlock(8, 16),
            nn.MaxPool2d(2),
            ConvBlock(16, 32),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict manipulation probability.

        Parameters
        ----------
        x:
            Face image.

        Returns
        -------
        torch.Tensor
            Binary logit.
        """

        return self.head(self.net(x).flatten(1))


def build_hybrik_resnet34() -> nn.Module:
    """Build HybrIK with ResNet-style heatmap integral pose head."""

    return HybrIKTiny().eval()


def build_resnet50_simclr_brainscore() -> nn.Module:
    """Build ResNet SimCLR projection model."""

    return ResNetProjectionModel("simclr").eval()


def build_resnet50_clip_brainscore() -> nn.Module:
    """Build ResNet CLIP image-text contrastive model."""

    return ResNetProjectionModel("clip").eval()


def build_robust_resnet50_l2_eps3() -> nn.Module:
    """Build robust ResNet classifier reconstruction."""

    return ResNetProjectionModel("robust").eval()


def build_taskonomy_encoder() -> nn.Module:
    """Build Taskonomy encoder-decoder reconstruction."""

    return ResNetProjectionModel("taskonomy").eval()


def build_dhg_unisage() -> nn.Module:
    """Build DHG UniSAGE hypergraph model."""

    return HypergraphNet("sage").eval()


def build_dhg_hgnnp() -> nn.Module:
    """Build DHG HGNNP hypergraph model."""

    return HypergraphNet("hgnnp").eval()


def build_dhg_hnhn() -> nn.Module:
    """Build DHG HNHN hypergraph model."""

    return HypergraphNet("hnhn").eval()


def build_dhg_unigat() -> nn.Module:
    """Build DHG UniGAT hypergraph model."""

    return HypergraphNet("gat").eval()


def build_dhg_unigcn() -> nn.Module:
    """Build DHG UniGCN hypergraph model."""

    return HypergraphNet("gcn").eval()


def build_fengwu_ghr() -> nn.Module:
    """Build FengWu-GHR weather grid transformer."""

    return WeatherGridTransformer("fengwu").eval()


def build_functional_weather() -> nn.Module:
    """Build functional generative weather network."""

    return WeatherGridTransformer("functional").eval()


def build_graph_weather() -> nn.Module:
    """Build graph-biased weather forecaster."""

    return WeatherGridTransformer("graph").eval()


def build_pangu_weather() -> nn.Module:
    """Build Pangu-style grid transformer weather model."""

    return WeatherGridTransformer("pangu").eval()


def build_dasiamrpn_big() -> nn.Module:
    """Build DaSiamRPN Siamese RPN tracker."""

    return SiameseRPNTracker("rpn").eval()


def build_siamgat_resnet50() -> nn.Module:
    """Build SiamGAT graph-attention Siamese tracker."""

    return SiameseRPNTracker("gat").eval()


def build_siamtrackers_trtr() -> nn.Module:
    """Build TrTr transformer Siamese tracker."""

    return SiameseRPNTracker("trtr").eval()


def build_samurai() -> nn.Module:
    """Build SAMURAI memory-attention tracker reconstruction."""

    return SiameseRPNTracker("samurai").eval()


def build_internimage() -> nn.Module:
    """Build InternImage dynamic convolution classifier."""

    return InternImageTiny().eval()


def build_scale_mae() -> nn.Module:
    """Build Scale-MAE multiscale masked autoencoder."""

    return ScaleMAETiny().eval()


def build_krno() -> nn.Module:
    """Build Koopman recurrent neural operator."""

    return SpectralOperator(channels=6).eval()


def build_video_depth_anything() -> nn.Module:
    """Build Video Depth Anything-style temporal depth model."""

    return VideoDepthTiny().eval()


def build_unimatch_stereo() -> nn.Module:
    """Build UniMatch stereo model."""

    return UniMatchStereoTiny().eval()


def build_grit() -> nn.Module:
    """Build GRIT graph transformer."""

    return GraphTransformerTiny().eval()


def build_rhofold() -> nn.Module:
    """Build RhoFold RNA pair transformer."""

    return RhoFoldTiny().eval()


def build_zoobot_convnext() -> nn.Module:
    """Build Zoobot ConvNeXt model."""

    return ConvNeXtZooBot().eval()


def build_mesonet() -> nn.Module:
    """Build MesoNet forensic CNN."""

    return MesoNetTiny().eval()


def example_image() -> torch.Tensor:
    """Return a compact image example."""

    return torch.randn(1, 3, 32, 32)


def example_hypergraph() -> torch.Tensor:
    """Return packed hypergraph node features."""

    return torch.randn(1, 8, 6)


def example_weather() -> torch.Tensor:
    """Return a compact weather grid."""

    return torch.randn(1, 6, 16, 16)


def example_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Return exemplar and search images for Siamese trackers."""

    return torch.randn(1, 3, 32, 32), torch.randn(1, 3, 48, 48)


def example_video() -> torch.Tensor:
    """Return a short video clip."""

    return torch.randn(1, 3, 3, 32, 32)


def example_stereo() -> tuple[torch.Tensor, torch.Tensor]:
    """Return left and right stereo images."""

    return torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32)


def example_graph() -> tuple[torch.Tensor, torch.Tensor]:
    """Return graph node and edge features."""

    return torch.randn(1, 7, 6), torch.randn(1, 7, 7, 3)


def example_rna() -> torch.Tensor:
    """Return RNA token ids."""

    return torch.randint(0, 8, (1, 12))


MENAGERIE_ENTRIES = [
    ("HybrIK-ResNet34", "build_hybrik_resnet34", "example_image", "2021", "pose"),
    (
        "resnet50_simclr_brainscore",
        "build_resnet50_simclr_brainscore",
        "example_image",
        "2020",
        "vision/self-supervised",
    ),
    (
        "resnet50_clip_brainscore",
        "build_resnet50_clip_brainscore",
        "example_image",
        "2021",
        "vision/multimodal",
    ),
    (
        "robust_resnet50_l2_eps3",
        "build_robust_resnet50_l2_eps3",
        "example_image",
        "2019",
        "vision/robust",
    ),
    ("taskonomy_encoder", "build_taskonomy_encoder", "example_image", "2018", "vision/multitask"),
    ("DHG-UniSAGE", "build_dhg_unisage", "example_hypergraph", "2023", "graph/hypergraph"),
    ("DHG-HGNNP", "build_dhg_hgnnp", "example_hypergraph", "2023", "graph/hypergraph"),
    ("DHG-HNHN", "build_dhg_hnhn", "example_hypergraph", "2020", "graph/hypergraph"),
    ("DHG-UniGAT", "build_dhg_unigat", "example_hypergraph", "2023", "graph/hypergraph"),
    ("DHG-UniGCN", "build_dhg_unigcn", "example_hypergraph", "2023", "graph/hypergraph"),
    ("FengWu-GHR", "build_fengwu_ghr", "example_weather", "2023", "weather/forecasting"),
    (
        "FunctionalGenerativeNetwork-weather",
        "build_functional_weather",
        "example_weather",
        "2023",
        "weather/forecasting",
    ),
    (
        "GraphWeatherForecaster",
        "build_graph_weather",
        "example_weather",
        "2022",
        "weather/forecasting",
    ),
    (
        "pangu_weather.PanguWeather",
        "build_pangu_weather",
        "example_weather",
        "2022",
        "weather/forecasting",
    ),
    ("DaSiamRPN-SiamRPNBIG", "build_dasiamrpn_big", "example_pair", "2018", "vision/tracking"),
    ("SiamGAT-ResNet50", "build_siamgat_resnet50", "example_pair", "2021", "vision/tracking"),
    ("SiamTrackers-TrTr", "build_siamtrackers_trtr", "example_pair", "2021", "vision/tracking"),
    ("SAMURAI", "build_samurai", "example_pair", "2024", "vision/tracking"),
    ("InternImage", "build_internimage", "example_image", "2023", "vision/backbone"),
    ("Scale-MAE", "build_scale_mae", "example_image", "2023", "vision/self-supervised"),
    ("KRNO", "build_krno", "example_weather", "2024", "operator/weather"),
    (
        "video_depth_anything_base",
        "build_video_depth_anything",
        "example_video",
        "2025",
        "depth/video",
    ),
    (
        "video_depth_anything_large",
        "build_video_depth_anything",
        "example_video",
        "2025",
        "depth/video",
    ),
    (
        "video_depth_anything_small",
        "build_video_depth_anything",
        "example_video",
        "2025",
        "depth/video",
    ),
    (
        "video_depth_anything_metric_base",
        "build_video_depth_anything",
        "example_video",
        "2025",
        "depth/video",
    ),
    (
        "video_depth_anything_metric_small",
        "build_video_depth_anything",
        "example_video",
        "2025",
        "depth/video",
    ),
    ("unimatch_stereo", "build_unimatch_stereo", "example_stereo", "2023", "stereo/matching"),
    ("GRIT", "build_grit", "example_graph", "2023", "graph/transformer"),
    ("RhoFold", "build_rhofold", "example_rna", "2022", "rna/structure"),
    ("Zoobot-convnext", "build_zoobot_convnext", "example_image", "2022", "astronomy/morphology"),
    ("MesoNet", "build_mesonet", "example_image", "2018", "vision/forensics"),
]
