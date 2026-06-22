"""Compact faithful PaddleSeg and geometry classics for dependency-gated entries.

The PaddleSeg entries below are random-initialized PyTorch reconstructions of
the load-bearing mechanisms used by the corresponding PaddleSeg/Paper models:
ASPP decoders, non-local/global-context blocks, EM attention, context encoding,
ENet/ESPNet real-time bottlenecks, transformer decoders, multi-branch real-time
segmentation, matting trimap fusion, and U-Net variants.  The geometry entries
keep the factorized implicit global convolution and permutation-equivariant
multi-view prediction primitives.

Paper: PaddleSeg toolkit, Liu et al. 2021; individual entries cite their
architecture paper in the catalog notes emitted for this reimplementation batch.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Convolution, batch normalization, and activation helper."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        """Initialize a compact convolutional block.

        Parameters
        ----------
        in_ch:
            Input channel count.
        out_ch:
            Output channel count.
        kernel:
            Kernel size.
        stride:
            Convolution stride.
        dilation:
            Dilation factor.
        groups:
            Convolution groups.
        """

        super().__init__()
        pad = dilation * (kernel // 2)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, pad, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution, normalization, and activation.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Activated feature map.
        """

        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparable(nn.Module):
    """Depthwise-separable convolution used by mobile segmentation models."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1) -> None:
        """Initialize depthwise and pointwise convolutions.

        Parameters
        ----------
        in_ch:
            Input channel count.
        out_ch:
            Output channel count.
        stride:
            Depthwise stride.
        dilation:
            Depthwise dilation.
        """

        super().__init__()
        self.depth = ConvBNAct(in_ch, in_ch, 3, stride, dilation, groups=in_ch)
        self.point = ConvBNAct(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise-separable convolution.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """

        return self.point(self.depth(x))


class TinyBackbone(nn.Module):
    """Small feature pyramid backbone returning four semantic scales."""

    def __init__(self, width: int = 16) -> None:
        """Initialize a compact four-stage backbone.

        Parameters
        ----------
        width:
            Base channel width.
        """

        super().__init__()
        self.s1 = ConvBNAct(3, width, 3, 1)
        self.s2 = ConvBNAct(width, width * 2, 3, 2)
        self.s3 = ConvBNAct(width * 2, width * 3, 3, 2)
        self.s4 = ConvBNAct(width * 3, width * 4, 3, 2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract a four-level feature pyramid.

        Parameters
        ----------
        x:
            Input image.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Features at progressively lower spatial resolutions.
        """

        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        return f1, f2, f3, f4


class ASPP(nn.Module):
    """Atrous spatial pyramid pooling module."""

    def __init__(self, in_ch: int, out_ch: int = 16, rates: tuple[int, ...] = (1, 2, 4)) -> None:
        """Initialize dilated ASPP branches and image pooling.

        Parameters
        ----------
        in_ch:
            Input channel count.
        out_ch:
            Branch output channel count.
        rates:
            Dilation rates.
        """

        super().__init__()
        self.branches = nn.ModuleList([ConvBNAct(in_ch, out_ch, 3, dilation=r) for r in rates])
        self.pool = ConvBNAct(in_ch, out_ch, 1)
        self.project = ConvBNAct(out_ch * (len(rates) + 1), out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate multi-rate and image-level context.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            ASPP feature map.
        """

        pooled = F.interpolate(
            self.pool(F.adaptive_avg_pool2d(x, 1)),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.project(torch.cat([branch(x) for branch in self.branches] + [pooled], dim=1))


class NonLocalBlock(nn.Module):
    """Disentangled non-local attention with centered pairwise affinity."""

    def __init__(self, channels: int) -> None:
        """Initialize query, key, value, and output projections.

        Parameters
        ----------
        channels:
            Feature channel count.
        """

        super().__init__()
        inner = max(4, channels // 2)
        self.theta = nn.Conv2d(channels, inner, 1)
        self.phi = nn.Conv2d(channels, inner, 1)
        self.g = nn.Conv2d(channels, inner, 1)
        self.out = nn.Conv2d(inner, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply centered non-local aggregation.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Context-enhanced feature map.
        """

        bsz, _, height, width = x.shape
        q = self.theta(x).flatten(2).transpose(1, 2)
        k = self.phi(x).flatten(2)
        q = q - q.mean(dim=1, keepdim=True)
        k = k - k.mean(dim=2, keepdim=True)
        attn = torch.softmax(torch.bmm(q, k) / (q.shape[-1] ** 0.5), dim=-1)
        v = self.g(x).flatten(2).transpose(1, 2)
        y = torch.bmm(attn, v).transpose(1, 2).reshape(bsz, -1, height, width)
        return x + self.out(y)


class EMABlock(nn.Module):
    """Expectation-maximization attention block with learned bases."""

    def __init__(self, channels: int, bases: int = 4, steps: int = 2) -> None:
        """Initialize EMA bases.

        Parameters
        ----------
        channels:
            Feature channel count.
        bases:
            Number of EM bases.
        steps:
            Number of compact EM iterations.
        """

        super().__init__()
        self.mu = nn.Parameter(torch.randn(1, channels, bases))
        self.steps = steps
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run EM attention reconstruction over spatial pixels.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            EMA-enhanced feature map.
        """

        bsz, channels, height, width = x.shape
        feats = x.flatten(2)
        mu = F.normalize(self.mu.expand(bsz, -1, -1), dim=1)
        for _ in range(self.steps):
            z = torch.softmax(torch.bmm(feats.transpose(1, 2), mu), dim=-1)
            mu = F.normalize(torch.bmm(feats, z) / (z.sum(dim=1, keepdim=True) + 1e-6), dim=1)
        recon = torch.bmm(mu, z.transpose(1, 2)).reshape(bsz, channels, height, width)
        return x + self.project(recon)


class EncodingBlock(nn.Module):
    """Context Encoding Network codeword residual aggregation."""

    def __init__(self, channels: int, codewords: int = 6) -> None:
        """Initialize codewords, scales, and channel gate.

        Parameters
        ----------
        channels:
            Feature channel count.
        codewords:
            Number of learned visual codewords.
        """

        super().__init__()
        self.codewords = nn.Parameter(torch.randn(codewords, channels))
        self.scales = nn.Parameter(torch.ones(codewords))
        self.gate = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual encoding and channel recalibration.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Gated feature map.
        """

        bsz, channels, _, _ = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        residual = tokens.unsqueeze(2) - self.codewords.view(1, 1, -1, channels)
        dist = residual.pow(2).sum(-1)
        assign = torch.softmax(-self.scales.view(1, 1, -1).abs() * dist, dim=2)
        encoded = (assign.unsqueeze(-1) * residual).sum(dim=(1, 2))
        gamma = torch.sigmoid(self.gate(encoded)).view(bsz, channels, 1, 1)
        return x + x * gamma


class HarDBlock(nn.Module):
    """HarDNet harmonic dense block with sparse power-of-two links."""

    def __init__(self, channels: int, layers: int = 4, growth: int = 8) -> None:
        """Initialize harmonically connected layers.

        Parameters
        ----------
        channels:
            Input and output channel count.
        layers:
            Number of internal layers.
        growth:
            Growth channels per layer.
        """

        super().__init__()
        self.layers = nn.ModuleList()
        in_counts = []
        for idx in range(layers):
            links = [idx - step for step in (1, 2, 4) if idx - step >= 0]
            in_count = channels + growth * len(links)
            in_counts.append(in_count)
            self.layers.append(ConvBNAct(in_count, growth, 3))
        self.out = ConvBNAct(channels + growth * layers, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply harmonic dense aggregation.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        torch.Tensor
            Output feature map.
        """

        feats = [x]
        made: list[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            links = [idx - step for step in (1, 2, 4) if idx - step >= 0]
            src = [x] + [made[link] for link in links]
            made.append(layer(torch.cat(src, dim=1)))
            feats.append(made[-1])
        return self.out(torch.cat(feats, dim=1))


class PSPModule(nn.Module):
    """PSPNet pyramid scene pooling over fixed spatial bins."""

    def __init__(self, channels: int, out_ch: int) -> None:
        """Initialize pooled-bin projections.

        Parameters
        ----------
        channels:
            Input channel count.
        out_ch:
            Output channel count.
        """

        super().__init__()
        self.bins = (1, 2, 3, 6)
        self.proj = nn.ModuleList([ConvBNAct(channels, out_ch, 1) for _ in self.bins])
        self.out = ConvBNAct(channels + out_ch * len(self.bins), out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate context from PSPNet pyramid pooling bins."""

        pooled = [
            F.interpolate(proj(F.adaptive_avg_pool2d(x, bin_size)), x.shape[-2:], mode="bilinear")
            for proj, bin_size in zip(self.proj, self.bins, strict=True)
        ]
        return self.out(torch.cat([x, *pooled], dim=1))


class SegNetUnpoolDecoder(nn.Module):
    """SegNet max-pooling indices plus max-unpool reconstruction decoder."""

    def __init__(self, channels: int, classes: int) -> None:
        """Initialize encoder, decoder, and segmentation head."""

        super().__init__()
        self.enc = ConvBNAct(3, channels, 3)
        self.dec = ConvBNAct(channels, channels, 3)
        self.head = nn.Conv2d(channels, classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode with pooling indices and decode with max unpooling."""

        feat = self.enc(image)
        pooled, indices = F.max_pool2d(feat, 2, 2, return_indices=True)
        up = F.max_unpool2d(pooled, indices, 2, 2, output_size=feat.shape)
        return self.head(self.dec(up))


class PPStrideFormerHead(nn.Module):
    """PP-MobileSeg StrideFormer, AAM, and VIM compact head."""

    def __init__(self, width: int, classes: int) -> None:
        """Initialize strided SEA attention, AAM voting, and valid interpolation."""

        super().__init__()
        self.strided_sea = nn.Conv2d(width * 4, width * 4, 3, stride=1, padding=1, groups=width * 4)
        self.semantic_vote = nn.Conv2d(width * 4, width, 1)
        self.detail_filter = nn.Conv2d(width, width, 1)
        self.aam = nn.Conv2d(width * 2, width, 1)
        self.classifier = nn.Conv2d(width, classes, 1)

    def forward(self, p2: torch.Tensor, p4_raw: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Apply StrideFormer context, AAM feature filtering, and VIM-style upsample."""

        semantic = self.semantic_vote(self.strided_sea(p4_raw))
        semantic = F.interpolate(semantic, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        detail = self.detail_filter(p2)
        vote = torch.sigmoid(semantic)
        fused = self.aam(torch.cat((detail * vote, semantic), dim=1))
        logits = self.classifier(fused)
        present = torch.sigmoid(F.adaptive_max_pool2d(logits, 1))
        valid_logits = logits * present
        return F.interpolate(
            valid_logits, size=image.shape[-2:], mode="bilinear", align_corners=False
        )


class RTFormerFusion(nn.Module):
    """RTFormer dual-resolution linear attention with cross-resolution fusion."""

    def __init__(self, width: int, classes: int) -> None:
        """Initialize high/low projections and linear attention maps."""

        super().__init__()
        self.high = nn.Conv2d(width, width, 1)
        self.low = nn.Conv2d(width * 4, width, 1)
        self.q = nn.Conv2d(width, width, 1)
        self.k = nn.Conv2d(width, width, 1)
        self.v = nn.Conv2d(width, width, 1)
        self.out = nn.Conv2d(width, classes, 1)

    def forward(self, p1: torch.Tensor, p4_raw: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Fuse high-resolution detail with low-resolution linear attention context."""

        high = self.high(p1)
        low = self.low(p4_raw)
        q = F.elu(self.q(high)).flatten(2).transpose(1, 2) + 1.0
        k = F.elu(self.k(low)).flatten(2).transpose(1, 2) + 1.0
        v = self.v(low).flatten(2).transpose(1, 2)
        kv = torch.bmm(k.transpose(1, 2), v)
        denom = torch.bmm(q, k.sum(1, keepdim=True).transpose(1, 2)).clamp_min(1e-4)
        ctx = torch.bmm(q, kv) / denom
        ctx = ctx.transpose(1, 2).reshape_as(high)
        return F.interpolate(
            self.out(high + ctx), image.shape[-2:], mode="bilinear", align_corners=False
        )


class SeaFormerBlock(nn.Module):
    """SeaFormer squeeze-enhanced axial attention and detail enhancement."""

    def __init__(self, width: int, classes: int) -> None:
        """Initialize axial squeeze projections and detail branch."""

        super().__init__()
        self.row = nn.Conv2d(width * 4, width, (1, 3), padding=(0, 1))
        self.col = nn.Conv2d(width * 4, width, (3, 1), padding=(1, 0))
        self.detail = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.out = nn.Conv2d(width, classes, 1)

    def forward(self, p1: torch.Tensor, p4_raw: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Combine squeeze axial context with enhanced spatial detail."""

        row = self.row(p4_raw.mean(2, keepdim=True)).expand(-1, -1, p4_raw.shape[2], -1)
        col = self.col(p4_raw.mean(3, keepdim=True)).expand(-1, -1, -1, p4_raw.shape[3])
        axial = F.interpolate(torch.sigmoid(row + col), p1.shape[-2:], mode="bilinear")
        detail = p1 + self.detail(p1)
        return F.interpolate(self.out(detail * axial), image.shape[-2:], mode="bilinear")


class SegNeXtMSCA(nn.Module):
    """SegNeXt multi-scale convolutional attention block."""

    def __init__(self, channels: int) -> None:
        """Initialize multi-scale strip depthwise convolutions."""

        super().__init__()
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.dw7h = nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels)
        self.dw7v = nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels)
        self.dw11h = nn.Conv2d(channels, channels, (1, 11), padding=(0, 5), groups=channels)
        self.dw11v = nn.Conv2d(channels, channels, (11, 1), padding=(5, 0), groups=channels)
        self.mix = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MSCA spatial gating from multi-scale convolutional features."""

        attn = self.dw5(x) + self.dw7v(self.dw7h(x)) + self.dw11v(self.dw11h(x))
        return x * torch.sigmoid(self.mix(attn))


class PatchTransformerSeg(nn.Module):
    """Pure patch Transformer segmentation path for Segmenter and SETR."""

    def __init__(self, patch: int, dim: int, classes: int, mask_decoder: bool) -> None:
        """Initialize patch embedding, transformer encoder, and decoder head."""

        super().__init__()
        self.patch = patch
        self.mask_decoder = mask_decoder
        self.embed = nn.Conv2d(3, dim, patch, patch)
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.class_tokens = nn.Parameter(torch.randn(1, classes, dim) * 0.02)
        self.proj = nn.Linear(dim, classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Segment directly from image patches without a convolutional backbone."""

        patches = self.embed(image)
        bsz, dim, height, width = patches.shape
        tokens = self.encoder(patches.flatten(2).transpose(1, 2))
        if self.mask_decoder:
            cls = self.class_tokens.expand(bsz, -1, -1)
            masks = torch.matmul(cls, tokens.transpose(1, 2)).reshape(bsz, -1, height, width)
        else:
            masks = self.proj(tokens).transpose(1, 2).reshape(bsz, -1, height, width)
        return F.interpolate(masks, image.shape[-2:], mode="bilinear", align_corners=False)


class DynamicMultiScaleConv(nn.Module):
    """DMNet dynamic multi-scale convolution with input-conditioned scale weights."""

    def __init__(self, channels: int) -> None:
        """Initialize parallel dilated filters and a scale generator."""

        super().__init__()
        self.branches = nn.ModuleList(
            [ConvBNAct(channels, channels, 3, dilation=r) for r in (1, 2, 3)]
        )
        self.generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // 2), 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(max(4, channels // 2), 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dynamically weighted multi-scale filters."""

        weights = torch.softmax(self.generator(x), dim=1)
        branches = torch.stack([branch(x) for branch in self.branches], dim=1)
        return (branches * weights.unsqueeze(2)).sum(dim=1)


class ENetBottleneck(nn.Module):
    """ENet bottleneck with projection, spatial convolution, and pooling side path."""

    def __init__(self, channels: int) -> None:
        """Initialize compact ENet downsample bottleneck."""

        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels // 2, 3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(channels // 2, channels, 1),
        )
        self.out = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ENet bottleneck and restore the input resolution."""

        pooled = F.max_pool2d(x, 2, 2, ceil_mode=True)
        y = self.main(x) + F.interpolate(pooled, size=self.main(x).shape[-2:], mode="nearest")
        return F.interpolate(self.out(y), size=x.shape[-2:], mode="bilinear", align_corners=False)


class ESPBlock(nn.Module):
    """ESPNet efficient spatial pyramid of dilated convolutions."""

    def __init__(self, channels: int) -> None:
        """Initialize reduce-split-transform-merge ESP branches."""

        super().__init__()
        split = max(4, channels // 4)
        self.reduce = ConvBNAct(channels, split, 1)
        self.branches = nn.ModuleList(
            [nn.Conv2d(split, split, 3, padding=r, dilation=r) for r in (1, 2, 4, 8)]
        )
        self.merge = ConvBNAct(split * 4, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hierarchical sum fusion over parallel dilated branches."""

        base = self.reduce(x)
        outs = []
        running = torch.zeros_like(base)
        for branch in self.branches:
            running = running + branch(base)
            outs.append(running)
        return x + self.merge(torch.cat(outs, dim=1))


class GloReUnit(nn.Module):
    """GloRe coordinate-to-interaction graph reasoning unit."""

    def __init__(self, channels: int, nodes: int = 8) -> None:
        """Initialize projection, graph convolution, and broadcast layers."""

        super().__init__()
        self.assign = nn.Conv2d(channels, nodes, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.graph = nn.Linear(nodes, nodes, bias=False)
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to graph nodes, reason there, and broadcast to coordinates."""

        bsz, channels, height, width = x.shape
        assign = torch.softmax(self.assign(x).flatten(2), dim=-1)
        value = self.value(x).flatten(2)
        nodes = torch.bmm(value, assign.transpose(1, 2))
        nodes = self.graph(nodes)
        back = torch.bmm(nodes, assign).view(bsz, channels, height, width)
        return x + self.out(back)


class EfficientFormerV2Block(nn.Module):
    """EfficientFormerV2 4D MetaBlock with mobile local attention."""

    def __init__(self, channels: int) -> None:
        """Initialize 4D token mixer and lightweight attention."""

        super().__init__()
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1), nn.GELU(), nn.Conv2d(channels * 2, channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local 4D MetaBlock mixing."""

        local = x + self.pool(x) - x.mean(dim=(-2, -1), keepdim=True)
        q, k, v = self.qkv(local).chunk(3, dim=1)
        return local + self.mlp(v * torch.sigmoid((q * k).mean(dim=1, keepdim=True)))


class DynamicKernelUpdate(nn.Module):
    """K-Net dynamic kernel update and mask regeneration."""

    def __init__(self, channels: int, queries: int = 6) -> None:
        """Initialize learnable kernels and update cell."""

        super().__init__()
        self.kernels = nn.Parameter(torch.randn(1, queries, channels) * 0.02)
        self.update = nn.GRUCell(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Update kernels from assigned regions and produce refined masks."""

        bsz, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        kernels = self.kernels.expand(bsz, -1, -1)
        masks = torch.matmul(kernels, tokens.transpose(1, 2))
        regions = torch.bmm(torch.softmax(masks, dim=-1), tokens)
        updated = self.update(regions.flatten(0, 1), kernels.flatten(0, 1)).view_as(kernels)
        return torch.matmul(updated, tokens.transpose(1, 2)).view(bsz, -1, height, width)


class PaddleSegClassic(nn.Module):
    """One compact segmentation model selected by a distinctive primitive."""

    def __init__(self, kind: str, classes: int = 5, width: int = 16) -> None:
        """Initialize a compact faithful reconstruction.

        Parameters
        ----------
        kind:
            Architecture key.
        classes:
            Number of segmentation classes.
        width:
            Base channel width.
        """

        super().__init__()
        self.kind = kind
        self.backbone = TinyBackbone(width)
        self.low = ConvBNAct(width, width, 1)
        self.proj2 = ConvBNAct(width * 2, width, 1)
        self.proj3 = ConvBNAct(width * 3, width, 1)
        self.proj4 = ConvBNAct(width * 4, width, 1)
        self.aspp = ASPP(width * 4, width)
        self.nonlocal_block = NonLocalBlock(width * 4)
        self.ema = EMABlock(width * 4)
        self.encoding = EncodingBlock(width * 4)
        self.hard = HarDBlock(width * 4)
        self.dw = DepthwiseSeparable(width * 4, width * 4)
        self.edge = nn.Conv2d(width, 1, 1)
        self.aux = nn.Conv2d(width, classes, 1)
        self.query = nn.Parameter(torch.randn(1, 6, width))
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width * 4, 4, width * 8, batch_first=True), 1
        )
        self.cls = nn.Conv2d(width, classes, 1)
        self.fuse = ConvBNAct(width * 4, width, 1)
        self.gate = nn.Conv2d(width * 2, width, 1)
        self.psp = PSPModule(width * 4, width)
        self.segnet = SegNetUnpoolDecoder(width, classes)
        self.pp_mobile = PPStrideFormerHead(width, classes)
        self.rtformer = RTFormerFusion(width, classes)
        self.seaformer = SeaFormerBlock(width, classes)
        self.msca = SegNeXtMSCA(width * 4)
        self.segmenter = PatchTransformerSeg(4, width * 2, classes, mask_decoder=True)
        self.setr = PatchTransformerSeg(4, width * 2, classes, mask_decoder=False)
        self.flow = nn.Conv2d(width * 2, 2, 3, padding=1)
        self.stdc_layers = nn.ModuleList(
            [ConvBNAct(width if idx == 0 else width // 2, width // 2, 3) for idx in range(4)]
        )
        self.stdc_out = ConvBNAct(width * 3, width, 1)
        self.teacher_backbone = TinyBackbone(width)
        self.nest = ConvBNAct(width * 2, width, 3)
        self.dynamic_ms = DynamicMultiScaleConv(width * 4)
        self.enet_block = ENetBottleneck(width * 4)
        self.esp_block = ESPBlock(width * 4)
        self.glore_unit = GloReUnit(width * 4)
        self.ginet_unit = GloReUnit(width * 4)
        self.effformer = EfficientFormerV2Block(width * 4)
        self.kernel_update = DynamicKernelUpdate(width)
        self.shape_stream = ConvBNAct(1, width, 3)
        self.detail_stream = ConvBNAct(3, width, 3)
        self.context_stream = ConvBNAct(width * 4, width, 1)

    def _decode(self, feats: list[torch.Tensor], image: torch.Tensor) -> torch.Tensor:
        """Decode aligned feature maps to logits.

        Parameters
        ----------
        feats:
            Feature maps already aligned to the highest resolution.
        image:
            Original image tensor.

        Returns
        -------
        torch.Tensor
            Segmentation logits at input resolution.
        """

        x = self.fuse(torch.cat(feats, dim=1))
        return F.interpolate(
            self.cls(x), size=image.shape[-2:], mode="bilinear", align_corners=False
        )

    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a feature map with a tiny transformer.

        Parameters
        ----------
        x:
            Feature map to tokenize.

        Returns
        -------
        torch.Tensor
            Transformer-encoded feature map.
        """

        bsz, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        encoded = self.trans(tokens)
        return encoded.transpose(1, 2).reshape(bsz, channels, height, width)

    def forward(self, image: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Segment an image with the configured architecture primitive.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Logits, or logits plus auxiliary branch for multi-head models.
        """

        f1, f2, f3, f4 = self.backbone(image)
        target = f1.shape[-2:]
        p1 = self.low(f1)
        p2 = F.interpolate(self.proj2(f2), size=target, mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.proj3(f3), size=target, mode="bilinear", align_corners=False)
        p4_raw = f4

        if self.kind == "deeplabv3p":
            p4 = F.interpolate(self.aspp(p4_raw), size=target, mode="bilinear", align_corners=False)
            return self._decode([p1, p2, p3, p4], image)
        if self.kind == "decoupled_segnet":
            body = F.interpolate(
                self.aspp(p4_raw), size=target, mode="bilinear", align_corners=False
            )
            edge_flow = torch.tanh(self.flow(torch.cat([p1, body], dim=1)))
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, target[0], device=image.device),
                torch.linspace(-1, 1, target[1], device=image.device),
                indexing="ij",
            )
            grid = torch.stack([xx, yy], dim=-1).unsqueeze(0) + 0.05 * edge_flow.permute(0, 2, 3, 1)
            body = F.grid_sample(body, grid, align_corners=False)
            edge = torch.sigmoid(self.edge(p1))
            logits = self._decode([p1 * edge, p2, p3, body], image)
            return logits, F.interpolate(
                edge, image.shape[-2:], mode="bilinear", align_corners=False
            )
        if self.kind == "pspnet":
            p4 = F.interpolate(self.psp(p4_raw), size=target, mode="bilinear", align_corners=False)
            return self._decode([p1, p2, p3, p4], image)
        if self.kind == "dmnet":
            p4 = self.proj4(self.dynamic_ms(p4_raw))
            return self._decode([p1, p2, p3, F.interpolate(p4, target)], image)
        if self.kind == "dnlnet":
            p4 = self.proj4(self.nonlocal_block(p4_raw))
            return self._decode([p1, p2, p3, F.interpolate(p4, target)], image)
        if self.kind == "emanet":
            p4 = self.proj4(self.ema(p4_raw))
            return self._decode([p1, p2, p3, F.interpolate(p4, target)], image)
        if self.kind == "encnet":
            p4 = self.proj4(self.encoding(p4_raw))
            return self._decode([p1, p2, p3, F.interpolate(p4, target)], image)
        if self.kind == "pp_mobileseg":
            return self.pp_mobile(p2, p4_raw, image)
        if self.kind == "enet":
            p4 = self.proj4(self.enet_block(p4_raw))
            return self._decode([p1, p2, p3, F.interpolate(p4, target)], image)
        if self.kind == "espnet":
            p4 = self.proj4(self.esp_block(p4_raw))
            return self._decode([p1, p2, p3, F.interpolate(p4, target)], image)
        if self.kind == "fastfcn":
            high = F.interpolate(
                self.proj4(p4_raw), size=target, mode="bilinear", align_corners=False
            )
            joint = torch.cat([p2, p3, high], dim=1)
            jpu = self.proj3(joint)
            return self._decode([p1, p2, p3, jpu], image)
        if self.kind == "fastscnn":
            learn_to_downsample = F.avg_pool2d(p1, 2, 2)
            global_extractor = self.proj4(self.dw(p4_raw))
            fused = p1 + F.interpolate(
                global_extractor, target, mode="bilinear", align_corners=False
            )
            return self._decode(
                [
                    fused,
                    F.interpolate(learn_to_downsample, target),
                    p3,
                    F.interpolate(global_extractor, target),
                ],
                image,
            )
        if self.kind in {"mobileseg"}:
            p4 = self.proj4(self.dw(p4_raw))
            gate = torch.sigmoid(self.gate(torch.cat([p1, F.interpolate(p4, target)], dim=1)))
            return self._decode([p1 * gate, p2, p3, F.interpolate(p4, target)], image)
        if self.kind == "hardnet":
            p4 = self.proj4(self.hard(p4_raw))
            return self._decode([p1, p2, p3, F.interpolate(p4, target)], image)
        if self.kind == "segnet":
            return self.segnet(image)
        if self.kind == "fcn":
            p4 = F.interpolate(
                self.proj4(p4_raw), size=target, mode="bilinear", align_corners=False
            )
            return self._decode([p1, p2, p3, p4], image)
        if self.kind == "gcnet":
            context = torch.softmax(p4_raw.flatten(2).mean(1), dim=-1).unsqueeze(1)
            global_vec = torch.bmm(p4_raw.flatten(2), context.transpose(1, 2)).view(
                image.shape[0], -1, 1, 1
            )
            p4 = self.proj4(p4_raw + global_vec)
            return self._decode([p1, p2, p3, F.interpolate(p4, target)], image)
        if self.kind == "glore":
            p4 = self.proj4(self.glore_unit(p4_raw))
            return self._decode([p1, p2, p3, F.interpolate(p4, target)], image)
        if self.kind == "ginet":
            p4 = self.proj4(self.ginet_unit(p4_raw) + p4_raw.mean(dim=(-2, -1), keepdim=True))
            return self._decode([p1, p2, p3, F.interpolate(p4, target)], image)
        if self.kind == "rtformer":
            return self.rtformer(p1, p4_raw, image)
        if self.kind == "seaformer":
            return self.seaformer(p1, p4_raw, image)
        if self.kind == "topformer":
            low = self.proj4(self._tokens_to_map(p4_raw))
            semantic = F.interpolate(low, size=target, mode="bilinear", align_corners=False)
            injected = p1 + semantic * torch.sigmoid(p1)
            return self._decode([injected, p2, p3, semantic], image)
        if self.kind in {"hrformer", "uhrnet"}:
            high = p1
            low = self.proj4(self._tokens_to_map(p4_raw))
            attn = torch.sigmoid(
                F.interpolate(low, size=target, mode="bilinear", align_corners=False)
            )
            if self.kind == "uhrnet":
                high = high + F.interpolate(low, size=target, mode="bilinear", align_corners=False)
            return self._decode([high * attn, p2, p3, attn], image)
        if self.kind == "mscale_ocrnet":
            small = self.proj4(self._tokens_to_map(p4_raw))
            large = self.proj4(self._tokens_to_map(F.avg_pool2d(p4_raw, 2, ceil_mode=True)))
            large = F.interpolate(large, small.shape[-2:], mode="bilinear", align_corners=False)
            object_context = torch.softmax((small * large).flatten(2), dim=-1).mean(
                dim=-1, keepdim=True
            )
            p4 = F.interpolate(
                small + large * object_context.unsqueeze(-1), target, mode="bilinear"
            )
            return self._decode([p1, p2, p3, p4], image)
        if self.kind == "segmenter":
            return self.segmenter(image)
        if self.kind == "setr":
            return self.setr(image)
        if self.kind == "efficientformerv2":
            p4 = self.proj4(self.effformer(p4_raw))
            return self._decode([p1, p2, p3, F.interpolate(p4, target)], image)
        if self.kind == "isanet":
            long = self._tokens_to_map(p4_raw[:, :, ::2, ::2])
            interlaced = p4_raw + F.interpolate(long, p4_raw.shape[-2:], mode="nearest")
            p4 = self.proj4(self._tokens_to_map(interlaced))
            return self._decode([p1, p2, p3, F.interpolate(p4, target)], image)
        if self.kind == "knet":
            p4 = self.proj4(p4_raw)
            masks = self.kernel_update(p4)
            masks = F.interpolate(
                masks, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
            return masks[:, :5]
        if self.kind == "pointrend":
            coarse = self._decode([p1, p2, p3, F.interpolate(self.proj4(p4_raw), target)], image)
            uncertainty = -coarse.softmax(dim=1).topk(2, dim=1).values.diff(dim=1).abs().squeeze(1)
            return coarse + 0.05 * uncertainty.unsqueeze(1)
        if self.kind == "maskformer":
            p4 = self.proj4(p4_raw)
            tokens = p4.flatten(2).transpose(1, 2)
            queries = self.query.expand(image.shape[0], -1, -1)
            masks = torch.matmul(queries, tokens.transpose(1, 2)).view(
                image.shape[0], -1, *p4.shape[-2:]
            )
            masks = F.interpolate(
                masks, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
            return masks[:, :5]
        if self.kind == "lpsnet":
            stage2 = p1 + p2
            stage3 = stage2 + p3
            return self._decode([p1, stage2, stage3, p3], image)
        if self.kind == "sfnet":
            high = F.interpolate(
                self.proj4(p4_raw), size=target, mode="bilinear", align_corners=False
            )
            flow = torch.tanh(self.flow(torch.cat((p1, high), dim=1))).permute(0, 2, 3, 1)
            base_y, base_x = torch.meshgrid(
                torch.linspace(-1, 1, target[0], device=image.device),
                torch.linspace(-1, 1, target[1], device=image.device),
                indexing="ij",
            )
            grid = torch.stack((base_x, base_y), dim=-1).unsqueeze(0) + 0.1 * flow
            aligned = F.grid_sample(high, grid, align_corners=False)
            return self._decode([p1, p2, p3, aligned], image)
        if self.kind == "lraspp":
            low = p1
            high = F.interpolate(
                self.proj4(p4_raw), size=target, mode="bilinear", align_corners=False
            )
            scale = torch.sigmoid(F.adaptive_avg_pool2d(high, 1))
            return self._decode([low, p2, p3, high * scale], image)
        if self.kind == "segnext":
            p4 = F.interpolate(self.proj4(self.msca(p4_raw)), size=target, mode="bilinear")
            return self._decode([p1, p2, p3, p4], image)
        if self.kind == "stdcseg":
            feats = [p1]
            cur = p1
            for layer in self.stdc_layers:
                cur = layer(cur)
                feats.append(cur)
            stdc = self.stdc_out(torch.cat(feats, dim=1))
            return self._decode([stdc, p2, p3, F.interpolate(self.proj4(p4_raw), target)], image)
        if self.kind == "gscnn":
            shape = self.shape_stream(image.mean(dim=1, keepdim=True))
            gate = torch.sigmoid(self.gate(torch.cat([shape, p1], dim=1)))
            edge = torch.sigmoid(self.edge(shape * gate))
            body = self._decode([p1, p2, p3, F.interpolate(self.proj4(p4_raw), target)], image)
            return body * F.interpolate(
                edge, size=image.shape[-2:], mode="bilinear", align_corners=False
            )
        if self.kind == "pidnet":
            detail = self.detail_stream(image)
            context = F.interpolate(
                self.context_stream(p4_raw), target, mode="bilinear", align_corners=False
            )
            boundary = torch.sigmoid(self.edge(detail))
            return self._decode([detail * boundary, p2, p3, context], image)
        if self.kind == "modnet":
            semantic = F.interpolate(
                self.context_stream(p4_raw), target, mode="bilinear", align_corners=False
            )
            detail = self.detail_stream(image)
            return torch.sigmoid(self.cls(semantic * torch.sigmoid(detail)))[:, :1]
        if self.kind == "portraitnet":
            matte = torch.sigmoid(
                self._decode([p1, p2, p3, F.interpolate(self.proj4(p4_raw), target)], image)
            )
            boundary = torch.sigmoid(
                F.interpolate(self.edge(p1), image.shape[-2:], mode="bilinear")
            )
            return matte[:, :1] * boundary
        if self.kind == "humanseg":
            matte = torch.sigmoid(
                self._decode([p1, p2, p3, F.interpolate(self.proj4(p4_raw), target)], image)
            )
            return matte[:, :1] * F.avg_pool2d(matte[:, :1], 3, stride=1, padding=1)
        if self.kind == "pp_matting":
            matte = torch.sigmoid(
                self._decode([p1, p2, p3, F.interpolate(self.proj4(p4_raw), target)], image)
            )
            trimap = torch.sigmoid(image[:, :1])
            return matte[:, :1] * trimap + (1.0 - trimap) * torch.sigmoid(self.edge(p1))
        if self.kind in {"unet", "attention_unet", "unet3p", "unetpp"}:
            up4 = F.interpolate(self.proj4(p4_raw), target, mode="bilinear", align_corners=False)
            dec3 = F.interpolate(p3, target, mode="bilinear", align_corners=False)
            if self.kind == "attention_unet":
                skip_gate = torch.sigmoid(self.gate(torch.cat((p1, up4), dim=1)))
                return self._decode([p1 * skip_gate, p2, dec3, up4], image)
            if self.kind == "unet3p":
                full = torch.cat((p1, p2, dec3, up4), dim=1)
                return F.interpolate(self.cls(self.fuse(full)), image.shape[-2:], mode="bilinear")
            if self.kind == "unetpp":
                nested = self.nest(
                    torch.cat((p1, F.interpolate(p2, target, mode="bilinear")), dim=1)
                )
                return self._decode([nested, p2, dec3, up4], image)
            return self._decode([p1, p2, dec3, up4], image)
        if self.kind == "stfpm":
            with torch.no_grad():
                t1_raw, t2_raw, t3_raw, t4_raw = self.teacher_backbone(image)
                teacher_feats = [
                    self.low(t1_raw),
                    F.interpolate(
                        self.proj2(t2_raw), size=target, mode="bilinear", align_corners=False
                    ),
                    F.interpolate(
                        self.proj3(t3_raw), size=target, mode="bilinear", align_corners=False
                    ),
                    F.interpolate(
                        self.proj4(t4_raw), size=target, mode="bilinear", align_corners=False
                    ),
                ]
            student_feats = [
                p1,
                p2,
                p3,
                F.interpolate(
                    self.proj4(p4_raw), size=target, mode="bilinear", align_corners=False
                ),
            ]
            pyramid_errors = [
                (F.normalize(student, dim=1) - F.normalize(teacher.detach(), dim=1)).pow(2)
                for student, teacher in zip(student_feats, teacher_feats, strict=True)
            ]
            score = sum(err.mean(dim=1, keepdim=True) for err in pyramid_errors) / len(
                pyramid_errors
            )
            return F.interpolate(score, size=image.shape[-2:], mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.proj4(p4_raw), size=target, mode="bilinear", align_corners=False)
        return self._decode([p1, p2, p3, p4], image)


class FIGConvNet(nn.Module):
    """Factorized implicit global convolution network for point CFD fields."""

    def __init__(self, points: int = 16, width: int = 24) -> None:
        """Initialize latent axes and point projections.

        Parameters
        ----------
        points:
            Number of compact input points.
        width:
            Hidden width.
        """

        super().__init__()
        self.x_axis = nn.Parameter(torch.randn(points, width))
        self.y_axis = nn.Parameter(torch.randn(points, width))
        self.z_axis = nn.Parameter(torch.randn(points, width))
        self.in_proj = nn.Linear(6, width)
        self.out = nn.Linear(width, 4)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Predict fields with factorized global implicit convolution.

        Parameters
        ----------
        points:
            Point coordinates and attributes with shape ``(B, N, 6)``.

        Returns
        -------
        torch.Tensor
            Per-point pressure/velocity-like outputs.
        """

        feat = self.in_proj(points)
        ax = torch.softmax(torch.matmul(feat, self.x_axis.T), dim=-1)
        ay = torch.softmax(torch.matmul(feat, self.y_axis.T), dim=-1)
        az = torch.softmax(torch.matmul(feat, self.z_axis.T), dim=-1)
        gx = torch.matmul(ax, torch.matmul(ax.transpose(1, 2), feat))
        gy = torch.matmul(ay, torch.matmul(ay.transpose(1, 2), feat))
        gz = torch.matmul(az, torch.matmul(az.transpose(1, 2), feat))
        return self.out(F.gelu(feat + (gx + gy + gz) / 3.0))


class Pi3Geometry(nn.Module):
    """Permutation-equivariant reference-free multi-view geometry model."""

    def __init__(self, views: int = 3, width: int = 24) -> None:
        """Initialize shared image encoder and equivariant heads.

        Parameters
        ----------
        views:
            Number of compact views.
        width:
            Hidden width.
        """

        super().__init__()
        self.views = views
        self.encoder = nn.Sequential(ConvBNAct(3, width, 3, 2), ConvBNAct(width, width, 3, 2))
        self.pose = nn.Linear(width * 2, 7)
        self.point_head = nn.Conv2d(width * 2, 3, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict pose and point maps without a fixed reference view.

        Parameters
        ----------
        images:
            Multi-view images with shape ``(B, V, 3, H, W)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Per-view affine-invariant poses and local point maps.
        """

        bsz, views, channels, height, width = images.shape
        flat = images.reshape(bsz * views, channels, height, width)
        local = self.encoder(flat)
        pooled = F.adaptive_avg_pool2d(local, 1).flatten(1).reshape(bsz, views, -1)
        global_set = pooled.mean(dim=1, keepdim=True).expand(-1, views, -1)
        fused_vec = torch.cat([pooled, global_set], dim=-1)
        pose = self.pose(fused_vec)
        global_map = global_set.reshape(bsz * views, -1, 1, 1).expand_as(local)
        points = self.point_head(torch.cat([local, global_map], dim=1))
        return pose, points.reshape(bsz, views, 3, *points.shape[-2:])


def _build(kind: str) -> nn.Module:
    """Build a compact PaddleSeg classic for ``kind``.

    Parameters
    ----------
    kind:
        Architecture key.

    Returns
    -------
    nn.Module
        Random-initialized compact model.
    """

    torch.set_num_threads(1)
    return PaddleSegClassic(kind)


def example_input() -> torch.Tensor:
    """Create a compact RGB segmentation input.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 3, 32, 32)``.
    """

    torch.set_num_threads(1)
    return torch.randn(1, 3, 32, 32)


def example_points() -> torch.Tensor:
    """Create compact point features for FIGConvNet.

    Returns
    -------
    torch.Tensor
        Point tensor with shape ``(1, 16, 6)``.
    """

    torch.set_num_threads(1)
    return torch.randn(1, 16, 6)


def example_views() -> torch.Tensor:
    """Create compact multi-view images for Pi3.

    Returns
    -------
    torch.Tensor
        Image tensor with shape ``(1, 3, 3, 32, 32)``.
    """

    torch.set_num_threads(1)
    return torch.randn(1, 3, 3, 32, 32)


def build_figconvnet() -> nn.Module:
    """Build compact FIGConvNet.

    Returns
    -------
    nn.Module
        Random-initialized FIGConvNet.
    """

    torch.set_num_threads(1)
    return FIGConvNet()


def build_pi3() -> nn.Module:
    """Build compact Pi3 geometry model.

    Returns
    -------
    nn.Module
        Random-initialized Pi3Geometry.
    """

    torch.set_num_threads(1)
    return Pi3Geometry()


def _make_builder(kind: str) -> Callable[[], nn.Module]:
    """Create a named build function for a PaddleSeg kind.

    Parameters
    ----------
    kind:
        Architecture key.

    Returns
    -------
    Callable[[], nn.Module]
        Builder function.
    """

    def build() -> nn.Module:
        """Build compact random-initialized PaddleSeg model.

        Returns
        -------
        nn.Module
            Compact model.
        """

        return _build(kind)

    build.__name__ = f"build_{kind}"
    return build


_KIND_BY_NAME: dict[str, str] = {
    "ppseg_decoupled_segnet": "decoupled_segnet",
    "paddleseg_deeplabv3p": "deeplabv3p",
    "paddleseg_dmnet": "dmnet",
    "ppseg_dmnet": "dmnet",
    "paddleseg_dnlnet": "dnlnet",
    "ppseg_dnlnet": "dnlnet",
    "paddleseg_efficientformerv2": "efficientformerv2",
    "ppseg_efficientformerv2": "efficientformerv2",
    "paddleseg_emanet": "emanet",
    "ppseg_emanet": "emanet",
    "paddleseg_encnet": "encnet",
    "ppseg_encnet": "encnet",
    "paddleseg_enet": "enet",
    "ppseg_enet": "enet",
    "paddleseg_espnet": "espnet",
    "ppseg_espnet": "espnet",
    "ppseg_espnetv1": "espnet",
    "paddleseg_fastfcn": "fastfcn",
    "ppseg_fastfcn": "fastfcn",
    "paddleseg_fastscnn": "fastscnn",
    "ppseg_hardnet": "hardnet",
    "paddleseg_fcn": "fcn",
    "paddleseg_gcnet": "gcnet",
    "paddleseg_ginet": "ginet",
    "ppseg_ginet": "ginet",
    "paddleseg_glore": "glore",
    "ppseg_glore": "glore",
    "paddleseg_gscnn": "gscnn",
    "ppseg_gscnn": "gscnn",
    "paddleseg_hardnet": "hardnet",
    "paddleseg_hrformer": "hrformer",
    "paddleseg_uhrnet": "uhrnet",
    "ppseg_uhrnet": "uhrnet",
    "paddleseg_isanet": "isanet",
    "ppseg_isanet": "isanet",
    "paddleseg_knet": "knet",
    "ppseg_knet": "knet",
    "ppseg_lpsnet": "lpsnet",
    "paddleseg_lraspp": "lraspp",
    "ppseg_lraspp": "lraspp",
    "paddleseg_maskformer": "maskformer",
    "ppseg_maskformer": "maskformer",
    "paddleseg_mobileseg": "mobileseg",
    "ppseg_modnet": "modnet",
    "paddleseg_mscale_ocrnet": "mscale_ocrnet",
    "ppseg_mscale_ocrnet": "mscale_ocrnet",
    "paddleseg_pidnet": "pidnet",
    "paddleseg_pointrend": "pointrend",
    "paddleseg_portraitnet": "portraitnet",
    "ppseg_portraitnet": "portraitnet",
    "paddleseg_pp_humanseg_lite": "humanseg",
    "ppseg_pp_humanseg_lite": "humanseg",
    "ppseg_pp_matting": "pp_matting",
    "paddleseg_pp_mobileseg": "pp_mobileseg",
    "ppseg_pp_mobileseg": "pp_mobileseg",
    "paddleseg_pspnet": "pspnet",
    "paddleseg_rtformer": "rtformer",
    "ppseg_rtformer": "rtformer",
    "paddleseg_seaformer": "seaformer",
    "ppseg_seaformer": "seaformer",
    "paddleseg_segmenter": "segmenter",
    "ppseg_segmenter": "segmenter",
    "paddleseg_segnet": "segnet",
    "paddleseg_segnext": "segnext",
    "ppseg_segnext": "segnext",
    "paddleseg_setr": "setr",
    "ppseg_setr": "setr",
    "paddleseg_sfnet": "sfnet",
    "ppseg_sfnet": "sfnet",
    "paddleseg_stdcseg": "stdcseg",
    "ppseg_stdcseg": "stdcseg",
    "paddleseg_stfpm": "stfpm",
    "paddleseg_topformer": "topformer",
    "ppseg_topformer": "topformer",
    "paddleseg_attention_unet": "attention_unet",
    "paddleseg_unet": "unet",
    "paddleseg_unet_3plus": "unet3p",
    "paddleseg_unet_plusplus": "unetpp",
    "ppseg_attention_unet": "attention_unet",
    "ppseg_unet_3plus": "unet3p",
}

for _kind in sorted(set(_KIND_BY_NAME.values())):
    globals()[f"build_{_kind}"] = _make_builder(_kind)

MENAGERIE_ENTRIES = [
    (name, f"build_{kind}", "example_input", "2021", "E5") for name, kind in _KIND_BY_NAME.items()
] + [
    ("FIGConvNet", "build_figconvnet", "example_points", "2025", "CFD"),
    ("Pi3-PermutationEquivariantGeometry", "build_pi3", "example_views", "2025", "3D"),
]
