"""More dependency-gated menagerie targets as compact classics.

Architectures covered here:

* YOLOX variants: CSPDarknet-style backbone, PAN/FPN fusion, anchor-free decoupled
  classification/objectness/regression head.
* Mimi: streaming convolutional neural audio codec with semantic/acoustic residual
  vector-quantizer token streams.
* MedSAM: SAM-style image encoder, box prompt encoder, and cross-attention mask
  decoder adapted for medical image segmentation.
* NBFNet: neural Bellman-Ford path reasoning with indicator, message, and aggregate
  components for link prediction.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_act(
    in_channels: int, out_channels: int, kernel: int = 3, stride: int = 1
) -> nn.Sequential:
    """Create a Conv-BN-SiLU block.

    Parameters
    ----------
    in_channels:
        Input channel count.
    out_channels:
        Output channel count.
    kernel:
        Convolution kernel size.
    stride:
        Convolution stride.

    Returns
    -------
    nn.Sequential
        Convolutional block.
    """

    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel, stride=stride, padding=kernel // 2, bias=False
        ),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(),
    )


class _CSPBlock(nn.Module):
    """Compact cross-stage-partial block."""

    def __init__(self, channels: int) -> None:
        """Initialize split, residual, and merge branches.

        Parameters
        ----------
        channels:
            Input and output channel count.
        """

        super().__init__()
        half = channels // 2
        self.left = _conv_bn_act(half, half)
        self.right = _conv_bn_act(half, half, 1)
        self.merge = _conv_bn_act(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP split-transform-concat.

        Parameters
        ----------
        x:
            Image feature tensor.

        Returns
        -------
        torch.Tensor
            Updated feature tensor.
        """

        a, b = x.chunk(2, dim=1)
        a = a + self.left(a)
        b = self.right(b)
        return self.merge(torch.cat([a, b], dim=1))


class _YOLOXBackbone(nn.Module):
    """Tiny CSPDarknet backbone returning P3, P4, and P5."""

    def __init__(self, width: int) -> None:
        """Initialize downsampling stages.

        Parameters
        ----------
        width:
            Base channel width.
        """

        super().__init__()
        self.stem = _conv_bn_act(3, width, 3, 2)
        self.s2 = nn.Sequential(_conv_bn_act(width, width * 2, 3, 2), _CSPBlock(width * 2))
        self.s3 = nn.Sequential(_conv_bn_act(width * 2, width * 4, 3, 2), _CSPBlock(width * 4))
        self.s4 = nn.Sequential(_conv_bn_act(width * 4, width * 8, 3, 2), _CSPBlock(width * 8))
        self.s5 = nn.Sequential(_conv_bn_act(width * 8, width * 16, 3, 2), _CSPBlock(width * 16))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multiscale features.

        Parameters
        ----------
        x:
            Input image.

        Returns
        -------
        list[torch.Tensor]
            ``[P3, P4, P5]`` feature maps.
        """

        x = self.s2(self.stem(x))
        p3 = self.s3(x)
        p4 = self.s4(p3)
        p5 = self.s5(p4)
        return [p3, p4, p5]


class _YOLOXPAFPN(nn.Module):
    """YOLOX path-aggregation FPN."""

    def __init__(self, width: int) -> None:
        """Initialize lateral and fusion blocks.

        Parameters
        ----------
        width:
            Backbone base width.
        """

        super().__init__()
        c3, c4, c5 = width * 4, width * 8, width * 16
        self.lat5 = _conv_bn_act(c5, c4, 1)
        self.fuse4 = _conv_bn_act(c4 + c4, c4, 1)
        self.lat4 = _conv_bn_act(c4, c3, 1)
        self.fuse3 = _conv_bn_act(c3 + c3, c3, 1)
        self.down3 = _conv_bn_act(c3, c3, 3, 2)
        self.out4 = _conv_bn_act(c3 + c4, c4, 1)
        self.down4 = _conv_bn_act(c4, c4, 3, 2)
        self.out5 = _conv_bn_act(c4 + c4, c5, 1)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Fuse top-down and bottom-up feature paths.

        Parameters
        ----------
        feats:
            Backbone features.

        Returns
        -------
        list[torch.Tensor]
            Fused pyramid features.
        """

        p3, p4, p5 = feats
        u5 = F.interpolate(self.lat5(p5), size=p4.shape[-2:], mode="nearest")
        n4 = self.fuse4(torch.cat([u5, p4], dim=1))
        u4 = F.interpolate(self.lat4(n4), size=p3.shape[-2:], mode="nearest")
        n3 = self.fuse3(torch.cat([u4, p3], dim=1))
        o4 = self.out4(torch.cat([self.down3(n3), n4], dim=1))
        o5 = self.out5(torch.cat([self.down4(o4), self.lat5(p5)], dim=1))
        return [n3, o4, o5]


class _YOLOXHead(nn.Module):
    """Anchor-free decoupled YOLOX detection head."""

    def __init__(self, channels: list[int], classes: int = 5) -> None:
        """Initialize per-scale stem and decoupled branches.

        Parameters
        ----------
        channels:
            Channel count per pyramid feature.
        classes:
            Number of object classes.
        """

        super().__init__()
        self.stems = nn.ModuleList([_conv_bn_act(ch, 32, 1) for ch in channels])
        self.cls_convs = nn.ModuleList([_conv_bn_act(32, 32) for _ in channels])
        self.reg_convs = nn.ModuleList([_conv_bn_act(32, 32) for _ in channels])
        self.cls_preds = nn.ModuleList([nn.Conv2d(32, classes, 1) for _ in channels])
        self.obj_preds = nn.ModuleList([nn.Conv2d(32, 1, 1) for _ in channels])
        self.reg_preds = nn.ModuleList([nn.Conv2d(32, 4, 1) for _ in channels])

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Predict class, objectness, and box maps for each scale.

        Parameters
        ----------
        feats:
            FPN feature maps.

        Returns
        -------
        list[torch.Tensor]
            Per-scale prediction tensors.
        """

        outs = []
        for idx, feat in enumerate(feats):
            stem = self.stems[idx](feat)
            cls = self.cls_preds[idx](self.cls_convs[idx](stem))
            reg_feat = self.reg_convs[idx](stem)
            reg = self.reg_preds[idx](reg_feat)
            obj = self.obj_preds[idx](reg_feat)
            outs.append(torch.cat([reg, obj, cls], dim=1))
        return outs


class YOLOXCompact(nn.Module):
    """Compact YOLOX detector."""

    def __init__(self, width: int = 8, classes: int = 5) -> None:
        """Initialize backbone, PAFPN, and decoupled head.

        Parameters
        ----------
        width:
            Base channel width.
        classes:
            Detection class count.
        """

        super().__init__()
        self.backbone = _YOLOXBackbone(width)
        self.neck = _YOLOXPAFPN(width)
        self.head = _YOLOXHead([width * 4, width * 8, width * 16], classes)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run YOLOX detection.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        list[torch.Tensor]
            Per-scale prediction maps.
        """

        return self.head(self.neck(self.backbone(x)))


class _ResidualVectorQuantizer(nn.Module):
    """Straight-through residual vector quantizer."""

    def __init__(self, codebooks: int, vocab: int, dim: int) -> None:
        """Initialize learned codebooks.

        Parameters
        ----------
        codebooks:
            Number of residual codebooks.
        vocab:
            Entries per codebook.
        dim:
            Latent dimension.
        """

        super().__init__()
        self.codes = nn.Parameter(torch.randn(codebooks, vocab, dim) * 0.05)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize latent frames by residual coding.

        Parameters
        ----------
        z:
            Latent sequence ``(batch, time, dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Quantized latent and code indices.
        """

        residual = z
        quantized = torch.zeros_like(z)
        all_indices = []
        for codebook in self.codes:
            dist = (residual.unsqueeze(-2) - codebook.view(1, 1, *codebook.shape)).pow(2).sum(-1)
            idx = dist.argmin(dim=-1)
            chosen = F.embedding(idx, codebook)
            quantized = quantized + chosen
            residual = residual - chosen
            all_indices.append(idx)
        return z + (quantized - z).detach(), torch.stack(all_indices, dim=-1)


class MimiCodecCompact(nn.Module):
    """Compact Mimi-style streaming neural audio codec."""

    def __init__(self, channels: int = 32, codebooks: int = 4, vocab: int = 32) -> None:
        """Initialize encoder, semantic/acoustic RVQ, and decoder.

        Parameters
        ----------
        channels:
            Latent channel count.
        codebooks:
            Compact residual codebook count.
        vocab:
            Entries per codebook.
        """

        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, 7, padding=3),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 4, stride=2, padding=1),
        )
        self.semantic = _ResidualVectorQuantizer(1, vocab, channels)
        self.acoustic = _ResidualVectorQuantizer(codebooks - 1, vocab, channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(channels, 1, 7, padding=3),
        )

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode, split-quantize, and reconstruct audio.

        Parameters
        ----------
        audio:
            Mono waveform ``(batch, 1, samples)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Reconstruction and semantic/acoustic token ids.
        """

        z = self.encoder(audio).transpose(1, 2)
        semantic, sem_ids = self.semantic(z)
        acoustic, ac_ids = self.acoustic(z - semantic.detach())
        recon = self.decoder((semantic + acoustic).transpose(1, 2))
        return recon, torch.cat([sem_ids, ac_ids], dim=-1)


class MedSAMCompact(nn.Module):
    """SAM-style promptable medical image segmentation model."""

    def __init__(self, dim: int = 48, heads: int = 4) -> None:
        """Initialize image encoder, prompt encoder, and mask decoder.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Number of cross-attention heads.
        """

        super().__init__()
        self.image_encoder = nn.Sequential(
            _conv_bn_act(1, 24, 3, 2),
            _conv_bn_act(24, dim, 3, 2),
            _conv_bn_act(dim, dim, 3),
        )
        self.box_mlp = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(dim, 24, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(24, 1, 4, stride=2, padding=1),
        )

    def forward(self, image_and_box: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Decode a prompt-conditioned segmentation mask.

        Parameters
        ----------
        image_and_box:
            Tuple of image tensor and normalized box prompt.

        Returns
        -------
        torch.Tensor
            Mask logits.
        """

        image, box = image_and_box
        feats = self.image_encoder(image)
        batch, channels, height, width = feats.shape
        img_tokens = feats.flatten(2).transpose(1, 2)
        prompt = self.box_mlp(box).unsqueeze(1)
        query = self.mask_token.expand(batch, -1, -1) + prompt
        decoded, _ = self.cross(query, img_tokens, img_tokens)
        gated = feats + decoded.transpose(1, 2).view(batch, channels, 1, 1)
        return self.up(gated)


class NBFNetCompact(nn.Module):
    """Neural Bellman-Ford network for compact link prediction."""

    def __init__(self, nodes: int = 8, relations: int = 3, dim: int = 24, steps: int = 3) -> None:
        """Initialize indicator, message, aggregate, and score components.

        Parameters
        ----------
        nodes:
            Number of nodes in the compact graph.
        relations:
            Number of relation types.
        dim:
            Hidden path representation dimension.
        steps:
            Bellman-Ford relaxation steps.
        """

        super().__init__()
        self.nodes = nodes
        self.steps = steps
        self.indicator = nn.Embedding(relations, dim)
        self.rel = nn.Embedding(relations, dim)
        self.message = nn.Linear(dim * 2, dim)
        self.aggregate = nn.GRUCell(dim, dim)
        self.score = nn.Linear(dim, 1)
        src = torch.tensor([0, 0, 1, 2, 2, 3, 4, 5, 6, 1])
        dst = torch.tensor([1, 2, 3, 3, 4, 5, 6, 7, 7, 6])
        etype = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 1])
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        self.register_buffer("etype", etype)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """Score query source, relation, and target triples.

        Parameters
        ----------
        query:
            Integer tensor ``(batch, 3)`` containing source, relation, target.

        Returns
        -------
        torch.Tensor
            Link logits.
        """

        batch = query.shape[0]
        src_q, rel_q, dst_q = query[:, 0], query[:, 1], query[:, 2]
        h = torch.zeros(batch, self.nodes, self.indicator.embedding_dim, device=query.device)
        h[torch.arange(batch), src_q] = self.indicator(rel_q)
        rel_features = self.rel(self.etype)
        for _ in range(self.steps):
            src_state = h[:, self.src]
            rel_state = rel_features.unsqueeze(0).expand(batch, -1, -1)
            msg = torch.tanh(self.message(torch.cat([src_state, rel_state], dim=-1)))
            agg = torch.zeros_like(h).index_add(1, self.dst, msg)
            h = self.aggregate(agg.reshape(-1, agg.shape[-1]), h.reshape(-1, h.shape[-1])).view_as(
                h
            )
        return self.score(h[torch.arange(batch), dst_q]).squeeze(-1)


def build_yolox(width: int = 8) -> nn.Module:
    """Build a compact YOLOX variant.

    Parameters
    ----------
    width:
        Base channel width.

    Returns
    -------
    nn.Module
        Random-init YOLOX model.
    """

    return YOLOXCompact(width=width)


def example_yolox() -> torch.Tensor:
    """Create example image input.

    Returns
    -------
    torch.Tensor
        Image tensor of shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


def build_mimi_moshi() -> nn.Module:
    """Build a compact Mimi codec.

    Returns
    -------
    nn.Module
        Random-init Mimi codec.
    """

    return MimiCodecCompact()


def example_mimi_moshi() -> torch.Tensor:
    """Create example waveform input.

    Returns
    -------
    torch.Tensor
        Waveform tensor of shape ``(1, 1, 256)``.
    """

    return torch.randn(1, 1, 256)


def build_medsam() -> nn.Module:
    """Build compact MedSAM.

    Returns
    -------
    nn.Module
        Random-init promptable segmenter.
    """

    return MedSAMCompact()


def example_medsam() -> tuple[torch.Tensor, torch.Tensor]:
    """Create example medical image and box prompt.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Image and normalized box tensors.
    """

    return torch.randn(1, 1, 64, 64), torch.tensor([[0.2, 0.2, 0.8, 0.8]])


def build_nbfnet() -> nn.Module:
    """Build compact NBFNet.

    Returns
    -------
    nn.Module
        Random-init neural Bellman-Ford link predictor.
    """

    return NBFNetCompact()


def example_nbfnet() -> torch.Tensor:
    """Create example link-prediction queries.

    Returns
    -------
    torch.Tensor
        Query triples of shape ``(2, 3)``.
    """

    return torch.tensor([[0, 1, 7], [2, 0, 6]])


MENAGERIE_ENTRIES = [
    ("yolox_darknet53", "build_yolox", "example_yolox", "2021", "E5"),
    ("yolox_l", "build_yolox", "example_yolox", "2021", "E5"),
    ("yolox_m", "build_yolox", "example_yolox", "2021", "E5"),
    ("yolox_nano", "build_yolox", "example_yolox", "2021", "E5"),
    ("yolox_s", "build_yolox", "example_yolox", "2021", "E5"),
    ("yolox_tiny", "build_yolox", "example_yolox", "2021", "E5"),
    ("yolox_x", "build_yolox", "example_yolox", "2021", "E5"),
    ("Mimi_moshi", "build_mimi_moshi", "example_mimi_moshi", "2024", "E5"),
    ("MedSAM", "build_medsam", "example_medsam", "2024", "E5"),
    ("NBFNet", "build_nbfnet", "example_nbfnet", "2021", "E5"),
]
