"""Dependency-gated OpenMMLab/Paddle classics reconstructed in base PyTorch.

This module supplies exact-name registry entries for install-hostile models from
OpenMMLab, PaddleOCR, PaddleDetection, and adjacent long-tail packages.  The
compact networks keep the distinctive architecture primitives: MaskFormer and
Mask2Former query-mask segmentation, VPD diffusion-conditioned segmentation,
BinsFormer adaptive-bin depth, LayoutLM-style KIE, table-structure recognizers,
and exact aliases to existing faithful Paddle/OCR/detection classics.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from menagerie.classics import cx_miscreimpl_longtail as longtail
from menagerie.classics import depth_anything_dpt
from menagerie.classics import paddledet_extra
from menagerie.classics import paddledet_faster_rcnn
from menagerie.classics import paddledet_rtdetrv3
from menagerie.classics import pgnet_resnet50_paddleocr
from menagerie.classics import ppocr_visionlan
from menagerie.classics import ppocr_vitstr
from menagerie.classics import reimpl2_4_compact
from menagerie.classics import reimpl3_10_tabseq
from menagerie.classics import reimpl6_dependency_gated
from menagerie.classics import spikingformer_variants
from menagerie.classics import svtrv2_base


class TinyBackboneFPN(nn.Module):
    """Small CNN backbone with top-down FPN feature fusion."""

    def __init__(self, width: int = 32) -> None:
        """Initialize backbone and lateral FPN layers.

        Parameters
        ----------
        width:
            Base channel width.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), nn.BatchNorm2d(width), nn.GELU()
        )
        self.c3 = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU())
        self.c4 = nn.Sequential(nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.GELU())
        self.c5 = nn.Sequential(nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1), nn.GELU())
        self.lat3 = nn.Conv2d(width, width, 1)
        self.lat4 = nn.Conv2d(width * 2, width, 1)
        self.lat5 = nn.Conv2d(width * 2, width, 1)
        self.out = nn.Conv2d(width * 3, width, 3, padding=1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode an RGB image into FPN features.

        Parameters
        ----------
        image:
            Input image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            Fine FPN feature plus three pyramid levels.
        """

        c3 = self.c3(self.stem(image))
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        p5 = self.lat5(c5)
        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        fused = self.out(
            torch.cat(
                [
                    p3,
                    F.interpolate(p4, size=p3.shape[-2:], mode="nearest"),
                    F.interpolate(p5, size=p3.shape[-2:], mode="nearest"),
                ],
                dim=1,
            )
        )
        return fused, p3, p4, p5


class MaskFormerCompact(nn.Module):
    """MaskFormer set-prediction segmentor with pixel decoder and mask queries."""

    def __init__(self, queries: int = 12, classes: int = 6, width: int = 32) -> None:
        """Initialize compact MaskFormer.

        Parameters
        ----------
        queries:
            Number of learned mask queries.
        classes:
            Number of semantic classes.
        width:
            Feature width.
        """

        super().__init__()
        self.backbone = TinyBackboneFPN(width)
        self.query = nn.Embedding(queries, width)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(width, 4, width * 2, batch_first=True, activation="gelu"),
            num_layers=2,
        )
        self.mask_embed = nn.Linear(width, width)
        self.class_embed = nn.Linear(width, classes + 1)
        self.pixel_proj = nn.Conv2d(width, width, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Predict class logits and query-conditioned masks.

        Parameters
        ----------
        image:
            RGB input image.

        Returns
        -------
        tuple[Tensor, Tensor]
            Query class logits and mask logits.
        """

        pixels, _, _, _ = self.backbone(image)
        memory = pixels.flatten(2).transpose(1, 2)
        query = self.query.weight.unsqueeze(0).expand(image.shape[0], -1, -1)
        decoded = self.decoder(query, memory)
        kernels = self.mask_embed(decoded)
        masks = torch.einsum("bqc,bchw->bqhw", kernels, self.pixel_proj(pixels))
        return self.class_embed(decoded), masks


class Mask2FormerCompact(nn.Module):
    """Mask2Former segmentor with masked-attention query refinement."""

    def __init__(self, queries: int = 12, classes: int = 6, width: int = 32) -> None:
        """Initialize compact Mask2Former.

        Parameters
        ----------
        queries:
            Number of object/mask queries.
        classes:
            Number of semantic classes.
        width:
            Feature width.
        """

        super().__init__()
        self.backbone = TinyBackboneFPN(width)
        self.query = nn.Embedding(queries, width)
        self.cross = nn.MultiheadAttention(width, 4, batch_first=True)
        self.self_attn = nn.MultiheadAttention(width, 4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )
        self.mask_embed = nn.Linear(width, width)
        self.class_embed = nn.Linear(width, classes + 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run masked-attention decoding over multi-scale pixel features.

        Parameters
        ----------
        image:
            RGB input image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class logits, mask logits, and final attention mask.
        """

        pixels, p3, p4, p5 = self.backbone(image)
        query = self.query.weight.unsqueeze(0).expand(image.shape[0], -1, -1)
        attention_mask = None
        for level in (p5, p4, p3):
            memory = level.flatten(2).transpose(1, 2)
            query = query + self.cross(query, memory, memory, need_weights=False)[0]
            query = query + self.self_attn(query, query, query, need_weights=False)[0]
            query = query + self.ffn(query)
            masks = torch.einsum("bqc,bchw->bqhw", self.mask_embed(query), pixels)
            pooled = F.adaptive_avg_pool2d(masks.sigmoid(), level.shape[-2:]).flatten(2)
            attention_mask = pooled / pooled.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return self.class_embed(query), masks, attention_mask


class VPDCompact(nn.Module):
    """VPD-style frozen diffusion UNet feature extractor plus segmentation head."""

    def __init__(self, width: int = 32, classes: int = 8) -> None:
        """Initialize compact VPD segmentor.

        Parameters
        ----------
        width:
            Feature width.
        classes:
            Number of segmentation classes.
        """

        super().__init__()
        self.t_embed = nn.Embedding(8, width)
        self.down = nn.Conv2d(3, width, 3, stride=2, padding=1)
        self.mid = nn.TransformerEncoderLayer(
            width, 4, width * 2, batch_first=True, activation="gelu"
        )
        self.up = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)
        self.adapter = nn.Conv2d(width, width, 1)
        self.head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, classes, 1)
        )

    def forward(self, image: Tensor) -> Tensor:
        """Segment an image using diffusion-timestep-conditioned features.

        Parameters
        ----------
        image:
            RGB image tensor.

        Returns
        -------
        Tensor
            Segmentation logits.
        """

        timestep = torch.full((image.shape[0],), 3, dtype=torch.long, device=image.device)
        feat = self.down(image) + self.t_embed(timestep).view(image.shape[0], -1, 1, 1)
        height, width = feat.shape[-2:]
        tokens = self.mid(feat.flatten(2).transpose(1, 2))
        feat = tokens.transpose(1, 2).reshape(image.shape[0], -1, height, width)
        return self.head(self.adapter(self.up(feat)))


class BinsFormerCompact(nn.Module):
    """BinsFormer monocular depth with transformer bin tokens and pixel-bin attention."""

    def __init__(self, bins: int = 16, width: int = 48) -> None:
        """Initialize compact BinsFormer.

        Parameters
        ----------
        bins:
            Number of adaptive depth bins.
        width:
            Feature width.
        """

        super().__init__()
        self.backbone = TinyBackboneFPN(width)
        self.bin_tokens = nn.Embedding(bins, width)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(width, 4, width * 2, batch_first=True, activation="gelu"),
            num_layers=2,
        )
        self.bin_width = nn.Linear(width, 1)
        self.assign = nn.Conv2d(width, bins, 1)

    def forward(self, image: Tensor) -> Tensor:
        """Predict dense depth from adaptive transformer bins.

        Parameters
        ----------
        image:
            RGB input image.

        Returns
        -------
        Tensor
            Dense depth map.
        """

        pixels, _, _, _ = self.backbone(image)
        memory = pixels.flatten(2).transpose(1, 2)
        tokens = self.bin_tokens.weight.unsqueeze(0).expand(image.shape[0], -1, -1)
        bins = self.decoder(tokens, memory)
        widths = F.softplus(self.bin_width(bins)).squeeze(-1)
        centers = torch.cumsum(widths, dim=-1)
        centers = centers / centers[:, -1:].clamp_min(1e-6)
        prob = torch.softmax(self.assign(pixels), dim=1)
        depth = (prob * centers[:, :, None, None]).sum(dim=1, keepdim=True)
        return F.interpolate(depth, size=image.shape[-2:], mode="bilinear", align_corners=False)


class LayoutLMKIECompact(nn.Module):
    """LayoutLM/LayoutXLM-style KIE with text, box, and image-region features."""

    def __init__(self, vocab: int = 256, dim: int = 64, labels: int = 12) -> None:
        """Initialize multimodal document token classifier.

        Parameters
        ----------
        vocab:
            Token vocabulary size.
        dim:
            Transformer width.
        labels:
            Entity label count.
        """

        super().__init__()
        self.word = nn.Embedding(vocab, dim)
        self.box = nn.Linear(4, dim)
        self.image = nn.Sequential(nn.Conv2d(3, dim, 4, stride=4), nn.GELU())
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, activation="gelu"),
            num_layers=2,
        )
        self.head = nn.Linear(dim, labels)

    def forward(self, payload: Tensor) -> Tensor:
        """Classify document tokens from packed ids, boxes, and page image.

        Parameters
        ----------
        payload:
            Packed tensor ``(B, 3 + T, 64)``.  The first three rows are an RGB
            page image; remaining rows contain token id and normalized box data.

        Returns
        -------
        Tensor
            Per-token KIE label logits.
        """

        image = payload[:, :3].view(payload.shape[0], 3, 8, 8)
        token_rows = payload[:, 3:]
        ids = token_rows[..., 0].abs().mul(255).long().clamp(0, 255)
        boxes = token_rows[..., 1:5].sigmoid()
        visual = self.image(image).flatten(2).transpose(1, 2)
        text = self.word(ids) + self.box(boxes)
        encoded = self.encoder(torch.cat([visual, text], dim=1))
        return self.head(encoded[:, visual.shape[1] :])


class TableMasterCompact(nn.Module):
    """TableMASTER-style table structure recognizer with MASTER decoder."""

    def __init__(self, vocab: int = 48, steps: int = 16, dim: int = 64) -> None:
        """Initialize table-structure encoder-decoder.

        Parameters
        ----------
        vocab:
            Structure token count.
        steps:
            Output decoding steps.
        dim:
            Transformer width.
        """

        super().__init__()
        self.patch = nn.Conv2d(1, dim, 4, stride=4)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, activation="gelu"),
            num_layers=2,
        )
        self.query = nn.Embedding(steps, dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, 4, dim * 2, batch_first=True, activation="gelu"),
            num_layers=2,
        )
        self.structure = nn.Linear(dim, vocab)
        self.cell_box = nn.Linear(dim, 4)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Decode table structure tokens and cell boxes.

        Parameters
        ----------
        image:
            Grayscale table image.

        Returns
        -------
        tuple[Tensor, Tensor]
            Structure logits and normalized cell boxes.
        """

        memory = self.encoder(self.patch(image).flatten(2).transpose(1, 2))
        query = self.query.weight.unsqueeze(0).expand(image.shape[0], -1, -1)
        decoded = self.decoder(query, memory)
        return self.structure(decoded), torch.sigmoid(self.cell_box(decoded))


class SLANetCompact(nn.Module):
    """SLANet table recognizer with SLA feature aggregation and dual heads."""

    def __init__(self, dim: int = 48, tokens: int = 40) -> None:
        """Initialize compact SLANet.

        Parameters
        ----------
        dim:
            Feature width.
        tokens:
            Number of table tokens.
        """

        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, dim, 3, stride=2, padding=1), nn.GELU())
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=3)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fuse = nn.Conv2d(dim * 2, dim, 1)
        self.seq = nn.GRU(dim, dim, batch_first=True, bidirectional=True)
        self.token = nn.Linear(dim * 2, tokens)
        self.box = nn.Linear(dim * 2, 4)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Predict SLANet table tokens and cell locations.

        Parameters
        ----------
        image:
            Grayscale table image.

        Returns
        -------
        tuple[Tensor, Tensor]
            Token logits and cell boxes.
        """

        feat = self.stem(image)
        glob = self.global_pool(feat).expand_as(feat)
        feat = self.fuse(torch.cat([self.local(feat), glob], dim=1))
        seq = feat.mean(dim=2).transpose(1, 2)
        encoded = self.seq(seq)[0]
        return self.token(encoded), torch.sigmoid(self.box(encoded))


class SimpleMLP(nn.Module):
    """Avalanche SimpleMLP-style feed-forward classifier."""

    def __init__(self, classes: int = 10) -> None:
        """Initialize MLP layers.

        Parameters
        ----------
        classes:
            Number of output classes.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Classify tabular inputs.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        Tensor
            Class logits.
        """

        return self.net(x)


class EWCMLP(SimpleMLP):
    """Avalanche EWC MLP with a Fisher-weighted penalty readout."""

    def __init__(self) -> None:
        """Initialize EWC model and Fisher diagonal buffer."""

        super().__init__(classes=10)
        self.register_buffer("fisher_diag", torch.ones(1, 10) * 0.05)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return logits and an EWC-style quadratic penalty.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        tuple[Tensor, Tensor]
            Class logits and penalty scalar per batch.
        """

        logits = super().forward(x)
        penalty = (self.fisher_diag * logits.square()).mean(dim=-1)
        return logits, penalty


class ModernNCAClassifier(nn.Module):
    """ModernNCA nearest-class-attention classifier."""

    def __init__(self, features: int = 20, classes: int = 4, prototypes: int = 3) -> None:
        """Initialize learned class prototypes.

        Parameters
        ----------
        features:
            Input feature count.
        classes:
            Class count.
        prototypes:
            Prototypes per class.
        """

        super().__init__()
        self.embed = nn.Sequential(nn.Linear(features, 48), nn.GELU(), nn.Linear(48, 24))
        self.prototypes = nn.Parameter(torch.randn(classes, prototypes, 24) * 0.1)
        self.temperature = nn.Parameter(torch.ones(()))

    def forward(self, x: Tensor) -> Tensor:
        """Classify features by prototype distances.

        Parameters
        ----------
        x:
            Tabular features.

        Returns
        -------
        Tensor
            Class logits.
        """

        emb = self.embed(x)
        dist = (emb[:, None, None, :] - self.prototypes[None]).square().sum(dim=-1)
        return -dist.amin(dim=-1) * self.temperature.exp()


class TabRClassifier(nn.Module):
    """TabR retrieval-augmented tabular classifier."""

    def __init__(self, features: int = 20, memory: int = 8, classes: int = 4) -> None:
        """Initialize learned retrieval memory.

        Parameters
        ----------
        features:
            Input feature count.
        memory:
            Number of memory rows.
        classes:
            Number of classes.
        """

        super().__init__()
        self.key = nn.Linear(features, 32)
        self.memory_x = nn.Parameter(torch.randn(memory, features) * 0.1)
        self.memory_y = nn.Parameter(torch.randn(memory, classes) * 0.1)
        self.head = nn.Linear(features + classes, classes)

    def forward(self, x: Tensor) -> Tensor:
        """Classify with soft nearest-neighbor memory retrieval.

        Parameters
        ----------
        x:
            Tabular features.

        Returns
        -------
        Tensor
            Class logits.
        """

        q = self.key(x)
        k = self.key(self.memory_x)
        weights = torch.softmax(torch.matmul(q, k.T) / q.shape[-1] ** 0.5, dim=-1)
        retrieved = torch.matmul(weights, self.memory_y)
        return self.head(torch.cat([x, retrieved], dim=-1))


class SDMGRPackedWrapper(nn.Module):
    """Single-input wrapper for SDMGR node-edge KIE."""

    def __init__(self) -> None:
        """Initialize wrapped SDMGR model."""

        super().__init__()
        self.core = reimpl2_4_compact.build_sdmgr_nodeedge()

    def forward(self, payload: Tensor) -> tuple[Tensor, Tensor]:
        """Run SDMGR from a packed document tensor.

        Parameters
        ----------
        payload:
            Packed token rows whose first channels encode features and boxes.

        Returns
        -------
        tuple[Tensor, Tensor]
            Node and edge logits.
        """

        rows = payload[:, 3:]
        feats = rows[..., :16]
        boxes = rows[..., :4].sigmoid()
        return self.core((feats, boxes))


class ALIGNNPackedWrapper(nn.Module):
    """Single-input wrapper for ALIGNN atom and line-graph message passing."""

    def __init__(self) -> None:
        """Initialize wrapped ALIGNN and fixed graph incidence."""

        super().__init__()
        self.core = longtail.build_alignn()
        self.register_buffer("src", torch.tensor([0, 1, 2, 3, 4, 0, 2, 1], dtype=torch.long))
        self.register_buffer("dst", torch.tensor([1, 2, 3, 4, 0, 2, 4, 3], dtype=torch.long))
        line_adj = torch.eye(8).roll(1, dims=0) + torch.eye(8).roll(-1, dims=0)
        self.register_buffer("line_adj", line_adj / line_adj.sum(dim=-1, keepdim=True))

    def forward(self, packed: Tensor) -> Tensor:
        """Run ALIGNN from packed atom and edge features.

        Parameters
        ----------
        packed:
            Tensor with five atom rows followed by eight edge rows.

        Returns
        -------
        Tensor
            Graph-level prediction.
        """

        atom = packed[:5, :8]
        edge = packed[5:13, :4]
        return self.core(atom, edge, self.src, self.dst, self.line_adj)


class OccupancyPackedWrapper(nn.Module):
    """Single-input wrapper for occupancy-network decoder."""

    def __init__(self) -> None:
        """Initialize wrapped occupancy decoder."""

        super().__init__()
        self.core = longtail.build_occupancy_network_decoder()

    def forward(self, packed: Tensor) -> Tensor:
        """Decode occupancy from packed query points and latent code.

        Parameters
        ----------
        packed:
            Tensor containing query coordinates and a repeated latent code.

        Returns
        -------
        Tensor
            Occupancy logits.
        """

        points = packed[..., :3]
        code = packed[:, 0, 3:19]
        return self.core(points, code)


class AvalancheMTPackedWrapper(nn.Module):
    """Single-input wrapper for Avalanche multi-task MLP."""

    def __init__(self) -> None:
        """Initialize wrapped multi-task MLP."""

        super().__init__()
        self.core = longtail.build_avalanche_mt_mlp()

    def forward(self, packed: Tensor) -> Tensor:
        """Classify packed features with task-conditioned heads.

        Parameters
        ----------
        packed:
            Feature rows with a final task-id channel.

        Returns
        -------
        Tensor
            Class logits.
        """

        features = packed[:, :20]
        task_id = packed[:, 20].abs().long() % 3
        return self.core(features, task_id)


class BrainLMCompact(nn.Module):
    """BrainLM-style masked fMRI patch transformer."""

    def __init__(self, regions: int = 16, time: int = 24, dim: int = 64) -> None:
        """Initialize fMRI token encoder and reconstruction head.

        Parameters
        ----------
        regions:
            Number of brain regions.
        time:
            Number of time samples.
        dim:
            Transformer width.
        """

        super().__init__()
        self.regions = regions
        self.time = time
        self.value = nn.Linear(1, dim)
        self.region = nn.Embedding(regions, dim)
        self.time_pos = nn.Embedding(time, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, activation="gelu"),
            num_layers=2,
        )
        self.reconstruct = nn.Linear(dim, 1)

    def forward(self, fmri: Tensor) -> Tensor:
        """Reconstruct masked fMRI region-time tokens.

        Parameters
        ----------
        fmri:
            fMRI tensor ``(B, regions, time)``.

        Returns
        -------
        Tensor
            Reconstructed fMRI signal.
        """

        batch, regions, time = fmri.shape
        values = self.value(fmri.reshape(batch, regions * time, 1))
        region_ids = torch.arange(regions, device=fmri.device).repeat_interleave(time)
        time_ids = torch.arange(time, device=fmri.device).repeat(regions)
        tokens = (
            values + self.region(region_ids).unsqueeze(0) + self.time_pos(time_ids).unsqueeze(0)
        )
        mask = (region_ids + time_ids).remainder(4).eq(0).view(1, -1, 1)
        tokens = torch.where(mask, self.mask_token.expand_as(tokens), tokens)
        encoded = self.encoder(tokens)
        return self.reconstruct(encoded).view(batch, regions, time)


class BehaviorTransformerCompact(nn.Module):
    """Behavior Transformer with discrete action bins and residual offsets."""

    def __init__(self, obs_dim: int = 12, actions: int = 8, bins: int = 16, dim: int = 64) -> None:
        """Initialize BeT observation transformer and action heads.

        Parameters
        ----------
        obs_dim:
            Observation feature count.
        actions:
            Action dimensions.
        bins:
            Number of action codebook bins.
        dim:
            Transformer width.
        """

        super().__init__()
        self.obs = nn.Linear(obs_dim, dim)
        self.codebook = nn.Parameter(torch.randn(bins, actions) * 0.1)
        self.pos = nn.Parameter(torch.randn(1, 12, dim) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, activation="gelu"),
            num_layers=2,
        )
        self.bin_head = nn.Linear(dim, bins)
        self.offset_head = nn.Linear(dim, actions)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Predict behavior-token bins and continuous residual actions.

        Parameters
        ----------
        obs:
            Observation sequence.

        Returns
        -------
        tuple[Tensor, Tensor]
            Bin logits and continuous actions.
        """

        hidden = self.encoder(self.obs(obs) + self.pos[:, : obs.shape[1]])
        logits = self.bin_head(hidden)
        weights = torch.softmax(logits, dim=-1)
        quantized = torch.matmul(weights, self.codebook)
        return logits, quantized + self.offset_head(hidden)


class ESMFoldCompact(nn.Module):
    """ESMFold-style protein language-model trunk plus folding head."""

    def __init__(self, vocab: int = 24, dim: int = 64) -> None:
        """Initialize compact ESMFold.

        Parameters
        ----------
        vocab:
            Amino-acid token vocabulary size.
        dim:
            Transformer width.
        """

        super().__init__()
        self.aa = nn.Embedding(vocab, dim)
        self.esm = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, activation="gelu"),
            num_layers=3,
        )
        self.pair = nn.Linear(dim * 2, dim)
        self.trunk = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, activation="gelu"),
            num_layers=1,
        )
        self.coord = nn.Linear(dim, 3)
        self.dist = nn.Linear(dim, 1)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Fold a protein sequence into coordinates and distance logits.

        Parameters
        ----------
        tokens:
            Amino-acid token ids.

        Returns
        -------
        tuple[Tensor, Tensor]
            Residue coordinates and pairwise distance logits.
        """

        single = self.esm(self.aa(tokens))
        left = single.unsqueeze(2).expand(-1, -1, single.shape[1], -1)
        right = single.unsqueeze(1).expand(-1, single.shape[1], -1, -1)
        pair = self.pair(torch.cat([left, right], dim=-1))
        pair_context = self.trunk(pair.mean(dim=2))
        coords = torch.cumsum(self.coord(single + pair_context), dim=1)
        dist_logits = self.dist(pair).squeeze(-1)
        return coords, dist_logits


def image_input() -> Tensor:
    """Return a compact RGB image.

    Returns
    -------
    Tensor
        Image tensor ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


def text_image() -> Tensor:
    """Return a compact grayscale text/table image.

    Returns
    -------
    Tensor
        Image tensor ``(1, 1, 32, 96)``.
    """

    return torch.randn(1, 1, 32, 96)


def tabular_input() -> Tensor:
    """Return compact tabular features.

    Returns
    -------
    Tensor
        Feature tensor.
    """

    return torch.randn(2, 20)


def layout_input() -> Tensor:
    """Return packed LayoutLM-style document payload.

    Returns
    -------
    Tensor
        Packed image/token tensor.
    """

    return torch.randn(1, 15, 64)


def fmri_input() -> Tensor:
    """Return compact fMRI region-time data.

    Returns
    -------
    Tensor
        fMRI tensor.
    """

    return torch.randn(1, 16, 24)


def behavior_input() -> Tensor:
    """Return compact behavior observation sequence.

    Returns
    -------
    Tensor
        Observation tensor.
    """

    return torch.randn(1, 12, 12)


def protein_tokens() -> Tensor:
    """Return compact amino-acid token sequence.

    Returns
    -------
    Tensor
        Protein token tensor.
    """

    return torch.randint(0, 24, (1, 16))


def build_maskformer() -> nn.Module:
    """Build compact MaskFormer segmentor.

    Returns
    -------
    nn.Module
        MaskFormer-style model.
    """

    return MaskFormerCompact().eval()


def build_mask2former() -> nn.Module:
    """Build compact Mask2Former segmentor.

    Returns
    -------
    nn.Module
        Mask2Former-style model.
    """

    return Mask2FormerCompact().eval()


def build_vpd() -> nn.Module:
    """Build compact VPD segmentor.

    Returns
    -------
    nn.Module
        VPD-style diffusion feature segmentor.
    """

    return VPDCompact().eval()


def build_binsformer() -> nn.Module:
    """Build compact BinsFormer.

    Returns
    -------
    nn.Module
        BinsFormer-style depth model.
    """

    return BinsFormerCompact().eval()


def build_layoutlm_kie() -> nn.Module:
    """Build compact LayoutLM KIE model.

    Returns
    -------
    nn.Module
        LayoutLM-style model.
    """

    return LayoutLMKIECompact().eval()


def build_table_master() -> nn.Module:
    """Build compact TableMASTER.

    Returns
    -------
    nn.Module
        TableMASTER-style recognizer.
    """

    return TableMasterCompact().eval()


def build_slanet() -> nn.Module:
    """Build compact SLANet.

    Returns
    -------
    nn.Module
        SLANet-style recognizer.
    """

    return SLANetCompact().eval()


def build_avalanche_simple_mlp() -> nn.Module:
    """Build Avalanche SimpleMLP.

    Returns
    -------
    nn.Module
        SimpleMLP classifier.
    """

    return SimpleMLP().eval()


def build_avalanche_ewc_mlp() -> nn.Module:
    """Build Avalanche EWC MLP.

    Returns
    -------
    nn.Module
        EWC-style MLP.
    """

    return EWCMLP().eval()


def build_modern_nca() -> nn.Module:
    """Build ModernNCA classifier.

    Returns
    -------
    nn.Module
        Prototype-distance classifier.
    """

    return ModernNCAClassifier().eval()


def build_tabr() -> nn.Module:
    """Build TabR classifier.

    Returns
    -------
    nn.Module
        Retrieval-augmented tabular classifier.
    """

    return TabRClassifier().eval()


def build_sdmgr_packed() -> nn.Module:
    """Build packed-input SDMGR KIE wrapper.

    Returns
    -------
    nn.Module
        SDMGR wrapper.
    """

    return SDMGRPackedWrapper().eval()


def build_alignn_packed() -> nn.Module:
    """Build packed-input ALIGNN wrapper.

    Returns
    -------
    nn.Module
        ALIGNN wrapper.
    """

    return ALIGNNPackedWrapper().eval()


def build_occupancy_packed() -> nn.Module:
    """Build packed-input occupancy decoder wrapper.

    Returns
    -------
    nn.Module
        Occupancy wrapper.
    """

    return OccupancyPackedWrapper().eval()


def build_avalanche_mt_packed() -> nn.Module:
    """Build packed-input Avalanche multi-task MLP wrapper.

    Returns
    -------
    nn.Module
        Multi-task MLP wrapper.
    """

    return AvalancheMTPackedWrapper().eval()


def build_brainlm() -> nn.Module:
    """Build compact BrainLM fMRI foundation model.

    Returns
    -------
    nn.Module
        BrainLM-style model.
    """

    return BrainLMCompact().eval()


def build_behavior_transformer_bet() -> nn.Module:
    """Build compact Behavior Transformer.

    Returns
    -------
    nn.Module
        BeT-style policy model.
    """

    return BehaviorTransformerCompact().eval()


def build_esmfold2() -> nn.Module:
    """Build compact ESMFold2-style model.

    Returns
    -------
    nn.Module
        ESMFold-style protein folder.
    """

    return ESMFoldCompact().eval()


def build_movinet_a1_pytorch() -> nn.Module:
    """Build compact MoViNet-A1 alias.

    Returns
    -------
    nn.Module
        MoViNet model.
    """

    return longtail.build_movinet_a0_pytorch().eval()


def build_movinet_a2_pytorch() -> nn.Module:
    """Build compact MoViNet-A2 alias.

    Returns
    -------
    nn.Module
        MoViNet model.
    """

    return longtail.build_movinet_a0_pytorch().eval()


def build_slanext() -> nn.Module:
    """Build compact SLANeXt with the SLANet primitive.

    Returns
    -------
    nn.Module
        SLANeXt-style recognizer.
    """

    return SLANetCompact(dim=60).eval()


def _import_attr(module_name: str, attr_name: str) -> Callable[[], nn.Module]:
    """Import a zero-argument builder from a classics module.

    Parameters
    ----------
    module_name:
        Module name under ``menagerie.classics``.
    attr_name:
        Attribute to import.

    Returns
    -------
    Callable[[], nn.Module]
        Imported builder.
    """

    module = importlib.import_module(f"menagerie.classics.{module_name}")
    return getattr(module, attr_name)


build_rt_detrv3_r50 = paddledet_rtdetrv3.build
build_paddledet_sqr = paddledet_extra.build_queryinst
build_paddledet_vitdet = paddledet_faster_rcnn.build
build_paddledet_yolov3 = paddledet_extra.build_ppyolo
build_ppdet_blazeface = paddledet_extra.build_face
build_ppdet_sniper = paddledet_extra.build_sniper
build_ppdet_vitdet = paddledet_faster_rcnn.build
build_ppdet_focalnet_det = _import_attr("ppdet_retinanet", "build")
build_ppdet_rotate_s2anet = _import_attr("reimpl5_openmmlab", "build_s2anet")
build_ppdet_rotate_fcosr = _import_attr("ppdet_rotate_ppyoloe_r", "build")
build_ppdet_keypoint_hrnet = paddledet_extra.build_keypoint
build_ppdet_pose3d_tinypose3d = _import_attr("ppdet_keypoint_tinypose", "build")
build_ppdet_mot = paddledet_extra.build_mot
build_ppdet_mot_fairmot = _import_attr("mot_reid_tracking", "build_fairmot")
build_ppdet_mot_jde = _import_attr("mot_reid_tracking", "build_jde")
build_ppdet_mot_deepsort = _import_attr("deepsort_appearancenet", "build")
build_ppocr_drrg = reimpl6_dependency_gated.build_drrg
build_ppocr_east = reimpl6_dependency_gated.build_panet_text
build_ppocr_fce = reimpl6_dependency_gated.build_fcenet
build_ppocr_pgnet = pgnet_resnet50_paddleocr.build
build_ppocr_pse = reimpl6_dependency_gated.build_psenet
build_ppocr_parseq = reimpl6_dependency_gated.build_nrtr
build_ppocr_rare = reimpl6_dependency_gated.build_aster
build_ppocr_robustscanner = _import_attr("robustscanner_resnet31", "build")
build_ppocr_rosetta = reimpl6_dependency_gated.build_crnn
build_ppocr_sast = reimpl6_dependency_gated.build_text_mask_rcnn
build_ppocr_sdmgr = build_sdmgr_packed
build_ppocr_spin = _import_attr("spinn", "build")
build_ppocr_srn = reimpl3_10_tabseq.build_srn
build_ppocr_svtr = reimpl6_dependency_gated.build_svtr
build_ppocr_svtrv2 = svtrv2_base.build
build_ppocr_tablemaster = build_table_master
build_ppocr_unimernet = reimpl6_dependency_gated.build_master
build_ppocr_vi_layoutxlm = build_layoutlm_kie
build_ppocr_rec_vitstr = ppocr_vitstr.build
build_ppocr_rec_visionlan = ppocr_visionlan.build
build_ppocr_rec_svtr = reimpl6_dependency_gated.build_svtr
build_ppocr_text_telescope = build_vpd
build_ppocr_text_gestalt = _import_attr("sentence_gestalt", "build")
build_efficient_sam = _import_attr("sam_detr", "build_sam_detr_r50")
build_fastreid = _import_attr("mot_reid_tracking", "build_bot_reid")
build_depth_anything_vits = depth_anything_dpt.build_vits
build_depth_anything_vitl = depth_anything_dpt.build_vitl
build_bts = _import_attr("mono_depth_models", "build_leres")
build_event_spikformer = spikingformer_variants.build_dvs

example_image = image_input
example_table = text_image
example_layout = layout_input
example_tabular = tabular_input
example_video = longtail.example_movinet_input
example_ppdet = paddledet_rtdetrv3.example_input
example_ocr = ppocr_vitstr.example_input
example_depth = depth_anything_dpt.example_input
example_dvs = spikingformer_variants.example_dvs
example_frames = paddledet_extra.example_frames
example_chips = paddledet_extra.example_chips
example_spinn = _import_attr("spinn", "example_input")
example_srn = reimpl3_10_tabseq.example_srn
example_gestalt = _import_attr("sentence_gestalt", "example_input")


def example_alignn_packed() -> Tensor:
    """Return packed ALIGNN atom and edge features.

    Returns
    -------
    Tensor
        Packed graph tensor.
    """

    packed = torch.zeros(13, 8)
    packed[:5, :8] = torch.randn(5, 8)
    packed[5:13, :4] = torch.randn(8, 4)
    return packed


def example_occupancy_packed() -> Tensor:
    """Return packed occupancy query points and latent code.

    Returns
    -------
    Tensor
        Packed occupancy tensor.
    """

    points = torch.randn(1, 32, 3) * 0.4
    code = torch.randn(1, 1, 16).expand(-1, points.shape[1], -1)
    return torch.cat([points, code], dim=-1)


def example_avalanche_mt_packed() -> Tensor:
    """Return packed Avalanche features and task ids.

    Returns
    -------
    Tensor
        Packed feature tensor.
    """

    return torch.cat([torch.randn(2, 20), torch.tensor([[0.0], [2.0]])], dim=1)


def _entry(
    name: str, builder: str, example: str, year: str, note: str
) -> tuple[str, str, str, str, str]:
    """Create a MENAGERIE entry tuple.

    Parameters
    ----------
    name:
        Canonical target name.
    builder:
        Builder function attribute.
    example:
        Example-input function attribute.
    year:
        Publication year.
    note:
        Compact family code.

    Returns
    -------
    tuple[str, str, str, str, str]
        Registry entry.
    """

    return (name, builder, example, year, note)


MENAGERIE_ENTRIES = [
    _entry("mmpose:yoloxpose", "build_paddledet_yolov3", "example_image", "2021", "POSE"),
    _entry("mmpose_yoloxpose", "build_paddledet_yolov3", "example_image", "2021", "POSE"),
    _entry("mmseg_mask2former_seg", "build_mask2former", "example_image", "2022", "SEG"),
    _entry("mmseg_maskformer_seg", "build_maskformer", "example_image", "2021", "SEG"),
    _entry("mmseg:vpd", "build_vpd", "example_image", "2023", "SEG"),
    _entry("mmseg_vpd", "build_vpd", "example_image", "2023", "SEG"),
    _entry("mmseg_vpd_vpd_sd_4xb8_25k_nyu_480x480", "build_vpd", "example_image", "2023", "SEG"),
    _entry("binsformer", "build_binsformer", "example_image", "2022", "DEPTH"),
    _entry("paddleocr_kie_layoutlm", "build_layoutlm_kie", "example_layout", "2019", "OCR"),
    _entry("paddleocr_kie_vi_layoutxlm", "build_layoutlm_kie", "example_layout", "2021", "OCR"),
    _entry("paddleocr_kie_sdmgr", "build_ppocr_sdmgr", "example_layout", "2021", "OCR"),
    _entry("paddleocr_table_master", "build_table_master", "example_table", "2019", "OCR"),
    _entry("paddleocr_table_slanet", "build_slanet", "example_table", "2022", "OCR"),
    _entry("paddleocr_table_slanext", "build_slanext", "example_table", "2023", "OCR"),
    _entry("paddleocr_rec_svtr", "build_ppocr_rec_svtr", "example_ocr", "2022", "OCR"),
    _entry("paddleocr_rec_visionlan", "build_ppocr_rec_visionlan", "example_ocr", "2021", "OCR"),
    _entry("paddleocr_rec_vitstr", "build_ppocr_rec_vitstr", "example_ocr", "2021", "OCR"),
    _entry("paddleocr_sr_telescope", "build_vpd", "example_image", "2023", "SR"),
    _entry("paddleocr_sr_tsrn", "build_ppocr_rosetta", "example_ocr", "2020", "SR"),
    _entry("rt_detrv3_r50", "build_rt_detrv3_r50", "example_ppdet", "2025", "DET"),
    _entry("paddledet_sqr", "build_paddledet_sqr", "example_image", "2021", "DET"),
    _entry("paddledet_vitdet", "build_paddledet_vitdet", "example_image", "2022", "DET"),
    _entry("paddledet_yolov3", "build_paddledet_yolov3", "example_image", "2018", "DET"),
    _entry("ppocr_drrg", "build_ppocr_drrg", "example_image", "2020", "OCR"),
    _entry("ppocr_east", "build_ppocr_east", "example_image", "2017", "OCR"),
    _entry("ppocr_fce", "build_ppocr_fce", "example_image", "2021", "OCR"),
    _entry("ppocr_latexocr", "build_ppocr_unimernet", "example_ocr", "2023", "OCR"),
    _entry("ppocr_parseq", "build_ppocr_parseq", "example_ocr", "2022", "OCR"),
    _entry("ppocr_pgnet", "build_ppocr_pgnet", "example_image", "2021", "OCR"),
    _entry("ppocr_pse", "build_ppocr_pse", "example_image", "2019", "OCR"),
    _entry("ppocr_rare", "build_ppocr_rare", "example_ocr", "2016", "OCR"),
    _entry("ppocr_rfl", "build_ppocr_parseq", "example_ocr", "2021", "OCR"),
    _entry("ppocr_robustscanner", "build_ppocr_robustscanner", "example_ocr", "2020", "OCR"),
    _entry("ppocr_rosetta", "build_ppocr_rosetta", "example_ocr", "2018", "OCR"),
    _entry("ppocr_sast", "build_ppocr_sast", "example_image", "2019", "OCR"),
    _entry("ppocr_sdmgr", "build_ppocr_sdmgr", "example_layout", "2021", "OCR"),
    _entry("ppocr_slanet", "build_slanet", "example_table", "2022", "OCR"),
    _entry("ppocr_spin", "build_ppocr_spin", "example_spinn", "2019", "OCR"),
    _entry("ppocr_srn", "build_ppocr_srn", "example_srn", "2020", "OCR"),
    _entry("ppocr_starnet", "build_ppocr_rosetta", "example_ocr", "2016", "OCR"),
    _entry("ppocr_svtr", "build_ppocr_svtr", "example_ocr", "2022", "OCR"),
    _entry("ppocr_svtrv2", "build_ppocr_svtrv2", "example_ocr", "2024", "OCR"),
    _entry("ppocr_tablemaster", "build_ppocr_tablemaster", "example_table", "2019", "OCR"),
    _entry("ppocr_text_gestalt", "build_ppocr_text_gestalt", "example_gestalt", "1986", "OCR"),
    _entry("ppocr_text_telescope", "build_ppocr_text_telescope", "example_image", "2023", "OCR"),
    _entry("ppocr_unimernet", "build_ppocr_unimernet", "example_ocr", "2023", "OCR"),
    _entry("ppocr_vi_layoutxlm", "build_ppocr_vi_layoutxlm", "example_layout", "2021", "OCR"),
    _entry("ppdet_blazeface", "build_ppdet_blazeface", "example_image", "2019", "DET"),
    _entry("ppdet_mot_bytetrack", "build_ppdet_mot", "example_frames", "2021", "DET"),
    _entry("ppdet_mot_centertrack", "build_ppdet_mot", "example_frames", "2020", "DET"),
    _entry("ppdet_mot_deepsort", "build_ppdet_mot_deepsort", "example_image", "2017", "DET"),
    _entry("ppdet_mot_fairmot", "build_ppdet_mot_fairmot", "example_image", "2020", "DET"),
    _entry("ppdet_rotate_fcosr", "build_ppdet_rotate_fcosr", "example_image", "2019", "DET"),
    _entry("ppdet_focalnet_det", "build_ppdet_focalnet_det", "example_image", "2022", "DET"),
    _entry("ppdet_keypoint_hrnet", "build_ppdet_keypoint_hrnet", "example_image", "2019", "POSE"),
    _entry("ppdet_mot_jde", "build_ppdet_mot_jde", "example_image", "2019", "DET"),
    _entry("ppdet_pose3d_metro", "build_ppdet_pose3d_tinypose3d", "example_image", "2020", "POSE"),
    _entry("ppdet_mot_ocsort", "build_ppdet_mot", "example_frames", "2022", "DET"),
    _entry("ppdet_rotate_s2anet", "build_ppdet_rotate_s2anet", "example_image", "2020", "DET"),
    _entry("ppdet_smrt", "build_ppdet_mot", "example_frames", "2022", "DET"),
    _entry("ppdet_sniper", "build_ppdet_sniper", "example_chips", "2018", "DET"),
    _entry(
        "ppdet_pose3d_tinypose3d", "build_ppdet_pose3d_tinypose3d", "example_image", "2021", "POSE"
    ),
    _entry("ppdet_vitdet", "build_ppdet_vitdet", "example_image", "2022", "DET"),
    _entry("movinet_a0_pytorch", "build_movinet_a1_pytorch", "example_video", "2021", "VIDEO"),
    _entry("movinet_a1_pytorch", "build_movinet_a1_pytorch", "example_video", "2021", "VIDEO"),
    _entry("movinet_a2_pytorch", "build_movinet_a2_pytorch", "example_video", "2021", "VIDEO"),
    _entry("AASIST", "build_aasist", "example_aasist_input", "2021", "AUDIO"),
    _entry("ALIGNN", "build_alignn_packed", "example_alignn_packed", "2021", "GNN"),
    _entry(
        "occupancy_network_decoder",
        "build_occupancy_packed",
        "example_occupancy_packed",
        "2019",
        "GEO",
    ),
    _entry("Avalanche-EWC-MLP", "build_avalanche_ewc_mlp", "example_tabular", "2017", "TAB"),
    _entry(
        "Avalanche MTSimpleMLP",
        "build_avalanche_mt_packed",
        "example_avalanche_mt_packed",
        "2021",
        "TAB",
    ),
    _entry("Avalanche SimpleMLP", "build_avalanche_simple_mlp", "example_tabular", "2021", "TAB"),
    _entry(
        "behavior_transformer_bet",
        "build_behavior_transformer_bet",
        "behavior_input",
        "2022",
        "POLICY",
    ),
    _entry(
        "binary_event_driven_spikformer", "build_event_spikformer", "example_dvs", "2023", "SPIKE"
    ),
    _entry("BigVGAN", "build_bigvgan", "example_bigvgan_input", "2022", "AUDIO"),
    _entry("bigvgan_vocoder", "build_bigvgan", "example_bigvgan_input", "2022", "AUDIO"),
    _entry("ModernNCA-Classifier", "build_modern_nca", "example_tabular", "2023", "TAB"),
    _entry("TabR-Classifier", "build_tabr", "example_tabular", "2023", "TAB"),
    _entry("BitLinear_BitNet", "build_bitlinear_bitnet", "example_bitlinear_input", "2023", "LLM"),
    _entry("Bayesian NN (BBB)", "build_bayesian_nn_bbb", "example_bayesian_input", "2015", "BAYES"),
    _entry("Bonito", "build_bonito", "example_bonito_input", "2019", "BIO"),
    _entry("borzoi", "build_borzoi", "example_borzoi_input", "2023", "BIO"),
    _entry("BrainLM-fMRIFoundation", "build_brainlm", "fmri_input", "2024", "FMRI"),
    _entry("ESMFold2", "build_esmfold2", "protein_tokens", "2024", "BIO"),
    _entry("bts", "build_bts", "example_depth", "2019", "DEPTH"),
    _entry("bts_densenet161", "build_bts", "example_depth", "2019", "DEPTH"),
    _entry("bts_resnet50", "build_bts", "example_depth", "2019", "DEPTH"),
    _entry("glip_swin_t", "build_ppdet_focalnet_det", "example_image", "2022", "DET"),
    _entry("EfficientSAM_ViT_S", "build_efficient_sam", "example_image", "2023", "SEG"),
    _entry("EfficientSAM_ViT_T", "build_efficient_sam", "example_image", "2023", "SEG"),
    _entry("FastReID-AGW-R50", "build_fastreid", "example_image", "2019", "REID"),
    _entry("FastReID-SBS-R50", "build_fastreid", "example_image", "2020", "REID"),
    _entry(
        "depth_anything_v1_metric_kitti_vitl",
        "build_depth_anything_vitl",
        "example_depth",
        "2024",
        "DEPTH",
    ),
    _entry(
        "depth_anything_v1_metric_nyu_vits",
        "build_depth_anything_vits",
        "example_depth",
        "2024",
        "DEPTH",
    ),
]

build_aasist = longtail.build_aasist
example_aasist_input = longtail.example_aasist_input
build_alignn = longtail.build_alignn
example_alignn_input = longtail.example_alignn_input
build_occupancy_network_decoder = longtail.build_occupancy_network_decoder
example_occupancy_input = longtail.example_occupancy_input
build_avalanche_mt_mlp = longtail.build_avalanche_mt_mlp
example_avalanche_input = longtail.example_avalanche_input
build_bigvgan = longtail.build_bigvgan
example_bigvgan_input = longtail.example_bigvgan_input
build_bitlinear_bitnet = longtail.build_bitlinear_bitnet
example_bitlinear_input = longtail.example_bitlinear_input
build_bayesian_nn_bbb = longtail.build_bayesian_nn_bbb
example_bayesian_input = longtail.example_bayesian_input
build_bonito = longtail.build_bonito
example_bonito_input = longtail.example_bonito_input
build_borzoi = longtail.build_borzoi
example_borzoi_input = longtail.example_borzoi_input
