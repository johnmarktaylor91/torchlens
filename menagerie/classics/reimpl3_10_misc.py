"""Compact miscellaneous dependency-gated classics.

Covered primitives include scGPT gene-value generative token modeling,
contrastive representation distillation, Kolmogorov-Arnold Transformer channel
mixing, PDFormer spatial-temporal masked attention, spiking Transformer/BERT
average-spike-rate attention, spiking point features, YOLO-style spiking heads,
Sonata/Topaz compact vision backbones, MIND multi-interest routing, and stereo
cost-volume matching.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _spike(x: torch.Tensor) -> torch.Tensor:
    """Return binary spikes from activations.

    Parameters
    ----------
    x:
        Activation tensor.

    Returns
    -------
    torch.Tensor
        Spike tensor.
    """

    return (x > x.mean()).to(x.dtype)


class ScGPTTiny(nn.Module):
    """scGPT-style generative pretrained Transformer for gene tokens."""

    def __init__(self) -> None:
        """Initialize gene/value embeddings and transformer."""

        super().__init__()
        self.gene = nn.Embedding(64, 24)
        self.value = nn.Linear(1, 24)
        layer = nn.TransformerEncoderLayer(24, 4, 48, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.expr = nn.Linear(24, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict masked expression values from gene/value tokens.

        Parameters
        ----------
        x:
            Tensor shaped ``(batch, genes, 2)`` with gene id and expression.

        Returns
        -------
        torch.Tensor
            Reconstructed expression values.
        """

        ids = x[..., 0].round().long().clamp(0, 63)
        y = self.gene(ids) + self.value(x[..., 1:2])
        return self.expr(self.encoder(y)).squeeze(-1)


class CRDTiny(nn.Module):
    """Contrastive Representation Distillation student-teacher projector."""

    def __init__(self) -> None:
        """Initialize student, teacher, and contrast projection heads."""

        super().__init__()
        self.student = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 12))
        self.teacher = nn.Sequential(nn.Linear(10, 24), nn.ReLU(), nn.Linear(24, 12))
        self.proj_s = nn.Linear(12, 8)
        self.proj_t = nn.Linear(12, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return student-teacher contrastive similarity matrix.

        Parameters
        ----------
        x:
            Input features.

        Returns
        -------
        torch.Tensor
            Contrastive logits.
        """

        s = F.normalize(self.proj_s(self.student(x)), dim=-1)
        t = F.normalize(self.proj_t(self.teacher(x)).detach(), dim=-1)
        return s @ t.t() / 0.07


class KANLayer(nn.Module):
    """Kolmogorov-Arnold spline-like channel mixer."""

    def __init__(self, dim: int = 24) -> None:
        """Initialize basis coefficients.

        Parameters
        ----------
        dim:
            Feature width.
        """

        super().__init__()
        self.coeff = nn.Parameter(torch.randn(dim, 4) * 0.1)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learned univariate basis expansion per channel.

        Parameters
        ----------
        x:
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Mixed features.
        """

        basis = torch.stack([x, x.pow(2), torch.sin(x), torch.cos(x)], dim=-1)
        return self.out((basis * self.coeff).sum(dim=-1))


class KATTransformerTiny(nn.Module):
    """Kolmogorov-Arnold Transformer replacing the MLP with KAN."""

    def __init__(self) -> None:
        """Initialize attention and KAN feed-forward block."""

        super().__init__()
        self.embed = nn.Embedding(64, 24)
        self.attn = nn.MultiheadAttention(24, 4, batch_first=True)
        self.kan = KANLayer(24)
        self.head = nn.Linear(24, 64)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Run KAT sequence model.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        x = self.embed(ids)
        y, _ = self.attn(x, x, x, need_weights=False)
        return self.head(x + y + self.kan(x + y))


class PDFormerTiny(nn.Module):
    """Propagation-delay-aware spatial-temporal Transformer."""

    def __init__(self) -> None:
        """Initialize temporal and spatial attention."""

        super().__init__()
        self.temporal = nn.MultiheadAttention(16, 4, batch_first=True)
        self.node = nn.Linear(3, 16)
        self.spatial = nn.MultiheadAttention(16, 4, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast traffic with delay-biased spatial attention.

        Parameters
        ----------
        x:
            Tensor shaped ``(batch, nodes, time, features)``.

        Returns
        -------
        torch.Tensor
            Node forecasts.
        """

        batch, nodes, time, feat = x.shape
        y = self.node(x).reshape(batch * nodes, time, 16)
        y, _ = self.temporal(y, y, y, need_weights=False)
        y = y[:, -1].reshape(batch, nodes, 16)
        delay = torch.arange(nodes, device=x.device)
        mask = (delay[:, None] - delay[None, :]).abs() > 2
        y, _ = self.spatial(y, y, y, attn_mask=mask, need_weights=False)
        return self.head(y).squeeze(-1)


class SpikingTransformerTiny(nn.Module):
    """Spiking Transformer/BERT with average-spike-rate attention."""

    def __init__(self, vocab: int = 64) -> None:
        """Initialize spiking language layers.

        Parameters
        ----------
        vocab:
            Vocabulary size.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab, 24)
        self.qkv = nn.Linear(24, 72)
        self.ff = nn.Sequential(nn.Linear(24, 48), nn.ReLU(), nn.Linear(48, 24))
        self.head = nn.Linear(24, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Run spike-rate self-attention.

        Parameters
        ----------
        ids:
            Token ids.

        Returns
        -------
        torch.Tensor
            Token logits.
        """

        x = _spike(self.embed(ids))
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = _spike(q), _spike(k), _spike(v)
        attn = torch.softmax(q @ k.transpose(-1, -2) / 24.0, dim=-1)
        y = attn @ v
        return self.head(y + _spike(self.ff(y)))


class SpikingResformerTiny(nn.Module):
    """Spiking ResFormer with residual stages and Dual Spike Self-Attention."""

    def __init__(self) -> None:
        """Initialize vision patch embedding, residual spike blocks, and head."""

        super().__init__()
        self.patch = nn.Conv2d(3, 24, 4, stride=4)
        self.res1 = nn.Conv2d(24, 24, 3, padding=1, groups=24)
        self.res2 = nn.Conv2d(24, 24, 1)
        self.qkv = nn.Conv2d(24, 72, 1)
        self.proj = nn.Conv2d(24, 24, 1)
        self.head = nn.Linear(24, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify images with residual DSSA-style spiking patch features."""

        h = _spike(self.patch(x))
        h = h + _spike(self.res2(_spike(self.res1(h))))
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q_spike = _spike(q).flatten(2).transpose(1, 2)
        k_spike = _spike(k).flatten(2)
        v_spike = _spike(v).flatten(2).transpose(1, 2)
        token_gate = torch.bmm(q_spike, k_spike).clamp_min(0.0) / max(h.shape[-2] * h.shape[-1], 1)
        channel_gate = torch.bmm(k_spike, v_spike).clamp_min(0.0) / max(v_spike.shape[1], 1)
        dssa = torch.bmm(token_gate, v_spike) + torch.bmm(q_spike, channel_gate)
        dssa_map = dssa.transpose(1, 2).reshape_as(h)
        h = h + _spike(self.proj(dssa_map))
        return self.head(h.mean(dim=(2, 3)))


class STAttenSpikformerTiny(nn.Module):
    """STAtten Spikformer with spiking token attention."""

    def __init__(self) -> None:
        """Initialize spiking patch tokens and token attention."""

        super().__init__()
        self.patch = nn.Conv2d(3, 24, 4, stride=4)
        self.qkv = nn.Linear(24, 72)
        self.temporal_gate = nn.Linear(24, 24)
        self.head = nn.Linear(24, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify images with spike-gated STAtten token mixing."""

        tokens = _spike(self.patch(x)).flatten(2).transpose(1, 2)
        q, k, v = self.qkv(tokens).chunk(3, dim=-1)
        spike_attn = torch.softmax(_spike(q) @ _spike(k).transpose(1, 2) / 24.0, dim=-1)
        mixed = spike_attn @ _spike(v)
        gated = mixed * torch.sigmoid(self.temporal_gate(tokens))
        return self.head(gated.mean(1))


class SpikePointTiny(nn.Module):
    """Point-based SNN feature extractor for event/point clouds."""

    def __init__(self) -> None:
        """Initialize point MLP and classifier."""

        super().__init__()
        self.local = nn.Linear(3, 16)
        self.global_fc = nn.Linear(32, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify point cloud with local and global spiking features.

        Parameters
        ----------
        x:
            Point tensor shaped ``(batch, points, 3)``.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        local = _spike(self.local(x))
        global_feat = local.max(dim=1).values
        pooled = torch.cat([global_feat, local.mean(dim=1)], dim=-1)
        return self.global_fc(pooled)


class SpikeYOLOTiny(nn.Module):
    """Spiking YOLO-style detector with objectness/class/regression heads."""

    def __init__(self) -> None:
        """Initialize spiking convolutional detector."""

        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.head = nn.Conv2d(16, 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run spiking detection head.

        Parameters
        ----------
        x:
            Image tensor.

        Returns
        -------
        torch.Tensor
            Dense detection predictions.
        """

        y = _spike(self.c1(x))
        y = _spike(self.c2(y))
        return self.head(y)


class StereoAnywhereTiny(nn.Module):
    """Stereo matching with correlation volume and recurrent refinement."""

    def __init__(self) -> None:
        """Initialize feature extractor and update block."""

        super().__init__()
        self.feat = nn.Conv2d(3, 8, 3, padding=1)
        self.update = nn.Conv2d(6, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate disparity from left/right images.

        Parameters
        ----------
        x:
            Concatenated stereo tensor shaped ``(batch, 6, height, width)``.

        Returns
        -------
        torch.Tensor
            Disparity map.
        """

        left = self.feat(x[:, :3])
        right = self.feat(x[:, 3:])
        costs = []
        for disp in range(4):
            costs.append((left * torch.roll(right, shifts=disp, dims=-1)).mean(dim=1, keepdim=True))
        prob = torch.softmax(torch.cat(costs, dim=1), dim=1)
        d = (prob * torch.arange(4, device=x.device, dtype=x.dtype).view(1, 4, 1, 1)).sum(
            dim=1, keepdim=True
        )
        return d + self.update(x)


class MINDTiny(nn.Module):
    """MIND multi-interest recommender with dynamic routing capsules."""

    def __init__(self) -> None:
        """Initialize item embeddings and routing projection."""

        super().__init__()
        self.item = nn.Embedding(128, 16)
        self.route = nn.Linear(16, 3)
        self.candidate = nn.Embedding(32, 16)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Score candidates using routed user interests.

        Parameters
        ----------
        ids:
            User history item ids.

        Returns
        -------
        torch.Tensor
            Candidate logits.
        """

        emb = self.item(ids)
        weights = torch.softmax(self.route(emb), dim=1)
        interests = torch.einsum("btk,btd->bkd", weights, emb)
        scores = torch.einsum("bkd,nd->bkn", interests, self.candidate.weight)
        return scores.max(dim=1).values


class ConvVisionTiny(nn.Module):
    """Compact named vision backbone for Topaz/Sonata-style classifiers."""

    def __init__(self) -> None:
        """Initialize convolutional classifier."""

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.SiLU(), nn.Conv2d(8, 16, 3, 2, 1), nn.SiLU()
        )
        self.head = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify image patches.

        Parameters
        ----------
        x:
            Image patch tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """

        return self.head(self.net(x).mean(dim=(2, 3)))


def example_gene_tokens() -> torch.Tensor:
    """Return scGPT gene/value tokens."""

    ids = torch.arange(8, dtype=torch.float32).view(1, 8, 1)
    vals = torch.rand(1, 8, 1)
    return torch.cat([ids, vals], dim=-1)


def example_dense() -> torch.Tensor:
    """Return dense feature input."""

    return torch.randn(3, 10)


def example_tokens() -> torch.Tensor:
    """Return token ids."""

    return torch.randint(0, 64, (1, 8))


def example_traffic() -> torch.Tensor:
    """Return traffic tensor."""

    return torch.randn(1, 5, 6, 3)


def example_points() -> torch.Tensor:
    """Return point cloud."""

    return torch.randn(1, 12, 3)


def example_image() -> torch.Tensor:
    """Return RGB image."""

    return torch.randn(1, 3, 16, 16)


def example_stereo() -> torch.Tensor:
    """Return stereo image pair."""

    return torch.randn(1, 6, 16, 16)


def example_history() -> torch.Tensor:
    """Return item history."""

    return torch.randint(0, 128, (1, 6))


def example_patch() -> torch.Tensor:
    """Return grayscale image patch."""

    return torch.randn(1, 1, 16, 16)


def build_scgpt() -> nn.Module:
    """Build scGPT model."""

    return ScGPTTiny()


def build_crd() -> nn.Module:
    """Build RepDistiller CRD model."""

    return CRDTiny()


def build_kat_transformer() -> nn.Module:
    """Build KAT Transformer model."""

    return KATTransformerTiny()


def build_st_pdformer() -> nn.Module:
    """Build ST-PDFormer model."""

    return PDFormerTiny()


def build_spiking_transformer() -> nn.Module:
    """Build spiking language Transformer."""

    return SpikingTransformerTiny()


def build_spiking_resformer() -> nn.Module:
    """Build spiking ResFormer vision model."""

    return SpikingResformerTiny()


def build_statten_spikformer() -> nn.Module:
    """Build STAtten Spikformer vision model."""

    return STAttenSpikformerTiny()


def build_spikepoint() -> nn.Module:
    """Build SpikePoint model."""

    return SpikePointTiny()


def build_spike_yolo() -> nn.Module:
    """Build spiking YOLO model."""

    return SpikeYOLOTiny()


def build_stereo_anywhere() -> nn.Module:
    """Build Stereo Anywhere compact model."""

    return StereoAnywhereTiny()


def build_mind() -> nn.Module:
    """Build MIND recommender."""

    return MINDTiny()


def build_topaz() -> nn.Module:
    """Build Topaz particle-picking classifier."""

    return ConvVisionTiny()


def build_sonata() -> nn.Module:
    """Build Sonata compact vision model."""

    return ConvVisionTiny()


MENAGERIE_ENTRIES = [
    ("scgpt", "build_scgpt", "example_gene_tokens", "2023", "single-cell/transformer"),
    ("Sonata", "build_sonata", "example_patch", "2024", "vision"),
    ("RepDistiller-CRD", "build_crd", "example_dense", "2020", "distillation"),
    ("KAT-transformer", "build_kat_transformer", "example_tokens", "2025", "transformer/kan"),
    ("ST-PDFormer", "build_st_pdformer", "example_traffic", "2023", "traffic/transformer"),
    ("spikebert", "build_spiking_transformer", "example_tokens", "2024", "spiking/language"),
    ("spikelm_bert", "build_spiking_transformer", "example_tokens", "2024", "spiking/language"),
    ("SpikePoint", "build_spikepoint", "example_points", "2024", "spiking/point-cloud"),
    ("spike_yolo", "build_spike_yolo", "example_image", "2024", "spiking/detection"),
    ("spikingbert", "build_spiking_transformer", "example_tokens", "2024", "spiking/language"),
    (
        "spikingresformer_ti",
        "build_spiking_resformer",
        "example_image",
        "2024",
        "spiking/vision-transformer",
    ),
    ("spikingtorch_bp", "build_spiking_transformer", "example_tokens", "2021", "spiking"),
    (
        "statten_spikformer",
        "build_statten_spikformer",
        "example_image",
        "2024",
        "spiking/transformer",
    ),
    ("stereo_anywhere", "build_stereo_anywhere", "example_stereo", "2024", "vision/stereo"),
    ("Topaz", "build_topaz", "example_patch", "2019", "vision/particle-picking"),
    ("MIND", "build_mind", "example_history", "2019", "recommender"),
]
