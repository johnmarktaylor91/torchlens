"""Compact classics for notable missing speech, audio, and music architectures.

Paper: Hershey et al. 2016, "Deep clustering: Discriminative embeddings for
segmentation and separation"; Chen et al. 2017, "Deep attractor network for
single-microphone speaker separation"; Stoller et al. 2018, "Wave-U-Net"; Huang
et al. 2019, "Music Transformer"; YAMNet TensorFlow Hub model card, 2019;
Niizumi et al. 2021, "BYOL for Audio"; Hao et al. 2021, "FullSubNet"; Chen
et al. 2022, "HTS-AT".
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DepthwiseSeparableConv2d(nn.Module):
    """MobileNet-style depthwise separable convolution block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize depthwise and pointwise convolutions.

        Parameters
        ----------
        in_channels:
            Number of input feature channels.
        out_channels:
            Number of output feature channels.
        stride:
            Stride for the depthwise convolution.
        """
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply depthwise separable convolution.

        Parameters
        ----------
        x:
            Input spectrogram tensor.

        Returns
        -------
        Tensor
            Convolved activations.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return F.relu(self.norm(x))


class YAMNetTiny(nn.Module):
    """YAMNet-style AudioSet classifier using MobileNetV1 blocks."""

    def __init__(self, num_classes: int = 32) -> None:
        """Initialize the compact YAMNet classifier.

        Parameters
        ----------
        num_classes:
            Number of audio event logits.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv2d(8, 16),
            DepthwiseSeparableConv2d(16, 32, stride=2),
            DepthwiseSeparableConv2d(32, 48),
            DepthwiseSeparableConv2d(48, 64, stride=2),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, waveform_patch: Tensor) -> Tensor:
        """Classify a log-mel patch.

        Parameters
        ----------
        waveform_patch:
            Log-mel-like patch with shape ``(batch, 1, time, mel)``.

        Returns
        -------
        Tensor
            Audio event logits.
        """
        x = self.blocks(self.stem(waveform_patch))
        return self.classifier(x.mean(dim=(-2, -1)))


class RelativeSelfAttention(nn.Module):
    """Causal self-attention with learned relative-position logits."""

    def __init__(self, d_model: int, num_heads: int, max_len: int) -> None:
        """Initialize projections and relative position embeddings.

        Parameters
        ----------
        d_model:
            Token embedding width.
        num_heads:
            Number of attention heads.
        max_len:
            Maximum sequence length for learned relative offsets.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.rel = nn.Parameter(torch.randn(num_heads, max_len, self.head_dim) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal content and relative attention.

        Parameters
        ----------
        x:
            Token embeddings with shape ``(batch, time, d_model)``.

        Returns
        -------
        Tensor
            Contextualized token embeddings.
        """
        batch, seq_len, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        content = torch.matmul(q, k.transpose(-1, -2))
        offsets = torch.arange(seq_len, device=x.device)
        rel_index = (offsets[:, None] - offsets[None, :]).clamp(min=0)
        rel = self.rel[:, rel_index]
        rel_logits = torch.einsum("bhtd,htsd->bhts", q, rel)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device).triu(1)
        weights = torch.softmax((content + rel_logits) / math.sqrt(self.head_dim) + mask, dim=-1)
        y = torch.matmul(weights, v).transpose(1, 2).reshape(batch, seq_len, width)
        return self.out(y)


class MusicTransformerTiny(nn.Module):
    """Music Transformer-style event language model with relative attention."""

    def __init__(
        self, vocab_size: int = 64, d_model: int = 48, num_heads: int = 4, max_len: int = 16
    ) -> None:
        """Initialize embeddings, relative attention, and logits head.

        Parameters
        ----------
        vocab_size:
            Size of the symbolic music event vocabulary.
        d_model:
            Token embedding width.
        num_heads:
            Number of attention heads.
        max_len:
            Maximum context length.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        self.attn = RelativeSelfAttention(d_model, num_heads, max_len)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        """Predict symbolic music event logits.

        Parameters
        ----------
        tokens:
            Integer music-event tokens with shape ``(batch, time)``.

        Returns
        -------
        Tensor
            Autoregressive logits.
        """
        x = self.embed(tokens) + self.pos[: tokens.shape[1]]
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return self.head(x)


class HTSATPatchMerge(nn.Module):
    """Swin-style patch merging for hierarchical audio tokens."""

    def __init__(self, dim: int) -> None:
        """Initialize patch-merge projection.

        Parameters
        ----------
        dim:
            Input token width.
        """
        super().__init__()
        self.proj = nn.Linear(4 * dim, 2 * dim)

    def forward(self, x: Tensor) -> Tensor:
        """Merge adjacent time-frequency token neighborhoods.

        Parameters
        ----------
        x:
            Token grid with shape ``(batch, height, width, dim)``.

        Returns
        -------
        Tensor
            Downsampled token grid.
        """
        x00 = x[:, 0::2, 0::2]
        x01 = x[:, 0::2, 1::2]
        x10 = x[:, 1::2, 0::2]
        x11 = x[:, 1::2, 1::2]
        return self.proj(torch.cat([x00, x01, x10, x11], dim=-1))


class HTSATTiny(nn.Module):
    """HTS-AT-style hierarchical audio transformer with token-semantic maps."""

    def __init__(self, num_classes: int = 24, dim: int = 32) -> None:
        """Initialize patch embedder, transformer stages, and semantic head.

        Parameters
        ----------
        num_classes:
            Number of sound classes.
        dim:
            First-stage token width.
        """
        super().__init__()
        self.patch = nn.Conv2d(1, dim, kernel_size=4, stride=4)
        self.stage1 = nn.TransformerEncoderLayer(dim, nhead=4, dim_feedforward=64, batch_first=True)
        self.merge = HTSATPatchMerge(dim)
        self.stage2 = nn.TransformerEncoderLayer(
            2 * dim, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.semantic = nn.Conv2d(2 * dim, num_classes, kernel_size=1)

    def forward(self, spectrogram: Tensor) -> Tensor:
        """Compute class logits and class activation maps.

        Parameters
        ----------
        spectrogram:
            Log-mel spectrogram with shape ``(batch, 1, time, mel)``.

        Returns
        -------
        Tensor
            Concatenated clip logits and pooled token-semantic map summaries.
        """
        x = self.patch(spectrogram).permute(0, 2, 3, 1)
        batch, height, width, dim = x.shape
        x = self.stage1(x.reshape(batch, height * width, dim)).reshape(batch, height, width, dim)
        x = self.merge(x)
        batch, height, width, dim = x.shape
        x = self.stage2(x.reshape(batch, height * width, dim)).reshape(batch, height, width, dim)
        maps = self.semantic(x.permute(0, 3, 1, 2))
        return torch.cat([maps.mean(dim=(-2, -1)), maps.amax(dim=(-2, -1))], dim=-1)


class ConvAudioEncoder(nn.Module):
    """Small convolutional audio encoder used by BYOL-A."""

    def __init__(self, out_dim: int = 48) -> None:
        """Initialize convolutional encoder layers.

        Parameters
        ----------
        out_dim:
            Output embedding width.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode an augmented audio view.

        Parameters
        ----------
        x:
            Log-mel view with shape ``(batch, 1, time, mel)``.

        Returns
        -------
        Tensor
            Audio representation vector.
        """
        return self.net(x)


class BYOLATiny(nn.Module):
    """BYOL-A-style online/target audio representation model."""

    def __init__(self, dim: int = 48, proj_dim: int = 32) -> None:
        """Initialize online and target branches.

        Parameters
        ----------
        dim:
            Encoder representation width.
        proj_dim:
            Projector and predictor width.
        """
        super().__init__()
        self.online = ConvAudioEncoder(dim)
        self.target = ConvAudioEncoder(dim)
        self.projector = nn.Sequential(
            nn.Linear(dim, proj_dim), nn.BatchNorm1d(proj_dim), nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, views: tuple[Tensor, Tensor]) -> Tensor:
        """Predict one augmented audio representation from another.

        Parameters
        ----------
        views:
            Pair of augmented spectrogram views.

        Returns
        -------
        Tensor
            Cosine similarities for the positive BYOL-A pair.
        """
        online_view, target_view = views
        prediction = F.normalize(self.predictor(self.projector(self.online(online_view))), dim=-1)
        with torch.no_grad():
            target = F.normalize(self.projector(self.target(target_view)), dim=-1)
        return (prediction * target).sum(dim=-1)


class FullSubNetTiny(nn.Module):
    """FullSubNet-style full-band plus sub-band speech enhancement model."""

    def __init__(self, freq_bins: int = 16, context: int = 3, hidden: int = 24) -> None:
        """Initialize full-band and sub-band recurrent branches.

        Parameters
        ----------
        freq_bins:
            Number of frequency bins.
        context:
            Number of neighboring bins in each sub-band window.
        hidden:
            Recurrent hidden width.
        """
        super().__init__()
        self.context = context
        self.fullband = nn.GRU(freq_bins, hidden, batch_first=True)
        self.subband = nn.GRU(context + hidden, hidden, batch_first=True)
        self.mask = nn.Linear(hidden, 2)

    def forward(self, noisy_mag: Tensor) -> Tensor:
        """Estimate a complex speech-enhancement mask.

        Parameters
        ----------
        noisy_mag:
            Magnitude features with shape ``(batch, time, freq)``.

        Returns
        -------
        Tensor
            Real/imaginary mask estimates with shape ``(batch, time, freq, 2)``.
        """
        full, _ = self.fullband(noisy_mag)
        pad = self.context // 2
        neighborhoods = noisy_mag.transpose(1, 2).unfold(dimension=1, size=self.context, step=1)
        if pad:
            noisy_pad = F.pad(noisy_mag.transpose(1, 2), (0, 0, pad, pad), mode="replicate")
            neighborhoods = noisy_pad.unfold(dimension=1, size=self.context, step=1)
        batch, freq, time, ctx = neighborhoods.shape
        full_expanded = full.unsqueeze(1).expand(batch, freq, time, full.shape[-1])
        sub_in = torch.cat([neighborhoods, full_expanded], dim=-1).reshape(batch * freq, time, -1)
        sub, _ = self.subband(sub_in)
        return self.mask(sub).reshape(batch, freq, time, 2).transpose(1, 2)


class DeepClusteringTiny(nn.Module):
    """Deep Clustering-style spectrogram embedding network."""

    def __init__(self, freq_bins: int = 12, embed_dim: int = 6, hidden: int = 24) -> None:
        """Initialize recurrent embedding network.

        Parameters
        ----------
        freq_bins:
            Number of spectrogram bins.
        embed_dim:
            Embedding width per time-frequency bin.
        hidden:
            Recurrent hidden width.
        """
        super().__init__()
        self.freq_bins = freq_bins
        self.embed_dim = embed_dim
        self.rnn = nn.LSTM(freq_bins, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * hidden, freq_bins * embed_dim)

    def forward(self, mixture: Tensor) -> Tensor:
        """Produce normalized embeddings and an affinity summary.

        Parameters
        ----------
        mixture:
            Mixture spectrogram with shape ``(batch, time, freq)``.

        Returns
        -------
        Tensor
            Low-rank affinity matrix over time-frequency bins.
        """
        hidden, _ = self.rnn(mixture)
        emb = self.proj(hidden).reshape(mixture.shape[0], -1, self.embed_dim)
        emb = F.normalize(emb, dim=-1)
        return torch.matmul(emb, emb.transpose(-1, -2))


class DeepAttractorTiny(nn.Module):
    """Deep Attractor Network-style separator."""

    def __init__(self, freq_bins: int = 12, embed_dim: int = 6, sources: int = 2) -> None:
        """Initialize embedding network and learned attractor queries.

        Parameters
        ----------
        freq_bins:
            Number of spectrogram bins.
        embed_dim:
            Embedding width per time-frequency bin.
        sources:
            Number of separated sources.
        """
        super().__init__()
        self.embedder = DeepClusteringTiny(freq_bins=freq_bins, embed_dim=embed_dim)
        self.rnn = nn.LSTM(freq_bins, 24, batch_first=True, bidirectional=True)
        self.embed = nn.Linear(48, freq_bins * embed_dim)
        self.attractors = nn.Parameter(torch.randn(sources, embed_dim) * 0.02)
        self.freq_bins = freq_bins
        self.embed_dim = embed_dim

    def forward(self, mixture: Tensor) -> Tensor:
        """Estimate source masks from learned attractors.

        Parameters
        ----------
        mixture:
            Mixture spectrogram with shape ``(batch, time, freq)``.

        Returns
        -------
        Tensor
            Source masks with shape ``(batch, sources, time, freq)``.
        """
        hidden, _ = self.rnn(mixture)
        emb = self.embed(hidden).reshape(mixture.shape[0], -1, self.embed_dim)
        emb = F.normalize(emb, dim=-1)
        attractors = F.normalize(self.attractors, dim=-1)
        masks = torch.softmax(torch.matmul(emb, attractors.t()), dim=-1)
        return masks.transpose(1, 2).reshape(mixture.shape[0], -1, mixture.shape[1], self.freq_bins)


class WaveUNetTiny(nn.Module):
    """Wave-U-Net-style multi-scale 1D source separator."""

    def __init__(self, sources: int = 2) -> None:
        """Initialize encoder, decoder, skip projections, and additive head.

        Parameters
        ----------
        sources:
            Number of output sources.
        """
        super().__init__()
        self.down1 = nn.Conv1d(1, 8, kernel_size=5, padding=2)
        self.down2 = nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2)
        self.down3 = nn.Conv1d(16, 24, kernel_size=5, stride=2, padding=2)
        self.up2 = nn.ConvTranspose1d(24, 16, kernel_size=4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose1d(32, 8, kernel_size=4, stride=2, padding=1)
        self.head = nn.Conv1d(16, sources, kernel_size=1)

    def forward(self, waveform: Tensor) -> Tensor:
        """Separate raw waveform sources with additive residual correction.

        Parameters
        ----------
        waveform:
            Mono waveform with shape ``(batch, 1, samples)``.

        Returns
        -------
        Tensor
            Source waveforms whose sum is tied to the input mixture.
        """
        skip1 = F.relu(self.down1(waveform))
        skip2 = F.relu(self.down2(skip1))
        bottleneck = F.relu(self.down3(skip2))
        x = F.relu(self.up2(bottleneck))
        x = F.relu(self.up1(torch.cat([x, skip2], dim=1)))
        raw_sources = self.head(torch.cat([x, skip1], dim=1))
        correction = (waveform - raw_sources.sum(dim=1, keepdim=True)) / raw_sources.shape[1]
        return raw_sources + correction


def build_yamnet() -> nn.Module:
    """Build a compact YAMNet-style classifier.

    Returns
    -------
    nn.Module
        Random-initialized compact model.
    """
    return YAMNetTiny()


def example_yamnet() -> Tensor:
    """Create a small log-mel patch for YAMNet.

    Returns
    -------
    Tensor
        Example input tensor.
    """
    return torch.randn(1, 1, 64, 32)


def build_music_transformer() -> nn.Module:
    """Build a compact Music Transformer.

    Returns
    -------
    nn.Module
        Random-initialized compact model.
    """
    return MusicTransformerTiny()


def example_music_transformer() -> Tensor:
    """Create symbolic music event tokens.

    Returns
    -------
    Tensor
        Example token tensor.
    """
    return torch.randint(0, 64, (1, 12))


def build_htsat() -> nn.Module:
    """Build a compact HTS-AT model.

    Returns
    -------
    nn.Module
        Random-initialized compact model.
    """
    return HTSATTiny()


def example_htsat() -> Tensor:
    """Create a log-mel spectrogram for HTS-AT.

    Returns
    -------
    Tensor
        Example input tensor.
    """
    return torch.randn(1, 1, 32, 32)


def build_byola() -> nn.Module:
    """Build a compact BYOL-A model.

    Returns
    -------
    nn.Module
        Random-initialized compact model.
    """
    return BYOLATiny()


def example_byola() -> tuple[Tensor, Tensor]:
    """Create paired augmented log-mel views.

    Returns
    -------
    tuple[Tensor, Tensor]
        Example view pair.
    """
    return torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32)


def build_fullsubnet() -> nn.Module:
    """Build a compact FullSubNet model.

    Returns
    -------
    nn.Module
        Random-initialized compact model.
    """
    return FullSubNetTiny()


def example_fullsubnet() -> Tensor:
    """Create noisy magnitude features.

    Returns
    -------
    Tensor
        Example magnitude tensor.
    """
    return torch.rand(1, 6, 16)


def build_deep_clustering() -> nn.Module:
    """Build a compact Deep Clustering separator.

    Returns
    -------
    nn.Module
        Random-initialized compact model.
    """
    return DeepClusteringTiny()


def example_deep_clustering() -> Tensor:
    """Create mixture magnitude features.

    Returns
    -------
    Tensor
        Example mixture tensor.
    """
    return torch.rand(1, 5, 12)


def build_deep_attractor() -> nn.Module:
    """Build a compact Deep Attractor Network.

    Returns
    -------
    nn.Module
        Random-initialized compact model.
    """
    return DeepAttractorTiny()


def example_deep_attractor() -> Tensor:
    """Create mixture magnitude features.

    Returns
    -------
    Tensor
        Example mixture tensor.
    """
    return torch.rand(1, 5, 12)


def build_waveunet() -> nn.Module:
    """Build a compact Wave-U-Net separator.

    Returns
    -------
    nn.Module
        Random-initialized compact model.
    """
    return WaveUNetTiny()


def example_waveunet() -> Tensor:
    """Create a short mono waveform.

    Returns
    -------
    Tensor
        Example waveform tensor.
    """
    return torch.randn(1, 1, 64)


MENAGERIE_ENTRIES = [
    ("YAMNet (MobileNetV1 AudioSet classifier)", "build_yamnet", "example_yamnet", "2019", "DE"),
    (
        "Music Transformer (relative-attention music LM)",
        "build_music_transformer",
        "example_music_transformer",
        "2019",
        "DE",
    ),
    (
        "HTS-AT (hierarchical token-semantic audio transformer)",
        "build_htsat",
        "example_htsat",
        "2022",
        "DE",
    ),
    ("BYOL-A (Bootstrap Your Own Latent for Audio)", "build_byola", "example_byola", "2021", "DE"),
    (
        "FullSubNet (full-band/sub-band speech enhancement)",
        "build_fullsubnet",
        "example_fullsubnet",
        "2021",
        "DE",
    ),
    (
        "Deep Clustering speech separation",
        "build_deep_clustering",
        "example_deep_clustering",
        "2016",
        "DE",
    ),
    (
        "Deep Attractor Network (DANet) speech separation",
        "build_deep_attractor",
        "example_deep_attractor",
        "2017",
        "DE",
    ),
    ("Wave-U-Net audio source separator", "build_waveunet", "example_waveunet", "2018", "DE"),
]
