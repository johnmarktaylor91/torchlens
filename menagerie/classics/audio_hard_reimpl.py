"""Compact audio and speech architectures for hard menagerie rows.

Paper: Gulati et al. 2020, "Conformer"; Subakan et al. 2021, "SepFormer";
Defossez et al. 2019/2021, "Demucs"; Défossez et al. 2022, "High Fidelity
Neural Audio Compression"; Baevski et al. 2020, "wav2vec 2.0"; Fu et al. 2021,
"MetricGAN+".

The implementations are random-init Torch-only forward graphs that preserve each
family's load-bearing structure: convolutional audio encoders, recurrent or
Transformer sequence modeling, dual-path separation masks, U-Net waveform
decoders, residual vector-quantization proxies, and objective speech-quality
heads. They intentionally avoid package-specific I/O wrappers and pretrained
bundles.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConformerBlock(nn.Module):
    """Conformer block with FFN, self-attention, and depthwise convolution."""

    def __init__(self, dim: int = 128, heads: int = 4) -> None:
        """Initialize block layers.

        Parameters
        ----------
        dim:
            Feature dimension.
        heads:
            Attention heads.
        """
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.conv_norm = nn.LayerNorm(dim)
        self.pointwise1 = nn.Conv1d(dim, dim * 2, 1)
        self.depthwise = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.pointwise2 = nn.Conv1d(dim, dim, 1)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply a Conformer block.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        Tensor
            Updated sequence.
        """
        x = x + 0.5 * self.ffn1(x)
        attn_in = self.attn_norm(x)
        x = x + self.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        conv = self.conv_norm(x).transpose(1, 2)
        gate, value = self.pointwise1(conv).chunk(2, dim=1)
        conv = self.pointwise2(F.silu(gate) * self.depthwise(value)).transpose(1, 2)
        return x + conv + 0.5 * self.ffn2(x)


class ConformerEncoder(nn.Module):
    """SpeechBrain-style Conformer encoder."""

    def __init__(self, input_dim: int = 256, dim: int = 128, layers: int = 2) -> None:
        """Initialize projection and Conformer layers.

        Parameters
        ----------
        input_dim:
            Input feature width.
        dim:
            Hidden width.
        layers:
            Number of Conformer blocks.
        """
        super().__init__()
        self.proj = nn.Linear(input_dim, dim)
        self.layers = nn.ModuleList([ConformerBlock(dim) for _ in range(layers)])
        self.out = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Encode acoustic features.

        Parameters
        ----------
        x:
            Acoustic tensor ``(batch, time, features)``.

        Returns
        -------
        Tensor
            Encoded sequence.
        """
        y = self.proj(x)
        for layer in self.layers:
            y = layer(y)
        return self.out(y)


class MetricGANPlusGenerator(nn.Module):
    """MetricGAN+ spectrogram enhancement generator."""

    def __init__(self, freq_bins: int = 257) -> None:
        """Initialize recurrent mask estimator.

        Parameters
        ----------
        freq_bins:
            Spectrogram frequency bins.
        """
        super().__init__()
        self.encoder = nn.Linear(freq_bins, 128)
        self.rnn = nn.GRU(128, 96, batch_first=True, bidirectional=True)
        self.mask = nn.Sequential(nn.Linear(192, freq_bins), nn.Sigmoid())

    def forward(self, spec: Tensor) -> Tensor:
        """Enhance a magnitude spectrogram.

        Parameters
        ----------
        spec:
            Spectrogram tensor ``(batch, time, freq)``.

        Returns
        -------
        Tensor
            Enhanced spectrogram.
        """
        hidden = torch.relu(self.encoder(spec))
        enhanced, _ = self.rnn(hidden)
        return spec * self.mask(enhanced)


class SepFormerSeparator(nn.Module):
    """Dual-path Transformer speech separator."""

    def __init__(self, sources: int = 2) -> None:
        """Initialize encoder, dual-path transformer, and decoder.

        Parameters
        ----------
        sources:
            Number of separated sources.
        """
        super().__init__()
        self.sources = sources
        self.encoder = nn.Conv1d(1, 64, 16, stride=8, padding=4)
        layer = nn.TransformerEncoderLayer(64, 4, 128, batch_first=True)
        self.intra = nn.TransformerEncoder(layer, num_layers=1)
        self.inter = nn.TransformerEncoder(layer, num_layers=1)
        self.mask = nn.Conv1d(64, sources * 64, 1)
        self.decoder = nn.ConvTranspose1d(64, 1, 16, stride=8, padding=4)

    def forward(self, wav: Tensor) -> Tensor:
        """Separate a mono waveform.

        Parameters
        ----------
        wav:
            Waveform tensor ``(batch, time)``.

        Returns
        -------
        Tensor
            Source waveforms ``(batch, sources, time)``.
        """
        encoded = F.relu(self.encoder(wav.unsqueeze(1)))
        seq = encoded.transpose(1, 2)
        seq = self.inter(self.intra(seq)).transpose(1, 2)
        masks = torch.sigmoid(self.mask(seq)).view(wav.shape[0], self.sources, 64, -1)
        decoded = [
            self.decoder(encoded * masks[:, index]).squeeze(1) for index in range(self.sources)
        ]
        return torch.stack(decoded, dim=1)[..., : wav.shape[-1]]


class TensorBiRNN(nn.Module):
    """Bidirectional recurrent bottleneck that returns a plain tensor."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Initialize forward and backward recurrent cells.

        Parameters
        ----------
        input_dim:
            Input feature width.
        hidden_dim:
            Hidden state width.
        """
        super().__init__()
        self.forward_cell = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.backward_cell = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        """Run bidirectional recurrence over a sequence.

        Parameters
        ----------
        x:
            Sequence tensor ``(batch, time, features)``.

        Returns
        -------
        Tensor
            Concatenated forward/backward hidden states.
        """
        batch = x.shape[0]
        hidden_f = torch.zeros(batch, self.hidden_dim, device=x.device, dtype=x.dtype)
        hidden_b = torch.zeros(batch, self.hidden_dim, device=x.device, dtype=x.dtype)
        forward_states = []
        backward_states = []
        for step in x.unbind(dim=1):
            hidden_f = torch.tanh(self.forward_cell(torch.cat((step, hidden_f), dim=-1)))
            forward_states.append(hidden_f)
        for step in reversed(x.unbind(dim=1)):
            hidden_b = torch.tanh(self.backward_cell(torch.cat((step, hidden_b), dim=-1)))
            backward_states.append(hidden_b)
        backward_states = list(reversed(backward_states))
        return torch.cat(
            (torch.stack(forward_states, dim=1), torch.stack(backward_states, dim=1)), dim=-1
        )


class DemucsLike(nn.Module):
    """Waveform U-Net source separator with recurrent bottleneck."""

    def __init__(self, channels: int = 2, sources: int = 4, hybrid: bool = False) -> None:
        """Initialize Demucs-style encoder/decoder.

        Parameters
        ----------
        channels:
            Audio channels.
        sources:
            Output source count.
        hybrid:
            Whether to include a Transformer bottleneck for HDemucs-style rows.
        """
        super().__init__()
        self.sources = sources
        self.channels = channels
        self.hybrid = hybrid
        self.enc1 = nn.Conv1d(channels, 32, 8, stride=4, padding=2)
        self.enc2 = nn.Conv1d(32, 64, 8, stride=4, padding=2)
        self.lstm = TensorBiRNN(64, 64)
        self.bridge = nn.Linear(128, 64)
        layer = nn.TransformerEncoderLayer(64, 4, 128, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.dec2 = nn.ConvTranspose1d(64, 32, 8, stride=4, padding=2)
        self.dec1 = nn.ConvTranspose1d(32, sources * channels, 8, stride=4, padding=2)

    def forward(self, wav: Tensor) -> Tensor:
        """Separate stereo waveform sources.

        Parameters
        ----------
        wav:
            Waveform tensor ``(batch, channels, time)``.

        Returns
        -------
        Tensor
            Source tensor ``(batch, sources, channels, time)``.
        """
        x1 = F.gelu(self.enc1(wav))
        x2 = F.gelu(self.enc2(x1))
        seq = self.lstm(x2.transpose(1, 2))
        seq = self.bridge(seq)
        if self.hybrid:
            seq = self.transformer(seq)
        y = F.gelu(self.dec2(seq.transpose(1, 2)))
        y = self.dec1(y)[..., : wav.shape[-1]]
        return y.view(wav.shape[0], self.sources, self.channels, -1)


class EnCodecLike(nn.Module):
    """EnCodec-style convolutional audio autoencoder with quantization proxy."""

    def __init__(self, channels: int = 2) -> None:
        """Initialize encoder and decoder.

        Parameters
        ----------
        channels:
            Audio channels.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 32, 7, padding=3),
            nn.ELU(),
            nn.Conv1d(32, 64, 8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv1d(64, 96, 8, stride=4, padding=2),
        )
        self.codebook = nn.Linear(96, 96)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(96, 64, 8, stride=4, padding=2),
            nn.ELU(),
            nn.ConvTranspose1d(64, 32, 8, stride=4, padding=2),
            nn.ELU(),
            nn.Conv1d(32, channels, 7, padding=3),
        )

    def forward(self, wav: Tensor) -> Tensor:
        """Reconstruct audio through a quantization proxy.

        Parameters
        ----------
        wav:
            Waveform tensor.

        Returns
        -------
        Tensor
            Reconstructed waveform.
        """
        z = self.encoder(wav).transpose(1, 2)
        z = self.codebook(torch.tanh(z)).transpose(1, 2)
        return self.decoder(z)[..., : wav.shape[-1]]


class Wav2Vec2Like(nn.Module):
    """wav2vec 2.0/XLSR-style convolutional Transformer encoder."""

    def __init__(self) -> None:
        """Initialize feature extractor and Transformer encoder."""
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(1, 32, 10, stride=5, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, 8, stride=4, padding=2),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(64, 4, 128, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.proj = nn.Linear(64, 32)

    def forward(self, wav: Tensor) -> Tensor:
        """Encode raw waveform frames.

        Parameters
        ----------
        wav:
            Waveform tensor ``(batch, time)``.

        Returns
        -------
        Tensor
            Context representations.
        """
        features = self.feature(wav.unsqueeze(1)).transpose(1, 2)
        return self.proj(self.encoder(features))


class SQUIMObjectiveLike(nn.Module):
    """Objective speech quality estimator with three metric heads."""

    def __init__(self) -> None:
        """Initialize quality estimator layers."""
        super().__init__()
        self.encoder = Wav2Vec2Like()
        self.head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(inplace=False), nn.Linear(32, 3))

    def forward(self, wav: Tensor) -> Tensor:
        """Estimate objective quality metrics.

        Parameters
        ----------
        wav:
            Waveform tensor.

        Returns
        -------
        Tensor
            STOI/PESQ/SI-SDR-like metric predictions.
        """
        encoded = self.encoder(wav).mean(dim=1)
        return self.head(encoded)


def build_conformer_encoder() -> nn.Module:
    """Build a Conformer encoder.

    Returns
    -------
    nn.Module
        Conformer encoder.
    """
    return ConformerEncoder()


def build_metricgan_plus() -> nn.Module:
    """Build a MetricGAN+ generator.

    Returns
    -------
    nn.Module
        Enhancement generator.
    """
    return MetricGANPlusGenerator()


def build_sepformer() -> nn.Module:
    """Build a SepFormer separator.

    Returns
    -------
    nn.Module
        Separator.
    """
    return SepFormerSeparator()


def build_demucs() -> nn.Module:
    """Build a Demucs-like separator.

    Returns
    -------
    nn.Module
        Separator.
    """
    return DemucsLike(hybrid=False)


def build_hdemucs() -> nn.Module:
    """Build an HDemucs-like hybrid separator.

    Returns
    -------
    nn.Module
        Separator.
    """
    return DemucsLike(hybrid=True)


def build_encodec() -> nn.Module:
    """Build an EnCodec-like autoencoder.

    Returns
    -------
    nn.Module
        Audio codec.
    """
    return EnCodecLike()


def build_squim_objective() -> nn.Module:
    """Build SQUIM objective estimator.

    Returns
    -------
    nn.Module
        Quality estimator.
    """
    return SQUIMObjectiveLike()


def build_wav2vec2_xlsr() -> nn.Module:
    """Build wav2vec2/XLSR-style encoder.

    Returns
    -------
    nn.Module
        Speech encoder.
    """
    return Wav2Vec2Like()
