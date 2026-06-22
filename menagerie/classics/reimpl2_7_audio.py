"""Compact neural-audio dependency-gated reimplementations for shard 7.

Sources checked: SNAC multi-scale neural audio codec (multi-resolution RVQ),
SpeechTokenizer ICLR 2024 (encoder-decoder with hierarchical RVQ whose first
layer is semantic), StableCodec neural codec descriptions (ConvNeXt/RVQ codec),
and WaveGrad (conditional diffusion vocoder refining waveform from mel).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ResidualVectorQuantizer(nn.Module):
    """Differentiable compact residual vector quantizer."""

    def __init__(self, channels: int, codebooks: int = 3, entries: int = 16) -> None:
        """Initialize residual codebooks.

        Parameters
        ----------
        channels:
            Latent channel dimension.
        codebooks:
            Number of residual codebooks.
        entries:
            Number of vectors per codebook.
        """
        super().__init__()
        self.codebooks = nn.Parameter(torch.randn(codebooks, entries, channels) * 0.05)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize a latent sequence with residual codebooks.

        Parameters
        ----------
        z:
            Latent sequence ``(batch, channels, time)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Quantized latent and stacked soft assignment probabilities.
        """
        residual = z.transpose(1, 2)
        quantized = torch.zeros_like(residual)
        probs_out = []
        for codebook in self.codebooks:
            dist = (residual[:, :, None, :] - codebook[None, None, :, :]).square().sum(dim=-1)
            probs = torch.softmax(-dist, dim=-1)
            chosen = torch.einsum("bte,ec->btc", probs, codebook)
            quantized = quantized + chosen
            residual = residual - chosen
            probs_out.append(probs)
        return quantized.transpose(1, 2), torch.stack(probs_out, dim=1)


class ConvCodecEncoder(nn.Module):
    """Small causal-ish convolutional audio encoder."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize strided waveform encoder.

        Parameters
        ----------
        channels:
            Hidden channel count.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, 7, padding=3),
            nn.GELU(),
            nn.Conv1d(channels, channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, 4, stride=2, padding=1),
            nn.GELU(),
        )

    def forward(self, audio: Tensor) -> Tensor:
        """Encode waveform samples.

        Parameters
        ----------
        audio:
            Waveform tensor.

        Returns
        -------
        Tensor
            Latent sequence.
        """
        return self.net(audio)


class ConvCodecDecoder(nn.Module):
    """Small transposed-convolution audio decoder."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize waveform decoder.

        Parameters
        ----------
        channels:
            Hidden channel count.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, 1, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent sequence to waveform.

        Parameters
        ----------
        z:
            Quantized latent sequence.

        Returns
        -------
        Tensor
            Reconstructed waveform.
        """
        return self.net(z)


class SNACCodec(nn.Module):
    """SNAC-style multi-scale RVQ neural audio codec."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize compact SNAC.

        Parameters
        ----------
        channels:
            Latent channel count.
        """
        super().__init__()
        self.encoder = ConvCodecEncoder(channels)
        self.coarse = ResidualVectorQuantizer(channels, codebooks=1)
        self.mid = ResidualVectorQuantizer(channels, codebooks=1)
        self.fine = ResidualVectorQuantizer(channels, codebooks=1)
        self.decoder = ConvCodecDecoder(channels)

    def forward(self, audio: Tensor) -> Tensor:
        """Encode, multi-scale quantize, upsample residuals, and decode.

        Parameters
        ----------
        audio:
            Waveform tensor ``(batch, 1, samples)``.

        Returns
        -------
        Tensor
            Reconstructed waveform.
        """
        z = self.encoder(audio)
        coarse_in = F.avg_pool1d(z, kernel_size=4, stride=4)
        q_coarse, _ = self.coarse(coarse_in)
        coarse_up = F.interpolate(q_coarse, size=z.shape[-1], mode="nearest")
        mid_in = F.avg_pool1d(z - coarse_up, kernel_size=2, stride=2)
        q_mid, _ = self.mid(mid_in)
        mid_up = F.interpolate(q_mid, size=z.shape[-1], mode="nearest")
        q_fine, _ = self.fine(z - coarse_up - mid_up)
        return self.decoder(coarse_up + mid_up + q_fine)


class SpeechTokenizerCodec(nn.Module):
    """SpeechTokenizer-style semantic-acoustic hierarchical RVQ codec."""

    def __init__(self, channels: int = 32, semantic_dim: int = 12) -> None:
        """Initialize compact SpeechTokenizer.

        Parameters
        ----------
        channels:
            Latent channel count.
        semantic_dim:
            Semantic teacher feature dimension.
        """
        super().__init__()
        self.encoder = ConvCodecEncoder(channels)
        self.rvq = ResidualVectorQuantizer(channels, codebooks=4)
        self.semantic_head = nn.Conv1d(channels, semantic_dim, 1)
        self.decoder = ConvCodecDecoder(channels)

    def forward(self, audio: Tensor) -> Tensor:
        """Return waveform reconstruction and semantic first-layer features.

        Parameters
        ----------
        audio:
            Waveform tensor.

        Returns
        -------
        Tensor
            Concatenated reconstruction summary and semantic-token summary.
        """
        z = self.encoder(audio)
        q, probs = self.rvq(z)
        recon = self.decoder(q)
        semantic = self.semantic_head(q)
        first_layer_confidence = probs[:, 0].amax(dim=-1).mean(dim=1, keepdim=True)
        semantic_summary = semantic.mean(dim=-1)
        return torch.cat((recon.mean(dim=-1), semantic_summary, first_layer_confidence), dim=-1)


class ConvNeXt1DBlock(nn.Module):
    """ConvNeXt-style 1D residual block used by StableCodec."""

    def __init__(self, channels: int) -> None:
        """Initialize depthwise and pointwise convolutions.

        Parameters
        ----------
        channels:
            Feature channels.
        """
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pw1 = nn.Linear(channels, channels * 4)
        self.pw2 = nn.Linear(channels * 4, channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply ConvNeXt residual mixing.

        Parameters
        ----------
        x:
            Sequence tensor.

        Returns
        -------
        Tensor
            Updated sequence tensor.
        """
        y = self.dw(x).transpose(1, 2)
        y = self.pw2(F.gelu(self.pw1(self.norm(y)))).transpose(1, 2)
        return x + y


class FiniteScalarQuantizer(nn.Module):
    """Finite scalar quantizer used by compact StableCodec."""

    def __init__(self, levels: int = 7) -> None:
        """Initialize scalar quantization levels.

        Parameters
        ----------
        levels:
            Number of finite scalar levels.
        """

        super().__init__()
        self.register_buffer("values", torch.linspace(-1.0, 1.0, levels))

    def forward(self, z: Tensor) -> Tensor:
        """Quantize latents independently to finite scalar levels."""

        dist = (z.unsqueeze(-1) - self.values.view(1, 1, 1, -1)).square()
        probs = torch.softmax(-20.0 * dist, dim=-1)
        return (probs * self.values.view(1, 1, 1, -1)).sum(dim=-1)


class StableCodec(nn.Module):
    """StableCodec-style Transformer codec with FSQ bottleneck."""

    def __init__(self, channels: int = 32) -> None:
        """Initialize compact StableCodec.

        Parameters
        ----------
        channels:
            Latent channel count.
        """
        super().__init__()
        self.pre = ConvCodecEncoder(channels)
        layer = nn.TransformerEncoderLayer(
            channels, 4, dim_feedforward=channels * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.fsq = FiniteScalarQuantizer()
        self.post = ConvCodecDecoder(channels)

    def forward(self, audio: Tensor) -> Tensor:
        """Compress and reconstruct audio with Transformer and FSQ codec.

        Parameters
        ----------
        audio:
            Waveform tensor.

        Returns
        -------
        Tensor
            Reconstructed waveform.
        """
        z = self.pre(audio).transpose(1, 2)
        z = self.transformer(z).transpose(1, 2)
        q = self.fsq(torch.tanh(z))
        return self.post(q)


class WaveGradDenoiser(nn.Module):
    """WaveGrad-style conditional waveform score network."""

    def __init__(self, mel_bins: int = 16, channels: int = 32) -> None:
        """Initialize compact WaveGrad denoiser.

        Parameters
        ----------
        mel_bins:
            Number of mel conditioning bins.
        channels:
            Hidden channel count.
        """
        super().__init__()
        self.noise_embed = nn.Linear(1, channels)
        self.mel_proj = nn.Conv1d(mel_bins, channels, 3, padding=1)
        self.down = nn.Conv1d(1, channels, 5, padding=2)
        self.res1 = ConvNeXt1DBlock(channels)
        self.res2 = ConvNeXt1DBlock(channels)
        self.up = nn.Conv1d(channels, 1, 5, padding=2)

    def forward(self, noisy_audio: Tensor, mel: Tensor, noise_level: Tensor) -> Tensor:
        """Estimate denoising gradient conditioned on mel and noise level.

        Parameters
        ----------
        noisy_audio:
            Current noisy waveform.
        mel:
            Log-mel spectrogram conditioning.
        noise_level:
            Diffusion noise-level scalar per batch item.

        Returns
        -------
        Tensor
            Estimated score/gradient waveform.
        """
        mel_up = F.interpolate(
            self.mel_proj(mel), size=noisy_audio.shape[-1], mode="linear", align_corners=False
        )
        emb = self.noise_embed(noise_level).unsqueeze(-1)
        h = self.down(noisy_audio) + mel_up + emb
        h = self.res2(self.res1(F.gelu(h)))
        return self.up(h)


def build_snac() -> nn.Module:
    """Build compact SNAC."""
    return SNACCodec()


def build_speechtokenizer() -> nn.Module:
    """Build compact SpeechTokenizer."""
    return SpeechTokenizerCodec()


def build_stablecodec() -> nn.Module:
    """Build compact StableCodec."""
    return StableCodec()


def build_wavegrad_lmnt() -> nn.Module:
    """Build compact WaveGrad."""
    return WaveGradDenoiser()


def example_audio() -> Tensor:
    """Return compact waveform input."""
    return torch.randn(1, 1, 256)


def example_wavegrad() -> tuple[Tensor, Tensor, Tensor]:
    """Return noisy waveform, mel conditioning, and noise level."""
    return torch.randn(1, 1, 256), torch.randn(1, 16, 32), torch.tensor([[0.3]])


MENAGERIE_ENTRIES = [
    ("SNAC", "build_snac", "example_audio", "2024", "DC"),
    ("SpeechTokenizer", "build_speechtokenizer", "example_audio", "2024", "DC"),
    ("StableCodec", "build_stablecodec", "example_audio", "2024", "DC"),
    ("WaveGrad_lmnt", "build_wavegrad_lmnt", "example_wavegrad", "2020", "DC"),
]
