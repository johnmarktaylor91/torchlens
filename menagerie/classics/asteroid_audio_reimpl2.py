"""Asteroid source-separation architectures in compact pure PyTorch form.

Sources: Asteroid model docs for DCUNet/DCCRNet, DPRNN-TasNet, DPTNet, and
SuDORMRF variants.

These reconstructions keep the architectural primitives that make the original
models distinct: encoder-masker-decoder separation, complex-style U-Net masks,
dual-path recurrent/Transformer chunk processing, and multi-resolution U-blocks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSeparatorBase(nn.Module):
    """Time-domain encoder-masker-decoder source-separation scaffold."""

    def __init__(self, channels: int = 32, n_src: int = 2) -> None:
        """Initialize the shared encoder and decoder.

        Parameters
        ----------
        channels:
            Latent channel count.
        n_src:
            Number of output sources.
        """

        super().__init__()
        self.n_src = n_src
        self.encoder = nn.Conv1d(1, channels, kernel_size=16, stride=8, padding=4)
        self.decoder = nn.ConvTranspose1d(channels, 1, kernel_size=16, stride=8, padding=4)

    def _decode_masks(
        self, encoded: torch.Tensor, masks: torch.Tensor, length: int
    ) -> torch.Tensor:
        """Apply masks and decode separated waveforms.

        Parameters
        ----------
        encoded:
            Encoded mixture, shape ``(batch, channels, frames)``.
        masks:
            Source masks, shape ``(batch, n_src, channels, frames)``.
        length:
            Requested waveform length.

        Returns
        -------
        torch.Tensor
            Separated audio, shape ``(batch, n_src, length)``.
        """

        outs = []
        for src in range(self.n_src):
            outs.append(self.decoder(encoded * masks[:, src])[:, 0, :length])
        return torch.stack(outs, dim=1)


class DualPathRNNBlock(nn.Module):
    """DPRNN block alternating intra-chunk and inter-chunk recurrence."""

    def __init__(self, channels: int) -> None:
        """Initialize intra- and inter-chunk recurrent paths.

        Parameters
        ----------
        channels:
            Latent channel count.
        """

        super().__init__()
        self.intra = nn.GRU(channels, channels, batch_first=True, bidirectional=True)
        self.intra_proj = nn.Linear(channels * 2, channels)
        self.inter = nn.GRU(channels, channels, batch_first=True, bidirectional=True)
        self.inter_proj = nn.Linear(channels * 2, channels)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run dual-path recurrent updates over fixed chunks.

        Parameters
        ----------
        x:
            Encoded features, shape ``(batch, channels, frames)``.

        Returns
        -------
        torch.Tensor
            Updated encoded features.
        """

        bsz, channels, frames = x.shape
        chunk = 8
        pad = (chunk - frames % chunk) % chunk
        y = F.pad(x, (0, pad)).transpose(1, 2)
        y = y.reshape(bsz, -1, chunk, channels)
        intra, _ = self.intra(y.reshape(-1, chunk, channels))
        y = y + self.intra_proj(intra).reshape_as(y)
        inter_in = y.transpose(1, 2).reshape(bsz * chunk, -1, channels)
        inter, _ = self.inter(inter_in)
        y = y + self.inter_proj(inter).reshape(bsz, chunk, -1, channels).transpose(1, 2)
        y = y.reshape(bsz, -1, channels)[:, :frames].transpose(1, 2)
        return self.norm(x + y)


class DPRNNTasNet(ConvSeparatorBase):
    """Compact DPRNN-TasNet separator."""

    def __init__(self) -> None:
        """Initialize DPRNN-TasNet layers."""

        super().__init__()
        self.blocks = nn.ModuleList([DualPathRNNBlock(32) for _ in range(2)])
        self.mask = nn.Conv1d(32, 64, kernel_size=1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Separate a waveform using DPRNN mask estimation.

        Parameters
        ----------
        wav:
            Mixture waveform, shape ``(batch, time)``.

        Returns
        -------
        torch.Tensor
            Two separated waveforms.
        """

        length = wav.shape[-1]
        enc = F.relu(self.encoder(wav.unsqueeze(1)))
        y = enc
        for block in self.blocks:
            y = block(y)
        masks = torch.sigmoid(self.mask(y)).view(wav.shape[0], 2, 32, -1)
        return self._decode_masks(enc, masks, length)


class DualPathTransformerBlock(nn.Module):
    """DPTNet block with intra- and inter-chunk Transformer encoders."""

    def __init__(self, channels: int) -> None:
        """Initialize dual-path Transformer layers.

        Parameters
        ----------
        channels:
            Latent channel count.
        """

        super().__init__()
        layer = nn.TransformerEncoderLayer(channels, 4, dim_feedforward=64, batch_first=True)
        self.intra = nn.TransformerEncoder(layer, num_layers=1)
        self.inter = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply intra- and inter-chunk self-attention.

        Parameters
        ----------
        x:
            Encoded features, shape ``(batch, channels, frames)``.

        Returns
        -------
        torch.Tensor
            Updated features.
        """

        bsz, channels, frames = x.shape
        chunk = 8
        pad = (chunk - frames % chunk) % chunk
        y = F.pad(x, (0, pad)).transpose(1, 2).reshape(bsz, -1, chunk, channels)
        y = y + self.intra(y.reshape(-1, chunk, channels)).reshape_as(y)
        z = y.transpose(1, 2).reshape(bsz * chunk, -1, channels)
        y = y + self.inter(z).reshape(bsz, chunk, -1, channels).transpose(1, 2)
        return self.norm(x + y.reshape(bsz, -1, channels)[:, :frames].transpose(1, 2))


class DPTNet(ConvSeparatorBase):
    """Compact DPTNet separator."""

    def __init__(self) -> None:
        """Initialize DPTNet layers."""

        super().__init__()
        self.blocks = nn.ModuleList([DualPathTransformerBlock(32) for _ in range(2)])
        self.mask = nn.Conv1d(32, 64, kernel_size=1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Separate a waveform with dual-path Transformers.

        Parameters
        ----------
        wav:
            Mixture waveform, shape ``(batch, time)``.

        Returns
        -------
        torch.Tensor
            Two separated waveforms.
        """

        length = wav.shape[-1]
        enc = F.relu(self.encoder(wav.unsqueeze(1)))
        y = enc
        for block in self.blocks:
            y = block(y)
        masks = torch.sigmoid(self.mask(y)).view(wav.shape[0], 2, 32, -1)
        return self._decode_masks(enc, masks, length)


class ComplexUNetMasker(nn.Module):
    """Two-dimensional complex-style U-Net mask estimator."""

    def __init__(self, recurrent: bool = False) -> None:
        """Initialize encoder, bottleneck, and decoder layers.

        Parameters
        ----------
        recurrent:
            Whether to insert a recurrent bottleneck as in DCCRNet.
        """

        super().__init__()
        self.down1 = nn.Conv2d(2, 16, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.recurrent = recurrent
        self.rnn = nn.GRU(32, 32, batch_first=True) if recurrent else None
        self.up1 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(32, 2, 4, stride=2, padding=1)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Estimate a complex-like mask from framed audio.

        Parameters
        ----------
        spec:
            Two-channel spectrogram-like tensor.

        Returns
        -------
        torch.Tensor
            Mask tensor with the same shape as ``spec``.
        """

        d1 = F.leaky_relu(self.down1(spec), 0.2)
        d2 = F.leaky_relu(self.down2(d1), 0.2)
        if self.rnn is not None:
            bsz, ch, freq, time = d2.shape
            seq = d2.permute(0, 2, 3, 1).reshape(bsz * freq, time, ch)
            seq, _ = self.rnn(seq)
            d2 = seq.reshape(bsz, freq, time, ch).permute(0, 3, 1, 2)
        u1 = F.relu(self.up1(d2))
        u1 = torch.cat((u1[..., : d1.shape[-2], : d1.shape[-1]], d1), dim=1)
        return torch.tanh(self.up2(u1))[..., : spec.shape[-2], : spec.shape[-1]]


class DCUNetLike(nn.Module):
    """Deep Complex U-Net or recurrent DCCRNet-style enhancer."""

    def __init__(self, recurrent: bool = False) -> None:
        """Initialize the compact complex U-Net.

        Parameters
        ----------
        recurrent:
            Insert a GRU bottleneck for DCCRNet.
        """

        super().__init__()
        self.masker = ComplexUNetMasker(recurrent)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Enhance a waveform via framed two-channel masking.

        Parameters
        ----------
        wav:
            Mixture waveform, shape ``(batch, time)``.

        Returns
        -------
        torch.Tensor
            Enhanced waveform-like tensor.
        """

        bsz, length = wav.shape
        frames = wav.unfold(-1, 32, 16)
        real = frames
        imag = torch.diff(F.pad(frames, (1, 0)), dim=-1)
        spec = torch.stack((real, imag), dim=1)
        mask = self.masker(spec)
        enhanced = (spec * mask).sum(dim=1).mean(dim=-1)
        return F.interpolate(
            enhanced.unsqueeze(1), size=length, mode="linear", align_corners=False
        )[:, 0]


class UConvBlock(nn.Module):
    """SuDORMRF multi-resolution U-convolution block."""

    def __init__(self, channels: int, improved: bool = False) -> None:
        """Initialize depthwise multi-scale convolutions.

        Parameters
        ----------
        channels:
            Latent channel count.
        improved:
            Add dense residual projection for the improved variant.
        """

        super().__init__()
        self.in_proj = nn.Conv1d(channels, channels, 1)
        self.down = nn.Conv1d(channels, channels, 5, stride=2, padding=2, groups=channels)
        self.mid = nn.Conv1d(channels, channels, 5, padding=2, groups=channels)
        self.up = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1, groups=channels)
        self.out_proj = nn.Conv1d(channels * (2 if improved else 1), channels, 1)
        self.improved = improved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a multi-resolution residual update.

        Parameters
        ----------
        x:
            Encoded features.

        Returns
        -------
        torch.Tensor
            Updated features.
        """

        z = F.prelu(self.in_proj(x), torch.tensor(0.25, device=x.device))
        low = F.relu(self.down(z))
        low = F.relu(self.mid(low))
        up = self.up(low)[..., : x.shape[-1]]
        joined = torch.cat((up, z), dim=1) if self.improved else up
        return x + self.out_proj(joined)


class SuDORMRFNet(ConvSeparatorBase):
    """Compact SuDORMRF separator with optional improved dense U-blocks."""

    def __init__(self, improved: bool = False) -> None:
        """Initialize SuDORMRF layers.

        Parameters
        ----------
        improved:
            Use improved UConv blocks.
        """

        super().__init__()
        self.blocks = nn.ModuleList([UConvBlock(32, improved) for _ in range(3)])
        self.mask = nn.Conv1d(32, 64, 1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Separate a waveform with SuDORMRF-style U-blocks.

        Parameters
        ----------
        wav:
            Mixture waveform.

        Returns
        -------
        torch.Tensor
            Two separated waveforms.
        """

        length = wav.shape[-1]
        enc = F.relu(self.encoder(wav.unsqueeze(1)))
        y = enc
        for block in self.blocks:
            y = block(y)
        masks = torch.sigmoid(self.mask(y)).view(wav.shape[0], 2, 32, -1)
        return self._decode_masks(enc, masks, length)


def build_dprnntasnet() -> nn.Module:
    """Build compact Asteroid DPRNN-TasNet."""

    return DPRNNTasNet()


def build_dptnet() -> nn.Module:
    """Build compact Asteroid DPTNet."""

    return DPTNet()


def build_dcunet() -> nn.Module:
    """Build compact Asteroid DCUNet."""

    return DCUNetLike(recurrent=False)


def build_dccrnet() -> nn.Module:
    """Build compact Asteroid DCCRNet."""

    return DCUNetLike(recurrent=True)


def build_sudormrf() -> nn.Module:
    """Build compact Asteroid SuDORMRFNet."""

    return SuDORMRFNet(improved=False)


def build_sudormrf_improved() -> nn.Module:
    """Build compact Asteroid SuDORMRFImprovedNet."""

    return SuDORMRFNet(improved=True)


def example_waveform() -> torch.Tensor:
    """Return a short waveform batch."""

    return torch.randn(1, 512)


MENAGERIE_ENTRIES = [
    ("asteroid_DCCRNet", "build_dccrnet", "example_waveform", "2020", "audio"),
    ("asteroid_DPRNNTasNet", "build_dprnntasnet", "example_waveform", "2020", "audio"),
    ("asteroid_DPTNet", "build_dptnet", "example_waveform", "2020", "audio"),
    (
        "asteroid_SuDORMRFImprovedNet",
        "build_sudormrf_improved",
        "example_waveform",
        "2021",
        "audio",
    ),
    ("asteroid_SuDORMRFNet", "build_sudormrf", "example_waveform", "2020", "audio"),
    ("asteroid_DCUNet", "build_dcunet", "example_waveform", "2019", "audio"),
]
