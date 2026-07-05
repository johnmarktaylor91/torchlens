# SOURCE: vendored from https://github.com/vivinousi/gw-detection-deep-learning @ master
"""Vendored AresGW challenge model staging module for TorchLens validation."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.fft import irfft, rfft, rfftfreq
from torch.nn import functional as F


class ResBlock(nn.Module):
    """Residual one-dimensional convolution block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize the residual block."""
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.x_transform = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.x_transform = nn.Identity()

        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual convolution block."""
        return F.relu(self.body(x) + self.x_transform(x))


class ResNet54(nn.Module):
    """Original 54-layer ResNet backbone from the AresGW challenge model."""

    def __init__(self) -> None:
        """Initialize the ResNet54 blocks and classification head."""
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlock(2, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            ResBlock(8, 16, stride=2),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 32, stride=2),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(16, 32, 64),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 2, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ResNet54 forward pass."""
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)


class DAINLayer(nn.Module):
    """Adaptive input normalization layer used by AresGW."""

    def __init__(
        self,
        mode: str = "full",
        mean_lr: float = 0.00001,
        gate_lr: float = 0.001,
        scale_lr: float = 0.00001,
        input_dim: int = 144,
    ) -> None:
        """Initialize the DAIN layer."""
        super().__init__()
        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        self.gating_layer = nn.Linear(input_dim, input_dim)
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a batch of time-series feature vectors."""
        avg = torch.mean(x, 2)
        adaptive_avg = self.mean_layer(avg)
        adaptive_avg = adaptive_avg.reshape(adaptive_avg.size(0), adaptive_avg.size(1), 1)
        x = x - adaptive_avg

        std = torch.mean(x**2, 2)
        std = torch.sqrt(std + self.eps)
        adaptive_std = self.scaling_layer(std)
        adaptive_std = torch.where(
            adaptive_std <= self.eps,
            torch.ones_like(adaptive_std),
            adaptive_std,
        )

        adaptive_std = adaptive_std.reshape(adaptive_std.size(0), adaptive_std.size(1), 1)
        x = x / adaptive_std

        avg = torch.mean(x, 2)
        gate = torch.sigmoid(self.gating_layer(avg))
        gate = gate.reshape(gate.size(0), gate.size(1), 1)
        return x * gate


def torch_inverse_spectrum_truncation(
    psd: torch.Tensor,
    max_filter_len: int,
    low_frequency_cutoff: float = 9.0,
    delta_f: float = 1.0,
    trunc_method: str = "hann",
) -> torch.Tensor:
    """Apply the inverse spectrum truncation routine from the source repo."""
    n_freq = (psd.size(1) - 1) * 2
    inv_asd = torch.zeros_like(psd)
    kmin = int(low_frequency_cutoff / delta_f)
    inv_asd[0, kmin : n_freq // 2] = (1.0 / psd[0, kmin : n_freq // 2]) ** 0.5

    q_value = irfft(inv_asd, n=n_freq, norm="forward")
    trunc_start = max_filter_len // 2
    trunc_end = n_freq - max_filter_len // 2
    if trunc_method == "hann":
        trunc_window = torch.hann_window(max_filter_len, dtype=torch.float64).to(psd.device)
        q_value[0, 0:trunc_start] *= trunc_window[-trunc_start:]
        q_value[0, trunc_end:] *= trunc_window[: max_filter_len // 2]
    if trunc_start < trunc_end:
        q_value[0, trunc_start:trunc_end] = 0
    psd_trunc = rfft(q_value, n=n_freq, norm="forward")
    psd_trunc *= psd_trunc.conj()

    psd = 1 / torch.abs(psd_trunc)
    return psd / 2


class Whiten(nn.Module):
    """FFT-based whitening module from the AresGW source."""

    def __init__(
        self,
        delta_t: float,
        low_frequency_cutoff: float = 15.0,
        m: float = 1.25,
        max_filter_len: float = 1.0,
        legacy: bool = True,
    ) -> None:
        """Initialize the whitening state."""
        super().__init__()
        self.max_filter_len = max_filter_len
        self.legacy = legacy
        self.delta_t = delta_t
        self.delta_f = 1 / m
        m /= delta_t
        self.m = int(m)
        self.d = int(m / 2)
        self.psd_est = None
        self.norm = nn.Parameter(torch.ones(2, 1281), requires_grad=False)
        self.frequencies = rfftfreq(self.m, d=self.delta_t)
        self.low_frequency_cutoff = low_frequency_cutoff

    def initialize(self, noise_t: torch.Tensor) -> None:
        """Estimate and store the whitening PSD from a noise tensor."""
        if noise_t.dim() == 2:
            noise_t = noise_t.unsqueeze(0).unsqueeze(2)
        n_channels = noise_t.size(1)
        psds = []
        for c_idx in range(n_channels):
            psd = self.estimate_psd(noise_t[:, c_idx, :, :].unsqueeze(1))
            psds.append(psd)
        self.psd_est = torch.cat(psds, dim=0)
        if self.legacy:
            self.norm.data = torch.sqrt(self.psd_est / self.delta_f)
            idx = int(self.low_frequency_cutoff / self.delta_f)
            self.norm.data[:, :idx] = self.norm.data[:, idx].view(-1, 1)
            self.norm.data[:, -1:] = self.norm.data[:, -2].view(-1, 1)
        else:
            self.norm.data = self.psd_est**0.5

    def estimate_psd(self, noise_t: torch.Tensor) -> torch.Tensor:
        """Estimate a PSD tensor from one channel of noise."""
        m_value = self.m
        d_value = self.d
        segments = F.unfold(noise_t, kernel_size=(1, m_value), stride=(1, d_value)).double()
        w_hann = torch.hann_window(segments.size(1), periodic=True, dtype=torch.float64).to(
            segments.device
        )
        segments_w = segments * w_hann.unsqueeze_(1)
        segments_fft = rfft(segments_w, dim=1, norm="forward")

        segments_sq_mag = torch.abs(segments_fft * segments_fft.conj())
        segments_sq_mag[0, 0, :] /= 2
        segments_sq_mag[0, -1, :] /= 2

        t_psd = torch.mean(segments_sq_mag, dim=2)
        t_psd *= 2 * self.delta_f * m_value / (w_hann * w_hann).sum()

        if t_psd.size(1) != 1281:
            t_psd = F.interpolate(t_psd.unsqueeze(1), 1281).squeeze(1)
            self.frequencies = rfftfreq(int(1.25 * 2048), d=self.delta_t)
        return torch_inverse_spectrum_truncation(
            t_psd,
            int(self.max_filter_len / self.delta_t),
            low_frequency_cutoff=15,
            delta_f=self.delta_f,
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Whiten a signal using the stored PSD estimate."""
        return self.whiten(signal)

    def whiten(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply frequency-domain whitening."""
        signal_f = rfft(signal.double(), dim=2, norm="forward")
        signal_t = irfft(signal_f / self.norm, norm="forward", n=signal.size(2))
        return signal_t.float()


class CropWhitenNet(nn.Module):
    """AresGW whitening, normalization, and classifier wrapper."""

    def __init__(
        self,
        net: nn.Module | None = None,
        norm: nn.Module | None = None,
        deploy: bool = False,
        m: float = 0.625,
        l_value: float = 0.5,
        f_value: float = 15.0,
    ) -> None:
        """Initialize the wrapper."""
        super().__init__()
        self.net = net
        self.norm = norm
        self.whiten = Whiten(
            1 / 2048, low_frequency_cutoff=f_value, m=m, max_filter_len=l_value, legacy=False
        )
        self.deploy = deploy
        self.step = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the original testing-time crop, whiten, normalize, and classify path."""
        segments_wh = []
        with torch.no_grad():
            c_value = x.size(1)
            for _i, sample in enumerate(x):
                self.whiten.initialize(sample)
                sample = sample.unsqueeze(0).unsqueeze(2)
                x_segments = F.unfold(sample, kernel_size=(1, 2560), stride=(1, 204)).contiguous()
                n_value = sample.size(0)
                l_value = x_segments.size(2)
                x_segments = (
                    x_segments.view(n_value, c_value, -1, l_value).permute(3, 1, 2, 0).squeeze_(3)
                )
                segments_wh.append(self.whiten(x_segments)[:, :, 256:-256])
            segments_wh_tensor = torch.cat(segments_wh)

        if self.norm is not None:
            segments_wh_tensor = self.norm(segments_wh_tensor)

        if self.net is None:
            return segments_wh_tensor
        return self.net(segments_wh_tensor)


def build_aresgw() -> CropWhitenNet:
    """Build the AresGW challenge model composition."""
    return CropWhitenNet(ResNet54(), DAINLayer(input_dim=2), m=1.25, l_value=0.25)


def example_input_aresgw() -> torch.Tensor:
    """Return a sample two-detector gravitational-wave segment."""
    return torch.randn(1, 2, 3072)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("AresGW", "build_aresgw", "example_input_aresgw", 2022, "CV3b"),
]
