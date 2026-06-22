"""MoCoGAN-HD: Large Motion in the Latent Space.

Tian et al. 2021.  arXiv:2101.04671.
Source: https://github.com/snap-research/MoCoGAN-HD

MoCoGAN-HD's distinctive primitive: an **RNN-based motion generator** that walks
the StyleGAN2 W latent space to produce a video.
  - A **content code** z_c (initial latent) is mapped through StyleGAN2's mapping
    network to an initial W vector.
  - A **motion RNN** (GRU) takes a per-frame motion code z_m (sampled i.i.d.) and
    the previous hidden state, and outputs a *latent residual* delta_w that is added
    to the initial W vector: w_t = w_0 + delta_w_t.
  - Each w_t is fed into a frozen StyleGAN2 synthesis network to generate frame t.
  - The result is a temporally coherent video where content is locked in W space
    and motion is a learned GRU trajectory through that space.

Here we reproduce the motion generator RNN:
  - Mapping network: z_c -> w_0  (initial W code).
  - GRU motion RNN: [z_m_t || h_{t-1}] -> delta_w_t + h_t.
  - Tiny synthesis network stub: w_t -> single conv -> frame (compact).
  - Wrapped to accept a single tensor (z_c || z_m_1 || ... || z_m_T) and return
    stacked frames (B, T, C, H, W) reshaped to (B*T, C, H, W).

Random init, CPU, small dims for compact tracing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MappingNetwork(nn.Module):
    def __init__(self, z_dim: int = 32, w_dim: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = z_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, w_dim), nn.LeakyReLU(0.2)]
            d = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MotionGRU(nn.Module):
    """GRU-based motion generator: per-frame motion noise -> latent residuals delta_w."""

    def __init__(self, z_m_dim: int = 16, w_dim: int = 64, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRUCell(z_m_dim, hidden)
        self.to_delta = nn.Linear(hidden, w_dim)

    def forward(self, z_m_seq: torch.Tensor, h0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z_m_seq: (B, T, z_m_dim) -- per-frame motion codes
        h0: (B, hidden) -- initial hidden state

        Returns:
            delta_w_seq: (B, T, w_dim) -- latent residuals
            h_T: (B, hidden) -- final hidden state
        """
        B, T, _ = z_m_seq.shape
        h = h0
        deltas = []
        for t in range(T):
            h = self.gru(z_m_seq[:, t], h)
            deltas.append(self.to_delta(h))
        return torch.stack(deltas, dim=1), h


class TinySynthesisNet(nn.Module):
    """Minimal synthesis stub: w -> 3-channel 16x16 frame (replaces full StyleGAN2 synthesis)."""

    def __init__(self, w_dim: int = 64, nf: int = 16) -> None:
        super().__init__()
        self.fc = nn.Linear(w_dim, nf * 4 * 4)
        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(nf, 3, 3, 1, 1),
            nn.Tanh(),
        )
        self.nf = nf

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        # w: (B*T, w_dim) -> frame: (B*T, 3, 8, 8)
        x = self.fc(w).view(-1, self.nf, 4, 4)
        return self.conv(x)


class MoCoGANHDRNNModule(nn.Module):
    """MoCoGAN-HD: content mapping + GRU motion RNN + frame synthesis.

    Input: flattened vector (z_c || z_m_1 || ... || z_m_T).
    Output: frames stacked as (B*T, 3, H, W).
    """

    def __init__(
        self,
        z_c_dim: int = 32,
        z_m_dim: int = 16,
        w_dim: int = 64,
        hidden: int = 64,
        n_frames: int = 4,
    ) -> None:
        super().__init__()
        self.z_c_dim = z_c_dim
        self.z_m_dim = z_m_dim
        self.n_frames = n_frames
        self.hidden = hidden
        self.mapping = MappingNetwork(z_c_dim, w_dim)
        self.motion_rnn = MotionGRU(z_m_dim, w_dim, hidden)
        self.synthesis = TinySynthesisNet(w_dim)
        # Initial hidden state from content code
        self.h_init = nn.Linear(w_dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, z_c_dim + n_frames * z_m_dim)
        B = x.shape[0]
        z_c = x[:, : self.z_c_dim]
        z_m_flat = x[:, self.z_c_dim :]  # (B, T * z_m_dim)
        z_m_seq = z_m_flat.view(B, self.n_frames, self.z_m_dim)

        # Content: initial W code
        w0 = self.mapping(z_c)  # (B, w_dim)
        # Initial hidden state
        h0 = torch.tanh(self.h_init(w0))  # (B, hidden)
        # Motion RNN: delta_w per frame
        delta_w, _ = self.motion_rnn(z_m_seq, h0)  # (B, T, w_dim)
        # w_t = w0 + delta_w_t for each frame
        w_t = w0.unsqueeze(1) + delta_w  # (B, T, w_dim)
        # Synthesize each frame
        w_flat = w_t.reshape(B * self.n_frames, -1)  # (B*T, w_dim)
        frames = self.synthesis(w_flat)  # (B*T, 3, H, W)
        return frames


def build_mocogan_hd_rnn_module() -> nn.Module:
    return MoCoGANHDRNNModule()


def example_input() -> torch.Tensor:
    # z_c (32) + 4 frames * z_m (16 each) = 96 total
    return torch.randn(1, 32 + 4 * 16)


MENAGERIE_ENTRIES = [
    (
        "MoCoGAN-HD RNN Module (GRU motion generator walking StyleGAN W latent space for video)",
        "build_mocogan_hd_rnn_module",
        "example_input",
        "2021",
        "DC",
    ),
]
