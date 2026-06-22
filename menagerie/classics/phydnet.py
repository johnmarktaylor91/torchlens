"""PhyDNet: Disentangling Physical Dynamics and Unknown Residuals.

Le Guen & Thome, CVPR 2020.
Paper: https://arxiv.org/abs/2003.01460
Source: https://github.com/vincent-leguen/PhyDNet

PhyDNet disentangles video/spatiotemporal prediction into:
  1. PhyCell: a PHYSICALLY-CONSTRAINED recurrent cell that models known PDE
     dynamics via a learned moment-based formulation (PDE moment bank).
  2. ConvLSTM cell: learns residual dynamics that the physical model can't capture.

The two branches run in parallel and their outputs are combined:
  total_prediction = physical_prediction + residual_prediction

PhyCell's key primitive:
  - Maintains a "moment bank" of M spatial derivative moments phi_k (convolutions)
    applied to the hidden state H. These approximate partial derivatives:
    phi_1 = df/dx, phi_2 = d^2f/dx^2, ... up to moment M.
  - The hidden state evolves via:
    H_{t+1} = H_t + sum_k a_k * phi_k(H_t)
    where a_k are learned (spatially varying) coefficients representing the PDE.
  - This is the "PDE moment network": constrained to produce dynamics consistent
    with a linear PDE on the hidden field.
  - A correction term (from encoder-decoder with the input) is subtracted to
    compensate for moment-basis truncation error.

Architecture for T-step prediction: at each step t,
  - Input frame x_t -> encoder -> feature map
  - PhyCell updates H with PDE moments + correction
  - ConvLSTM updates (H_lstm, C_lstm) with standard gated dynamics
  - Combined hidden -> decoder -> predicted frame x_{t+1}

Simplifications: 2 input frames, predict 1 step, spatial 8x8,
hidden_dim=32, M=4 moments (2nd-order PDE basis), single ConvLSTM layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhyCellMoments(nn.Module):
    """PDE moment bank: M learnable convolutional filters approximating spatial derivatives.

    phi_k is a 3x3 convolution initialized to approximate the k-th spatial moment.
    The PhyCell applies sum_k a_k * phi_k(H) as the PDE right-hand side.
    """

    def __init__(self, hidden_dim: int, M: int = 4, kernel_size: int = 3) -> None:
        super().__init__()
        self.M = M
        self.hidden_dim = hidden_dim
        # M convolutional filters (the PDE moment bank) -- shared across channels
        # Each filter is applied independently to each channel
        self.moment_filters = nn.ModuleList(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=hidden_dim,
                    bias=False,
                )
                for _ in range(M)
            ]
        )
        # Learned PDE coefficients: spatial-map (B, M, H, W) comes from input
        # Here we use a fixed spatially-shared 1x1 conv to produce the M coefficients
        # from the hidden state itself (feedback)
        self.coeff_net = nn.Conv2d(hidden_dim, M, kernel_size=1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (B, hidden_dim, h, w)
        # Compute coefficients: (B, M, h, w)
        a = self.coeff_net(H)  # (B, M, h, w)

        # Compute PDE right-hand side: sum_k a_k * phi_k(H)
        rhs = torch.zeros_like(H)
        for k, phi_k in enumerate(self.moment_filters):
            phi_H = phi_k(H)  # (B, hidden_dim, h, w)
            a_k = a[:, k : k + 1, :, :]  # (B, 1, h, w)
            rhs = rhs + a_k * phi_H
        return rhs


class PhyCell(nn.Module):
    """Physically-constrained recurrent cell (PDE moment cell).

    H_{t+1} = H_t + dt * sum_k a_k * phi_k(H_t) + correction(x_t, H_t)
    """

    def __init__(self, input_dim: int, hidden_dim: int, M: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # PDE moment bank
        self.moments = PhyCellMoments(hidden_dim, M)

        # Correction network: takes input + hidden, outputs correction to subtract
        self.correction = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=1)

        # Gate: blend between PDE prediction and correction
        self.gate = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim, h, w), H: (B, hidden_dim, h, w)
        # PDE step: H_phys = H + dt * moments(H)
        H_phys = H + 0.1 * self.moments(H)

        # Correction from input
        xh = torch.cat([x, H], dim=1)
        corr = self.correction(xh)
        g = self.gate(xh)

        # Updated hidden: gated blend with correction
        H_new = H_phys - g * corr
        return H_new


class ConvLSTMCell(nn.Module):
    """Standard ConvLSTM cell for residual dynamics."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        p = kernel_size // 2
        # All 4 gates in one conv
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim, kernel_size=kernel_size, padding=p
        )

    def forward(self, x: torch.Tensor, H: torch.Tensor, C: torch.Tensor) -> tuple:
        xh = torch.cat([x, H], dim=1)
        gates = self.conv(xh)
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        C_new = f * C + i * g
        H_new = o * torch.tanh(C_new)
        return H_new, C_new


class PhyDNet(nn.Module):
    """PhyDNet: disentangled physical + residual spatiotemporal prediction.

    Takes T input frames, predicts T_pred next frames.
    Two branches run in parallel: PhyCell (physical) + ConvLSTM (residual).
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 32,
        M: int = 4,
        H: int = 8,
        W: int = 8,
        T_in: int = 2,
        T_pred: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.H, self.W = H, W
        self.T_in = T_in
        self.T_pred = T_pred

        # Encoder: frame -> feature map
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )

        # PhyCell branch
        self.phy_cell = PhyCell(hidden_dim, hidden_dim, M)

        # ConvLSTM branch (residual)
        self.conv_lstm = ConvLSTMCell(hidden_dim, hidden_dim)

        # Decoder: combined hidden -> predicted frame
        # Combine physical + residual hidden states
        self.decoder = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_in, C, H, W) -- input sequence
        B, T, C, H, W = x.shape

        # Initialize hidden states
        H_phy = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        H_res = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
        C_res = torch.zeros(B, self.hidden_dim, H, W, device=x.device)

        # Encoding phase: process T_in input frames
        for t in range(T):
            feat = self.encoder(x[:, t])  # (B, hidden_dim, H, W)
            H_phy = self.phy_cell(feat, H_phy)
            H_res, C_res = self.conv_lstm(feat, H_res, C_res)

        # Prediction phase: generate T_pred future frames
        preds = []
        for _ in range(self.T_pred):
            combined = torch.cat([H_phy, H_res], dim=1)  # (B, 2*hidden_dim, H, W)
            pred = self.decoder(combined)  # (B, C, H, W)
            preds.append(pred)

            # Feed prediction back (closed-loop)
            feat = self.encoder(pred)
            H_phy = self.phy_cell(feat, H_phy)
            H_res, C_res = self.conv_lstm(feat, H_res, C_res)

        return torch.stack(preds, dim=1)  # (B, T_pred, C, H, W)


def build_phydnet() -> nn.Module:
    return PhyDNet(in_channels=1, hidden_dim=32, M=4, H=8, W=8, T_in=2, T_pred=1)


def example_input_phydnet() -> torch.Tensor:
    # (B=1, T_in=2, C=1, H=8, W=8): 2 input frames at 8x8
    return torch.randn(1, 2, 1, 8, 8)


MENAGERIE_ENTRIES = [
    ("PhyDNet", "build_phydnet", "example_input_phydnet", "2020", "DC"),
]
