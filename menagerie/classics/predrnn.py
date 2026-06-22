"""PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning.

Wang et al., NeurIPS 2017 (PredRNN) + NeurIPS 2021 (PredRNN-V2).
Paper: https://arxiv.org/abs/2103.09504 (PredRNN-V2 / PredRNN++)
Source: https://github.com/thuml/predrnn-pytorch

PredRNN introduces the SPATIOTEMPORAL LSTM (ST-LSTM) cell with a distinctive
ZIGZAG MEMORY FLOW across layers and time steps.

Standard stacked RNN: memory flows independently within each layer (horizontal).
PredRNN: introduces a SECOND MEMORY STATE M (spatiotemporal memory) that flows
in a ZIGZAG pattern:
  - Within a time step: M flows DOWN through layers (layer 1 -> 2 -> ... -> L)
  - Across time steps: M flows from the LAST layer back to the FIRST layer
    of the NEXT time step
  This creates a "zigzag" memory pathway that connects all layers at all times.

ST-LSTM cell equations:
  i, f, g, o = standard LSTM gates from (X, H)
  i', f', g' = spatial memory gates from (X, H, M)  [the additional gates]
  C_t = f * C_{t-1} + i * g
  M_t = f' * M_{t-1} + i' * g'                      [spatial memory update]
  H_t = o * tanh(W_11 * C_t + W_12 * M_t)           [fuse both memories]

The M state flows across layers (unlike C which is layer-local).

Unrolled: 3 time steps, 2 stacked ST-LSTM layers; spatial 8x8, hidden 32 channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class STLSTMCell(nn.Module):
    """Spatiotemporal LSTM (ST-LSTM) cell -- the PredRNN primitive.

    Maintains two memory states:
      - C: temporal memory (flows horizontally within a layer across time)
      - M: spatiotemporal memory (flows diagonally in zigzag across layers+time)
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.h_ch = hidden_channels
        p = kernel_size // 2

        # Standard LSTM gates from (X, H) -> C update
        self.conv_XH = nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels, kernel_size=kernel_size, padding=p
        )
        # Spatial memory gates from (X, H, M) -> M update + H update
        self.conv_XHM = nn.Conv2d(
            in_channels + 2 * hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=p,
        )
        # Fuse C and M for H output
        self.conv_fuse = nn.Conv2d(2 * hidden_channels, hidden_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,  # (B, in_ch, H, W)
        H: torch.Tensor,  # (B, h_ch, H, W) -- temporal hidden
        C: torch.Tensor,  # (B, h_ch, H, W) -- temporal memory
        M: torch.Tensor,  # (B, h_ch, H, W) -- spatiotemporal memory (zigzag)
    ) -> tuple:
        # -- Temporal gates (from X, H) --
        xh = torch.cat([x, H], dim=1)
        temporal = self.conv_XH(xh)
        i, f, g, o = temporal.chunk(4, dim=1)
        i, f = torch.sigmoid(i), torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        # Temporal memory update
        C_new = f * C + i * g

        # -- Spatiotemporal gates (from X, H, M) --
        xhm = torch.cat([x, H, M], dim=1)
        spatial = self.conv_XHM(xhm)
        ip, fp, gp, _ = spatial.chunk(4, dim=1)
        ip, fp = torch.sigmoid(ip), torch.sigmoid(fp)
        gp = torch.tanh(gp)
        # Spatiotemporal memory update (zigzag flow)
        M_new = fp * M + ip * gp

        # Fuse C and M for output hidden state
        H_new = o * torch.tanh(self.conv_fuse(torch.cat([C_new, M_new], dim=1)))
        return H_new, C_new, M_new


class PredRNN(nn.Module):
    """PredRNN with zigzag memory flow across 2 ST-LSTM layers.

    Takes T_in input frames, predicts T_pred future frames.
    Uses the zigzag M memory flow: within each time step, M flows down through
    layers; across time steps, M from the last layer feeds into the first layer
    of the next step.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        n_layers: int = 2,
        kernel_size: int = 5,
        H: int = 8,
        W: int = 8,
        T_in: int = 3,
        T_pred: int = 1,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.h_ch = hidden_channels
        self.H, self.W = H, W
        self.T_in = T_in
        self.T_pred = T_pred

        # ST-LSTM cells for each layer
        cells = []
        for l in range(n_layers):
            in_ch = in_channels if l == 0 else hidden_channels
            cells.append(STLSTMCell(in_ch, hidden_channels, kernel_size))
        self.cells = nn.ModuleList(cells)

        # Output: last hidden -> predicted frame
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1),
        )

    def _init_states(self, B: int, device: torch.device) -> tuple:
        """Initialize H, C per layer and M (single zigzag M)."""
        H_list = [
            torch.zeros(B, self.h_ch, self.H, self.W, device=device) for _ in range(self.n_layers)
        ]
        C_list = [
            torch.zeros(B, self.h_ch, self.H, self.W, device=device) for _ in range(self.n_layers)
        ]
        # Single M state that zigzags (same spatial size)
        M = torch.zeros(B, self.h_ch, self.H, self.W, device=device)
        return H_list, C_list, M

    def _step(self, x: torch.Tensor, H_list: list, C_list: list, M: torch.Tensor) -> tuple:
        """One time step: process x through all layers with zigzag M."""
        inp = x
        new_H_list = []
        new_C_list = []

        for l, cell in enumerate(self.cells):
            H_new, C_new, M = cell(inp, H_list[l], C_list[l], M)
            new_H_list.append(H_new)
            new_C_list.append(C_new)
            inp = H_new  # feed to next layer

        # M after last layer flows to first layer of next time step (zigzag)
        return new_H_list, new_C_list, M

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_in, C, H, W)
        B, T, C, H, W = x.shape
        H_list, C_list, M = self._init_states(B, x.device)

        # Encoding phase: process T_in input frames
        for t in range(T):
            H_list, C_list, M = self._step(x[:, t], H_list, C_list, M)

        # Prediction phase
        preds = []
        last_pred = x[:, -1]  # start from last input frame
        for _ in range(self.T_pred):
            H_list, C_list, M = self._step(last_pred, H_list, C_list, M)
            pred = self.output_conv(H_list[-1])  # (B, C, H, W)
            preds.append(pred)
            last_pred = pred

        return torch.stack(preds, dim=1)  # (B, T_pred, C, H, W)


def build_predrnn() -> nn.Module:
    return PredRNN(
        in_channels=1, hidden_channels=32, n_layers=2, kernel_size=5, H=8, W=8, T_in=3, T_pred=1
    )


def example_input_predrnn() -> torch.Tensor:
    # (B=1, T_in=3, C=1, H=8, W=8): 3 input frames at 8x8
    return torch.randn(1, 3, 1, 8, 8)


MENAGERIE_ENTRIES = [
    ("PredRNN (Spatiotemporal LSTM)", "build_predrnn", "example_input_predrnn", "2017", "DC"),
]
