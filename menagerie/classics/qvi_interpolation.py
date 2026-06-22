"""QVI: Quadratic Video Interpolation with acceleration-aware flow.

Liu et al. (2020), "Video Frame Interpolation via Deformable Separable Convolution".
"Quadratic Video Interpolation" (NeurIPS 2019), arXiv:1911.00627.
Source: https://github.com/xuxy09/QVI

Distinctive primitives:
  1. QUADRATIC MOTION MODEL: given 4 frames (I_{t-1}, I_t, I_{t+1}, I_{t+2}) and
     their pairwise optical flows, estimate a QUADRATIC (acceleration-aware) flow at
     fractional time tau in [0, 1]:
       v(tau) = (1 - tau) * F_{t->t+1} + tau * F_{t+1->t+2}  (linear component)
              + tau * (1 - tau) * a                             (acceleration term)
     where a = F_{t->t+1} - F_{t+1->t+2}   (flow "second difference").
     This 4-frame quadratic model captures acceleration, unlike linear 2-frame models.
  2. FLOW REVERSAL: convert forward flows to reversed flows for backward warping via
     a soft-argmax flow reversal layer.
  3. SYNTHESIS NET: takes warped frames + flows + masks -> synthesised interpolated frame.

For the atlas: we use pre-computed (random) flows instead of PWC-Net, and reproduce
the quadratic flow combination + bilinear warp + synthesis net.
Compact config: C=16, H=W=8, 4 input frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# Bilinear warp (shared with RIFE concept)
# ==============================================================


def warp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Bilinear backward warp. img: (B, C, H, W), flow: (B, 2, H, W)."""
    B, C, H, W = img.shape
    gy, gx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=img.device),
        torch.arange(W, dtype=torch.float32, device=img.device),
        indexing="ij",
    )
    grid = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    new_grid = grid + flow
    new_grid[:, 0] = 2.0 * new_grid[:, 0] / max(W - 1, 1) - 1.0
    new_grid[:, 1] = 2.0 * new_grid[:, 1] / max(H - 1, 1) - 1.0
    new_grid = new_grid.permute(0, 2, 3, 1)
    return F.grid_sample(img, new_grid, mode="bilinear", align_corners=True, padding_mode="border")


# ==============================================================
# Flow reversal (simplified soft-argmax version)
# ==============================================================


class FlowReversal(nn.Module):
    """Simplified flow reversal: invert flow via soft-argmax weighting.

    For the atlas: use the negative + a small learned correction.
    Full QVI uses a differentiable flow-reversal with a softmax splatting.
    """

    def __init__(self, c: int = 16) -> None:
        super().__init__()
        # Correction network: flow -> correction
        self.corr = nn.Sequential(
            nn.Conv2d(2, c, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(c, 2, 3, padding=1),
        )

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """Approximate reversed flow: -flow + learned correction."""
        return -flow + self.corr(flow)


# ==============================================================
# QVI quadratic flow combiner
# ==============================================================


class QuadraticFlowCombiner(nn.Module):
    """Combine two flows (F01, F12) with quadratic motion model at time tau.

    v(tau) = (1 - tau) * F01 + tau * F12 + tau * (1 - tau) * (F01 - F12)
           = F01 - tau * F01 + tau * F12 + tau * F01 - tau^2 * F01 - tau * F12 + tau^2 * F12
    Simplification: v(tau) = (1 - tau + tau*(1-tau)) * F01 + (tau - tau*(1-tau)) * F12
    """

    def forward(self, F01: torch.Tensor, F12: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        """F01, F12: (B, 2, H, W), tau: scalar -> (B, 2, H, W) combined flow."""
        acc = F01 - F12  # acceleration term (second difference)
        v_tau = (1.0 - tau) * F01 + tau * F12 + tau * (1.0 - tau) * acc
        return v_tau


# ==============================================================
# Synthesis net
# ==============================================================


class SynthesisNet(nn.Module):
    """Synthesis: warped frames + flows + mask -> interpolated frame."""

    def __init__(self, c: int = 16) -> None:
        super().__init__()
        # Input: warp0(3) + warp1(3) + warp2(3) + warp3(3) + flow_fwd(2) + flow_bwd(2) + mask(1) = 17
        self.net = nn.Sequential(
            nn.Conv2d(17, c, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(c, c, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(c, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        f_fwd: torch.Tensor,
        f_bwd: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([w0, w1, w2, w3, f_fwd, f_bwd, mask], dim=1)
        return self.net(x)


# ==============================================================
# QVI full model
# ==============================================================


class QVIModel(nn.Module):
    """QVI: 4-frame quadratic motion model + warp + synthesis.

    Input: (B, 4*3 + 3*2, H, W) = 4 RGB frames (12ch) + 3 pairwise flows (6ch).
      Flows: F_{0->1} (2ch), F_{1->2} (2ch), F_{2->3} (2ch)
    Output: (B, 3, H, W) interpolated middle frame.
    Packed as (B, 18, H, W).
    """

    def __init__(self, c: int = 16) -> None:
        super().__init__()
        self.quad_combine = QuadraticFlowCombiner()
        self.flow_reversal = FlowReversal(c)
        # Mask prediction network: given flows -> soft mask
        self.mask_net = nn.Sequential(
            nn.Conv2d(4, c, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(c, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.synthesis = SynthesisNet(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 18, H, W) -- 4 frames (12ch) + 3 flows (6ch)."""
        I0 = x[:, 0:3]
        I1 = x[:, 3:6]
        I2 = x[:, 6:9]
        I3 = x[:, 9:12]
        F01 = x[:, 12:14]
        F12 = x[:, 14:16]
        F23 = x[:, 16:18]

        # Quadratic flow at tau=0.5: between I1 and I2
        f_fwd = self.quad_combine(F01, F12, tau=0.5)  # forward flow -> midpoint
        f_bwd = self.quad_combine(F23, -F12, tau=0.5)  # backward flow from I2

        # Reverse for backward warp
        f_fwd_rev = self.flow_reversal(f_fwd)
        f_bwd_rev = self.flow_reversal(f_bwd)

        # Warp 4 frames
        w0 = warp(I0, f_fwd)
        w1 = warp(I1, f_fwd_rev)
        w2 = warp(I2, f_bwd_rev)
        w3 = warp(I3, f_bwd)

        # Mask
        mask_in = torch.cat([f_fwd[:, :1], f_fwd[:, 1:], f_bwd[:, :1], f_bwd[:, 1:]], dim=1)
        mask = self.mask_net(mask_in)

        # Synthesise
        return self.synthesis(w0, w1, w2, w3, f_fwd, f_bwd, mask)


def build_qvi_interpolation() -> nn.Module:
    return QVIModel(c=16).eval()


def example_input() -> torch.Tensor:
    """(1, 18, 8, 8) -- 4 frames (12ch) + 3 flows (6ch), 8x8."""
    return torch.randn(1, 18, 8, 8)


MENAGERIE_ENTRIES = [
    (
        "QVI (Quadratic Video Interpolation: 4-frame acceleration-aware quadratic flow + warp + synthesis)",
        "build_qvi_interpolation",
        "example_input",
        "2020",
        "DC",
    ),
]
