"""PDE-Net and PDE-Refiner compact physics-operator models.

PDE-Net: Long et al. 2018, arXiv:1710.09668.
PDE-Refiner: Lippe et al. 2023, arXiv:2308.05732.

PDE-Net learns finite-difference differential operators and combines them in a
time-stepping update.  PDE-Refiner refines coarse PDE states with residual
convolutional denoising conditioned on the current state and a coarse prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PDENetStep(nn.Module):
    """Learned finite-difference PDE-Net step."""

    def __init__(self, channels: int = 2) -> None:
        """Initialize derivative filters and nonlinear update.

        Parameters
        ----------
        channels:
            Number of state channels.
        """

        super().__init__()
        self.dx = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), groups=channels, bias=False)
        self.dy = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), groups=channels, bias=False)
        self.lap = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.mix = nn.Conv2d(channels * 4, channels, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Advance a PDE state by one learned Euler step.

        Parameters
        ----------
        state:
            PDE state ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Updated PDE state.
        """

        features = torch.cat([state, self.dx(state), self.dy(state), self.lap(state)], dim=1)
        return state + 0.1 * self.mix(torch.tanh(features))


class PDENet(nn.Module):
    """Compact PDE-Net with recurrent learned operator steps."""

    def __init__(self, steps: int = 3) -> None:
        """Initialize the recurrent PDE-Net.

        Parameters
        ----------
        steps:
            Number of time steps.
        """

        super().__init__()
        self.steps = steps
        self.step = PDENetStep()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Run several learned PDE steps.

        Parameters
        ----------
        state:
            Initial PDE state.

        Returns
        -------
        torch.Tensor
            Final PDE state.
        """

        for _ in range(self.steps):
            state = self.step(state)
        return state


class PDERefiner(nn.Module):
    """Compact PDE-Refiner with iterative diffusion-style denoising."""

    def __init__(self, channels: int = 2, refinement_steps: int = 4) -> None:
        """Initialize a coarse predictor and residual refiner.

        Parameters
        ----------
        channels:
            Number of PDE state channels.
        refinement_steps:
            Number of denoising refinement steps.
        """

        super().__init__()
        self.refinement_steps = refinement_steps
        self.coarse = nn.Conv2d(channels, channels, 5, padding=2)
        self.step_embed = nn.Embedding(refinement_steps, channels)
        self.refine = nn.Sequential(
            nn.Conv2d(channels * 3, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, channels, 3, padding=1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Refine a coarse PDE forecast.

        Parameters
        ----------
        state:
            PDE state tensor.

        Returns
        -------
        torch.Tensor
            Refined PDE forecast.
        """

        coarse = self.coarse(state)
        refined = coarse
        for step in range(self.refinement_steps):
            sigma = 1.0 / float(step + 1)
            step_ids = torch.full((state.shape[0],), step, dtype=torch.long, device=state.device)
            step_cond = self.step_embed(step_ids).view(state.shape[0], -1, 1, 1)
            step_cond = step_cond.expand(-1, -1, state.shape[-2], state.shape[-1])
            denoise = self.refine(torch.cat([state, refined, step_cond], dim=1))
            refined = refined - sigma * torch.tanh(denoise) + (1.0 - sigma) * (coarse - refined)
        return refined


def build_pde_net() -> nn.Module:
    """Build compact PDE-Net.

    Returns
    -------
    nn.Module
        Random-init PDE-Net.
    """

    return PDENet()


def build_pde_refiner() -> nn.Module:
    """Build compact PDE-Refiner.

    Returns
    -------
    nn.Module
        Random-init PDE-Refiner.
    """

    return PDERefiner()


def example_state() -> torch.Tensor:
    """Create a compact PDE state.

    Returns
    -------
    torch.Tensor
        State tensor ``(1, 2, 24, 24)``.
    """

    return torch.randn(1, 2, 24, 24)


MENAGERIE_ENTRIES = [
    ("PDE-Net", "build_pde_net", "example_state", "2018", "DYN"),
    ("PDE-Refiner", "build_pde_refiner", "example_state", "2023", "DYN"),
]
