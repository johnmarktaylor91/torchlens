"""PCNN (Pulse-Coupled Neural Network / Eckhorn Linking-Field Model), 1990.

Eckhorn's Linking-Field model of primary visual cortex synchrony:
each neuron has a *feeding* channel F driven by the external stimulus and a
*linking* channel L driven by nearby neurons; they combine multiplicatively:

    U_i = F_i * (1 + beta * L_i)

A dynamic threshold Theta fires when U exceeds it; after firing the threshold
jumps up and then decays exponentially:

    Y_i[t] = step(U_i[t] - Theta_i[t])
    Theta_i[t+1] = V_theta * Y_i[t] + Theta_i[t] * exp(-alpha_theta)
    F_i[t+1]     = S_i     + F_i[t] * exp(-alpha_F)    (leaky integrate stim)
    L_i[t+1]     = sum_j W_ij * Y_j + L_i[t] * exp(-alpha_L)

Smooth sigmoid "pulse" replaces the hard step to make the module trace-clean
without data-dependent Python control flow on tensor values.

Paper: Eckhorn, Reitboeck, Arndt, Dicke 1990, "Feature Linking via
       Synchronization among Distributed Assemblies: Simulations of Results
       from Cat Visual Cortex," Neural Computation.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _soft_fire(u: Tensor, theta: Tensor, mu: float = 20.0) -> Tensor:
    """Smooth sigmoid approximation of step(U - Theta).

    Parameters
    ----------
    u:
        Internal activity ``U``.
    theta:
        Dynamic threshold ``Theta`` (same shape as ``u``).
    mu:
        Sigmoid slope (higher = closer to hard step).

    Returns
    -------
    Tensor
        Soft firing probability in (0, 1).
    """
    return torch.sigmoid(mu * (u - theta))


class PCNNLinkingField(nn.Module):
    """Pulse-Coupled Neural Network with leaky feeding + linking fields.

    A fixed number of time steps are unrolled in forward(); the smooth sigmoid
    pulse approximation makes all ops differentiable and trace-clean.
    """

    def __init__(
        self,
        height: int = 64,
        width: int = 64,
        n_steps: int = 5,
        beta: float = 0.2,
        alpha_F: float = 0.1,
        alpha_L: float = 0.2,
        alpha_theta: float = 0.1,
        V_theta: float = 5.0,
        link_kernel_size: int = 3,
        mu_fire: float = 20.0,
    ) -> None:
        """Initialize PCNN parameters.

        Parameters
        ----------
        height:
            Neuron grid height (pixels).
        width:
            Neuron grid width (pixels).
        n_steps:
            Number of time steps to unroll.
        beta:
            Linking field coupling strength.
        alpha_F:
            Feeding channel decay rate.
        alpha_L:
            Linking channel decay rate.
        alpha_theta:
            Dynamic threshold decay rate.
        V_theta:
            Threshold jump amplitude on firing.
        link_kernel_size:
            Spatial size of the linking-field convolutional kernel.
        mu_fire:
            Sigmoid slope for smooth firing function.
        """
        super().__init__()
        self.n_steps = n_steps
        self.beta = beta
        self.alpha_F = alpha_F
        self.alpha_L = alpha_L
        self.alpha_theta = alpha_theta
        self.V_theta = V_theta
        self.mu_fire = mu_fire
        # Learned linking-field kernel (spatial coupling among neurons)
        k = link_kernel_size
        self.link_kernel = nn.Parameter(torch.ones(1, 1, k, k) / (k * k))

    def forward(self, stim: Tensor) -> Tensor:
        """Unroll n_steps of PCNN dynamics.

        Parameters
        ----------
        stim:
            External stimulus (image) with shape ``(B, 1, H, W)``.
            Pixel values should be in [0, 1] for stable dynamics.

        Returns
        -------
        Tensor
            Stacked firing maps over time, shape ``(B, n_steps, H, W)``.
            Each slice is a soft pulse image in (0, 1).
        """
        B, C, H, W = stim.shape
        pad = self.link_kernel.shape[-1] // 2

        F_state = stim.clone()  # feeding channel
        L_state = torch.zeros_like(stim)
        Theta = torch.ones_like(stim) * 0.5

        fires = []
        decay_F = float(torch.exp(torch.tensor(-self.alpha_F)).item())
        decay_L = float(torch.exp(torch.tensor(-self.alpha_L)).item())
        decay_T = float(torch.exp(torch.tensor(-self.alpha_theta)).item())

        for _ in range(self.n_steps):
            # Internal activity: feeding modulated by linking
            U = F_state * (1.0 + self.beta * L_state)
            # Soft pulse
            Y = _soft_fire(U, Theta, mu=self.mu_fire)
            fires.append(Y)
            # Update channels
            F_state = stim + F_state * decay_F
            L_state = F.conv2d(Y, self.link_kernel, padding=pad) + L_state * decay_L
            Theta = self.V_theta * Y + Theta * decay_T

        return torch.stack(fires, dim=1).squeeze(2)  # (B, n_steps, H, W)


def build() -> nn.Module:
    """Build a small PCNN linking-field module.

    Returns
    -------
    nn.Module
        Configured ``PCNNLinkingField`` instance.
    """
    return PCNNLinkingField(height=64, width=64, n_steps=5)


def example_input() -> Tensor:
    """Create a normalized grayscale image for the PCNN.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 1, 64, 64)`` in [0, 1].
    """
    return torch.rand(1, 1, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "PCNN (Eckhorn Pulse-Coupled / Linking-Field)",
        "build",
        "example_input",
        "1990",
        "RT",
    )
]
