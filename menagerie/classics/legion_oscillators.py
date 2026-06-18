"""LEGION (Locally Excitatory Globally Inhibitory Oscillators), 1995, Wang and Terman.

A model of visual scene segmentation through synchronization of relaxation
oscillators.  Each pixel drives one oscillator; locally-connected oscillators
that receive correlated inputs synchronize (excitatory coupling), while the
global inhibitor ensures that only one synchronized group fires at a time
(winner-take-all across segments).

Oscillator dynamics (Wang 1995 simplified form):
    dx_i = 3*x_i - x_i^3 + 2 - y_i + I_i + S_i   (fast variable)
    dy_i = epsilon * (alpha + gamma*tanh(beta*x_i) - y_i)   (slow variable)
    S_i  = rho * sum_{j in N(i)} H(x_j - threshold)        (local coupling)
    z     = sigma * H(max_i x_i - theta_z)                  (global inhibitor)

where H is a smooth sigmoid approximation of the Heaviside function (so the
module traces cleanly -- no data-dependent branching on tensor values).

A few discrete Euler steps are unrolled in forward() to show the dynamics.

Paper: Wang and Terman 1995, "Locally Excitatory Globally Inhibitory Oscillator
       Networks," IEEE Transactions on Neural Networks.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _smooth_step(u: Tensor, mu: float = 10.0) -> Tensor:
    """Differentiable smooth approximation of the Heaviside step function.

    H(u) ~ sigmoid(mu * u)

    Parameters
    ----------
    u:
        Input tensor (arbitrary shape).
    mu:
        Steepness; larger = closer to hard step.

    Returns
    -------
    Tensor
        Values in (0, 1) approximating H(u).
    """
    return torch.sigmoid(mu * u)


class LEGIONGrid(nn.Module):
    """2D grid of LEGION relaxation oscillators with local-E global-I coupling.

    State: (x, y, z) where x and y are per-pixel scalars and z is one global
    scalar.  An external input image I drives x via additive injection.

    A fixed number of Euler steps are unrolled in forward().
    """

    def __init__(
        self,
        height: int = 32,
        width: int = 32,
        n_steps: int = 4,
        epsilon: float = 0.02,
        alpha: float = -0.2,
        gamma: float = 3.0,
        beta: float = 3.0,
        rho: float = 0.1,
        sigma: float = 1.0,
        theta_x: float = 0.0,
        theta_z: float = 0.5,
        dt: float = 0.05,
    ) -> None:
        """Initialize LEGION grid parameters.

        Parameters
        ----------
        height:
            Grid height (pixels).
        width:
            Grid width (pixels).
        n_steps:
            Number of Euler time steps unrolled in forward().
        epsilon:
            Slow-variable time scale (small = slow).
        alpha, gamma, beta:
            Parameters of the slow nullcline.
        rho:
            Local excitatory coupling strength.
        sigma:
            Global inhibitor strength.
        theta_x:
            Threshold for local coupling Heaviside.
        theta_z:
            Threshold for global inhibitor Heaviside.
        dt:
            Euler step size.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.n_steps = n_steps
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.rho = rho
        self.sigma = sigma
        self.theta_x = theta_x
        self.theta_z = theta_z
        self.dt = dt

    def _local_coupling(self, x: Tensor) -> Tensor:
        """Compute local excitatory neighborhood coupling via nearest-neighbors.

        Parameters
        ----------
        x:
            Fast-variable map with shape ``(B, 1, H, W)``.

        Returns
        -------
        Tensor
            Coupling signal with shape ``(B, 1, H, W)``.
        """
        # Neighbor sum via 3x3 avg conv (excludes center with kernel trick)
        F.avg_pool2d(x, kernel_size=3, stride=1, padding=1) * 9.0 - x
        firing = _smooth_step(x - self.theta_x)
        return self.rho * F.avg_pool2d(firing, kernel_size=3, stride=1, padding=1) * 8.0

    def forward(self, stim: Tensor) -> Tensor:
        """Unroll n_steps of LEGION dynamics.

        Parameters
        ----------
        stim:
            External stimulus (image) with shape ``(B, 1, H, W)``.

        Returns
        -------
        Tensor
            Fast variable x after n_steps, shape ``(B, 1, H, W)``.
            Values near +1 indicate actively-firing oscillators (one segment).
        """
        B, C, H, W = stim.shape
        # Initialize states
        x = torch.zeros_like(stim)
        y = torch.full_like(stim, -1.5)
        z = stim.new_zeros(B, 1, 1, 1)  # global inhibitor scalar

        for _ in range(self.n_steps):
            # Local coupling
            coupling = self._local_coupling(x)
            # dx: fast variable (cubic nullcline + coupling - global inhibitor)
            dx = 3.0 * x - x.pow(3) + 2.0 - y + stim + coupling - self.sigma * z
            # dy: slow variable
            dy = self.epsilon * (self.alpha + self.gamma * torch.tanh(self.beta * x) - y)
            # dz: global inhibitor driven by any active oscillator
            max_x = x.flatten(1).max(dim=-1).values.view(B, 1, 1, 1)
            dz = _smooth_step(max_x - self.theta_z) - z

            x = x + self.dt * dx
            y = y + self.dt * dy
            z = z + self.dt * dz

        return x


def build() -> nn.Module:
    """Build a small LEGION oscillator grid.

    Returns
    -------
    nn.Module
        Configured ``LEGIONGrid`` instance.
    """
    return LEGIONGrid(height=32, width=32, n_steps=4)


def example_input() -> Tensor:
    """Create a grayscale image stimulus for the LEGION grid.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 1, 32, 32)``.
    """
    return torch.randn(1, 1, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "LEGION (Locally Excitatory Globally Inhibitory Oscillators)",
        "build",
        "example_input",
        "1995",
        "RT",
    )
]
