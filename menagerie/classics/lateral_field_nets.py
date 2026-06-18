"""Lateral-field and substrate-like differentiable nets, 2019-2020.

Paper: Kraska 2018, "The Case for Learned Index Structures"; de Avila Belbute-
Peres 2018, "End-to-End Differentiable Physics"; Mordvintsev 2020, "Growing
Neural Cellular Automata." These compact modules retain the core differentiable
mechanisms while omitting training loops and task-specific datasets.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RecursiveModelIndex(nn.Module):
    """Two-stage learned index with soft routing to leaf regressors."""

    def __init__(self, n_leaves: int = 4, hidden: int = 8) -> None:
        """Initialize root and leaf regressors.

        Parameters
        ----------
        n_leaves
            Number of second-stage regressors.
        hidden
            Hidden width for the root router.
        """
        super().__init__()
        self.n_leaves = n_leaves
        self.root = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, n_leaves))
        self.leaves = nn.ModuleList([nn.Linear(1, 1) for _ in range(n_leaves)])

    def forward(self, key: Tensor) -> Tensor:
        """Predict normalized positions for sorted scalar keys.

        Parameters
        ----------
        key
            Sorted-key tensor with shape ``(batch, 1)``.

        Returns
        -------
        Tensor
            Position estimate tensor with shape ``(batch, 1)``.
        """
        route = torch.softmax(self.root(key), dim=-1)
        leaf_outputs = torch.stack([leaf(key).squeeze(-1) for leaf in self.leaves], dim=-1)
        position = (route * leaf_outputs).sum(dim=-1, keepdim=True)
        return torch.sigmoid(position)


class DifferentiablePhysicsStep(nn.Module):
    """Learnable mass-spring system rolled with semi-implicit Euler updates."""

    def __init__(self, n_bodies: int = 4, steps: int = 3, dt: float = 0.05) -> None:
        """Initialize physical parameters.

        Parameters
        ----------
        n_bodies
            Number of point masses.
        steps
            Number of Euler integration steps.
        dt
            Integration time step.
        """
        super().__init__()
        self.steps = steps
        self.dt = dt
        self.stiffness_raw = nn.Parameter(torch.randn(n_bodies, n_bodies) * 0.05)
        self.mass_raw = nn.Parameter(torch.zeros(n_bodies))
        self.damping = nn.Parameter(torch.full((n_bodies, 1), 0.05))

    def forward(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Roll positions and velocities through a spring Euler integrator.

        Parameters
        ----------
        inputs
            Tuple ``(position, velocity)``, each shaped ``(batch, n_bodies, 3)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated position and velocity tensors.
        """
        position, velocity = inputs
        stiffness = self.stiffness_raw @ self.stiffness_raw.T
        mass = F.softplus(self.mass_raw).view(1, -1, 1) + 0.1
        damping = F.softplus(self.damping).view(1, -1, 1)
        for _ in range(self.steps):
            spring_force = -torch.einsum("ij,bjd->bid", stiffness, position)
            force = spring_force - damping * velocity
            velocity = velocity + self.dt * force / mass
            position = position + self.dt * velocity
        return position, velocity


class SelfClassifyingNCA(nn.Module):
    """Self-classifying neural cellular automaton with fixed perception filters."""

    def __init__(self, channels: int = 19, classes: int = 10, steps: int = 4) -> None:
        """Initialize perception buffers and update network.

        Parameters
        ----------
        channels
            Number of cellular state channels.
        classes
            Number of class-logit channels to pool from the state.
        steps
            Number of cellular update iterations.
        """
        super().__init__()
        self.channels = channels
        self.classes = classes
        self.steps = steps
        identity = torch.zeros(channels, 1, 3, 3)
        identity[:, :, 1, 1] = 1.0
        sobel_x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]) / 8.0
        sobel_y = sobel_x.T
        self.register_buffer("identity_kernel", identity)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))
        self.update = nn.Sequential(
            nn.Conv2d(channels * 3, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, channels, kernel_size=1, bias=False),
        )

    def _perceive(self, state: Tensor) -> Tensor:
        """Collect identity and Sobel perceptions for each channel.

        Parameters
        ----------
        state
            Cellular state tensor of shape ``(batch, channels, height, width)``.

        Returns
        -------
        Tensor
            Concatenated perception tensor.
        """
        identity = F.conv2d(state, self.identity_kernel, padding=1, groups=self.channels)
        grad_x = F.conv2d(state, self.sobel_x, padding=1, groups=self.channels)
        grad_y = F.conv2d(state, self.sobel_y, padding=1, groups=self.channels)
        return torch.cat((identity, grad_x, grad_y), dim=1)

    def forward(self, state: Tensor) -> Tensor:
        """Run cellular updates and pool class logits.

        Parameters
        ----------
        state
            NCA state tensor with shape ``(batch, channels, height, width)``.

        Returns
        -------
        Tensor
            Class logits with shape ``(batch, classes)``.
        """
        for _ in range(self.steps):
            state = state + 0.1 * self.update(self._perceive(state))
        return state[:, 1 : self.classes + 1].mean(dim=(2, 3))


def build_rmi_learned_index() -> nn.Module:
    """Build a two-stage recursive model index.

    Returns
    -------
    nn.Module
        Configured ``RecursiveModelIndex`` instance.
    """
    return RecursiveModelIndex()


def example_input_rmi_learned_index() -> Tensor:
    """Create a sorted-key example.

    Returns
    -------
    Tensor
        Example input with shape ``(4, 1)``.
    """
    return torch.linspace(0.0, 1.0, steps=4).unsqueeze(-1)


def build_differentiable_physics_integrator_step() -> nn.Module:
    """Build a differentiable physics integrator module.

    Returns
    -------
    nn.Module
        Configured ``DifferentiablePhysicsStep`` instance.
    """
    return DifferentiablePhysicsStep()


def example_input_differentiable_physics_integrator_step() -> tuple[Tensor, Tensor]:
    """Create position and velocity examples.

    Returns
    -------
    tuple[Tensor, Tensor]
        Position and velocity tensors, each with shape ``(1, 4, 3)``.
    """
    position = torch.randn(1, 4, 3) * 0.1
    velocity = torch.zeros(1, 4, 3)
    return position, velocity


def build_self_classifying_nca() -> nn.Module:
    """Build a small self-classifying NCA.

    Returns
    -------
    nn.Module
        Configured ``SelfClassifyingNCA`` instance.
    """
    return SelfClassifyingNCA()


def example_input_self_classifying_nca() -> Tensor:
    """Create an NCA state example.

    Returns
    -------
    Tensor
        Example input with shape ``(1, 19, 28, 28)``.
    """
    state = torch.zeros(1, 19, 28, 28)
    state[:, 0, 12:16, 12:16] = 1.0
    return state


MENAGERIE_ENTRIES = [
    (
        "Recursive Model Index (RMI learned index)",
        "build_rmi_learned_index",
        "example_input_rmi_learned_index",
        "2019",
        "CH-D",
    ),
    (
        "Differentiable physics integrator step",
        "build_differentiable_physics_integrator_step",
        "example_input_differentiable_physics_integrator_step",
        "2019",
        "CH-D",
    ),
    (
        "Self-classifying NCA",
        "build_self_classifying_nca",
        "example_input_self_classifying_nca",
        "2020",
        "CH-D",
    ),
]
