"""Matrix Capsules with EM routing.

Hinton, Sabour, and Frosst, 2018, "Matrix Capsules with EM Routing".  Capsules
carry a logistic activation and a 4x4 pose matrix; higher capsules are inferred
from transformed lower-capsule votes by an expectation-maximization routing
procedure.  This compact reconstruction keeps the conv stem, primary capsules,
learned vote transforms, EM routing, and class capsule activations.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MatrixCapsulesEM(nn.Module):
    """Compact matrix capsule classifier with EM routing."""

    def __init__(self, in_caps: int = 8, out_caps: int = 5, routing_iters: int = 3) -> None:
        """Initialize capsule transforms and EM parameters.

        Parameters
        ----------
        in_caps:
            Number of primary capsule types.
        out_caps:
            Number of class capsules.
        routing_iters:
            Number of EM routing iterations.
        """
        super().__init__()
        self.routing_iters = routing_iters
        self.stem = nn.Conv2d(1, 16, 5, stride=2, padding=2)
        self.pose = nn.Conv2d(16, in_caps * 16, 1)
        self.act = nn.Conv2d(16, in_caps, 1)
        self.transforms = nn.Parameter(torch.randn(in_caps, out_caps, 16, 16) * 0.04)
        self.beta_u = nn.Parameter(torch.zeros(out_caps, 16))
        self.beta_a = nn.Parameter(torch.zeros(out_caps))

    def forward(self, image: Tensor) -> Tensor:
        """Route primary capsules to class activations.

        Parameters
        ----------
        image:
            Grayscale image tensor with shape ``(batch, 1, height, width)``.

        Returns
        -------
        Tensor
            Class capsule activations.
        """
        feat = F.relu(self.stem(image))
        batch = image.shape[0]
        poses = self.pose(feat).permute(0, 2, 3, 1).reshape(batch, -1, self.transforms.shape[0], 16)
        acts = (
            torch.sigmoid(self.act(feat))
            .permute(0, 2, 3, 1)
            .reshape(batch, -1, self.transforms.shape[0])
        )
        votes = torch.einsum("bnci,coij->bncoj", poses, self.transforms)
        votes = votes.reshape(batch, -1, self.transforms.shape[1], 16)
        acts = acts.reshape(batch, -1, 1)
        responsibilities = torch.softmax(torch.zeros_like(votes[..., 0]), dim=-1)
        activation = torch.ones(
            batch, self.transforms.shape[1], device=image.device, dtype=image.dtype
        )
        for _ in range(self.routing_iters):
            weights = responsibilities * acts
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
            mean = (weights.unsqueeze(-1) * votes).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
            var = (weights.unsqueeze(-1) * (votes - mean).square()).sum(dim=1) / denom.squeeze(
                1
            ).unsqueeze(-1)
            cost = (self.beta_u + torch.log(var.clamp_min(1.0e-6))).sum(dim=-1)
            activation = torch.sigmoid(self.beta_a - cost)
            log_prob = -0.5 * ((votes - mean).square() / var.unsqueeze(1).clamp_min(1.0e-6)).sum(
                dim=-1
            )
            responsibilities = torch.softmax(
                log_prob + torch.log(activation.unsqueeze(1).clamp_min(1.0e-6)), dim=-1
            )
        return activation


def build() -> nn.Module:
    """Build compact Matrix Capsules with EM routing.

    Returns
    -------
    nn.Module
        Random-initialized capsule network.
    """
    return MatrixCapsulesEM().eval()


def example_input() -> Tensor:
    """Return a small grayscale image.

    Returns
    -------
    Tensor
        Tensor with shape ``(1, 1, 16, 16)``.
    """
    return torch.randn(1, 1, 16, 16)


MENAGERIE_ENTRIES = [
    ("matrix_capsules_em", "build", "example_input", "2018", "DC"),
]
