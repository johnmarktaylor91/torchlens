"""NMRF stereo: neural Markov Random Field inference for stereo matching.

Paper: Neural Markov Random Field for Stereo Matching, Guan, Wang, and Liu, CVPR 2024.

The distinctive mechanism is not a plain cost volume.  NMRF learns unary and pairwise
potential functions and performs variational/message-passing inference over a pruned
set of disparity proposals.  This compact reconstruction keeps a DPN-like proposal
embedding, learned unary terms, learned self-/neighbor-pair messages, and iterative
belief updates over disparity labels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralMessagePassing(nn.Module):
    """Learned MRF pairwise message block over disparity proposals."""

    def __init__(self, labels: int, hidden: int) -> None:
        """Initialize pairwise potential networks.

        Parameters
        ----------
        labels:
            Number of disparity labels.
        hidden:
            Hidden feature width.
        """

        super().__init__()
        self.self_edge = nn.Linear(labels, labels, bias=False)
        self.neighbor_edge = nn.Sequential(
            nn.Linear(hidden + labels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, labels),
        )

    def forward(self, belief: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute neural self-edge and spatial-neighbor messages.

        Parameters
        ----------
        belief:
            Current label belief ``(batch, labels, height, width)``.
        context:
            Image context features ``(batch, hidden, height, width)``.

        Returns
        -------
        torch.Tensor
            Message tensor with the same shape as ``belief``.
        """

        pooled = (
            F.avg_pool2d(belief, kernel_size=3, stride=1, padding=1)
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        self_msg = self.self_edge(belief.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        pair_in = torch.cat([context.permute(0, 2, 3, 1), pooled], dim=-1)
        neigh_msg = self.neighbor_edge(pair_in).permute(0, 3, 1, 2)
        return self_msg + neigh_msg


class NMRFStereo(nn.Module):
    """Compact learned-potential stereo MRF with iterative belief refinement."""

    def __init__(self, labels: int = 5, hidden: int = 16, iters: int = 3) -> None:
        """Initialize feature, proposal, unary, and message networks.

        Parameters
        ----------
        labels:
            Number of retained disparity proposals.
        hidden:
            Feature width.
        iters:
            Number of variational message-passing refinements.
        """

        super().__init__()
        self.labels = labels
        self.iters = iters
        self.feature = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
        )
        self.proposal = nn.Conv2d(2 * hidden, labels, 1)
        self.unary = nn.Sequential(
            nn.Conv2d(2 * hidden + labels, hidden, 1), nn.ReLU(), nn.Conv2d(hidden, labels, 1)
        )
        self.message = NeuralMessagePassing(labels, hidden)
        self.register_buffer("disp_values", torch.linspace(0.0, float(labels - 1), labels))

    def forward(self, stereo: torch.Tensor) -> torch.Tensor:
        """Estimate a disparity map from left/right images.

        Parameters
        ----------
        stereo:
            Stereo pair ``(batch, 2, 3, height, width)``.

        Returns
        -------
        torch.Tensor
            Soft expected disparity ``(batch, height, width)``.
        """

        left = stereo[:, 0]
        right = stereo[:, 1]
        lf = self.feature(left)
        rf = self.feature(right)
        corr = []
        for d in range(self.labels):
            shifted = torch.roll(rf, shifts=-d, dims=-1)
            corr.append((lf * shifted).mean(dim=1, keepdim=True))
        proposal_logits = self.proposal(torch.cat([lf, rf], dim=1))
        proposals = torch.softmax(proposal_logits, dim=1)
        unary = self.unary(torch.cat([lf, rf, torch.cat(corr, dim=1)], dim=1))
        belief = torch.softmax(unary + proposals, dim=1)
        for _ in range(self.iters):
            msg = self.message(belief, lf)
            belief = torch.softmax(unary + msg, dim=1)
        disp = (belief * self.disp_values.view(1, self.labels, 1, 1)).sum(dim=1)
        return disp


def build() -> nn.Module:
    """Build compact NMRF stereo.

    Returns
    -------
    nn.Module
        Random-init stereo MRF.
    """

    return NMRFStereo()


def example_input() -> torch.Tensor:
    """Create a small stereo pair.

    Returns
    -------
    torch.Tensor
        Example stereo input ``(1, 2, 3, 16, 16)``.
    """

    return torch.randn(1, 2, 3, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "nmrf_stereo",
        "build",
        "example_input",
        "2024",
        "vision/stereo",
    ),
]
