"""SMoG: self-supervised visual representation learning with semantic grouping.

Wen et al. (NeurIPS 2022), "Self-Supervised Visual Representation Learning with
Semantic Grouping".  Public descriptions of SMoG emphasize an online/target
Siamese visual encoder, projection/prediction heads, and semantic group
prototypes that provide group-level targets in addition to instance alignment.

This compact random-init reconstruction keeps those distinctive pieces:
two augmented views are encoded by shared and momentum-style branches, projected,
predicted, compared to normalized prototypes, and returned with both online and
target group assignments.  The convolutional encoder is intentionally small so
TorchLens can render the graph without requiring the original training package.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """Small CNN encoder used in place of the paper's large ImageNet backbone."""

    def __init__(self, width: int = 24, out_dim: int = 64) -> None:
        """Initialize the compact image encoder.

        Parameters
        ----------
        width:
            Number of channels in the first convolutional stage.
        out_dim:
            Output feature dimension after global pooling.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=False),
            nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=False),
        )
        self.stage = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=False),
        )
        self.head = nn.Linear(width * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image batch into global visual features.

        Parameters
        ----------
        x:
            Image tensor of shape ``(batch, 3, height, width)``.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape ``(batch, out_dim)``.
        """

        x = self.stage(self.stem(x))
        x = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        return self.head(x)


class MLPHead(nn.Module):
    """Two-layer projection or prediction head."""

    def __init__(self, dim: int = 64, hidden: int = 96, out_dim: int = 64) -> None:
        """Initialize the MLP head.

        Parameters
        ----------
        dim:
            Input feature dimension.
        hidden:
            Hidden layer dimension.
        out_dim:
            Output embedding dimension.
        """

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=False),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP head.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, dim)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, out_dim)``.
        """

        return self.net(x)


class SMoG(nn.Module):
    """Compact SMoG-style Siamese encoder with semantic group prototypes."""

    def __init__(self, dim: int = 64, num_groups: int = 8) -> None:
        """Initialize the SMoG reconstruction.

        Parameters
        ----------
        dim:
            Embedding dimension for the projection space.
        num_groups:
            Number of semantic group prototypes.
        """

        super().__init__()
        self.online_encoder = ConvEncoder(out_dim=dim)
        self.target_encoder = ConvEncoder(out_dim=dim)
        self.projector = MLPHead(dim=dim, hidden=96, out_dim=dim)
        self.predictor = MLPHead(dim=dim, hidden=96, out_dim=dim)
        self.target_projector = MLPHead(dim=dim, hidden=96, out_dim=dim)
        self.prototypes = nn.Parameter(torch.randn(num_groups, dim))

    def _branch(self, x: torch.Tensor, target: bool = False) -> torch.Tensor:
        """Run one SMoG encoder/projection branch.

        Parameters
        ----------
        x:
            Image tensor.
        target:
            Whether to use the target branch.

        Returns
        -------
        torch.Tensor
            L2-normalized projected representation.
        """

        if target:
            z = self.target_projector(self.target_encoder(x))
        else:
            z = self.projector(self.online_encoder(x))
        return F.normalize(z, dim=-1)

    def forward(self, views: tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode two augmented views and compute instance/group predictions.

        Parameters
        ----------
        views:
            Pair of image tensors representing two augmentations.

        Returns
        -------
        dict[str, torch.Tensor]
            Online predictions, target embeddings, and semantic group logits.
        """

        x1, x2 = views
        online_1 = F.normalize(self.predictor(self._branch(x1)), dim=-1)
        online_2 = F.normalize(self.predictor(self._branch(x2)), dim=-1)
        target_1 = self._branch(x1, target=True)
        target_2 = self._branch(x2, target=True)
        prototypes = F.normalize(self.prototypes, dim=-1)
        return {
            "online_1": online_1,
            "online_2": online_2,
            "target_1": target_1,
            "target_2": target_2,
            "group_logits_1": online_1 @ prototypes.t(),
            "group_logits_2": online_2 @ prototypes.t(),
            "target_groups_1": target_1 @ prototypes.t(),
            "target_groups_2": target_2 @ prototypes.t(),
        }


def build() -> nn.Module:
    """Build the compact SMoG model.

    Returns
    -------
    nn.Module
        Random-init SMoG reconstruction in evaluation mode.
    """

    return SMoG().eval()


def example_input() -> tuple[torch.Tensor, torch.Tensor]:
    """Return two small image views for tracing.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Two tensors of shape ``(2, 3, 32, 32)``.
    """

    return torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "SMoG",
        "build",
        "example_input",
        "2022",
        "DC",
    ),
]
