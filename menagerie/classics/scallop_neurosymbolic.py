"""Scallopy-style differentiable neurosymbolic model.

Scallopy exposes Scallop logic programs to Python and can combine neural
perception with provenance-semiring reasoning.  This compact Torch classic does
not embed the Rust solver.  Instead, it keeps the faithful-core atlas: a small
perception CNN produces soft facts, then fixed tensor contractions apply a tiny
logic program using product soft-AND and probabilistic soft-OR.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptionNet(nn.Module):
    """Small CNN that emits object-attribute fact probabilities."""

    def __init__(self, objects: int = 3, attributes: int = 4) -> None:
        """Initialize image perception layers.

        Parameters
        ----------
        objects:
            Number of object slots in the downstream logic program.
        attributes:
            Number of attribute symbols predicted for each object.
        """

        super().__init__()
        self.objects = objects
        self.attributes = attributes
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, objects * attributes),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict soft object-attribute facts from an image.

        Parameters
        ----------
        image:
            Batch of grayscale images with shape ``(batch, 1, height, width)``.

        Returns
        -------
        torch.Tensor
            Soft fact probabilities with shape ``(batch, objects, attributes)``.
        """

        logits = self.head(self.conv(image)).view(-1, self.objects, self.attributes)
        return torch.softmax(logits, dim=-1)


class DifferentiableLogicLayer(nn.Module):
    """Fixed tensorized Scallop-style rule application layer."""

    def __init__(self, objects: int = 3, attributes: int = 4, relations: int = 3) -> None:
        """Create fixed rule tensors for differentiable logical inference.

        Parameters
        ----------
        objects:
            Number of object slots.
        attributes:
            Number of attribute symbols.
        relations:
            Number of relation predicates produced by the program.
        """

        super().__init__()
        self.objects = objects
        same_attr = torch.eye(attributes)
        successor = torch.zeros(attributes, attributes)
        successor[:-1, 1:] = torch.eye(attributes - 1)
        distant = 1.0 - same_attr
        self.register_buffer("rules", torch.stack([same_attr, successor, distant])[:relations])
        self.register_buffer("query_weights", torch.tensor([1.0, 0.7, 0.4])[:relations])

    def _pairwise_rule_scores(self, facts: torch.Tensor) -> torch.Tensor:
        """Apply binary predicate rules by matrix contraction.

        Parameters
        ----------
        facts:
            Soft fact probabilities of shape ``(batch, objects, attributes)``.

        Returns
        -------
        torch.Tensor
            Pairwise relation scores of shape ``(batch, relations, objects, objects)``.
        """

        return torch.einsum("nia,rac,njc->nrij", facts, self.rules, facts)

    def _soft_or(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        """Aggregate alternative proofs with probabilistic soft-OR.

        Parameters
        ----------
        scores:
            Alternative proof probabilities.
        dim:
            Dimension containing alternatives.

        Returns
        -------
        torch.Tensor
            Aggregated proof probabilities with ``dim`` removed.
        """

        return 1.0 - torch.prod(1.0 - scores.clamp(0.0, 1.0), dim=dim)

    def forward(self, facts: torch.Tensor) -> torch.Tensor:
        """Infer query scores from soft perception facts.

        Parameters
        ----------
        facts:
            Soft object-attribute facts of shape ``(batch, objects, attributes)``.

        Returns
        -------
        torch.Tensor
            Query logits for three symbolic outcomes.
        """

        pair_scores = self._pairwise_rule_scores(facts)
        mask = 1.0 - torch.eye(self.objects, device=facts.device).view(
            1, 1, self.objects, self.objects
        )
        pair_scores = pair_scores * mask
        relation_exists = self._soft_or(pair_scores.flatten(start_dim=2), dim=-1)
        first_object_facts = facts[:, 0]
        anchored_rule = torch.einsum("na,rac,nc->nr", first_object_facts, self.rules, facts[:, 1])
        weighted_relation = relation_exists * self.query_weights.view(1, -1)
        return torch.cat([weighted_relation, anchored_rule.mean(dim=-1, keepdim=True)], dim=-1)


class ScallopNeurosymbolicNet(nn.Module):
    """Perception-to-logic Scallopy-style differentiable model."""

    def __init__(self) -> None:
        """Initialize perception and differentiable logic modules."""

        super().__init__()
        self.perception = PerceptionNet()
        self.logic = DifferentiableLogicLayer()
        self.classifier = nn.Linear(4, 3)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Classify an image through soft facts and tensorized rules.

        Parameters
        ----------
        image:
            Batch of grayscale images with shape ``(batch, 1, 16, 16)``.

        Returns
        -------
        torch.Tensor
            Class logits derived from differentiable logical queries.
        """

        facts = self.perception(image)
        proof_scores = self.logic(facts)
        return self.classifier(proof_scores)


def build() -> nn.Module:
    """Build the compact Scallopy-style neurosymbolic classic.

    Returns
    -------
    nn.Module
        Random-init perception plus differentiable tensor-logic model.
    """

    return ScallopNeurosymbolicNet()


def example_input() -> torch.Tensor:
    """Return a compact image input for the perception front-end.

    Returns
    -------
    torch.Tensor
        Grayscale image tensor of shape ``(1, 1, 16, 16)``.
    """

    return torch.linspace(0.0, 1.0, 16 * 16, dtype=torch.float32).view(1, 1, 16, 16)


MENAGERIE_ENTRIES = [
    ("scallopy", "build", "example_input", 2023, "CD"),
]
