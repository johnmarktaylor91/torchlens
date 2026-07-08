"""PyKEEN-style knowledge graph embedding scorers.

Paper: representative KGE architectures collected in PyKEEN, including TransE,
RESCAL, DistMult, ComplEx, RotatE, ConvKB, NTN, TuckER, BoxE, PairRE, and related
embedding scorers.

This module is a Torch-only random-init reimplementation of the scoring paths used
by the PyKEEN catalog rows. It intentionally focuses on ``score_hrt``-style
forward computation for small synthetic triples: entity and relation embeddings,
variant-specific interaction layers, and a score per ``(head, relation, tail)``.
Training losses, negative sampling, and PyKEEN's triples-factory plumbing are not
part of the architecture being traced here.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PyKEENScorer(nn.Module):
    """Compact KGE scorer with variant-specific interaction formulas."""

    def __init__(
        self,
        variant: str,
        num_entities: int = 16,
        num_relations: int = 6,
        embedding_dim: int = 24,
    ) -> None:
        """Initialize entity/relation parameters and auxiliary scoring layers.

        Parameters
        ----------
        variant:
            Canonical lowercase scorer name.
        num_entities:
            Number of synthetic entities.
        num_relations:
            Number of synthetic relations.
        embedding_dim:
            Real embedding width. Complex/quaternion variants split this width into
            equal components internally.
        """
        super().__init__()
        self.variant = variant
        self.embedding_dim = embedding_dim
        self.entity = nn.Embedding(num_entities, embedding_dim)
        self.relation = nn.Embedding(num_relations, embedding_dim)
        self.tail_relation = nn.Embedding(num_relations, embedding_dim)
        self.head_projection = nn.Embedding(num_entities, embedding_dim)
        self.tail_projection = nn.Embedding(num_entities, embedding_dim)
        self.relation_projection = nn.Embedding(num_relations, embedding_dim)
        self.relation_matrix = nn.Embedding(num_relations, embedding_dim * embedding_dim)
        self.ntn_tensor = nn.Embedding(num_relations, 4 * embedding_dim * embedding_dim)
        self.ntn_linear = nn.Linear(embedding_dim * 2, 4)
        self.ntn_out = nn.Linear(4, 1)
        self.convkb = nn.Conv1d(3, 8, kernel_size=3, padding=1)
        self.convkb_out = nn.Linear(8 * embedding_dim, 1)
        self.proje = nn.Linear(embedding_dim * 2, embedding_dim)
        self.proje_out = nn.Linear(embedding_dim, 1)
        self.ermlp = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
        self.tucker_core = nn.Parameter(
            torch.randn(embedding_dim, embedding_dim, embedding_dim) * 0.03
        )
        self.autosf_weights = nn.Parameter(torch.randn(4, embedding_dim) * 0.05)
        self.box_base = nn.Embedding(num_entities, embedding_dim)
        self.box_delta = nn.Embedding(num_entities, embedding_dim)

    def forward(self, triples_float: Tensor) -> Tensor:
        """Score a batch of ``(head, relation, tail)`` triples.

        Parameters
        ----------
        triples_float:
            Tensor with shape ``(batch, 3)``. Values are cast to integer IDs inside
            the module to keep TorchLens validation stable for index-based models.

        Returns
        -------
        Tensor
            One score per input triple.
        """
        triples = triples_float.long()
        heads = triples[:, 0].remainder(self.entity.num_embeddings)
        relations = triples[:, 1].remainder(self.relation.num_embeddings)
        tails = triples[:, 2].remainder(self.entity.num_embeddings)
        head = self.entity(heads)
        relation = self.relation(relations)
        tail = self.entity(tails)
        return self._score(head, relation, tail, heads, relations, tails)

    def _score(
        self,
        head: Tensor,
        relation: Tensor,
        tail: Tensor,
        heads: Tensor,
        relations: Tensor,
        tails: Tensor,
    ) -> Tensor:
        """Dispatch to the requested KGE scoring formula.

        Parameters
        ----------
        head:
            Head entity embeddings.
        relation:
            Relation embeddings.
        tail:
            Tail entity embeddings.
        heads:
            Integer head IDs.
        relations:
            Integer relation IDs.
        tails:
            Integer tail IDs.

        Returns
        -------
        Tensor
            Score vector.
        """
        if self.variant in {"transe", "transf"}:
            return -(head + relation - tail).norm(p=1, dim=-1)
        if self.variant == "transh":
            normal = F.normalize(self.relation_projection(relations), dim=-1)
            return -(
                self._project_hyperplane(head, normal)
                + relation
                - self._project_hyperplane(tail, normal)
            ).norm(dim=-1)
        if self.variant in {"transr", "transd"}:
            projected_head = self._relation_matrix(relations).bmm(head.unsqueeze(-1)).squeeze(-1)
            projected_tail = self._relation_matrix(relations).bmm(tail.unsqueeze(-1)).squeeze(-1)
            if self.variant == "transd":
                projected_head = projected_head + self.head_projection(heads)
                projected_tail = projected_tail + self.tail_projection(tails)
            return -(projected_head + relation - projected_tail).norm(dim=-1)
        if self.variant in {"distmult", "cp-kge", "simple-kge"}:
            rel_tail = self.tail_relation(relations) if self.variant == "simple-kge" else relation
            return (head * relation * tail + tail * rel_tail * head).sum(dim=-1)
        if self.variant in {"complex", "hole"}:
            return self._complex_score(head, relation, tail)
        if self.variant == "rotate":
            phase = torch.tanh(relation[:, : self.embedding_dim // 2]) * math.pi
            re_head, im_head = self._split_complex(head)
            re_tail, im_tail = self._split_complex(tail)
            re_rot = re_head * torch.cos(phase) - im_head * torch.sin(phase)
            im_rot = re_head * torch.sin(phase) + im_head * torch.cos(phase)
            return -torch.cat((re_rot - re_tail, im_rot - im_tail), dim=-1).norm(dim=-1)
        if self.variant == "quate":
            return self._quaternion_score(head, relation, tail)
        if self.variant == "rescal":
            return (
                head.unsqueeze(1)
                .bmm(self._relation_matrix(relations))
                .bmm(tail.unsqueeze(-1))
                .flatten()
            )
        if self.variant == "tucker":
            core_relation = torch.einsum("bd,dej->bej", relation, self.tucker_core)
            return head.unsqueeze(1).bmm(core_relation).bmm(tail.unsqueeze(-1)).flatten()
        if self.variant == "ntn":
            tensor = self.ntn_tensor(relations).view(-1, 4, self.embedding_dim, self.embedding_dim)
            bilinear = torch.einsum("bd,bkde,be->bk", head, tensor, tail)
            hidden = torch.tanh(bilinear + self.ntn_linear(torch.cat((head, tail), dim=-1)))
            return self.ntn_out(hidden).flatten()
        if self.variant == "convkb":
            stacked = torch.stack((head, relation, tail), dim=1)
            conv = torch.relu(self.convkb(stacked))
            return self.convkb_out(conv.flatten(start_dim=1)).flatten()
        if self.variant in {"ermlp", "ermlpe"}:
            features = torch.cat(
                (head, relation, tail if self.variant == "ermlp" else head * tail), dim=-1
            )
            return self.ermlp(features).flatten()
        if self.variant == "proje":
            return self.proje_out(
                torch.tanh(self.proje(torch.cat((head * relation, tail), dim=-1)))
            ).flatten()
        if self.variant == "boxe":
            lower = self.box_base(heads)
            upper = lower + F.softplus(self.box_delta(heads))
            point = tail + relation
            penalty = torch.relu(lower - point) + torch.relu(point - upper)
            return -penalty.sum(dim=-1)
        if self.variant == "pairre":
            rel_head = relation
            rel_tail = self.tail_relation(relations)
            return -(head * rel_head - tail * rel_tail).norm(dim=-1)
        if self.variant == "mure":
            matrix = torch.diag_embed(torch.tanh(relation))
            return -((matrix.bmm(head.unsqueeze(-1)).squeeze(-1) + relation) - tail).norm(dim=-1)
        if self.variant == "kg2e":
            variance = F.softplus(self.tail_relation(relations))
            return -((head + relation - tail).pow(2) / variance).sum(dim=-1)
        if self.variant == "toruse":
            distance = head + relation - tail
            wrapped = torch.minimum(distance.abs(), 1.0 - distance.abs().remainder(1.0))
            return -wrapped.norm(dim=-1)
        if self.variant == "structured-embedding":
            return -(
                self._relation_matrix(relations).bmm(head.unsqueeze(-1)).squeeze(-1) - tail
            ).norm(dim=-1)
        if self.variant == "unstructured-model":
            return -(head - tail).norm(dim=-1)
        if self.variant == "rgcn-kge":
            aggregated = torch.tanh(head + relation + self.tail_relation(relations))
            return (aggregated * tail).sum(dim=-1)
        if self.variant == "distma":
            return (torch.sin(head + relation) * torch.cos(tail)).sum(dim=-1)
        if self.variant == "crosse":
            crossover = torch.tanh(head * relation + relation)
            return (crossover * tail).sum(dim=-1)
        if self.variant == "autosf":
            terms = torch.stack(
                (head * relation, relation * tail, head * tail, head + relation - tail)
            )
            return (terms * self.autosf_weights[:, None, :]).sum(dim=(0, 2))
        return (head * relation * tail).sum(dim=-1)

    def _relation_matrix(self, relations: Tensor) -> Tensor:
        """Return relation-specific dense matrices.

        Parameters
        ----------
        relations:
            Relation IDs.

        Returns
        -------
        Tensor
            Matrix tensor with shape ``(batch, dim, dim)``.
        """
        return self.relation_matrix(relations).view(-1, self.embedding_dim, self.embedding_dim)

    def _split_complex(self, value: Tensor) -> tuple[Tensor, Tensor]:
        """Split an embedding into real and imaginary halves.

        Parameters
        ----------
        value:
            Real-valued embedding tensor.

        Returns
        -------
        tuple[Tensor, Tensor]
            Real and imaginary components.
        """
        half = self.embedding_dim // 2
        return value[:, :half], value[:, half : half * 2]

    def _complex_score(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        """Compute ComplEx/HolE-style circular correlation score.

        Parameters
        ----------
        head:
            Head embeddings.
        relation:
            Relation embeddings.
        tail:
            Tail embeddings.

        Returns
        -------
        Tensor
            Score vector.
        """
        head_re, head_im = self._split_complex(head)
        rel_re, rel_im = self._split_complex(relation)
        tail_re, tail_im = self._split_complex(tail)
        return (
            head_re * rel_re * tail_re
            + head_im * rel_re * tail_im
            + head_re * rel_im * tail_im
            - head_im * rel_im * tail_re
        ).sum(dim=-1)

    def _quaternion_score(self, head: Tensor, relation: Tensor, tail: Tensor) -> Tensor:
        """Compute a QuatE Hamilton-product score.

        Parameters
        ----------
        head:
            Head embeddings.
        relation:
            Relation embeddings.
        tail:
            Tail embeddings.

        Returns
        -------
        Tensor
            Score vector.
        """
        chunks = 4
        h0, h1, h2, h3 = head.chunk(chunks, dim=-1)
        r0, r1, r2, r3 = F.normalize(relation, dim=-1).chunk(chunks, dim=-1)
        t0, t1, t2, t3 = tail.chunk(chunks, dim=-1)
        p0 = h0 * r0 - h1 * r1 - h2 * r2 - h3 * r3
        p1 = h0 * r1 + h1 * r0 + h2 * r3 - h3 * r2
        p2 = h0 * r2 - h1 * r3 + h2 * r0 + h3 * r1
        p3 = h0 * r3 + h1 * r2 - h2 * r1 + h3 * r0
        return (p0 * t0 + p1 * t1 + p2 * t2 + p3 * t3).sum(dim=-1)

    @staticmethod
    def _project_hyperplane(value: Tensor, normal: Tensor) -> Tensor:
        """Project embeddings onto a TransH relation hyperplane.

        Parameters
        ----------
        value:
            Entity embeddings.
        normal:
            Unit relation normal vectors.

        Returns
        -------
        Tensor
            Projected embeddings.
        """
        return value - (value * normal).sum(dim=-1, keepdim=True) * normal


def build_model(variant: str = "distmult") -> nn.Module:
    """Build a PyKEEN-style scorer.

    Parameters
    ----------
    variant:
        Canonical lowercase scorer name.

    Returns
    -------
    nn.Module
        Random-init KGE scorer.
    """
    return PyKEENScorer(variant=variant)


def example_input() -> Tensor:
    """Return a small batch of synthetic triple IDs.

    Returns
    -------
    Tensor
        Float tensor with shape ``(4, 3)`` carrying entity/relation IDs.
    """
    return torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 0.0, 3.0], [3.0, 2.0, 0.0]])
