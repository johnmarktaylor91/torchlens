"""OpenKE knowledge-graph embedding models.

OpenKE supports classic KGE scorers including RESCAL, DistMult, ComplEx,
Analogy, TransE, TransH, TransR, TransD, SimplE, and RotatE.  This module keeps
their distinctive random-initialized scoring functions in compact PyTorch form:
bilinear semantic matching, complex-valued rotations/products, and translational
distance models with relation-specific projections.

Sources: OpenKE model list (THUNLP), TransE/TransH/TransR/TransD papers,
RESCAL, DistMult, ComplEx, ANALOGY, SimplE, and RotatE.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

ModelName = Literal[
    "analogy",
    "complex",
    "distmult",
    "rescal",
    "rotate",
    "simple",
    "transd",
    "transe",
    "transh",
    "transr",
]


class OpenKEScorer(nn.Module):
    """Compact OpenKE-style triple scorer.

    Parameters
    ----------
    model_name:
        Which KGE scoring family to instantiate.
    num_entities:
        Number of entity embeddings.
    num_relations:
        Number of relation embeddings.
    dim:
        Embedding dimension.
    """

    def __init__(
        self,
        model_name: ModelName,
        num_entities: int = 17,
        num_relations: int = 7,
        dim: int = 12,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.dim = dim
        complex_dim = dim * 2
        self.ent = nn.Embedding(num_entities, complex_dim)
        self.rel = nn.Embedding(num_relations, complex_dim)
        self.ent_tail = nn.Embedding(num_entities, dim)
        self.rel_tail = nn.Embedding(num_relations, dim)
        self.rel_mat = nn.Embedding(num_relations, dim * dim)
        self.rel_norm = nn.Embedding(num_relations, dim)
        self.ent_proj = nn.Embedding(num_entities, dim)
        self.rel_proj = nn.Embedding(num_relations, dim)
        self.rel_proj_mat = nn.Embedding(num_relations, dim * dim)

    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """Score integer triples.

        Parameters
        ----------
        triples:
            Long tensor of shape ``(batch, 3)`` with head, relation, tail IDs.

        Returns
        -------
        torch.Tensor
            Score tensor of shape ``(batch,)``.
        """

        h_id, r_id, t_id = triples[:, 0], triples[:, 1], triples[:, 2]
        h = self.ent(h_id)[:, : self.dim]
        t = self.ent(t_id)[:, : self.dim]
        r = self.rel(r_id)[:, : self.dim]
        if self.model_name == "distmult":
            return (h * r * t).sum(dim=-1)
        if self.model_name == "rescal":
            mat = self.rel_mat(r_id).view(-1, self.dim, self.dim)
            return torch.bmm(torch.bmm(h.unsqueeze(1), mat), t.unsqueeze(-1)).flatten()
        if self.model_name == "complex":
            return self._complex_score(h_id, r_id, t_id)
        if self.model_name == "analogy":
            return self._analogy_score(h_id, r_id, t_id)
        if self.model_name == "simple":
            return self._simple_score(h_id, r_id, t_id)
        if self.model_name == "rotate":
            return self._rotate_score(h_id, r_id, t_id)
        if self.model_name == "transh":
            return self._transh_score(h, r_id, t)
        if self.model_name == "transr":
            return self._transr_score(h, r_id, t)
        if self.model_name == "transd":
            return self._transd_score(h_id, r_id, t_id)
        return -torch.linalg.vector_norm(h + r - t, ord=1, dim=-1)

    def _split_complex(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a real tensor into real and imaginary halves.

        Parameters
        ----------
        values:
            Tensor with last dimension ``2 * dim``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Real and imaginary components.
        """

        return values[..., : self.dim], values[..., self.dim :]

    def _complex_score(
        self, h_id: torch.Tensor, r_id: torch.Tensor, t_id: torch.Tensor
    ) -> torch.Tensor:
        """Compute the ComplEx Hermitian product score.

        Parameters
        ----------
        h_id:
            Head entity IDs.
        r_id:
            Relation IDs.
        t_id:
            Tail entity IDs.

        Returns
        -------
        torch.Tensor
            ComplEx scores.
        """

        hr, hi = self._split_complex(self.ent(h_id))
        rr, ri = self._split_complex(self.rel(r_id))
        tr, ti = self._split_complex(self.ent(t_id))
        return (hr * rr * tr + hr * ri * ti + hi * rr * ti - hi * ri * tr).sum(dim=-1)

    def _analogy_score(
        self, h_id: torch.Tensor, r_id: torch.Tensor, t_id: torch.Tensor
    ) -> torch.Tensor:
        """Compute ANALOGY's normal-matrix bilinear score.

        Parameters
        ----------
        h_id:
            Head entity IDs.
        r_id:
            Relation IDs.
        t_id:
            Tail entity IDs.

        Returns
        -------
        torch.Tensor
            ANALOGY scores.
        """

        base = self._complex_score(h_id, r_id, t_id)
        h = self.ent(h_id)[:, : self.dim]
        r = self.rel(r_id)[:, : self.dim]
        t = self.ent(t_id)[:, : self.dim]
        return base + (h * r * t).sum(dim=-1)

    def _simple_score(
        self, h_id: torch.Tensor, r_id: torch.Tensor, t_id: torch.Tensor
    ) -> torch.Tensor:
        """Compute SimplE's averaged head/tail CP scores.

        Parameters
        ----------
        h_id:
            Head entity IDs.
        r_id:
            Relation IDs.
        t_id:
            Tail entity IDs.

        Returns
        -------
        torch.Tensor
            SimplE scores.
        """

        h_head = self.ent(h_id)[:, : self.dim]
        h_tail = self.ent_tail(h_id)
        t_head = self.ent(t_id)[:, : self.dim]
        t_tail = self.ent_tail(t_id)
        rel = self.rel(r_id)[:, : self.dim]
        inv_rel = self.rel_tail(r_id)
        forward = (h_head * rel * t_tail).sum(dim=-1)
        inverse = (t_head * inv_rel * h_tail).sum(dim=-1)
        return 0.5 * (forward + inverse)

    def _rotate_score(
        self, h_id: torch.Tensor, r_id: torch.Tensor, t_id: torch.Tensor
    ) -> torch.Tensor:
        """Compute RotatE's complex rotation distance score.

        Parameters
        ----------
        h_id:
            Head entity IDs.
        r_id:
            Relation IDs.
        t_id:
            Tail entity IDs.

        Returns
        -------
        torch.Tensor
            Negative complex distance scores.
        """

        hr, hi = self._split_complex(self.ent(h_id))
        tr, ti = self._split_complex(self.ent(t_id))
        phase = torch.tanh(self.rel(r_id)[:, : self.dim]) * torch.pi
        rr, ri = torch.cos(phase), torch.sin(phase)
        pred_r = hr * rr - hi * ri
        pred_i = hr * ri + hi * rr
        return -torch.stack((pred_r - tr, pred_i - ti), dim=-1).norm(dim=-1).sum(dim=-1)

    def _transh_score(self, h: torch.Tensor, r_id: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute TransH's hyperplane-projected translation score.

        Parameters
        ----------
        h:
            Head embeddings.
        r_id:
            Relation IDs.
        t:
            Tail embeddings.

        Returns
        -------
        torch.Tensor
            Negative L2 distance scores.
        """

        normal = F.normalize(self.rel_norm(r_id), dim=-1)
        r = self.rel(r_id)[:, : self.dim]
        hp = h - (h * normal).sum(dim=-1, keepdim=True) * normal
        tp = t - (t * normal).sum(dim=-1, keepdim=True) * normal
        return -torch.linalg.vector_norm(hp + r - tp, dim=-1)

    def _transr_score(self, h: torch.Tensor, r_id: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute TransR's relation-space projection score.

        Parameters
        ----------
        h:
            Head embeddings.
        r_id:
            Relation IDs.
        t:
            Tail embeddings.

        Returns
        -------
        torch.Tensor
            Negative L2 distance scores.
        """

        mat = self.rel_proj_mat(r_id).view(-1, self.dim, self.dim)
        hp = torch.bmm(mat, h.unsqueeze(-1)).squeeze(-1)
        tp = torch.bmm(mat, t.unsqueeze(-1)).squeeze(-1)
        r = self.rel(r_id)[:, : self.dim]
        return -torch.linalg.vector_norm(hp + r - tp, dim=-1)

    def _transd_score(
        self, h_id: torch.Tensor, r_id: torch.Tensor, t_id: torch.Tensor
    ) -> torch.Tensor:
        """Compute TransD's dynamic mapping-vector score.

        Parameters
        ----------
        h_id:
            Head entity IDs.
        r_id:
            Relation IDs.
        t_id:
            Tail entity IDs.

        Returns
        -------
        torch.Tensor
            Negative L2 distance scores.
        """

        h = self.ent(h_id)[:, : self.dim]
        t = self.ent(t_id)[:, : self.dim]
        hp = self.ent_proj(h_id)
        tp = self.ent_proj(t_id)
        rp = self.rel_proj(r_id)
        r = self.rel(r_id)[:, : self.dim]
        h_map = h + (h * hp).sum(dim=-1, keepdim=True) * rp
        t_map = t + (t * tp).sum(dim=-1, keepdim=True) * rp
        return -torch.linalg.vector_norm(h_map + r - t_map, dim=-1)


def _build(model_name: ModelName) -> OpenKEScorer:
    """Build a compact OpenKE scorer.

    Parameters
    ----------
    model_name:
        OpenKE model key.

    Returns
    -------
    OpenKEScorer
        Random-initialized scorer.
    """

    return OpenKEScorer(model_name)


def example_input() -> torch.Tensor:
    """Create a tiny batch of integer triples.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``(5, 3)``.
    """

    return torch.tensor([[0, 0, 1], [2, 1, 4], [5, 2, 3], [6, 3, 8], [9, 4, 10]])


def build_analogy() -> OpenKEScorer:
    """Build OpenKE ANALOGY.

    Returns
    -------
    OpenKEScorer
        ANALOGY scorer.
    """

    return _build("analogy")


def build_complex() -> OpenKEScorer:
    """Build OpenKE ComplEx.

    Returns
    -------
    OpenKEScorer
        ComplEx scorer.
    """

    return _build("complex")


def build_distmult() -> OpenKEScorer:
    """Build OpenKE DistMult.

    Returns
    -------
    OpenKEScorer
        DistMult scorer.
    """

    return _build("distmult")


def build_rescal() -> OpenKEScorer:
    """Build OpenKE RESCAL.

    Returns
    -------
    OpenKEScorer
        RESCAL scorer.
    """

    return _build("rescal")


def build_rotate() -> OpenKEScorer:
    """Build OpenKE RotatE.

    Returns
    -------
    OpenKEScorer
        RotatE scorer.
    """

    return _build("rotate")


def build_simple() -> OpenKEScorer:
    """Build OpenKE SimplE.

    Returns
    -------
    OpenKEScorer
        SimplE scorer.
    """

    return _build("simple")


def build_transd() -> OpenKEScorer:
    """Build OpenKE TransD.

    Returns
    -------
    OpenKEScorer
        TransD scorer.
    """

    return _build("transd")


def build_transe() -> OpenKEScorer:
    """Build OpenKE TransE.

    Returns
    -------
    OpenKEScorer
        TransE scorer.
    """

    return _build("transe")


def build_transh() -> OpenKEScorer:
    """Build OpenKE TransH.

    Returns
    -------
    OpenKEScorer
        TransH scorer.
    """

    return _build("transh")


def build_transr() -> OpenKEScorer:
    """Build OpenKE TransR.

    Returns
    -------
    OpenKEScorer
        TransR scorer.
    """

    return _build("transr")


MENAGERIE_ENTRIES = [
    ("OpenKE-Analogy", "build_analogy", "example_input", "2017", "graph/embedding"),
    ("OpenKE-ComplEx", "build_complex", "example_input", "2016", "graph/embedding"),
    ("OpenKE-DistMult", "build_distmult", "example_input", "2015", "graph/embedding"),
    ("OpenKE-RESCAL", "build_rescal", "example_input", "2011", "graph/embedding"),
    ("OpenKE-RotatE", "build_rotate", "example_input", "2019", "graph/embedding"),
    ("OpenKE-SimplE", "build_simple", "example_input", "2018", "graph/embedding"),
    ("OpenKE-TransD", "build_transd", "example_input", "2015", "graph/embedding"),
    ("OpenKE-TransE", "build_transe", "example_input", "2013", "graph/embedding"),
    ("OpenKE-TransH", "build_transh", "example_input", "2014", "graph/embedding"),
    ("OpenKE-TransR", "build_transr", "example_input", "2015", "graph/embedding"),
]
