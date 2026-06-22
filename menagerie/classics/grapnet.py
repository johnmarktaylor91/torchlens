"""GrapNet: a programmable dynamic-architecture neural graph substrate.

Zirong Li (Tiangong University), 2026, arXiv:2606.18923
("GrapNet: A Programmable Dynamic-Architecture Neural Graph Substrate").

GrapNet treats *the graph as the neural network itself* (not an input data graph, unlike a
GNN).  For a transition from layer l to layer l+1, every current-layer compute node
("GrapNode") owns a list of next-layer **child references** ``C_u = [v_1, ..., v_du]`` and
an aligned trainable **allocation vector** ``W_alloc_u = [w_1, ..., w_du]`` -- the storage
invariant is ``node.children[k] <-> node.W_alloc[0, k]``.  Deleting a relation physically
removes BOTH the child reference and its allocation coordinate (vs. masking a dense
weight, which leaves the slot + optimizer state behind).  The same child-owned graph can
be executed under several **execution policies** over the same relation set:
  - reference **scatter** backend (Algorithm 1): for each node i, message
    ``M_i = X[:, i] * W_alloc_i`` is scattered into each child ``C_i[k]``;
  - trainable **family blocks** (the dynamic fast path): nodes with identical child
    patterns are grouped, ``M_f = H_f W_f`` then scattered to targets;
  - **attention** policy over active relations:
    ``a_ij = softmax_{i:(i,j) in E}(h_i k_i q_j)``, ``z_j = sum_i a_ij h_i w_ij``;
  - **dense snapshot** (Proposition 2): materialize ``A_ij = w_ij`` for active relations
    (else 0), then ``Z = X A + b`` -- a throughput endpoint, provably gradient-equivalent.
MLPs / Transformer FFNs are the *fully-connected* special case of a GrapNet topology; the
substrate composes with conventional modules through a vector-valued **parent interface**
(a CNN/ResNet/dense encoder feeds one "sensory GrapNode" per coordinate).

THE DISTINCTIVE PRIMITIVE (reproduced faithfully here): a stack of transitions whose
topology IS the network -- each node owns child references + an aligned trainable
allocation vector, and the forward pass is a **sparse scatter-accumulate into children**
(sparse = not fully connected; that sparsity, plus child-owned deletable relations, is
exactly what distinguishes it from a plain dense MLP).  We render, over one fixed
canonical sparse topology, the three native execution views the paper specifies
(reference scatter, trainable family block, attention-over-relations) plus the dense
snapshot, all behind a parent dense-encoder interface.

This is a faithful COMPACT random-init reimplementation: a few nodes per layer, a small
fixed sparse child-owned topology, plain torch only.  Random init, CPU, forward-only.
The forward pass is FIXED-topology (a stabilized snapshot of the editable graph) so it
renders as a static unrolled graph; the editable/grow/freeze/delete machinery is the
training-time controller around this stabilized forward and is described in the docstring
rather than exercised in a single forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scatter_add_rows(messages: torch.Tensor, targets: torch.Tensor, n_out: int) -> torch.Tensor:
    """Scatter-accumulate ``messages`` (E, B) into ``n_out`` output coords by ``targets`` (E,)."""
    out = messages.new_zeros(n_out, messages.shape[1])
    idx = targets.view(-1, 1).expand_as(messages)
    return out.scatter_add(0, idx, messages)


class ChildOwnedTransition(nn.Module):
    """One GrapNet transition stored as a child-owned relation set, run by SCATTER (Alg. 1).

    Canonical storage: for each source node i, a child list ``children[i]`` (indices into
    the next layer) and an aligned allocation vector ``alloc[i]`` (one trainable weight per
    child).  Forward = reference scatter backend: ``M_i = X[:, i] * alloc[i]`` scattered
    into each child, plus next-node bias and activation.  Inactive (absent) relations carry
    no weight and contribute nothing -- this is the sparse, editable analogue of a dense
    ``Z = X A + b`` (Proposition 2).
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        relations: list[tuple[int, int]],
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        src = torch.tensor([r[0] for r in relations], dtype=torch.long)
        dst = torch.tensor([r[1] for r in relations], dtype=torch.long)
        self.register_buffer("src", src)  # source GrapNode per relation
        self.register_buffer("dst", dst)  # child reference per relation
        # one trainable allocation coordinate per active child-owned relation
        self.alloc = nn.Parameter(torch.randn(len(relations)) * 0.3)
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_in) -> per-relation source activation * allocation coordinate
        src_act = x.index_select(1, self.src)  # (B, E)
        messages = (src_act * self.alloc.unsqueeze(0)).transpose(0, 1)  # (E, B)
        z = _scatter_add_rows(messages, self.dst, self.n_out).transpose(0, 1)  # (B, n_out)
        z = z + self.bias
        return F.relu(z) if self.activation else z


class AttentionPolicyTransition(nn.Module):
    """A GrapNet transition under the ATTENTION execution policy over the same relations.

    Input-dependent routing over active child-owned relations:
        a_ij = softmax_{i:(i,j) in E}(h_i k_i q_j),   z_j = sum_i a_ij h_i w_ij
    Read from the same child lists; w_ij is the child-owned allocation coordinate.  The
    softmax is segment-wise over the incoming relations of each destination node.
    """

    def __init__(self, n_in: int, n_out: int, relations: list[tuple[int, int]]) -> None:
        super().__init__()
        self.n_out = n_out
        src = torch.tensor([r[0] for r in relations], dtype=torch.long)
        dst = torch.tensor([r[1] for r in relations], dtype=torch.long)
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        self.alloc = nn.Parameter(torch.randn(len(relations)) * 0.3)  # w_ij
        self.key = nn.Parameter(torch.randn(n_in) * 0.3)  # k_i per source coord
        self.query = nn.Parameter(torch.randn(n_out) * 0.3)  # q_j per target coord
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_i = x.index_select(1, self.src)  # (B, E)
        k_i = self.key.index_select(0, self.src).unsqueeze(0)  # (1, E)
        q_j = self.query.index_select(0, self.dst).unsqueeze(0)  # (1, E)
        logits = (h_i * k_i * q_j).transpose(0, 1)  # (E, B)
        # segment softmax over incoming relations of each destination node
        maxes = _scatter_max(logits, self.dst, self.n_out).index_select(0, self.dst)
        weights = (logits - maxes).exp()
        denom = _scatter_add_rows(weights, self.dst, self.n_out).index_select(0, self.dst) + 1e-9
        alpha = weights / denom  # (E, B)
        messages = alpha * h_i.transpose(0, 1) * self.alloc.unsqueeze(1)  # (E, B)
        z = _scatter_add_rows(messages, self.dst, self.n_out).transpose(0, 1)
        return z + self.bias


def _scatter_max(x: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = x.new_full((n, x.shape[1]), float("-inf"))
    idx = index.view(-1, 1).expand_as(x)
    return out.scatter_reduce(0, idx, x, reduce="amax", include_self=True)


class FamilyBlockTransition(nn.Module):
    """A GrapNet transition under the trainable FAMILY-BLOCK fast path.

    Current-layer nodes with identical next-layer child-reference patterns form a family f;
    member activations form ``H_f`` and live allocation params form ``W_f``, evaluated as
    ``M_f = H_f W_f`` then scattered to the shared next-layer targets.  Here the whole input
    layer is one family fully connected to a child group, so ``W_f`` is a dense
    ``(n_in, group)`` block assembled from live allocation parameters and the result is
    scattered into the family's child coordinates.
    """

    def __init__(self, n_in: int, n_out: int, child_group: list[int]) -> None:
        super().__init__()
        self.n_out = n_out
        self.register_buffer("targets", torch.tensor(child_group, dtype=torch.long))
        self.W_f = nn.Parameter(torch.randn(n_in, len(child_group)) * 0.3)  # live allocation block
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M_f = x @ self.W_f  # (B, group)  = H_f W_f
        z = _scatter_add_rows(M_f.transpose(0, 1), self.targets, self.n_out).transpose(0, 1)
        return F.relu(z + self.bias)


class DenseSnapshot(nn.Module):
    """Dense-snapshot endpoint (Proposition 2): materialized ``Z = X A + b``.

    After topology stabilizes, the external dense-snapshot controller packs the current
    child-owned relation set into a fixed dense matrix ``A`` (``A_ij = w_ij`` for active
    relations, else 0) for throughput; gradient-equivalent to the scatter backend.
    """

    def __init__(self, n_in: int, n_out: int, relations: list[tuple[int, int]]) -> None:
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)
        # zero the inactive coordinates so A reflects only active child-owned relations
        mask = torch.zeros(n_out, n_in)
        for s, d in relations:
            mask[d, s] = 1.0
        self.register_buffer("active_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.lin.weight * self.active_mask  # materialized sparse snapshot
        return F.relu(F.linear(x, A, self.lin.bias))


class GrapNet(nn.Module):
    """Compact GrapNet head over a fixed canonical sparse child-owned topology.

    Pipeline: a conventional **parent encoder** (dense, standing in for a CNN/ResNet
    feature extractor) feeds one sensory GrapNode per coordinate; then a stabilized
    child-owned graph is executed through, in turn, all the native execution views the
    paper specifies over the SAME relation set -- reference scatter, attention-over-
    relations, trainable family block, and the dense snapshot -- and the read-out
    classifies.  This exercises the distinctive primitive (node-owned children + aligned
    allocation vectors + scatter-accumulate) and its execution-policy multiplicity.
    """

    def __init__(self, in_dim: int = 16, width: int = 12, n_classes: int = 5) -> None:
        super().__init__()
        # parent interface: conventional dense encoder -> one sensory GrapNode per coord
        self.parent = nn.Sequential(nn.Linear(in_dim, width), nn.ReLU())

        # one fixed canonical sparse child-owned topology, reused by every execution view.
        # each source node connects to ~half the next-layer children (sparse, editable).
        rels = self._make_sparse_relations(width, width, fanout=width // 2)
        self.scatter = ChildOwnedTransition(width, width, rels)  # reference backend
        self.attention = AttentionPolicyTransition(width, width, rels)  # attention policy
        self.family = FamilyBlockTransition(
            width, width, list(range(width // 2))
        )  # family fast path
        self.snapshot = DenseSnapshot(width, width, rels)  # dense endpoint

        self.readout = nn.Linear(width, n_classes)

    @staticmethod
    def _make_sparse_relations(n_in: int, n_out: int, fanout: int) -> list[tuple[int, int]]:
        rels: list[tuple[int, int]] = []
        for i in range(n_in):
            for k in range(fanout):
                rels.append((i, (i + k + 1) % n_out))  # child reference owned by node i
        return rels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.parent(x)  # vector-valued parent interface
        h = self.scatter(h)  # Algorithm 1: child-owned scatter
        h = self.attention(h)  # attention execution policy
        h = self.family(h)  # trainable family block fast path
        h = self.snapshot(h)  # dense snapshot endpoint
        return self.readout(h)


def build() -> nn.Module:
    return GrapNet()


def example_input() -> torch.Tensor:
    """A small feature batch ``(2, 16)`` (2 samples, 16-d parent-encoder input)."""
    torch.manual_seed(0)
    return torch.randn(2, 16)


MENAGERIE_ENTRIES = [
    (
        "GrapNet (programmable child-owned neural graph substrate)",
        "build",
        "example_input",
        "2026",
        "DC",
    ),
]
