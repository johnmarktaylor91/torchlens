"""Chemprop D-MPNN: directed message-passing neural network on molecular graphs.

Yang et al., "Analyzing Learned Molecular Representations for Property Prediction",
JCIM 2019. https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237
Heid et al., Chemprop (JCIM 2024). Source: https://github.com/chemprop/chemprop

The D-MPNN's distinctive contribution is passing messages along **directed bonds**
(edges) rather than atoms, which reduces over-smoothing / message-tottering.
Faithful message-passing equations (Chemprop):

  initial:  m_{i->j}^0 = ReLU(W_i [a_j ; b_{i->j}])
  update:   m_{j->i}^{t+1} = ReLU(m_{j->i}^0 + W_h * sum_{k in N(j)\\{i}} m_{k->j}^t)
  readout:  a-state h_v = ReLU(W_o [a_v ; sum_{k in N(v)} m_{k->v}^T])
            molecule vector = mean_v h_v  ->  2-layer FFN -> property

Published hyperparameters: bond-message width 300, depth = 3 message-passing steps,
FFN = 2 hidden layers of 300. The "reverse-message" subtraction term enforces the
directed (i->j excludes j->i) aggregation.

This is a faithful reimplementation of the exact directed message passing + readout
+ FFN. The molecular graph (which is the real model's ``BatchMolGraph`` input) is
represented here as fixed node/edge tensors the forward consumes: atom features,
bond features, and the bond connectivity (a2b / b2a / b2revb index tensors) for a
small random molecular graph. Random init, forward-only. Widths reduced so the
unrolled atlas graph renders quickly.

Faithful-core simplifications (honest, not lies):
  - hidden width 64 and depth 3 on a small 6-atom / 6-bond graph (vs width 300 on
    real molecules); the directed message-passing math is identical.
  - the input ``BatchMolGraph`` is materialized as fixed atom/bond/index tensors
    (a small fixed molecular graph), so the example input is a tuple of small
    tensors the forward consumes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Small fixed molecular graph (a 6-membered ring) as directed bonds.
# Each undirected bond becomes two directed bonds (i->j and j->i).
_RING_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
_ATOM_FDIM = 16
_BOND_FDIM = 8


class DMPNNEncoder(nn.Module):
    """Directed message-passing encoder (Chemprop MPNEncoder)."""

    def __init__(
        self,
        atom_fdim: int = _ATOM_FDIM,
        bond_fdim: int = _BOND_FDIM,
        hidden: int = 64,
        depth: int = 3,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.hidden = hidden
        # W_i maps [atom_feat ; bond_feat] -> hidden (directed-bond input)
        self.W_i = nn.Linear(atom_fdim + bond_fdim, hidden, bias=False)
        self.W_h = nn.Linear(hidden, hidden, bias=False)
        self.W_o = nn.Linear(atom_fdim + hidden, hidden)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.0)

    def forward(
        self,
        f_atoms: torch.Tensor,  # (n_atoms, atom_fdim)
        f_bonds: torch.Tensor,  # (n_bonds, atom_fdim + bond_fdim) directed-bond inputs
        a2b: torch.Tensor,  # (n_atoms, max_deg) incoming directed bonds per atom (+1 pad idx 0)
        b2a: torch.Tensor,  # (n_bonds,) source atom of each directed bond
        b2revb: torch.Tensor,  # (n_bonds,) reverse directed bond index
    ) -> torch.Tensor:
        # initial directed-bond hidden state
        input_bond = self.W_i(f_bonds)  # (n_bonds, hidden)
        message = self.act(input_bond)

        for _ in range(self.depth - 1):
            # aggregate incoming messages per atom, then map back to bonds,
            # subtracting the reverse message (directed aggregation).
            # message padded with a zero row at index 0 for gather.
            padded = torch.cat([message.new_zeros(1, self.hidden), message], dim=0)
            nei_message = padded[a2b]  # (n_atoms, max_deg, hidden)
            a_message = nei_message.sum(dim=1)  # (n_atoms, hidden)
            rev_message = message[b2revb]  # (n_bonds, hidden)
            message = a_message[b2a] - rev_message  # directed: exclude reverse bond
            message = self.act(input_bond + self.W_h(message))
            message = self.dropout(message)

        # atom readout: sum incoming bond messages per atom
        padded = torch.cat([message.new_zeros(1, self.hidden), message], dim=0)
        nei_message = padded[a2b].sum(dim=1)  # (n_atoms, hidden)
        a_input = torch.cat([f_atoms, nei_message], dim=1)
        atom_hiddens = self.dropout(self.act(self.W_o(a_input)))  # (n_atoms, hidden)
        # molecule vector = mean over atoms
        return atom_hiddens.mean(dim=0, keepdim=True)  # (1, hidden)


class MoleculeModel(nn.Module):
    """Chemprop MoleculeModel: D-MPNN encoder + 2-hidden-layer readout FFN."""

    def __init__(self, hidden: int = 64, ffn_hidden: int = 64, n_tasks: int = 1) -> None:
        super().__init__()
        self.encoder = DMPNNEncoder(hidden=hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, n_tasks),
        )

    def forward(self, graph: tuple) -> torch.Tensor:
        f_atoms, f_bonds, a2b, b2a, b2revb = graph
        mol_vec = self.encoder(f_atoms, f_bonds, a2b, b2a, b2revb)
        return self.ffn(mol_vec)


def _build_molgraph():
    """Materialize a small fixed molecular graph as Chemprop-style index tensors."""
    n_atoms = 6
    directed = []  # (src, dst)
    for i, j in _RING_EDGES:
        directed.append((i, j))
        directed.append((j, i))
    n_bonds = len(directed)
    b2a = torch.tensor([d[0] for d in directed], dtype=torch.long)  # source atom
    # reverse bond: directed list is [e, rev(e), ...] so revb pairs are 2k<->2k+1
    revb = []
    for k in range(0, n_bonds, 2):
        revb.append(k + 1)
        revb.append(k)
    b2revb = torch.tensor(revb, dtype=torch.long)
    # a2b: incoming directed bonds per atom (bonds whose dst == atom), 1-indexed, 0=pad
    incoming = {a: [] for a in range(n_atoms)}
    for bidx, (src, dst) in enumerate(directed):
        incoming[dst].append(bidx + 1)  # +1 for pad-at-0 convention
    max_deg = max(len(v) for v in incoming.values())
    a2b = torch.zeros(n_atoms, max_deg, dtype=torch.long)
    for a, bonds in incoming.items():
        for k, b in enumerate(bonds):
            a2b[a, k] = b
    f_atoms = torch.randn(n_atoms, _ATOM_FDIM)
    # directed-bond input = [atom_feat(src) ; bond_feat]
    bond_feats = torch.randn(n_bonds, _BOND_FDIM)
    f_bonds = torch.cat([f_atoms[b2a], bond_feats], dim=1)
    return f_atoms, f_bonds, a2b, b2a, b2revb


def build() -> nn.Module:
    return MoleculeModel()


def example_input() -> tuple:
    """Fixed small molecular graph as (f_atoms, f_bonds, a2b, b2a, b2revb) tensors."""
    return _build_molgraph()


MENAGERIE_ENTRIES = [
    (
        "Chemprop D-MPNN (directed bond message-passing)",
        "build",
        "example_input",
        "2019",
        "DC",
    ),
]
