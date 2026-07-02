# FAITHFUL PORT of https://github.com/wengong-jin/nips17-rexgen @ master (original framework: TensorFlow 0.x/1.x)
# Ported from USPTO/core-wln-global/{models.py,nntrain.py,ioutils.py,mol_graph.py} and
# utils/nn.py. The original repo (Jin, Coley, Green & Jaakkola, NeurIPS 2017, "Predicting
# Organic Reaction Outcomes with Weisfeiler-Lehman Network") is written against a very old
# TF API (Python-2 `xrange`, positional `tf.concat(axis, values)`, raw `tf.get_variable`
# session graphs) that cannot run under any installed TF -- and TF is not in the declared
# base-lib set for this project regardless. Every layer below is a direct 1:1 transcription
# of the real TF ops (same gates, same neighbor-masking sum-pool, same pairwise-attention
# reactivity head), only the tensor-framework plumbing (tf.Session/placeholders -> torch
# nn.Module/forward) changed.
#
# Two composed pieces, both from the real repo:
#   1. `rcnn_wl_last` (models.py) -- the WLN graph encoder: atom features are embedded,
#      then `depth` rounds of neighbor-gathered atom+bond message passing (masked sum over
#      up to `max_nb` neighbors) update each atom's WL-style hidden representation.
#   2. The core-wln-global reactivity head (nntrain.py lines 63-86) -- builds all-pairs atom
#      representations from the encoder output, folds in the "binary" pairwise
#      bond/component features (ioutils.get_bin_feature), and passes them through a
#      pairwise self-attention layer (att_hidden/att_score/att_context) before a final
#      pairwise reactivity-score projection.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"

# mol_graph.py: atom_fdim = len(elem_list) + 6 + 6 + 6 + 1, elem_list has 65 entries
ATOM_FDIM = 65 + 6 + 6 + 6 + 1  # = 82
BOND_FDIM = 6
MAX_NB = 10
# ioutils.py: binary_fdim = 4 + bond_fdim
BINARY_FDIM = 4 + BOND_FDIM  # = 10


class LinearND(nn.Module):
    """Port of utils/nn.py `linearND`: a Linear applied over the last dim of an
    arbitrary-rank tensor (TF's manual reshape-to-2D-matmul-reshape-back), with an
    optional bias (bias omitted when the real call site passes `init_bias=None`)."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)


class WLNEncoder(nn.Module):
    """Port of models.py `rcnn_wl_last`: WL-network graph convolution encoder."""

    def __init__(self, hidden_size=100, depth=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.atom_embedding = LinearND(
            ATOM_FDIM, hidden_size, bias=False
        )  # init_bias=None -> no bias

        # "WL" scope, reused across depth (real code: tf.variable_scope("WL", reuse=(i>0)))
        self.nei_atom = LinearND(hidden_size, hidden_size, bias=False)  # init_bias=None
        self.nei_bond = LinearND(BOND_FDIM, hidden_size, bias=False)  # init_bias=None
        self.self_atom = LinearND(hidden_size, hidden_size, bias=False)  # init_bias=None
        self.label_U2 = LinearND(hidden_size + BOND_FDIM, hidden_size)  # default bias
        self.label_U1 = LinearND(hidden_size * 2, hidden_size)  # default bias

    def forward(self, input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask):
        # atom_graph/bond_graph: (batch, n_atoms, max_nb) int64 neighbor indices (no
        # leading batch-index channel here -- gather is done per-batch-row via
        # torch.gather rather than TF's flattened gather_nd with a batch-idx column).
        atom_features = F.relu(self.atom_embedding(input_atom))
        batch_size, n_atoms, _ = atom_features.shape

        # mask_nei: (batch, n_atoms, max_nb, 1), 1 for real neighbors else 0
        nb_range = torch.arange(MAX_NB, device=num_nbs.device).view(1, 1, MAX_NB)
        mask_nei = (nb_range < num_nbs.unsqueeze(-1)).float().unsqueeze(-1)

        for _ in range(self.depth):
            fatom_nei = _gather_nei(atom_features, atom_graph)  # (batch, n_atoms, max_nb, hidden)
            fbond_nei = _gather_nei(input_bond, bond_graph)  # (batch, n_atoms, max_nb, bond_fdim)

            h_nei_atom = self.nei_atom(fatom_nei)
            h_nei_bond = self.nei_bond(fbond_nei)
            h_nei = h_nei_atom * h_nei_bond
            f_nei = (h_nei * mask_nei).sum(dim=2)  # masked sum over neighbors

            f_self = self.self_atom(atom_features)
            # layers.append(f_nei * f_self * node_mask) -- kept implicitly via `kernels`
            # being the *last* layer's value per real code (kernels = layers[-1]).

            l_nei = torch.cat([fatom_nei, fbond_nei], dim=-1)
            nei_label = F.relu(self.label_U2(l_nei))
            nei_label = (nei_label * mask_nei).sum(dim=2)
            new_label = torch.cat([atom_features, nei_label], dim=-1)
            new_label = self.label_U1(new_label)
            atom_features = F.relu(new_label)

        kernels = f_nei * f_self * node_mask
        fp = kernels.sum(dim=1)
        return kernels, fp


def _gather_nei(features, graph):
    # features: (batch, n_atoms, feat), graph: (batch, n_atoms, max_nb) neighbor index
    # into the n_atoms axis of `features` (per-batch-row gather, matching TF's
    # per-example gather_nd([batch_idx, neighbor_idx])).
    batch_size, n_atoms, max_nb = graph.shape
    feat_dim = features.shape[-1]
    idx = graph.reshape(batch_size, n_atoms * max_nb, 1).expand(-1, -1, feat_dim)
    gathered = torch.gather(features, 1, idx)
    return gathered.view(batch_size, n_atoms, max_nb, feat_dim)


class WLNReactionCenter(nn.Module):
    """Port of the core-wln-global reactivity head from nntrain.py (lines 63-86): WLN
    encoder followed by an all-pairs attention-gated reactivity scorer."""

    def __init__(self, hidden_size=100, depth=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = WLNEncoder(hidden_size=hidden_size, depth=depth)

        self.att_atom_feature = LinearND(hidden_size, hidden_size, bias=False)  # init_bias=None
        self.att_bin_feature = LinearND(BINARY_FDIM, hidden_size)  # default bias
        self.att_scores = LinearND(hidden_size, 1)  # default bias

        self.atom_feature = LinearND(hidden_size, hidden_size, bias=False)  # init_bias=None
        self.bin_feature = LinearND(BINARY_FDIM, hidden_size, bias=False)  # init_bias=None
        self.ctx_feature = LinearND(hidden_size, hidden_size)  # default bias

        self.scores = LinearND(hidden_size, 1)

    def forward(self, input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask, binary):
        node_mask = node_mask.unsqueeze(-1)
        atom_hiddens, _ = self.encoder(
            input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask
        )

        batch_size, n_atoms, hidden = atom_hiddens.shape
        atom_hiddens1 = atom_hiddens.view(batch_size, 1, n_atoms, hidden)
        atom_hiddens2 = atom_hiddens.view(batch_size, n_atoms, 1, hidden)
        atom_pair = atom_hiddens1 + atom_hiddens2  # (batch, n_atoms, n_atoms, hidden)

        att_hidden = F.relu(self.att_atom_feature(atom_pair) + self.att_bin_feature(binary))
        att_score = torch.sigmoid(self.att_scores(att_hidden))
        att_context = att_score * atom_hiddens1
        att_context = att_context.sum(dim=2)  # (batch, n_atoms, hidden)

        att_context1 = att_context.view(batch_size, 1, n_atoms, hidden)
        att_context2 = att_context.view(batch_size, n_atoms, 1, hidden)
        att_pair = att_context1 + att_context2

        pair_hidden = (
            self.atom_feature(atom_pair) + self.bin_feature(binary) + self.ctx_feature(att_pair)
        )
        pair_hidden = F.relu(pair_hidden)
        pair_hidden = pair_hidden.view(batch_size, n_atoms * n_atoms, hidden)

        score = self.scores(pair_hidden).squeeze(-1)  # (batch, n_atoms * n_atoms)
        return score


# --- staging harness ---
def build_wln_reaction_center():
    return WLNReactionCenter(hidden_size=16, depth=2)


def example_input_wln_reaction_center():
    torch.manual_seed(0)
    batch_size, n_atoms = 2, 6
    input_atom = torch.randn(batch_size, n_atoms, ATOM_FDIM)
    input_bond = torch.randn(batch_size, n_atoms, BOND_FDIM)
    # neighbor index graphs: each atom "sees" the next two atoms (toy ring topology)
    atom_graph = torch.zeros(batch_size, n_atoms, MAX_NB, dtype=torch.long)
    bond_graph = torch.zeros(batch_size, n_atoms, MAX_NB, dtype=torch.long)
    num_nbs = torch.full((batch_size, n_atoms), 2, dtype=torch.long)
    for i in range(n_atoms):
        atom_graph[:, i, 0] = (i + 1) % n_atoms
        atom_graph[:, i, 1] = (i - 1) % n_atoms
        bond_graph[:, i, 0] = i
        bond_graph[:, i, 1] = (i - 1) % n_atoms
    node_mask = torch.ones(batch_size, n_atoms)
    binary = torch.randn(batch_size, n_atoms, n_atoms, BINARY_FDIM)
    return (input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask, binary)


MENAGERIE_ENTRIES = [
    (
        "WLN_Reaction_Center",
        "build_wln_reaction_center",
        "example_input_wln_reaction_center",
        "2017",
        "ported-pytorch",
    ),
]
