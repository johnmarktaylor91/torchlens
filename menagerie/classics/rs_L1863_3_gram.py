# FAITHFUL PORT of mp2893/gram @ master (gram.py, original framework:
# Theano, using Python-2-only `cPickle`/`iteritems`/`xrange` -- cannot be
# installed/run in the modern base env; Theano is unmaintained and its
# public releases predate Python 3 wheels for current interpreters).
# Transcribes the GRAM forward graph mechanism-for-mechanism from
# `init_params` / `generate_attention` / `gru_layer` / `softmax_layer` /
# `build_model` in the real code:
#   - `W_emb`: a single embedding table of size
#     (inputDimSize + numAncestors, embDimSize) shared by leaf codes AND
#     their ontology ancestors (GRAM's key idea: represent a leaf medical
#     code as an attention-weighted combination of itself + its ancestors
#     in the ICD9/CCS hierarchy, at every level of the tree).
#   - `generate_attention`: per level, concatenate the leaf embedding with
#     each ancestor embedding, pass through a `tanh(W_attention @ . +
#     b_attention)` MLP, score with `v_attention`, softmax over the
#     ancestor axis -- an additive (Bahdanau-style) attention over the
#     code's own ontology path.
#   - per-level embeddings are summed (`T.concatenate(embList, axis=0)`
#     over levels) into one code-embedding table, then a multi-hot visit
#     vector `x` is projected via `tanh(x @ emb)` into the GRU input space
#     (this is the visit-level embedding fed to the recurrent encoder).
#   - `gru_layer`: a GRU written out by hand from its update/reset/candidate
#     gate equations (`stepFn`); reproduced here as an explicit custom GRU
#     cell (not `nn.GRU`) to keep the exact gate wiring
#     (`r,z,h_tilde` slices of a single fused `Wx = x @ W_gru + b_gru`
#     matmul, matching `_slice`/`stepFn`).
#   - `softmax_layer`: `Linear` + softmax over the class axis at every
#     timestep (per-visit multi-label diagnosis-code prediction).
# Training-time-only machinery (adadelta optimizer, cross-entropy loss,
# masking/padding utilities, tree-building from pickled ICD9 ontology files)
# is intentionally not ported; the embedding-attention + GRU + softmax
# forward graph is preserved in full.
"""GRAM: Graph-based Attention Model for medical code representation learning.

Choi et al., KDD 2017, "GRAM: Graph-based Attention Model for Healthcare
Representation Learning" (arxiv:1611.07012). Each ICD9/CCS diagnosis code is
embedded as an attention-weighted sum over itself and its ancestors in the
clinical ontology tree (5 levels including the root), giving rare/novel
codes robust representations borrowed from more frequent ancestor concepts.
Per-visit multi-hot code vectors are projected into this attention-derived
embedding space and encoded with a GRU for sequential diagnosis prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class OntologyAttentionEmbedding(nn.Module):
    """One ontology level's `generate_attention` + ancestor-weighted embedding lookup.

    ``leaves``/``ancestors`` are fixed integer index tensors: for every leaf
    code visible at this ontology level, ``leaves`` repeats that code's index
    once per ancestor slot and ``ancestors`` holds the corresponding ancestor
    code index (including the code itself and the artificial root), mirroring
    `build_tree` in the original.
    """

    def __init__(
        self,
        emb_table: nn.Embedding,
        attention_dim: int,
        leaves: torch.Tensor,
        ancestors: torch.Tensor,
    ):
        super().__init__()
        self.emb_table = emb_table
        emb_dim = emb_table.embedding_dim
        self.attention = nn.Linear(emb_dim * 2, attention_dim)
        self.v_attention = nn.Parameter(torch.empty(attention_dim).uniform_(-0.1, 0.1))
        self.register_buffer("leaves", leaves)
        self.register_buffer("ancestors", ancestors)

    def forward(self) -> torch.Tensor:
        # (n_leaves, n_ancestors, emb_dim)
        leaf_emb = self.emb_table(self.leaves)
        anc_emb = self.emb_table(self.ancestors)
        attn_input = torch.cat([leaf_emb, anc_emb], dim=-1)
        mlp_out = torch.tanh(self.attention(attn_input))
        pre_attention = mlp_out @ self.v_attention
        attention = F.softmax(pre_attention, dim=-1)  # over ancestor axis
        weighted = anc_emb * attention.unsqueeze(-1)
        return weighted.sum(dim=1)  # (n_leaves, emb_dim)


class GRAMGRUCell(nn.Module):
    """Hand-rolled GRU cell matching `gru_layer`'s `stepFn` gate wiring exactly."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w_gru = nn.Linear(input_dim, 3 * hidden_dim, bias=True)
        self.u_gru = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)

    def forward(self, x_emb: torch.Tensor) -> torch.Tensor:
        # x_emb: (batch, time, input_dim) -> returns (batch, time, hidden_dim)
        batch, timesteps, _ = x_emb.shape
        h = x_emb.new_zeros(batch, self.hidden_dim)
        wx_all = self.w_gru(x_emb)  # (batch, time, 3*hidden_dim)
        outputs = []
        hd = self.hidden_dim
        for t in range(timesteps):
            wx = wx_all[:, t, :]
            uh = self.u_gru(h)
            r = torch.sigmoid(wx[:, 0:hd] + uh[:, 0:hd])
            z = torch.sigmoid(wx[:, hd : 2 * hd] + uh[:, hd : 2 * hd])
            h_tilde = torch.tanh(wx[:, 2 * hd : 3 * hd] + r * uh[:, 2 * hd : 3 * hd])
            h = z * h + (1.0 - z) * h_tilde
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class GRAM(nn.Module):
    """Full GRAM forward graph: ontology-attention embedding -> GRU -> softmax."""

    def __init__(
        self,
        input_dim_size: int,
        num_ancestors: int,
        emb_dim_size: int,
        hidden_dim_size: int,
        attention_dim_size: int,
        num_class: int,
        level_trees: list,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        vocab_size = input_dim_size + num_ancestors
        self.emb_table = nn.Embedding(vocab_size, emb_dim_size)
        self.levels = nn.ModuleList(
            [
                OntologyAttentionEmbedding(self.emb_table, attention_dim_size, leaves, ancestors)
                for leaves, ancestors in level_trees
            ]
        )
        self.input_dim_size = input_dim_size
        self.gru = GRAMGRUCell(emb_dim_size, hidden_dim_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output = nn.Linear(hidden_dim_size, num_class)

    def code_embedding(self) -> torch.Tensor:
        # concatenate per-level attention-weighted embeddings over the leaf axis,
        # matching `T.concatenate(embList, axis=0)` in `build_model`
        return torch.cat([level() for level in self.levels], dim=0)[: self.input_dim_size]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, input_dim_size) multi-hot visit vectors
        emb = self.code_embedding()  # (input_dim_size, emb_dim_size)
        x_emb = torch.tanh(x @ emb)  # (batch, time, emb_dim_size)
        hidden = self.gru(x_emb)  # (batch, time, hidden_dim_size)
        hidden = self.dropout(hidden)
        logits = self.output(hidden)  # (batch, time, num_class)
        y_hat = F.softmax(logits, dim=-1)
        return y_hat


# ---------------------------------------------------------------------------
# Staging build/example helpers. A tiny synthetic 2-level ontology
# (leaf-level + artificial-root level) stands in for the 5-level CCS/ICD9
# tree the real code loads from pickled files, scaled down for fast tracing.
# ---------------------------------------------------------------------------


def _make_toy_ontology(input_dim_size: int, num_ancestors: int):
    # `build_tree` in the original returns `leaves`/`ancestors` of shape
    # (n_leaves, ancSize), where every leaf code is padded to the same
    # per-level ancestor-path length (`leaves.append([k] * ancSize)`).
    # level 1: each leaf's own path is [itself, parent-slot, root-slot].
    parent_idx = input_dim_size
    root_idx = input_dim_size + num_ancestors - 1
    leaves_l1 = torch.arange(input_dim_size).unsqueeze(1).repeat(1, 3)  # (n_leaves, 3)
    anc_l1 = torch.stack(
        [
            torch.arange(input_dim_size),
            torch.full((input_dim_size,), parent_idx),
            torch.full((input_dim_size,), root_idx),
        ],
        dim=1,
    )  # (n_leaves, 3)

    # level 2: every leaf attends to just the shared root ancestor slot.
    leaves_l2 = torch.arange(input_dim_size).unsqueeze(1)  # (n_leaves, 1)
    anc_l2 = torch.full((input_dim_size, 1), root_idx)

    return [(leaves_l1, anc_l1), (leaves_l2, anc_l2)]


def build_gram():
    input_dim_size = 20
    num_ancestors = 4
    level_trees = _make_toy_ontology(input_dim_size, num_ancestors)
    model = GRAM(
        input_dim_size=input_dim_size,
        num_ancestors=num_ancestors,
        emb_dim_size=16,
        hidden_dim_size=24,
        attention_dim_size=12,
        num_class=input_dim_size,
        level_trees=level_trees,
        dropout_rate=0.0,
    )
    model.eval()
    return model


def example_input_gram():
    torch.manual_seed(0)
    batch, timesteps, input_dim_size = 2, 5, 20
    x = (torch.rand(batch, timesteps, input_dim_size) > 0.85).float()
    return (x,)


MENAGERIE_ENTRIES = [
    ("GRAM", "build_gram", "example_input_gram", 2017, "ported-pytorch"),
]
