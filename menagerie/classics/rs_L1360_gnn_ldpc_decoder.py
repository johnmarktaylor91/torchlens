# FAITHFUL PORT of https://github.com/NVlabs/gnn-decoder @ 26630040 (original framework: TensorFlow/Keras
#   via the Sionna link-level simulator)
#
# Ported file: gnn.py -> `MLP`, `GNN_BP`, `UpdateEmbeddings` (S. Cammerer, J. Hoydis, F. Ait Aoudia,
# A. Keller, "Graph Neural Networks for Channel Decoding", arXiv:2207.14742, 2022).
#
# The repo's `GNN_BP` decoder is a fully differentiable graph neural network that learns a
# generalized message-passing (belief-propagation-style) algorithm over the Tanner graph of an
# error-correcting code's parity-check matrix, for LDPC / BCH channel decoding. The core decoder
# class (`GNN_BP`/`UpdateEmbeddings`/`MLP`) only depends on TensorFlow/Keras and numpy -- no
# `sionna`-specific ops appear inside `call()`/`build()` -- but the repo's `pip install sionna`
# stack (used only for the surrounding end-to-end channel/encoder/mapper simulation and for
# `tf.ragged.constant` message routing) is not a base-env package, and the file is TF/Keras (not
# torch). Per the rung ladder this is a RUNG-3 FAITHFUL PORT: every op of `GNN_BP.build`/`call`,
# `UpdateEmbeddings.build`/`call`, and `MLP.build`/`call` is transcribed faithfully into base-env
# torch (Keras `Dense` -> `nn.Linear`; `tf.ragged.constant` per-vertex edge-index lists -> a plain
# Python list-of-index-tensors used identically for `index_select`+segment-reduce gather; the
# einsum-free reduce_{sum,mean,max,min} aggregation over incoming per-edge messages is reproduced
# with the equivalent `torch.index_add_`/`scatter_reduce` semantics). `use_attributes` (optional
# trainable per-node/per-edge attribute vectors, off by default in the original) is preserved.
#
# Original TF/Keras call flow (see gnn.py):
#   h_vn = Dense(num_embed_dims)(llr[..., None])                 # VN embedding init
#   h_cn = zeros([batch, num_cn, num_embed_dims])                # CN embedding init
#   for i in range(num_iter):
#       h_cn = UpdateEmbeddings(...)(h_vn, h_cn)   # "from VN to CN" (edges flipped)
#       h_vn = UpdateEmbeddings(...)(h_cn, h_vn)   # "from CN to VN"
#   llr_hat = squeeze(Dense(1)(h_vn), axis=-1)
#
# UpdateEmbeddings.call(h_from, h_to):
#   features = concat([gather(h_from, from_ind, axis=1), gather(h_to, to_ind, axis=1)], axis=-1)
#   messages = msg_mlp(features)                                  # per-edge MLP
#   m[v] = reduce_{sum,mean,max,min} over messages of edges incident to vertex v (via gather_ind)
#   h_to_new = embed_mlp(concat([m, h_to], axis=-1))              # per-vertex update MLP

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class MLP(nn.Module):
    """Faithful port of gnn.MLP (a simple Keras `Dense`-stack Layer)."""

    def __init__(self, in_dim, units, activations, use_bias):
        super().__init__()
        layers = []
        prev = in_dim
        for u, act, ub in zip(units, activations, use_bias):
            layers.append(nn.Linear(prev, u, bias=ub))
            if act == "relu":
                layers.append(nn.ReLU())
            elif act == "tanh":
                layers.append(nn.Tanh())
            elif act is None:
                pass
            else:
                raise ValueError(f"unsupported activation {act!r}")
            prev = u
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class UpdateEmbeddings(nn.Module):
    """Faithful port of gnn.UpdateEmbeddings: computes per-edge messages between the
    "from" and "to" vertex sets of a bipartite Tanner-graph edge list, then aggregates
    incoming messages at each "to" vertex and updates its embedding via a second MLP.
    """

    def __init__(
        self,
        num_msg_dims,
        num_hidden_units,
        num_mlp_layers,
        from_to_ind,
        gather_ind,
        reduce_op="sum",
        activation="relu",
        use_attributes=False,
        node_attribute_dims=0,
        msg_attribute_dims=0,
        use_bias=False,
    ):
        super().__init__()
        self._num_msg_dims = num_msg_dims
        self._num_hidden_units = num_hidden_units
        self._num_mlp_layers = num_mlp_layers
        # from_to_ind: [num_edges, 2] numpy array; column 0 = "from" vertex id, column 1 = "to"
        # vertex id, one row per edge (mirrors the Tanner-graph edge list `self._edges`).
        self.register_buffer("_from_ind", torch.as_tensor(from_to_ind[:, 0], dtype=torch.long))
        self.register_buffer("_to_ind", torch.as_tensor(from_to_ind[:, 1], dtype=torch.long))
        # gather_ind: ragged list (one entry per "to" vertex) of edge indices incident to that
        # vertex -- mirrors tf.ragged.constant(cn_edges) / tf.ragged.constant(vn_edges).
        self._gather_ind = [torch.as_tensor(np.asarray(g), dtype=torch.long) for g in gather_ind]
        self._reduce_op = reduce_op
        self._activation = activation
        self._use_attributes = use_attributes
        self._node_attribute_dims = node_attribute_dims
        self._msg_attribute_dims = msg_attribute_dims
        self._use_bias = use_bias
        self._built = False

        if self._use_attributes:
            num_nodes = len(self._gather_ind)
            num_edges = from_to_ind.shape[0]
            self._g_node = nn.Parameter(torch.zeros(num_nodes, node_attribute_dims))
            self._g_msg = nn.Parameter(torch.zeros(num_edges, msg_attribute_dims))

    def _build(self, num_embed_dims):
        units = [self._num_hidden_units] * (self._num_mlp_layers - 1) + [self._num_msg_dims]
        activations = [self._activation] * (self._num_mlp_layers - 1) + [None]
        use_bias = [self._use_bias] * self._num_mlp_layers
        msg_in_dim = 2 * num_embed_dims + (self._msg_attribute_dims if self._use_attributes else 0)
        self.msg_mlp = MLP(msg_in_dim, units, activations, use_bias)

        units2 = list(units)
        units2[-1] = num_embed_dims
        m_dim = self._num_msg_dims + (self._node_attribute_dims if self._use_attributes else 0)
        embed_in_dim = m_dim + num_embed_dims
        self.embed_mlp = MLP(embed_in_dim, units2, activations, use_bias)
        self._built = True

    def forward(self, h_from, h_to):
        if not self._built:
            self._build(h_to.shape[-1])

        # features = concat([gather(h_from, from_ind, axis=1), gather(h_to, to_ind, axis=1)], -1)
        f_from = h_from.index_select(1, self._from_ind)
        f_to = h_to.index_select(1, self._to_ind)
        features = torch.cat([f_from, f_to], dim=-1)

        if self._use_attributes:
            attr = self._g_msg.unsqueeze(0).expand(features.shape[0], -1, -1)
            features = torch.cat([features, attr], dim=-1)

        messages = self.msg_mlp(features)  # [batch, num_edges, num_msg_dims]

        # Aggregate incoming messages at each "to" vertex per its ragged edge-index list.
        m_list = []
        for idx in self._gather_ind:
            if idx.numel() == 0:
                m_list.append(
                    torch.zeros(messages.shape[0], messages.shape[-1], device=messages.device)
                )
                continue
            gathered = messages.index_select(1, idx)  # [batch, deg, num_msg_dims]
            if self._reduce_op == "sum":
                m_list.append(gathered.sum(dim=1))
            elif self._reduce_op == "mean":
                m_list.append(gathered.mean(dim=1))
            elif self._reduce_op == "max":
                m_list.append(gathered.max(dim=1).values)
            elif self._reduce_op == "min":
                m_list.append(gathered.min(dim=1).values)
            else:
                raise ValueError("unknown reduce operation")
        m = torch.stack(m_list, dim=1)  # [batch, num_vertices, num_msg_dims]

        if self._use_attributes:
            attr = self._g_node.unsqueeze(0).expand(m.shape[0], -1, -1)
            m = torch.cat([m, attr], dim=-1)

        h_to_new = self.embed_mlp(torch.cat([m, h_to], dim=-1))
        return h_to_new


class GNN_BP(nn.Module):
    """Faithful port of gnn.GNN_BP: GNN-based message-passing decoder over a code's
    parity-check-matrix Tanner graph."""

    def __init__(
        self,
        pcm,
        num_embed_dims,
        num_msg_dims,
        num_hidden_units,
        num_mlp_layers,
        num_iter,
        reduce_op="mean",
        activation="tanh",
        output_all_iter=False,
        clip_llr_to=None,
        use_attributes=False,
        node_attribute_dims=0,
        msg_attribute_dims=0,
        use_bias=False,
    ):
        super().__init__()
        pcm = np.asarray(pcm)
        self._num_cn = pcm.shape[0]
        self._num_vn = pcm.shape[1]

        edges = np.stack(np.where(pcm), axis=1)  # [num_edges, 2]: (cn_id, vn_id)
        self._edges = edges

        cn_edges = [np.where(edges[:, 0] == i)[0] for i in range(self._num_cn)]
        vn_edges = [np.where(edges[:, 1] == i)[0] for i in range(self._num_vn)]

        self._num_embed_dims = num_embed_dims
        self._num_iter = num_iter
        self._output_all_iter = output_all_iter
        self._clip_llr_to = clip_llr_to

        self.llr_embed = nn.Linear(1, num_embed_dims, bias=use_bias)
        self.llr_inv_embed = nn.Linear(num_embed_dims, 1, bias=use_bias)

        self.update_h_cn = UpdateEmbeddings(
            num_msg_dims,
            num_hidden_units,
            num_mlp_layers,
            np.flip(edges, 1),  # "from VN to CN"
            cn_edges,
            reduce_op,
            activation,
            use_attributes,
            node_attribute_dims,
            msg_attribute_dims,
            use_bias,
        )
        self.update_h_vn = UpdateEmbeddings(
            num_msg_dims,
            num_hidden_units,
            num_mlp_layers,
            edges,  # "from CN to VN"
            vn_edges,
            reduce_op,
            activation,
            use_attributes,
            node_attribute_dims,
            msg_attribute_dims,
            use_bias,
        )

    def llr_to_embed(self, llr):
        return self.llr_embed(llr.unsqueeze(-1))

    def embed_to_llr(self, h_vn):
        return self.llr_inv_embed(h_vn).squeeze(-1)

    def forward(self, llr):
        batch_size = llr.shape[0]
        if self._clip_llr_to is not None:
            llr = torch.clamp(llr, -self._clip_llr_to, self._clip_llr_to)

        h_vn = self.llr_to_embed(llr)
        h_cn = torch.zeros(batch_size, self._num_cn, self._num_embed_dims, device=llr.device)

        llr_hat_all = []
        for _ in range(self._num_iter):
            h_cn = self.update_h_cn(h_vn, h_cn)
            h_vn = self.update_h_vn(h_cn, h_vn)
            if self._output_all_iter:
                llr_hat_all.append(self.embed_to_llr(h_vn))

        if self._output_all_iter:
            return llr_hat_all
        return self.embed_to_llr(h_vn)


# ---- tiny build/example (architecture unmodified from the real repo) ----


def _small_regular_ldpc_pcm(num_vn=12, num_cn=6, dv=3, seed=0):
    """Build a small regular-degree parity-check matrix for tracing (Gallager-style random
    construction so every check node gets a non-empty edge set)."""
    rng = np.random.RandomState(seed)
    pcm = np.zeros((num_cn, num_vn), dtype=np.float32)
    for v in range(num_vn):
        cns = rng.choice(num_cn, size=dv, replace=False)
        for c in cns:
            pcm[c, v] = 1.0
    # guarantee no all-zero check row (degenerate CN with no edges breaks the ragged gather)
    for c in range(num_cn):
        if pcm[c].sum() == 0:
            pcm[c, rng.randint(num_vn)] = 1.0
    return pcm


def build_gnn_ldpc_decoder():
    pcm = _small_regular_ldpc_pcm(num_vn=12, num_cn=6, dv=3)
    model = GNN_BP(
        pcm=pcm,
        num_embed_dims=8,
        num_msg_dims=8,
        num_hidden_units=16,
        num_mlp_layers=2,
        num_iter=4,
        reduce_op="mean",
        activation="tanh",
    )
    model.eval()
    return model


def example_input_gnn_ldpc_decoder():
    """llr: [batch_size, num_vn] float32 tensor of channel LLRs."""
    batch = 3
    return torch.randn(batch, 12)


MENAGERIE_ENTRIES = [
    (
        "GNN-LDPC-Decoder",
        build_gnn_ldpc_decoder,
        example_input_gnn_ldpc_decoder,
        2022,
        "ported-pytorch",
    ),
]
