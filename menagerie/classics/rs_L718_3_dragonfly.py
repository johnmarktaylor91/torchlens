# SOURCE: vendored from https://github.com/ETHmodlab/dragonfly_gen @ master
#
# DRAGONFLY (Atz, Isert, Schneider et al., "Prospective de novo drug design with deep
# interactome learning", Nature Communications 2024) -- a ligand-based generative model for
# interactome-aware drug design: a graph-transformer / EGNN molecular-graph encoder
# (equivariant message passing + a Graph Multiset Transformer multi-head pooling readout)
# whose latent embedding conditions an autoregressive LSTM SMILES/SELFIES decoder. Vendored
# verbatim (architecture-relevant classes only) from the repo's own files:
#   https://raw.githubusercontent.com/ETHmodlab/dragonfly_gen/master/dragonfly_gen/genfromligand/net.py
#   https://raw.githubusercontent.com/ETHmodlab/dragonfly_gen/master/dragonfly_gen/gml/pygmt.py
#
# What is kept: GraphTransformer (the real 2D-molecular-graph encoder: 4 categorical atom
# embeddings -> pre-EGNN MLP -> a stack of real EGNN_sparse message-passing kernels -> a
# post-EGNN MLP -> `pooling_heads` parallel GraphMultisetTransformer multi-head attention
# poolers -> post-pooling MLPs), EGNN_sparse (the real PyG `MessagePassing` equivariant
# graph-neural-network layer with edge/node MLPs and LayerNorm -- edge_mlp -> edge_norm1 ->
# aggregate -> edge_norm2 -> node_mlp -> node_norm2 -> residual, all unmodified),
# GraphMultisetTransformer / MAB / SAB / PMA (vendored from `dragonfly_gen/gml/pygmt.py`,
# itself PyTorch Geometric's own Graph Multiset Pooling operator vendored by the DRAGONFLY
# authors) -- every mechanism in the real trainable network, transcribed unmodified. This
# staging build exercises `GraphTransformer`, the primary ligand-side molecular-graph encoder
# (the file's `LSTM` class is the paired autoregressive SMILES decoder used at generation
# time, architecturally a plain `nn.LSTM` wrapper and not vendored separately here since it
# adds no new mechanism beyond `nn.Embedding`+`nn.LSTM`).
#
# What is dropped (infra plumbing, not part of the forward-pass computation graph):
# `dragonfly_gen/genfromligand/sampling.py` (RDKit/argparse/configparser-driven CLI sampling
# script; imports `rdkit`, not an installed base lib here) and
# `dragonfly_gen/drugtargetgraph/utils.py` (SMILES/SELFIES tokenization vocab, also
# RDKit-dependent) are not vendored -- neither defines architecture, both only consume
# `GraphTransformer`/`LSTM` for CLI-driven molecule generation. `scatter_sum`/`broadcast`
# (used only by the sibling structure-based `EGNN` encoder in the same file, not by
# `GraphTransformer`) are likewise not vendored. `EGNN_sparse.propagate` is rewritten to
# build `x_i`/`x_j` directly from `edge_index` and call the real `message`/`aggregate`
# sequence with the identical inter-step norm/MLP/residual computation, replacing the
# original's `self.inspector.distribute(...)`/`self._collect(...)` calls -- PyTorch
# Geometric-internal auto-dispatch plumbing (not architecture) that no longer exists in
# current torch_geometric releases; every real learnable layer and the exact computation
# order is unchanged. This staging module builds a synthetic-graph `example_input_dragonfly()`
# (random atom/edge features of the exact
# categorical ranges `GraphTransformer.forward` expects) in place of RDKit-parsed molecules.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, Size, Tensor

MENAGERIE_ZOO = "vendored-pytorch"


def weights_init(m):
    """Xavier uniform weight initialization."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# from dragonfly_gen/gml/pygmt.py (verbatim) -- Graph Multiset Transformer
# pooling operator vendored by the DRAGONFLY authors from PyTorch Geometric
# ---------------------------------------------------------------------------
class MAB(torch.nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, Conv=None, layer_norm=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        self.fc_q = nn.Linear(dim_Q, dim_V)

        if Conv is None:
            self.layer_k = nn.Linear(dim_K, dim_V)
            self.layer_v = nn.Linear(dim_K, dim_V)
        else:
            self.layer_k = Conv(dim_K, dim_V)
            self.layer_v = Conv(dim_K, dim_V)

        if layer_norm:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)

        self.fc_o = nn.Linear(dim_V, dim_V)

    def reset_parameters(self):
        self.fc_q.reset_parameters()
        self.layer_k.reset_parameters()
        self.layer_v.reset_parameters()
        if self.layer_norm:
            self.ln0.reset_parameters()
            self.ln1.reset_parameters()
        self.fc_o.reset_parameters()

    def forward(self, Q, K, graph=None, mask=None):
        import math

        from torch_geometric.utils import to_dense_batch

        Q = self.fc_q(Q)

        if graph is not None:
            x, edge_index, batch = graph
            K, V = self.layer_k(x, edge_index), self.layer_v(x, edge_index)
            K, _ = to_dense_batch(K, batch)
            V, _ = to_dense_batch(V, batch)
        else:
            K, V = self.layer_k(K), self.layer_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), dim=0)
        K_ = torch.cat(K.split(dim_split, 2), dim=0)
        V_ = torch.cat(V.split(dim_split, 2), dim=0)

        if mask is not None:
            mask = torch.cat([mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1, 2))
            attention_score = attention_score / math.sqrt(self.dim_V)
            A = torch.softmax(mask + attention_score, 1)
        else:
            A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 1)

        out = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        if self.layer_norm:
            out = self.ln0(out)

        out = out + self.fc_o(out).relu()

        if self.layer_norm:
            out = self.ln1(out)

        return out


class SAB(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, Conv=None, layer_norm=False):
        super().__init__()
        self.mab = MAB(
            in_channels, in_channels, out_channels, num_heads, Conv=Conv, layer_norm=layer_norm
        )

    def reset_parameters(self):
        self.mab.reset_parameters()

    def forward(self, x, graph=None, mask=None):
        return self.mab(x, x, graph, mask)


class PMA(torch.nn.Module):
    def __init__(self, channels, num_heads, num_seeds, Conv=None, layer_norm=False):
        super().__init__()
        self.S = torch.nn.Parameter(torch.Tensor(1, num_seeds, channels))
        self.mab = MAB(channels, channels, channels, num_heads, Conv=Conv, layer_norm=layer_norm)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.S)
        self.mab.reset_parameters()

    def forward(self, x, graph=None, mask=None):
        return self.mab(self.S.repeat(x.size(0), 1, 1), x, graph, mask)


class GraphMultisetTransformer(torch.nn.Module):
    r"""The global Graph Multiset Transformer pooling operator from
    "Accurate Learning of Graph Representations with Graph Multiset Pooling"
    <https://arxiv.org/abs/2102.11533>."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        Conv=None,
        num_nodes=300,
        pooling_ratio=0.25,
        pool_sequences=["GMPool_G", "SelfAtt", "GMPool_I"],
        num_heads=4,
        layer_norm=False,
    ):
        import math

        from torch_geometric.nn import GCNConv

        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.Conv = Conv or GCNConv
        self.num_nodes = num_nodes
        self.pooling_ratio = pooling_ratio
        self.pool_sequences = pool_sequences
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

        self.pools = torch.nn.ModuleList()
        num_out_nodes = math.ceil(num_nodes * pooling_ratio)
        for i, pool_type in enumerate(pool_sequences):
            if pool_type not in ["GMPool_G", "GMPool_I", "SelfAtt"]:
                raise ValueError(
                    "Elements in 'pool_sequences' should be one of 'GMPool_G', 'GMPool_I', or 'SelfAtt'"
                )

            if i == len(pool_sequences) - 1:
                num_out_nodes = 1

            if pool_type == "GMPool_G":
                self.pools.append(
                    PMA(
                        hidden_channels,
                        num_heads,
                        num_out_nodes,
                        Conv=self.Conv,
                        layer_norm=layer_norm,
                    )
                )
                num_out_nodes = math.ceil(num_out_nodes * self.pooling_ratio)

            elif pool_type == "GMPool_I":
                self.pools.append(
                    PMA(hidden_channels, num_heads, num_out_nodes, Conv=None, layer_norm=layer_norm)
                )
                num_out_nodes = math.ceil(num_out_nodes * self.pooling_ratio)

            elif pool_type == "SelfAtt":
                self.pools.append(
                    SAB(
                        hidden_channels,
                        hidden_channels,
                        num_heads,
                        Conv=None,
                        layer_norm=layer_norm,
                    )
                )

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()

    def forward(self, x, batch, edge_index=None):
        from torch_geometric.utils import to_dense_batch

        x = self.lin1(x)
        batch_x, mask = to_dense_batch(x, batch)
        mask = (~mask).unsqueeze(1).to(dtype=x.dtype) * -1e9

        for i, (name, pool) in enumerate(zip(self.pool_sequences, self.pools)):
            graph = (x, edge_index, batch) if name == "GMPool_G" else None
            batch_x = pool(batch_x, graph, mask)
            mask = None

        return self.lin2(batch_x.squeeze(1))


# ---------------------------------------------------------------------------
# from genfromligand/net.py (verbatim) -- EGNN_sparse message-passing kernel
# ---------------------------------------------------------------------------
class EGNN_sparse(MessagePassing):
    """torch geometric message-passing layer for 2D molecular graphs."""

    def __init__(self, feats_dim, m_dim=32, dropout=0.1, aggr="add", **kwargs):
        assert aggr in {"add", "sum", "max", "mean"}, "pool method must be a valid option"

        kwargs.setdefault("aggr", aggr)
        super(EGNN_sparse, self).__init__(**kwargs)

        self.feats_dim = feats_dim
        self.m_dim = m_dim

        self.edge_input_dim = feats_dim * 2

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.edge_norm1 = nn.LayerNorm(m_dim)
        self.edge_norm2 = nn.LayerNorm(m_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.edge_input_dim * 2, m_dim),
            nn.SiLU(),
        )

        self.node_norm1 = nn.LayerNorm(feats_dim)
        self.node_norm2 = nn.LayerNorm(feats_dim)

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, feats_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(feats_dim * 2, feats_dim),
        )

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, edge_index: Adj):
        hidden_out = self.propagate(edge_index, x=x)
        return hidden_out

    def message(self, x_i, x_j):
        m_ij = self.edge_mlp(torch.cat([x_i, x_j], dim=-1))
        return m_ij

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        # NOTE: the original repo code drove this step via
        # `self.inspector.distribute(...)` + `self._collect(...)`, PyTorch Geometric
        # internal auto-dispatch plumbing (not architecture) that no longer exists in
        # current torch_geometric releases. Rewritten here to build `x_i`/`x_j`
        # directly from `edge_index` and call the same real `message`/`aggregate`
        # sequence with the identical inter-step LayerNorm/MLP/residual computation
        # the original `propagate` override performed -- the architecture itself
        # (edge_mlp -> edge_norm1 -> aggregate -> edge_norm2 -> node_mlp -> node_norm2
        # -> residual) is unchanged.
        x = kwargs["x"]
        row, col = edge_index[0], edge_index[1]  # source_to_target: j=row, i=col
        x_j = x.index_select(0, row)
        x_i = x.index_select(0, col)

        m_ij = self.message(x_i, x_j)
        m_ij = self.edge_norm1(m_ij)

        m_i = self.aggregate(m_ij, index=col, dim_size=x.size(0))
        m_i = self.edge_norm2(m_i)

        hidden_feats = self.node_norm1(kwargs["x"])
        hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
        hidden_out = self.node_norm2(hidden_out)
        hidden_out = kwargs["x"] + hidden_out

        return self.update(hidden_out)


# ---------------------------------------------------------------------------
# from genfromligand/net.py (verbatim) -- top-level ligand-graph encoder
# ---------------------------------------------------------------------------
class GraphTransformer(nn.Module):
    def __init__(self, n_kernels=3, rnn_dim=1024, property_dim=6, pooling_heads=8):
        super(GraphTransformer, self).__init__()

        self.num_embeddings_atom = 22
        self.num_embeddings_residue = 255
        self.embeddings_dim = 64
        self.pdb_prop_dim = 32
        self.m_dim = 16
        self.kernel_dim = 128
        self.n_kernels = n_kernels
        self.aggr = "add"
        self.pooling_heads = pooling_heads
        self.property = property_dim
        self.rnn_dim = rnn_dim

        dropout = 0.1
        self.dropout = nn.Dropout(dropout)

        self.atom_emb = nn.Embedding(num_embeddings=11, embedding_dim=self.embeddings_dim)
        self.is_ring_emb = nn.Embedding(num_embeddings=2, embedding_dim=self.embeddings_dim)
        self.hyb_emb = nn.Embedding(num_embeddings=4, embedding_dim=self.embeddings_dim)
        self.arom_emb = nn.Embedding(num_embeddings=2, embedding_dim=self.embeddings_dim)

        self.pre_egnn_mlp = nn.Sequential(
            nn.Linear(self.embeddings_dim * 4, self.kernel_dim * 2),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.kernel_dim * 2, self.kernel_dim),
        )

        self.kernels = nn.ModuleList()
        for _ in range(self.n_kernels):
            self.kernels.append(
                EGNN_sparse(
                    feats_dim=self.kernel_dim,
                    m_dim=self.m_dim,
                    aggr=self.aggr,
                )
            )

        self.post_egnn_mlp = nn.Sequential(
            nn.Linear(self.kernel_dim * self.n_kernels, self.kernel_dim),
            self.dropout,
            nn.SiLU(),
            nn.Linear(self.kernel_dim, self.kernel_dim),
            nn.SiLU(),
            nn.Linear(self.kernel_dim, self.kernel_dim),
            nn.SiLU(),
        )

        self.transformers = nn.ModuleList()
        for _ in range(self.pooling_heads):
            self.transformers.append(
                GraphMultisetTransformer(
                    in_channels=self.kernel_dim,
                    hidden_channels=self.kernel_dim,
                    out_channels=self.kernel_dim,
                    pool_sequences=["GMPool_G", "SelfAtt", "GMPool_I"],
                    num_heads=1,
                    layer_norm=True,
                )
            )

        if self.property == 6:
            self.mol_property_lin = nn.Linear(self.property, self.kernel_dim)
            self.mlp_input_dim = self.kernel_dim * (self.pooling_heads + 1)
            self.mol_property_lin.apply(weights_init)
        elif self.property == 1:
            self.mol_property_lin = nn.Linear(self.property, self.kernel_dim)
            self.mlp_input_dim = self.kernel_dim * (self.pooling_heads + 1)
            self.mol_property_lin.apply(weights_init)
        else:
            self.mlp_input_dim = self.kernel_dim * self.pooling_heads

        self.post_pooling_mlps = nn.ModuleList()
        for _ in range(2):
            self.post_pooling_mlps.append(
                nn.Sequential(
                    nn.Linear(self.mlp_input_dim, self.rnn_dim),
                    self.dropout,
                    nn.SiLU(),
                    nn.Linear(self.rnn_dim, self.rnn_dim),
                    nn.SiLU(),
                    nn.Linear(self.rnn_dim, self.rnn_dim),
                )
            )

        self.transformers.apply(weights_init)
        self.kernels.apply(weights_init)
        self.post_egnn_mlp.apply(weights_init)
        self.post_pooling_mlps.apply(weights_init)
        nn.init.xavier_uniform_(self.atom_emb.weight)
        nn.init.xavier_uniform_(self.is_ring_emb.weight)
        nn.init.xavier_uniform_(self.hyb_emb.weight)
        nn.init.xavier_uniform_(self.arom_emb.weight)

    def forward(self, g_batch):
        features = self.pre_egnn_mlp(
            torch.cat(
                [
                    self.atom_emb(g_batch.atomids),
                    self.is_ring_emb(g_batch.is_ring),
                    self.hyb_emb(g_batch.hyb),
                    self.arom_emb(g_batch.arom),
                ],
                dim=1,
            )
        )

        feature_list = []
        for kernel in self.kernels:
            feature_list.append(kernel(x=features, edge_index=g_batch.edge_index))

        features = torch.cat(feature_list, dim=1)
        features = self.post_egnn_mlp(features)

        feature_list = []
        for transformer in self.transformers:
            feature_list.append(
                transformer(x=features, batch=g_batch.batch, edge_index=g_batch.edge_index)
            )

        features = torch.cat(feature_list, dim=1)

        if self.property == 6:
            features = torch.cat([features, self.mol_property_lin(g_batch.properties)], dim=1)
        elif self.property == 1:
            features = torch.cat([features, self.mol_property_lin(g_batch.sim)], dim=1)

        feature_list = []
        for mlp in self.post_pooling_mlps:
            feature_list.append(mlp(features).unsqueeze(0))

        features = torch.cat(feature_list, dim=0)
        del feature_list

        features = (
            features,
            torch.zeros(2, features.size(1), features.size(2)).to(features.device),
        )

        return features


# ---------------------------------------------------------------------------
# staging glue (not part of the original architecture)
# ---------------------------------------------------------------------------
def build_dragonfly():
    # Tiny GraphTransformer: fewer EGNN kernels / pooling heads / rnn_dim than the
    # published config (n_kernels=3, pooling_heads=8, rnn_dim=1024), same mechanisms.
    return GraphTransformer(n_kernels=2, rnn_dim=32, property_dim=6, pooling_heads=2)


def example_input_dragonfly():
    # Synthetic molecular graph batch with the exact categorical feature ranges
    # GraphTransformer.forward reads (atomids in [0,11), is_ring in [0,2), hyb in
    # [0,4), arom in [0,2)) in place of RDKit-parsed real molecules (rdkit is not an
    # installed base lib here).
    generator = torch.Generator().manual_seed(0)
    n_atoms_per_mol = [6, 5]
    atomids, is_ring, hyb, arom, batch_idx, edges = [], [], [], [], [], []
    offset = 0
    for mol_i, n in enumerate(n_atoms_per_mol):
        atomids.append(torch.randint(0, 11, (n,), generator=generator))
        is_ring.append(torch.randint(0, 2, (n,), generator=generator))
        hyb.append(torch.randint(0, 4, (n,), generator=generator))
        arom.append(torch.randint(0, 2, (n,), generator=generator))
        batch_idx.append(torch.full((n,), mol_i, dtype=torch.long))
        # simple ring connectivity, both directions (undirected graph as edge pairs)
        for a in range(n):
            b = (a + 1) % n
            edges.append([offset + a, offset + b])
            edges.append([offset + b, offset + a])
        offset += n

    data = Data()
    data.atomids = torch.cat(atomids)
    data.is_ring = torch.cat(is_ring)
    data.hyb = torch.cat(hyb)
    data.arom = torch.cat(arom)
    data.batch = torch.cat(batch_idx)
    data.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    data.properties = torch.rand(len(n_atoms_per_mol), 6, generator=generator)
    return (data,)


MENAGERIE_ENTRIES = [
    ("DRAGONFLY", "build_dragonfly", "example_input_dragonfly", 2024, "vendored-pytorch"),
]
