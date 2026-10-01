# FAITHFUL PORT of mims-harvard/decagon @ master (original framework: TensorFlow 1.x)
# https://raw.githubusercontent.com/mims-harvard/decagon/master/decagon/deep/layers.py
# https://raw.githubusercontent.com/mims-harvard/decagon/master/decagon/deep/model.py
# https://raw.githubusercontent.com/mims-harvard/decagon/master/decagon/deep/inits.py
#
# Zitnik, Agrawal, Leskovec 2018 (Bioinformatics/ISMB) "Modeling polypharmacy side effects
# with graph convolutional networks" -- Decagon: a multirelational graph convolutional
# encoder (per-node-type, per-edge-type GraphConvolution(Sparse)Multi stack, matching
# `DecagonModel._build()`) followed by a DEDICOM tensor-factorization decoder for
# multi-relational link prediction (drug-drug side-effect types). The reference code cannot
# run in the base env: TF1.x graph-mode APIs (`tf.variable_scope`, `tf.sparse_tensor_dense_
# matmul`, `tf.app.flags`, `tf.diag`) were removed in TF2 and are not installed here. This
# port transcribes the real two-layer multirelational GCN encoder (GraphConvolutionSparseMulti
# -> relu -> GraphConvolutionMulti, summed across edge types per node type, L2-normalized per
# layer) and the real DEDICOM decoder (`relation = diag(local_variation)`; `rec = (row @
# relation) @ global_interaction @ relation @ col.T`; sigmoid) faithfully into a
# self-contained torch.nn.Module, replacing only the TF1 sparse-placeholder plumbing with
# eager torch ops over dense adjacency matrices. Simplified to the paper's core two node
# types (drug, protein) with two edge types (drug-protein bipartite, drug-drug multirelational
# side-effect) to keep the trace self-contained, matching the reference's node_type/edge_type
# dict-of-adjacency-matrices design exactly for those edge types.
"""Decagon: multirelational GCN encoder + DEDICOM decoder for polypharmacy prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def glorot_init(input_dim, output_dim):
    """Port of decagon/deep/inits.py::weight_variable_glorot."""
    init_range = (6.0 / (input_dim + output_dim)) ** 0.5
    return nn.Parameter(torch.empty(input_dim, output_dim).uniform_(-init_range, init_range))


class GraphConvolutionSparseMulti(nn.Module):
    """Port of decagon/deep/layers.py::GraphConvolutionSparseMulti (dense-adjacency
    variant -- the reference's `tf.sparse_tensor_dense_matmul` degenerates to a regular
    matmul once the sparse placeholders are materialized as dense tensors, which is what
    happens here)."""

    def __init__(self, input_dim, output_dim, num_types, dropout=0.0):
        super().__init__()
        self.num_types = num_types
        self.dropout = dropout
        self.weights = nn.ParameterList(
            [glorot_init(input_dim, output_dim) for _ in range(num_types)]
        )

    def forward(self, x, adj_mats):
        """x: (n_src, input_dim); adj_mats: list of (n_dst, n_src) per relation-type k."""
        outputs = []
        for k in range(self.num_types):
            h = F.dropout(x, p=self.dropout, training=self.training)
            h = h @ self.weights[k]
            h = adj_mats[k] @ h
            outputs.append(F.relu(h))
        out = torch.stack(outputs, dim=0).sum(dim=0)
        out = F.normalize(out, p=2, dim=1)
        return out


class GraphConvolutionMulti(nn.Module):
    """Port of decagon/deep/layers.py::GraphConvolutionMulti."""

    def __init__(self, input_dim, output_dim, num_types, dropout=0.0):
        super().__init__()
        self.num_types = num_types
        self.dropout = dropout
        self.weights = nn.ParameterList(
            [glorot_init(input_dim, output_dim) for _ in range(num_types)]
        )

    def forward(self, x, adj_mats):
        outputs = []
        for k in range(self.num_types):
            h = F.dropout(x, p=self.dropout, training=self.training)
            h = h @ self.weights[k]
            h = adj_mats[k] @ h
            outputs.append(F.relu(h))
        out = torch.stack(outputs, dim=0).sum(dim=0)
        out = F.normalize(out, p=2, dim=1)
        return out


class DEDICOMDecoder(nn.Module):
    """Port of decagon/deep/layers.py::DEDICOMDecoder -- tensor-factorization decoder
    used for the multirelational drug-drug (polypharmacy side-effect) edge type."""

    def __init__(self, input_dim, num_types, dropout=0.0):
        super().__init__()
        self.num_types = num_types
        self.dropout = dropout
        self.global_interaction = glorot_init(input_dim, input_dim)
        self.local_variation = nn.ParameterList(
            [nn.Parameter(glorot_init(input_dim, 1).reshape(-1)) for _ in range(num_types)]
        )

    def forward(self, inputs_row, inputs_col):
        outputs = []
        for k in range(self.num_types):
            row = F.dropout(inputs_row, p=self.dropout, training=self.training)
            col = F.dropout(inputs_col, p=self.dropout, training=self.training)
            relation = torch.diag(self.local_variation[k])
            product1 = row @ relation
            product2 = product1 @ self.global_interaction
            product3 = product2 @ relation
            rec = product3 @ col.t()
            outputs.append(torch.sigmoid(rec))
        return torch.stack(outputs, dim=0)


class DecagonModel(nn.Module):
    """Port of decagon/deep/model.py::DecagonModel, specialized to the paper's two core
    node types (0=drug, 1=protein) and two edge types matching `_build()`'s loop structure:
      - protein-protein interaction graph (single relation type)
      - drug-protein target graph (bipartite, single relation type)
      - drug-drug polypharmacy graph (multirelational, `n_drug_drug_rel` side-effect types),
        decoded with the real DEDICOM decoder.
    """

    def __init__(self, n_drug, n_protein, feat_dim, hidden1, hidden2, n_drug_drug_rel):
        super().__init__()
        self.n_drug = n_drug
        self.n_protein = n_protein
        self.n_drug_drug_rel = n_drug_drug_rel

        # layer 1: sparse-feature GCN per (node_type -> node_type) edge, matching
        # DecagonModel._build()'s `self.hidden1[i]` accumulation over `edge_types`.
        self.gc1_pp = GraphConvolutionSparseMulti(
            feat_dim, hidden1, num_types=1
        )  # protein<-protein
        self.gc1_dp = GraphConvolutionSparseMulti(feat_dim, hidden1, num_types=1)  # drug<-protein
        self.gc1_pd = GraphConvolutionSparseMulti(feat_dim, hidden1, num_types=1)  # protein<-drug
        self.gc1_dd = GraphConvolutionSparseMulti(
            feat_dim, hidden1, num_types=n_drug_drug_rel
        )  # drug<-drug

        # layer 2: dense GCN, same edge-type structure, matching `self.embeddings_reltyp`.
        self.gc2_pp = GraphConvolutionMulti(hidden1, hidden2, num_types=1)
        self.gc2_dp = GraphConvolutionMulti(hidden1, hidden2, num_types=1)
        self.gc2_pd = GraphConvolutionMulti(hidden1, hidden2, num_types=1)
        self.gc2_dd = GraphConvolutionMulti(hidden1, hidden2, num_types=n_drug_drug_rel)

        # decoder for the drug-drug polypharmacy edge type, matching `edge_type2decoder`.
        self.dedicom = DEDICOMDecoder(hidden2, num_types=n_drug_drug_rel)

    def forward(self, drug_feat, protein_feat, adj_pp, adj_dp, adj_pd, adj_dd):
        """
        drug_feat: (n_drug, feat_dim); protein_feat: (n_protein, feat_dim)
        adj_pp: list[1] of (n_protein, n_protein) protein-protein adjacency
        adj_dp: list[1] of (n_drug, n_protein) drug<-protein adjacency
        adj_pd: list[1] of (n_protein, n_drug) protein<-drug adjacency
        adj_dd: list[n_drug_drug_rel] of (n_drug, n_drug) per-side-effect drug-drug adjacency
        """
        # layer 1 (sparse-feature GCN), summed per node type across incident edge types,
        # matching `self.hidden1[i] = relu(add_n(hid1))`.
        h1_protein = F.relu(self.gc1_pp(protein_feat, adj_pp) + self.gc1_pd(drug_feat, adj_pd))
        h1_drug = F.relu(self.gc1_dp(protein_feat, adj_dp) + self.gc1_dd(drug_feat, adj_dd))

        # layer 2 (dense GCN), matching `self.embeddings[i] = add_n(embeds)`.
        emb_protein = self.gc2_pp(h1_protein, adj_pp) + self.gc2_pd(h1_drug, adj_pd)
        emb_drug = self.gc2_dp(h1_protein, adj_dp) + self.gc2_dd(h1_drug, adj_dd)

        # DEDICOM decoder over the drug-drug multirelational edge type.
        reconstructions = self.dedicom(emb_drug, emb_drug)  # (n_drug_drug_rel, n_drug, n_drug)
        return reconstructions, emb_drug, emb_protein


def build_decagon():
    return DecagonModel(
        n_drug=8, n_protein=6, feat_dim=10, hidden1=16, hidden2=8, n_drug_drug_rel=3
    )


def example_input_decagon():
    torch.manual_seed(0)
    n_drug, n_protein, feat_dim = 8, 6, 10
    n_drug_drug_rel = 3
    drug_feat = torch.randn(n_drug, feat_dim)
    protein_feat = torch.randn(n_protein, feat_dim)
    adj_pp = [torch.rand(n_protein, n_protein)]
    adj_dp = [torch.rand(n_drug, n_protein)]
    adj_pd = [torch.rand(n_protein, n_drug)]
    adj_dd = [torch.rand(n_drug, n_drug) for _ in range(n_drug_drug_rel)]
    return (drug_feat, protein_feat, adj_pp, adj_dp, adj_pd, adj_dd)


MENAGERIE_ENTRIES = [
    ("Decagon", "build_decagon", "example_input_decagon", 2018, "ported"),
]
