# SOURCE: vendored from FraudDetection/FRAUDRE @ main
# https://raw.githubusercontent.com/FraudDetection/FRAUDRE/main/model.py
# https://raw.githubusercontent.com/FraudDetection/FRAUDRE/main/layers.py
#
# Zhang, Zhou, Ye, Wang, Zhang 2021 (ICDM Best Student Paper) "FRAUDRE: Fraud Detection
# Dual-Resistant to Graph Inconsistency and Imbalance" -- a heterogeneous-graph fraud
# detector for multi-relation graphs (e.g. Amazon/YelpChi review graphs) that stacks
# fraud-aware convolution layers: each layer's `InterAgg` runs one `IntraAgg` per relation
# (mean-aggregate each node's per-relation neighbor embeddings, then concatenate with the
# "ego minus neighbor-mean" difference feature -- the "graph-inconsistency-resistant"
# feature), then combines the 3 relation-specific outputs with a learned softmax attention
# (`weight_inter_agg`) and concatenates back onto the node's own embedding. Two such
# convolution layers are stacked (K=2), and a final `MODEL` head projects the full
# multi-scale concatenated embedding (`weight_model`/`weight_model2`) plus a parallel
# first-layer-only MLP head (`weight_mlp`) to per-class scores, with prior-corrected
# logits at train time (dual-resistant to label imbalance via the log-prior offset).
#
# `MLP_`, `InterAgg`, `IntraAgg`, `weight_inter_agg` are copied verbatim from the real
# `layers.py`. `MODEL` is copied verbatim from the real `model.py` (only the unused
# `Variable` import and CUDA-only `.cuda()` construction paths are dropped -- this module
# never sets `cuda=True`). No architectural changes were made. `IntraAgg.forward` builds a
# dense neighbor mask on CPU per call, exactly as in the original (`embed_matrix.cpu()`),
# which is fine at menagerie's tiny node counts.
#
# The real model does not consume a fixed-shape input tensor -- per `train.py`, `features`
# is an `nn.Embedding` over the full node set, `adj_lists` is a list of 3 python
# dict[node_id -> set[neighbor_id]] adjacency lists (one per relation, built via
# `sparse_to_adjlist` from a `scipy` sparse graph plus self-loops), and the model's public
# forward pass takes a plain python list of integer node ids (`nodes`). We reproduce this
# real calling convention: a tiny synthetic 40-node graph with 3 random relations (mirroring
# the real `sparse_to_adjlist` self-loop + up-to-10-neighbor sampling), and a batch of node
# ids as a python list, exactly like `train.py`'s `batch_nodes`.

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def weight_inter_agg(num_relations, neigh_feats, embed_dim, alpha, n, cuda):
    """
    Weight inter-relation aggregator
    :param num_relations: number of relations in the graph
    :param neigh_feats: intra_relation aggregated neighbor embeddings for each aggregation
    :param embed_dim: the dimension of output embedding
    :param alpha: weight paramter for each relation
    :param n: number of nodes in a batch
    :param cuda: whether use GPU
    """

    neigh_h = neigh_feats.t()

    w = F.softmax(alpha, dim=1)

    if cuda:
        aggregated = torch.zeros(size=(embed_dim, n)).cuda()
    else:
        aggregated = torch.zeros(size=(embed_dim, n))

    for r in range(num_relations):
        aggregated += torch.mul(w[:, r].unsqueeze(1).repeat(1, n), neigh_h[:, r * n : (r + 1) * n])

    return aggregated.t()


class MLP_(nn.Module):
    """
    the ego-feature embedding module
    """

    def __init__(self, features, input_dim, output_dim, cuda=False):
        super(MLP_, self).__init__()

        self.features = features
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cuda = cuda
        self.mlp_layer = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, nodes):
        if self.cuda:
            batch_features = self.features(torch.cuda.LongTensor(nodes))
        else:
            batch_features = self.features(torch.LongTensor(nodes))

        if self.cuda:
            self.mlp_layer.cuda()

        result = self.mlp_layer(batch_features)

        result = F.relu(result)

        return result


class InterAgg(nn.Module):
    """
    the fraud-aware convolution module
    Inter aggregation layer
    """

    def __init__(self, features, embed_dim, adj_lists, intraggs, cuda=False):
        """
        Initialize the inter-relation aggregator
        :param features: the input embeddings for all nodes
        :param embed_dim: the dimension need to be aggregated
        :param adj_lists: a list of adjacency lists for each single-relation graph
        :param intraggs: the intra-relation aggregatore used by each single-relation graph
        :param cuda: whether to use GPU
        """

        super(InterAgg, self).__init__()

        self.features = features
        self.dropout = 0.6
        self.adj_lists = adj_lists
        self.intra_agg1 = intraggs[0]
        self.intra_agg2 = intraggs[1]
        self.intra_agg3 = intraggs[2]
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.intra_agg1.cuda = cuda
        self.intra_agg2.cuda = cuda
        self.intra_agg3.cuda = cuda

        self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim * 2, 3))

        init.xavier_uniform_(self.alpha)

    def forward(self, nodes, train_flag=True):
        """
        nodes: a list of batch node ids
        """

        if not isinstance(nodes, list):
            nodes = nodes.cpu().numpy().tolist()

        to_neighs = []

        # adj_lists = [relation1, relation2, relation3]

        for adj_list in self.adj_lists:
            to_neighs.append([set(adj_list[int(node)]) for node in nodes])

        # find unique nodes and their neighbors used in current batch
        unique_nodes = set.union(
            set.union(*to_neighs[0]), set.union(*to_neighs[1]), set.union(*to_neighs[2], set(nodes))
        )

        # id mapping
        unique_nodes_new_index = {n: i for i, n in enumerate(list(unique_nodes))}

        if self.cuda:
            batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
        else:
            batch_features = self.features(torch.LongTensor(list(unique_nodes)))

        # get neighbor node id list for each batch node and relation
        r1_list = [set(to_neigh) for to_neigh in to_neighs[0]]
        r2_list = [set(to_neigh) for to_neigh in to_neighs[1]]
        r3_list = [set(to_neigh) for to_neigh in to_neighs[2]]

        center_nodes_new_index = [unique_nodes_new_index[int(n)] for n in nodes]

        self_feats = batch_features[center_nodes_new_index]

        r1_feats = self.intra_agg1.forward(
            batch_features[:, -self.embed_dim :],
            nodes,
            r1_list,
            unique_nodes_new_index,
            self_feats[:, -self.embed_dim :],
        )
        r2_feats = self.intra_agg2.forward(
            batch_features[:, -self.embed_dim :],
            nodes,
            r2_list,
            unique_nodes_new_index,
            self_feats[:, -self.embed_dim :],
        )
        r3_feats = self.intra_agg3.forward(
            batch_features[:, -self.embed_dim :],
            nodes,
            r3_list,
            unique_nodes_new_index,
            self_feats[:, -self.embed_dim :],
        )

        neigh_feats = torch.cat((r1_feats, r2_feats, r3_feats), dim=0)

        n = len(nodes)

        attention_layer_outputs = weight_inter_agg(
            len(self.adj_lists), neigh_feats, self.embed_dim * 2, self.alpha, n, self.cuda
        )

        result = torch.cat((self_feats, attention_layer_outputs), dim=1)

        return result


class IntraAgg(nn.Module):
    """
    the fraud-aware convolution module
    Intra Aggregation Layer
    """

    def __init__(self, cuda=False):
        super(IntraAgg, self).__init__()

        self.cuda = cuda

    def forward(self, embedding, nodes, neighbor_lists, unique_nodes_new_index, self_feats):
        """
        Code partially from https://github.com/williamleif/graphsage-simple/
        :param nodes: list of nodes in a batch
        :param embedding: embedding of all nodes in a batch
        :param neighbor_lists: neighbor node id list for each batch node in one relation
        :param unique_nodes_new_index
        """

        # find unique nodes
        unique_nodes_list = list(set.union(*neighbor_lists))

        # id mapping
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

        mask = torch.zeros(len(neighbor_lists), len(unique_nodes))

        column_indices = [
            unique_nodes[n] for neighbor_list in neighbor_lists for n in neighbor_list
        ]
        row_indices = [i for i in range(len(neighbor_lists)) for _ in range(len(neighbor_lists[i]))]

        mask[row_indices, column_indices] = 1

        num_neigh = mask.sum(1, keepdim=True)
        mask = torch.div(mask, num_neigh)

        neighbors_new_index = [unique_nodes_new_index[n] for n in unique_nodes_list]

        embed_matrix = embedding[neighbors_new_index]

        embed_matrix = embed_matrix.cpu()

        _feats_1 = mask.mm(embed_matrix)
        if self.cuda:
            _feats_1 = _feats_1.cuda()

        # difference
        _feats_2 = self_feats - _feats_1
        return torch.cat((_feats_1, _feats_2), dim=1)


class MODEL(nn.Module):
    def __init__(self, K, num_classes, embed_dim, agg, prior):
        super(MODEL, self).__init__()
        """
        Initialize the model
        :param K: the number of CONVOLUTION layers of the model
        :param num_classes: number of classes (2 in our paper)
        :param embed_dim: the output dimension of MLP layer
        :agg: the inter-relation aggregator that output the final embedding
        :prior:prior
        """

        self.agg = agg

        self.K = K
        self.prior = prior
        self.xent = nn.CrossEntropyLoss()
        self.embed_dim = embed_dim
        self.fun = nn.LeakyReLU(0.3)

        self.weight_mlp = nn.Parameter(torch.FloatTensor(self.embed_dim, num_classes))
        self.weight_model = nn.Parameter(
            torch.FloatTensor((int(math.pow(2, K + 1) - 1) * self.embed_dim), 64)
        )

        self.weight_model2 = nn.Parameter(torch.FloatTensor(64, num_classes))

        init.xavier_uniform_(self.weight_mlp)
        init.xavier_uniform_(self.weight_model)
        init.xavier_uniform_(self.weight_model2)

    def forward(self, nodes, train_flag=True):
        embedding = self.agg(nodes, train_flag)

        scores_model = embedding.mm(self.weight_model)
        scores_model = self.fun(scores_model)
        scores_model = scores_model.mm(self.weight_model2)

        scores_mlp = embedding[:, 0 : self.embed_dim].mm(self.weight_mlp)
        scores_mlp = self.fun(scores_mlp)

        return scores_model, scores_mlp

    def to_prob(self, nodes, train_flag=False):
        scores_model, scores_mlp = self.forward(nodes, train_flag)
        scores_model = torch.sigmoid(scores_model)
        return scores_model

    def loss(self, nodes, labels, train_flag=True):
        scores_model, scores_mlp = self.forward(nodes, train_flag)

        scores_model = scores_model + torch.log(self.prior)
        scores_mlp = scores_mlp + torch.log(self.prior)

        loss_model = self.xent(scores_model, labels.squeeze())
        final_loss = loss_model
        return final_loss


def _sparse_to_adjlist(n_nodes, n_edges, seed):
    """Mirrors the real utlis.sparse_to_adjlist: builds an undirected adjacency dict with
    self-loops from a random small edge list, capping each node's neighbor set at 10 (same
    `random.sample` cap the original applies to real scipy-sparse relation graphs)."""
    rng = random.Random(seed)
    adj_lists = {i: {i} for i in range(n_nodes)}
    for _ in range(n_edges):
        a, b = rng.randrange(n_nodes), rng.randrange(n_nodes)
        adj_lists[a].add(b)
        adj_lists[b].add(a)
    return {
        k: (set(rng.sample(sorted(v), 10)) if len(v) >= 10 else v) for k, v in adj_lists.items()
    }


def build_fraudre():
    n_nodes = 40
    feat_dim = 12
    embed_dim = 8

    features = nn.Embedding(n_nodes, feat_dim)
    features.weight = nn.Parameter(torch.randn(n_nodes, feat_dim), requires_grad=False)

    relation1 = _sparse_to_adjlist(n_nodes, 60, seed=1)
    relation2 = _sparse_to_adjlist(n_nodes, 60, seed=2)
    relation3 = _sparse_to_adjlist(n_nodes, 60, seed=3)
    adj_lists = [relation1, relation2, relation3]

    mlp = MLP_(features, feat_dim, embed_dim, cuda=False)

    intra1_1, intra1_2, intra1_3 = IntraAgg(cuda=False), IntraAgg(cuda=False), IntraAgg(cuda=False)
    agg1 = InterAgg(
        lambda nodes: mlp(nodes), embed_dim, adj_lists, [intra1_1, intra1_2, intra1_3], cuda=False
    )

    intra2_1, intra2_2, intra2_3 = IntraAgg(cuda=False), IntraAgg(cuda=False), IntraAgg(cuda=False)
    agg2 = InterAgg(
        lambda nodes: agg1(nodes),
        embed_dim * 2,
        adj_lists,
        [intra2_1, intra2_2, intra2_3],
        cuda=False,
    )

    prior = torch.tensor([0.9, 0.1], dtype=torch.float32)
    return MODEL(2, 2, embed_dim, agg2, prior)


def example_input_fraudre():
    batch_nodes = list(range(10))
    return (batch_nodes,)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("FRAUDRE", "build_fraudre", "example_input_fraudre", 2021, "vendored"),
]
