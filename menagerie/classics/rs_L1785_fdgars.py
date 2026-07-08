# FAITHFUL PORT of safe-graph/DGFraud @ master (original framework: TensorFlow 1.x)
"""FdGars: Fraudster Detection via Graph Convolutional Networks in Online App
Review System.

Faithfully ported from the TensorFlow-1.x (session/placeholder based, requires
tf.contrib) reference implementation in safe-graph/DGFraud:
  - algorithms/FdGars/FdGars.py   (FdGars model: multi-metapath GCN stack + classifier)
  - base_models/models.py         (GCN: 2-layer GraphConvolution stack, adapted from tkipf/gcn)
  - base_models/layers.py         (GraphConvolution layer)
  - base_models/inits.py          (glorot init)

The reference code cannot run in the base env: it is written against
TensorFlow 1.x graph-mode APIs (``tf.placeholder``, ``tf.variable_scope``,
``tf.Session``, ``tf.contrib.layers``) that were removed in TF2 and are not
installed here. This port transcribes the real computational graph -- per
metapath: two stacked GraphConvolution layers (glorot-init weight matmul with
the dense per-metapath adjacency, batch-normalization over the output,
L2-normalize + ReLU on the final layer) -- faithfully into a self-contained
torch.nn.Module, replacing only the TF1 plumbing (placeholders/session/graph
construction) with eager torch ops. The gather-by-batch-index + softmax
classification head at the end is preserved as in the reference
``forward_propagation``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def glorot_init(shape):
    """Glorot & Bengio (AISTATS 2010) init, matching base_models/inits.py::glorot."""
    init_range = (6.0 / (shape[0] + shape[1])) ** 0.5
    return nn.Parameter(torch.empty(*shape).uniform_(-init_range, init_range))


class GraphConvolution(nn.Module):
    """Port of base_models/layers.py::GraphConvolution.

    Reference _call(): dropout -> x @ W -> support @ (x @ W) -> batch-norm ->
    optional bias -> optional L2-normalize -> activation.
    """

    def __init__(self, input_dim, output_dim, act=F.relu, dropout=0.0, bias=False, norm=False):
        super().__init__()
        self.act = act
        self.dropout = dropout
        self.bias_flag = bias
        self.norm = norm
        self.weight = glorot_init((input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        # eps matches variance_epsilon=0.001 in the reference tf.nn.batch_normalization call
        self.bn = nn.BatchNorm1d(output_dim, eps=0.001, affine=False)

    def forward(self, x, support):
        # support: (nodes, nodes) dense per-metapath adjacency (self.support[self.index] in ref)
        x = F.dropout(x, p=self.dropout, training=self.training)
        pre_sup = x @ self.weight
        output = support @ pre_sup
        output = self.bn(output)
        if self.bias_flag:
            output = output + self.bias
        if self.norm:
            return F.normalize(self.act(output), p=2, dim=None, eps=1e-12)
        return self.act(output)


class GCN(nn.Module):
    """Port of base_models/models.py::GCN -- two stacked GraphConvolution layers
    for a single metapath, matching FdGars.forward_propagation()'s per-index GCN().
    """

    def __init__(self, input_dim, gcn_output1, output_dim):
        super().__init__()
        self.layer1 = GraphConvolution(input_dim, gcn_output1, act=F.relu, dropout=0.0, norm=False)
        self.layer2 = GraphConvolution(gcn_output1, output_dim, act=F.relu, dropout=0.0, norm=True)

    def forward(self, x, support):
        h = self.layer1(x, support)
        h = self.layer2(h, support)
        return h


class FdGarsModel(nn.Module):
    """Port of algorithms/FdGars/FdGars.py::FdGars.

    Parameters (matching the reference constructor):
        nodes: total node count
        class_size: number of output classes
        gcn_output1: first GCN layer's hidden unit count
        meta: number of metapaths (adjacency matrices)
        embedding: input node feature dim
        encoding: final GCN embedding dim per node (== classifier logits width
            reused as class_size in the reference softmax; kept faithful below)
    """

    def __init__(self, nodes, class_size, gcn_output1, meta, embedding, encoding):
        super().__init__()
        self.nodes = nodes
        self.meta = meta
        self.class_size = class_size
        self.encoding = encoding
        self.gcns = nn.ModuleList([GCN(embedding, gcn_output1, encoding) for _ in range(meta)])

    def forward(self, x, adj):
        """
        Args:
            x: (nodes, embedding) node feature matrix (placeholders['x'] in ref)
            adj: (meta, nodes, nodes) per-metapath dense adjacency (placeholders['a'] in ref)

        Returns: (batch_index_all_nodes, class_size) sigmoid probabilities, matching
        the reference forward_propagation() which gathers ALL nodes via
        one_hot(batch_index, nodes) @ gcn_emb, then softmax + sigmoid.
        """
        gcn_embs = []
        for i in range(self.meta):
            gcn_out = self.gcns[i](x, adj[i])  # (nodes, encoding)
            gcn_embs.append(gcn_out.reshape(1, self.nodes * self.encoding))
        gcn_emb = torch.cat(gcn_embs, dim=0)  # (meta, nodes*encoding)
        gcn_emb = gcn_emb.reshape(self.nodes, self.encoding)

        # classification: batch_data = one_hot(batch_index, nodes) @ gcn_emb
        # with batch_index = arange(nodes) this reduces to gcn_emb itself, but we
        # keep the explicit gather (as in the reference) for a faithful op trace.
        batch_index = torch.arange(self.nodes)
        one_hot = F.one_hot(batch_index, num_classes=self.nodes).float()
        batch_data = one_hot @ gcn_emb
        logits = F.softmax(batch_data, dim=-1)
        probabilities = torch.sigmoid(logits)
        return probabilities


def build_fdgars():
    # NOTE: the reference forward_propagation() does
    #   gcn_emb = tf.concat(gcn_emb, 0)              # (meta, nodes*encoding)
    #   gcn_emb = tf.reshape(gcn_emb, [nodes, encoding])
    # which is only shape-consistent when meta == 1 (a quirk of the original
    # reference code, faithfully preserved here rather than "fixed").
    nodes = 12
    meta = 1
    embedding = 8
    return FdGarsModel(
        nodes=nodes,
        class_size=nodes,
        gcn_output1=16,
        meta=meta,
        embedding=embedding,
        encoding=nodes,
    )


def example_input_fdgars():
    torch.manual_seed(0)
    nodes = 12
    meta = 1
    embedding = 8
    x = torch.randn(nodes, embedding)
    adj = torch.rand(meta, nodes, nodes)
    return (x, adj)


MENAGERIE_ENTRIES = [
    ("FdGars", "build_fdgars", "example_input_fdgars", 2019, "SOURCE_AVAILABLE"),
]
