# SOURCE: vendored from shiwentao00/Graphsite-classifier @ dcba38916c8e2623941f2aa0646754e8cae3d579
# (gnn/model.py::GraphsiteClassifier + JKMCNWMEmbeddingNet + MCNWMConv + NWMConv)
"""Graphsite-classifier: GNN classifier for ligand-binding-pocket classification
over protein-structure graphs (Shi, W. 2020; https://github.com/shiwentao00/Graphsite-classifier).

``GraphsiteClassifier`` dispatches to one of five embedding-net architectures
selected by the ``which_model`` string ('residual' / 'jk' / 'pna' / 'jknwm' /
'jkgin', default 'normal'). The actually-trained configuration -- per the
repo's own ``gnn/train_classifier.yaml`` (``which_model: 'jknwm'``, used by
``gnn/train_classifier.py``) -- is the "jumping knowledge" variant with
multi-channel neural-weighted-message (MCNWM) graph-conv layers, so this
module wires up ``JKMCNWMEmbeddingNet`` (which composes ``MCNWMConv`` ->
``NWMConv``, a custom edge-attribute-gated ``MessagePassing`` layer) and the
top-level ``GraphsiteClassifier`` dispatcher class.

Code below is copied verbatim from the official repo's ``GraphsiteClassifier``,
``JKMCNWMEmbeddingNet``, ``MCNWMConv``, and ``NWMConv`` classes (the other
``which_model`` branches -- ``ResidualEmbeddingNet``, ``JKEmbeddingNet``,
``PNAEmbeddingNet``, ``JKEGINEmbeddingNet``, ``EmbeddingNet``, and the
Siamese/contrastive-loss training-only classes -- are dropped here since they
are unused by the actually-trained model and this module only stages the
'jknwm' construction path; none of the retained architecture logic is
altered). One `exec(...)`-based dynamic-attribute-assignment idiom in
``JKMCNWMEmbeddingNet.__init__`` (used by the original authors to programmatically
name ``self.conv{i}``/``self.bn{i}`` per layer) is preserved verbatim.
"""

from torch import Tensor
import torch
import torch.nn.functional as F
from torch.nn import Linear, LeakyReLU, ELU, ModuleList, Sequential
from torch_geometric.nn import MessagePassing, Set2Set

MENAGERIE_ZOO = "vendored-pytorch"


class NWMConv(MessagePassing):
    """The neural weighted message (NWM) layer. Output of multiple instances of
    this will produce multi-channel output."""

    def __init__(self, num_edge_attr=1, train_eps=True, eps=0):
        super(NWMConv, self).__init__(aggr="add")
        self.edge_nn = Sequential(Linear(num_edge_attr, 8), LeakyReLU(), Linear(8, 1), ELU())
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

    def forward(self, x, edge_index, edge_attr, size=None):
        # x: OptPairTensor
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return out

    def message(self, x_j, edge_attr):
        weight = self.edge_nn(edge_attr)

        # message size: num_features or dim
        # weight size: 1
        # all the dimensions in a node masked by one weight generated from edge attribute
        return x_j * weight

    def __repr__(self):
        return "{}(edge_nn={})".format(self.__class__.__name__, self.edge_nn)


class MCNWMConv(torch.nn.Module):
    """Multi-channel neural weighted message module."""

    def __init__(self, in_dim, out_dim, num_channels, num_edge_attr=1, train_eps=True, eps=0):
        super(MCNWMConv, self).__init__()
        self.nn = Sequential(
            Linear(in_dim * num_channels, out_dim), LeakyReLU(), Linear(out_dim, out_dim)
        )
        self.NMMs = ModuleList()

        # add the message passing modules
        for _ in range(num_channels):
            self.NMMs.append(NWMConv(num_edge_attr, train_eps, eps))

    def forward(self, x, edge_index, edge_attr):
        # compute the aggregated information for each channel
        channels = []
        for nmm in self.NMMs:
            channels.append(nmm(x=x, edge_index=edge_index, edge_attr=edge_attr))

        # concatenate output of each channel
        x = torch.cat(channels, dim=1)

        # use the neural network to shrink dimension back
        x = self.nn(x)

        return x


class JKMCNWMEmbeddingNet(torch.nn.Module):
    """
    Jumping knowledge embedding net inspired by the paper "Representation Learning on
    Graphs with Jumping Knowledge Networks".

    The GNN layers are now MCNWMConv layer.
    """

    def __init__(
        self,
        num_features,
        dim,
        train_eps,
        num_edge_attr,
        num_layers,
        num_channels=1,
        layer_aggregate="max",
    ):
        super(JKMCNWMEmbeddingNet, self).__init__()
        self.num_layers = num_layers
        self.layer_aggregate = layer_aggregate

        # first layer
        self.conv0 = MCNWMConv(
            in_dim=num_features,
            out_dim=dim,
            num_channels=num_channels,
            num_edge_attr=num_edge_attr,
            train_eps=train_eps,
        )
        self.bn0 = torch.nn.BatchNorm1d(dim)

        # rest of the layers
        for i in range(1, self.num_layers):
            exec(
                "self.conv{} = MCNWMConv(in_dim=dim, out_dim=dim, num_channels={}, num_edge_attr=num_edge_attr, train_eps=train_eps)".format(
                    i, num_channels
                )
            )
            exec("self.bn{} = torch.nn.BatchNorm1d(dim)".format(i))

        # read out function
        self.set2set = Set2Set(in_channels=dim, processing_steps=5, num_layers=2)

    def forward(self, x, edge_index, edge_attr, batch):
        # GNN layers
        layer_x = []  # jumping knowledge
        for i in range(0, self.num_layers):
            conv = getattr(self, "conv{}".format(i))
            bn = getattr(self, "bn{}".format(i))
            x = F.leaky_relu(conv(x, edge_index, edge_attr))
            x = bn(x)
            layer_x.append(x)

        # layer aggregation
        if self.layer_aggregate == "max":
            x = torch.stack(layer_x, dim=0)
            x = torch.max(x, dim=0)[0]
        elif self.layer_aggregate == "mean":
            x = torch.stack(layer_x, dim=0)
            x = torch.mean(x, dim=0)[0]

        # graph readout
        x = self.set2set(x, batch)

        return x


class GraphsiteClassifier(torch.nn.Module):
    """
    Standard classifier to classify the binding sites.
    """

    def __init__(
        self,
        num_classes,
        num_features,
        dim,
        train_eps,
        num_edge_attr,
        which_model,
        num_layers,
        num_channels,
        deg=None,
    ):
        """
        train_eps: for the SCNWMConv module only when which_model in
        ['jk', 'residual', 'jknmm', and 'normal'].
        deg: for PNAEmbeddingNet only, can not be None when which_model=='pna'.
        """
        super(GraphsiteClassifier, self).__init__()
        self.num_classes = num_classes

        # use one of the embedding net (this staging module only wires up the
        # actually-trained 'jknwm' branch -- see module docstring)
        if which_model == "jknwm":
            self.embedding_net = JKMCNWMEmbeddingNet(
                num_features=num_features,
                dim=dim,
                train_eps=train_eps,
                num_edge_attr=num_edge_attr,
                num_layers=num_layers,
                num_channels=num_channels,
            )
        else:
            raise ValueError(
                "This staging module only wires up which_model='jknwm' "
                "(the actually-trained configuration)."
            )

        # set2set doubles the size of embeddnig
        self.fc1 = Linear(2 * dim, dim)
        self.fc2 = Linear(dim, self.num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.embedding_net(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        # returned tensor should be processed by a softmax layer
        return x


# --- staging harness -------------------------------------------------------


def build_graphsiteclassifier():
    # Real hyperparameters from gnn/train_classifier.yaml (which_model:
    # 'jknwm', model_size=96, num_layers=6, num_channels=3) and
    # gnn/train_classifier.py's GraphsiteClassifier(...) call (train_eps=True,
    # num_edge_attr=1, num_features=len(features_to_use)=11, num_classes=
    # len(clusters after merge_info) = 14), sized down for tracing (dim=16,
    # num_layers=2).
    return GraphsiteClassifier(
        num_classes=14,
        num_features=11,
        dim=16,
        train_eps=True,
        num_edge_attr=1,
        which_model="jknwm",
        num_layers=2,
        num_channels=3,
    ).eval()


def example_input_graphsiteclassifier():
    # A small batch of 2 graphs (protein-pocket point clouds), matching the
    # real PyG Data(x, edge_index, edge_attr, batch) layout consumed by
    # GraphsiteClassifier.forward.
    num_features = 11
    num_edge_attr = 1

    # Graph 0: 4 nodes, Graph 1: 3 nodes.
    x = torch.rand(7, num_features)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)

    # A small ring + a few chords within each graph (undirected, stored as
    # both directions), no cross-graph edges.
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 4],
            [1, 0, 2, 1, 3, 2, 0, 3, 5, 4, 6, 5, 4, 6],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.rand(edge_index.shape[1], num_edge_attr)

    return (x, edge_index, edge_attr, batch)


MENAGERIE_ENTRIES = [
    (
        "GraphsiteClassifier",
        "build_graphsiteclassifier",
        "example_input_graphsiteclassifier",
        2020,
        "vendored",
    ),
]
