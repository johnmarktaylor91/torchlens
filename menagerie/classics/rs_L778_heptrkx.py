# SOURCE: vendored from https://github.com/HEPTrkX/heptrkx-gnn-tracking @ master
# (models/gnn.py)
#
# HEPTrkX: message-passing graph neural network for particle-track
# segment/hit classification in high-energy physics detector data (the
# HEP.TrkX collaboration, precursor to the exatrkx line of work). The model
# alternates an EdgeNetwork (scores candidate hit-hit segments) and a
# NodeNetwork (aggregates weighted in/out neighbor messages into new hit
# features) over dense bipartite adjacency matrices Ri/Ro, rather than a
# torch_geometric sparse edge_index -- this is the real repo's own
# representation (see docstrings below), not a menagerie simplification. No
# architecture was altered; only the module-level docstring/imports were kept
# as-is since the file already has zero repo-relative imports (pure torch/nn).

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class EdgeNetwork(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """

    def __init__(self, input_dim, hidden_dim=8, hidden_activation=nn.Tanh):
        super(EdgeNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            hidden_activation(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, X, Ri, Ro):
        # Select the features of the associated nodes
        bo = torch.bmm(Ro.transpose(1, 2), X)
        bi = torch.bmm(Ri.transpose(1, 2), X)
        B = torch.cat([bo, bi], dim=2)
        # Apply the network to each edge
        return self.network(B).squeeze(-1)


class NodeNetwork(nn.Module):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """

    def __init__(self, input_dim, output_dim, hidden_activation=nn.Tanh):
        super(NodeNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim * 3, output_dim),
            hidden_activation(),
            nn.Linear(output_dim, output_dim),
            hidden_activation(),
        )

    def forward(self, X, e, Ri, Ro):
        bo = torch.bmm(Ro.transpose(1, 2), X)
        bi = torch.bmm(Ri.transpose(1, 2), X)
        Rwo = Ro * e[:, None]
        Rwi = Ri * e[:, None]
        mi = torch.bmm(Rwi, bo)
        mo = torch.bmm(Rwo, bi)
        M = torch.cat([mi, mo, X], dim=2)
        return self.network(M)


class GNNSegmentClassifier(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """

    def __init__(self, input_dim=2, hidden_dim=8, n_iters=3, hidden_activation=nn.Tanh):
        super(GNNSegmentClassifier, self).__init__()
        self.n_iters = n_iters
        # Setup the input network
        self.input_network = nn.Sequential(nn.Linear(input_dim, hidden_dim), hidden_activation())
        # Setup the edge network
        self.edge_network = EdgeNetwork(input_dim + hidden_dim, hidden_dim, hidden_activation)
        # Setup the node layers
        self.node_network = NodeNetwork(input_dim + hidden_dim, hidden_dim, hidden_activation)

    def forward(self, inputs):
        """Apply forward pass of the model"""
        X, Ri, Ro = inputs
        # Apply input network to get hidden representation
        H = self.input_network(X)
        # Shortcut connect the inputs onto the hidden representation
        H = torch.cat([H, X], dim=-1)
        # Loop over iterations of edge and node networks
        for i in range(self.n_iters):
            # Apply edge network
            e = self.edge_network(H, Ri, Ro)
            # Apply node network
            H = self.node_network(H, e, Ri, Ro)
            # Shortcut connect the inputs onto the hidden representation
            H = torch.cat([H, X], dim=-1)
        # Apply final edge network
        return self.edge_network(H, Ri, Ro)


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------

_N_HITS = 12
_N_EDGES = 20
_INPUT_DIM = 2  # (r, phi) or (r, z) hit coordinates, per the repo's datasets/hitgraphs.py
_HIDDEN_DIM = 8


def build_heptrkx():
    torch.manual_seed(0)
    model = GNNSegmentClassifier(input_dim=_INPUT_DIM, hidden_dim=_HIDDEN_DIM, n_iters=3)
    model.eval()
    return model


def example_input_heptrkx():
    torch.manual_seed(0)
    # X: hit features (batch, n_hits, input_dim)
    X = torch.randn(1, _N_HITS, _INPUT_DIM)
    # Ri, Ro: dense bipartite incidence matrices (batch, n_hits, n_edges) --
    # the real repo's `datasets/graph.py` builds exactly this representation
    # (one-hot per edge: which hit is the "in" node, which is the "out" node).
    Ri = torch.zeros(1, _N_HITS, _N_EDGES)
    Ro = torch.zeros(1, _N_HITS, _N_EDGES)
    for edge in range(_N_EDGES):
        in_hit = edge % _N_HITS
        out_hit = (edge + 1) % _N_HITS
        Ri[0, in_hit, edge] = 1.0
        Ro[0, out_hit, edge] = 1.0
    return ((X, Ri, Ro),)


MENAGERIE_ENTRIES = [
    (
        "HEPTrkX GNN Segment Classifier",
        "build_heptrkx",
        "example_input_heptrkx",
        2019,
        MENAGERIE_ZOO,
    ),
]
