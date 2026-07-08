# SOURCE: vendored from https://github.com/HEPTrkX/heptrkx-gnn-tracking @ master (c622ed038244)
# (models/gnn.py's EdgeNetwork + NodeNetwork + GNNSegmentClassifier)
#
# HEPTrkX GNN segment classifier (Ju et al., "Graph Neural Networks for
# Particle Reconstruction in High Energy Physics detectors", CTD 2018,
# arXiv:1810.06111), built for the TrackML particle-tracking challenge.
# Classifies candidate track "segments" (edges between detector hits) as
# real-vs-fake using a recurrent message-passing GNN: an EdgeNetwork scores
# edges from the incidence-matrix-selected endpoint features, a NodeNetwork
# aggregates weighted-by-edge-score neighbor features (via the same
# incidence matrices Ri/Ro) into updated node features, and the two networks
# are applied `n_iters` times with an input-feature shortcut concatenated
# back in after each iteration, before a final EdgeNetwork pass emits the
# per-edge segment-classification score. No architecture altered.

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


def build_trackml_gnn():
    torch.manual_seed(0)
    model = GNNSegmentClassifier(input_dim=2, hidden_dim=8, n_iters=2)
    model.eval()
    return model


def example_input_trackml_gnn():
    torch.manual_seed(0)
    batch_size = 1
    n_nodes = 10
    n_edges = 16
    input_dim = 2

    X = torch.randn(batch_size, n_nodes, input_dim)
    # Ri/Ro are dense incidence matrices (node x edge) mapping each edge to
    # its incoming (Ri) / outgoing (Ro) node, one-hot per edge column.
    Ri = torch.zeros(batch_size, n_nodes, n_edges)
    Ro = torch.zeros(batch_size, n_nodes, n_edges)
    torch.manual_seed(1)
    in_idx = torch.randint(0, n_nodes, (n_edges,))
    out_idx = torch.randint(0, n_nodes, (n_edges,))
    for e in range(n_edges):
        Ri[0, in_idx[e], e] = 1.0
        Ro[0, out_idx[e], e] = 1.0

    return ((X, Ri, Ro),)


MENAGERIE_ENTRIES = [
    (
        "HEPTrkX GNN (TrackML segment classifier)",
        "build_trackml_gnn",
        "example_input_trackml_gnn",
        2018,
        MENAGERIE_ZOO,
    ),
]
