# SOURCE: vendored from wenqi006/SlideGraph @ main
# https://raw.githubusercontent.com/wenqi006/SlideGraph/main/GNN_pr.py
#
# "SlideGraph+: Whole Slide Image Level Graphs to Predict HER2 Status in Breast
# Cancer" (Lu, Minhas et al., Medical Image Analysis 2022). SlideGraph builds a
# whole-slide-image-level graph from nuclei/tile topology and classifies it
# with a multi-layer GIN/EdgeConv graph network with per-layer readout pooling
# (JK-style: predictions from every layer are pooled and summed, either by
# pooling node CLASSIFICATION SCORES or by pooling node EMBEDDINGS before the
# final linear layer, selected by `gembed`). The `GNN` class below is
# transcribed VERBATIM from GNN_pr.py (only the training-loop / evaluation
# helpers -- NetWrapper, decision_function, EnsembleDecisionScoring,
# StratifiedSampler -- and their sklearn/tqdm imports, which are not part of
# the model architecture, are omitted; no change to the GNN module itself).
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import EdgeConv, GINConv, global_add_pool, global_max_pool, global_mean_pool

MENAGERIE_ZOO = "vendored-pytorch"


# ---- GNN_pr.py (verbatim) ----
class GNN(torch.nn.Module):
    def __init__(
        self,
        dim_features,
        dim_target,
        layers=[6, 6],
        pooling="max",
        dropout=0.0,
        conv="GINConv",
        gembed=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        dim_features : TYPE Int
            DESCRIPTION. Number of features of each node
        dim_target : TYPE Int
            DESCRIPTION. Number of outputs
        layers : TYPE, optional List of number of nodes in each layer
            DESCRIPTION. The default is [6,6].
        pooling : TYPE, optional
            DESCRIPTION. The default is 'max'.
        dropout : TYPE, optional
            DESCRIPTION. The default is 0.0.
        conv : TYPE, optional Layer type string {'GINConv','EdgeConv'} supported
            DESCRIPTION. The default is 'GINConv'.
        gembed : TYPE, optional Graph Embedding
            DESCRIPTION. The default is False. Pool node scores or pool node features
        **kwargs : TYPE
            DESCRIPTION.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super(GNN, self).__init__()
        self.dropout = dropout
        self.embeddings_dim = layers
        self.no_layers = len(self.embeddings_dim)
        self.first_h = []
        self.nns = []
        self.convs = []
        self.linears = []
        self.pooling = {"max": global_max_pool, "mean": global_mean_pool, "add": global_add_pool}[
            pooling
        ]
        self.gembed = gembed  # if True then learn graph embedding for final classification (classify pooled node features) otherwise pool node decision scores

        for layer, out_emb_dim in enumerate(self.embeddings_dim):
            if layer == 0:
                self.first_h = Sequential(
                    Linear(dim_features, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()
                )
                self.linears.append(Linear(out_emb_dim, dim_target))

            else:
                input_emb_dim = self.embeddings_dim[layer - 1]
                self.linears.append(Linear(out_emb_dim, dim_target))
                if conv == "GINConv":
                    subnet = Sequential(
                        Linear(input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()
                    )
                    self.nns.append(subnet)
                    self.convs.append(
                        GINConv(self.nns[-1], **kwargs)
                    )  # Eq. 4.2 eps=100, train_eps=False
                elif conv == "EdgeConv":
                    subnet = Sequential(
                        Linear(2 * input_emb_dim, out_emb_dim), BatchNorm1d(out_emb_dim), ReLU()
                    )
                    self.nns.append(subnet)
                    self.convs.append(
                        EdgeConv(self.nns[-1], **kwargs)
                    )  # DynamicEdgeConv#EdgeConv                aggr='mean'

                else:
                    raise NotImplementedError

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        self.linears = torch.nn.ModuleList(self.linears)  # has got one more for initial input

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = 0
        pooling = self.pooling
        Z = 0
        for layer in range(self.no_layers):
            if layer == 0:
                x = self.first_h(x)
                z = self.linears[layer](x)
                Z += z
                dout = F.dropout(pooling(z, batch), p=self.dropout, training=self.training)
                out += dout
            else:
                x = self.convs[layer - 1](x, edge_index)
                if not self.gembed:
                    z = self.linears[layer](x)
                    Z += z
                    dout = F.dropout(pooling(z, batch), p=self.dropout, training=self.training)
                else:
                    dout = F.dropout(
                        self.linears[layer](pooling(x, batch)),
                        p=self.dropout,
                        training=self.training,
                    )
                out += dout
        return out, Z, x


def build_slidegraph():
    torch.manual_seed(0)
    return GNN(
        dim_features=8,
        dim_target=2,
        layers=[6, 6, 6],
        pooling="mean",
        dropout=0.0,
        conv="GINConv",
        gembed=False,
    )


def example_input_slidegraph():
    torch.manual_seed(0)
    n_nodes_per_graph = [12, 9]
    n_nodes = sum(n_nodes_per_graph)
    dim_features = 8

    x = torch.randn(n_nodes, dim_features)

    edges = []
    offset = 0
    for n in n_nodes_per_graph:
        for i in range(n - 1):
            edges.append((offset + i, offset + i + 1))
            edges.append((offset + i + 1, offset + i))
        offset += n
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    batch = torch.cat(
        [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(n_nodes_per_graph)]
    )

    data = Data(x=x, edge_index=edge_index, batch=batch)
    return (data,)


MENAGERIE_ENTRIES = [
    ("SlideGraph", "build_slidegraph", "example_input_slidegraph", 2022, "vendored"),
]
