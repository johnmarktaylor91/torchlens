# SOURCE: vendored from thinng/GraphDTA @ master
#   models/gcn.py :: GCNNet
#
# GraphDTA (Nguyen et al., "GraphDTA: predicting drug-target binding affinity
# with graph neural networks", Bioinformatics 2021). Dual-branch architecture:
# a torch_geometric GCN tower reads the drug's molecular graph (atom features +
# bonds), a 1D-conv tower reads the target protein's residue-index sequence,
# and the two pooled representations are concatenated into an MLP regression
# head predicting binding affinity. Vendored verbatim (imports/formatting only
# adjusted); no architectural changes.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_max_pool as gmp

MENAGERIE_ZOO = "vendored-pytorch"


# ---- models/gcn.py (verbatim, GCN-based model) ----


class GCNNet(torch.nn.Module):
    def __init__(
        self,
        n_output=1,
        n_filters=32,
        embed_dim=128,
        num_features_xd=78,
        num_features_xt=25,
        output_dim=128,
        dropout=0.2,
    ):
        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd * 4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32 * 121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
        target = data.target

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)  # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # 1d conv layers
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out


def build_graphdta_gcn():
    torch.manual_seed(0)
    return GCNNet(
        n_output=1,
        n_filters=32,
        embed_dim=128,
        num_features_xd=78,
        num_features_xt=25,
        output_dim=128,
        dropout=0.0,
    ).eval()


def example_input_graphdta_gcn():
    torch.manual_seed(0)
    num_features_xd = 78
    # two tiny molecular graphs (drug SMILES graphs) batched via torch_geometric
    n_nodes_1, n_nodes_2 = 5, 7
    data_list = []
    for n_nodes in (n_nodes_1, n_nodes_2):
        x = torch.randn(n_nodes, num_features_xd)
        # simple path graph edges (undirected, both directions listed)
        src = list(range(n_nodes - 1)) + list(range(1, n_nodes))
        dst = list(range(1, n_nodes)) + list(range(n_nodes - 1))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        target = torch.randint(0, 25, (1000,), dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index, target=target.unsqueeze(0)))
    batch = Batch.from_data_list(data_list)
    # GraphDTA's dataloader collates `target` per-graph into (batch, 1000); Batch.from_data_list
    # concatenates along dim 0 since target was stored per-node-shaped -- reshape explicitly.
    batch.target = torch.stack([d.target.squeeze(0) for d in data_list], dim=0)
    return (batch,)


MENAGERIE_ENTRIES = [
    ("GraphDTA-GCN", build_graphdta_gcn, example_input_graphdta_gcn, 2021, "vendored-pytorch"),
]
