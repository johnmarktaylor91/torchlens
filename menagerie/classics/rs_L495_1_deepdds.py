# SOURCE: vendored from Sinwang404/DeepDDS @ master (models/gat_gcn.py: GAT_GCN)
#
# DeepDDS: predicting synergistic drug combinations via a GNN over each drug's molecular
# graph fused with cell-line gene-expression features (Briefings in Bioinformatics 2022).
# This is the repo's GAT_GCN variant: a GATConv layer (10 attention heads) followed by a
# GCNConv layer over the drug's atom graph, global max+mean pooling to a fixed-size drug
# embedding, fused (via concatenation) with a 1D-conv-processed protein/cell feature branch,
# then dense layers to the synergy-score output. Copied verbatim from the real repo file;
# only the unused `heatmap.get_map` import (a plotting-only helper with no bearing on the
# forward pass, present in the sibling models/gat.py but not needed here) is omitted, and
# the drug2 duplicate forward call in the original multi-drug DDS setup is collapsed to a
# single drug input (this GAT_GCN class, as written in the repo, already takes one drug
# graph + one target/cell feature -- no architecture changes).
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import global_max_pool as gmp

MENAGERIE_ZOO = "vendored-pytorch"


# GCN-CNN based model
class GAT_GCN(torch.nn.Module):
    def __init__(
        self,
        n_output=1,
        num_features_xd=78,
        num_features_xt=25,
        n_filters=32,
        embed_dim=128,
        output_dim=128,
        dropout=0.2,
    ):
        super(GAT_GCN, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd * 10, num_features_xd * 10)
        self.fc_g1 = torch.nn.Linear(num_features_xd * 10 * 2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32 * 121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)  # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

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


def build_deepdds():
    return GAT_GCN(n_output=1, num_features_xd=78, num_features_xt=25)


def example_input_deepdds():
    # One molecular graph (drug), atom feature dim 78 (matches num_features_xd), plus a
    # 1000-length integer target/cell-feature sequence (matches conv_xt_1's in_channels=1000
    # and embedding_xt's num_features_xt=25 vocab).
    torch.manual_seed(0)
    num_atoms = 20
    x = torch.randn(num_atoms, 78)
    edge_index = torch.randint(0, num_atoms, (2, 40))
    target = torch.randint(0, 25, (1, 1000))
    data = Data(x=x, edge_index=edge_index)
    data.target = target
    batch = Batch.from_data_list([data])
    return (batch,)


MENAGERIE_ENTRIES = [
    ("DeepDDS", build_deepdds, example_input_deepdds, 2022, "SOURCE_AVAILABLE"),
]
