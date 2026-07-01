# SOURCE: vendored from OptiMaL-PSE-Lab/DeepDock @ main (deepdock/models.py)
# https://github.com/OptiMaL-PSE-Lab/DeepDock/blob/main/deepdock/models.py
#
# DeepDock: geometric deep learning model for protein-ligand binding pose
# prediction via a mixture-density-network over pairwise ligand-target
# distances. Two graph encoders (LigandNet, TargetNet), each a stack of
# MetaLayer (edge+node message passing) blocks followed by a residual
# MetaLayer stack, feed a combined MLP + mixture-density output head.
#
# Only import/name-path fixes were made (deepdock.utils.distributions ->
# inlined MixtureSameFamily import removed since it is unused by the
# forward pass; deepdock.* relative imports flattened to this single file).
# The architecture itself (ResBlock, EdgeModel, NodeModel, TargetNet,
# LigandNet, DeepDock) is transcribed unmodified from the real repo.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import MetaLayer
from torch_geometric.data import Data


def compute_cluster_batch_index(cluster, batch):
    max_prev_batch = 0
    for i in range(batch.max().item() + 1):
        cluster[batch == i] += max_prev_batch
        max_prev_batch = cluster[batch == i].max().item() + 1
    return cluster


class NodeSampling(nn.Module):
    def __init__(self, nodes_per_graph):
        super(NodeSampling, self).__init__()

        self.num = nodes_per_graph

    def forward(self, x):
        if self.training:
            max_prev_batch = 0
            idx = []
            counts = torch.unique(x.batch, return_counts=True)[1]

            for i in range(x.batch.max().item() + 1):
                idx.append(torch.randperm(counts[i])[: self.num] + max_prev_batch)
                max_prev_batch += counts[i]
            idx = torch.cat(idx)

            x.batch = x.batch[idx]
            x.pos = x.pos[idx]
            x.x = x.x[idx]

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.15):
        super(ResBlock, self).__init__()

        self.projectDown_node = nn.Linear(in_channels, in_channels // 4)
        self.projectDown_edge = nn.Linear(in_channels, in_channels // 4)
        self.bn1_node = nn.BatchNorm1d(in_channels // 4)
        self.bn1_edge = nn.BatchNorm1d(in_channels // 4)

        self.conv = MetaLayer(
            edge_model=EdgeModel(in_channels // 4),
            node_model=NodeModel(in_channels // 4),
            global_model=None,
        )

        self.projectUp_node = nn.Linear(in_channels // 4, in_channels)
        self.projectUp_edge = nn.Linear(in_channels // 4, in_channels)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2_node = nn.BatchNorm1d(in_channels)
        nn.init.zeros_(self.bn2_node.weight)
        self.bn2_edge = nn.BatchNorm1d(in_channels)
        nn.init.zeros_(self.bn2_edge.weight)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        h_node = F.elu(self.bn1_node(self.projectDown_node(x)))
        h_edge = F.elu(self.bn1_edge(self.projectDown_edge(edge_attr)))
        h_node, h_edge, _ = self.conv(h_node, edge_index, h_edge, None, batch)

        h_node = self.dropout(self.bn2_node(self.projectUp_node(h_node)))
        data.x = F.elu(h_node + x)

        h_edge = self.dropout(self.bn2_edge(self.projectUp_edge(h_edge)))
        data.edge_attr = F.elu(h_edge + edge_attr)

        return data


class EdgeModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(EdgeModel, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels * 3, in_channels), nn.BatchNorm1d(in_channels), nn.ELU()
        )

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels), nn.BatchNorm1d(in_channels), nn.ELU()
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels), nn.BatchNorm1d(in_channels), nn.ELU()
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class TargetNet(nn.Module):
    def __init__(
        self, in_channels, edge_features=3, hidden_dim=128, residual_layers=20, dropout_rate=0.15
    ):
        super(TargetNet, self).__init__()

        self.node_encoder = nn.Linear(in_channels, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        self.conv1 = MetaLayer(
            edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None
        )
        self.conv2 = MetaLayer(
            edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None
        )
        self.conv3 = MetaLayer(
            edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None
        )
        layers = [
            ResBlock(in_channels=hidden_dim, dropout_rate=dropout_rate)
            for i in range(residual_layers)
        ]
        self.resnet = nn.Sequential(*layers)

    def forward(self, data):
        data.edge_attr = None
        data = T.Cartesian(norm=False, max_value=None, cat=False)(data)

        data.x = self.node_encoder(data.x)
        data.edge_attr = self.edge_encoder(data.edge_attr)
        data.x, data.edge_attr, _ = self.conv1(
            data.x, data.edge_index, data.edge_attr, None, data.batch
        )
        data.x, data.edge_attr, _ = self.conv2(
            data.x, data.edge_index, data.edge_attr, None, data.batch
        )
        data.x, data.edge_attr, _ = self.conv3(
            data.x, data.edge_index, data.edge_attr, None, data.batch
        )
        data = self.resnet(data)

        return data


class LigandNet(nn.Module):
    def __init__(
        self, in_channels, edge_features=6, hidden_dim=128, residual_layers=20, dropout_rate=0.15
    ):
        super(LigandNet, self).__init__()

        self.node_encoder = nn.Linear(in_channels, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        self.conv1 = MetaLayer(
            edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None
        )
        self.conv2 = MetaLayer(
            edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None
        )
        self.conv3 = MetaLayer(
            edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None
        )
        layers = [
            ResBlock(in_channels=hidden_dim, dropout_rate=dropout_rate)
            for i in range(residual_layers)
        ]
        self.resnet = nn.Sequential(*layers)

    def forward(self, data):
        data.x = self.node_encoder(data.x)
        data.edge_attr = self.edge_encoder(data.edge_attr)
        data.x, data.edge_attr, _ = self.conv1(
            data.x, data.edge_index, data.edge_attr, None, data.batch
        )
        data.x, data.edge_attr, _ = self.conv2(
            data.x, data.edge_index, data.edge_attr, None, data.batch
        )
        data.x, data.edge_attr, _ = self.conv3(
            data.x, data.edge_index, data.edge_attr, None, data.batch
        )
        data = self.resnet(data)

        return data


class DeepDock(nn.Module):
    def __init__(
        self,
        ligand_model,
        target_model,
        hidden_dim,
        n_gaussians,
        dropout_rate=0.15,
        nodes_per_target=None,
        dist_threhold=1000,
    ):
        super(DeepDock, self).__init__()

        self.ligand_model = ligand_model
        self.target_model = target_model
        if nodes_per_target:
            self.node_sampling = NodeSampling(nodes_per_graph=nodes_per_target)
        else:
            self.node_sampling = None
        self.MLP = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
        )

        self.z_pi = nn.Linear(hidden_dim, n_gaussians)
        self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
        self.z_mu = nn.Linear(hidden_dim, n_gaussians)
        self.atom_types = nn.Linear(128, 28)
        self.bond_types = nn.Linear(256, 6)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dist_threhold = dist_threhold

    def forward(self, data_ligand, data_target, y=None):
        h_l = self.ligand_model(data_ligand)
        h_t = self.target_model(data_target)
        if self.node_sampling:
            h_t = self.node_sampling(h_t)

        h_l_x, l_mask = to_dense_batch(h_l.x, h_l.batch, fill_value=0)
        h_t_x, t_mask = to_dense_batch(h_t.x, h_t.batch, fill_value=0)
        h_l_pos, _ = to_dense_batch(h_l.pos, h_l.batch, fill_value=0)
        h_t_pos, _ = to_dense_batch(h_t.pos, h_t.batch, fill_value=0)

        assert h_l_x.size(0) == h_t_x.size(0), "Encountered unequal batch-sizes"
        (B, N_l, C_out), N_t = h_l_x.size(), h_t_x.size(1)  # noqa: F841 (C_out unused in real repo code)

        # Combine and mask
        h_l_x = h_l_x.unsqueeze(-2)
        h_l_x = h_l_x.repeat(1, 1, N_t, 1)  # [B, N_l, N_t, C_out]

        h_t_x = h_t_x.unsqueeze(-3)
        h_t_x = h_t_x.repeat(1, N_l, 1, 1)  # [B, N_l, N_t, C_out]

        C = torch.cat((h_l_x, h_t_x), -1)
        self.C_mask = C_mask = l_mask.view(B, N_l, 1) & t_mask.view(B, 1, N_t)
        self.C = C = C[C_mask]
        C = self.MLP(C)

        # Get batch indexes for ligand-target combined features
        C_batch = torch.tensor(range(B)).unsqueeze(-1).unsqueeze(-1)
        C_batch = C_batch.repeat(1, N_l, N_t)[C_mask].to(self.device)

        # Outputs
        pi = F.softmax(self.z_pi(C), -1)
        sigma = F.elu(self.z_sigma(C)) + 1.1
        mu = F.elu(self.z_mu(C)) + 1
        dist = self.compute_euclidean_distances_matrix(h_l_pos, h_t_pos)[C_mask]
        atom_types = self.atom_types(h_l.x)
        bond_types = self.bond_types(
            torch.cat([h_l.x[h_l.edge_index[0]], h_l.x[h_l.edge_index[1]]], axis=1)
        )

        return pi, sigma, mu, dist.unsqueeze(1).detach(), atom_types, bond_types, C_batch

    def compute_euclidean_distances_matrix(self, X, Y):
        # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        # (X-Y)^2 = X^2 + Y^2 -2XY
        X = X.double()
        Y = Y.double()

        dists = (
            -2 * torch.bmm(X, Y.permute(0, 2, 1))
            + torch.sum(Y**2, axis=-1).unsqueeze(1)
            + torch.sum(X**2, axis=-1).unsqueeze(-1)
        )
        return dists**0.5


# ---------------------------------------------------------------------------
# Menagerie staging entries
# ---------------------------------------------------------------------------


def build_deepdock():
    torch.manual_seed(0)
    # hidden_dim must stay 128 for both encoders: DeepDock.MLP hardcodes an
    # input width of 256 = ligand_hidden_dim + target_hidden_dim (real repo
    # constant, see models.py `self.MLP = nn.Sequential(nn.Linear(256, ...`).
    ligand_model = LigandNet(in_channels=17, edge_features=6, hidden_dim=128, residual_layers=1)
    target_model = TargetNet(in_channels=6, edge_features=3, hidden_dim=128, residual_layers=1)
    model = DeepDock(
        ligand_model, target_model, hidden_dim=64, n_gaussians=10, nodes_per_target=None
    )
    model.eval()
    return model


def example_input_deepdock():
    torch.manual_seed(0)

    def make_graph(n_nodes, in_channels, edge_features, n_edges):
        x = torch.randn(n_nodes, in_channels)
        pos = torch.randn(n_nodes, 3)
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        edge_attr = torch.randn(n_edges, edge_features)
        batch = torch.zeros(n_nodes, dtype=torch.long)
        return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

    data_ligand = make_graph(n_nodes=8, in_channels=17, edge_features=6, n_edges=14)
    data_target = make_graph(n_nodes=10, in_channels=6, edge_features=3, n_edges=18)
    return (data_ligand, data_target)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    ("DeepDock", "build_deepdock", "example_input_deepdock", 2022, "vendored-pytorch"),
]
