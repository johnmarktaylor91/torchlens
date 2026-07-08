# SOURCE: vendored from https://github.com/henry-yeh/DeepACO @ 9a756a3fe6cf9627ecc166110a02fdbb61f2a0d7
# Original file: tsp/net.py (TSP variant of the DeepACO GNN heuristic-parameterization network)
# Imports/paths only were adjusted to be self-contained; architecture is verbatim.
import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data


# GNN for edge embeddings
class EmbNet(nn.Module):
    def __init__(self, depth=12, feats=2, units=32, act_fn="silu", agg_fn="mean"):  # TODO feats=1
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f"global_{agg_fn}_pool")
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(1, self.units)
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr):
        x = x
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)
        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + self.act_fn(
                self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0]))
            )
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return w


# general class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device

    def __init__(self, units_list, act_fn):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList(
            [nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)]
        )

    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            else:
                x = torch.sigmoid(x)  # last layer
        return x


# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, depth=3, units=32, preds=1, act_fn="silu"):
        self.units = units
        self.preds = preds
        super().__init__([self.units] * depth + [self.preds], act_fn)

    def forward(self, x):
        return super().forward(x).squeeze(dim=-1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_net = EmbNet()
        self.par_net_phe = ParNet()
        self.par_net_heu = ParNet()

    def forward(self, pyg):
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        emb = self.emb_net(x, edge_index, edge_attr)
        heu = self.par_net_heu(emb)
        return heu

    def freeze_gnn(self):
        for param in self.emb_net.parameters():
            param.requires_grad = False

    @staticmethod
    def reshape(pyg, vector):
        """Turn phe/heu vector into matrix with zero padding"""
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        matrix = torch.zeros(size=(n_nodes, n_nodes), device=device)
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = vector
        return matrix


# --- TorchLens menagerie staging harness (not part of the original repo) ---


def build_deepaco_tsp():
    return Net()


def example_input_deepaco_tsp():
    # Tiny synthetic fully-connected TSP instance: 6 nodes, 2D coordinates as
    # node features, edge_attr = Euclidean distance (matches EmbNet(feats=2)).
    n = 6
    torch.manual_seed(0)
    coords = torch.rand(n, 2)
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = (coords[edge_index[0]] - coords[edge_index[1]]).norm(dim=-1, keepdim=True)
    return (Data(x=coords, edge_index=edge_index, edge_attr=edge_attr),)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepACO-TSP", build_deepaco_tsp, example_input_deepaco_tsp, 2023, "vendored-pytorch"),
]
