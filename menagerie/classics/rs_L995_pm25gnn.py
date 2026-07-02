# SOURCE: vendored from shuowang-ai/PM2.5-GNN @ 471fc60775f80492f4f224203d172868bc6eebac
# (model/cells.py, model/PM25_GNN.py)
#
# PM2.5-GNN: domain-knowledge-enhanced graph neural network for city-level PM2.5
# forecasting (Wang et al., ACM SIGSPATIAL 2020). GraphGNN encodes a fixed city-graph
# with a physically-motivated edge weight (wind speed/direction projected onto the
# inter-city bearing) via an edge MLP + torch_scatter aggregation; PM25_GNN autoregresses
# a GRUCell (custom, torch-native GRU cell reimplementation from cells.py -- not
# nn.GRUCell) over the graph-augmented features for pred_len steps. GRUCell/PM25_GNN/
# GraphGNN below are the REAL model code from the listed files, copied verbatim. Only
# base-lib deps: torch, numpy, torch_scatter (installed).

import numpy as np
import torch
from torch import nn
from torch.nn import Sequential, Linear, Sigmoid
from torch.nn import functional as F
from torch.nn import Parameter
from torch_scatter import scatter_add  # , scatter_sub  # no scatter sub in lastest PyG


# --- model/cells.py -----------------------------------------------------------


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(-1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy


# --- model/PM25_GNN.py ---------------------------------------------------------


class GraphGNN(nn.Module):
    def __init__(self, device, edge_index, edge_attr, in_dim, out_dim, wind_mean, wind_std):
        super(GraphGNN, self).__init__()
        self.device = device
        self.edge_index = torch.LongTensor(edge_index).to(self.device)
        self.edge_attr = torch.Tensor(np.float32(edge_attr))
        self.edge_attr_norm = (self.edge_attr - self.edge_attr.mean(dim=0)) / self.edge_attr.std(
            dim=0
        )
        self.w = Parameter(torch.rand([1]))
        self.b = Parameter(torch.rand([1]))
        self.wind_mean = torch.Tensor(np.float32(wind_mean)).to(self.device)
        self.wind_std = torch.Tensor(np.float32(wind_std)).to(self.device)
        e_h = 32
        e_out = 30
        n_out = out_dim
        self.edge_mlp = Sequential(
            Linear(in_dim * 2 + 2 + 1, e_h),
            Sigmoid(),
            Linear(e_h, e_out),
            Sigmoid(),
        )
        self.node_mlp = Sequential(
            Linear(e_out, n_out),
            Sigmoid(),
        )

    def forward(self, x):
        self.edge_index = self.edge_index.to(self.device)
        self.edge_attr = self.edge_attr.to(self.device)
        self.w = self.w.to(self.device)
        self.b = self.b.to(self.device)

        edge_src, edge_target = self.edge_index
        node_src = x[:, edge_src]
        node_target = x[:, edge_target]

        src_wind = (
            node_src[:, :, -2:] * self.wind_std[None, None, :] + self.wind_mean[None, None, :]
        )
        src_wind_speed = src_wind[:, :, 0]
        src_wind_direc = src_wind[:, :, 1]
        self.edge_attr_ = self.edge_attr[None, :, :].repeat(node_src.size(0), 1, 1)
        city_dist = self.edge_attr_[:, :, 0]
        city_direc = self.edge_attr_[:, :, 1]

        theta = torch.abs(city_direc - src_wind_direc)
        edge_weight = F.relu(3 * src_wind_speed * torch.cos(theta) / city_dist)
        edge_weight = edge_weight.to(self.device)
        edge_attr_norm = (
            self.edge_attr_norm[None, :, :].repeat(node_src.size(0), 1, 1).to(self.device)
        )
        out = torch.cat([node_src, node_target, edge_attr_norm, edge_weight[:, :, None]], dim=-1)

        out = self.edge_mlp(out)
        out_add = scatter_add(out, edge_target, dim=1, dim_size=x.size(1))
        # out_sub = scatter_sub(out, edge_src, dim=1, dim_size=x.size(1))
        out_sub = scatter_add(
            out.neg(), edge_src, dim=1, dim_size=x.size(1)
        )  # For higher version of PyG.

        out = out_add + out_sub
        out = self.node_mlp(out)

        return out


class PM25_GNN(nn.Module):
    def __init__(
        self,
        hist_len,
        pred_len,
        in_dim,
        city_num,
        batch_size,
        device,
        edge_index,
        edge_attr,
        wind_mean,
        wind_std,
    ):
        super(PM25_GNN, self).__init__()

        self.device = device
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.city_num = city_num
        self.batch_size = batch_size

        self.in_dim = in_dim
        self.hid_dim = 64
        self.out_dim = 1
        self.gnn_out = 13

        self.fc_in = nn.Linear(self.in_dim, self.hid_dim)
        self.graph_gnn = GraphGNN(
            self.device, edge_index, edge_attr, self.in_dim, self.gnn_out, wind_mean, wind_std
        )
        self.gru_cell = GRUCell(self.in_dim + self.gnn_out, self.hid_dim)
        self.fc_out = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, pm25_hist, feature):
        pm25_pred = []
        h0 = torch.zeros(self.batch_size * self.city_num, self.hid_dim).to(self.device)
        hn = h0
        xn = pm25_hist[:, -1]
        for i in range(self.pred_len):
            x = torch.cat((xn, feature[:, self.hist_len + i]), dim=-1)

            xn_gnn = x
            xn_gnn = xn_gnn.contiguous()
            xn_gnn = self.graph_gnn(xn_gnn)
            x = torch.cat([xn_gnn, x], dim=-1)

            hn = self.gru_cell(x, hn)
            xn = hn.view(self.batch_size, self.city_num, self.hid_dim)
            xn = self.fc_out(xn)
            pm25_pred.append(xn)

        pm25_pred = torch.stack(pm25_pred, dim=1)

        return pm25_pred


# --- staging entry points ------------------------------------------------------


def build_pm25gnn():
    """Tiny PM2.5-GNN for tracing: small city_num/hist_len/pred_len/in_dim, and a
    small synthetic city graph (edge_index/edge_attr are normally precomputed from
    real station lat/lon in the KnowAir dataset -- here a tiny random 6-node ring
    graph stands in for that geography, since GraphGNN's actual architecture
    (edge MLP + scatter aggregation over whatever edge_index/edge_attr it is given)
    is unchanged by the specific graph)."""
    torch.manual_seed(0)
    city_num = 6
    # in_dim = feature_dim + pm25_dim per the real train.py construction call
    # (in_dim = train_data.feature.shape[-1] + train_data.pm25.shape[-1]): forward()
    # concatenates xn (pm25, 1 channel) with feature (feat_dim channels) into a single
    # in_dim-wide tensor before feeding GraphGNN, whose edge_mlp is sized off in_dim.
    feat_dim = 7
    in_dim = feat_dim + 1  # = 8; last 2 feature channels are wind speed/direction
    hist_len = 3
    pred_len = 2
    batch_size = 1
    device = "cpu"

    # Small ring graph over city_num cities: (src, dst) pairs both directions.
    src = list(range(city_num))
    dst = [(i + 1) % city_num for i in range(city_num)]
    edge_index = np.array([src + dst, dst + src], dtype=np.int64)
    n_edge = edge_index.shape[1]
    # edge_attr columns: [city_dist, city_direc]
    edge_attr = np.stack(
        [np.full(n_edge, 10.0, dtype=np.float32), np.linspace(0, 3.0, n_edge, dtype=np.float32)],
        axis=1,
    )
    wind_mean = np.array([2.0, 0.0], dtype=np.float32)
    wind_std = np.array([1.0, 1.0], dtype=np.float32)

    model = PM25_GNN(
        hist_len=hist_len,
        pred_len=pred_len,
        in_dim=in_dim,
        city_num=city_num,
        batch_size=batch_size,
        device=device,
        edge_index=edge_index,
        edge_attr=edge_attr,
        wind_mean=wind_mean,
        wind_std=wind_std,
    )
    model.eval()
    return model


def example_input_pm25gnn():
    # pm25_hist: [B, hist_len, city_num, 1] past PM2.5 concentrations.
    # feature: [B, hist_len+pred_len, city_num, feat_dim] meteorological features
    # (forward() indexes feature[:, hist_len + i] for each future step i, and
    # GraphGNN reads the last 2 channels of the per-step feature as wind speed/dir).
    # cat(xn[1 channel], feature[feat_dim channels]) = in_dim channels into GraphGNN.
    torch.manual_seed(0)
    batch_size, city_num, feat_dim = 1, 6, 7
    hist_len, pred_len = 3, 2
    pm25_hist = torch.rand(batch_size, hist_len, city_num, 1)
    feature = torch.rand(batch_size, hist_len + pred_len, city_num, feat_dim)
    return (pm25_hist, feature)


MENAGERIE_ENTRIES = [
    ("PM2.5-GNN", "build_pm25gnn", "example_input_pm25gnn", 2020, "vendored-pytorch"),
]
