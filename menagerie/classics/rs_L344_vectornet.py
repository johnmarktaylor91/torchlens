# SOURCE: vendored from Liang-ZX/VectorNet @ master
# https://raw.githubusercontent.com/Liang-ZX/VectorNet/master/vectornet.py
# https://raw.githubusercontent.com/Liang-ZX/VectorNet/master/subgraph_net.py
# https://raw.githubusercontent.com/Liang-ZX/VectorNet/master/gnn.py
#
# Gao, Sun, Shen, Pang, Yang, Schmid, Ross, Casas 2020 (CVPR) "VectorNet: Encoding
# HD Maps and Agent Dynamics from Vectorized Representation" -- encodes both the
# target agent's trajectory and the surrounding HD-map polylines as sets of
# "vectors" (consecutive-point segments), runs each polyline through a shared
# hierarchical `SubgraphNet` (3 stacked fully-connected + maxpool "subgraph"
# layers that fuse per-node and global polyline features), L2-normalizes the
# resulting per-polyline embeddings, and feeds the whole polyline set (one node
# per polyline: the trajectory polyline + every map polyline) through a single
# global self-attention layer (`GraphAttentionNet`, a fully-connected-graph
# single-head QKV attention block) before decoding the target agent's future
# trajectory offsets via a 2-layer MLP head. This is the official community
# PyTorch re-implementation referenced by the queue (no official Waymo/Argoverse
# open-source release of VectorNet itself).
#
# `SubgraphNet_Layer`, `SubgraphNet` (subgraph_net.py), `GraphAttentionNet`
# (gnn.py), and `VectorNet` (vectornet.py) are copied verbatim from the real
# source files (only the `from subgraph_net import ...` / `from gnn import ...`
# local-package imports were inlined into one file, and unused
# matplotlib/debug imports were dropped -- no architectural change). We exercise
# `VectorNet._forward_train`, the code path that runs BOTH sub-networks (the
# trajectory subgraph net AND the map subgraph net over multiple polylines) plus
# the graph attention net and the offset-decoding head -- i.e. the full
# architecture as actually used during training, rather than the
# `_forward_test` branch which only runs the trajectory subgraph net.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubgraphNet_Layer(nn.Module):
    def __init__(self, input_channels=128, hidden_channels=64):
        super().__init__()
        self.fc = nn.Linear(input_channels, hidden_channels)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, input):
        hidden = self.fc(input).unsqueeze(0)
        encode_data = F.relu(F.layer_norm(hidden, hidden.size()[1:]))
        kernel_size = encode_data.size()[1]
        maxpool = nn.MaxPool1d(kernel_size)
        polyline_feature = maxpool(encode_data.transpose(1, 2)).squeeze()
        polyline_feature = polyline_feature.repeat(kernel_size, 1)
        output = torch.cat([encode_data.squeeze(), polyline_feature], 1)
        return output


class SubgraphNet(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.sublayer1 = SubgraphNet_Layer(input_channels)
        self.sublayer2 = SubgraphNet_Layer()
        self.sublayer3 = SubgraphNet_Layer()

    def forward(self, input):
        out1 = self.sublayer1(input)
        out2 = self.sublayer2(out1)
        out3 = self.sublayer3(out2)
        kernel_size = out3.size()[0]
        maxpool = nn.MaxPool1d(kernel_size)
        polyline_feature = maxpool(out3.unsqueeze(1).transpose(0, 2)).squeeze()
        return polyline_feature


class GraphAttentionNet(nn.Module):
    def __init__(self, in_dim=128, key_dim=64, value_dim=64):
        super().__init__()
        self.queryFC = nn.Linear(in_dim, key_dim)
        nn.init.kaiming_normal_(self.queryFC.weight)
        self.keyFC = nn.Linear(in_dim, key_dim)
        nn.init.kaiming_normal_(self.keyFC.weight)
        self.valueFC = nn.Linear(in_dim, value_dim)
        nn.init.kaiming_normal_(self.valueFC.weight)

    def forward(self, polyline_feature):
        p_query = F.relu(self.queryFC(polyline_feature))
        p_key = F.relu(self.keyFC(polyline_feature))
        p_value = F.relu(self.valueFC(polyline_feature))
        query_result = p_query.mm(p_key.t())
        query_result = query_result / (p_key.shape[1] ** 0.5)
        attention = F.softmax(query_result, dim=1)
        output = attention.mm(p_value)
        return output + p_query


class VectorNet(nn.Module):
    def __init__(self, traj_features=4, map_features=8, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = dict(device=torch.device("cpu"))
        self.cfg = cfg
        self.traj_subgraphnet = SubgraphNet(traj_features)
        self.map_subgraphnet = SubgraphNet(map_features)
        self.graphnet = GraphAttentionNet()
        prediction_step = 2 * (49 - self.cfg["last_observe"])
        self.fc = nn.Linear(64, 64)
        nn.init.kaiming_normal_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, prediction_step)
        nn.init.kaiming_normal_(self.fc2.weight)

        self.loss_fn = nn.MSELoss(size_average=False, reduce=True)

    def _forward_train(self, trajectory_batch, vectormap_batch):
        batch_size = trajectory_batch.size()[0]

        label = trajectory_batch[:, self.cfg["last_observe"] :, 2:4]

        predict_list = []
        for i in range(batch_size):
            polyline_list = []
            polyline_list.append(
                self.traj_subgraphnet(trajectory_batch[i, : self.cfg["last_observe"]]).unsqueeze(0)
            )

            for vec_map in vectormap_batch:
                vec_map = vec_map.to(device=self.cfg["device"], dtype=torch.float)
                map_feature = self.map_subgraphnet(vec_map.squeeze())
                polyline_list.append(map_feature.unsqueeze(0))

            polyline_feature = F.normalize(torch.cat(polyline_list, dim=0), p=2, dim=1)
            out = self.graphnet(polyline_feature)
            decoded_data_perstep = self.fc2(
                F.relu(self.layer_norm(self.fc(out[0].unsqueeze(0))))
            ).view(1, -1, 2)
            decoded_data = torch.cumsum(decoded_data_perstep, dim=0)
            predict_list.append(decoded_data)
        predict_batch = torch.cat(predict_list, dim=0)
        loss = self.loss_fn(predict_batch, label)
        return loss

    def _forward_test(self, trajectory_batch):
        batch_size = trajectory_batch.size()[0]

        traj_label = trajectory_batch[:, self.cfg["last_observe"] :, 2:4]
        result, label = dict(), dict()
        for i in range(batch_size):
            polyline_feature = self.traj_subgraphnet(
                trajectory_batch[i, : self.cfg["last_observe"]]
            ).unsqueeze(0)
            polyline_feature = F.normalize(polyline_feature, p=2, dim=1)
            out = self.graphnet(polyline_feature)
            decoded_data_perstep = self.fc2(F.relu(self.layer_norm(self.fc(out)))).view(-1, 2)
            decoded_data = torch.cumsum(decoded_data_perstep, dim=0)
            key = str(trajectory_batch[i, 0, -1].int().item())
            predict_step = self.cfg["predict_step"]
            result.update({key: decoded_data[:predict_step]})
            label.update({key: traj_label[i, :predict_step]})
        return result, label

    def forward(self, trajectory, vectormap):
        if self.training:
            return self._forward_train(trajectory, vectormap)
        else:
            return self._forward_test(trajectory)


def build_vectornet():
    cfg = dict(device=torch.device("cpu"), last_observe=20, predict_step=10)
    model = VectorNet(traj_features=6, map_features=8, cfg=cfg)
    model.train()
    return model


def example_input_vectornet():
    # trajectory_batch: [batch_size, 49, 6] (traj_features=6, matches source comment)
    trajectory_batch = torch.randn(2, 49, 6)
    # vectormap_batch: list of per-polyline map-vector tensors, each [n_vectors, 8]
    vectormap_batch = [torch.randn(1, 18, 8) for _ in range(3)]
    return (trajectory_batch, vectormap_batch)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("VectorNet", "build_vectornet", "example_input_vectornet", 2020, "vendored"),
]
