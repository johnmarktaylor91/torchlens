# SOURCE: vendored from https://github.com/xincoder/GRIP @ master
# (model.py, layers/graph.py, layers/graph_conv_block.py, layers/graph_operation_layer.py,
#  layers/seq2seq.py)
# GRIP: Graph-based Interaction-aware Trajectory Prediction. IROS 2019.
"""GRIP: spatio-temporal graph convolution + per-class seq2seq GRU trajectory decoder.

Vendored verbatim from xincoder/GRIP `model.py` + `layers/*.py`. Architecture is
unmodified; only this header/build/example wrapper were added for menagerie staging.
One minimal fix: `Seq2Seq`/`EncoderRNN`/`DecoderRNN` accept an `isCuda` flag that,
when True, unconditionally calls `.cuda()` on freshly allocated tensors -- the build
below constructs with `isCuda=False` so the module runs on CPU, matching how the
original repo's own CPU users would configure it (the flag exists precisely to gate
this behavior).
"""

import numpy as np
import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# layers/graph.py
# ---------------------------------------------------------------------------
class Graph:
    """The Graph Representation
    How to use:
        1. graph = Graph(max_hop=1)
        2. A = graph.get_adjacency()
        3. A = code to modify A
        4. normalized_A = graph.normalize_adjacency(A)
    """

    def __init__(self, num_node=120, max_hop=1):
        self.max_hop = max_hop
        self.num_node = num_node

    def get_adjacency(self, A):
        # compute hop steps
        self.hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = np.stack(transfer_mat) > 0
        for d in range(self.max_hop, -1, -1):
            self.hop_dis[arrive_mat[d]] = d

        # compute adjacency
        valid_hop = range(0, self.max_hop + 1)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        return adjacency

    def normalize_adjacency(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)

        valid_hop = range(0, self.max_hop + 1)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = AD[self.hop_dis == hop]
        return A


# ---------------------------------------------------------------------------
# layers/graph_operation_layer.py
# ---------------------------------------------------------------------------
class ConvTemporalGraphical(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x, A):
        assert A.size(1) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()

        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,nkvw->nctw", (x, A))

        return x.contiguous(), A


# ---------------------------------------------------------------------------
# layers/graph_conv_block.py
# ---------------------------------------------------------------------------
class Graph_Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=False),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


# ---------------------------------------------------------------------------
# layers/seq2seq.py
# ---------------------------------------------------------------------------
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, isCuda=True):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.isCuda = isCuda
        self.lstm = nn.GRU(input_size, hidden_size * 30, num_layers, batch_first=True)

    def forward(self, input):
        output, hidden = self.lstm(input)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout=0.5, isCuda=True):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.isCuda = isCuda
        self.lstm = nn.GRU(hidden_size, output_size * 30, num_layers, batch_first=True)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(output_size * 30, output_size)
        self.tanh = nn.Tanh()

    def forward(self, encoded_input, hidden):
        decoded_output, hidden = self.lstm(encoded_input, hidden)
        decoded_output = self.dropout(decoded_output)
        decoded_output = self.linear(decoded_output)
        return decoded_output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, isCuda=True):
        super(Seq2Seq, self).__init__()
        self.isCuda = isCuda
        self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
        self.decoder = DecoderRNN(hidden_size, hidden_size, num_layers, dropout, isCuda)

    def forward(
        self, in_data, last_location, pred_length, teacher_forcing_ratio=0, teacher_location=None
    ):
        batch_size = in_data.shape[0]
        out_dim = self.decoder.output_size
        self.pred_length = pred_length

        outputs = torch.zeros(batch_size, self.pred_length, out_dim)
        if self.isCuda:
            outputs = outputs.cuda()

        encoded_output, hidden = self.encoder(in_data)
        decoder_input = last_location
        for t in range(self.pred_length):
            now_out, hidden = self.decoder(decoder_input, hidden)
            now_out += decoder_input
            outputs[:, t : t + 1] = now_out
            teacher_force = np.random.random() < teacher_forcing_ratio
            decoder_input = (
                teacher_location[:, t : t + 1]
                if (teacher_location is not None) and teacher_force
                else now_out
            )
        return outputs


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, in_channels, graph_args, edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = np.ones((graph_args["max_hop"] + 1, graph_args["num_node"], graph_args["num_node"]))

        # build networks
        spatial_kernel_size = np.shape(A)[0]
        temporal_kernel_size = 5
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # best
        self.st_gcn_networks = nn.ModuleList(
            (
                nn.BatchNorm2d(in_channels),
                Graph_Conv_Block(in_channels, 64, kernel_size, 1, residual=True, **kwargs),
                Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
                Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
            )
        )

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(np.shape(A))) for i in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.num_node = num_node = self.graph.num_node  # noqa: F841 (verbatim from upstream)
        self.out_dim_per_node = out_dim_per_node = 2  # (x, y) coordinate
        self.seq2seq_car = Seq2Seq(
            input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=False
        )
        self.seq2seq_human = Seq2Seq(
            input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=False
        )
        self.seq2seq_bike = Seq2Seq(
            input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=False
        )

    def reshape_for_lstm(self, feature):
        # prepare for skeleton prediction model
        """
        N: batch_size
        C: channel
        T: time_step
        V: nodes
        """
        N, C, T, V = feature.size()
        now_feat = feature.permute(0, 3, 2, 1).contiguous()  # to (N, V, T, C)
        now_feat = now_feat.view(N * V, T, C)
        return now_feat

    def reshape_from_lstm(self, predicted):
        # predicted (N*V, T, C)
        NV, T, C = predicted.size()
        now_feat = predicted.view(
            -1, self.num_node, T, self.out_dim_per_node
        )  # (N, T, V, C) -> (N, C, T, V) [(N, V, T, C)]
        now_feat = now_feat.permute(0, 3, 2, 1).contiguous()  # (N, C, T, V)
        return now_feat

    def forward(
        self, pra_x, pra_A, pra_pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=None
    ):
        x = pra_x

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            if type(gcn) is nn.BatchNorm2d:
                x = gcn(x)
            else:
                x, _ = gcn(x, pra_A + importance)

        # prepare for seq2seq lstm model
        graph_conv_feature = self.reshape_for_lstm(x)
        last_position = self.reshape_for_lstm(
            pra_x[:, :2]
        )  # (N, C, T, V)[:, :2] -> (N, T, V*2) [(N*V, T, C)]

        if pra_teacher_forcing_ratio > 0 and pra_teacher_location is not None:
            pra_teacher_location = self.reshape_for_lstm(pra_teacher_location)

        # now_predict.shape = (N, T, V*C)
        now_predict_car = self.seq2seq_car(
            in_data=graph_conv_feature,
            last_location=last_position[:, -1:, :],
            pred_length=pra_pred_length,
            teacher_forcing_ratio=pra_teacher_forcing_ratio,
            teacher_location=pra_teacher_location,
        )
        now_predict_car = self.reshape_from_lstm(now_predict_car)  # (N, C, T, V)

        now_predict_human = self.seq2seq_human(
            in_data=graph_conv_feature,
            last_location=last_position[:, -1:, :],
            pred_length=pra_pred_length,
            teacher_forcing_ratio=pra_teacher_forcing_ratio,
            teacher_location=pra_teacher_location,
        )
        now_predict_human = self.reshape_from_lstm(now_predict_human)  # (N, C, T, V)

        now_predict_bike = self.seq2seq_bike(
            in_data=graph_conv_feature,
            last_location=last_position[:, -1:, :],
            pred_length=pra_pred_length,
            teacher_forcing_ratio=pra_teacher_forcing_ratio,
            teacher_location=pra_teacher_location,
        )
        now_predict_bike = self.reshape_from_lstm(now_predict_bike)  # (N, C, T, V)

        now_predict = (now_predict_car + now_predict_human + now_predict_bike) / 3.0

        return now_predict


# ---------------------------------------------------------------------------
# Menagerie staging glue
# ---------------------------------------------------------------------------
_NUM_NODE = 12  # reduced from the paper's 120
_MAX_HOP = 2
_IN_CHANNELS = 4
_HIST_FRAMES = 6
_PRED_LENGTH = 6


def build_grip():
    graph_args = {"max_hop": _MAX_HOP, "num_node": _NUM_NODE}
    return Model(in_channels=_IN_CHANNELS, graph_args=graph_args, edge_importance_weighting=True)


def example_input_grip():
    batch = 1
    pra_x = torch.randn(batch, _IN_CHANNELS, _HIST_FRAMES, _NUM_NODE)

    graph = Graph(num_node=_NUM_NODE, max_hop=_MAX_HOP)
    rng = np.random.RandomState(0)
    raw_adj = (rng.rand(_NUM_NODE, _NUM_NODE) > 0.5).astype(np.float64)
    np.fill_diagonal(raw_adj, 1)
    adjacency = graph.get_adjacency(raw_adj)
    normalized_A = graph.normalize_adjacency(adjacency)  # [max_hop+1, V, V]
    pra_A = (
        torch.from_numpy(normalized_A).float().unsqueeze(0).repeat(batch, 1, 1, 1)
    )  # [N, max_hop+1, V, V]

    pra_pred_length = _PRED_LENGTH
    pra_teacher_forcing_ratio = 0
    pra_teacher_location = None

    return (pra_x, pra_A, pra_pred_length, pra_teacher_forcing_ratio, pra_teacher_location)


MENAGERIE_ENTRIES = [
    ("GRIP", "build_grip", "example_input_grip", 2019, "vendored-pytorch"),
]
