# SOURCE: vendored from hazdzz/STGCN @ main
# https://raw.githubusercontent.com/hazdzz/STGCN/main/model/layers.py
# https://raw.githubusercontent.com/hazdzz/STGCN/main/model/models.py
#
# STGCN (IJCAI 2018): Yu, Yin, Zhu, "Spatio-Temporal Graph Convolutional
# Networks: A Deep Learning Framework for Traffic Forecasting". The official
# VeritasYin/STGCN_IJCAI-18 repo is TensorFlow; hazdzz/STGCN is the
# well-established pure-PyTorch reimplementation (independently used/cited
# widely) and is what the queue row's "PyTorch hazdzz/stgcn" note points at.
#
# Vendored verbatim (only trivial edit: `from model import layers` ->
# module-relative reference, since we vendor both files into one module).
# Architecture is untouched.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

MENAGERIE_ZOO = "vendored-pytorch"


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat(
                [x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)],
                dim=1,
            )
        else:
            x = x

        return x


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        enable_padding=False,
        dilation=1,
        groups=1,
        bias=True,
    ):
        if enable_padding == True:
            self.__padding = (kernel_size - 1) * dilation
        else:
            self.__padding = 0
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]

        return result


class CausalConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        enable_padding=False,
        dilation=1,
        groups=1,
        bias=True,
    ):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [
                int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))
            ]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)

        return result


class TemporalConvLayer(nn.Module):
    # Temporal Convolution Layer (GLU)
    #
    #        |--------------------------------| * residual connection *
    #        |                                |
    #        |    |--->--- casualconv2d ----- + -------|
    # -------|----|                                   (x) ------>
    #             |--->--- casualconv2d --- sigmoid ---|

    # param x: tensor, [bs, c_in, ts, n_vertex]

    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        if act_func == "glu" or act_func == "gtu":
            self.causal_conv = CausalConv2d(
                in_channels=c_in,
                out_channels=2 * c_out,
                kernel_size=(Kt, 1),
                enable_padding=False,
                dilation=1,
            )
        else:
            self.causal_conv = CausalConv2d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=(Kt, 1),
                enable_padding=False,
                dilation=1,
            )
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.act_func = act_func

    def forward(self, x):
        x_in = self.align(x)[:, :, self.Kt - 1 :, :]
        x_causal_conv = self.causal_conv(x)

        if self.act_func == "glu" or self.act_func == "gtu":
            x_p = x_causal_conv[:, : self.c_out, :, :]
            x_q = x_causal_conv[:, -self.c_out :, :, :]

            if self.act_func == "glu":
                x = torch.mul((x_p + x_in), torch.sigmoid(x_q))
            else:
                x = torch.mul(torch.tanh(x_p + x_in), torch.sigmoid(x_q))

        elif self.act_func == "relu":
            x = self.relu(x_causal_conv + x_in)

        elif self.act_func == "silu":
            x = self.silu(x_causal_conv + x_in)

        else:
            raise NotImplementedError(
                f"ERROR: The activation function {self.act_func} is not implemented."
            )

        return x


class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso, bias):
        super(ChebGraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))

        if self.Ks - 1 < 0:
            raise ValueError(
                f"ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.Ks}."
            )
        elif self.Ks - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum("hi,btij->bthj", self.gso, x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x
            x_1 = torch.einsum("hi,btij->bthj", self.gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(
                    torch.einsum("hi,btij->bthj", 2 * self.gso, x_list[k - 1]) - x_list[k - 2]
                )

        x = torch.stack(x_list, dim=2)

        cheb_graph_conv = torch.einsum("btkhi,kij->bthj", x, self.weight)

        if self.bias is not None:
            cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)
        else:
            cheb_graph_conv = cheb_graph_conv

        return cheb_graph_conv


class GraphConv(nn.Module):
    def __init__(self, c_in, c_out, gso, bias):
        super(GraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))

        first_mul = torch.einsum("hi,btij->bthj", self.gso, x)
        second_mul = torch.einsum("bthi,ij->bthj", first_mul, self.weight)

        if self.bias is not None:
            graph_conv = torch.add(second_mul, self.bias)
        else:
            graph_conv = second_mul

        return graph_conv


class GraphConvLayer(nn.Module):
    def __init__(self, graph_conv_type, c_in, c_out, Ks, gso, bias):
        super(GraphConvLayer, self).__init__()
        self.graph_conv_type = graph_conv_type
        self.c_in = c_in
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        self.Ks = Ks
        self.gso = gso
        if self.graph_conv_type == "cheb_graph_conv":
            self.cheb_graph_conv = ChebGraphConv(c_out, c_out, Ks, gso, bias)
        elif self.graph_conv_type == "graph_conv":
            self.graph_conv = GraphConv(c_out, c_out, gso, bias)

    def forward(self, x):
        x_gc_in = self.align(x)
        if self.graph_conv_type == "cheb_graph_conv":
            x_gc = self.cheb_graph_conv(x_gc_in)
        elif self.graph_conv_type == "graph_conv":
            x_gc = self.graph_conv(x_gc_in)
        x_gc = x_gc.permute(0, 3, 1, 2)
        x_gc_out = torch.add(x_gc, x_gc_in)

        return x_gc_out


class STConvBlock(nn.Module):
    # STConv Block contains 'TGTND' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv or GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    def __init__(
        self,
        Kt,
        Ks,
        n_vertex,
        last_block_channel,
        channels,
        act_func,
        graph_conv_type,
        gso,
        bias,
        droprate,
    ):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.graph_conv(x)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)

        return x


class OutputBlock(nn.Module):
    # Output block contains 'TNFF' structure
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(
        self, Ko, last_block_channel, channels, end_channel, n_vertex, act_func, bias, droprate
    ):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Ko, last_block_channel, channels[0], n_vertex, act_func)
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1], bias=bias)
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel, bias=bias)
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x).permute(0, 3, 1, 2)

        return x


class STGCNChebGraphConv(nn.Module):
    # STGCNChebGraphConv contains 'TGTND TGTND TNFF' structure
    # ChebGraphConv is the graph convolution from ChebyNet.

    def __init__(self, args, blocks, n_vertex):
        super(STGCNChebGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(
                STConvBlock(
                    args.Kt,
                    args.Ks,
                    n_vertex,
                    blocks[l][-1],
                    blocks[l + 1],
                    args.act_func,
                    args.graph_conv_type,
                    args.gso,
                    args.enable_bias,
                    args.droprate,
                )
            )
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = OutputBlock(
                Ko,
                blocks[-3][-1],
                blocks[-2],
                blocks[-1][0],
                n_vertex,
                args.act_func,
                args.enable_bias,
                args.droprate,
            )
        elif self.Ko == 0:
            self.fc1 = nn.Linear(
                in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias
            )
            self.fc2 = nn.Linear(
                in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias
            )
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        return x


# --- Menagerie staging wrapper --------------------------------------------------
#
# The real model is constructed from an argparse.Namespace `args` (see the real
# main.py CLI) + a `blocks` channel schedule + `n_vertex` graph size. We supply
# a tiny stand-in Namespace and a small random symmetric graph-shift-operator
# (gso) instead of building it via scipy from a real traffic-sensor adjacency
# CSV, matching the real code path `args.gso = torch.from_numpy(gso)...` in
# `data_preparate()`. Kt/Ks/blocks follow the real main.py defaults, shrunk
# just enough (n_his, stblock_num, n_vertex) for a fast trace.


class _Args:
    def __init__(self, n_his, Kt, Ks, gso, n_vertex):
        self.n_his = n_his
        self.Kt = Kt
        self.Ks = Ks
        self.act_func = "glu"
        self.graph_conv_type = "cheb_graph_conv"
        self.gso = gso
        self.enable_bias = True
        self.droprate = 0.0


_N_VERTEX = 6
_N_HIS = 9
_KT = 3
_KS = 3
_STBLOCK_NUM = 1


def _make_blocks(stblock_num, Ko):
    blocks = [[1]]
    for _ in range(stblock_num):
        blocks.append([16, 8, 16])
    if Ko == 0:
        blocks.append([32])
    elif Ko > 0:
        blocks.append([32, 32])
    blocks.append([1])
    return blocks


def build_stgcn():
    torch.manual_seed(0)
    # Symmetric random graph-shift-operator standing in for a real
    # scipy-computed sym_renorm_lap Chebyshev GSO (real code path in
    # utility.calc_gso / calc_chebynet_gso); architecture consumes it as a
    # plain float32 [n_vertex, n_vertex] tensor either way.
    raw = torch.randn(_N_VERTEX, _N_VERTEX)
    gso = (raw + raw.t()) / 2
    Ko = _N_HIS - (_STBLOCK_NUM) * 2 * (_KT - 1)
    args = _Args(n_his=_N_HIS, Kt=_KT, Ks=_KS, gso=gso, n_vertex=_N_VERTEX)
    blocks = _make_blocks(_STBLOCK_NUM, Ko)
    return STGCNChebGraphConv(args, blocks, _N_VERTEX)


def example_input_stgcn():
    return torch.randn(1, 1, _N_HIS, _N_VERTEX)


MENAGERIE_ENTRIES = [
    ("STGCN (ChebGraphConv)", build_stgcn, example_input_stgcn, 2018, "CODE"),
]
