# SOURCE: vendored from Xiefeng69/EpiGNN @ 1c0bb8c2
# https://raw.githubusercontent.com/Xiefeng69/EpiGNN/main/src/models.py
# https://raw.githubusercontent.com/Xiefeng69/EpiGNN/main/src/layers.py
#
# Xie, Wang, Ma, Wang, Chen 2022 (ECML-PKDD) "EpiGNN: Exploring Spatial Transmission with
# Graph Neural Network for Regional Epidemic Forecasting". The classes below (EpiGNN,
# RegionAwareConv, ConvBranch, GraphLearner, GraphConvLayer) are copied verbatim from the
# real repo's src/models.py and src/layers.py -- only the ablation-module imports (unused
# by the plain forward pass) and the getLaplaceMat helper (copied inline from src/utils.py)
# were adjusted so this file is self-contained under base libs (torch, numpy). EpiGNN's
# constructor takes an `args` namespace and a `data` object exposing precomputed adjacency
# / degree tensors (data.adj, data.m, data.degree_adj) -- build_epignn() below assembles
# minimal stand-ins for both, matching the real train.py wiring (Region-Aware multi-scale
# conv backbone -> global/local risk encodings -> learned graph -> stacked GraphConvLayer
# GCN -> linear output head), at tiny size (m=6 regions, window=8).
"""EpiGNN: spatial transmission risk encoding + region-aware graph learner for epidemic forecasting."""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from src/utils.py (graph-Laplacian helper only) ---
def getLaplaceMat(batch_size, m, adj):
    i_mat = torch.eye(m).to(adj.device)
    i_mat = i_mat.unsqueeze(0)
    o_mat = torch.ones(m).to(adj.device)
    o_mat = o_mat.unsqueeze(0)
    i_mat = i_mat.expand(batch_size, m, m)
    o_mat = o_mat.expand(batch_size, m, m)
    adj = torch.where(adj > 0, o_mat, adj)
    d_mat = torch.sum(adj, dim=2)  # attention: dim=2
    d_mat = d_mat.unsqueeze(2)
    d_mat = d_mat + 1e-12
    d_mat = torch.pow(d_mat, -1)
    d_mat = d_mat.expand(d_mat.shape[0], d_mat.shape[1], d_mat.shape[1])
    d_mat = i_mat * d_mat
    laplace_mat = torch.bmm(d_mat, adj)
    return laplace_mat


# --- vendored from src/layers.py ---
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.act = nn.ELU()
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            stdv = 1.0 / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter("bias", None)

    def forward(self, feature, adj):
        support = torch.matmul(feature, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            return self.act(output + self.bias)
        else:
            return self.act(output)


class GraphLearner(nn.Module):
    def __init__(self, hidden_dim, tanhalpha=1):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tanhalpha

    def forward(self, embedding):
        # embedding [batchsize, hidden_dim]
        nodevec1 = self.linear1(embedding)
        nodevec2 = self.linear2(embedding)
        nodevec1 = self.alpha * nodevec1
        nodevec2 = self.alpha * nodevec2
        nodevec1 = torch.tanh(nodevec1)
        nodevec2 = torch.tanh(nodevec2)

        adj = torch.bmm(nodevec1, nodevec2.permute(0, 2, 1)) - torch.bmm(
            nodevec2, nodevec1.permute(0, 2, 1)
        )
        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))
        return adj


class ConvBranch(nn.Module):
    def __init__(
        self, m, in_channels, out_channels, kernel_size, dilation_factor, hidP=1, isPool=True
    ):
        super().__init__()
        self.m = m
        self.isPool = isPool
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(kernel_size, 1), dilation=(dilation_factor, 1)
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        if self.isPool:
            self.pooling = nn.AdaptiveMaxPool2d((hidP, m))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.isPool:
            x = self.pooling(x)
        x = x.view(batch_size, -1, self.m)
        return x


class RegionAwareConv(nn.Module):
    def __init__(self, P, m, k, hidP, dilation_factor=2):
        super(RegionAwareConv, self).__init__()
        self.P = P
        self.m = m
        self.k = k
        self.hidP = hidP
        self.conv_l1 = ConvBranch(
            m=self.m,
            in_channels=1,
            out_channels=self.k,
            kernel_size=3,
            dilation_factor=1,
            hidP=self.hidP,
        )
        self.conv_l2 = ConvBranch(
            m=self.m,
            in_channels=1,
            out_channels=self.k,
            kernel_size=5,
            dilation_factor=1,
            hidP=self.hidP,
        )
        self.conv_p1 = ConvBranch(
            m=self.m,
            in_channels=1,
            out_channels=self.k,
            kernel_size=3,
            dilation_factor=dilation_factor,
            hidP=self.hidP,
        )
        self.conv_p2 = ConvBranch(
            m=self.m,
            in_channels=1,
            out_channels=self.k,
            kernel_size=5,
            dilation_factor=dilation_factor,
            hidP=self.hidP,
        )
        self.conv_g = ConvBranch(
            m=self.m,
            in_channels=1,
            out_channels=self.k,
            kernel_size=self.P,
            dilation_factor=1,
            hidP=None,
            isPool=False,
        )
        self.activate = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, 1, self.P, self.m)
        # local pattern
        x_l1 = self.conv_l1(x)
        x_l2 = self.conv_l2(x)
        x_local = torch.cat([x_l1, x_l2], dim=1)
        # periodic pattern
        x_p1 = self.conv_p1(x)
        x_p2 = self.conv_p2(x)
        x_period = torch.cat([x_p1, x_p2], dim=1)
        # global
        x_global = self.conv_g(x)
        # concat and activate
        x = torch.cat([x_local, x_period, x_global], dim=1).permute(0, 2, 1)
        x = self.activate(x)
        return x


# --- vendored from src/models.py ---
class EpiGNN(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        # arguments setting
        self.adj = data.adj
        self.m = data.m
        self.w = args.window
        self.n_layer = args.n_layer
        self.droprate = args.dropout
        self.hidR = args.hidR
        self.hidA = args.hidA
        self.hidP = args.hidP
        self.k = args.k
        self.s = args.s
        self.n = args.n
        self.res = args.res
        self.hw = args.hw
        self.dropout = nn.Dropout(self.droprate)

        if self.hw > 0:
            self.highway = nn.Linear(self.hw, 1)

        if args.extra:
            self.extra = True
            self.external = data.external
        else:
            self.extra = False

        # Feature embedding
        self.hidR = self.k * 4 * self.hidP + self.k
        self.backbone = RegionAwareConv(P=self.w, m=self.m, k=self.k, hidP=self.hidP)

        # global
        self.WQ = nn.Linear(self.hidR, self.hidA)
        self.WK = nn.Linear(self.hidR, self.hidA)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.t_enc = nn.Linear(1, self.hidR)

        # local
        self.degree = data.degree_adj
        self.s_enc = nn.Linear(1, self.hidR)

        # external resources
        self.external_parameter = nn.Parameter(
            torch.FloatTensor(self.m, self.m), requires_grad=True
        )

        # Graph Generator and GCN
        self.d_gate = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)
        self.graphGen = GraphLearner(self.hidR)
        self.GNNBlocks = nn.ModuleList(
            [GraphConvLayer(in_features=self.hidR, out_features=self.hidR) for i in range(self.n)]
        )

        # prediction
        if self.res == 0:
            self.output = nn.Linear(self.hidR * 2, 1)
        else:
            self.output = nn.Linear(self.hidR * (self.n + 1), 1)

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)  # best
            else:
                stdv = 1.0 / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, x, index, isEval=False):
        batch_size = x.shape[0]  # batchsize, w, m

        # step 1: Use multi-scale convolution to extract feature embedding (SEFNet => RAConv).
        temp_emb = self.backbone(x)

        # step 2: generate global transmission risk encoding.
        query = self.WQ(temp_emb)  # batch, N, hidden
        query = self.dropout(query)
        key = self.WK(temp_emb)
        key = self.dropout(key)
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = torch.sum(attn, dim=-1)
        attn = attn.unsqueeze(2)
        t_enc = self.t_enc(attn)
        t_enc = self.dropout(t_enc)

        # step 3: generate local transmission risk encoding.
        d = self.degree.unsqueeze(1)
        s_enc = self.s_enc(d)
        s_enc = self.dropout(s_enc)

        # Three embedding fusion.
        feat_emb = temp_emb + t_enc + s_enc

        # step 4: Region-Aware Graph Learner
        if self.extra:
            extra_adj_list = []
            zeros_mt = torch.zeros((self.m, self.m)).to(self.adj.device)
            for i in range(batch_size):
                offset = 20
                if i - offset >= 0:
                    idx = i - offset  # noqa: F841 (kept verbatim from real repo; unused branch when extra=False)
                    extra_adj_list.append(self.external[index[i], :, :].unsqueeze(0))
                else:
                    extra_adj_list.append(zeros_mt.unsqueeze(0))
            extra_info = torch.concat(extra_adj_list, dim=0)
            external_info = torch.mul(self.external_parameter, extra_info)
            external_info = F.relu(external_info)

        # apply Graph Learner to generate a graph
        d_mat = torch.mm(d, d.permute(1, 0))
        d_mat = torch.mul(self.d_gate, d_mat)
        d_mat = torch.sigmoid(d_mat)
        spatial_adj = torch.mul(d_mat, self.adj)
        adj = self.graphGen(temp_emb)

        if self.extra:
            adj = adj + spatial_adj + external_info
        else:
            adj = adj + spatial_adj

        # get laplace adjacent matrix
        laplace_adj = getLaplaceMat(batch_size, self.m, adj)

        # Graph Convolution Network
        node_state = feat_emb
        node_state_list = []
        for layer in self.GNNBlocks:
            node_state = layer(node_state, laplace_adj)
            node_state = self.dropout(node_state)
            node_state_list.append(node_state)

        # Final prediction
        node_state = torch.cat([node_state, feat_emb], dim=-1)
        res = self.output(node_state).squeeze(2)
        # highway means autoregressive model
        if self.hw > 0:
            z = x[:, -self.hw :, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z

        if isEval:
            imd = (adj, attn)
        else:
            imd = None

        return res, imd


def build_epignn():
    torch.manual_seed(0)
    m = 6  # number of regions
    window = 12  # must exceed largest dilated-conv receptive field: (5-1)*2+1 = 9
    args = SimpleNamespace(
        window=window,
        n_layer=1,
        dropout=0.2,
        hidR=32,
        hidA=16,
        hidP=1,
        k=8,
        s=1,
        n=2,
        res=0,
        hw=0,
        extra=False,
    )
    data = SimpleNamespace(
        adj=torch.rand(m, m),
        m=m,
        degree_adj=torch.rand(m),
    )
    return EpiGNN(args, data)


def example_input_epignn():
    torch.manual_seed(0)
    batch_size = 2
    window = 12
    m = 6
    x = torch.randn(batch_size, window, m)
    index = torch.arange(batch_size)
    return (x, index)


MENAGERIE_ENTRIES = [
    ("EpiGNN", "build_epignn", "example_input_epignn", 2022, "vendored"),
]
