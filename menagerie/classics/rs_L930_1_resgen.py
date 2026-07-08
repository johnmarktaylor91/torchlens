# SOURCE: vendored from https://github.com/OdinZhang/ResGen @ main
# Files vendored (near-verbatim, only import paths + package prefix changed to be
# self-contained in this staging module): models/embedding.py (GVP), models/utils.py
# (GVLinear/GVPerceptronVN/VNLinear/VNLeakyReLU/MessageModule), models/common.py
# (GaussianSmearing/EdgeExpansion/SmoothCrossEntropyLoss), models/interaction/cftfm.py
# (CFTransformerEncoderVN), models/fields/classifier.py (SpatialClassifierVN), models/
# frontier.py (FrontierLayerVN), models/position.py (PositionPredictor), models/ResGen.py
# (ResGen, embed_compose_GVP).
#
# ResGen (Zhang & Liu, Chem. Sci. 2023) is a 3D structure-based ligand-generation model:
# a geometric-vector-perceptron (GVP) protein/ligand atom encoder feeds a vector-neuron
# (VN) transformer message-passing encoder (CFTransformerEncoderVN) over a protein-ligand
# "compose" k-NN graph, followed by frontier/position/element/bond prediction heads used
# autoregressively at generation time.
#
# TorchLens integration: the real model has no plain `forward(x)` -- it is invoked via
# `ResGen.get_loss(...)` (training, needs full contrastive-sampling supervision) or
# `ResGen.sample_focal(...)` (inference: frontier-atom classification over a protein-only
# "compose" graph, the entry point `sample_init` starts from). `sample_focal` is REAL
# unmodified inference code from ResGen.py (only the `unique()` import dependency was
# dropped since sample_focal never calls it), so it is used as-is as the traced entry point.
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import Module, Linear
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter_sum, scatter_softmax
from math import pi as PI

MENAGERIE_ZOO = "vendored-pytorch"

EPS = 1e-6


# ---------------------------------------------------------------------------
# models/embedding.py :: GVP  (Geometric Vector Perceptron)
# ---------------------------------------------------------------------------
def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


class GVP(nn.Module):
    """Geometric Vector Perceptron. See ResGen models/embedding.py."""

    def __init__(
        self, in_dims, out_dims, h_dim=None, activations=(F.relu, torch.sigmoid), vector_gate=False
    ):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate:
                    self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(_norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3, device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)

        return (s, v) if self.vo else s


# ---------------------------------------------------------------------------
# models/utils.py :: VN-GVP primitives
# ---------------------------------------------------------------------------
class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, *args, **kwargs)

    def forward(self, x):
        return self.map_to_feat(x.transpose(-2, -1)).transpose(-2, -1)


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.01):
        super().__init__()
        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        d = self.map_to_dir(x.transpose(-2, -1)).transpose(-2, -1)
        dotprod = (x * d).sum(-1, keepdim=True)
        mask = (dotprod >= 0).to(x.dtype)
        d_norm_sq = (d * d).sum(-1, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class GVLinear(Module):
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector):
        super().__init__()
        dim_hid = max(in_vector, out_vector)
        self.lin_vector = VNLinear(in_vector, dim_hid, bias=False)
        self.lin_vector2 = VNLinear(dim_hid, out_vector, bias=False)
        self.scalar_to_vector_gates = Linear(out_scalar, out_vector)
        self.lin_scalar = Linear(in_scalar + dim_hid, out_scalar, bias=False)

    def forward(self, features):
        feat_scalar, feat_vector = features
        feat_vector_inter = self.lin_vector(feat_vector)
        feat_vector_norm = torch.norm(feat_vector_inter, p=2, dim=-1)
        feat_scalar_cat = torch.cat([feat_vector_norm, feat_scalar], dim=-1)

        out_scalar = self.lin_scalar(feat_scalar_cat)
        out_vector = self.lin_vector2(feat_vector_inter)

        gating = torch.sigmoid(self.scalar_to_vector_gates(out_scalar)).unsqueeze(dim=-1)
        out_vector = gating * out_vector
        return out_scalar, out_vector


class GVPerceptronVN(Module):
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector):
        super().__init__()
        self.gv_linear = GVLinear(in_scalar, in_vector, out_scalar, out_vector)
        self.act_sca = nn.LeakyReLU()
        self.act_vec = VNLeakyReLU(out_vector)

    def forward(self, x):
        sca, vec = self.gv_linear(x)
        vec = self.act_vec(vec)
        sca = self.act_sca(sca)
        return sca, vec


class MessageModule(Module):
    def __init__(self, node_sca, node_vec, edge_sca, edge_vec, out_sca, out_vec, cutoff=10.0):
        super().__init__()
        hid_sca, hid_vec = edge_sca, edge_vec
        self.cutoff = cutoff
        self.node_gvlinear = GVLinear(node_sca, node_vec, out_sca, out_vec)
        self.edge_gvp = GVPerceptronVN(edge_sca, edge_vec, hid_sca, hid_vec)

        self.sca_linear = Linear(hid_sca, out_sca)
        self.e2n_linear = Linear(hid_sca, out_vec)
        self.n2e_linear = Linear(out_sca, out_vec)
        self.edge_vnlinear = VNLinear(hid_vec, out_vec)

        self.out_gvlienar = GVLinear(out_sca, out_vec, out_sca, out_vec)

    def forward(self, node_features, edge_features, edge_index_node, dist_ij=None, annealing=False):
        node_scalar, node_vector = self.node_gvlinear(node_features)
        node_scalar, node_vector = node_scalar[edge_index_node], node_vector[edge_index_node]
        edge_scalar, edge_vector = self.edge_gvp(edge_features)

        y_scalar = node_scalar * self.sca_linear(edge_scalar)
        y_node_vector = self.e2n_linear(edge_scalar).unsqueeze(-1) * node_vector
        y_edge_vector = self.n2e_linear(node_scalar).unsqueeze(-1) * self.edge_vnlinear(edge_vector)
        y_vector = y_node_vector + y_edge_vector

        output = self.out_gvlienar((y_scalar, y_vector))

        if annealing:
            C = 0.5 * (torch.cos(dist_ij * PI / self.cutoff) + 1.0)
            C = C * (dist_ij <= self.cutoff) * (dist_ij >= 0.0)
            output = [output[0] * C.view(-1, 1), output[1] * C.view(-1, 1, 1)]
        return output


# ---------------------------------------------------------------------------
# models/common.py :: geometry featurization + losses
# ---------------------------------------------------------------------------
class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=50):
        super().__init__()
        self.stop = stop
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.clamp_max(self.stop)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class EdgeExpansion(nn.Module):
    def __init__(self, edge_channels):
        super().__init__()
        self.nn = nn.Linear(in_features=1, out_features=edge_channels, bias=False)

    def forward(self, edge_vector):
        edge_vector = edge_vector / (torch.norm(edge_vector, p=2, dim=1, keepdim=True) + 1e-7)
        expansion = self.nn(edge_vector.unsqueeze(-1)).transpose(1, -1)
        return expansion


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets, n_classes, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1), self.smoothing)
        lsm = F.log_softmax(inputs, -1)
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)
        loss = -(targets * lsm).sum(-1)
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss


# ---------------------------------------------------------------------------
# models/interaction/cftfm.py :: CFTransformerEncoderVN
# ---------------------------------------------------------------------------
class AttentionInteractionBlockVN(Module):
    def __init__(
        self, hidden_channels, edge_channels, num_edge_types, key_channels, num_heads=1, cutoff=10.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.distance_expansion = GaussianSmearing(
            stop=cutoff, num_gaussians=edge_channels - num_edge_types
        )
        self.vector_expansion = EdgeExpansion(edge_channels)

        self.message_module = MessageModule(
            hidden_channels[0],
            hidden_channels[1],
            edge_channels,
            edge_channels,
            hidden_channels[0],
            hidden_channels[1],
            cutoff,
        )

        self.centroid_lin = GVLinear(
            hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1]
        )
        self.act_sca = nn.LeakyReLU()
        self.act_vec = VNLeakyReLU(hidden_channels[1])
        self.out_transform = GVLinear(
            hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1]
        )

        self.layernorm_sca = nn.LayerNorm([hidden_channels[0]])
        self.layernorm_vec = nn.LayerNorm([hidden_channels[1], 3])

    def forward(self, x, edge_index, edge_feature, edge_vector):
        scalar, vector = x
        N = scalar.size(0)
        row, col = edge_index

        edge_dist = torch.norm(edge_vector, dim=-1, p=2)
        edge_sca_feat = torch.cat([self.distance_expansion(edge_dist), edge_feature], dim=-1)
        edge_vec_feat = self.vector_expansion(edge_vector)

        msg_j_sca, msg_j_vec = self.message_module(
            x, (edge_sca_feat, edge_vec_feat), col, edge_dist, annealing=True
        )

        aggr_msg_sca = scatter_sum(msg_j_sca, row, dim=0, dim_size=N)
        aggr_msg_vec = scatter_sum(msg_j_vec, row, dim=0, dim_size=N)
        x_out_sca, x_out_vec = self.centroid_lin(x)
        out_sca = x_out_sca + aggr_msg_sca
        out_vec = x_out_vec + aggr_msg_vec

        out_sca = self.layernorm_sca(out_sca)
        out_vec = self.layernorm_vec(out_vec)
        out = self.out_transform((self.act_sca(out_sca), self.act_vec(out_vec)))
        return out


class CFTransformerEncoderVN(Module):
    def __init__(
        self,
        hidden_channels=(256, 64),
        edge_channels=64,
        num_edge_types=4,
        key_channels=128,
        num_heads=4,
        num_interactions=6,
        k=32,
        cutoff=10.0,
    ):
        super().__init__()
        self.hidden_channels = list(hidden_channels)
        self.edge_channels = edge_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff

        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlockVN(
                hidden_channels=self.hidden_channels,
                edge_channels=edge_channels,
                num_edge_types=num_edge_types,
                key_channels=key_channels,
                num_heads=num_heads,
                cutoff=cutoff,
            )
            self.interactions.append(block)

    @property
    def out_sca(self):
        return self.hidden_channels[0]

    @property
    def out_vec(self):
        return self.hidden_channels[1]

    def forward(self, node_attr, pos, edge_index, edge_feature):
        edge_vector = pos[edge_index[0]] - pos[edge_index[1]]
        h = list(node_attr)
        for interaction in self.interactions:
            delta_h = interaction(h, edge_index, edge_feature, edge_vector)
            h[0] = h[0] + delta_h[0]
            h[1] = h[1] + delta_h[1]
        return h


def get_interaction_vn(config):
    if config.name == "cftfm":
        return CFTransformerEncoderVN(
            hidden_channels=[config.hidden_channels, config.hidden_channels_vec],
            edge_channels=config.edge_channels,
            key_channels=config.key_channels,
            num_heads=config.num_heads,
            num_interactions=config.num_interactions,
            k=config.knn,
            cutoff=config.cutoff,
        )
    raise NotImplementedError("Unknown encoder: %s" % config.name)


# ---------------------------------------------------------------------------
# models/fields/classifier.py :: SpatialClassifierVN  (only what sample_focal needs:
# FrontierLayerVN below reuses the encoder output directly, so the field module is not
# required for the traced sample_focal() path but is kept for a faithful ResGen.__init__)
# ---------------------------------------------------------------------------
class AttentionBias(Module):
    def __init__(self, num_heads, hidden_channels, cutoff=10.0, num_bond_types=3):
        super().__init__()
        num_edge_types = num_bond_types + 1
        self.num_bond_types = num_bond_types
        self.distance_expansion = GaussianSmearing(
            stop=cutoff, num_gaussians=hidden_channels[0] - num_edge_types - 1
        )
        self.vector_expansion = EdgeExpansion(hidden_channels[1])
        self.gvlinear = GVLinear(hidden_channels[0], hidden_channels[1], num_heads, num_heads)

    def forward(self, tri_edge_index, tri_edge_feat, pos_compose):
        node_a, node_b = tri_edge_index
        pos_a = pos_compose[node_a]
        pos_b = pos_compose[node_b]
        vector = pos_a - pos_b
        dist = torch.norm(vector, p=2, dim=-1)

        dist_feat = self.distance_expansion(dist)
        sca_feat = torch.cat([dist_feat, tri_edge_feat], dim=-1)
        vec_feat = self.vector_expansion(vector)
        output_sca, output_vec = self.gvlinear([sca_feat, vec_feat])
        output_vec = (output_vec * output_vec).sum(-1)
        return output_sca, output_vec


class AttentionEdges(Module):
    def __init__(self, hidden_channels, key_channels, num_heads=1, num_bond_types=3):
        super().__init__()
        assert (hidden_channels[0] % num_heads == 0) and (hidden_channels[1] % num_heads == 0)
        assert (key_channels[0] % num_heads == 0) and (key_channels[1] % num_heads == 0)

        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads

        self.q_lin = GVLinear(
            hidden_channels[0], hidden_channels[1], key_channels[0], key_channels[1]
        )
        self.k_lin = GVLinear(
            hidden_channels[0], hidden_channels[1], key_channels[0], key_channels[1]
        )
        self.v_lin = GVLinear(
            hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1]
        )

        self.atten_bias_lin = AttentionBias(
            self.num_heads, hidden_channels, num_bond_types=num_bond_types
        )
        self.layernorm_sca = nn.LayerNorm([hidden_channels[0]])
        self.layernorm_vec = nn.LayerNorm([hidden_channels[1], 3])

    def forward(
        self,
        edge_attr,
        edge_index,
        pos_compose,
        index_real_cps_edge_for_atten,
        tri_edge_index,
        tri_edge_feat,
    ):
        scalar, vector = edge_attr
        N = scalar.size(0)

        h_queries = self.q_lin(edge_attr)
        h_queries = (
            h_queries[0].view(N, self.num_heads, -1),
            h_queries[1].view(N, self.num_heads, -1, 3),
        )
        h_keys = self.k_lin(edge_attr)
        h_keys = (h_keys[0].view(N, self.num_heads, -1), h_keys[1].view(N, self.num_heads, -1, 3))
        h_values = self.v_lin(edge_attr)
        h_values = (
            h_values[0].view(N, self.num_heads, -1),
            h_values[1].view(N, self.num_heads, -1, 3),
        )

        index_edge_i_list, index_edge_j_list = index_real_cps_edge_for_atten

        atten_bias = self.atten_bias_lin(tri_edge_index, tri_edge_feat, pos_compose)

        queries_i = [h_queries[0][index_edge_i_list], h_queries[1][index_edge_i_list]]
        keys_j = [h_keys[0][index_edge_j_list], h_keys[1][index_edge_j_list]]

        qk_ij = [
            (queries_i[0] * keys_j[0]).sum(-1),
            (queries_i[1] * keys_j[1]).sum(-1).sum(-1),
        ]

        alpha = [atten_bias[0] + qk_ij[0], atten_bias[1] + qk_ij[1]]
        alpha = [
            scatter_softmax(alpha[0], index_edge_i_list, dim=0),
            scatter_softmax(alpha[1], index_edge_i_list, dim=0),
        ]

        values_j = [h_values[0][index_edge_j_list], h_values[1][index_edge_j_list]]
        num_attens = len(index_edge_j_list)
        output = [
            scatter_sum(
                (alpha[0].unsqueeze(-1) * values_j[0]).view(num_attens, -1),
                index_edge_i_list,
                dim=0,
                dim_size=N,
            ),
            scatter_sum(
                (alpha[1].unsqueeze(-1).unsqueeze(-1) * values_j[1]).view(num_attens, -1, 3),
                index_edge_i_list,
                dim=0,
                dim_size=N,
            ),
        ]

        output = [edge_attr[0] + output[0], edge_attr[1] + output[1]]
        output = [self.layernorm_sca(output[0]), self.layernorm_vec(output[1])]
        return output


class SpatialClassifierVN(Module):
    def __init__(
        self,
        num_classes,
        num_bond_types,
        in_sca,
        in_vec,
        num_filters,
        edge_channels,
        num_heads,
        k=32,
        cutoff=10.0,
    ):
        super().__init__()
        self.num_bond_types = num_bond_types
        self.message_module = MessageModule(
            in_sca, in_vec, edge_channels, edge_channels, num_filters[0], num_filters[1], cutoff
        )

        self.nn_edge_ij = nn.Sequential(
            GVPerceptronVN(edge_channels, edge_channels, num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1]),
        )

        self.classifier = nn.Sequential(
            GVPerceptronVN(num_filters[0], num_filters[1], num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_classes, 1),
        )

        self.edge_feat = nn.Sequential(
            GVPerceptronVN(
                num_filters[0] * 2 + in_sca,
                num_filters[1] * 2 + in_vec,
                num_filters[0],
                num_filters[1],
            ),
            GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1]),
        )
        self.edge_atten = AttentionEdges(num_filters, num_filters, num_heads, num_bond_types)
        self.edge_pred = GVLinear(num_filters[0], num_filters[1], num_bond_types + 1, 1)

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.distance_expansion_3A = GaussianSmearing(stop=3.0, num_gaussians=edge_channels)
        self.vector_expansion = EdgeExpansion(edge_channels)
        self.k = k
        self.cutoff = cutoff

    def forward(
        self,
        pos_query,
        edge_index_query,
        pos_compose,
        node_attr_compose,
        edge_index_q_cps_knn,
        index_real_cps_edge_for_atten=(),
        tri_edge_index=(),
        tri_edge_feat=(),
    ):
        vec_ij = pos_query[edge_index_q_cps_knn[0]] - pos_compose[edge_index_q_cps_knn[1]]
        dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)
        edge_ij = self.distance_expansion(dist_ij), self.vector_expansion(vec_ij)

        h = self.message_module(
            node_attr_compose, edge_ij, edge_index_q_cps_knn[1], dist_ij, annealing=True
        )

        y = [
            scatter_add(h[0], index=edge_index_q_cps_knn[0], dim=0, dim_size=pos_query.size(0)),
            scatter_add(h[1], index=edge_index_q_cps_knn[0], dim=0, dim_size=pos_query.size(0)),
        ]

        y_cls, _ = self.classifier(y)

        if len(edge_index_query) != 0 and edge_index_query.size(1) > 0:
            idx_node_i = edge_index_query[0]
            node_mol_i = [y[0][idx_node_i], y[1][idx_node_i]]
            idx_node_j = edge_index_query[1]
            node_mol_j = [node_attr_compose[0][idx_node_j], node_attr_compose[1][idx_node_j]]
            vec_ij = pos_query[idx_node_i] - pos_compose[idx_node_j]
            dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)

            edge_ij = self.distance_expansion_3A(dist_ij), self.vector_expansion(vec_ij)
            edge_feat = self.nn_edge_ij(edge_ij)

            edge_attr = (
                torch.cat([node_mol_i[0], node_mol_j[0], edge_feat[0]], dim=-1),
                torch.cat([node_mol_i[1], node_mol_j[1], edge_feat[1]], dim=1),
            )
            edge_attr = self.edge_feat(edge_attr)
            edge_attr = self.edge_atten(
                edge_attr,
                edge_index_query,
                pos_compose,
                index_real_cps_edge_for_atten,
                tri_edge_index,
                tri_edge_feat,
            )
            edge_pred, _ = self.edge_pred(edge_attr)
        else:
            edge_pred = torch.empty([0, self.num_bond_types + 1], device=pos_query.device)

        return y_cls, edge_pred


def get_field_vn(config, num_classes, num_bond_types, in_sca, in_vec):
    if config.name == "classifier":
        return SpatialClassifierVN(
            num_classes=num_classes,
            num_bond_types=num_bond_types,
            in_vec=in_vec,
            in_sca=in_sca,
            num_filters=[config.num_filters, config.num_filters_vec],
            edge_channels=config.edge_channels,
            num_heads=config.num_heads,
            k=config.knn,
            cutoff=config.cutoff,
        )
    raise NotImplementedError("Unknown field: %s" % config.name)


# ---------------------------------------------------------------------------
# models/frontier.py :: FrontierLayerVN
# ---------------------------------------------------------------------------
class FrontierLayerVN(Module):
    def __init__(self, in_sca, in_vec, hidden_dim_sca, hidden_dim_vec):
        super().__init__()
        self.net = nn.Sequential(
            GVPerceptronVN(in_sca, in_vec, hidden_dim_sca, hidden_dim_vec),
            GVLinear(hidden_dim_sca, hidden_dim_vec, 1, 1),
        )

    def forward(self, h_att, idx_ligans):
        h_att_ligand = [h_att[0][idx_ligans], h_att[1][idx_ligans]]
        pred = self.net(h_att_ligand)
        pred = pred[0]
        return pred


# ---------------------------------------------------------------------------
# models/position.py :: PositionPredictor
# ---------------------------------------------------------------------------
class PositionPredictor(Module):
    def __init__(self, in_sca, in_vec, num_filters, n_component):
        super().__init__()
        self.n_component = n_component
        self.gvp = nn.Sequential(
            GVPerceptronVN(in_sca, in_vec, num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1]),
        )
        self.mu_net = GVLinear(num_filters[0], num_filters[1], n_component, n_component)
        self.logsigma_net = GVLinear(num_filters[0], num_filters[1], n_component, n_component)
        self.pi_net = GVLinear(num_filters[0], num_filters[1], n_component, 1)

    def forward(self, h_compose, idx_focal, pos_compose):
        h_focal = [h[idx_focal] for h in h_compose]
        pos_focal = pos_compose[idx_focal]

        feat_focal = self.gvp(h_focal)
        relative_mu = self.mu_net(feat_focal)[1]
        logsigma = self.logsigma_net(feat_focal)[1]
        sigma = torch.exp(logsigma)
        pi = self.pi_net(feat_focal)[0]
        pi = F.softmax(pi, dim=1)

        abs_mu = relative_mu + pos_focal.unsqueeze(1).expand_as(relative_mu)
        return relative_mu, abs_mu, sigma, pi


# ---------------------------------------------------------------------------
# models/ResGen.py :: embed_compose_GVP + ResGen
# ---------------------------------------------------------------------------
def embed_compose_GVP(
    compose_feature,
    compose_vec,
    idx_ligand,
    idx_protein,
    ligand_atom_emb,
    protein_res_emb,
    emb_dim,
    ligand_atom_feature=13,
):
    protein_nodes = (compose_feature[idx_protein], compose_vec[idx_protein])
    ligand_nodes = (
        compose_feature[idx_ligand][:, :ligand_atom_feature],
        compose_vec[idx_ligand][:, 0, :].unsqueeze(-2),
    )
    h_protein = protein_res_emb(protein_nodes)
    h_ligand = ligand_atom_emb(ligand_nodes)

    h_sca = torch.zeros([len(compose_feature), emb_dim[0]]).to(h_ligand[0])
    h_vec = torch.zeros([len(compose_feature), emb_dim[1], 3]).to(h_ligand[1])
    h_sca[idx_ligand], h_sca[idx_protein] = h_ligand[0], h_protein[0]
    h_vec[idx_ligand], h_vec[idx_protein] = h_ligand[1], h_protein[1]
    return [h_sca, h_vec]


class ResGen(Module):
    """
    :protein_res_feature_dim: (scalar dim, vector dim) of input protein-residue features
        default: (27, 3)  # 6(dihedral) + 20(AA) + 1(is_mol_atom)
    :ligand_atom_feature_dim: (scalar dim, vector dim) of input ligand-atom features
        default: (13, 1)
    """

    def __init__(
        self, config, num_classes, num_bond_types, protein_res_feature_dim, ligand_atom_feature_dim
    ):
        super().__init__()
        self.config = config
        self.num_bond_types = num_bond_types

        self.emb_dim = [config.hidden_channels, config.hidden_channels_vec]
        self.protein_res_emb = GVP(protein_res_feature_dim, self.emb_dim)
        self.ligand_atom_emb = GVP(ligand_atom_feature_dim, self.emb_dim)

        self.encoder = get_interaction_vn(config.encoder)
        in_sca, in_vec = self.encoder.out_sca, self.encoder.out_vec
        self.field = get_field_vn(
            config.field,
            num_classes=num_classes,
            num_bond_types=num_bond_types,
            in_sca=in_sca,
            in_vec=in_vec,
        )
        self.frontier_pred = FrontierLayerVN(
            in_sca=in_sca, in_vec=in_vec, hidden_dim_sca=128, hidden_dim_vec=32
        )
        self.pos_predictor = PositionPredictor(
            in_sca=in_sca,
            in_vec=in_vec,
            num_filters=[config.position.num_filters] * 2,
            n_component=config.position.n_component,
        )

        self.smooth_cross_entropy = SmoothCrossEntropyLoss(reduction="mean", smoothing=0.1)
        self.bceloss_with_logits = nn.BCEWithLogitsLoss()

    def sample_focal(
        self,
        compose_feature,
        compose_vec,
        compose_pos,
        idx_ligand,
        idx_protein,
        compose_knn_edge_index,
        compose_knn_edge_feature,
        n_samples=-1,
        frontier_threshold=0,
    ):
        """Real ResGen inference entry point (identical to upstream ResGen.py). Encodes the
        protein/ligand 'compose' k-NN graph with the GVP+VN-transformer encoder and predicts
        frontier-atom logits -- the first autoregressive-generation step used by sample_init().
        """
        h_compose = embed_compose_GVP(
            compose_feature,
            compose_vec,
            idx_ligand,
            idx_protein,
            self.ligand_atom_emb,
            self.protein_res_emb,
            self.emb_dim,
        )
        h_compose = self.encoder(
            node_attr=h_compose,
            pos=compose_pos,
            edge_index=compose_knn_edge_index,
            edge_feature=compose_knn_edge_feature,
        )
        if len(idx_ligand) == 0:
            idx_ligand = idx_protein
        y_frontier_pred = self.frontier_pred(h_compose, idx_ligand)[:, 0]
        ind_frontier = y_frontier_pred > frontier_threshold
        has_frontier = torch.sum(ind_frontier) > 0
        if has_frontier:
            idx_frontier = idx_ligand[ind_frontier]
            p_frontier = torch.sigmoid(y_frontier_pred[ind_frontier])
            idx_focal_in_compose = torch.nonzero(ind_frontier)[:, 0]
            p_focal = p_frontier
            return (
                has_frontier,
                idx_frontier,
                p_frontier,
                idx_focal_in_compose,
                p_focal,
                h_compose,
            )
        return (has_frontier, h_compose)


# ---------------------------------------------------------------------------
# Staging harness (tiny random-init construction + example input)
# ---------------------------------------------------------------------------
class ResGenFrontierHead(Module):
    """Thin wrapper so TorchLens traces a plain forward(): builds the protein-only compose
    graph exactly as ResGen.sample_init() does (idx_ligand=empty) and returns frontier logits.
    """

    def __init__(self, resgen: ResGen):
        super().__init__()
        self.resgen = resgen

    def forward(
        self,
        compose_feature,
        compose_vec,
        compose_pos,
        idx_protein,
        compose_knn_edge_index,
        compose_knn_edge_feature,
    ):
        idx_ligand = torch.empty(0, dtype=torch.long, device=idx_protein.device)
        result = self.resgen.sample_focal(
            compose_feature,
            compose_vec,
            compose_pos,
            idx_ligand,
            idx_protein,
            compose_knn_edge_index,
            compose_knn_edge_feature,
        )
        # result[-1] is always h_compose = [h_sca, h_vec]; return the scalar frontier channel
        h_compose = result[-1]
        return h_compose[0]


def _tiny_config():
    return SimpleNamespace(
        hidden_channels=16,
        hidden_channels_vec=8,
        encoder=SimpleNamespace(
            name="cftfm",
            hidden_channels=16,
            hidden_channels_vec=8,
            edge_channels=16,
            key_channels=8,
            num_heads=2,
            num_interactions=2,
            cutoff=10.0,
            knn=8,
        ),
        field=SimpleNamespace(
            name="classifier",
            num_filters=16,
            num_filters_vec=4,
            edge_channels=16,
            num_heads=2,
            cutoff=10.0,
            knn=8,
        ),
        position=SimpleNamespace(num_filters=16, n_component=3),
    )


def build_resgen_frontier():
    config = _tiny_config()
    resgen = ResGen(
        config=config,
        num_classes=8,
        num_bond_types=3,
        protein_res_feature_dim=(27, 3),
        ligand_atom_feature_dim=(13, 1),
    )
    return ResGenFrontierHead(resgen)


def example_input_resgen_frontier():
    n_protein = 12
    compose_feature = torch.randn(n_protein, 27)
    compose_vec = torch.randn(n_protein, 3, 3)
    compose_pos = torch.randn(n_protein, 3) * 5.0
    idx_protein = torch.arange(n_protein, dtype=torch.long)

    k = 4
    src = torch.arange(n_protein).repeat_interleave(k)
    dst = torch.randint(0, n_protein, (n_protein * k,))
    compose_knn_edge_index = torch.stack([src, dst], dim=0)
    compose_knn_edge_feature = torch.cat(
        [
            torch.ones(compose_knn_edge_index.size(1), 1),
            torch.zeros(compose_knn_edge_index.size(1), 3),
        ],
        dim=-1,
    )

    return (
        compose_feature,
        compose_vec,
        compose_pos,
        idx_protein,
        compose_knn_edge_index,
        compose_knn_edge_feature,
    )


MENAGERIE_ENTRIES = [
    ("ResGen", build_resgen_frontier, example_input_resgen_frontier, 2023, MENAGERIE_ZOO),
]
