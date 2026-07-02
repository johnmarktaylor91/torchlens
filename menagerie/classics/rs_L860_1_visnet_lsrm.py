# SOURCE: vendored from liyy2/LSR-MP @ main (2026-07-01)
# Files combined from the real repo (paths as in upstream):
#   lightnp/LSRM/models/lsrm_modules.py                    (Visnorm_shared_LSRMNorm2_2branchSerial --
#                                                            the ViSNet-LSRM model class, and its
#                                                            Node_Edge_Fea_Init / Edge_Feat_Init /
#                                                            Bipartite_Edge_Feat_Init helper modules)
#   lightnp/LSRM/models/long_short_interact_modules.py     (LongShortIneractModel_dis_direct and
#                                                            LongShortIneractModel_dis_direct_vector2_drop
#                                                            -- the long-short-range message-passing
#                                                            interaction block, the paper's contribution)
#   lightnp/LSRM/models/output_net.py                      (OutputNet, subset used by the model)
#   lightnp/LSRM/models/utils.py                           (get_distance)
#   lightnp/LSRM/models/torchmdnet/models/torchmd_norm.py  (EquivariantMultiHeadAttention -- the
#                                                            short-range ViSNet-style attention block)
#   lightnp/LSRM/models/torchmdnet/models/utils.py         (ExpNormalSmearing, GaussianSmearing,
#                                                            NeighborEmbedding, CosineCutoff, norm,
#                                                            vec_layernorm, max_min_norm,
#                                                            act_class_mapping)
#
# ViSNet-LSRM ("Long-Short-Range Message-Passing", ICLR 2024, Li et al.) extends ViSNet's short-range
# equivariant attention with a second, long-range branch: atoms are grouped, group (super-node)
# embeddings are computed, and a bipartite node<->group message-passing block
# (LongShortIneractModel_dis_direct*) exchanges information across the long-range graph in parallel
# with the short-range ViSNet attention stack; the two branches are concatenated before the output head
# (`Visnorm_shared_LSRMNorm2_2branchSerial`, the "2branchSerial" variant). Code is transcribed verbatim
# from the real repo; only import paths were flattened into this single file.
#
# Fix (minimal, non-architectural): `lsrm_modules.py` imports `from ..utils import conditional_grad`,
# but `lightnp/LSRM/utils.py` does not exist in the repo (the package `__init__.py` itself references
# a missing `.utils` submodule -- the upstream repo is in a broken state for that one import). This is
# NOT part of the model architecture: it is the standard `conditional_grad` decorator used verbatim
# across the OCP/TorchMD-Net ecosystem (identical implementation in facebookresearch/fairchem,
# mlcommons/hpc open_catalyst, kyonofx/MDsim, divelab/AIRS -- all descended from the same
# open-catalyst-project/ocp `common/utils.py`). Supplied here verbatim from that canonical source so
# the real model classes below are otherwise untouched.
#
# Original license: MIT (per repo).

import math
from functools import wraps
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models.schnet import GaussianSmearing as _PygGaussianSmearing
from torch_geometric.utils import remove_self_loops, softmax
from torch_scatter import scatter

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# conditional_grad -- canonical OCP/TorchMD-Net-ecosystem utility (see note above);
# not part of lsrm_modules.py itself, supplied because the repo's own `..utils` is missing.
# ---------------------------------------------------------------------------
def conditional_grad(dec):
    """Decorator to enable/disable grad depending on whether force/energy predictions are made."""

    def decorator(func):
        @wraps(func)
        def cls_method(self, *args, **kwargs):
            f = func
            if getattr(self, "regress_forces", False) and not getattr(self, "direct_forces", 0):
                f = dec(func)
            return f(self, *args, **kwargs)

        return cls_method

    return decorator


# ---------------------------------------------------------------------------
# lightnp/LSRM/models/torchmdnet/models/utils.py
# ---------------------------------------------------------------------------
def norm(vec, dim=1, keepdim=False):
    return torch.square(vec + 1e-6).sum(dim=dim, keepdim=keepdim).sqrt() + 1e-6


def vec_layernorm(vec, norm_fn):
    dist = torch.norm(vec, dim=1, keepdim=True)
    if (dist == 0).all():
        return torch.zeros_like(vec)
    tmp_dist = torch.where(dist == 0, torch.ones_like(dist), dist)
    direct = vec / tmp_dist
    return F.relu(norm_fn(dist)) * direct


def max_min_norm(dist):
    max_val, _ = torch.max(dist, dim=-1)
    min_val, _ = torch.min(dist, dim=-1)
    delta = (max_val - min_val).view(-1)
    delta = torch.where(delta == 0, torch.ones_like(delta), delta)
    dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)
    return dist


class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff_lower, cutoff_upper, max_z=100):
        super(NeighborEmbedding, self).__init__(aggr="add")
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(GaussianSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(self.cutoff_lower, self.cutoff_upper, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


rbf_class_mapping = {"gauss": GaussianSmearing, "expnorm": ExpNormalSmearing}

act_class_mapping = {
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class EquivariantMultiHeadAttention(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        distance_influence,
        num_heads,
        activation,
        attn_activation,
        cutoff_lower,
        cutoff_upper,
        last_layer=False,
    ):
        super(EquivariantMultiHeadAttention, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of attention heads ({num_heads})"
        )
        if isinstance(activation, str):
            activation = act_class_mapping[activation]

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)

        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels)
            self.src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            self.dk_proj = nn.Linear(hidden_channels, hidden_channels)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            self.dv_proj = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def vector_rejection(self, vec, d_ij):
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.s_proj.weight)
        self.s_proj.bias.data.fill_(0)

        if not self.last_layer:
            nn.init.xavier_uniform_(self.f_proj.weight)
            self.f_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.src_proj.weight)
            nn.init.xavier_uniform_(self.trg_proj.weight)

        nn.init.xavier_uniform_(self.vec_proj.weight)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        x = self.layernorm(x)
        vec = vec_layernorm(vec, max_min_norm)
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dv_proj is not None
            else None
        )

        x, vec_out = self.propagate(
            edge_index, q=q, k=k, v=v, vec=vec, dk=dk, dv=dv, r_ij=r_ij, d_ij=d_ij, size=None
        )
        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if not self.last_layer:
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):
        if dk is None:
            attn = (q_i * k_j).sum(dim=-1)
        else:
            attn = (q_i * k_j * dk).sum(dim=-1)

        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        if dv is not None:
            v_j = v_j * dv

        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)

        s1, s2 = torch.split(self.act(self.s_proj(v_j)), self.hidden_channels, dim=1)
        vec = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)

        return v_j, vec

    def edge_update(self, vec_i, vec_j, d_ij, f_ij):
        w1 = self.vector_rejection(self.trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        return self.act(self.f_proj(f_ij)) * w_dot

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


# ---------------------------------------------------------------------------
# lightnp/LSRM/models/utils.py
# ---------------------------------------------------------------------------
def get_distance(source_pos, target_pos, edge_index):
    edge_vec = source_pos[edge_index[0]] - target_pos[edge_index[1]]
    mask = edge_index[0] == edge_index[1]
    edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
    edge_weight[~mask] = torch.norm(edge_vec[~mask], p=2, dim=1)
    return edge_index, edge_weight, edge_vec


# ---------------------------------------------------------------------------
# lightnp/LSRM/models/output_net.py (OutputNet only)
# ---------------------------------------------------------------------------
class OutputNet(nn.Module):
    def __init__(
        self,
        hidden_channels,
        act="silu",
        dipole=False,
        mean=None,
        std=None,
        atomref=None,
        scale=None,
        mean_std_adder="molecule_level",
    ) -> None:
        __MEAN_STD_ADDER__ = ["atom_level", "molecule_level"]
        super().__init__()
        self.dipole = dipole
        self.scale = scale
        self.readout = "sum"
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("atomref", atomref)
        act_class = act_class_mapping[act]
        self.output_network = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            act_class(),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.mean_std_adder = mean_std_adder
        assert self.mean_std_adder in __MEAN_STD_ADDER__
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.output_network[0].weight)
        self.output_network[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.output_network[2].weight)
        self.output_network[2].bias.data.fill_(0)

    def forward(self, h, v, data):
        z = data.atomic_numbers.long()
        pos = data.pos
        batch = data.batch
        h = self.output_network(h)
        if self.dipole:
            c = scatter(z * pos, batch, dim=0) / scatter(z, batch, dim=0)
            h = h * (pos - c.index_select(0, batch))

        if (
            not self.dipole
            and self.mean is not None
            and self.std is not None
            and self.mean_std_adder == "atom_level"
        ):
            h = h * self.std + self.mean

        out = scatter(h, batch, dim=0, reduce=self.readout)

        if (
            not self.dipole
            and self.mean is not None
            and self.std is not None
            and self.mean_std_adder == "molecule_level"
        ):
            out = out * self.std + self.mean

        if self.atomref is not None:
            out = out + scatter(self.atomref[z], batch, dim=0, reduce=self.readout)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return out


# ---------------------------------------------------------------------------
# lightnp/LSRM/models/long_short_interact_modules.py
# ---------------------------------------------------------------------------
class LongShortIneractModel_dis_direct(MessagePassing):
    def __init__(
        self, hidden_channels, num_gaussians, cutoff, norm=False, act="silu", num_heads=8, **kwargs
    ):
        super().__init__(aggr="add", node_dim=0)
        self.act = act_class_mapping[act]()
        self.norm = norm
        self.layernorm_node = nn.LayerNorm(hidden_channels)
        self.layernorm_group = nn.LayerNorm(hidden_channels)
        self.layernorm_node_vec = nn.LayerNorm(hidden_channels)
        self.layernorm_group_vec = nn.LayerNorm(hidden_channels)
        self.model_2 = nn.ModuleDict(
            {
                "q": nn.Linear(hidden_channels, hidden_channels),
                "k": nn.Linear(hidden_channels, hidden_channels),
                "val": nn.Linear(hidden_channels, hidden_channels),
                "mlp_scalar_pos": nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    self.act,
                    nn.Linear(hidden_channels, hidden_channels),
                ),
                "mlp_scalar_vec": nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    self.act,
                    nn.Linear(hidden_channels, hidden_channels),
                ),
                "linears": nn.ModuleList(
                    [nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(6)]
                ),
            }
        )
        self.model_1 = None
        self.num_heads = num_heads
        self.attn_channels = hidden_channels // num_heads
        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm_node.reset_parameters()
        self.layernorm_group.reset_parameters()
        for model in [self.model_1, self.model_2]:
            if model is None:
                continue
            for _, value in model.items():
                if isinstance(value, nn.ModuleList):
                    for m in value.modules():
                        if isinstance(m, nn.Linear):
                            torch.nn.init.xavier_uniform_(m.weight)
                elif isinstance(value, nn.Linear):
                    torch.nn.init.xavier_uniform_(value.weight)
                    value.bias.data.fill_(0)
                else:
                    pass

    def forward(
        self,
        edge_index,
        node_embedding,
        node_pos,
        node_vec,
        group_embedding,
        group_pos,
        group_vec,
        edge_attr,
        edge_weight,
        edge_vec,
    ):
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            group_embedding = self.layernorm_group(group_embedding)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]

        attn_2, val_2 = self.calculate_attention(
            node_embedding,
            group_embedding,
            edge_index[0],
            edge_index[1],
            edge_attr,
            self.model_2,
            "silu",
        )
        m_s_node, m_v_node = self.propagate(
            edge_index.flip(0),
            size=(num_groups, num_nodes),
            x=(group_embedding, node_embedding),
            v=group_vec[edge_index[1]],
            u_ij=-edge_vec,
            d_ij=edge_weight,
            attn_score=attn_2,
            val=val_2[edge_index[1]],
            mode="group_to_node",
        )

        v_node_1 = self.model_2["linears"][2](node_vec)
        v_node_2 = self.model_2["linears"][3](node_vec)
        dx_node = (v_node_1 * v_node_2).sum(dim=1) * self.model_2["linears"][4](
            m_s_node
        ) + self.model_2["linears"][5](m_s_node)
        dv_node = m_v_node + self.model_2["linears"][0](m_s_node).unsqueeze(1) * self.model_2[
            "linears"
        ][1](node_vec)
        return dx_node, dv_node

    def calculate_attention(
        self, x_1, x_2, x1_index, x2_index, expanded_edge_weight, model, attn_type
    ):
        __supported_attn__ = ["softmax", "silu"]
        q = model["q"](x_1).reshape(-1, self.num_heads, self.attn_channels)
        k = model["k"](x_2).reshape(-1, self.num_heads, self.attn_channels)
        val = model["val"](x_2).reshape(-1, self.num_heads, self.attn_channels)

        q_i = q[x1_index]
        k_j = k[x2_index]

        expanded_edge_weight = expanded_edge_weight.reshape(-1, self.num_heads, self.attn_channels)
        attn = q_i * k_j * expanded_edge_weight
        attn = attn.sum(dim=-1) / math.sqrt(self.attn_channels)

        if attn_type == "softmax":
            attn = softmax(attn, x1_index, dim=0)
        elif attn_type == "silu":
            attn = act_class_mapping["silu"]()(attn)
        else:
            raise NotImplementedError(
                f"Attention type {attn_type} is not supported, supported types are {__supported_attn__}"
            )
        return attn, val

    def message(self, x_i, x_j, v, u_ij, d_ij, attn_score, val, mode):
        if mode == "node_to_group":
            model = self.model_1
        else:
            model = self.model_2

        m_s_ij = val * attn_score.unsqueeze(-1)
        m_s_ij = m_s_ij.reshape(-1, self.num_heads * self.attn_channels)
        m_v_ij = model["mlp_scalar_pos"](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1) + model[
            "mlp_scalar_vec"
        ](m_s_ij).unsqueeze(1) * (v)
        return m_s_ij, m_v_ij

    def aggregate(self, features, index, ptr, dim_size):
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec


class LongShortIneractModel_dis_direct_vector2_drop(LongShortIneractModel_dis_direct):
    def __init__(
        self,
        hidden_channels,
        num_gaussians,
        cutoff,
        norm=False,
        act="silu",
        num_heads=8,
        p=0.1,
        **kwargs,
    ):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm, act, num_heads)
        self.dropout_s = nn.Dropout(p)
        self.dropout_v = nn.Dropout(p)
        self.p = p

    def forward(
        self,
        edge_index,
        node_embedding,
        node_pos,
        node_vec,
        group_embedding,
        group_pos,
        group_vec,
        edge_attr,
        edge_weight,
        edge_vec,
    ):
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            group_embedding = self.layernorm_group(group_embedding)
            group_vec = vec_layernorm(group_vec, max_min_norm)
        if self.p > 0:
            group_embedding = self.dropout_s(group_embedding)
            group_vec = self.dropout_v(group_vec)

        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]

        attn_2, val_2 = self.calculate_attention(
            node_embedding,
            group_embedding,
            edge_index[0],
            edge_index[1],
            edge_attr,
            self.model_2,
            "silu",
        )
        m_s_node, m_v_node = self.propagate(
            edge_index.flip(0),
            size=(num_groups, num_nodes),
            x=(group_embedding, node_embedding),
            v=group_vec[edge_index[1]],
            u_ij=-edge_vec,
            d_ij=edge_weight,
            attn_score=attn_2,
            val=val_2[edge_index[1]],
            mode="group_to_node",
        )

        v_node_1 = self.model_2["linears"][2](node_vec)
        v_node_2 = self.model_2["linears"][3](node_vec)
        dx_node = (v_node_1 * v_node_2).sum(dim=1) * self.model_2["linears"][4](
            m_s_node
        ) + self.model_2["linears"][5](m_s_node)
        dv_node = m_v_node + self.model_2["linears"][0](m_s_node).unsqueeze(1) * self.model_2[
            "linears"
        ][1](node_vec)
        return dx_node, dv_node


# ---------------------------------------------------------------------------
# lightnp/LSRM/models/lsrm_modules.py -- the ViSNet-LSRM model itself
# ---------------------------------------------------------------------------
class Node_Edge_Fea_Init(nn.Module):
    def __init__(
        self,
        max_z=100,
        rbf_type="expnorm",
        num_rbf=50,
        trainable_rbf=True,
        hidden_channels=128,
        cutoff_lower=0,
        cutoff_upper=5,
        neighbor_embedding=True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(max_z, hidden_channels)

        if rbf_type == "expnorm":
            rbf = ExpNormalSmearing
        elif rbf_type == "":
            rbf = GaussianSmearing
        else:
            assert False
        self.distance_encoder = rbf(
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            num_rbf=num_rbf,
            trainable=trainable_rbf,
        )
        self.rbf_linear = nn.Linear(num_rbf, hidden_channels)
        if neighbor_embedding:
            self.neighbor_embedding = NeighborEmbedding(
                hidden_channels, num_rbf, cutoff_lower, cutoff_upper, max_z
            )
        else:
            self.neighbor_embedding = None

    def forward(self, z, pos, edge_index):
        node_embedding = self.embedding(z)
        node_vec = torch.zeros(
            node_embedding.size(0), 3, node_embedding.size(1), device=node_embedding.device
        )
        edge_index, edge_weight, edge_vec = get_distance(pos, pos, edge_index)
        edge_attr = self.distance_encoder(edge_weight)
        edge_vec = edge_vec / norm(edge_vec, keepdim=True)
        if self.neighbor_embedding is not None:
            node_embedding = self.neighbor_embedding(
                z, node_embedding, edge_index, edge_weight, edge_attr
            )
        edge_attr = self.rbf_linear(edge_attr)
        return node_embedding, node_vec, edge_index, edge_weight, edge_attr, edge_vec


class Edge_Feat_Init(nn.Module):
    def __init__(
        self,
        rbf_type="expnorm",
        num_rbf=50,
        trainable_rbf=True,
        hidden_channels=128,
        cutoff_lower=0,
        cutoff_upper=5,
    ):
        super().__init__()
        if rbf_type == "expnorm":
            rbf = ExpNormalSmearing
        elif rbf_type == "":
            rbf = GaussianSmearing
        else:
            assert False
        self.distance_encoder = rbf(
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            num_rbf=num_rbf,
            trainable=trainable_rbf,
        )
        self.rbf_linear = nn.Linear(num_rbf, hidden_channels)

    def forward(self, pos, edge_index):
        edge_index, edge_weight, edge_vec = get_distance(pos, pos, edge_index)
        edge_attr = self.distance_encoder(edge_weight)
        edge_vec = edge_vec / norm(edge_vec, keepdim=True)
        edge_attr = self.rbf_linear(edge_attr)
        return edge_index, edge_weight, edge_attr, edge_vec


class Bipartite_Edge_Feat_Init(nn.Module):
    def __init__(
        self,
        rbf_type="expnorm",
        num_rbf=50,
        trainable_rbf=True,
        hidden_channels=128,
        cutoff_lower=0,
        cutoff_upper=10,
    ):
        super().__init__()
        if rbf_type == "expnorm":
            rbf = ExpNormalSmearing
        elif rbf_type == "":
            rbf = GaussianSmearing
        else:
            assert False
        self.distance_encoder = rbf(
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            num_rbf=num_rbf,
            trainable=trainable_rbf,
        )
        self.rbf_linear = nn.Linear(num_rbf, hidden_channels)

    def forward(self, edge_index, node_pos, group_pos, *args, **kwargs):
        edge_vec = node_pos[edge_index[0]] - group_pos[edge_index[1]]
        edge_weight = norm(edge_vec, dim=1)
        edge_vec = edge_vec / edge_weight.unsqueeze(1)
        edge_attr = self.distance_encoder(edge_weight)
        edge_attr = self.rbf_linear(edge_attr)
        return edge_index, edge_weight, edge_attr, edge_vec


class Visnorm_shared_LSRMNorm2_2branchSerial(nn.Module):
    """The ViSNet-LSRM model (ICLR 2024): short-range ViSNet-style attention branch +
    long-range node<->group interaction branch, concatenated before the output head."""

    def __init__(
        self,
        regress_forces=True,
        hidden_channels=128,
        num_layers=6,
        num_rbf=50,
        rbf_type="expnorm",
        trainable_rbf=True,
        neighbor_embedding=True,
        short_cutoff_upper=10,
        long_cutoff_upper=10,
        mean=None,
        std=None,
        atom_ref=None,
        max_z=100,
        group_center="center_of_mass",
        tf_writer=None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.regress_forces = regress_forces
        self.num_layers = num_layers
        self.group_center = group_center
        self.tf_writer = tf_writer
        self.t = 0

        self.node_fea_init = Node_Edge_Fea_Init(
            max_z=max_z,
            rbf_type=rbf_type,
            num_rbf=num_rbf,
            trainable_rbf=trainable_rbf,
            hidden_channels=hidden_channels,
            cutoff_lower=0,
            cutoff_upper=short_cutoff_upper,
            neighbor_embedding=neighbor_embedding,
        )
        self.mlp_node_fea = nn.Linear(hidden_channels, 2 * hidden_channels)
        self.mlp_node_vec_fea = nn.Linear(hidden_channels, 2 * hidden_channels, bias=False)

        self.edge_fea_init = Edge_Feat_Init(
            rbf_type=rbf_type,
            num_rbf=num_rbf,
            trainable_rbf=trainable_rbf,
            hidden_channels=hidden_channels,
            cutoff_lower=0,
            cutoff_upper=short_cutoff_upper,
        )

        self.bipartite_edge_fea_init = Bipartite_Edge_Feat_Init(
            rbf_type=rbf_type,
            num_rbf=num_rbf,
            trainable_rbf=trainable_rbf,
            hidden_channels=hidden_channels,
            cutoff_lower=0,
            cutoff_upper=long_cutoff_upper,
        )

        self.long_cutoff_upper = long_cutoff_upper

        self.visnet_att0 = nn.ModuleList()
        self.longshortinteract_models = nn.ModuleList()
        for _ in range(self.num_layers):
            self.visnet_att0.append(
                EquivariantMultiHeadAttention(
                    hidden_channels,
                    distance_influence="both",
                    num_heads=8,
                    activation="silu",
                    attn_activation="silu",
                    cutoff_lower=0,
                    cutoff_upper=short_cutoff_upper,
                    last_layer=False,
                )
            )

        config = kwargs["config"]
        self.long_num_layers = config["long_num_layers"]
        for i in range(self.long_num_layers):
            self.longshortinteract_models.append(
                LongShortIneractModel_dis_direct_vector2_drop(
                    hidden_channels,
                    num_gaussians=50,
                    cutoff=self.long_cutoff_upper,
                    norm=True,
                    max_group_num=3,
                    act="silu",
                    num_heads=8,
                    p=config["dropout"],
                )
            )

        self.out_norm1 = nn.LayerNorm(hidden_channels)
        self.out_norm2 = nn.LayerNorm(hidden_channels)
        self.out_energy = OutputNet(
            hidden_channels * 2,
            act="silu",
            dipole=False,
            mean=mean,
            std=std,
            atomref=atom_ref,
            scale=None,
        )

    @conditional_grad(torch.enable_grad())
    def forward(self, data, *args, **kwargs):
        if self.regress_forces:
            data.pos.requires_grad_(True)

        data.edge_index = remove_self_loops(data.edge_index)[0]
        z = data.atomic_numbers.long()
        pos = data.pos
        labels = data.labels
        atomic_numbers = data.atomic_numbers

        if z.dim() == 2:
            z = z.squeeze()
        device = pos.device
        if self.group_center == "geometric":
            group_pos = scatter(pos, data.labels, reduce="mean", dim=0)
        elif self.group_center == "center_of_mass":
            group_pos = scatter(pos * atomic_numbers, labels, reduce="sum", dim=0) / scatter(
                atomic_numbers, labels, reduce="sum", dim=0
            )
        else:
            assert False
        node_id, group_id = data.interaction_graph[0], data.interaction_graph[1]
        node_group_dis = torch.sqrt(torch.sum((pos[node_id] - group_pos[group_id]) ** 2, dim=1))
        data.interaction_graph = data.interaction_graph[:, node_group_dis <= self.long_cutoff_upper]
        group_embedding = None
        group_vec = torch.zeros((group_pos.shape[0], 3, self.hidden_channels), device=device)
        (
            node_embedding,
            node_vec,
            edge_index_short,
            edge_weight_short,
            edge_attr_short,
            edge_vec_short,
        ) = self.node_fea_init(z, pos, data.edge_index)
        edge_index_bipartite, edge_weight_bipartite, edge_attr_bipartite, edge_vec_bipartite = (
            self.bipartite_edge_fea_init(data.interaction_graph, pos, group_pos)
        )
        node_embedding_short, node_embedding_long = torch.split(
            self.mlp_node_fea(node_embedding), self.hidden_channels, dim=-1
        )
        node_vec_short, node_vec_long = torch.split(
            self.mlp_node_vec_fea(node_vec), self.hidden_channels, dim=-1
        )
        for idx in range(self.num_layers):
            delta_node_embedding_short, delta_node_vec_short, dedge_attr_short = self.visnet_att0[
                idx
            ](
                node_embedding_short,
                node_vec_short,
                edge_index_short,
                edge_weight_short,
                edge_attr_short,
                edge_vec_short,
            )
            node_embedding_short = node_embedding_short + delta_node_embedding_short
            node_vec_short = node_vec_short + delta_node_vec_short
            edge_attr_short = edge_attr_short + dedge_attr_short
        if self.long_num_layers != 0:
            node_embedding_long = node_embedding_short
            node_vec_long = node_vec_short
        else:
            node_embedding_long = node_embedding_long * 0
            node_vec_long = node_vec_long * 0
        for idx in range(self.long_num_layers):
            group_embedding = scatter(node_embedding_long, labels, dim=0, reduce="mean")
            group_vec = scatter(node_vec_long, labels, dim=0, reduce="mean")
            delta_node_embedding_long, delta_node_vec_long = self.longshortinteract_models[idx](
                edge_index=edge_index_bipartite,
                node_embedding=node_embedding_long,
                node_pos=pos,
                node_vec=node_vec_long,
                group_embedding=group_embedding,
                group_pos=group_pos,
                group_vec=group_vec,
                edge_attr=edge_attr_bipartite,
                edge_weight=edge_weight_bipartite,
                edge_vec=edge_vec_bipartite,
            )

            node_embedding_long = node_embedding_long + delta_node_embedding_long
            node_vec_long = node_vec_long + delta_node_vec_long

        node_embedding_short = self.out_norm1(node_embedding_short)
        node_vec_short = vec_layernorm(node_vec_short, max_min_norm)
        node_embedding_long = self.out_norm2(node_embedding_long)
        node_vec_long = vec_layernorm(node_vec_long, max_min_norm)

        node_embedding_short = torch.cat([node_embedding_short, node_embedding_long], dim=-1)
        node_vec_short = torch.cat([node_vec_short, node_vec_long], dim=-1)

        energy = self.out_energy(node_embedding_short, node_vec_short, data)

        if self.regress_forces:
            forces = (
                -1
                * (
                    torch.autograd.grad(
                        energy,
                        data.pos,
                        grad_outputs=torch.ones_like(energy),
                        create_graph=True if self.training else False,
                        retain_graph=True if self.training else False,
                    )[0]
                )
            )
            return {"energy": energy, "forces": forces}
        else:
            return {"energy": energy}


# ---------------------------------------------------------------------------
# Menagerie build/example plumbing
# ---------------------------------------------------------------------------
def build_visnet_lsrm():
    """Tiny random-init ViSNet-LSRM model (short-range attention + 1 long-range interaction layer)."""
    torch.manual_seed(0)
    return Visnorm_shared_LSRMNorm2_2branchSerial(
        regress_forces=False,
        hidden_channels=32,
        num_layers=2,
        num_rbf=16,
        rbf_type="expnorm",
        trainable_rbf=True,
        neighbor_embedding=True,
        short_cutoff_upper=6.0,
        long_cutoff_upper=8.0,
        max_z=20,
        group_center="center_of_mass",
        config={"long_num_layers": 1, "dropout": 0.1},
    )


def example_input_visnet_lsrm():
    """A small molecule split into 2 groups (torch_geometric-style Data object) -- 8 atoms."""
    from torch_geometric.data import Data

    torch.manual_seed(0)
    n_atoms = 8
    pos = torch.randn(n_atoms, 3) * 2.0
    atomic_numbers = torch.randint(1, 10, (n_atoms, 1)).float()
    batch = torch.zeros(n_atoms, dtype=torch.long)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    edge_index = radius_graph(pos, r=6.0, loop=False)

    node_ids = torch.arange(n_atoms).repeat_interleave(2)
    group_ids = torch.tensor([0, 1]).repeat(n_atoms)
    interaction_graph = torch.stack([node_ids, group_ids], dim=0)

    data = Data(pos=pos, atomic_numbers=atomic_numbers, batch=batch)
    data.edge_index = edge_index
    data.labels = labels
    data.interaction_graph = interaction_graph
    return (data,)


MENAGERIE_ENTRIES = [
    ("ViSNet-LSRM", build_visnet_lsrm, example_input_visnet_lsrm, 2024, "REAL"),
]
