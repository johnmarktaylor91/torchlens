# SOURCE: vendored from XiaolongTang23/HPNet @ main (HPNet-Argoverse variant)
# Files combined:
#   HPNet-Argoverse/model/HPNet.py (HPNet.forward() control flow, reproduced as a plain
#       nn.Module -- see note below)
#   HPNet-Argoverse/modules/backbone.py (Backbone -- the historical-prediction-attention
#       propose+refine trajectory decoder)
#   HPNet-Argoverse/modules/map_encoder.py (MapEncoder -- centerline/lane graph encoder)
#   HPNet-Argoverse/layers/graph_attention.py (GraphAttention -- the shared triple
#       factorized-attention message-passing block)
#   HPNet-Argoverse/layers/two_layer_mlp.py (TwoLayerMLP)
#   HPNet-Argoverse/utils/process_data.py + utils/init_weights.py (geometry helpers,
#       weight init)
#
# HPNet (Tang et al., "HPNet: Dynamic Trajectory Forecasting with Historical Prediction
# Attention", CVPR 2024) is a multi-agent, multi-modal trajectory forecaster for Argoverse: a
# MapEncoder builds lane/centerline graph embeddings via GraphAttention (a shared
# torch_geometric MessagePassing block used identically for every edge type in the model), a
# Backbone embeds per-agent per-timestep "mode" tokens and runs alternating agent-agent
# (m2m_a), historical (m2m_h), and mode-self (m2m_s) attention -- the paper's "triple
# factorized attention" -- to propose K-mode trajectories per historical timestep, then
# repeats the same attention scheme over trajectory anchors to refine the proposals and emit
# per-mode probabilities. The "historical prediction attention" mechanism is exactly this:
# every historical timestep keeps its own live set of K trajectory proposals which attend to
# each other across time (m2m_h edges) as well as across agents and modes.
#
# Import-only fixes applied (no architectural change):
#   - The original `HPNet.py` is a `pytorch_lightning.LightningModule` that additionally
#     imports `argoverse.evaluation.competition_util.generate_forecasting_h5` (only used in
#     `on_test_end`, a submission-writing hook) and defines `training_step`/`validation_step`/
#     `test_step`/`configure_optimizers` (all training/eval-loop scaffolding, not needed to
#     build/trace the model). This file's `HPNet` is a plain `nn.Module` reproducing exactly
#     `HPNet.forward()`: `lane_embs = self.MapEncoder(data); pred = self.Backbone(data,
#     l_embs=lane_embs)`. `Backbone` and `MapEncoder` themselves, and every submodule they
#     use (`GraphAttention`, `TwoLayerMLP`, all `utils.process_data` geometry helpers), are
#     vendored unmodified.
#   - The example input builds a `torch_geometric.data.HeteroData` object with the same
#     field layout the repo's `ArgoverseV1Dataset.process()` produces (`data['agent']`,
#     `data['lane']`, `data['centerline']`, and the `('centerline','lane')` /
#     `('lane','lane')` edge-type stores), including the `batch`/`ptr` fields that
#     `torch_geometric.data.Batch.from_data_list()` would normally attach -- set directly
#     since we trace a single already-batched scene rather than going through the
#     dataset/dataloader pipeline. `data['lane']['visible_mask']` mirrors the repo's default
#     (`torch.ones(num_lanes, dtype=bool)`, from `transforms/lane_random_occlusion.py`'s
#     un-occluded base case -- occlusion is an optional 10%-ratio augmentation, not applied
#     here).
#   - `Backbone.forward()`/`MapEncoder.forward()` build several `*_valid_mask` boolean
#     tensors via `unsqueeze`+`&`+`drop_edge_between_samples` (elementwise `*` against a
#     broadcast batch-comparison mask) before feeding them to
#     `torch_geometric.utils.dense_to_sparse`, whose current (2.7.0) implementation calls
#     `.view(-1, adj.size(-1))` on that tensor. The repo's own mask-construction chain never
#     calls `.contiguous()`, and on this torch_geometric version the resulting non-contiguous
#     boolean tensor makes `.view()` raise `RuntimeError: ... Use .reshape(...) instead` --
#     this reproduces with plain torch_geometric alone (no torchlens involved), i.e. it is a
#     real repo/library version-compatibility bug, not an architectural issue. Fixed here by
#     routing every `dense_to_sparse` call through a local `_dense_to_sparse` wrapper that
#     calls `.contiguous()` on the input first (identical values, contiguous memory layout
#     only -- no behavioral change).
#
# MENAGERIE_ZOO = "vendored-pytorch"

import math
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import dense_to_sparse as _dense_to_sparse_impl
from torch_geometric.utils import softmax


def dense_to_sparse(adj: torch.Tensor):
    # NOTE: `.contiguous()` before calling into torch_geometric's `dense_to_sparse` (which
    # internally does `.view(...)`); see module header "Import-only fixes applied".
    return _dense_to_sparse_impl(adj.contiguous())


# ---------------------------------------------------------------------------
# utils/init_weights.py
# ---------------------------------------------------------------------------
def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif "weight_hh" in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif "weight_hr" in name:
                nn.init.xavier_uniform_(param)
            elif "bias_ih" in name:
                nn.init.zeros_(param)
            elif "bias_hh" in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif "weight_hh" in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif "bias_ih" in name:
                nn.init.zeros_(param)
            elif "bias_hh" in name:
                nn.init.zeros_(param)


# ---------------------------------------------------------------------------
# utils/process_data.py
# ---------------------------------------------------------------------------
def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


def get_index_of_A_in_B(list_A: Optional[List[Any]], list_B: Optional[List[Any]]) -> List[int]:
    if not list_A or not list_B:
        return []
    set_B = set(list_B)
    return [list_B.index(i) for i in list_A if i in set_B]


def generate_clockwise_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    matrix = (
        torch.zeros_like(angle)
        .unsqueeze(-1)
        .repeat_interleave(2, -1)
        .unsqueeze(-1)
        .repeat_interleave(2, -1)
    )
    matrix[..., 0, 0] = torch.cos(angle)
    matrix[..., 0, 1] = torch.sin(angle)
    matrix[..., 1, 0] = -torch.sin(angle)
    matrix[..., 1, 1] = torch.cos(angle)
    return matrix


def generate_counterclockwise_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    matrix = (
        torch.zeros_like(angle)
        .unsqueeze(-1)
        .repeat_interleave(2, -1)
        .unsqueeze(-1)
        .repeat_interleave(2, -1)
    )
    matrix[..., 0, 0] = torch.cos(angle)
    matrix[..., 0, 1] = -torch.sin(angle)
    matrix[..., 1, 0] = torch.sin(angle)
    matrix[..., 1, 1] = torch.cos(angle)
    return matrix


def compute_angles_lengths_2D(vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    length = torch.norm(vectors, dim=-1)
    theta = torch.atan2(vectors[..., 1], vectors[..., 0])
    return length, theta


def drop_edge_between_samples(
    valid_mask: torch.Tensor, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        batch_matrix = batch.unsqueeze(-1) == batch.unsqueeze(-2)
    else:
        batch_src, batch_dst = batch
        batch_matrix = batch_src.unsqueeze(-1) == batch_dst.unsqueeze(-2)
    valid_mask = valid_mask * batch_matrix.unsqueeze(0)
    return valid_mask


def transform_traj_to_local_coordinate(
    traj: torch.Tensor, position: torch.Tensor, heading: torch.Tensor
) -> torch.Tensor:
    traj = traj - position.unsqueeze(-2)
    rotation_matrix = generate_clockwise_rotation_matrix(heading)
    traj = torch.matmul(rotation_matrix.unsqueeze(-3), traj.unsqueeze(-1)).squeeze(-1)
    return traj


def transform_traj_to_global_coordinate(
    traj: torch.Tensor, position: torch.Tensor, heading: torch.Tensor
) -> torch.Tensor:
    rotation_matrix = generate_counterclockwise_rotation_matrix(heading)
    traj = torch.matmul(rotation_matrix.unsqueeze(-3), traj.unsqueeze(-1)).squeeze(-1)
    traj = traj + position.unsqueeze(-2)
    return traj


def transform_point_to_local_coordinate(
    point: torch.Tensor, position: torch.Tensor, heading: torch.Tensor
) -> torch.Tensor:
    point = point - position
    rotation_matrix = generate_clockwise_rotation_matrix(heading)
    point = torch.matmul(rotation_matrix, point.unsqueeze(-1)).squeeze(-1)
    return point


def generate_reachable_matrix(edge_index: torch.Tensor, num_hops: int, max_nodes: int) -> list:
    values = torch.ones(edge_index.size(1), device=edge_index.device)
    sparse_mat = torch.sparse_coo_tensor(edge_index, values, torch.Size([max_nodes, max_nodes]))

    reach_matrices = []
    current_matrix = sparse_mat.clone()
    for _ in range(num_hops):
        current_matrix = current_matrix.coalesce()
        current_matrix = torch.sparse_coo_tensor(
            current_matrix.indices(),
            torch.ones_like(current_matrix.values()),
            current_matrix.size(),
        )

        edge_index_now = current_matrix.coalesce().indices()
        reach_matrices.append(edge_index_now)

        next_matrix = torch.sparse.mm(current_matrix, sparse_mat)
        current_matrix = next_matrix
    return reach_matrices


# ---------------------------------------------------------------------------
# layers/two_layer_mlp.py
# ---------------------------------------------------------------------------
class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(TwoLayerMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(init_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.mlp(input)


# ---------------------------------------------------------------------------
# layers/graph_attention.py
# ---------------------------------------------------------------------------
class GraphAttention(MessagePassing):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        has_edge_attr: bool,
        if_self_attention: bool,
        **kwargs,
    ) -> None:
        super(GraphAttention, self).__init__(aggr="add", node_dim=0, **kwargs)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.has_edge_attr = has_edge_attr
        self.if_self_attention = if_self_attention

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        if has_edge_attr:
            self.edge_k = nn.Linear(hidden_dim, hidden_dim)
            self.edge_v = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.attn_drop = nn.Dropout(dropout)
        if if_self_attention:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
        else:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
            self.mha_prenorm_dst = nn.LayerNorm(hidden_dim)
        if has_edge_attr:
            self.mha_prenorm_edge = nn.LayerNorm(hidden_dim)
        self.ffn_prenorm = nn.LayerNorm(hidden_dim)
        self.apply(init_weights)

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.if_self_attention:
            x_src = x_dst = self.mha_prenorm_src(x)
        else:
            x_src, x_dst = x
            x_src = self.mha_prenorm_src(x_src)
            x_dst = self.mha_prenorm_dst(x_dst)
        if self.has_edge_attr:
            edge_attr = self.mha_prenorm_edge(edge_attr)
        x_dst = x_dst + self._mha_layer(x_src, x_dst, edge_index, edge_attr)
        x_dst = x_dst + self._ffn_layer(self.ffn_prenorm(x_dst))
        return x_dst

    def message(
        self,
        x_dst_i: torch.Tensor,
        x_src_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        query_i = self.q(x_dst_i).view(-1, self.num_heads, self.head_dim)
        key_j = self.k(x_src_j).view(-1, self.num_heads, self.head_dim)
        value_j = self.v(x_src_j).view(-1, self.num_heads, self.head_dim)
        if self.has_edge_attr:
            key_j = key_j + self.edge_k(edge_attr).view(-1, self.num_heads, self.head_dim)
            value_j = value_j + self.edge_v(edge_attr).view(-1, self.num_heads, self.head_dim)
        scale = self.head_dim**0.5
        weight = (query_i * key_j).sum(dim=-1) / scale
        weight = softmax(weight, index, ptr)
        weight = self.attn_drop(weight)
        return (value_j * weight.unsqueeze(-1)).view(-1, self.num_heads * self.head_dim)

    def _mha_layer(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.propagate(edge_index=edge_index, edge_attr=edge_attr, x_dst=x_dst, x_src=x_src)

    def _ffn_layer(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


# ---------------------------------------------------------------------------
# modules/map_encoder.py
# ---------------------------------------------------------------------------
class MapEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_hops: int, num_heads: int, dropout: float) -> None:
        super(MapEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout

        self._l2l_edge_type = ["adjacent", "predecessor", "successor"]

        self.c_emb_layer = TwoLayerMLP(input_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.l_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.c2l_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.l2l_emb_layer = TwoLayerMLP(input_dim=7, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.c2l_attn_layer = GraphAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            has_edge_attr=True,
            if_self_attention=False,
        )
        self.l2l_attn_layer = GraphAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            has_edge_attr=True,
            if_self_attention=True,
        )

        self.apply(init_weights)

    def forward(self, data) -> torch.Tensor:
        c_length = data["centerline"]["length"]
        c_embs = self.c_emb_layer(input=c_length.unsqueeze(-1))  # [(C1,...,Cb),D]

        l_length = data["lane"]["length"]
        l_is_intersection = data["lane"]["is_intersection"]
        l_turn_direction = data["lane"]["turn_direction"]
        l_traffic_control = data["lane"]["traffic_control"]
        l_input = torch.stack(
            [l_length, l_is_intersection, l_turn_direction, l_traffic_control], dim=-1
        )  # [(M1,...,Mb),4]
        l_embs = self.l_emb_layer(input=l_input)  # [(M1,...,Mb),D]

        # c2l
        c2l_position_c = data["centerline"]["position"]
        c2l_position_l = data["lane"]["position"]
        c2l_heading_c = data["centerline"]["heading"]
        c2l_heading_l = data["lane"]["heading"]
        c2l_edge_index = data["centerline", "lane"]["centerline_to_lane_edge_index"]
        c2l_edge_vector = transform_point_to_local_coordinate(
            c2l_position_c[c2l_edge_index[0]],
            c2l_position_l[c2l_edge_index[1]],
            c2l_heading_l[c2l_edge_index[1]],
        )
        c2l_edge_attr_length, c2l_edge_attr_theta = compute_angles_lengths_2D(c2l_edge_vector)
        c2l_edge_attr_heading = wrap_angle(
            c2l_heading_c[c2l_edge_index[0]] - c2l_heading_l[c2l_edge_index[1]]
        )
        c2l_edge_attr_input = torch.stack(
            [c2l_edge_attr_length, c2l_edge_attr_theta, c2l_edge_attr_heading], dim=-1
        )
        c2l_edge_attr_embs = self.c2l_emb_layer(input=c2l_edge_attr_input)

        # l2l
        l2l_position = data["lane"]["position"]
        l2l_heading = data["lane"]["heading"]
        l2l_edge_index = []
        l2l_edge_attr_type = []
        l2l_edge_attr_hop = []

        l2l_adjacent_edge_index = data["lane", "lane"]["adjacent_edge_index"]
        num_adjacent_edges = l2l_adjacent_edge_index.size(1)
        l2l_edge_index.append(l2l_adjacent_edge_index)
        l2l_edge_attr_type.append(
            F.one_hot(
                torch.tensor(self._l2l_edge_type.index("adjacent")),
                num_classes=len(self._l2l_edge_type),
            )
            .to(l2l_adjacent_edge_index.device)
            .repeat(num_adjacent_edges, 1)
        )
        l2l_edge_attr_hop.append(
            torch.ones(num_adjacent_edges, device=l2l_adjacent_edge_index.device)
        )

        num_lanes = data["lane"]["num_nodes"]
        l2l_predecessor_edge_index = data["lane", "lane"]["predecessor_edge_index"]
        l2l_predecessor_edge_index_all = generate_reachable_matrix(
            l2l_predecessor_edge_index, self.num_hops, num_lanes
        )
        for i in range(self.num_hops):
            num_edges_now = l2l_predecessor_edge_index_all[i].size(1)
            l2l_edge_index.append(l2l_predecessor_edge_index_all[i])
            l2l_edge_attr_type.append(
                F.one_hot(
                    torch.tensor(self._l2l_edge_type.index("predecessor")),
                    num_classes=len(self._l2l_edge_type),
                )
                .to(l2l_predecessor_edge_index.device)
                .repeat(num_edges_now, 1)
            )
            l2l_edge_attr_hop.append(
                (i + 1) * torch.ones(num_edges_now, device=l2l_predecessor_edge_index.device)
            )

        l2l_successor_edge_index = data["lane", "lane"]["successor_edge_index"]
        l2l_successor_edge_index_all = generate_reachable_matrix(
            l2l_successor_edge_index, self.num_hops, num_lanes
        )
        for i in range(self.num_hops):
            num_edges_now = l2l_successor_edge_index_all[i].size(1)
            l2l_edge_index.append(l2l_successor_edge_index_all[i])
            l2l_edge_attr_type.append(
                F.one_hot(
                    torch.tensor(self._l2l_edge_type.index("successor")),
                    num_classes=len(self._l2l_edge_type),
                )
                .to(l2l_successor_edge_index.device)
                .repeat(num_edges_now, 1)
            )
            l2l_edge_attr_hop.append(
                (i + 1) * torch.ones(num_edges_now, device=l2l_successor_edge_index.device)
            )

        l2l_edge_index = torch.cat(l2l_edge_index, dim=1)
        l2l_edge_attr_type = torch.cat(l2l_edge_attr_type, dim=0)
        l2l_edge_attr_hop = torch.cat(l2l_edge_attr_hop, dim=0)
        l2l_edge_vector = transform_point_to_local_coordinate(
            l2l_position[l2l_edge_index[0]],
            l2l_position[l2l_edge_index[1]],
            l2l_heading[l2l_edge_index[1]],
        )
        l2l_edge_attr_length, l2l_edge_attr_theta = compute_angles_lengths_2D(l2l_edge_vector)
        l2l_edge_attr_heading = wrap_angle(
            l2l_heading[l2l_edge_index[0]] - l2l_heading[l2l_edge_index[1]]
        )
        l2l_edge_attr_input = torch.cat(
            [
                l2l_edge_attr_length.unsqueeze(-1),
                l2l_edge_attr_theta.unsqueeze(-1),
                l2l_edge_attr_heading.unsqueeze(-1),
                l2l_edge_attr_hop.unsqueeze(-1),
                l2l_edge_attr_type,
            ],
            dim=-1,
        )
        l2l_edge_attr_embs = self.l2l_emb_layer(input=l2l_edge_attr_input)

        # attention
        l_embs = self.c2l_attn_layer(
            x=[c_embs, l_embs], edge_index=c2l_edge_index, edge_attr=c2l_edge_attr_embs
        )
        l_embs = self.l2l_attn_layer(
            x=l_embs, edge_index=l2l_edge_index, edge_attr=l2l_edge_attr_embs
        )

        return l_embs


# ---------------------------------------------------------------------------
# modules/backbone.py
# ---------------------------------------------------------------------------
class Backbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        pos_duration: int,
        pred_duration: int,
        a2a_radius: float,
        l2a_radius: float,
        num_attn_layers: int,
        num_modes: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super(Backbone, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.pos_duration = pos_duration
        self.pred_duration = pred_duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.num_attn_layers = num_attn_layers
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.dropout = dropout

        self.mode_tokens = nn.Embedding(num_modes, hidden_dim)

        self.a_emb_layer = TwoLayerMLP(input_dim=1, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.l2m_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.t2m_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.m2m_h_emb_layer = TwoLayerMLP(
            input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim
        )
        self.m2m_a_emb_layer = TwoLayerMLP(
            input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim
        )
        self.m2m_s_emb_layer = TwoLayerMLP(
            input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim
        )

        self.l2m_attn_layer = GraphAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            has_edge_attr=True,
            if_self_attention=False,
        )
        self.t2m_attn_layer = GraphAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            has_edge_attr=True,
            if_self_attention=False,
        )

        self.m2m_h_attn_layers = nn.ModuleList(
            [
                GraphAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    has_edge_attr=True,
                    if_self_attention=True,
                )
                for _ in range(num_attn_layers)
            ]
        )
        self.m2m_a_attn_layers = nn.ModuleList(
            [
                GraphAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    has_edge_attr=True,
                    if_self_attention=True,
                )
                for _ in range(num_attn_layers)
            ]
        )
        self.m2m_s_attn_layers = nn.ModuleList(
            [
                GraphAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    has_edge_attr=False,
                    if_self_attention=True,
                )
                for _ in range(num_attn_layers)
            ]
        )

        self.traj_propose = TwoLayerMLP(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.num_future_steps * 2
        )

        self.proposal_to_anchor = TwoLayerMLP(
            input_dim=self.num_future_steps * 2, hidden_dim=hidden_dim, output_dim=hidden_dim
        )

        self.l2n_attn_layer = GraphAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            has_edge_attr=True,
            if_self_attention=False,
        )
        self.t2n_attn_layer = GraphAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            has_edge_attr=True,
            if_self_attention=False,
        )

        self.n2n_h_attn_layers = nn.ModuleList(
            [
                GraphAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    has_edge_attr=True,
                    if_self_attention=True,
                )
                for _ in range(num_attn_layers)
            ]
        )
        self.n2n_a_attn_layers = nn.ModuleList(
            [
                GraphAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    has_edge_attr=True,
                    if_self_attention=True,
                )
                for _ in range(num_attn_layers)
            ]
        )
        self.n2n_s_attn_layers = nn.ModuleList(
            [
                GraphAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    has_edge_attr=True,
                    if_self_attention=True,
                )
                for _ in range(num_attn_layers)
            ]
        )

        self.traj_refine = TwoLayerMLP(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.num_future_steps * 2
        )

        self.prob_decoder = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.prob_norm = nn.Softmax(dim=-1)

        self.apply(init_weights)

    def forward(self, data, l_embs: torch.Tensor) -> torch.Tensor:
        # initialization
        a_length = data["agent"]["length"]  # [(N1,...,Nb),H]
        a_embs = self.a_emb_layer(input=a_length.unsqueeze(-1))  # [(N1,...,Nb),H,D]

        num_all_agent = a_length.size(0)
        m_embs = self.mode_tokens.weight.unsqueeze(0).repeat_interleave(
            self.num_historical_steps, 0
        )  # [H,K,D]
        m_embs = (
            m_embs.unsqueeze(0).repeat_interleave(num_all_agent, 0).reshape(-1, self.hidden_dim)
        )

        m_batch = data["agent"]["batch"].unsqueeze(1).repeat_interleave(self.num_modes, 1)
        m_position = (
            data["agent"]["position"][:, : self.num_historical_steps]
            .unsqueeze(2)
            .repeat_interleave(self.num_modes, 2)
        )
        m_heading = data["agent"]["heading"].unsqueeze(2).repeat_interleave(self.num_modes, 2)
        m_valid_mask = (
            data["agent"]["visible_mask"][:, : self.num_historical_steps]
            .unsqueeze(2)
            .repeat_interleave(self.num_modes, 2)
        )

        # t2m edge
        t2m_position_t = data["agent"]["position"][:, : self.num_historical_steps].reshape(-1, 2)
        t2m_position_m = m_position.reshape(-1, 2)
        t2m_heading_t = data["agent"]["heading"].reshape(-1)
        t2m_heading_m = m_heading.reshape(-1)
        t2m_valid_mask_t = data["agent"]["visible_mask"][:, : self.num_historical_steps]
        t2m_valid_mask_m = m_valid_mask.reshape(num_all_agent, -1)
        t2m_valid_mask = t2m_valid_mask_t.unsqueeze(2) & t2m_valid_mask_m.unsqueeze(1)
        t2m_edge_index = dense_to_sparse(t2m_valid_mask)[0]
        t2m_edge_index = t2m_edge_index[
            :, torch.floor(t2m_edge_index[1] / self.num_modes) >= t2m_edge_index[0]
        ]
        t2m_edge_index = t2m_edge_index[
            :,
            torch.floor(t2m_edge_index[1] / self.num_modes) - t2m_edge_index[0]
            <= self.pos_duration,
        ]
        t2m_edge_vector = transform_point_to_local_coordinate(
            t2m_position_t[t2m_edge_index[0]],
            t2m_position_m[t2m_edge_index[1]],
            t2m_heading_m[t2m_edge_index[1]],
        )
        t2m_edge_attr_length, t2m_edge_attr_theta = compute_angles_lengths_2D(t2m_edge_vector)
        t2m_edge_attr_heading = wrap_angle(
            t2m_heading_t[t2m_edge_index[0]] - t2m_heading_m[t2m_edge_index[1]]
        )
        t2m_edge_attr_interval = t2m_edge_index[0] - torch.floor(t2m_edge_index[1] / self.num_modes)
        t2m_edge_attr_input = torch.stack(
            [
                t2m_edge_attr_length,
                t2m_edge_attr_theta,
                t2m_edge_attr_heading,
                t2m_edge_attr_interval,
            ],
            dim=-1,
        )
        t2m_edge_attr_embs = self.t2m_emb_layer(input=t2m_edge_attr_input)

        # l2m edge
        l2m_position_l = data["lane"]["position"]
        l2m_position_m = m_position.reshape(-1, 2)
        l2m_heading_l = data["lane"]["heading"]
        l2m_heading_m = m_heading.reshape(-1)
        l2m_batch_l = data["lane"]["batch"]
        l2m_batch_m = (
            m_batch.unsqueeze(1).repeat_interleave(self.num_historical_steps, 1).reshape(-1)
        )
        l2m_valid_mask_l = data["lane"]["visible_mask"]
        l2m_valid_mask_m = m_valid_mask.reshape(-1)
        l2m_valid_mask = l2m_valid_mask_l.unsqueeze(1) & l2m_valid_mask_m.unsqueeze(0)
        l2m_valid_mask = drop_edge_between_samples(l2m_valid_mask, batch=(l2m_batch_l, l2m_batch_m))
        l2m_edge_index = dense_to_sparse(l2m_valid_mask)[0]
        l2m_edge_index = l2m_edge_index[
            :,
            torch.norm(
                l2m_position_l[l2m_edge_index[0]] - l2m_position_m[l2m_edge_index[1]], p=2, dim=-1
            )
            < self.l2a_radius,
        ]
        l2m_edge_vector = transform_point_to_local_coordinate(
            l2m_position_l[l2m_edge_index[0]],
            l2m_position_m[l2m_edge_index[1]],
            l2m_heading_m[l2m_edge_index[1]],
        )
        l2m_edge_attr_length, l2m_edge_attr_theta = compute_angles_lengths_2D(l2m_edge_vector)
        l2m_edge_attr_heading = wrap_angle(
            l2m_heading_l[l2m_edge_index[0]] - l2m_heading_m[l2m_edge_index[1]]
        )
        l2m_edge_attr_input = torch.stack(
            [l2m_edge_attr_length, l2m_edge_attr_theta, l2m_edge_attr_heading], dim=-1
        )
        l2m_edge_attr_embs = self.l2m_emb_layer(input=l2m_edge_attr_input)

        # m2m_a edge
        m2m_a_position = m_position.permute(1, 2, 0, 3).reshape(-1, 2)
        m2m_a_heading = m_heading.permute(1, 2, 0).reshape(-1)
        m2m_a_batch = data["agent"]["batch"]
        m2m_a_valid_mask = m_valid_mask.permute(1, 2, 0).reshape(
            self.num_historical_steps * self.num_modes, -1
        )
        m2m_a_valid_mask = m2m_a_valid_mask.unsqueeze(2) & m2m_a_valid_mask.unsqueeze(1)
        m2m_a_valid_mask = drop_edge_between_samples(m2m_a_valid_mask, m2m_a_batch)
        m2m_a_edge_index = dense_to_sparse(m2m_a_valid_mask)[0]
        m2m_a_edge_index = m2m_a_edge_index[:, m2m_a_edge_index[1] != m2m_a_edge_index[0]]
        m2m_a_edge_index = m2m_a_edge_index[
            :,
            torch.norm(
                m2m_a_position[m2m_a_edge_index[1]] - m2m_a_position[m2m_a_edge_index[0]],
                p=2,
                dim=-1,
            )
            < self.a2a_radius,
        ]
        m2m_a_edge_vector = transform_point_to_local_coordinate(
            m2m_a_position[m2m_a_edge_index[0]],
            m2m_a_position[m2m_a_edge_index[1]],
            m2m_a_heading[m2m_a_edge_index[1]],
        )
        m2m_a_edge_attr_length, m2m_a_edge_attr_theta = compute_angles_lengths_2D(m2m_a_edge_vector)
        m2m_a_edge_attr_heading = wrap_angle(
            m2m_a_heading[m2m_a_edge_index[0]] - m2m_a_heading[m2m_a_edge_index[1]]
        )
        m2m_a_edge_attr_input = torch.stack(
            [m2m_a_edge_attr_length, m2m_a_edge_attr_theta, m2m_a_edge_attr_heading], dim=-1
        )
        m2m_a_edge_attr_embs = self.m2m_a_emb_layer(input=m2m_a_edge_attr_input)

        # m2m_h edge
        m2m_h_position = m_position.permute(2, 0, 1, 3).reshape(-1, 2)
        m2m_h_heading = m_heading.permute(2, 0, 1).reshape(-1)
        m2m_h_valid_mask = m_valid_mask.permute(2, 0, 1).reshape(-1, self.num_historical_steps)
        m2m_h_valid_mask = m2m_h_valid_mask.unsqueeze(2) & m2m_h_valid_mask.unsqueeze(1)
        m2m_h_edge_index = dense_to_sparse(m2m_h_valid_mask)[0]
        m2m_h_edge_index = m2m_h_edge_index[:, m2m_h_edge_index[1] > m2m_h_edge_index[0]]
        m2m_h_edge_index = m2m_h_edge_index[
            :, m2m_h_edge_index[1] - m2m_h_edge_index[0] <= self.pred_duration
        ]
        m2m_h_edge_vector = transform_point_to_local_coordinate(
            m2m_h_position[m2m_h_edge_index[0]],
            m2m_h_position[m2m_h_edge_index[1]],
            m2m_h_heading[m2m_h_edge_index[1]],
        )
        m2m_h_edge_attr_length, m2m_h_edge_attr_theta = compute_angles_lengths_2D(m2m_h_edge_vector)
        m2m_h_edge_attr_heading = wrap_angle(
            m2m_h_heading[m2m_h_edge_index[0]] - m2m_h_heading[m2m_h_edge_index[1]]
        )
        m2m_h_edge_attr_interval = m2m_h_edge_index[0] - m2m_h_edge_index[1]
        m2m_h_edge_attr_input = torch.stack(
            [
                m2m_h_edge_attr_length,
                m2m_h_edge_attr_theta,
                m2m_h_edge_attr_heading,
                m2m_h_edge_attr_interval,
            ],
            dim=-1,
        )
        m2m_h_edge_attr_embs = self.m2m_h_emb_layer(input=m2m_h_edge_attr_input)

        # m2m_s edge
        m2m_s_valid_mask = m_valid_mask.transpose(0, 1).reshape(-1, self.num_modes)
        m2m_s_valid_mask = m2m_s_valid_mask.unsqueeze(2) & m2m_s_valid_mask.unsqueeze(1)
        m2m_s_edge_index = dense_to_sparse(m2m_s_valid_mask)[0]
        m2m_s_edge_index = m2m_s_edge_index[:, m2m_s_edge_index[0] != m2m_s_edge_index[1]]

        # t2m attention
        t_embs = a_embs.reshape(-1, self.hidden_dim)
        m_embs_t = self.t2m_attn_layer(
            x=[t_embs, m_embs], edge_index=t2m_edge_index, edge_attr=t2m_edge_attr_embs
        )

        # l2m attention
        m_embs_l = self.l2m_attn_layer(
            x=[l_embs, m_embs], edge_index=l2m_edge_index, edge_attr=l2m_edge_attr_embs
        )

        m_embs = m_embs_t + m_embs_l
        m_embs = (
            m_embs.reshape(
                num_all_agent, self.num_historical_steps, self.num_modes, self.hidden_dim
            )
            .transpose(0, 1)
            .reshape(-1, self.hidden_dim)
        )
        # mode attention
        for i in range(self.num_attn_layers):
            m_embs = (
                m_embs.reshape(
                    self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim
                )
                .transpose(1, 2)
                .reshape(-1, self.hidden_dim)
            )
            m_embs = self.m2m_a_attn_layers[i](
                x=m_embs, edge_index=m2m_a_edge_index, edge_attr=m2m_a_edge_attr_embs
            )
            m_embs = (
                m_embs.reshape(
                    self.num_historical_steps, self.num_modes, num_all_agent, self.hidden_dim
                )
                .permute(1, 2, 0, 3)
                .reshape(-1, self.hidden_dim)
            )
            m_embs = self.m2m_h_attn_layers[i](
                x=m_embs, edge_index=m2m_h_edge_index, edge_attr=m2m_h_edge_attr_embs
            )
            m_embs = (
                m_embs.reshape(
                    self.num_modes, num_all_agent, self.num_historical_steps, self.hidden_dim
                )
                .transpose(0, 2)
                .reshape(-1, self.hidden_dim)
            )
            m_embs = self.m2m_s_attn_layers[i](x=m_embs, edge_index=m2m_s_edge_index)
        m_embs = (
            m_embs.reshape(
                self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim
            )
            .transpose(0, 1)
            .reshape(-1, self.hidden_dim)
        )

        # generate traj
        traj_propose = self.traj_propose(m_embs).reshape(
            num_all_agent, self.num_historical_steps, self.num_modes, self.num_future_steps, 2
        )
        traj_propose = transform_traj_to_global_coordinate(traj_propose, m_position, m_heading)

        # generate anchor
        proposal = traj_propose.detach()

        n_batch = m_batch
        n_position = proposal[:, :, :, self.num_future_steps // 2, :]
        _, n_heading = compute_angles_lengths_2D(
            proposal[:, :, :, self.num_future_steps // 2, :]
            - proposal[:, :, :, (self.num_future_steps // 2) - 1, :]
        )
        n_valid_mask = m_valid_mask

        proposal = transform_traj_to_local_coordinate(proposal, n_position, n_heading)
        anchor = self.proposal_to_anchor(proposal.reshape(-1, self.num_future_steps * 2))
        n_embs = anchor

        # t2n edge
        t2n_position_t = data["agent"]["position"][:, : self.num_historical_steps].reshape(-1, 2)
        t2n_position_n = n_position.reshape(-1, 2)
        t2n_heading_t = data["agent"]["heading"].reshape(-1)
        t2n_heading_n = n_heading.reshape(-1)
        t2n_valid_mask_t = data["agent"]["visible_mask"][:, : self.num_historical_steps]
        t2n_valid_mask_n = n_valid_mask.reshape(num_all_agent, -1)
        t2n_valid_mask = t2n_valid_mask_t.unsqueeze(2) & t2n_valid_mask_n.unsqueeze(1)
        t2n_edge_index = dense_to_sparse(t2n_valid_mask)[0]
        t2n_edge_index = t2n_edge_index[
            :, torch.floor(t2n_edge_index[1] / self.num_modes) >= t2n_edge_index[0]
        ]
        t2n_edge_index = t2n_edge_index[
            :,
            torch.floor(t2n_edge_index[1] / self.num_modes) - t2n_edge_index[0]
            <= self.pos_duration,
        ]
        t2n_edge_vector = transform_point_to_local_coordinate(
            t2n_position_t[t2n_edge_index[0]],
            t2n_position_n[t2n_edge_index[1]],
            t2n_heading_n[t2n_edge_index[1]],
        )
        t2n_edge_attr_length, t2n_edge_attr_theta = compute_angles_lengths_2D(t2n_edge_vector)
        t2n_edge_attr_heading = wrap_angle(
            t2n_heading_t[t2n_edge_index[0]] - t2n_heading_n[t2n_edge_index[1]]
        )
        t2n_edge_attr_interval = (
            t2n_edge_index[0]
            - torch.floor(t2n_edge_index[1] / self.num_modes)
            - self.num_future_steps // 2
        )
        t2n_edge_attr_input = torch.stack(
            [
                t2n_edge_attr_length,
                t2n_edge_attr_theta,
                t2n_edge_attr_heading,
                t2n_edge_attr_interval,
            ],
            dim=-1,
        )
        t2n_edge_attr_embs = self.t2m_emb_layer(input=t2n_edge_attr_input)

        # l2n edge
        l2n_position_l = data["lane"]["position"]
        l2n_position_n = n_position.reshape(-1, 2)
        l2n_heading_l = data["lane"]["heading"]
        l2n_heading_n = n_heading.reshape(-1)
        l2n_batch_l = data["lane"]["batch"]
        l2n_batch_n = (
            n_batch.unsqueeze(1).repeat_interleave(self.num_historical_steps, 1).reshape(-1)
        )
        l2n_valid_mask_l = data["lane"]["visible_mask"]
        l2n_valid_mask_n = n_valid_mask.reshape(-1)
        l2n_valid_mask = l2n_valid_mask_l.unsqueeze(1) & l2n_valid_mask_n.unsqueeze(0)
        l2n_valid_mask = drop_edge_between_samples(l2n_valid_mask, batch=(l2n_batch_l, l2n_batch_n))
        l2n_edge_index = dense_to_sparse(l2n_valid_mask)[0]
        l2n_edge_index = l2n_edge_index[
            :,
            torch.norm(
                l2n_position_l[l2n_edge_index[0]] - l2n_position_n[l2n_edge_index[1]], p=2, dim=-1
            )
            < self.l2a_radius,
        ]
        l2n_edge_vector = transform_point_to_local_coordinate(
            l2n_position_l[l2n_edge_index[0]],
            l2n_position_n[l2n_edge_index[1]],
            l2n_heading_n[l2n_edge_index[1]],
        )
        l2n_edge_attr_length, l2n_edge_attr_theta = compute_angles_lengths_2D(l2n_edge_vector)
        l2n_edge_attr_heading = wrap_angle(
            l2n_heading_l[l2n_edge_index[0]] - l2n_heading_n[l2n_edge_index[1]]
        )
        l2n_edge_attr_input = torch.stack(
            [l2n_edge_attr_length, l2n_edge_attr_theta, l2n_edge_attr_heading], dim=-1
        )
        l2n_edge_attr_embs = self.l2m_emb_layer(input=l2n_edge_attr_input)

        # n2n_a edge
        n2n_a_position = n_position.permute(1, 2, 0, 3).reshape(-1, 2)
        n2n_a_heading = n_heading.permute(1, 2, 0).reshape(-1)
        n2n_a_batch = data["agent"]["batch"]
        n2n_a_valid_mask = n_valid_mask.permute(1, 2, 0).reshape(
            self.num_historical_steps * self.num_modes, -1
        )
        n2n_a_valid_mask = n2n_a_valid_mask.unsqueeze(2) & n2n_a_valid_mask.unsqueeze(1)
        n2n_a_valid_mask = drop_edge_between_samples(n2n_a_valid_mask, n2n_a_batch)
        n2n_a_edge_index = dense_to_sparse(n2n_a_valid_mask)[0]
        n2n_a_edge_index = n2n_a_edge_index[:, n2n_a_edge_index[1] != n2n_a_edge_index[0]]
        n2n_a_edge_index = n2n_a_edge_index[
            :,
            torch.norm(
                n2n_a_position[n2n_a_edge_index[1]] - n2n_a_position[n2n_a_edge_index[0]],
                p=2,
                dim=-1,
            )
            < self.a2a_radius,
        ]
        n2n_a_edge_vector = transform_point_to_local_coordinate(
            n2n_a_position[n2n_a_edge_index[0]],
            n2n_a_position[n2n_a_edge_index[1]],
            n2n_a_heading[n2n_a_edge_index[1]],
        )
        n2n_a_edge_attr_length, n2n_a_edge_attr_theta = compute_angles_lengths_2D(n2n_a_edge_vector)
        n2n_a_edge_attr_heading = wrap_angle(
            n2n_a_heading[n2n_a_edge_index[0]] - n2n_a_heading[n2n_a_edge_index[1]]
        )
        n2n_a_edge_attr_input = torch.stack(
            [n2n_a_edge_attr_length, n2n_a_edge_attr_theta, n2n_a_edge_attr_heading], dim=-1
        )
        n2n_a_edge_attr_embs = self.m2m_a_emb_layer(input=n2n_a_edge_attr_input)

        # n2n_h edge
        n2n_h_position = n_position.permute(2, 0, 1, 3).reshape(-1, 2)
        n2n_h_heading = n_heading.permute(2, 0, 1).reshape(-1)
        n2n_h_valid_mask = n_valid_mask.permute(2, 0, 1).reshape(-1, self.num_historical_steps)
        n2n_h_valid_mask = n2n_h_valid_mask.unsqueeze(2) & n2n_h_valid_mask.unsqueeze(1)
        n2n_h_edge_index = dense_to_sparse(n2n_h_valid_mask)[0]
        n2n_h_edge_index = n2n_h_edge_index[:, n2n_h_edge_index[1] > n2n_h_edge_index[0]]
        n2n_h_edge_index = n2n_h_edge_index[
            :, n2n_h_edge_index[1] - n2n_h_edge_index[0] <= self.pred_duration
        ]
        n2n_h_edge_vector = transform_point_to_local_coordinate(
            n2n_h_position[n2n_h_edge_index[0]],
            n2n_h_position[n2n_h_edge_index[1]],
            n2n_h_heading[n2n_h_edge_index[1]],
        )
        n2n_h_edge_attr_length, n2n_h_edge_attr_theta = compute_angles_lengths_2D(n2n_h_edge_vector)
        n2n_h_edge_attr_heading = wrap_angle(
            n2n_h_heading[n2n_h_edge_index[0]] - n2n_h_heading[n2n_h_edge_index[1]]
        )
        n2n_h_edge_attr_interval = n2n_h_edge_index[0] - n2n_h_edge_index[1]
        n2n_h_edge_attr_input = torch.stack(
            [
                n2n_h_edge_attr_length,
                n2n_h_edge_attr_theta,
                n2n_h_edge_attr_heading,
                n2n_h_edge_attr_interval,
            ],
            dim=-1,
        )
        n2n_h_edge_attr_embs = self.m2m_h_emb_layer(input=n2n_h_edge_attr_input)

        # n2n_s edge
        n2n_s_position = n_position.transpose(0, 1).reshape(-1, 2)
        n2n_s_heading = n_heading.transpose(0, 1).reshape(-1)
        n2n_s_valid_mask = n_valid_mask.transpose(0, 1).reshape(-1, self.num_modes)
        n2n_s_valid_mask = n2n_s_valid_mask.unsqueeze(2) & n2n_s_valid_mask.unsqueeze(1)
        n2n_s_edge_index = dense_to_sparse(n2n_s_valid_mask)[0]
        n2n_s_edge_index = n2n_s_edge_index[:, n2n_s_edge_index[0] != n2n_s_edge_index[1]]
        n2n_s_edge_vector = transform_point_to_local_coordinate(
            n2n_s_position[n2n_s_edge_index[0]],
            n2n_s_position[n2n_s_edge_index[1]],
            n2n_s_heading[n2n_s_edge_index[1]],
        )
        n2n_s_edge_attr_length, n2n_s_edge_attr_theta = compute_angles_lengths_2D(n2n_s_edge_vector)
        n2n_s_edge_attr_heading = wrap_angle(
            n2n_s_heading[n2n_s_edge_index[0]] - n2n_s_heading[n2n_s_edge_index[1]]
        )
        n2n_s_edge_attr_input = torch.stack(
            [n2n_s_edge_attr_length, n2n_s_edge_attr_theta, n2n_s_edge_attr_heading], dim=-1
        )
        n2n_s_edge_attr_embs = self.m2m_s_emb_layer(input=n2n_s_edge_attr_input)

        # t2n attention
        t_embs = a_embs.reshape(-1, self.hidden_dim)
        n_embs_t = self.t2n_attn_layer(
            x=[t_embs, n_embs], edge_index=t2n_edge_index, edge_attr=t2n_edge_attr_embs
        )

        # l2n attention
        n_embs_l = self.l2n_attn_layer(
            x=[l_embs, n_embs], edge_index=l2n_edge_index, edge_attr=l2n_edge_attr_embs
        )

        n_embs = n_embs_t + n_embs_l
        n_embs = (
            n_embs.reshape(
                num_all_agent, self.num_historical_steps, self.num_modes, self.hidden_dim
            )
            .transpose(0, 1)
            .reshape(-1, self.hidden_dim)
        )
        # mode attention
        for i in range(self.num_attn_layers):
            n_embs = (
                n_embs.reshape(
                    self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim
                )
                .transpose(1, 2)
                .reshape(-1, self.hidden_dim)
            )
            n_embs = self.n2n_a_attn_layers[i](
                x=n_embs, edge_index=n2n_a_edge_index, edge_attr=n2n_a_edge_attr_embs
            )
            n_embs = (
                n_embs.reshape(
                    self.num_historical_steps, self.num_modes, num_all_agent, self.hidden_dim
                )
                .permute(1, 2, 0, 3)
                .reshape(-1, self.hidden_dim)
            )
            n_embs = self.n2n_h_attn_layers[i](
                x=n_embs, edge_index=n2n_h_edge_index, edge_attr=n2n_h_edge_attr_embs
            )
            n_embs = (
                n_embs.reshape(
                    self.num_modes, num_all_agent, self.num_historical_steps, self.hidden_dim
                )
                .transpose(0, 2)
                .reshape(-1, self.hidden_dim)
            )
            n_embs = self.n2n_s_attn_layers[i](
                x=n_embs, edge_index=n2n_s_edge_index, edge_attr=n2n_s_edge_attr_embs
            )
        n_embs = (
            n_embs.reshape(
                self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim
            )
            .transpose(0, 1)
            .reshape(-1, self.hidden_dim)
        )

        # generate refinement
        traj_refine = self.traj_refine(n_embs).reshape(
            num_all_agent, self.num_historical_steps, self.num_modes, self.num_future_steps, 2
        )
        traj_output = transform_traj_to_global_coordinate(
            proposal + traj_refine, n_position, n_heading
        )

        # generate prob
        prob_output = self.prob_decoder(n_embs.detach()).reshape(
            -1, self.num_historical_steps, self.num_modes
        )
        prob_output = self.prob_norm(prob_output)

        return traj_propose, traj_output, prob_output


# ---------------------------------------------------------------------------
# model/HPNet.py -- HPNet.forward() control flow, as a plain nn.Module (see header note)
# ---------------------------------------------------------------------------
class HPNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        pos_duration: int,
        pred_duration: int,
        a2a_radius: float,
        l2a_radius: float,
        num_visible_steps: int,
        num_modes: int,
        num_attn_layers: int,
        num_hops: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super(HPNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.pos_duration = pos_duration
        self.pred_duration = pred_duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.num_visible_steps = num_visible_steps
        self.num_modes = num_modes
        self.num_attn_layers = num_attn_layers
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout

        self.Backbone = Backbone(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            pos_duration=pos_duration,
            pred_duration=pred_duration,
            a2a_radius=a2a_radius,
            l2a_radius=l2a_radius,
            num_attn_layers=num_attn_layers,
            num_modes=num_modes,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.MapEncoder = MapEncoder(
            hidden_dim=hidden_dim, num_hops=num_hops, num_heads=num_heads, dropout=dropout
        )

    def forward(self, data):
        lane_embs = self.MapEncoder(data=data)
        pred = self.Backbone(data=data, l_embs=lane_embs)
        return pred


# ---------------------------------------------------------------------------
# menagerie staging entry points
# ---------------------------------------------------------------------------
MENAGERIE_ZOO = "vendored-pytorch"


def build_hpnet():
    # Mirrors the repo's train.py / add_model_specific_args defaults, sized down for a
    # fast-tracing smoke config (hidden_dim shrunk from 128 -> 16; layer/hop counts kept
    # small but nonzero to exercise every branch).
    return HPNet(
        hidden_dim=16,
        num_historical_steps=6,
        num_future_steps=8,
        pos_duration=6,
        pred_duration=6,
        a2a_radius=50,
        l2a_radius=50,
        num_visible_steps=2,
        num_modes=3,
        num_attn_layers=1,
        num_hops=2,
        num_heads=2,
        dropout=0.1,
    )


def example_input_hpnet():
    # Small synthetic single-scene HeteroData: 4 agents, 4 lanes each with 2 centerlines, a
    # simple lane graph (adjacent + predecessor + successor edges), all visible. Field layout
    # mirrors ArgoverseV1Dataset.process() (see module header).
    torch.manual_seed(0)
    num_historical_steps = 6
    num_future_steps = 8
    num_steps = num_historical_steps + num_future_steps
    n_agents = 4
    n_lanes = 4
    n_centerlines_per_lane = 2

    data = HeteroData()

    # --- agent ---
    agent_position = torch.randn(n_agents, num_steps, 2)
    agent_heading = torch.rand(n_agents, num_historical_steps) * 2 * math.pi - math.pi
    agent_length = torch.rand(n_agents, num_historical_steps)
    visible_mask = torch.ones(n_agents, num_steps, dtype=torch.bool)

    data["agent"]["num_nodes"] = n_agents
    data["agent"]["agent_index"] = 0
    data["agent"]["visible_mask"] = visible_mask
    data["agent"]["position"] = agent_position
    data["agent"]["heading"] = agent_heading
    data["agent"]["length"] = agent_length
    data["agent"]["batch"] = torch.zeros(n_agents, dtype=torch.long)

    # --- lane ---
    lane_position = torch.randn(n_lanes, 2)
    lane_heading = torch.rand(n_lanes) * 2 * math.pi - math.pi
    lane_length = torch.rand(n_lanes) * 10
    lane_is_intersection = torch.zeros(n_lanes, dtype=torch.uint8)
    lane_turn_direction = torch.zeros(n_lanes, dtype=torch.uint8)
    lane_traffic_control = torch.zeros(n_lanes, dtype=torch.uint8)
    lane_visible_mask = torch.ones(n_lanes, dtype=torch.bool)

    data["lane"]["num_nodes"] = n_lanes
    data["lane"]["position"] = lane_position
    data["lane"]["length"] = lane_length
    data["lane"]["heading"] = lane_heading
    data["lane"]["is_intersection"] = lane_is_intersection
    data["lane"]["turn_direction"] = lane_turn_direction
    data["lane"]["traffic_control"] = lane_traffic_control
    data["lane"]["visible_mask"] = lane_visible_mask
    data["lane"]["batch"] = torch.zeros(n_lanes, dtype=torch.long)

    # --- centerline ---
    num_centerlines_total = n_lanes * n_centerlines_per_lane
    centerline_position = torch.randn(num_centerlines_total, 2)
    centerline_heading = torch.rand(num_centerlines_total) * 2 * math.pi - math.pi
    centerline_length = torch.rand(num_centerlines_total) * 5

    data["centerline"]["num_nodes"] = num_centerlines_total
    data["centerline"]["position"] = centerline_position
    data["centerline"]["heading"] = centerline_heading
    data["centerline"]["length"] = centerline_length

    # centerline -> lane: each lane owns n_centerlines_per_lane consecutive centerlines
    centerline_to_lane_edge_index = torch.stack(
        [
            torch.arange(num_centerlines_total, dtype=torch.long),
            torch.arange(n_lanes, dtype=torch.long).repeat_interleave(n_centerlines_per_lane),
        ],
        dim=0,
    )
    data["centerline", "lane"]["centerline_to_lane_edge_index"] = centerline_to_lane_edge_index

    # lane -> lane: a simple chain (0->1->2->3) doubling as adjacent/predecessor/successor
    chain_src = torch.arange(n_lanes - 1, dtype=torch.long)
    chain_dst = torch.arange(1, n_lanes, dtype=torch.long)
    lane_adjacent_edge_index = torch.stack([chain_src, chain_dst], dim=0)
    lane_predecessor_edge_index = torch.stack([chain_src, chain_dst], dim=0)
    lane_successor_edge_index = torch.stack([chain_dst, chain_src], dim=0)

    data["lane", "lane"]["adjacent_edge_index"] = lane_adjacent_edge_index
    data["lane", "lane"]["predecessor_edge_index"] = lane_predecessor_edge_index
    data["lane", "lane"]["successor_edge_index"] = lane_successor_edge_index

    return (data,)


MENAGERIE_ENTRIES = [
    ("HPNet", build_hpnet, example_input_hpnet, 2024, MENAGERIE_ZOO),
]
