# SOURCE: vendored from https://github.com/rainmaker22/SMART @ main
# (smart/modules/smart_decoder.py, smart/modules/map_decoder.py,
#  smart/modules/agent_decoder.py, smart/layers/attention_layer.py,
#  smart/layers/fourier_embedding.py, smart/layers/mlp_layer.py,
#  smart/utils/geometry.py, smart/utils/weight_init.py)
# SMART: Scalable Multi-agent Real-time Motion Generation via Next-token
# Prediction. NeurIPS 2024. Waymo Open Sim Agents Challenge 2024 champion.
"""SMARTDecoder: map/agent tokenized-trajectory next-token-prediction decoder.

Vendored verbatim from rainmaker22/SMART `smart/modules/{smart_decoder,
map_decoder,agent_decoder}.py` plus their `smart/layers/*` and
`smart/utils/{geometry,weight_init}.py` dependencies. This stages
`SMARTDecoder` (the pure nn.Module architecture) rather than the top-level
`SMART(pl.LightningModule)` in `smart/model/smart.py`, because that top-level
class additionally imports `waymo_open_dataset` (a non-base package, used only
for its Sim Agents submission proto, not architecture) -- `SMARTDecoder` itself
has no such dependency. The real learned trajectory-cluster tokenizer assets
(`smart/tokens/cluster_frame_5_2048.pkl`, `smart/tokens/map_traj_token5.pkl`)
are fetched verbatim from the repo and loaded here exactly as
`SMART.get_trajectory_token` / `SMART.init_map_token` do in
`smart/model/smart.py`, so the token embeddings are the real trained tokenizer
data, not synthetic placeholders. Architecture is unmodified; only this
header/build/example wrapper (including a hand-built minimal `HeteroData`
matching the fields consumed by `SMARTDecoder.forward`, in place of the
repo's Waymo-scenario preprocessing pipeline, which is data loading, not
architecture) were added for menagerie staging.
"""

import math
import os
import pickle
import sys
import urllib.request
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_cluster import radius, radius_graph
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

MENAGERIE_ZOO = "vendored-pytorch"

_TOKEN_URLS = {
    "cluster_frame_5_2048.pkl": (
        "https://raw.githubusercontent.com/rainmaker22/SMART/main/"
        "smart/tokens/cluster_frame_5_2048.pkl"
    ),
    "map_traj_token5.pkl": (
        "https://raw.githubusercontent.com/rainmaker22/SMART/main/smart/tokens/map_traj_token5.pkl"
    ),
}
_TOKEN_CACHE_DIR = os.path.join(os.path.dirname(__file__), "_smart_tokens_cache")


def _fetch_token_file(name: str) -> str:
    os.makedirs(_TOKEN_CACHE_DIR, exist_ok=True)
    dst = os.path.join(_TOKEN_CACHE_DIR, name)
    if not os.path.exists(dst):
        urllib.request.urlretrieve(_TOKEN_URLS[name], dst)  # noqa: S310
    return dst


# ---------------------------------------------------------------------------
# smart/utils/geometry.py
# ---------------------------------------------------------------------------
def angle_between_2d_vectors(ctr_vector: torch.Tensor, nbr_vector: torch.Tensor) -> torch.Tensor:
    return torch.atan2(
        ctr_vector[..., 0] * nbr_vector[..., 1] - ctr_vector[..., 1] * nbr_vector[..., 0],
        (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1),
    )


def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


# ---------------------------------------------------------------------------
# smart/utils/weight_init.py
# ---------------------------------------------------------------------------
def weight_init(m: nn.Module) -> None:
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
    elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
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
    elif isinstance(m, (nn.GRU, nn.GRUCell)):
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
# smart/layers/mlp_layer.py
# ---------------------------------------------------------------------------
class MLPLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(MLPLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ---------------------------------------------------------------------------
# smart/layers/fourier_embedding.py
# ---------------------------------------------------------------------------
class FourierEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_freq_bands: int) -> None:
        super(FourierEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.freqs = nn.Embedding(input_dim, num_freq_bands) if input_dim != 0 else None
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(num_freq_bands * 2 + 1, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(input_dim)
            ]
        )
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(weight_init)

    def forward(
        self,
        continuous_inputs: Optional[torch.Tensor] = None,
        categorical_embs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if continuous_inputs is None:
            if categorical_embs is not None:
                x = torch.stack(categorical_embs).sum(dim=0)
            else:
                raise ValueError("Both continuous_inputs and categorical_embs are None")
        else:
            x = continuous_inputs.unsqueeze(-1) * self.freqs.weight * 2 * math.pi
            x = torch.cat([x.cos(), x.sin(), continuous_inputs.unsqueeze(-1)], dim=-1)
            continuous_embs: List[Optional[torch.Tensor]] = [None] * self.input_dim
            for i in range(self.input_dim):
                continuous_embs[i] = self.mlps[i](x[:, i])
            x = torch.stack(continuous_embs).sum(dim=0)
            if categorical_embs is not None:
                x = x + torch.stack(categorical_embs).sum(dim=0)
        return self.to_out(x)


class MLPEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(MLPEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.apply(weight_init)

    def forward(
        self,
        continuous_inputs: Optional[torch.Tensor] = None,
        categorical_embs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if continuous_inputs is None:
            if categorical_embs is not None:
                x = torch.stack(categorical_embs).sum(dim=0)
            else:
                raise ValueError("Both continuous_inputs and categorical_embs are None")
        else:
            x = self.mlp(continuous_inputs)
            if categorical_embs is not None:
                x = x + torch.stack(categorical_embs).sum(dim=0)
        return x


# ---------------------------------------------------------------------------
# smart/layers/attention_layer.py
# ---------------------------------------------------------------------------
class AttentionLayer(MessagePassing):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        bipartite: bool,
        has_pos_emb: bool,
        **kwargs,
    ) -> None:
        super(AttentionLayer, self).__init__(aggr="add", node_dim=0, **kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.has_pos_emb = has_pos_emb
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_k = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
        self.to_v = nn.Linear(hidden_dim, head_dim * num_heads)
        if has_pos_emb:
            self.to_k_r = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
            self.to_v_r = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_s = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_g = nn.Linear(head_dim * num_heads + hidden_dim, head_dim * num_heads)
        self.to_out = nn.Linear(head_dim * num_heads, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.ff_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        if bipartite:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = nn.LayerNorm(hidden_dim)
        else:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = self.attn_prenorm_x_src
        if has_pos_emb:
            self.attn_prenorm_r = nn.LayerNorm(hidden_dim)
        self.attn_postnorm = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)
        self.ff_postnorm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)

    def forward(self, x, r, edge_index: torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            x_src = x_dst = self.attn_prenorm_x_src(x)
        else:
            x_src, x_dst = x
            x_src = self.attn_prenorm_x_src(x_src)
            x_dst = self.attn_prenorm_x_dst(x_dst)
            x = x[1]
        if self.has_pos_emb and r is not None:
            r = self.attn_prenorm_r(r)
        x = x + self.attn_postnorm(self._attn_block(x_src, x_dst, r, edge_index))
        x = x + self.ff_postnorm(self._ff_block(self.ff_prenorm(x)))
        return x

    def message(
        self, q_i, k_j, v_j, r, index: torch.Tensor, ptr: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.has_pos_emb and r is not None:
            k_j = k_j + self.to_k_r(r).view(-1, self.num_heads, self.head_dim)
            v_j = v_j + self.to_v_r(r).view(-1, self.num_heads, self.head_dim)
        sim = (q_i * k_j).sum(dim=-1) * self.scale
        attn = softmax(sim, index, ptr)
        self.attention_weight = attn.sum(-1).detach()
        attn = self.attn_drop(attn)
        return v_j * attn.unsqueeze(-1)

    def update(self, inputs: torch.Tensor, x_dst: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.num_heads * self.head_dim)
        g = torch.sigmoid(self.to_g(torch.cat([inputs, x_dst], dim=-1)))
        return inputs + g * (self.to_s(x_dst) - inputs)

    def _attn_block(self, x_src, x_dst, r, edge_index: torch.Tensor) -> torch.Tensor:
        q = self.to_q(x_dst).view(-1, self.num_heads, self.head_dim)
        k = self.to_k(x_src).view(-1, self.num_heads, self.head_dim)
        v = self.to_v(x_src).view(-1, self.num_heads, self.head_dim)
        agg = self.propagate(edge_index=edge_index, x_dst=x_dst, q=q, k=k, v=v, r=r)
        return self.to_out(agg)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_mlp(x)


# ---------------------------------------------------------------------------
# smart/modules/map_decoder.py
# ---------------------------------------------------------------------------
class SMARTMapDecoder(nn.Module):
    def __init__(
        self,
        dataset: str,
        input_dim: int,
        hidden_dim: int,
        num_historical_steps: int,
        pl2pl_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        map_token,
    ) -> None:
        super(SMARTMapDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.pl2pl_radius = pl2pl_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        if input_dim == 2:
            input_dim_r_pt2pt = 3
        elif input_dim == 3:
            input_dim_r_pt2pt = 4
        else:
            raise ValueError("{} is not a valid dimension".format(input_dim))

        self.type_pt_emb = nn.Embedding(17, hidden_dim)
        self.side_pt_emb = nn.Embedding(4, hidden_dim)
        self.polygon_type_emb = nn.Embedding(4, hidden_dim)
        self.light_pl_emb = nn.Embedding(4, hidden_dim)

        self.r_pt2pt_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2pt, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.pt2pt_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.token_size = 1024
        self.token_predict_head = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.token_size
        )
        input_dim_token = 22
        self.token_emb = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.map_token = map_token
        self.apply(weight_init)
        self.mask_pt = False

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        pt_valid_mask = data["pt_token"]["pt_valid_mask"]  # noqa: F841 (only used when self.mask_pt=True, matching source)
        pt_pred_mask = data["pt_token"]["pt_pred_mask"]
        pt_target_mask = data["pt_token"]["pt_target_mask"]

        pos_pt = data["pt_token"]["position"][:, : self.input_dim].contiguous()
        orient_pt = data["pt_token"]["orientation"].contiguous()
        orient_vector_pt = torch.stack([orient_pt.cos(), orient_pt.sin()], dim=-1)
        token_sample_pt = self.map_token["traj_src"].to(pos_pt.device).to(torch.float)
        pt_token_emb_src = self.token_emb(token_sample_pt.view(token_sample_pt.shape[0], -1))
        pt_token_emb = pt_token_emb_src[data["pt_token"]["token_idx"]]

        if self.input_dim == 2:
            x_pt = pt_token_emb
        elif self.input_dim == 3:
            x_pt = pt_token_emb
        else:
            raise ValueError("{} is not a valid dimension".format(self.input_dim))

        token2pl = data[("pt_token", "to", "map_polygon")]["edge_index"]
        token_light_type = data["map_polygon"]["light_type"][token2pl[1]]
        x_pt_categorical_embs = [
            self.type_pt_emb(data["pt_token"]["type"].long()),
            self.polygon_type_emb(data["pt_token"]["pl_type"].long()),
            self.light_pl_emb(token_light_type.long()),
        ]
        x_pt = x_pt + torch.stack(x_pt_categorical_embs).sum(dim=0)
        edge_index_pt2pt = radius_graph(
            x=pos_pt[:, :2], r=self.pl2pl_radius, batch=None, loop=False, max_num_neighbors=100
        )
        rel_pos_pt2pt = pos_pt[edge_index_pt2pt[0]] - pos_pt[edge_index_pt2pt[1]]
        rel_orient_pt2pt = wrap_angle(
            orient_pt[edge_index_pt2pt[0]] - orient_pt[edge_index_pt2pt[1]]
        )
        if self.input_dim == 2:
            r_pt2pt = torch.stack(
                [
                    torch.norm(rel_pos_pt2pt[:, :2], p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=orient_vector_pt[edge_index_pt2pt[1]],
                        nbr_vector=rel_pos_pt2pt[:, :2],
                    ),
                    rel_orient_pt2pt,
                ],
                dim=-1,
            )
        else:
            raise ValueError("{} is not a valid dimension".format(self.input_dim))
        r_pt2pt = self.r_pt2pt_emb(continuous_inputs=r_pt2pt, categorical_embs=None)
        for i in range(self.num_layers):
            x_pt = self.pt2pt_layers[i](x_pt, r_pt2pt, edge_index_pt2pt)

        next_token_prob = self.token_predict_head(x_pt[pt_pred_mask])
        next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)
        _, next_token_idx = torch.topk(next_token_prob_softmax, k=min(10, self.token_size), dim=-1)
        next_token_index_gt = data["pt_token"]["token_idx"][pt_target_mask]

        return {
            "x_pt": x_pt,
            "map_next_token_idx": next_token_idx,
            "map_next_token_prob": next_token_prob,
            "map_next_token_idx_gt": next_token_index_gt,
            "map_next_token_eval_mask": pt_pred_mask[pt_pred_mask],
        }


# ---------------------------------------------------------------------------
# smart/modules/agent_decoder.py
# ---------------------------------------------------------------------------
class SMARTAgentDecoder(nn.Module):
    def __init__(
        self,
        dataset: str,
        input_dim: int,
        hidden_dim: int,
        num_historical_steps: int,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        token_data: Dict,
        token_size=512,
    ) -> None:
        super(SMARTAgentDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_x_a = 2
        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3
        input_dim_token = 8

        self.type_a_emb = nn.Embedding(4, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

        self.x_a_emb = FourierEmbedding(
            input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.r_t_emb = FourierEmbedding(
            input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.r_pt2a_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.r_a2a_emb = FourierEmbedding(
            input_dim=input_dim_r_a2a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.token_emb_veh = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.token_emb_ped = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.token_emb_cyc = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.fusion_emb = MLPEmbedding(input_dim=self.hidden_dim * 2, hidden_dim=self.hidden_dim)

        self.t_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.pt2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.token_size = token_size
        self.token_predict_head = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=self.token_size
        )
        self.trajectory_token = token_data["token"]
        self.trajectory_token_traj = token_data["traj"]
        self.trajectory_token_all = token_data["token_all"]
        self.apply(weight_init)
        self.shift = 5
        self.beam_size = 5
        self.hist_mask = True

    def agent_token_embedding(
        self, data, agent_category, agent_token_index, pos_a, head_vector_a, inference=False
    ):
        num_agent, num_step, traj_dim = pos_a.shape
        motion_vector_a = torch.cat(
            [
                pos_a.new_zeros(data["agent"]["num_nodes"], 1, self.input_dim),
                pos_a[:, 1:] - pos_a[:, :-1],
            ],
            dim=1,
        )

        agent_type = data["agent"]["type"]
        veh_mask = agent_type == 0
        cyc_mask = agent_type == 2
        ped_mask = agent_type == 1
        trajectory_token_veh = (
            torch.from_numpy(self.trajectory_token["veh"]).clone().to(pos_a.device).to(torch.float)
        )
        self.agent_token_emb_veh = self.token_emb_veh(
            trajectory_token_veh.view(trajectory_token_veh.shape[0], -1)
        )
        trajectory_token_ped = (
            torch.from_numpy(self.trajectory_token["ped"]).clone().to(pos_a.device).to(torch.float)
        )
        self.agent_token_emb_ped = self.token_emb_ped(
            trajectory_token_ped.view(trajectory_token_ped.shape[0], -1)
        )
        trajectory_token_cyc = (
            torch.from_numpy(self.trajectory_token["cyc"]).clone().to(pos_a.device).to(torch.float)
        )
        self.agent_token_emb_cyc = self.token_emb_cyc(
            trajectory_token_cyc.view(trajectory_token_cyc.shape[0], -1)
        )

        agent_token_emb = torch.zeros((num_agent, num_step, self.hidden_dim), device=pos_a.device)
        agent_token_emb[veh_mask] = self.agent_token_emb_veh[agent_token_index[veh_mask]]
        agent_token_emb[ped_mask] = self.agent_token_emb_ped[agent_token_index[ped_mask]]
        agent_token_emb[cyc_mask] = self.agent_token_emb_cyc[agent_token_index[cyc_mask]]

        vel = data["agent"]["token_velocity"]  # noqa: F841 (unused in real code too, kept for fidelity)

        categorical_embs = [
            self.type_a_emb(data["agent"]["type"].long()).repeat_interleave(
                repeats=num_step, dim=0
            ),
            self.shape_emb(
                data["agent"]["shape"][:, self.num_historical_steps - 1, :]
            ).repeat_interleave(repeats=num_step, dim=0),
        ]
        feature_a = torch.stack(
            [
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
                ),
            ],
            dim=-1,
        )

        x_a = self.x_a_emb(
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
            categorical_embs=categorical_embs,
        )
        x_a = x_a.view(-1, num_step, self.hidden_dim)

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)

        return feat_a, None

    def build_temporal_edge(
        self, pos_a, head_a, head_vector_a, num_agent, mask, inference_mask=None
    ):
        from torch_geometric.utils import dense_to_sparse

        pos_t = pos_a.reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)
        hist_mask = mask.clone()

        if self.hist_mask and self.training:
            hist_mask[
                torch.arange(mask.shape[0]).unsqueeze(1),
                torch.randint(0, mask.shape[1], (num_agent, min(10, mask.shape[1]))),
            ] = False
            mask_t = hist_mask.unsqueeze(2) & hist_mask.unsqueeze(1)
        elif inference_mask is not None:
            mask_t = hist_mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            mask_t = hist_mask.unsqueeze(2) & hist_mask.unsqueeze(1)

        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[
            :, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift
        ]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [
                torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]
                ),
                rel_head_t,
                edge_index_t[0] - edge_index_t[1],
            ],
            dim=-1,
        )
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t, r_t

    def build_interaction_edge(self, pos_a, head_a, head_vector_a, batch_s, mask_s):
        from torch_geometric.utils import subgraph

        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        edge_index_a2a = radius_graph(
            x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False, max_num_neighbors=300
        )
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [
                torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]
                ),
                rel_head_a2a,
            ],
            dim=-1,
        )
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        return edge_index_a2a, r_a2a

    def build_map2agent_edge(
        self, data, num_step, agent_category, pos_a, head_a, head_vector_a, mask, batch_s, batch_pl
    ):
        mask_pl2a = mask.clone()
        mask_pl2a = mask_pl2a.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        pos_pl = data["pt_token"]["position"][:, : self.input_dim].contiguous()
        orient_pl = data["pt_token"]["orientation"].contiguous()
        pos_pl = pos_pl.repeat(num_step, 1)
        orient_pl = orient_pl.repeat(num_step)
        edge_index_pl2a = radius(
            x=pos_s[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2a_radius,
            batch_x=batch_s,
            batch_y=batch_pl,
            max_num_neighbors=300,
        )
        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
        r_pl2a = torch.stack(
            [
                torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]
                ),
                rel_orient_pl2a,
            ],
            dim=-1,
        )
        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        return edge_index_pl2a, r_pl2a

    def forward(
        self, data: HeteroData, map_enc: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pos_a = data["agent"]["token_pos"]
        head_a = data["agent"]["token_heading"]
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        num_agent, num_step, traj_dim = pos_a.shape
        agent_category = data["agent"]["category"]
        agent_token_index = data["agent"]["token_idx"]
        feat_a, _ = self.agent_token_embedding(
            data, agent_category, agent_token_index, pos_a, head_vector_a
        )

        agent_valid_mask = data["agent"]["agent_valid_mask"].clone()
        mask = agent_valid_mask
        edge_index_t, r_t = self.build_temporal_edge(pos_a, head_a, head_vector_a, num_agent, mask)

        batch_s = torch.arange(num_step, device=pos_a.device).repeat_interleave(
            data["agent"]["num_nodes"]
        )
        batch_pl = torch.arange(num_step, device=pos_a.device).repeat_interleave(
            data["pt_token"]["num_nodes"]
        )

        mask_s = mask.transpose(0, 1).reshape(-1)
        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a, head_a, head_vector_a, batch_s, mask_s
        )
        mask[agent_category != 3] = False
        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            data, num_step, agent_category, pos_a, head_a, head_vector_a, mask, batch_s, batch_pl
        )

        for i in range(self.num_layers):
            feat_a = feat_a.reshape(-1, self.hidden_dim)
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
            feat_a = (
                feat_a.reshape(-1, num_step, self.hidden_dim)
                .transpose(0, 1)
                .reshape(-1, self.hidden_dim)
            )
            feat_a = self.pt2a_attn_layers[i](
                (
                    map_enc["x_pt"]
                    .repeat_interleave(repeats=num_step, dim=0)
                    .reshape(-1, num_step, self.hidden_dim)
                    .transpose(0, 1)
                    .reshape(-1, self.hidden_dim),
                    feat_a,
                ),
                r_pl2a,
                edge_index_pl2a,
            )
            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.reshape(num_step, -1, self.hidden_dim).transpose(0, 1)

        next_token_prob = self.token_predict_head(feat_a)
        next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)
        _, next_token_idx = torch.topk(next_token_prob_softmax, k=min(10, self.token_size), dim=-1)

        next_token_index_gt = agent_token_index.roll(shifts=-1, dims=1)
        next_token_eval_mask = mask.clone()
        next_token_eval_mask = (
            next_token_eval_mask
            * next_token_eval_mask.roll(shifts=-1, dims=1)
            * next_token_eval_mask.roll(shifts=1, dims=1)
        )
        next_token_eval_mask[:, -1] = False

        return {
            "x_a": feat_a,
            "next_token_idx": next_token_idx,
            "next_token_prob": next_token_prob,
            "next_token_idx_gt": next_token_index_gt,
            "next_token_eval_mask": next_token_eval_mask,
        }


# ---------------------------------------------------------------------------
# smart/modules/smart_decoder.py
# ---------------------------------------------------------------------------
class SMARTDecoder(nn.Module):
    def __init__(
        self,
        dataset: str,
        input_dim: int,
        hidden_dim: int,
        num_historical_steps: int,
        pl2pl_radius: float,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_map_layers: int,
        num_agent_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        map_token: Dict,
        token_data: Dict,
        use_intention=False,
        token_size=512,
    ) -> None:
        super(SMARTDecoder, self).__init__()
        self.map_encoder = SMARTMapDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            map_token=map_token,
        )
        self.agent_encoder = SMARTAgentDecoder(
            dataset=dataset,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            token_size=token_size,
            token_data=token_data,
        )
        self.map_enc = None

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)
        agent_enc = self.agent_encoder(data, map_enc)
        return {**map_enc, **agent_enc}


# ---------------------------------------------------------------------------
# menagerie staging wrapper
# ---------------------------------------------------------------------------
# configs/train/train_scalable.yaml: Model + Model.decoder, verbatim values, with
# layer counts shrunk (num_map_layers 3->1, num_agent_layers 6->1) and hidden_dim
# shrunk (128->32) purely for fast tracing; every mechanism (Fourier/MLP token
# embeddings, radius-graph message passing, temporal/map/agent attention) is the
# real architecture, unmodified.
_MODEL_CFG = dict(
    dataset="waymo",
    input_dim=2,
    hidden_dim=32,
    output_dim=2,
    num_heads=4,
    head_dim=8,
    dropout=0.0,
    num_freq_bands=8,
    num_historical_steps=11,
    decoder=dict(
        num_map_layers=1,
        num_agent_layers=1,
        a2a_radius=60.0,
        pl2pl_radius=10.0,
        pl2a_radius=30.0,
        time_span=30,
        token_size=2048,
    ),
)


def _load_tokens():
    cluster_path = _fetch_token_file("cluster_frame_5_2048.pkl")
    map_path = _fetch_token_file("map_traj_token5.pkl")
    with open(cluster_path, "rb") as f:
        token_data = pickle.load(f)  # noqa: S301 (repo's own trusted asset)
    with open(map_path, "rb") as f:
        map_token_traj = pickle.load(f)  # noqa: S301
    map_token = {"traj_src": torch.from_numpy(map_token_traj["traj_src"]).to(torch.float)}
    return token_data, map_token


def build_smart():
    token_data, map_token = _load_tokens()
    dec = _MODEL_CFG["decoder"]
    return SMARTDecoder(
        dataset=_MODEL_CFG["dataset"],
        input_dim=_MODEL_CFG["input_dim"],
        hidden_dim=_MODEL_CFG["hidden_dim"],
        num_historical_steps=_MODEL_CFG["num_historical_steps"],
        pl2pl_radius=dec["pl2pl_radius"],
        time_span=dec["time_span"],
        pl2a_radius=dec["pl2a_radius"],
        a2a_radius=dec["a2a_radius"],
        num_freq_bands=_MODEL_CFG["num_freq_bands"],
        num_map_layers=dec["num_map_layers"],
        num_agent_layers=dec["num_agent_layers"],
        num_heads=_MODEL_CFG["num_heads"],
        head_dim=_MODEL_CFG["head_dim"],
        dropout=_MODEL_CFG["dropout"],
        map_token={"traj_src": map_token["traj_src"]},
        token_data=token_data,
        token_size=dec["token_size"],
    )


def example_input_smart():
    """Small hand-built HeteroData with the fields SMARTDecoder.forward consumes
    (map_decoder.forward + agent_decoder.forward, forward()-not-inference() path),
    matching smart/datasets/preprocess.py's `pt_token`/`map_polygon`/`agent` schema
    at toy scale. Positions are spread over a small area so pl2pl/pl2a/a2a radius
    graphs (10/30/60) are non-trivially connected.
    """
    torch.manual_seed(0)
    n_pt = 6  # map polyline points
    n_polygon = 2  # map polygons the points belong to
    n_agent = 3
    num_step = (
        _MODEL_CFG["num_historical_steps"] // 5 + 1
    )  # SMART operates on 5-step "shift" tokens

    data = HeteroData()

    # --- pt_token (map points) ---
    data["pt_token"].num_nodes = n_pt
    data["pt_token"].position = torch.cat(
        [torch.randn(n_pt, 2) * 5.0, torch.zeros(n_pt, 1)], dim=-1
    )
    data["pt_token"].orientation = torch.rand(n_pt) * 2 * math.pi - math.pi
    data["pt_token"].token_idx = torch.randint(0, 1024, (n_pt,))
    data["pt_token"].type = torch.randint(0, 17, (n_pt,))
    data["pt_token"].pl_type = torch.randint(0, 4, (n_pt,))
    valid = torch.ones(n_pt, dtype=torch.bool)
    data["pt_token"].pt_valid_mask = valid
    pred_mask = torch.zeros(n_pt, dtype=torch.bool)
    pred_mask[: max(1, n_pt // 2)] = True
    data["pt_token"].pt_pred_mask = pred_mask
    data["pt_token"].pt_target_mask = pred_mask.roll(1)

    # --- map_polygon ---
    data["map_polygon"].num_nodes = n_polygon
    data["map_polygon"].light_type = torch.randint(0, 4, (n_polygon,))

    # --- pt_token -> map_polygon edges ---
    pl_idx_list = torch.randint(0, n_polygon, (n_pt,))
    token2pl = torch.stack([torch.arange(n_pt), pl_idx_list], dim=0)
    data["pt_token", "to", "map_polygon"].edge_index = token2pl

    # --- agent ---
    data["agent"].num_nodes = n_agent
    data["agent"].token_pos = torch.randn(n_agent, num_step, 2) * 5.0
    data["agent"].token_heading = torch.rand(n_agent, num_step) * 2 * math.pi - math.pi
    data["agent"].category = torch.full((n_agent,), 3, dtype=torch.long)
    data["agent"].token_idx = torch.randint(
        0, _MODEL_CFG["decoder"]["token_size"], (n_agent, num_step)
    )
    data["agent"].agent_valid_mask = torch.ones(n_agent, num_step, dtype=torch.bool)
    data["agent"].type = torch.randint(0, 3, (n_agent,))  # 0=veh, 1=ped, 2=cyc (avoid unused idx 3)
    # shape is indexed at the raw historical-step index (num_historical_steps - 1),
    # NOT the shift-token step index used for token_pos/token_heading -- needs its
    # own num_historical_steps-length time axis (agent_decoder.py:agent_token_embedding).
    data["agent"].shape = torch.rand(n_agent, _MODEL_CFG["num_historical_steps"], 3)
    data["agent"].token_velocity = torch.randn(n_agent, num_step, 2)

    return (data,)


MENAGERIE_ENTRIES = [
    ("SMART", "build_smart", "example_input_smart", 2024, "vendored-pytorch"),
]

# torch_geometric.nn.conv.MessagePassing (AttentionLayer's base class, above)
# resolves message()'s type-hint globals via `sys.modules[cls.__module__].__dict__`
# at subclass-__init__ time (torch_geometric.inspector.Inspector._globals). When
# this file is loaded via importlib.util.spec_from_file_location (as the menagerie
# staging/integrate pipeline's `_integrate_staged_modules` does), the module
# executes under its own __name__ but is never registered in sys.modules, so that
# lookup raises KeyError the first time an AttentionLayer is constructed (inside
# build_smart(), at trace time, i.e. after this module has finished executing).
# Building a fresh module object from our own __spec__ and populating it with a
# (by-now-complete) snapshot of this module's globals fixes that without needing
# the importer's cooperation. A plain `import` (where sys.modules[__name__] is
# already populated before this line runs) leaves this branch a no-op.
if __name__ not in sys.modules and __spec__ is not None:
    import importlib.util as _importlib_util

    _sys_modules_proxy = _importlib_util.module_from_spec(__spec__)
    _sys_modules_proxy.__dict__.update(globals())
    sys.modules[__name__] = _sys_modules_proxy
