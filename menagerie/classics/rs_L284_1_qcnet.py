# SOURCE: vendored from https://github.com/ZikangZhou/QCNet @ main
# (layers/{attention_layer,fourier_embedding,mlp_layer}.py,
#  modules/{qcnet_encoder,qcnet_map_encoder,qcnet_agent_encoder,qcnet_decoder}.py,
#  utils/{geometry,graph,weight_init,list}.py)
"""QCNet (CVPR 2023) -- Query-Centric Trajectory Prediction.

QCNet predicts multimodal future trajectories for traffic agents from an HD map +
observed-agent-history heterogeneous graph (Argoverse 2 motion forecasting schema). The
real model is `predictors/qcnet.py:QCNet(pl.LightningModule)`, which is a thin
`pytorch_lightning` training wrapper around two pure-torch nn.Modules:
`modules/qcnet_encoder.py:QCNetEncoder` (map encoder + agent encoder, message-passing
attention over PyG heterogeneous graphs) and `modules/qcnet_decoder.py:QCNetDecoder`
(recurrent anchor-free multimodal trajectory decoder with propose/refine stages).

This vendors `QCNetEncoder` + `QCNetDecoder` and their real submodules
(`AttentionLayer` built on `torch_geometric.nn.conv.MessagePassing`, `FourierEmbedding`,
`MLPLayer`) and utility functions (`angle_between_2d_vectors`, `wrap_angle`, `merge_edges`,
`bipartite_dense_to_sparse`, `weight_init`) VERBATIM (architecture untouched; only the
`from layers import ...` / `from utils import ...` package-relative imports were flattened
into this single file, and the `pl.LightningModule` training wrapper -- optimizer/loss/
metric bookkeeping only, not part of the forward architecture -- was dropped in favor of a
plain nn.Module `QCNetNet` that calls the same `encoder(data); decoder(data, scene_enc)`
forward path as the real `QCNet.forward`). `torch_geometric` + `torch_cluster` are real,
installed base libs the real code already depends on (`radius`, `radius_graph`,
`MessagePassing`, `dense_to_sparse`, `softmax`, `coalesce`, `degree`).

The model consumes a `torch_geometric.data.HeteroData` scene graph with node types
`agent`/`map_point`/`map_polygon` and edge types `('map_point','to','map_polygon')` /
`('map_polygon','to','map_polygon')`, matching the Argoverse 2 dataset schema the repo's
own `datasets/argoverse_v2_dataset.py` builds. `example_input_qcnet()` constructs a tiny
synthetic HeteroData with the same fields/dtypes/shapes real preprocessed AV2 scenes carry
(`dataset='argoverse_v2'`, `input_dim=2`), so this is a MODULE (multi-tensor input via a
graph object), not a single-tensor recipe.
"""

import math
import sys
from typing import Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius, radius_graph
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import coalesce, degree, dense_to_sparse, softmax

# torch_geometric's MessagePassing.inspector resolves `message()`'s type-hinted params
# (e.g. `Optional[torch.Tensor]`) via `sys.modules[self.__module__].__dict__`. If this file
# is loaded via importlib.util.module_from_spec() without the resulting module object also
# being registered in sys.modules under its own name, that lookup KeyErrors at
# AttentionLayer construction time. Registering the currently-executing module object here
# is load-time scaffolding only (mirrors what a normal `import` statement always does) --
# it does not touch the vendored AttentionLayer architecture below.
if __name__ not in sys.modules:
    import types as _types

    sys.modules[__name__] = _types.ModuleType(__name__)

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# utils/geometry.py, utils/graph.py, utils/weight_init.py, utils/list.py
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


def merge_edges(
    edge_indices: List[torch.Tensor],
    edge_attrs: Optional[List[torch.Tensor]] = None,
    reduce: str = "add",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    edge_index = torch.cat(edge_indices, dim=1)
    if edge_attrs is not None:
        edge_attr = torch.cat(edge_attrs, dim=0)
    else:
        edge_attr = None
    return coalesce(edge_index=edge_index, edge_attr=edge_attr, reduce=reduce)


def bipartite_dense_to_sparse(adj: torch.Tensor) -> torch.Tensor:
    index = adj.nonzero(as_tuple=True)
    if len(index) == 3:
        batch_src = index[0] * adj.size(1)
        batch_dst = index[0] * adj.size(2)
        index = (batch_src + index[1], batch_dst + index[2])
    return torch.stack(index, dim=0)


def unbatch(src: torch.Tensor, batch: torch.Tensor, dim: int = 0) -> List[torch.Tensor]:
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


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
# layers/fourier_embedding.py, layers/mlp_layer.py, layers/attention_layer.py
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

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        r: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
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
        self,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        v_j: torch.Tensor,
        r: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.has_pos_emb and r is not None:
            k_j = k_j + self.to_k_r(r).view(-1, self.num_heads, self.head_dim)
            v_j = v_j + self.to_v_r(r).view(-1, self.num_heads, self.head_dim)
        sim = (q_i * k_j).sum(dim=-1) * self.scale
        attn = softmax(sim, index, ptr)
        attn = self.attn_drop(attn)
        return v_j * attn.unsqueeze(-1)

    def update(self, inputs: torch.Tensor, x_dst: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.num_heads * self.head_dim)
        g = torch.sigmoid(self.to_g(torch.cat([inputs, x_dst], dim=-1)))
        return inputs + g * (self.to_s(x_dst) - inputs)

    def _attn_block(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        r: Optional[torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        q = self.to_q(x_dst).view(-1, self.num_heads, self.head_dim)
        k = self.to_k(x_src).view(-1, self.num_heads, self.head_dim)
        v = self.to_v(x_src).view(-1, self.num_heads, self.head_dim)
        agg = self.propagate(edge_index=edge_index, x_dst=x_dst, q=q, k=k, v=v, r=r)
        return self.to_out(agg)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_mlp(x)


# ---------------------------------------------------------------------------
# modules/qcnet_map_encoder.py, modules/qcnet_agent_encoder.py,
# modules/qcnet_encoder.py, modules/qcnet_decoder.py
# ---------------------------------------------------------------------------


class QCNetMapEncoder(nn.Module):
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
    ) -> None:
        super(QCNetMapEncoder, self).__init__()
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

        if dataset == "argoverse_v2":
            if input_dim == 2:
                input_dim_x_pt = 1
                input_dim_x_pl = 0
                input_dim_r_pt2pl = 3
                input_dim_r_pl2pl = 3
            elif input_dim == 3:
                input_dim_x_pt = 2
                input_dim_x_pl = 1
                input_dim_r_pt2pl = 4
                input_dim_r_pl2pl = 4
            else:
                raise ValueError("{} is not a valid dimension".format(input_dim))
        else:
            raise ValueError("{} is not a valid dataset".format(dataset))

        self.type_pt_emb = nn.Embedding(17, hidden_dim)
        self.side_pt_emb = nn.Embedding(3, hidden_dim)
        self.type_pl_emb = nn.Embedding(4, hidden_dim)
        self.int_pl_emb = nn.Embedding(3, hidden_dim)
        self.type_pl2pl_emb = nn.Embedding(5, hidden_dim)
        self.x_pt_emb = FourierEmbedding(
            input_dim=input_dim_x_pt, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.x_pl_emb = FourierEmbedding(
            input_dim=input_dim_x_pl, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.r_pt2pl_emb = FourierEmbedding(
            input_dim=input_dim_r_pt2pl, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.r_pl2pl_emb = FourierEmbedding(
            input_dim=input_dim_r_pl2pl, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.pt2pl_layers = nn.ModuleList(
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
        self.pl2pl_layers = nn.ModuleList(
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
        self.apply(weight_init)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        pos_pt = data["map_point"]["position"][:, : self.input_dim].contiguous()
        orient_pt = data["map_point"]["orientation"].contiguous()
        pos_pl = data["map_polygon"]["position"][:, : self.input_dim].contiguous()
        orient_pl = data["map_polygon"]["orientation"].contiguous()
        orient_vector_pl = torch.stack([orient_pl.cos(), orient_pl.sin()], dim=-1)

        if self.input_dim == 2:
            x_pt = data["map_point"]["magnitude"].unsqueeze(-1)
            x_pl = None
        else:
            x_pt = torch.stack(
                [data["map_point"]["magnitude"], data["map_point"]["height"]], dim=-1
            )
            x_pl = data["map_polygon"]["height"].unsqueeze(-1)
        x_pt_categorical_embs = [
            self.type_pt_emb(data["map_point"]["type"].long()),
            self.side_pt_emb(data["map_point"]["side"].long()),
        ]
        x_pl_categorical_embs = [
            self.type_pl_emb(data["map_polygon"]["type"].long()),
            self.int_pl_emb(data["map_polygon"]["is_intersection"].long()),
        ]
        x_pt = self.x_pt_emb(continuous_inputs=x_pt, categorical_embs=x_pt_categorical_embs)
        x_pl = self.x_pl_emb(continuous_inputs=x_pl, categorical_embs=x_pl_categorical_embs)

        edge_index_pt2pl = data["map_point", "to", "map_polygon"]["edge_index"]
        rel_pos_pt2pl = pos_pt[edge_index_pt2pl[0]] - pos_pl[edge_index_pt2pl[1]]
        rel_orient_pt2pl = wrap_angle(
            orient_pt[edge_index_pt2pl[0]] - orient_pl[edge_index_pt2pl[1]]
        )
        if self.input_dim == 2:
            r_pt2pl = torch.stack(
                [
                    torch.norm(rel_pos_pt2pl[:, :2], p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=orient_vector_pl[edge_index_pt2pl[1]],
                        nbr_vector=rel_pos_pt2pl[:, :2],
                    ),
                    rel_orient_pt2pl,
                ],
                dim=-1,
            )
        else:
            r_pt2pl = torch.stack(
                [
                    torch.norm(rel_pos_pt2pl[:, :2], p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=orient_vector_pl[edge_index_pt2pl[1]],
                        nbr_vector=rel_pos_pt2pl[:, :2],
                    ),
                    rel_pos_pt2pl[:, -1],
                    rel_orient_pt2pl,
                ],
                dim=-1,
            )
        r_pt2pl = self.r_pt2pl_emb(continuous_inputs=r_pt2pl, categorical_embs=None)

        edge_index_pl2pl = data["map_polygon", "to", "map_polygon"]["edge_index"]
        edge_index_pl2pl_radius = radius_graph(
            x=pos_pl[:, :2],
            r=self.pl2pl_radius,
            batch=data["map_polygon"]["batch"] if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=300,
        )
        type_pl2pl = data["map_polygon", "to", "map_polygon"]["type"]
        type_pl2pl_radius = type_pl2pl.new_zeros(edge_index_pl2pl_radius.size(1), dtype=torch.uint8)
        edge_index_pl2pl, type_pl2pl = merge_edges(
            edge_indices=[edge_index_pl2pl_radius, edge_index_pl2pl],
            edge_attrs=[type_pl2pl_radius, type_pl2pl],
            reduce="max",
        )
        rel_pos_pl2pl = pos_pl[edge_index_pl2pl[0]] - pos_pl[edge_index_pl2pl[1]]
        rel_orient_pl2pl = wrap_angle(
            orient_pl[edge_index_pl2pl[0]] - orient_pl[edge_index_pl2pl[1]]
        )
        if self.input_dim == 2:
            r_pl2pl = torch.stack(
                [
                    torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=orient_vector_pl[edge_index_pl2pl[1]],
                        nbr_vector=rel_pos_pl2pl[:, :2],
                    ),
                    rel_orient_pl2pl,
                ],
                dim=-1,
            )
        else:
            r_pl2pl = torch.stack(
                [
                    torch.norm(rel_pos_pl2pl[:, :2], p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=orient_vector_pl[edge_index_pl2pl[1]],
                        nbr_vector=rel_pos_pl2pl[:, :2],
                    ),
                    rel_pos_pl2pl[:, -1],
                    rel_orient_pl2pl,
                ],
                dim=-1,
            )
        r_pl2pl = self.r_pl2pl_emb(
            continuous_inputs=r_pl2pl, categorical_embs=[self.type_pl2pl_emb(type_pl2pl.long())]
        )

        for i in range(self.num_layers):
            x_pl = self.pt2pl_layers[i]((x_pt, x_pl), r_pt2pl, edge_index_pt2pl)
            x_pl = self.pl2pl_layers[i](x_pl, r_pl2pl, edge_index_pl2pl)
        x_pl = x_pl.repeat_interleave(repeats=self.num_historical_steps, dim=0).reshape(
            -1, self.num_historical_steps, self.hidden_dim
        )

        return {"x_pt": x_pt, "x_pl": x_pl}


class QCNetAgentEncoder(nn.Module):
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
    ) -> None:
        super(QCNetAgentEncoder, self).__init__()
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

        input_dim_x_a = 4
        input_dim_r_t = 4
        input_dim_r_pl2a = 3
        input_dim_r_a2a = 3

        self.type_a_emb = nn.Embedding(10, hidden_dim)
        self.x_a_emb = FourierEmbedding(
            input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.r_t_emb = FourierEmbedding(
            input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.r_pl2a_emb = FourierEmbedding(
            input_dim=input_dim_r_pl2a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.r_a2a_emb = FourierEmbedding(
            input_dim=input_dim_r_a2a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
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
        self.pl2a_attn_layers = nn.ModuleList(
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
        self.apply(weight_init)

    def forward(
        self, data: HeteroData, map_enc: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        mask = data["agent"]["valid_mask"][:, : self.num_historical_steps].contiguous()
        pos_a = data["agent"]["position"][
            :, : self.num_historical_steps, : self.input_dim
        ].contiguous()
        motion_vector_a = torch.cat(
            [
                pos_a.new_zeros(data["agent"]["num_nodes"], 1, self.input_dim),
                pos_a[:, 1:] - pos_a[:, :-1],
            ],
            dim=1,
        )
        head_a = data["agent"]["heading"][:, : self.num_historical_steps].contiguous()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pos_pl = data["map_polygon"]["position"][:, : self.input_dim].contiguous()

        vel = data["agent"]["velocity"][
            :, : self.num_historical_steps, : self.input_dim
        ].contiguous()
        categorical_embs = [
            self.type_a_emb(data["agent"]["type"].long()).repeat_interleave(
                repeats=self.num_historical_steps, dim=0
            ),
        ]

        x_a = torch.stack(
            [
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
                ),
                torch.norm(vel[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=vel[:, :, :2]),
            ],
            dim=-1,
        )
        x_a = self.x_a_emb(
            continuous_inputs=x_a.view(-1, x_a.size(-1)), categorical_embs=categorical_embs
        )
        x_a = x_a.view(-1, self.num_historical_steps, self.hidden_dim)

        pos_t = pos_a.reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)
        mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span]
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

        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        mask_s = mask.transpose(0, 1).reshape(-1)
        pos_pl = pos_pl.repeat(self.num_historical_steps, 1)
        orient_pl = (
            data["map_polygon"]["orientation"].contiguous().repeat(self.num_historical_steps)
        )
        if isinstance(data, Batch):
            batch_s = torch.cat(
                [
                    data["agent"]["batch"] + data.num_graphs * t
                    for t in range(self.num_historical_steps)
                ],
                dim=0,
            )
            batch_pl = torch.cat(
                [
                    data["map_polygon"]["batch"] + data.num_graphs * t
                    for t in range(self.num_historical_steps)
                ],
                dim=0,
            )
        else:
            batch_s = torch.arange(
                self.num_historical_steps, device=pos_a.device
            ).repeat_interleave(data["agent"]["num_nodes"])
            batch_pl = torch.arange(
                self.num_historical_steps, device=pos_pl.device
            ).repeat_interleave(data["map_polygon"]["num_nodes"])
        edge_index_pl2a = radius(
            x=pos_s[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2a_radius,
            batch_x=batch_s,
            batch_y=batch_pl,
            max_num_neighbors=300,
        )
        edge_index_pl2a = edge_index_pl2a[:, mask_s[edge_index_pl2a[1]]]
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
        r_pl2a = self.r_pl2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        edge_index_a2a = radius_graph(
            x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False, max_num_neighbors=300
        )
        # NOTE: real repo calls torch_geometric.utils.subgraph(subset=mask_s, edge_index=edge_index_a2a)[0].
        # Reproduced inline (mask both endpoints) to avoid importing an extra symbol.
        keep = mask_s[edge_index_a2a[0]] & mask_s[edge_index_a2a[1]]
        edge_index_a2a = edge_index_a2a[:, keep]
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

        for i in range(self.num_layers):
            x_a = x_a.reshape(-1, self.hidden_dim)
            x_a = self.t_attn_layers[i](x_a, r_t, edge_index_t)
            x_a = (
                x_a.reshape(-1, self.num_historical_steps, self.hidden_dim)
                .transpose(0, 1)
                .reshape(-1, self.hidden_dim)
            )
            x_a = self.pl2a_attn_layers[i](
                (map_enc["x_pl"].transpose(0, 1).reshape(-1, self.hidden_dim), x_a),
                r_pl2a,
                edge_index_pl2a,
            )
            x_a = self.a2a_attn_layers[i](x_a, r_a2a, edge_index_a2a)
            x_a = x_a.reshape(self.num_historical_steps, -1, self.hidden_dim).transpose(0, 1)

        return {"x_a": x_a}


class QCNetEncoder(nn.Module):
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
    ) -> None:
        super(QCNetEncoder, self).__init__()
        self.map_encoder = QCNetMapEncoder(
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
        )
        self.agent_encoder = QCNetAgentEncoder(
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
        )

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        map_enc = self.map_encoder(data)
        agent_enc = self.agent_encoder(data, map_enc)
        return {**map_enc, **agent_enc}


class QCNetDecoder(nn.Module):
    def __init__(
        self,
        dataset: str,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        output_head: bool,
        num_historical_steps: int,
        num_future_steps: int,
        num_modes: int,
        num_recurrent_steps: int,
        num_t2m_steps: Optional[int],
        pl2m_radius: float,
        a2m_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
    ) -> None:
        super(QCNetDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_t2m_steps = num_t2m_steps if num_t2m_steps is not None else num_historical_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_r_t = 4
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3

        self.mode_emb = nn.Embedding(num_modes, hidden_dim)
        self.r_t2m_emb = FourierEmbedding(
            input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.r_pl2m_emb = FourierEmbedding(
            input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.r_a2m_emb = FourierEmbedding(
            input_dim=input_dim_r_a2m, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.y_emb = FourierEmbedding(
            input_dim=output_dim + output_head, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands
        )
        self.traj_emb = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bias=True,
            batch_first=False,
            dropout=0.0,
            bidirectional=False,
        )
        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.t2m_propose_attn_layers = nn.ModuleList(
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
        self.pl2m_propose_attn_layers = nn.ModuleList(
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
        self.a2m_propose_attn_layers = nn.ModuleList(
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
        self.m2m_propose_attn_layer = AttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            bipartite=False,
            has_pos_emb=False,
        )
        self.t2m_refine_attn_layers = nn.ModuleList(
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
        self.pl2m_refine_attn_layers = nn.ModuleList(
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
        self.a2m_refine_attn_layers = nn.ModuleList(
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
        self.m2m_refine_attn_layer = AttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            bipartite=False,
            has_pos_emb=False,
        )
        self.to_loc_propose_pos = MLPLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_future_steps * output_dim // num_recurrent_steps,
        )
        self.to_scale_propose_pos = MLPLayer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_future_steps * output_dim // num_recurrent_steps,
        )
        self.to_loc_refine_pos = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_future_steps * output_dim
        )
        self.to_scale_refine_pos = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_future_steps * output_dim
        )
        if output_head:
            self.to_loc_propose_head = MLPLayer(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=num_future_steps // num_recurrent_steps,
            )
            self.to_conc_propose_head = MLPLayer(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=num_future_steps // num_recurrent_steps,
            )
            self.to_loc_refine_head = MLPLayer(
                input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_future_steps
            )
            self.to_conc_refine_head = MLPLayer(
                input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_future_steps
            )
        else:
            self.to_loc_propose_head = None
            self.to_conc_propose_head = None
            self.to_loc_refine_head = None
            self.to_conc_refine_head = None
        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.apply(weight_init)

    def forward(
        self, data: HeteroData, scene_enc: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pos_m = data["agent"]["position"][:, self.num_historical_steps - 1, : self.input_dim]
        head_m = data["agent"]["heading"][:, self.num_historical_steps - 1]
        head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)

        x_t = scene_enc["x_a"].reshape(-1, self.hidden_dim)
        x_pl = scene_enc["x_pl"][:, self.num_historical_steps - 1].repeat(self.num_modes, 1)
        x_a = scene_enc["x_a"][:, -1].repeat(self.num_modes, 1)
        m = self.mode_emb.weight.repeat(scene_enc["x_a"].size(0), 1)

        mask_src = data["agent"]["valid_mask"][:, : self.num_historical_steps].contiguous()
        mask_src[:, : self.num_historical_steps - self.num_t2m_steps] = False
        mask_dst = data["agent"]["predict_mask"].any(dim=-1, keepdim=True).repeat(1, self.num_modes)

        pos_t = data["agent"]["position"][:, : self.num_historical_steps, : self.input_dim].reshape(
            -1, self.input_dim
        )
        head_t = data["agent"]["heading"][:, : self.num_historical_steps].reshape(-1)
        edge_index_t2m = bipartite_dense_to_sparse(
            mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1)
        )
        rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]]
        rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]])
        r_t2m = torch.stack(
            [
                torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]
                ),
                rel_head_t2m,
                (edge_index_t2m[0] % self.num_historical_steps) - self.num_historical_steps + 1,
            ],
            dim=-1,
        )
        r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m, categorical_embs=None)
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))
        r_t2m = r_t2m.repeat_interleave(repeats=self.num_modes, dim=0)

        pos_pl = data["map_polygon"]["position"][:, : self.input_dim]
        orient_pl = data["map_polygon"]["orientation"]
        edge_index_pl2m = radius(
            x=pos_m[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2m_radius,
            batch_x=data["agent"]["batch"] if isinstance(data, Batch) else None,
            batch_y=data["map_polygon"]["batch"] if isinstance(data, Batch) else None,
            max_num_neighbors=300,
        )
        edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]]
        rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]]
        rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]])
        r_pl2m = torch.stack(
            [
                torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]
                ),
                rel_orient_pl2m,
            ],
            dim=-1,
        )
        r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None)
        edge_index_pl2m = torch.cat(
            [
                edge_index_pl2m
                + i
                * edge_index_pl2m.new_tensor(
                    [[data["map_polygon"]["num_nodes"]], [data["agent"]["num_nodes"]]]
                )
                for i in range(self.num_modes)
            ],
            dim=1,
        )
        r_pl2m = r_pl2m.repeat(self.num_modes, 1)

        edge_index_a2m = radius_graph(
            x=pos_m[:, :2],
            r=self.a2m_radius,
            batch=data["agent"]["batch"] if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=300,
        )
        edge_index_a2m = edge_index_a2m[
            :, mask_src[:, -1][edge_index_a2m[0]] & mask_dst[edge_index_a2m[1], 0]
        ]
        rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]]
        rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]])
        r_a2m = torch.stack(
            [
                torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_m[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]
                ),
                rel_head_a2m,
            ],
            dim=-1,
        )
        r_a2m = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)
        edge_index_a2m = torch.cat(
            [
                edge_index_a2m + i * edge_index_a2m.new_tensor([data["agent"]["num_nodes"]])
                for i in range(self.num_modes)
            ],
            dim=1,
        )
        r_a2m = r_a2m.repeat(self.num_modes, 1)

        edge_index_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[0]

        locs_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        scales_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        locs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        concs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        for t in range(self.num_recurrent_steps):
            for i in range(self.num_layers):
                m = m.reshape(-1, self.hidden_dim)
                m = self.t2m_propose_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
                m = (
                    m.reshape(-1, self.num_modes, self.hidden_dim)
                    .transpose(0, 1)
                    .reshape(-1, self.hidden_dim)
                )
                m = self.pl2m_propose_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                m = self.a2m_propose_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                m = (
                    m.reshape(self.num_modes, -1, self.hidden_dim)
                    .transpose(0, 1)
                    .reshape(-1, self.hidden_dim)
                )
            m = self.m2m_propose_attn_layer(m, None, edge_index_m2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim)
            locs_propose_pos[t] = self.to_loc_propose_pos(m)
            scales_propose_pos[t] = self.to_scale_propose_pos(m)
            if self.output_head:
                locs_propose_head[t] = self.to_loc_propose_head(m)
                concs_propose_head[t] = self.to_conc_propose_head(m)
        loc_propose_pos = torch.cumsum(
            torch.cat(locs_propose_pos, dim=-1).view(
                -1, self.num_modes, self.num_future_steps, self.output_dim
            ),
            dim=-2,
        )
        scale_propose_pos = (
            torch.cumsum(
                F.elu_(
                    torch.cat(scales_propose_pos, dim=-1).view(
                        -1, self.num_modes, self.num_future_steps, self.output_dim
                    ),
                    alpha=1.0,
                )
                + 1.0,
                dim=-2,
            )
            + 0.1
        )
        if self.output_head:
            loc_propose_head = torch.cumsum(
                torch.tanh(torch.cat(locs_propose_head, dim=-1).unsqueeze(-1)) * math.pi, dim=-2
            )
            conc_propose_head = 1.0 / (
                torch.cumsum(
                    F.elu_(torch.cat(concs_propose_head, dim=-1).unsqueeze(-1)) + 1.0, dim=-2
                )
                + 0.02
            )
            m = self.y_emb(
                torch.cat(
                    [loc_propose_pos.detach(), wrap_angle(loc_propose_head.detach())], dim=-1
                ).view(-1, self.output_dim + 1)
            )
        else:
            loc_propose_head = loc_propose_pos.new_zeros(
                (loc_propose_pos.size(0), self.num_modes, self.num_future_steps, 1)
            )
            conc_propose_head = scale_propose_pos.new_zeros(
                (scale_propose_pos.size(0), self.num_modes, self.num_future_steps, 1)
            )
            m = self.y_emb(loc_propose_pos.detach().view(-1, self.output_dim))
        m = m.reshape(-1, self.num_future_steps, self.hidden_dim).transpose(0, 1)
        m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0)
        for i in range(self.num_layers):
            m = self.t2m_refine_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
            m = (
                m.reshape(-1, self.num_modes, self.hidden_dim)
                .transpose(0, 1)
                .reshape(-1, self.hidden_dim)
            )
            m = self.pl2m_refine_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
            m = self.a2m_refine_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
            m = (
                m.reshape(self.num_modes, -1, self.hidden_dim)
                .transpose(0, 1)
                .reshape(-1, self.hidden_dim)
            )
        m = self.m2m_refine_attn_layer(m, None, edge_index_m2m)
        m = m.reshape(-1, self.num_modes, self.hidden_dim)
        loc_refine_pos = self.to_loc_refine_pos(m).view(
            -1, self.num_modes, self.num_future_steps, self.output_dim
        )
        loc_refine_pos = loc_refine_pos + loc_propose_pos.detach()
        scale_refine_pos = (
            F.elu_(
                self.to_scale_refine_pos(m).view(
                    -1, self.num_modes, self.num_future_steps, self.output_dim
                ),
                alpha=1.0,
            )
            + 1.0
            + 0.1
        )
        if self.output_head:
            loc_refine_head = torch.tanh(self.to_loc_refine_head(m).unsqueeze(-1)) * math.pi
            loc_refine_head = loc_refine_head + loc_propose_head.detach()
            conc_refine_head = 1.0 / (
                F.elu_(self.to_conc_refine_head(m).unsqueeze(-1)) + 1.0 + 0.02
            )
        else:
            loc_refine_head = loc_refine_pos.new_zeros(
                (loc_refine_pos.size(0), self.num_modes, self.num_future_steps, 1)
            )
            conc_refine_head = scale_refine_pos.new_zeros(
                (scale_refine_pos.size(0), self.num_modes, self.num_future_steps, 1)
            )
        pi = self.to_pi(m).squeeze(-1)

        return {
            "loc_propose_pos": loc_propose_pos,
            "scale_propose_pos": scale_propose_pos,
            "loc_propose_head": loc_propose_head,
            "conc_propose_head": conc_propose_head,
            "loc_refine_pos": loc_refine_pos,
            "scale_refine_pos": scale_refine_pos,
            "loc_refine_head": loc_refine_head,
            "conc_refine_head": conc_refine_head,
            "pi": pi,
        }


class QCNetNet(nn.Module):
    """Plain nn.Module wrapper matching the real QCNet.forward: encoder then decoder.

    Replaces `predictors/qcnet.py:QCNet(pl.LightningModule)` -- that class adds only
    optimizer/loss/metric bookkeeping around this exact `encoder(data); decoder(data, ...)`
    call, which is preserved verbatim.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.encoder = QCNetEncoder(
            dataset=kwargs["dataset"],
            input_dim=kwargs["input_dim"],
            hidden_dim=kwargs["hidden_dim"],
            num_historical_steps=kwargs["num_historical_steps"],
            pl2pl_radius=kwargs["pl2pl_radius"],
            time_span=kwargs["time_span"],
            pl2a_radius=kwargs["pl2a_radius"],
            a2a_radius=kwargs["a2a_radius"],
            num_freq_bands=kwargs["num_freq_bands"],
            num_map_layers=kwargs["num_map_layers"],
            num_agent_layers=kwargs["num_agent_layers"],
            num_heads=kwargs["num_heads"],
            head_dim=kwargs["head_dim"],
            dropout=kwargs["dropout"],
        )
        self.decoder = QCNetDecoder(
            dataset=kwargs["dataset"],
            input_dim=kwargs["input_dim"],
            hidden_dim=kwargs["hidden_dim"],
            output_dim=kwargs["output_dim"],
            output_head=kwargs["output_head"],
            num_historical_steps=kwargs["num_historical_steps"],
            num_future_steps=kwargs["num_future_steps"],
            num_modes=kwargs["num_modes"],
            num_recurrent_steps=kwargs["num_recurrent_steps"],
            num_t2m_steps=kwargs["num_t2m_steps"],
            pl2m_radius=kwargs["pl2m_radius"],
            a2m_radius=kwargs["a2m_radius"],
            num_freq_bands=kwargs["num_freq_bands"],
            num_layers=kwargs["num_dec_layers"],
            num_heads=kwargs["num_heads"],
            head_dim=kwargs["head_dim"],
            dropout=kwargs["dropout"],
        )

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        scene_enc = self.encoder(data)
        return self.decoder(data, scene_enc)


_HIDDEN_DIM = 32
_NUM_HIST = 6
_NUM_FUT = 12
_NUM_MODES = 3
_NUM_RECURRENT = 3


def build_qcnet():
    return QCNetNet(
        dataset="argoverse_v2",
        input_dim=2,
        hidden_dim=_HIDDEN_DIM,
        output_dim=2,
        output_head=True,
        num_historical_steps=_NUM_HIST,
        num_future_steps=_NUM_FUT,
        num_modes=_NUM_MODES,
        num_recurrent_steps=_NUM_RECURRENT,
        num_freq_bands=8,
        num_map_layers=1,
        num_agent_layers=1,
        num_dec_layers=1,
        num_heads=2,
        head_dim=8,
        dropout=0.0,
        pl2pl_radius=150.0,
        time_span=None,
        pl2a_radius=50.0,
        a2a_radius=50.0,
        num_t2m_steps=None,
        pl2m_radius=150.0,
        a2m_radius=150.0,
    )


def example_input_qcnet():
    torch.manual_seed(0)
    num_agents = 4
    num_pts = 10
    num_polys = 3

    data = HeteroData()

    data["agent"].num_nodes = num_agents
    data["agent"].position = torch.randn(num_agents, _NUM_HIST, 2)
    data["agent"].heading = torch.rand(num_agents, _NUM_HIST) * 2 * math.pi - math.pi
    data["agent"].velocity = torch.randn(num_agents, _NUM_HIST, 2)
    data["agent"].valid_mask = torch.ones(num_agents, _NUM_HIST, dtype=torch.bool)
    data["agent"].predict_mask = torch.ones(num_agents, _NUM_HIST + _NUM_FUT, dtype=torch.bool)
    data["agent"].type = torch.randint(0, 10, (num_agents,))

    data["map_point"].num_nodes = num_pts
    data["map_point"].position = torch.randn(num_pts, 2)
    data["map_point"].orientation = torch.rand(num_pts) * 2 * math.pi - math.pi
    data["map_point"].magnitude = torch.rand(num_pts)
    data["map_point"].type = torch.randint(0, 17, (num_pts,))
    data["map_point"].side = torch.randint(0, 3, (num_pts,))

    data["map_polygon"].num_nodes = num_polys
    data["map_polygon"].position = torch.randn(num_polys, 2)
    data["map_polygon"].orientation = torch.rand(num_polys) * 2 * math.pi - math.pi
    data["map_polygon"].type = torch.randint(0, 4, (num_polys,))
    data["map_polygon"].is_intersection = torch.randint(0, 3, (num_polys,))

    pt2pl_src = torch.arange(num_pts) % num_polys
    edge_index_pt2pl = torch.stack([torch.arange(num_pts), pt2pl_src], dim=0)
    data["map_point", "to", "map_polygon"].edge_index = edge_index_pt2pl

    pl_src, pl_dst = torch.meshgrid(torch.arange(num_polys), torch.arange(num_polys), indexing="ij")
    keep = pl_src != pl_dst
    data["map_polygon", "to", "map_polygon"].edge_index = torch.stack(
        [pl_src[keep], pl_dst[keep]], dim=0
    )
    data["map_polygon", "to", "map_polygon"].type = torch.randint(
        0, 5, (int(keep.sum().item()),), dtype=torch.uint8
    )

    return (data,)


MENAGERIE_ENTRIES = [
    ("QCNet", "build_qcnet", "example_input_qcnet", 2023, "vendored-pytorch"),
]
