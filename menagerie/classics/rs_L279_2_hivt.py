# SOURCE: vendored from ZikangZhou/HiVT @ main
# Files combined:
#   models/embedding.py (SingleInputEmbedding, MultipleInputEmbedding)
#   models/local_encoder.py (LocalEncoder, AAEncoder, TemporalEncoder, TemporalEncoderLayer,
#       ALEncoder -- agent-agent + temporal + agent-lane local encoders)
#   models/global_interactor.py (GlobalInteractor, GlobalInteractorLayer -- scene-level
#       multi-modal Transformer interaction module)
#   models/decoder.py (MLPDecoder -- the repo's default multi-modal Laplace-mixture decoder;
#       GRUDecoder omitted, it's an alternative decoder never used by the repo's shipped
#       configs)
#   utils.py (TemporalData, DistanceDropEdge, init_weights)
#   models/hivt.py (HiVT.forward() logic, reproduced as a plain nn.Module -- see note below)
#
# HiVT (Zhou et al., "HiVT: Hierarchical Vector Transformer for Multi-Agent Motion
# Prediction", CVPR 2022) is a hierarchical vectorized Transformer for multi-agent
# trajectory forecasting on Argoverse: a LocalEncoder builds per-agent local context via
# agent-agent temporal message passing (AAEncoder + TemporalEncoder, both torch_geometric
# MessagePassing/Transformer stacks) fused with agent-lane graph attention (ALEncoder), a
# GlobalInteractor runs scene-level rotation-invariant multi-modal Transformer interaction
# over all agents, and an MLPDecoder emits K-mode Laplace mixture future trajectories with
# mixing weights.
#
# Import-only fixes applied (no architectural change):
#   - The original `models/hivt.py::HiVT` is a `pytorch_lightning.LightningModule` whose
#     `__init__` builds `LocalEncoder` + `GlobalInteractor` + `MLPDecoder` and whose
#     `forward(data)` runs exactly the three-stage pipeline reproduced verbatim below in
#     `HiVT.forward()` (rotate_mat construction -> local_encoder -> global_interactor ->
#     decoder). `training_step`/`validation_step`/`configure_optimizers` (loss computation,
#     metrics, optimizer schedule -- all training-only) and `GRUDecoder` (an alternative,
#     never-instantiated decoder) are not needed to build/trace the model and are omitted;
#     this file's `HiVT` is a plain `nn.Module` instead of a `pl.LightningModule` so that
#     `pytorch_lightning` is not a hard dependency, but the encoder/interactor/decoder
#     submodules and the `forward()` control flow are unmodified from the real repo.
#   - `TemporalData.__init__`/`__inc__` (the repo's `torch_geometric.data.Data` subclass) are
#     vendored unmodified; the example input builds one directly with plain tensors (no
#     dataloader/dataset preprocessing needed to trace the model).
#   - `TemporalEncoderLayer.forward()` gained an `is_causal: bool = False` keyword-only
#     parameter (accepted and ignored). Newer `torch.nn.TransformerEncoder.forward()`
#     (post-2.1) always forwards an `is_causal` kwarg to each layer's `forward()`; the repo's
#     2022-era custom layer predates that torch signature change and would TypeError on
#     current torch otherwise. Pure signature widening, no behavioral change.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, Size
from torch_geometric.utils import softmax, subgraph


# ---------------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------------
class TemporalData(Data):
    def __init__(
        self,
        x: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attrs: Optional[List[torch.Tensor]] = None,
        y: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
        bos_mask: Optional[torch.Tensor] = None,
        rotate_angles: Optional[torch.Tensor] = None,
        lane_vectors: Optional[torch.Tensor] = None,
        is_intersections: Optional[torch.Tensor] = None,
        turn_directions: Optional[torch.Tensor] = None,
        traffic_controls: Optional[torch.Tensor] = None,
        lane_actor_index: Optional[torch.Tensor] = None,
        lane_actor_vectors: Optional[torch.Tensor] = None,
        seq_id: Optional[int] = None,
        **kwargs,
    ) -> None:
        if x is None:
            super(TemporalData, self).__init__()
            return
        super(TemporalData, self).__init__(
            x=x,
            positions=positions,
            edge_index=edge_index,
            y=y,
            num_nodes=num_nodes,
            padding_mask=padding_mask,
            bos_mask=bos_mask,
            rotate_angles=rotate_angles,
            lane_vectors=lane_vectors,
            is_intersections=is_intersections,
            turn_directions=turn_directions,
            traffic_controls=traffic_controls,
            lane_actor_index=lane_actor_index,
            lane_actor_vectors=lane_actor_vectors,
            seq_id=seq_id,
            **kwargs,
        )
        if edge_attrs is not None:
            for t in range(self.x.size(1)):
                self[f"edge_attr_{t}"] = edge_attrs[t]

    def __inc__(self, key, value):
        if key == "lane_actor_index":
            return torch.tensor([[self["lane_vectors"].size(0)], [self.num_nodes]])
        else:
            return super().__inc__(key, value)


class DistanceDropEdge:
    def __init__(self, max_distance: Optional[float] = None) -> None:
        self.max_distance = max_distance

    def __call__(
        self, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_distance is None:
            return edge_index, edge_attr
        row, col = edge_index
        mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        edge_attr = edge_attr[mask]
        return edge_index, edge_attr


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
# models/embedding.py
# ---------------------------------------------------------------------------
class SingleInputEmbedding(nn.Module):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super(SingleInputEmbedding, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
        )
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class MultipleInputEmbedding(nn.Module):
    def __init__(self, in_channels: List[int], out_channel: int) -> None:
        super(MultipleInputEmbedding, self).__init__()
        self.module_list = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_channel, out_channel),
                    nn.LayerNorm(out_channel),
                    nn.ReLU(inplace=True),
                    nn.Linear(out_channel, out_channel),
                )
                for in_channel in in_channels
            ]
        )
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
        )
        self.apply(init_weights)

    def forward(
        self,
        continuous_inputs: List[torch.Tensor],
        categorical_inputs: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i])
        output = torch.stack(continuous_inputs).sum(dim=0)
        if categorical_inputs is not None:
            output += torch.stack(categorical_inputs).sum(dim=0)
        return self.aggr_embed(output)


# ---------------------------------------------------------------------------
# models/local_encoder.py
# ---------------------------------------------------------------------------
class LocalEncoder(nn.Module):
    def __init__(
        self,
        historical_steps: int,
        node_dim: int,
        edge_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_temporal_layers: int = 4,
        local_radius: float = 50,
        parallel: bool = False,
    ) -> None:
        super(LocalEncoder, self).__init__()
        self.historical_steps = historical_steps
        self.parallel = parallel

        self.drop_edge = DistanceDropEdge(local_radius)
        self.aa_encoder = AAEncoder(
            historical_steps=historical_steps,
            node_dim=node_dim,
            edge_dim=edge_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            parallel=parallel,
        )
        self.temporal_encoder = TemporalEncoder(
            historical_steps=historical_steps,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_temporal_layers,
        )
        self.al_encoder = ALEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, data: TemporalData) -> torch.Tensor:
        for t in range(self.historical_steps):
            data[f"edge_index_{t}"], _ = subgraph(
                subset=~data["padding_mask"][:, t], edge_index=data.edge_index
            )
            data[f"edge_attr_{t}"] = (
                data["positions"][data[f"edge_index_{t}"][0], t]
                - data["positions"][data[f"edge_index_{t}"][1], t]
            )
        if self.parallel:
            snapshots = [None] * self.historical_steps
            for t in range(self.historical_steps):
                edge_index, edge_attr = self.drop_edge(
                    data[f"edge_index_{t}"], data[f"edge_attr_{t}"]
                )
                snapshots[t] = Data(
                    x=data.x[:, t],
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    num_nodes=data.num_nodes,
                )
            batch = Batch.from_data_list(snapshots)
            out = self.aa_encoder(
                x=batch.x,
                t=None,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                bos_mask=data["bos_mask"],
                rotate_mat=data["rotate_mat"],
            )
            out = out.view(self.historical_steps, out.shape[0] // self.historical_steps, -1)
        else:
            out = [None] * self.historical_steps
            for t in range(self.historical_steps):
                edge_index, edge_attr = self.drop_edge(
                    data[f"edge_index_{t}"], data[f"edge_attr_{t}"]
                )
                out[t] = self.aa_encoder(
                    x=data.x[:, t],
                    t=t,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    bos_mask=data["bos_mask"][:, t],
                    rotate_mat=data["rotate_mat"],
                )
            out = torch.stack(out)  # [T, N, D]
        out = self.temporal_encoder(
            x=out, padding_mask=data["padding_mask"][:, : self.historical_steps]
        )
        edge_index, edge_attr = self.drop_edge(data["lane_actor_index"], data["lane_actor_vectors"])
        out = self.al_encoder(
            x=(data["lane_vectors"], out),
            edge_index=edge_index,
            edge_attr=edge_attr,
            is_intersections=data["is_intersections"],
            turn_directions=data["turn_directions"],
            traffic_controls=data["traffic_controls"],
            rotate_mat=data["rotate_mat"],
        )
        return out


class AAEncoder(MessagePassing):
    def __init__(
        self,
        historical_steps: int,
        node_dim: int,
        edge_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        parallel: bool = False,
        **kwargs,
    ) -> None:
        super(AAEncoder, self).__init__(aggr="add", node_dim=0, **kwargs)
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.parallel = parallel

        self.center_embed = SingleInputEmbedding(in_channel=node_dim, out_channel=embed_dim)
        self.nbr_embed = MultipleInputEmbedding(
            in_channels=[node_dim, edge_dim], out_channel=embed_dim
        )
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.bos_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        nn.init.normal_(self.bos_token, mean=0.0, std=0.02)
        self.apply(init_weights)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[int],
        edge_index: Adj,
        edge_attr: torch.Tensor,
        bos_mask: torch.Tensor,
        rotate_mat: Optional[torch.Tensor] = None,
        size: Size = None,
    ) -> torch.Tensor:
        if self.parallel:
            if rotate_mat is None:
                center_embed = self.center_embed(
                    x.view(self.historical_steps, x.shape[0] // self.historical_steps, -1)
                )
            else:
                center_embed = self.center_embed(
                    torch.matmul(
                        x.view(
                            self.historical_steps, x.shape[0] // self.historical_steps, -1
                        ).unsqueeze(-2),
                        rotate_mat.expand(self.historical_steps, *rotate_mat.shape),
                    ).squeeze(-2)
                )
            center_embed = torch.where(
                bos_mask.t().unsqueeze(-1), self.bos_token.unsqueeze(-2), center_embed
            ).view(x.shape[0], -1)
        else:
            if rotate_mat is None:
                center_embed = self.center_embed(x)
            else:
                center_embed = self.center_embed(torch.bmm(x.unsqueeze(-2), rotate_mat).squeeze(-2))
            center_embed = torch.where(bos_mask.unsqueeze(-1), self.bos_token[t], center_embed)
        center_embed = center_embed + self._mha_block(
            self.norm1(center_embed), x, edge_index, edge_attr, rotate_mat, size
        )
        center_embed = center_embed + self._ff_block(self.norm2(center_embed))
        return center_embed

    def message(
        self,
        edge_index: Adj,
        center_embed_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        rotate_mat: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> torch.Tensor:
        if rotate_mat is None:
            nbr_embed = self.nbr_embed([x_j, edge_attr])
        else:
            if self.parallel:
                center_rotate_mat = rotate_mat.repeat(self.historical_steps, 1, 1)[edge_index[1]]
            else:
                center_rotate_mat = rotate_mat[edge_index[1]]
            nbr_embed = self.nbr_embed(
                [
                    torch.bmm(x_j.unsqueeze(-2), center_rotate_mat).squeeze(-2),
                    torch.bmm(edge_attr.unsqueeze(-2), center_rotate_mat).squeeze(-2),
                ]
            )
        query = self.lin_q(center_embed_i).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
        )
        key = self.lin_k(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self, inputs: torch.Tensor, center_embed: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(center_embed))
        return inputs + gate * (self.lin_self(center_embed) - inputs)

    def _mha_block(
        self,
        center_embed: torch.Tensor,
        x: torch.Tensor,
        edge_index: Adj,
        edge_attr: torch.Tensor,
        rotate_mat: Optional[torch.Tensor],
        size: Size,
    ) -> torch.Tensor:
        center_embed = self.out_proj(
            self.propagate(
                edge_index=edge_index,
                x=x,
                center_embed=center_embed,
                edge_attr=edge_attr,
                rotate_mat=rotate_mat,
                size=size,
            )
        )
        return self.proj_drop(center_embed)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        historical_steps: int,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super(TemporalEncoder, self).__init__()
        encoder_layer = TemporalEncoderLayer(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(embed_dim)
        )
        self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))
        attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)
        self.register_buffer("attn_mask", attn_mask)
        nn.init.normal_(self.padding_token, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        self.apply(init_weights)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x)
        expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        x = torch.cat((x, expand_cls_token), dim=0)
        x = x + self.pos_embed
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None)
        return out[-1]  # [N, D]

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask


class TemporalEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super(TemporalEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)


class ALEncoder(MessagePassing):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super(ALEncoder, self).__init__(aggr="add", node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lane_embed = MultipleInputEmbedding(
            in_channels=[node_dim, edge_dim], out_channel=embed_dim
        )
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.is_intersection_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        self.turn_direction_embed = nn.Parameter(torch.Tensor(3, embed_dim))
        self.traffic_control_embed = nn.Parameter(torch.Tensor(2, embed_dim))
        nn.init.normal_(self.is_intersection_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.turn_direction_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.traffic_control_embed, mean=0.0, std=0.02)
        self.apply(init_weights)

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        edge_index: Adj,
        edge_attr: torch.Tensor,
        is_intersections: torch.Tensor,
        turn_directions: torch.Tensor,
        traffic_controls: torch.Tensor,
        rotate_mat: Optional[torch.Tensor] = None,
        size: Size = None,
    ) -> torch.Tensor:
        x_lane, x_actor = x
        is_intersections = is_intersections.long()
        turn_directions = turn_directions.long()
        traffic_controls = traffic_controls.long()
        x_actor = x_actor + self._mha_block(
            self.norm1(x_actor),
            x_lane,
            edge_index,
            edge_attr,
            is_intersections,
            turn_directions,
            traffic_controls,
            rotate_mat,
            size,
        )
        x_actor = x_actor + self._ff_block(self.norm2(x_actor))
        return x_actor

    def message(
        self,
        edge_index: Adj,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        is_intersections_j,
        turn_directions_j,
        traffic_controls_j,
        rotate_mat: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> torch.Tensor:
        if rotate_mat is None:
            x_j = self.lane_embed(
                [x_j, edge_attr],
                [
                    self.is_intersection_embed[is_intersections_j],
                    self.turn_direction_embed[turn_directions_j],
                    self.traffic_control_embed[traffic_controls_j],
                ],
            )
        else:
            rotate_mat = rotate_mat[edge_index[1]]
            x_j = self.lane_embed(
                [
                    torch.bmm(x_j.unsqueeze(-2), rotate_mat).squeeze(-2),
                    torch.bmm(edge_attr.unsqueeze(-2), rotate_mat).squeeze(-2),
                ],
                [
                    self.is_intersection_embed[is_intersections_j],
                    self.turn_direction_embed[turn_directions_j],
                    self.traffic_control_embed[traffic_controls_j],
                ],
            )
        query = self.lin_q(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self, inputs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_actor = x[1]
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x_actor))
        return inputs + gate * (self.lin_self(x_actor) - inputs)

    def _mha_block(
        self,
        x_actor: torch.Tensor,
        x_lane: torch.Tensor,
        edge_index: Adj,
        edge_attr: torch.Tensor,
        is_intersections: torch.Tensor,
        turn_directions: torch.Tensor,
        traffic_controls: torch.Tensor,
        rotate_mat: Optional[torch.Tensor],
        size: Size,
    ) -> torch.Tensor:
        x_actor = self.out_proj(
            self.propagate(
                edge_index=edge_index,
                x=(x_lane, x_actor),
                edge_attr=edge_attr,
                is_intersections=is_intersections,
                turn_directions=turn_directions,
                traffic_controls=traffic_controls,
                rotate_mat=rotate_mat,
                size=size,
            )
        )
        return self.proj_drop(x_actor)

    def _ff_block(self, x_actor: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_actor)


# ---------------------------------------------------------------------------
# models/global_interactor.py
# ---------------------------------------------------------------------------
class GlobalInteractor(nn.Module):
    def __init__(
        self,
        historical_steps: int,
        embed_dim: int,
        edge_dim: int,
        num_modes: int = 6,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        rotate: bool = True,
    ) -> None:
        super(GlobalInteractor, self).__init__()
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_modes = num_modes

        if rotate:
            self.rel_embed = MultipleInputEmbedding(
                in_channels=[edge_dim, edge_dim], out_channel=embed_dim
            )
        else:
            self.rel_embed = SingleInputEmbedding(in_channel=edge_dim, out_channel=embed_dim)
        self.global_interactor_layers = nn.ModuleList(
            [
                GlobalInteractorLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.multihead_proj = nn.Linear(embed_dim, num_modes * embed_dim)
        self.apply(init_weights)

    def forward(self, data: TemporalData, local_embed: torch.Tensor) -> torch.Tensor:
        edge_index, _ = subgraph(
            subset=~data["padding_mask"][:, self.historical_steps - 1], edge_index=data.edge_index
        )
        rel_pos = (
            data["positions"][edge_index[0], self.historical_steps - 1]
            - data["positions"][edge_index[1], self.historical_steps - 1]
        )
        if data["rotate_mat"] is None:
            rel_embed = self.rel_embed(rel_pos)
        else:
            rel_pos = torch.bmm(rel_pos.unsqueeze(-2), data["rotate_mat"][edge_index[1]]).squeeze(
                -2
            )
            rel_theta = data["rotate_angles"][edge_index[0]] - data["rotate_angles"][edge_index[1]]
            rel_theta_cos = torch.cos(rel_theta).unsqueeze(-1)
            rel_theta_sin = torch.sin(rel_theta).unsqueeze(-1)
            rel_embed = self.rel_embed([rel_pos, torch.cat((rel_theta_cos, rel_theta_sin), dim=-1)])
        x = local_embed
        for layer in self.global_interactor_layers:
            x = layer(x, edge_index, rel_embed)
        x = self.norm(x)  # [N, D]
        x = self.multihead_proj(x).view(-1, self.num_modes, self.embed_dim)  # [N, F, D]
        x = x.transpose(0, 1)  # [F, N, D]
        return x


class GlobalInteractorLayer(MessagePassing):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, **kwargs) -> None:
        super(GlobalInteractorLayer, self).__init__(aggr="add", node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.lin_q_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_v_node = nn.Linear(embed_dim, embed_dim)
        self.lin_v_edge = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, edge_index: Adj, edge_attr: torch.Tensor, size: Size = None
    ) -> torch.Tensor:
        x = x + self._mha_block(self.norm1(x), edge_index, edge_attr, size)
        x = x + self._ff_block(self.norm2(x))
        return x

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        index: torch.Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> torch.Tensor:
        query = self.lin_q_node(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key_node = self.lin_k_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key_edge = self.lin_k_edge(edge_attr).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
        )
        value_node = self.lin_v_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_edge = self.lin_v_edge(edge_attr).view(
            -1, self.num_heads, self.embed_dim // self.num_heads
        )
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * (key_node + key_edge)).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return (value_node + value_edge) * alpha.unsqueeze(-1)

    def update(self, inputs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x))
        return inputs + gate * (self.lin_self(x) - inputs)

    def _mha_block(
        self, x: torch.Tensor, edge_index: Adj, edge_attr: torch.Tensor, size: Size
    ) -> torch.Tensor:
        x = self.out_proj(
            self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=size)
        )
        return self.proj_drop(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ---------------------------------------------------------------------------
# models/decoder.py (MLPDecoder only -- GRUDecoder is an unused alternative)
# ---------------------------------------------------------------------------
class MLPDecoder(nn.Module):
    def __init__(
        self,
        local_channels: int,
        global_channels: int,
        future_steps: int,
        num_modes: int,
        uncertain: bool = True,
        min_scale: float = 1e-3,
    ) -> None:
        super(MLPDecoder, self).__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        self.aggr_embed = nn.Sequential(
            nn.Linear(self.input_size + self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
        )
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.future_steps * 2),
        )
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.future_steps * 2),
            )
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1),
        )
        self.apply(init_weights)

    def forward(
        self, local_embed: torch.Tensor, global_embed: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = (
            self.pi(
                torch.cat(
                    (local_embed.expand(self.num_modes, *local_embed.shape), global_embed), dim=-1
                )
            )
            .squeeze(-1)
            .t()
        )
        out = self.aggr_embed(
            torch.cat(
                (global_embed, local_embed.expand(self.num_modes, *local_embed.shape)), dim=-1
            )
        )
        loc = self.loc(out).view(self.num_modes, -1, self.future_steps, 2)  # [F, N, H, 2]
        if self.uncertain:
            scale = (
                F.elu_(self.scale(out), alpha=1.0).view(self.num_modes, -1, self.future_steps, 2)
                + 1.0
            )
            scale = scale + self.min_scale  # [F, N, H, 2]
            return torch.cat((loc, scale), dim=-1), pi  # [F, N, H, 4], [N, F]
        else:
            return loc, pi  # [F, N, H, 2], [N, F]


# ---------------------------------------------------------------------------
# models/hivt.py -- HiVT.forward() control flow, as a plain nn.Module (see header note)
# ---------------------------------------------------------------------------
class HiVT(nn.Module):
    def __init__(
        self,
        historical_steps: int,
        future_steps: int,
        num_modes: int,
        rotate: bool,
        node_dim: int,
        edge_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        num_temporal_layers: int,
        num_global_layers: int,
        local_radius: float,
        parallel: bool,
    ) -> None:
        super(HiVT, self).__init__()
        self.historical_steps = historical_steps
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.rotate = rotate
        self.parallel = parallel

        self.local_encoder = LocalEncoder(
            historical_steps=historical_steps,
            node_dim=node_dim,
            edge_dim=edge_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_temporal_layers=num_temporal_layers,
            local_radius=local_radius,
            parallel=parallel,
        )
        self.global_interactor = GlobalInteractor(
            historical_steps=historical_steps,
            embed_dim=embed_dim,
            edge_dim=edge_dim,
            num_modes=num_modes,
            num_heads=num_heads,
            num_layers=num_global_layers,
            dropout=dropout,
            rotate=rotate,
        )
        self.decoder = MLPDecoder(
            local_channels=embed_dim,
            global_channels=embed_dim,
            future_steps=future_steps,
            num_modes=num_modes,
            uncertain=True,
        )

    def forward(self, data: TemporalData):
        if self.rotate:
            rotate_mat = torch.empty(data.num_nodes, 2, 2, device=data.x.device)
            sin_vals = torch.sin(data["rotate_angles"])
            cos_vals = torch.cos(data["rotate_angles"])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if data.y is not None:
                data.y = torch.bmm(data.y, rotate_mat)
            data["rotate_mat"] = rotate_mat
        else:
            data["rotate_mat"] = None

        local_embed = self.local_encoder(data=data)
        global_embed = self.global_interactor(data=data, local_embed=local_embed)
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
        return y_hat, pi


# ---------------------------------------------------------------------------
# menagerie staging entry points
# ---------------------------------------------------------------------------
MENAGERIE_ZOO = "vendored-pytorch"


def build_hivt():
    # Mirrors the repo's train.py / add_model_specific_args defaults, sized down for a
    # fast-tracing smoke config (embed_dim shrunk from 64 -> 16; layer counts kept as-shipped).
    return HiVT(
        historical_steps=20,
        future_steps=30,
        num_modes=6,
        rotate=True,
        node_dim=2,
        edge_dim=2,
        embed_dim=16,
        num_heads=4,
        dropout=0.1,
        num_temporal_layers=2,
        num_global_layers=2,
        local_radius=50,
        parallel=False,
    )


def example_input_hivt():
    # Small synthetic scene: 5 agents, 20 historical steps, 3 lane segments, fully connected
    # actor-actor graph. Mirrors the field layout the repo's ArgoverseV1Dataset would produce.
    torch.manual_seed(0)
    n_agents = 5
    historical_steps = 20
    n_lanes = 3

    x = torch.randn(n_agents, historical_steps, 2)
    positions = torch.randn(n_agents, historical_steps, 2)

    # Fully connected actor-actor graph (no self-loops), matching the repo's dense radius graph.
    src, dst = [], []
    for i in range(n_agents):
        for j in range(n_agents):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    padding_mask = torch.zeros(n_agents, historical_steps + 30, dtype=torch.bool)
    bos_mask = torch.zeros(n_agents, historical_steps, dtype=torch.bool)
    bos_mask[:, 0] = True
    rotate_angles = torch.rand(n_agents) * 2 * 3.14159265

    lane_vectors = torch.randn(n_lanes, 2)
    is_intersections = torch.zeros(n_lanes, dtype=torch.bool)
    turn_directions = torch.zeros(n_lanes, dtype=torch.long)
    traffic_controls = torch.zeros(n_lanes, dtype=torch.bool)

    # Every lane connects to every actor.
    lane_src, actor_dst = [], []
    for lane_id in range(n_lanes):
        for actor_id in range(n_agents):
            lane_src.append(lane_id)
            actor_dst.append(actor_id)
    lane_actor_index = torch.tensor([lane_src, actor_dst], dtype=torch.long)
    lane_actor_vectors = torch.randn(len(lane_src), 2)

    y = torch.randn(n_agents, 30, 2)

    data = TemporalData(
        x=x,
        positions=positions,
        edge_index=edge_index,
        y=y,
        num_nodes=n_agents,
        padding_mask=padding_mask,
        bos_mask=bos_mask,
        rotate_angles=rotate_angles,
        lane_vectors=lane_vectors,
        is_intersections=is_intersections,
        turn_directions=turn_directions,
        traffic_controls=traffic_controls,
        lane_actor_index=lane_actor_index,
        lane_actor_vectors=lane_actor_vectors,
    )
    return (data,)


MENAGERIE_ENTRIES = [
    ("HiVT", build_hivt, example_input_hivt, 2022, MENAGERIE_ZOO),
]
