# SOURCE: vendored from https://github.com/zhejz/TrafficBots @ main
# (src/models/traffic_bots.py, src/models/goal_manager.py, src/models/latent_encoder.py,
#  src/models/modules/*.py, src/utils/transform_utils.py; CC BY-NC 4.0 license, code copied verbatim
#  from the official zhejz/TrafficBots repo -- only import paths were flattened into this single
#  staging file so it can run standalone in the base torchlens env; the nn.Module classes,
#  forward() bodies, and math are unmodified real repository code)
"""TrafficBots (Zhang et al., ICRA 2023) -- conditional-VAE + cross-attention Transformer
policy for closed-loop multi-agent traffic simulation on the Waymo Open Motion Dataset.

This staging module wires the REAL `TrafficBots` nn.Module (and its real sub-modules
MapEncoder / InputPeEncoder / GoalManager / LatentEncoder / MultiAgentTF / MultiAgentGRULoop /
TemporalAggregate / AddLatentGoal / TransformerBlock / Attention / MLP) together with a tiny
synthetic scene-centric batch (agents/map/traffic-lights) matching the shapes documented in
src/data_modules/scene_centric.py, then runs one autoregressive forward step exactly as
TrafficBots.encode_input_features -> init -> forward is called from the real training loop
(src/pl_modules/waymo_motion.py).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# src/utils/transform_utils.py (only the torch helpers TrafficBots needs)
# ---------------------------------------------------------------------------
def cast_rad(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def torch_rad2rot(rad: Tensor) -> Tensor:
    _cos = torch.cos(rad)
    _sin = torch.sin(rad)
    return torch.stack(
        [torch.stack([_cos, -_sin], dim=-1), torch.stack([_sin, _cos], dim=-1)], dim=-2
    )


def torch_pos2local(in_pos: Tensor, local_pos: Tensor, local_rot: Tensor) -> Tensor:
    return torch.matmul(in_pos - local_pos, local_rot)


def torch_pos2global(in_pos: Tensor, local_pos: Tensor, local_rot: Tensor) -> Tensor:
    return torch.matmul(in_pos, local_rot.transpose(-1, -2)) + local_pos


# ---------------------------------------------------------------------------
# src/models/modules/mlp.py
# ---------------------------------------------------------------------------
def _get_activation(activation: str, inplace: bool) -> nn.Module:
    if activation == "relu":
        return nn.ReLU(inplace=inplace)
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU(inplace=inplace)
    elif activation == "elu":
        return nn.ELU(inplace=inplace)
    elif activation == "rrelu":
        return nn.RReLU(inplace=inplace)
    raise RuntimeError("activation {} not implemented".format(activation))


class MLP(nn.Module):
    def __init__(
        self,
        fc_dims,
        dropout_p: Optional[float] = None,
        use_layernorm: bool = False,
        activation: str = "relu",
        end_layer_activation: bool = True,
        init_weight_norm: bool = False,
        init_bias: Optional[float] = None,
        use_batchnorm: bool = False,
    ) -> None:
        super(MLP, self).__init__()
        assert len(fc_dims) >= 2
        assert not (use_layernorm and use_batchnorm)
        layers: List[nn.Module] = []
        for i in range(0, len(fc_dims) - 1):
            fc = nn.Linear(fc_dims[i], fc_dims[i + 1])
            if init_weight_norm:
                fc.weight.data *= 1.0 / fc.weight.norm(dim=1, p=2, keepdim=True)
            if init_bias is not None and i == len(fc_dims) - 2:
                fc.bias.data *= 0
                fc.bias.data += init_bias
            layers.append(fc)
            if i < len(fc_dims) - 2:
                if use_layernorm:
                    layers.append(nn.LayerNorm(fc_dims[i + 1]))
                elif use_batchnorm:
                    layers.append(nn.BatchNorm1d(fc_dims[i + 1]))
                if dropout_p is not None:
                    layers.append(nn.Dropout(p=dropout_p))
                layers.append(_get_activation(activation, inplace=True))
            if i == len(fc_dims) - 2:
                if end_layer_activation:
                    if use_layernorm:
                        layers.append(nn.LayerNorm(fc_dims[i + 1]))
                    elif use_batchnorm:
                        layers.append(nn.BatchNorm1d(fc_dims[i + 1]))
                    if dropout_p is not None:
                        layers.append(nn.Dropout(p=dropout_p))
                    self.end_layer_activation = _get_activation(activation, inplace=True)
                else:
                    self.end_layer_activation = None

        self.input_dim = fc_dims[0]
        self.output_dim = fc_dims[-1]
        self.fc_layers = nn.Sequential(*layers)

    def forward(
        self, x: Tensor, valid_mask: Optional[Tensor] = None, fill_invalid: float = 0.0
    ) -> Tensor:
        x = self.fc_layers(x.flatten(0, -2)).view(*x.shape[:-1], self.output_dim)
        if valid_mask is not None:
            x.masked_fill_(~valid_mask.unsqueeze(-1), fill_invalid)
        if self.end_layer_activation is not None:
            self.end_layer_activation(x)
        return x


# ---------------------------------------------------------------------------
# src/models/modules/attention.py
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(
        self, d_model: int, n_head: int, dropout_p: float = 0.0, bias: bool = True
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        assert self.d_head * n_head == d_model, "d_model must be divisible by n_head"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * d_model, d_model)))
        self.out_proj_weight = nn.Parameter(torch.empty((d_model, d_model)))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))
            self.out_proj_bias = nn.Parameter(torch.empty(d_model))
        else:
            self.register_parameter("in_proj_bias", None)
            self.register_parameter("out_proj_bias", None)

        self.dropout = nn.Dropout(p=dropout_p, inplace=False) if dropout_p > 0 else None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj_bias, 0.0)

    def forward(
        self,
        src: Tensor,
        tgt: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        n_batch, n_src, _ = src.shape
        if tgt is None:
            n_tgt = n_src
            qkv = F.linear(src, self.in_proj_weight, self.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            n_tgt = tgt.shape[-2]
            w_src, w_tgt = self.in_proj_weight.split([self.d_model, self.d_model * 2])
            b_src, b_tgt = None, None
            if self.in_proj_bias is not None:
                b_src, b_tgt = self.in_proj_bias.split([self.d_model, self.d_model * 2])
            q = F.linear(src, w_src, b_src)
            kv = F.linear(tgt, w_tgt, b_tgt)
            k, v = kv.chunk(2, dim=-1)

        attn_invalid_mask = None
        if tgt_padding_mask is not None:
            attn_invalid_mask = tgt_padding_mask
            if attn_invalid_mask.dim() == 2:
                attn_invalid_mask = attn_invalid_mask.unsqueeze(1).expand(-1, n_src, -1)
        if attn_mask is not None:
            if attn_invalid_mask is None:
                attn_invalid_mask = attn_mask
            else:
                attn_invalid_mask = attn_invalid_mask | attn_mask

        mask_no_tgt_valid = None
        if attn_invalid_mask is not None:
            mask_no_tgt_valid = attn_invalid_mask.all(-1)
            if mask_no_tgt_valid.any():
                attn_invalid_mask = attn_invalid_mask & (~mask_no_tgt_valid.unsqueeze(-1))
            else:
                mask_no_tgt_valid = None

        if k.dim() == 3:
            q = q.view(n_batch, n_src, self.n_head, self.d_head).transpose(1, 2).contiguous()
            k = k.view(n_batch, n_tgt, self.n_head, self.d_head).transpose(1, 2).contiguous()
            v = v.view(n_batch, n_tgt, self.n_head, self.d_head).transpose(1, 2).contiguous()
            attn = torch.matmul(q, k.transpose(-2, -1))
        else:
            k = k.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
            v = v.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
            q = q.view(n_batch, n_src, self.n_head, self.d_head).transpose(1, 2).unsqueeze(3)
            attn = torch.sum(q * k, dim=-1)

        if attn_invalid_mask is not None:
            attn = attn.masked_fill(attn_invalid_mask.unsqueeze(1), float("-inf"))

        attn = torch.softmax(attn / math.sqrt(self.d_head), dim=-1)
        if self.dropout is not None:
            attn = self.dropout(attn)

        if v.dim() == 4:
            out = torch.matmul(attn, v)
        else:
            out = torch.sum(v * attn.unsqueeze(-1), dim=3)

        out = out.transpose(1, 2).flatten(2, 3)
        out = F.linear(out, self.out_proj_weight, self.out_proj_bias)

        if mask_no_tgt_valid is not None:
            out = out.masked_fill(mask_no_tgt_valid.unsqueeze(-1), 0)

        if need_weights:
            attn_weights = attn.mean(1)
            if mask_no_tgt_valid is not None:
                attn_weights = attn_weights.masked_fill(mask_no_tgt_valid.unsqueeze(-1), 0)
            return out, attn_weights
        else:
            return out, None


# ---------------------------------------------------------------------------
# src/models/modules/transformer.py
# ---------------------------------------------------------------------------
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "elu":
        return F.elu
    raise RuntimeError("activation should be relu/gelu/elu, not {}".format(activation))


class TransformerCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_feedforward: int,
        dropout_p: float,
        activation: str,
        norm_first: bool,
        decoder_self_attn: bool,
        bias: bool,
    ) -> None:
        super().__init__()
        self.norm_first = norm_first
        self.d_feedforward = d_feedforward
        self.decoder_self_attn = decoder_self_attn
        inplace = False

        self.dropout = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None
        self.activation = _get_activation_fn(activation)
        self.norm1 = nn.LayerNorm(d_model)

        if self.decoder_self_attn:
            self.attn_src = Attention(
                d_model=d_model, n_head=n_head, dropout_p=dropout_p, bias=bias
            )
            self.norm_src = nn.LayerNorm(d_model)
            self.dropout_src = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None

        if self.norm_first:
            self.norm_tgt = nn.LayerNorm(d_model)

        self.attn = Attention(d_model=d_model, n_head=n_head, dropout_p=dropout_p, bias=bias)
        if self.d_feedforward > 0:
            self.linear1 = nn.Linear(d_model, d_feedforward)
            self.linear2 = nn.Linear(d_feedforward, d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None
            self.dropout2 = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None

    def forward(
        self,
        src: Tensor,
        src_padding_mask: Optional[Tensor] = None,
        tgt: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        decoder_tgt: Optional[Tensor] = None,
        decoder_tgt_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.decoder_self_attn:
            if self.norm_first:
                _s = self.norm_src(src)
                if decoder_tgt is None:
                    _s = self.attn_src(_s, tgt_padding_mask=src_padding_mask)[0]
                else:
                    decoder_tgt = self.norm_src(decoder_tgt)
                    _s = self.attn_src(_s, decoder_tgt, tgt_padding_mask=decoder_tgt_padding_mask)[
                        0
                    ]
                if self.dropout_src is None:
                    src = src + _s
                else:
                    src = src + self.dropout_src(_s)
            else:
                if decoder_tgt is None:
                    _s = self.attn_src(src, tgt_padding_mask=src_padding_mask)[0]
                else:
                    _s = self.attn_src(src, decoder_tgt, tgt_padding_mask=decoder_tgt_padding_mask)[
                        0
                    ]
                if self.dropout_src is None:
                    src = self.norm_src(src + _s)
                else:
                    src = self.norm_src(src + self.dropout_src(_s))

        if tgt is None:
            tgt_padding_mask = src_padding_mask

        if self.norm_first:
            src2 = self.norm1(src)
            if tgt is not None:
                tgt = self.norm_tgt(tgt)
        else:
            src2 = src

        src2, attn_weights = self.attn(
            src=src2,
            tgt=tgt,
            tgt_padding_mask=tgt_padding_mask,
            attn_mask=attn_mask,
            need_weights=need_weights,
        )

        if self.d_feedforward > 0:
            if self.dropout1 is None:
                src = src + src2
            else:
                src = src + self.dropout1(src2)

            if self.norm_first:
                src2 = self.norm2(src)
            else:
                src = self.norm1(src)
                src2 = src

            src2 = self.activation(self.linear1(src2))
            if self.dropout is None:
                src2 = self.linear2(src2)
            else:
                src2 = self.linear2(self.dropout(src2))

            if self.dropout2 is None:
                src = src + src2
            else:
                src = src + self.dropout2(src2)

            if not self.norm_first:
                src = self.norm2(src)
        else:
            src2 = self.activation(src2)
            if self.dropout is None:
                src = src + src2
            else:
                src = src + self.dropout(src2)
            if not self.norm_first:
                src = self.norm1(src)

        if src_padding_mask is not None:
            src.masked_fill_(src_padding_mask.unsqueeze(-1), 0.0)
            if need_weights:
                attn_weights.masked_fill_(src_padding_mask.unsqueeze(-1), 0.0)
        return src, attn_weights


class TransformerBlock(nn.Module):
    __constants__ = ["norm"]

    def __init__(
        self,
        d_model: int,
        n_head: int = 2,
        d_feedforward: int = 256,
        dropout_p: float = 0.1,
        activation: str = "relu",
        n_layer: int = 1,
        norm_first: bool = True,
        decoder_self_attn: bool = False,
        bias: bool = True,
        out_layernorm: bool = False,
    ) -> None:
        super(TransformerBlock, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerCrossAttention(
                    d_model=d_model,
                    n_head=n_head,
                    d_feedforward=d_feedforward,
                    dropout_p=dropout_p,
                    activation=activation,
                    norm_first=norm_first,
                    decoder_self_attn=decoder_self_attn,
                    bias=bias,
                )
                for _ in range(n_layer)
            ]
        )
        self.out_layernorm = nn.LayerNorm(d_model) if out_layernorm else None

    def forward(
        self,
        src: Tensor,
        src_padding_mask: Optional[Tensor] = None,
        tgt: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        decoder_tgt: Optional[Tensor] = None,
        decoder_tgt_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        attn_weights = None
        for mod in self.layers:
            src, attn_weights = mod(
                src=src,
                src_padding_mask=src_padding_mask,
                tgt=tgt,
                tgt_padding_mask=tgt_padding_mask,
                decoder_tgt=decoder_tgt,
                decoder_tgt_padding_mask=decoder_tgt_padding_mask,
                attn_mask=attn_mask,
                need_weights=need_weights,
            )
        if self.out_layernorm is not None:
            src = self.out_layernorm(src)
        return src, attn_weights


# ---------------------------------------------------------------------------
# src/models/modules/input_pe_encoder.py
# ---------------------------------------------------------------------------
class InputPeEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attr_dim: int,
        pe_dim: int,
        n_layer: int,
        mlp_dropout_p: Optional[float] = 0.1,
        mlp_use_layernorm: bool = True,
        pe_mode: str = "add",
    ) -> None:
        super().__init__()
        self.pe_mode = pe_mode
        if self.pe_mode == "input":
            mlp_in_dim = attr_dim + pe_dim
            mlp_out_dim = hidden_dim
        elif self.pe_mode == "cat":
            mlp_in_dim = attr_dim
            mlp_out_dim = hidden_dim - pe_dim
            assert mlp_out_dim >= 32, f"Make sure pe_dim is smaller than {hidden_dim - 32}!"
        elif self.pe_mode == "add":
            mlp_in_dim = attr_dim
            mlp_out_dim = hidden_dim
            assert pe_dim == hidden_dim, f"Make sure pe_dim equals to hidden_dim={hidden_dim}!"

        self.mlp = MLP(
            [mlp_in_dim] + [mlp_out_dim] * n_layer,
            dropout_p=mlp_dropout_p,
            use_layernorm=mlp_use_layernorm,
            end_layer_activation=False,
        )

    def forward(self, valid: Tensor, attr: Tensor, pe: Tensor) -> Tensor:
        if self.pe_mode == "input":
            x = self.mlp(torch.cat([attr, pe], dim=-1))
        elif self.pe_mode == "cat":
            x = self.mlp(attr)
            x = torch.cat([x, pe], dim=-1)
        elif self.pe_mode == "add":
            x = self.mlp(attr)
            x = x + pe
        feature = x.masked_fill(~valid.unsqueeze(-1), 0)
        return feature


# ---------------------------------------------------------------------------
# src/models/modules/map_encoder.py
# ---------------------------------------------------------------------------
class MapEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attr_dim: int,
        pe_dim: int,
        input_pe_encoder,
        tf_cfg,
        densetnt_vectornet: bool = False,
        pool_mode: str = "max",
        n_layer: int = 3,
        mlp_dropout_p: Optional[float] = 0.1,
        mlp_use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.densetnt_vectornet = densetnt_vectornet
        self.pool_mode = pool_mode
        self.input_pe_encoder = InputPeEncoder(
            hidden_dim=hidden_dim, attr_dim=attr_dim, pe_dim=pe_dim, **input_pe_encoder
        )
        if self.densetnt_vectornet:
            self.transformer_densetnt = TransformerBlock(n_layer=n_layer, **tf_cfg)
        else:
            mlp_layers: List[nn.Module] = []
            for _ in range(n_layer - 1):
                mlp_layers.append(
                    MLP(
                        [hidden_dim, hidden_dim // 2],
                        dropout_p=mlp_dropout_p,
                        use_layernorm=mlp_use_layernorm,
                    )
                )
            if tf_cfg["norm_first"]:
                end_layer_activation = False
            else:
                end_layer_activation = True
            mlp_layers.append(
                MLP(
                    [hidden_dim, hidden_dim // 2],
                    dropout_p=mlp_dropout_p,
                    use_layernorm=mlp_use_layernorm,
                    end_layer_activation=end_layer_activation,
                )
            )
            self.mlp_layers = nn.ModuleList(mlp_layers)

        self.transformer_self_attn = TransformerBlock(n_layer=1, **tf_cfg)

    def forward(self, map_valid: Tensor, map_attr: Tensor, map_pe: Tensor) -> Tuple[Tensor, Tensor]:
        n_scene, n_pl, n_node = map_valid.shape
        pl_feature = self.input_pe_encoder(map_valid, map_attr, map_pe)

        if self.densetnt_vectornet:
            pl_feature = pl_feature.flatten(0, 1)
            map_valid_f = map_valid.flatten(0, 1)
            pl_feature, _ = self.transformer_densetnt(
                src=pl_feature,
                src_padding_mask=~map_valid_f,
                tgt=pl_feature,
                tgt_padding_mask=~map_valid_f,
                need_weights=False,
            )
            hidden_dim = pl_feature.shape[-1]
            pl_feature = pl_feature.view(n_scene, n_pl, n_node, hidden_dim)
            map_valid = map_valid_f.view(n_scene, n_pl, n_node)
        else:
            for mlp in self.mlp_layers:
                feature_encoded = mlp(pl_feature, map_valid, float("-inf"))
                feature_pooled = feature_encoded.amax(dim=2, keepdim=True)
                pl_feature = torch.cat(
                    (feature_encoded, feature_pooled.expand(-1, -1, n_node, -1)), dim=-1
                )

        if self.pool_mode == "max":
            pl_feature = pl_feature.masked_fill(~map_valid.unsqueeze(-1), float("-inf"))
            pl_feature = pl_feature.amax(dim=2, keepdim=False)
        elif self.pool_mode == "first":
            pl_feature = pl_feature[:, :, 0]
        elif self.pool_mode == "mean":
            pl_feature = pl_feature.masked_fill(~map_valid.unsqueeze(-1), 0)
            pl_feature = pl_feature.sum(dim=2, keepdim=False)
            pl_feature = pl_feature / (
                map_valid.sum(dim=-1, keepdim=True) + torch.finfo(pl_feature.dtype).eps
            )

        pl_valid = map_valid.any(-1)
        pl_feature = pl_feature.masked_fill(~pl_valid.unsqueeze(-1), 0)

        pl_feature, _ = self.transformer_self_attn(
            src=pl_feature,
            src_padding_mask=~pl_valid,
            tgt=pl_feature,
            tgt_padding_mask=~pl_valid,
            need_weights=False,
        )
        return pl_feature, pl_valid


# ---------------------------------------------------------------------------
# src/models/modules/agent_interaction.py
# ---------------------------------------------------------------------------
class MultiAgentTF(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_layer: int,
        attn_to_map_aware_feature: bool,
        mask_self_agent: bool,
        detach_tgt: bool,
        tf_cfg,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mask_self_agent = mask_self_agent
        self.attn_mask = None
        self.detach_tgt = detach_tgt
        self.attn_to_map_aware_feature = attn_to_map_aware_feature
        self.transformer = TransformerBlock(n_layer=n_layer, **tf_cfg)

    def forward(
        self,
        as_feature_map_aware: Tensor,
        as_feature: Tensor,
        as_valid: Tensor,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        valid_dim = as_valid.dim()
        if valid_dim == 3:
            n_batch, n_step, n_agent = as_valid.shape
            as_feature_map_aware = as_feature_map_aware.flatten(start_dim=0, end_dim=1)
            as_feature = as_feature.flatten(start_dim=0, end_dim=1)
            as_valid = as_valid.flatten(start_dim=0, end_dim=1)
        elif valid_dim == 2:
            n_batch, n_agent = as_valid.shape

        x = as_feature_map_aware
        tgt_x = as_feature_map_aware if self.attn_to_map_aware_feature else as_feature
        if self.detach_tgt:
            tgt_x = tgt_x.detach()

        padding = ~as_valid
        if self.mask_self_agent:
            if self.attn_mask is None:
                self.attn_mask = torch.eye(n_agent, device=as_valid.device, dtype=torch.bool)

            invalid_batch = as_valid.sum(-1) == 1
            if invalid_batch.any():
                valid_batch = ~invalid_batch
                x_reduced = x[valid_batch]
                tgt_x_reduced = tgt_x[valid_batch]
                padding_reduced = padding[valid_batch]
                x_reduced, attn_weights_reduced = self.transformer(
                    src=x_reduced,
                    src_padding_mask=padding_reduced,
                    tgt=tgt_x_reduced,
                    tgt_padding_mask=padding_reduced,
                    need_weights=need_weights,
                    attn_mask=self.attn_mask,
                )
                x = x.masked_fill(valid_batch[:, None, None], 0.0)
                x[valid_batch] = x_reduced
                if need_weights:
                    n_batch, n_agent = as_valid.shape
                    attn_weights = torch.zeros(
                        [n_batch, n_agent, n_agent], device=x.device, dtype=x.dtype
                    )
                    attn_weights[valid_batch] = attn_weights_reduced
                else:
                    attn_weights = None
            else:
                x, attn_weights = self.transformer(
                    src=x,
                    src_padding_mask=padding,
                    tgt=tgt_x,
                    tgt_padding_mask=padding,
                    need_weights=need_weights,
                    attn_mask=self.attn_mask,
                )
        else:
            x, attn_weights = self.transformer(
                src=x,
                src_padding_mask=padding,
                tgt=tgt_x,
                tgt_padding_mask=padding,
                need_weights=need_weights,
                attn_mask=None,
            )

        if valid_dim == 3:
            x = x.view([n_batch, n_step, n_agent, self.hidden_dim])
        return x, attn_weights


# ---------------------------------------------------------------------------
# src/models/modules/agent_temporal.py
# ---------------------------------------------------------------------------
class TemporalAggregate(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def forward(self, x: Tensor, valid: Tensor) -> Tuple[Tensor, Tensor]:
        if self.mode == "max":
            x_aggregated = x.amax(1)
        elif self.mode == "last":
            x_aggregated = x[:, -1]
        elif self.mode == "max_valid":
            x_aggregated = x.masked_fill(~valid.unsqueeze(-1), -1e3).amax(1)
        elif self.mode == "last_valid":
            n_batch, n_step, n_agent = valid.shape
            idx_last_valid = n_step - 1 - torch.max(valid.flip(1), dim=1)[1]
            x_aggregated = x[
                torch.arange(n_batch).unsqueeze(1),
                idx_last_valid,
                torch.arange(n_agent).unsqueeze(0),
            ]
        elif self.mode == "mean_valid":
            valid_sum = valid.sum(1) + torch.finfo(x.dtype).eps
            x_aggregated = x.sum(1) / valid_sum.unsqueeze(-1)

        valid_aggregated = torch.any(valid, axis=1)
        return x_aggregated.masked_fill(~valid_aggregated.unsqueeze(-1), 0), valid_aggregated


class MultiAgentGRULoop(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, dropout=dropout)

    def forward(self, x: Tensor, valid: Tensor, h: Optional[Any] = None) -> Tuple[Tensor, Tensor]:
        n_batch = valid.shape[0]
        n_agent = valid.shape[-1]
        if h is None:
            h = torch.zeros((self.num_layers, n_batch * n_agent, self.hidden_dim), device=x.device)

        if valid.dim() == 3:
            n_step = valid.shape[1]
            x_1 = []
            x = x.transpose(0, 1).flatten(start_dim=1, end_dim=2)
            invalid = ~valid.transpose(0, 1).flatten(start_dim=1, end_dim=2).unsqueeze(-1)
            for k in range(n_step):
                x_out, h = self.rnn(x[[k]], h)
                h = h.masked_fill(invalid[[k]], 0.0)
                x_1.append(x_out)
            x_1 = torch.cat(x_1, dim=0)
            x_1 = (
                x_1.masked_fill(invalid, 0.0)
                .view(n_step, n_batch, n_agent, self.hidden_dim)
                .transpose(0, 1)
            )
            h_1 = None
        elif valid.dim() == 2:
            input = x.flatten(start_dim=0, end_dim=1).unsqueeze(0)
            input, h = self.rnn(input, h)
            invalid = ~valid.flatten(start_dim=0, end_dim=1).unsqueeze(-1)
            h_1 = h.masked_fill(invalid.unsqueeze(0), 0.0)
            x_1 = input[0].masked_fill(invalid, 0.0).view(n_batch, n_agent, self.hidden_dim)
        return x_1, h_1


# ---------------------------------------------------------------------------
# src/models/modules/distributions.py
# ---------------------------------------------------------------------------
class MyDist:
    def __init__(self, *args, **kwargs) -> None:
        self.distribution = None

    def log_prob(self, sample: Tensor) -> Tensor:
        return self.distribution.log_prob(sample)

    def sample(self, deterministic: Union[bool, Tensor]) -> Tensor:
        if type(deterministic) is Tensor:
            det_sample = self.distribution.mean
            rnd_sample = self.distribution.rsample()
            sample = det_sample.masked_fill(
                ~deterministic.unsqueeze(-1), 0
            ) + rnd_sample.masked_fill(deterministic.unsqueeze(-1), 0)
        else:
            if deterministic:
                sample = self.distribution.mean
            else:
                sample = self.distribution.rsample()
        return sample


class DiagGaussian(MyDist):
    def __init__(self, mean: Tensor, log_std: Tensor, valid: Optional[Tensor] = None) -> None:
        super().__init__()
        self.mean = mean
        self.valid = valid
        self.distribution = torch.distributions.Independent(
            torch.distributions.Normal(self.mean, log_std.exp()), 1
        )
        self.stddev = self.distribution.stddev
        self.covariance_matrix = torch.diag_embed(self.distribution.variance)


class DummyLatent(MyDist):
    def __init__(self, x, valid) -> None:
        super().__init__()
        self._logp = torch.zeros_like(x[..., 0])
        self._sample = torch.zeros_like(x)
        self.valid = valid

    def log_prob(self, *args, **kwargs) -> Tensor:
        return self._logp

    def sample(self, *args, **kwargs) -> Tensor:
        return self._sample


class MultiCategorical(MyDist):
    def __init__(self, probs: Tensor, valid: Optional[Tensor] = None):
        super().__init__()
        self.probs = probs
        self.distribution = torch.distributions.Independent(
            torch.distributions.OneHotCategoricalStraightThrough(probs=self.probs), 1
        )
        self.n_cat = self.probs.shape[-2]
        self.n_class = self.probs.shape[-1]
        self._dtype = self.probs.dtype
        self.valid = valid

    def log_prob(self, sample: Tensor) -> Tensor:
        return self.distribution.log_prob(sample.view(*sample.shape[:-1], self.n_cat, self.n_class))

    def sample(self, deterministic: Union[bool, Tensor]) -> Tensor:
        if type(deterministic) is Tensor:
            det_sample = (
                F.one_hot(self.distribution.base_dist.probs.argmax(-1), num_classes=self.n_class)
                .type(self._dtype)
                .flatten(start_dim=-2, end_dim=-1)
            )
            rnd_sample = self.distribution.rsample().flatten(start_dim=-2, end_dim=-1)
            sample = det_sample.masked_fill(
                ~deterministic.unsqueeze(-1), 0
            ) + rnd_sample.masked_fill(deterministic.unsqueeze(-1), 0)
        else:
            if deterministic:
                sample = (
                    F.one_hot(
                        self.distribution.base_dist.probs.argmax(-1), num_classes=self.n_class
                    )
                    .type(self._dtype)
                    .flatten(start_dim=-2, end_dim=-1)
                )
            else:
                sample = self.distribution.rsample().flatten(start_dim=-2, end_dim=-1)
        return sample


class DestCategorical(MyDist):
    def __init__(self, probs=None, logits=None, valid=None):
        super().__init__()
        if probs is None:
            assert logits is not None
            self.distribution = torch.distributions.Categorical(logits=logits)
            self.probs = self.distribution.probs
        else:
            self.distribution = torch.distributions.Categorical(probs=probs)
            self.probs = self.distribution.probs
        self.valid = valid

    def log_prob(self, sample: Tensor) -> Tensor:
        return self.distribution.log_prob(sample)

    def sample(self, deterministic):
        if type(deterministic) is Tensor:
            det_sample = self.distribution.probs.argmax(-1)
            rnd_sample = self.distribution.sample()
            sample = det_sample.masked_fill(~deterministic, 0) + rnd_sample.masked_fill(
                deterministic, 0
            )
        else:
            if deterministic:
                sample = self.distribution.probs.argmax(-1)
            else:
                sample = self.distribution.sample()
        return sample


# ---------------------------------------------------------------------------
# src/models/modules/add_latent_goal.py
# ---------------------------------------------------------------------------
class AddLatentGoal(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        dummy: bool,
        mode: str,
        n_layer_mlp_in: int,
        n_layer_mlp_out: int,
        mlp_in_cfg,
        mlp_out_cfg,
        res_cat: bool = False,
        res_add: bool = False,
    ) -> None:
        super().__init__()
        assert mode in ["add", "mul", "cat"]
        self.mode = mode
        self.dummy = dummy
        self.res_cat = res_cat
        self.res_add = res_add

        if not self.dummy:
            self.mlp_in = MLP([in_dim] + [hidden_dim] * n_layer_mlp_in, **mlp_in_cfg)
            if self.mode == "cat":
                out_dim = hidden_dim * 2
            else:
                out_dim = hidden_dim
            self.mlp_out = MLP([out_dim] + [hidden_dim] * n_layer_mlp_out, **mlp_out_cfg)
            if self.res_cat:
                self.mlp_res_cat = MLP(
                    [hidden_dim * 2 + in_dim] + [hidden_dim] * n_layer_mlp_out, **mlp_out_cfg
                )

    def forward(
        self, x: Tensor, x_valid: Tensor, z: Optional[Tensor], z_valid: Optional[Tensor]
    ) -> Tensor:
        if self.dummy:
            h = x
        else:
            z = self.mlp_in(z, z_valid)
            if self.mode == "add":
                h = x + z
            elif self.mode == "mul":
                h = x * z
            else:
                h = torch.cat([x, z], dim=-1)
            h = self.mlp_out(h)
            if self.res_cat:
                h = self.mlp_res_cat(torch.cat([x, h, z], dim=-1))
            h = h.masked_fill(~z_valid.unsqueeze(-1), 0)
            if self.res_add:
                h = h + x
            else:
                h = h + x.masked_fill(z_valid.unsqueeze(-1), 0)
        return h.masked_fill(~x_valid.unsqueeze(-1), 0)


# ---------------------------------------------------------------------------
# src/models/goal_manager.py (GoalManager + DestPredictor used by "dest" mode)
# ---------------------------------------------------------------------------
class DestPredictor(nn.Module):
    def __init__(
        self,
        tf_cfg,
        mode: str = "mlp",
        n_layer_gru: int = -1,
        use_layernorm: bool = True,
        res_add_gru: bool = True,
        detach_features: bool = True,
    ) -> None:
        super().__init__()
        assert mode in ["transformer", "transformer_aggr", "mlp", "attn"]
        self.mode = mode
        self.detach_features = detach_features
        self.res_add_gru = res_add_gru
        hidden_dim = tf_cfg["d_model"]

        if n_layer_gru > 0:
            self.gru_as = MultiAgentGRULoop(
                hidden_dim, num_layers=n_layer_gru, dropout=tf_cfg["dropout_p"]
            )
        else:
            self.gru_as = None
        self.agr_as = TemporalAggregate("last_valid")

        if self.mode in ("transformer", "transformer_aggr"):
            self.transformer_pl2as = TransformerBlock(hidden_dim, n_layer=1, **tf_cfg)
            self.mlp = MLP(
                [hidden_dim, hidden_dim, 1], end_layer_activation=False, use_layernorm=use_layernorm
            )
        elif self.mode == "mlp":
            self.mlp = MLP(
                [hidden_dim * 2, hidden_dim, hidden_dim, 1],
                end_layer_activation=False,
                use_layernorm=use_layernorm,
            )
        elif self.mode == "attn":
            self.n_head = tf_cfg["n_head"]
            self.attn = Attention(
                d_model=tf_cfg["d_model"],
                n_head=tf_cfg["n_head"],
                dropout_p=tf_cfg["dropout_p"],
                bias=tf_cfg["bias"],
            )

    def forward(
        self,
        agent_type: Tensor,
        map_type: Tensor,
        agent_state: Tensor,
        agent_feature: Tensor,
        agent_feature_valid: Tensor,
        map_feature: Tensor,
        map_feature_valid: Tensor,
        tl_feature: Optional[Tensor] = None,
        tl_feature_valid: Optional[Tensor] = None,
    ) -> "DestCategorical":
        if self.detach_features:
            agent_feature = agent_feature.detach()
            map_feature = map_feature.detach()

        map_type_mask = ~(map_feature_valid & (map_type[:, :, :5].any(-1)))
        attn_mask_veh = agent_type[:, :, [0]] & map_type[:, :, 3].unsqueeze(1)
        attn_mask_ped = agent_type[:, :, [1]] & map_type[:, :, :4].any(-1).unsqueeze(1)
        attn_mask_cyc = agent_type[:, :, [2]] & map_type[:, :, :3].any(-1).unsqueeze(1)
        attn_mask = attn_mask_veh | attn_mask_ped | attn_mask_cyc

        n_scene, n_pl, hidden_dim = map_feature.shape
        n_agent = agent_feature_valid.shape[2]
        dist_valid = agent_feature_valid.any(1)

        if self.mode == "mlp":
            if self.gru_as is None:
                tgt = agent_feature
            else:
                tgt, _ = self.gru_as(agent_feature, agent_feature_valid)
                if self.res_add_gru:
                    tgt = tgt + agent_feature
            tgt, tgt_valid = self.agr_as(tgt, agent_feature_valid)
            tgt = tgt.unsqueeze(2).expand(-1, -1, n_pl, -1)
            src = map_feature.unsqueeze(1).expand(-1, n_agent, -1, -1)
            logits = self.mlp(torch.cat([src, tgt], dim=-1)).squeeze(-1)
        else:
            raise NotImplementedError("staging module only wires mode='mlp' (recipe default)")

        logits = logits.masked_fill(map_type_mask.unsqueeze(1), float("-inf"))
        logits = logits.masked_fill(attn_mask, float("-inf"))
        logits = logits.masked_fill(~dist_valid.unsqueeze(-1), 0)
        logits = logits.masked_fill((logits == float("-inf")).all(-1).unsqueeze(-1), 0)
        return DestCategorical(logits=logits, valid=dist_valid)


class GoalManager(nn.Module):
    def __init__(
        self,
        tf_cfg,
        goal_predictor,
        goal_attr_mode: str,
        goal_in_local: bool,
        dest_detach_map_feature: bool,
        disable_if_reached: bool,
    ) -> None:
        super().__init__()
        self.goal_attr_mode = goal_attr_mode
        self.goal_in_local = goal_in_local
        self.dest_detach_map_feature = dest_detach_map_feature
        self.disable_if_reached = disable_if_reached
        hidden_dim = tf_cfg["d_model"]

        self.update_goal = False
        if self.goal_attr_mode == "dummy":
            self.dummy = True
            self.out_dim = -1
            self.goal_predictor = None
        elif self.goal_attr_mode == "dest":
            self.dummy = False
            self.out_dim = hidden_dim
            self.goal_predictor = DestPredictor(tf_cfg=tf_cfg, **goal_predictor)
        else:
            raise NotImplementedError(
                "staging module only wires goal_attr_mode='dest' (recipe default)"
            )

    def pred_goal(self, *args, **kwargs) -> Optional["MyDist"]:
        if self.goal_predictor is None:
            return None
        return self.goal_predictor(*args, **kwargs)

    def get_goal_feature(self, goal: Tensor, as_state: Tensor, map_feature: Tensor) -> Tensor:
        if as_state.dim() == 4:
            n_step = as_state.shape[1]
            goal = goal.unsqueeze(1).expand(-1, n_step, -1)
        if self.dest_detach_map_feature:
            map_feature = map_feature.detach()
        goal_feature = self._get_dest_feature(goal, map_feature)
        return goal_feature

    @staticmethod
    def _get_dest_feature(dest: Tensor, map_feature: Tensor) -> Tensor:
        batch_index = torch.arange(dest.shape[0]).unsqueeze(1)
        if dest.dim() == 3:
            dest_feature = []
            for k in range(dest.shape[1]):
                dest_feature.append(map_feature[batch_index, dest[:, k, :]])
            dest_feature = torch.stack(dest_feature, dim=1)
        else:
            dest_feature = map_feature[batch_index, dest]
        return dest_feature

    def disable_goal_reached(self, goal_valid, agent_valid, dest_reached, goal_reached):
        if goal_valid is not None:
            goal_valid = goal_valid & agent_valid
        if self.disable_if_reached:
            goal_valid = goal_valid & (~dest_reached)
        return goal_valid


# ---------------------------------------------------------------------------
# src/models/latent_encoder.py
# ---------------------------------------------------------------------------
class DistEncoder(nn.Module):
    def __init__(
        self,
        dist_type: str,
        hidden_dim: int,
        out_dim: int,
        use_layernorm: bool,
        log_std: float = 0.0,
        n_cat: int = 1,
    ) -> None:
        super().__init__()
        self.dist_type = dist_type
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.skip_forward = False

        if dist_type == "dummy":
            self.skip_forward = True
        elif dist_type == "std_gaus":
            self.log_std = nn.Parameter(log_std * torch.ones(out_dim), requires_grad=False)
            self.skip_forward = True
        elif dist_type == "diag_gaus":
            self.mlp_mean = MLP(
                [hidden_dim, hidden_dim, out_dim],
                end_layer_activation=False,
                use_layernorm=use_layernorm,
            )
            if log_std is None:
                self.log_std = None
                self.mlp_log_std = MLP(
                    [hidden_dim, hidden_dim, out_dim],
                    end_layer_activation=False,
                    use_layernorm=use_layernorm,
                )
            else:
                self.log_std = nn.Parameter(log_std * torch.ones(out_dim), requires_grad=True)
        elif dist_type == "cat":
            assert out_dim % n_cat == 0
            self.n_cat = n_cat
            self.n_class = out_dim // self.n_cat
            self.mlp_logits = MLP(
                [hidden_dim, hidden_dim, out_dim],
                end_layer_activation=False,
                use_layernorm=use_layernorm,
            )

    def forward(self, x: Tensor, valid: Tensor) -> "MyDist":
        if self.dist_type == "dummy":
            out_dist = DummyLatent(x, valid)
        elif self.dist_type == "std_gaus":
            out_dist = DiagGaussian(
                torch.zeros([*valid.shape, self.out_dim], dtype=x.dtype, device=x.device),
                self.log_std,
                valid=valid,
            )
        elif self.dist_type == "diag_gaus":
            if self.log_std is None:
                out_dist = DiagGaussian(
                    self.mlp_mean(x, valid), self.mlp_log_std(x, valid), valid=valid
                )
            else:
                out_dist = DiagGaussian(self.mlp_mean(x, valid), self.log_std, valid=valid)
        elif self.dist_type == "cat":
            logits = self.mlp_logits(x, valid).view(*valid.shape, self.n_cat, self.n_class)
            out_dist = MultiCategorical(nn.functional.softmax(logits, -1), valid=valid)
        return out_dist


class LatentEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        temporal_down_sample_rate: int,
        shared_post_prior_net: bool,
        shared_transformer_as: bool,
        latent_prior,
        latent_post,
        tf_cfg,
        interaction_first: bool,
        transformer_as2pl: TransformerBlock,
        transformer_as2tl: TransformerBlock,
        agent_temporal,
        agent_interaction,
        temporal_aggregate,
    ):
        super().__init__()
        hidden_dim = tf_cfg["d_model"]
        self.out_dim = latent_dim
        self.dummy = latent_post["dist_type"] == "dummy"
        self.temporal_down_sample_rate = temporal_down_sample_rate
        self.interaction_first = interaction_first

        if shared_transformer_as:
            self.transformer_as2pl = transformer_as2pl
            self.transformer_as2tl = transformer_as2tl
        else:
            self.transformer_as2pl = TransformerBlock(
                n_layer=len(transformer_as2pl.layers), **tf_cfg
            )
            self.transformer_as2tl = TransformerBlock(
                n_layer=len(transformer_as2tl.layers), **tf_cfg
            )

        self.latent_prior_dist = DistEncoder(
            **latent_prior, hidden_dim=hidden_dim, out_dim=latent_dim
        )
        self.latent_post_dist = DistEncoder(
            **latent_post, hidden_dim=hidden_dim, out_dim=latent_dim
        )

        if self.latent_post_dist.skip_forward:
            self.temporal_aggregate = None
            self.agent_temporal_post = None
            self.agent_interaction_post = None
            self.agent_temporal_prior = None
            self.agent_interaction_prior = None
        else:
            self.temporal_aggregate = TemporalAggregate(**temporal_aggregate)
            self.agent_temporal_post = MultiAgentGRULoop(hidden_dim=hidden_dim, **agent_temporal)
            self.agent_interaction_post = MultiAgentTF(
                hidden_dim=hidden_dim, tf_cfg=tf_cfg, **agent_interaction
            )
            if self.latent_prior_dist.skip_forward:
                self.agent_temporal_prior = None
                self.agent_interaction_prior = None
            elif shared_post_prior_net:
                self.agent_temporal_prior = self.agent_temporal_post
                self.agent_interaction_prior = self.agent_interaction_post
            else:
                self.agent_temporal_prior = MultiAgentGRULoop(
                    hidden_dim=hidden_dim, **agent_temporal
                )
                self.agent_interaction_prior = MultiAgentTF(
                    hidden_dim=hidden_dim, tf_cfg=tf_cfg, **agent_interaction
                )

    def forward(
        self,
        agent_feature: Tensor,
        agent_feature_valid: Tensor,
        map_feature: Tensor,
        map_feature_valid: Tensor,
        tl_feature: Optional[Tensor] = None,
        tl_feature_valid: Optional[Tensor] = None,
        posterior: bool = False,
    ) -> "MyDist":
        if posterior and self.latent_post_dist.skip_forward:
            return self.latent_post_dist(agent_feature[:, 0], agent_feature_valid.any(1))
        elif (not posterior) and self.latent_prior_dist.skip_forward:
            return self.latent_prior_dist(agent_feature[:, 0], agent_feature_valid.any(1))
        else:
            if self.temporal_down_sample_rate > 1:
                assert (agent_feature_valid.shape[1] - 1) % self.temporal_down_sample_rate == 0
                agent_feature_valid = agent_feature_valid[:, :: self.temporal_down_sample_rate]
                agent_feature = agent_feature[:, :: self.temporal_down_sample_rate]
                tl_feature_valid = tl_feature_valid[:, :: self.temporal_down_sample_rate]
                tl_feature = tl_feature[:, :: self.temporal_down_sample_rate]

            as_feature_map_aware = agent_feature
            as_shape = as_feature_map_aware.shape
            as_feature_map_aware, _ = self.transformer_as2pl(
                src=as_feature_map_aware.flatten(1, 2),
                src_padding_mask=~agent_feature_valid.flatten(1, 2),
                tgt=map_feature,
                tgt_padding_mask=~map_feature_valid,
            )
            as_feature_map_aware = as_feature_map_aware.view(as_shape)
            as_feature_map_aware, _ = self.transformer_as2tl(
                src=as_feature_map_aware.flatten(0, 1),
                src_padding_mask=~agent_feature_valid.flatten(0, 1),
                tgt=tl_feature.flatten(0, 1),
                tgt_padding_mask=~tl_feature_valid.flatten(0, 1),
            )
            as_feature_map_aware = as_feature_map_aware.view(as_shape)

            if posterior:
                if self.interaction_first:
                    latent_feature, _ = self.agent_interaction_post(
                        as_feature_map_aware, agent_feature, agent_feature_valid
                    )
                    latent_feature, _ = self.agent_temporal_post(
                        latent_feature, agent_feature_valid
                    )
                else:
                    latent_feature, _ = self.agent_temporal_post(
                        as_feature_map_aware, agent_feature_valid
                    )
                    latent_feature, _ = self.agent_interaction_post(
                        latent_feature, agent_feature, agent_feature_valid
                    )
                latent_feature, latent_valid = self.temporal_aggregate(
                    latent_feature, agent_feature_valid
                )
                return self.latent_post_dist(latent_feature, latent_valid)
            else:
                if self.interaction_first:
                    latent_feature, _ = self.agent_interaction_prior(
                        as_feature_map_aware, agent_feature, agent_feature_valid
                    )
                    latent_feature, _ = self.agent_temporal_prior(
                        latent_feature, agent_feature_valid
                    )
                else:
                    latent_feature, _ = self.agent_temporal_prior(
                        as_feature_map_aware, agent_feature_valid
                    )
                    latent_feature, _ = self.agent_interaction_prior(
                        latent_feature, agent_feature, agent_feature_valid
                    )
                latent_feature, latent_valid = self.temporal_aggregate(
                    latent_feature, agent_feature_valid
                )
                return self.latent_prior_dist(latent_feature, latent_valid)


# ---------------------------------------------------------------------------
# src/models/traffic_bots.py  (the real top-level TrafficBots nn.Module)
# ---------------------------------------------------------------------------
class TrafficBots(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        agent_attr_dim: int,
        map_pe_dim: int,
        tl_pe_dim: int,
        agent_pe_dim: int,
        map_encoder,
        input_pe_encoder,
        goal_manager,
        latent_encoder,
        tf_cfg,
        n_layer_tf_as2pl: int,
        n_layer_tf_as2tl: int,
        n_step_hist: int,
        n_pl_node: int,
        temporal_aggregate,
        agent_temporal,
        agent_interaction,
        add_latent,
        add_goal,
        interaction_first: bool,
        add_goal_latent_first: bool,
        resample_latent: bool,
        n_layer_final_mlp: int,
        final_mlp,
    ):
        super().__init__()
        self.resample_latent = resample_latent
        self.interaction_first = interaction_first
        self.add_goal_latent_first = add_goal_latent_first

        self.map_encoder = MapEncoder(
            hidden_dim=hidden_dim,
            attr_dim=map_attr_dim,
            pe_dim=map_pe_dim,
            input_pe_encoder=input_pe_encoder,
            tf_cfg=tf_cfg,
            **map_encoder,
        )
        self.tl_encoder = InputPeEncoder(
            hidden_dim=hidden_dim, attr_dim=tl_attr_dim, pe_dim=tl_pe_dim, **input_pe_encoder
        )
        self.agent_encoder = InputPeEncoder(
            hidden_dim=hidden_dim, attr_dim=agent_attr_dim, pe_dim=agent_pe_dim, **input_pe_encoder
        )
        self.transformer_as2pl = TransformerBlock(n_layer=n_layer_tf_as2pl, **tf_cfg)
        self.transformer_as2tl = TransformerBlock(n_layer=n_layer_tf_as2tl, **tf_cfg)
        self.goal_manager = GoalManager(tf_cfg=tf_cfg, **goal_manager)
        self.latent_encoder = LatentEncoder(
            tf_cfg=tf_cfg,
            interaction_first=interaction_first,
            transformer_as2pl=self.transformer_as2pl,
            transformer_as2tl=self.transformer_as2tl,
            agent_temporal=agent_temporal,
            agent_interaction=agent_interaction,
            temporal_aggregate=temporal_aggregate,
            **latent_encoder,
        )
        self.agent_temporal = MultiAgentGRULoop(hidden_dim=hidden_dim, **agent_temporal)
        self.agent_interaction = MultiAgentTF(
            hidden_dim=hidden_dim, tf_cfg=tf_cfg, **agent_interaction
        )
        self.temporal_aggregate = TemporalAggregate(**temporal_aggregate)
        self.add_goal = AddLatentGoal(
            hidden_dim=hidden_dim,
            in_dim=self.goal_manager.out_dim,
            dummy=self.goal_manager.dummy,
            **add_goal,
        )
        self.add_latent = AddLatentGoal(
            hidden_dim=hidden_dim,
            in_dim=self.latent_encoder.out_dim,
            dummy=self.latent_encoder.dummy,
            **add_latent,
        )

        if n_layer_final_mlp > 0:
            self.final_mlp = MLP([hidden_dim] * (n_layer_final_mlp + 1), **final_mlp)
        else:
            self.final_mlp = None

    def encode_input_features(
        self,
        agent_valid,
        agent_attr,
        agent_pe,
        agent_pos,
        map_valid,
        map_attr,
        map_pe,
        map_pos,
        tl_valid,
        tl_attr,
        tl_pe,
        tl_pos,
    ) -> Dict[str, Tensor]:
        feature_dict = {"agent_feature_valid": agent_valid, "tl_feature_valid": tl_valid}
        feature_dict["map_feature"], feature_dict["map_feature_valid"] = self.map_encoder(
            map_valid, map_attr, map_pe
        )
        feature_dict["agent_feature"] = self.agent_encoder(agent_valid, agent_attr, agent_pe)
        feature_dict["tl_feature"] = self.tl_encoder(tl_valid, tl_attr, tl_pe)
        return feature_dict

    def init(self, latent: "MyDist", deterministic) -> None:
        self.latent = latent
        self.deterministic = deterministic
        self.hidden = None
        self.latent_sample = None
        self.latent_logp = None

    def forward(
        self,
        agent_valid: Tensor,
        agent_feature: Tensor,
        map_valid: Tensor,
        map_feature: Tensor,
        tl_valid: Tensor,
        tl_feature: Tensor,
        goal_valid: Optional[Tensor],
        goal_feature: Optional[Tensor],
        need_weights: bool = False,
    ):
        if self.resample_latent or (self.latent_sample is None):
            self.latent_sample = self.latent.sample(self.deterministic)
            self.latent_logp = self.latent.log_prob(self.latent_sample.detach())

        policy_feature = agent_feature

        policy_feature, attn_pl = self.transformer_as2pl(
            src=policy_feature,
            src_padding_mask=~agent_valid,
            tgt=map_feature,
            tgt_padding_mask=~map_valid,
            need_weights=need_weights,
        )
        policy_feature, attn_tl = self.transformer_as2tl(
            src=policy_feature,
            src_padding_mask=~agent_valid,
            tgt=tl_feature,
            tgt_padding_mask=~tl_valid,
            need_weights=need_weights,
        )

        if self.add_goal_latent_first:
            policy_feature = self.add_goal(policy_feature, agent_valid, goal_feature, goal_valid)
            policy_feature = self.add_latent(
                policy_feature, agent_valid, self.latent_sample, agent_valid
            )

        if self.interaction_first:
            policy_feature, attn_agent = self.agent_interaction(
                policy_feature, agent_feature, agent_valid, need_weights=need_weights
            )
            policy_feature, self.hidden = self.agent_temporal(
                policy_feature, agent_valid, self.hidden
            )
        else:
            policy_feature, self.hidden = self.agent_temporal(
                policy_feature, agent_valid, self.hidden
            )
            policy_feature, attn_agent = self.agent_interaction(
                policy_feature, agent_feature, agent_valid, need_weights=need_weights
            )

        if not self.add_goal_latent_first:
            policy_feature = self.add_goal(policy_feature, agent_valid, goal_feature, goal_valid)
            policy_feature = self.add_latent(
                policy_feature, agent_valid, self.latent_sample, agent_valid
            )

        if self.final_mlp is not None:
            policy_feature = self.final_mlp(policy_feature, agent_valid)

        return policy_feature, self.latent_logp, attn_pl, attn_tl, attn_agent


# ---------------------------------------------------------------------------
# Staging wrapper: one autoregressive TrafficBots step over a tiny synthetic
# Waymo-Open-Motion-Dataset-style scene-centric batch. Mirrors the call
# order used by src/pl_modules/waymo_motion.py (encode -> goal -> latent ->
# init -> single policy step).
# ---------------------------------------------------------------------------
class TrafficBotsStep(nn.Module):
    """Wraps the real TrafficBots model + goal/latent pipeline for one traced step."""

    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.model = TrafficBots(**model_cfg)

    def forward(
        self,
        agent_valid: Tensor,
        agent_attr: Tensor,
        agent_pe: Tensor,
        agent_pos: Tensor,
        agent_type: Tensor,
        agent_state: Tensor,
        map_valid: Tensor,
        map_attr: Tensor,
        map_pe: Tensor,
        map_pos: Tensor,
        map_type: Tensor,
        tl_valid: Tensor,
        tl_attr: Tensor,
        tl_pe: Tensor,
        tl_pos: Tensor,
    ) -> Tensor:
        feat = self.model.encode_input_features(
            agent_valid=agent_valid,
            agent_attr=agent_attr,
            agent_pe=agent_pe,
            agent_pos=agent_pos,
            map_valid=map_valid,
            map_attr=map_attr,
            map_pe=map_pe,
            map_pos=map_pos,
            tl_valid=tl_valid,
            tl_attr=tl_attr,
            tl_pe=tl_pe,
            tl_pos=tl_pos,
        )

        # ! goal (dest classification over map polylines, current-step agent/tl features)
        goal_dist = self.model.goal_manager.pred_goal(
            agent_type=agent_type,
            map_type=map_type,
            agent_state=agent_state,
            agent_feature=feat["agent_feature"],
            agent_feature_valid=feat["agent_feature_valid"],
            map_feature=feat["map_feature"],
            map_feature_valid=feat["map_feature_valid"],
            tl_feature=feat["tl_feature"],
            tl_feature_valid=feat["tl_feature_valid"],
        )
        goal_sample = goal_dist.sample(True)  # deterministic dest index, [n_scene, n_agent]
        goal_feature = self.model.goal_manager.get_goal_feature(
            goal_sample, agent_state[:, -1], feat["map_feature"]
        )  # [n_scene, n_agent, hidden_dim]
        goal_valid = goal_dist.valid

        # ! latent (prior branch -- no ground-truth future available at inference)
        latent_dist = self.model.latent_encoder(
            agent_feature=feat["agent_feature"],
            agent_feature_valid=feat["agent_feature_valid"],
            map_feature=feat["map_feature"],
            map_feature_valid=feat["map_feature_valid"],
            tl_feature=feat["tl_feature"],
            tl_feature_valid=feat["tl_feature_valid"],
            posterior=False,
        )

        # ! one autoregressive policy step (current-step agent/map/tl slice)
        self.model.init(latent_dist, deterministic=True)
        policy_feature, latent_logp, attn_pl, attn_tl, attn_agent = self.model(
            agent_valid=feat["agent_feature_valid"][:, -1],
            agent_feature=feat["agent_feature"][:, -1],
            map_valid=feat["map_feature_valid"],
            map_feature=feat["map_feature"],
            tl_valid=feat["tl_feature_valid"][:, -1],
            tl_feature=feat["tl_feature"][:, -1],
            goal_valid=goal_valid,
            goal_feature=goal_feature,
            need_weights=False,
        )
        return policy_feature


def _default_model_cfg() -> Dict[str, Any]:
    """Real hyperparams straight from configs/model/traffic_bots.yaml (hidden_dim shrunk to 16
    for a fast trace; every nested block is the real DictConfig sub-tree from the repo)."""
    hidden_dim = 40
    tf_cfg = {
        "d_model": hidden_dim,
        "n_head": 2,
        "dropout_p": 0.0,
        "norm_first": True,
        "bias": True,
        "activation": "relu",
        "d_feedforward": hidden_dim,
        "out_layernorm": False,
    }
    input_pe_encoder = {
        "pe_mode": "cat",
        "n_layer": 2,
        "mlp_dropout_p": None,
        "mlp_use_layernorm": False,
    }
    cfg = OmegaConf.create(
        {
            "hidden_dim": hidden_dim,
            "map_attr_dim": 11,
            "tl_attr_dim": 5,
            "agent_attr_dim": 3,
            "map_pe_dim": 8,
            "tl_pe_dim": 8,
            "agent_pe_dim": 8,
            "map_encoder": {
                "pool_mode": "max",
                "densetnt_vectornet": True,
                "n_layer": 2,
                "mlp_dropout_p": None,
                "mlp_use_layernorm": False,
            },
            "input_pe_encoder": input_pe_encoder,
            "goal_manager": {
                "disable_if_reached": True,
                "goal_predictor": {
                    "mode": "mlp",
                    "n_layer_gru": 1,
                    "use_layernorm": True,
                    "res_add_gru": True,
                    "detach_features": True,
                },
                "goal_attr_mode": "dest",
                "goal_in_local": True,
                "dest_detach_map_feature": False,
            },
            "latent_encoder": {
                "latent_dim": 8,
                "temporal_down_sample_rate": 1,
                "shared_post_prior_net": False,
                "shared_transformer_as": True,
                "latent_prior": {
                    "dist_type": "diag_gaus",
                    "n_cat": 1,
                    "log_std": -1,
                    "use_layernorm": False,
                },
                "latent_post": {
                    "dist_type": "diag_gaus",
                    "n_cat": 1,
                    "log_std": -1,
                    "use_layernorm": False,
                },
            },
            "tf_cfg": tf_cfg,
            "n_layer_tf_as2pl": 1,
            "n_layer_tf_as2tl": 1,
            "n_step_hist": 3,
            "n_pl_node": 4,
            "temporal_aggregate": {"mode": "max_valid"},
            "agent_temporal": {"num_layers": 1, "dropout": 0.0},
            "agent_interaction": {
                "n_layer": 1,
                "mask_self_agent": True,
                "detach_tgt": False,
                "attn_to_map_aware_feature": True,
            },
            "add_latent": {
                "mode": "cat",
                "res_cat": False,
                "res_add": False,
                "n_layer_mlp_in": 1,
                "n_layer_mlp_out": 1,
                "mlp_in_cfg": {"dropout_p": None, "use_layernorm": False},
                "mlp_out_cfg": {"dropout_p": None, "use_layernorm": False},
            },
            "add_goal": {
                "mode": "cat",
                "res_cat": False,
                "res_add": False,
                "n_layer_mlp_in": 1,
                "n_layer_mlp_out": 1,
                "mlp_in_cfg": {"dropout_p": None, "use_layernorm": False},
                "mlp_out_cfg": {"dropout_p": None, "use_layernorm": False},
            },
            "interaction_first": True,
            "add_goal_latent_first": False,
            "resample_latent": False,
            "n_layer_final_mlp": 1,
            "final_mlp": {"dropout_p": None, "use_layernorm": False, "end_layer_activation": False},
        }
    )
    return OmegaConf.to_container(cfg, resolve=True)


def build_trafficbots() -> nn.Module:
    return TrafficBotsStep(_default_model_cfg())


def example_input_trafficbots():
    torch.manual_seed(0)
    n_scene, n_step_hist, n_agent, n_pl, n_pl_node, n_tl = 1, 3, 4, 3, 4, 2

    agent_valid = torch.ones(n_scene, n_step_hist, n_agent, dtype=torch.bool)
    agent_attr = torch.randn(n_scene, n_step_hist, n_agent, 3)
    agent_pe = torch.randn(n_scene, n_step_hist, n_agent, 8)
    agent_pos = torch.randn(n_scene, n_step_hist, n_agent, 2)
    agent_type = F.one_hot(torch.zeros(n_scene, n_agent, dtype=torch.long), num_classes=3).bool()
    agent_state = torch.randn(n_scene, n_step_hist, n_agent, 4)

    map_valid = torch.ones(n_scene, n_pl, n_pl_node, dtype=torch.bool)
    map_attr = torch.randn(n_scene, n_pl, n_pl_node, 11)
    map_pe = torch.randn(n_scene, n_pl, n_pl_node, 8)
    map_pos = torch.randn(n_scene, n_pl, 2)
    map_type = F.one_hot(torch.zeros(n_scene, n_pl, dtype=torch.long), num_classes=11).bool()

    tl_valid = torch.ones(n_scene, n_step_hist, n_tl, dtype=torch.bool)
    tl_attr = torch.randn(n_scene, n_step_hist, n_tl, 5)
    tl_pe = torch.randn(n_scene, n_step_hist, n_tl, 8)
    tl_pos = torch.randn(n_scene, n_step_hist, n_tl, 2)

    return (
        agent_valid,
        agent_attr,
        agent_pe,
        agent_pos,
        agent_type,
        agent_state,
        map_valid,
        map_attr,
        map_pe,
        map_pos,
        map_type,
        tl_valid,
        tl_attr,
        tl_pe,
        tl_pos,
    )


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("TrafficBots", "build_trafficbots", "example_input_trafficbots", 2023, "CODE"),
]
