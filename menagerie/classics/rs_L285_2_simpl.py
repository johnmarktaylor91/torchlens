# SOURCE: vendored from https://github.com/HKUST-Aerial-Robotics/SIMPL @ main
# (simpl/simpl.py, utils/utils.py:gpu/init_weights, config/simpl_cfg.py:AdvCfg.get_net_cfg)
# SIMPL: A Simple and Efficient Multi-agent Motion Prediction Baseline for Autonomous
# Driving. RA-L 2024.
"""SIMPL: actor/lane polyline encoders fused via a symmetric fusion transformer,
decoded by a multi-modal MLP trajectory head.

Vendored verbatim from HKUST-Aerial-Robotics/SIMPL `simpl/simpl.py`, with the
`gpu`/`init_weights` helpers from `utils/utils.py` inlined (trivial device-move
and weight-init utilities, not architecture). One import was fixed for Python 3:
`from fractions import gcd` (Python-2-only) -> `from math import gcd`. Net config
values are copied verbatim from `config/simpl_cfg.py:AdvCfg.get_net_cfg()`.
Architecture is otherwise unmodified; only this header/build/example wrapper
were added for menagerie staging.
"""

import math
from math import gcd
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import MultiheadAttention
from torch.nn import functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# utils/utils.py (gpu, init_weights only)
# ---------------------------------------------------------------------------
def gpu(data, device):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x, device=device) for x in data]
    elif isinstance(data, dict):
        data = {key: gpu(_data, device=device) for key, _data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().to(device, non_blocking=True)
    return data


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


# ---------------------------------------------------------------------------
# simpl/simpl.py
# ---------------------------------------------------------------------------
class Conv1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm="GN", ng=32, act=True):
        super(Conv1d, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]

        self.conv = nn.Conv1d(
            n_in,
            n_out,
            kernel_size=kernel_size,
            padding=(int(kernel_size) - 1) // 2,
            stride=stride,
            bias=False,
        )

        if norm == "GN":
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == "BN":
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit("SyncBN has not been added!")

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm="GN", ng=32, act=True):
        super(Res1d, self).__init__()
        assert norm in ["GN", "BN", "SyncBN"]
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(
            n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == "GN":
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == "BN":
            self.bn1 = nn.BatchNorm1d(n_out)
            self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit("SyncBN has not been added!")

        if stride != 1 or n_out != n_in:
            if norm == "GN":
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out),
                )
            elif norm == "BN":
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(n_out),
                )
            else:
                exit("SyncBN has not been added!")
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


class ActorNet(nn.Module):
    """
    Actor feature extractor with Conv1D
    """

    def __init__(self, n_in=3, hidden_size=128, n_fpn_scale=4):
        super(ActorNet, self).__init__()
        norm = "GN"
        ng = 1

        n_out = [2 ** (5 + s) for s in range(n_fpn_scale)]  # [32, 64, 128]
        blocks = [Res1d] * n_fpn_scale
        num_blocks = [2] * n_fpn_scale

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], hidden_size, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(hidden_size, hidden_size, norm=norm, ng=ng)

    def forward(self, actors: Tensor) -> Tensor:
        out = actors

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        out = self.output(out)[:, :, -1]
        return out


class PointAggregateBlock(nn.Module):
    def __init__(self, hidden_size: int, aggre_out: bool, dropout: float = 0.1) -> None:
        super(PointAggregateBlock, self).__init__()
        self.aggre_out = aggre_out

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def _global_maxpool_aggre(self, feat):
        return F.adaptive_max_pool1d(feat.permute(0, 2, 1), 1).permute(0, 2, 1)

    def forward(self, x_inp):
        x = self.fc1(x_inp)  # [N_{lane}, 10, hidden_size]
        x_aggre = self._global_maxpool_aggre(x)
        x_aggre = torch.cat([x, x_aggre.repeat([1, x.shape[1], 1])], dim=-1)

        out = self.norm(x_inp + self.fc2(x_aggre))
        if self.aggre_out:
            return self._global_maxpool_aggre(out).squeeze()
        else:
            return out


class LaneNet(nn.Module):
    def __init__(self, device, in_size=10, hidden_size=128, dropout=0.1):
        super(LaneNet, self).__init__()
        self.device = device

        self.proj = nn.Sequential(
            nn.Linear(in_size, hidden_size), nn.LayerNorm(hidden_size), nn.ReLU(inplace=True)
        )
        self.aggre1 = PointAggregateBlock(hidden_size=hidden_size, aggre_out=False, dropout=dropout)
        self.aggre2 = PointAggregateBlock(hidden_size=hidden_size, aggre_out=True, dropout=dropout)

    def forward(self, feats):
        x = self.proj(feats)  # [N_{lane}, 10, hidden_size]
        x = self.aggre1(x)
        x = self.aggre2(x)  # [N_{lane}, hidden_size]
        return x


class SftLayer(nn.Module):
    def __init__(
        self,
        device,
        d_edge: int = 128,
        d_model: int = 128,
        d_ffn: int = 2048,
        n_head: int = 8,
        dropout: float = 0.1,
        update_edge: bool = True,
    ) -> None:
        super(SftLayer, self).__init__()
        self.device = device
        self.update_edge = update_edge

        self.proj_memory = nn.Sequential(
            nn.Linear(d_model + d_model + d_edge, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )

        if self.update_edge:
            self.proj_edge = nn.Sequential(
                nn.Linear(d_model, d_edge), nn.LayerNorm(d_edge), nn.ReLU(inplace=True)
            )
            self.norm_edge = nn.LayerNorm(d_edge)

        self.multihead_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=False
        )

        # Feedforward model
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, node: Tensor, edge: Tensor, edge_mask: Optional[Tensor]) -> Tensor:
        """
        input:
            node:       (N, d_model)
            edge:       (N, N, d_model)
            edge_mask:  (N, N)
        """
        # update node
        x, edge, memory = self._build_memory(node, edge)
        x_prime, _ = self._mha_block(x, memory, attn_mask=None, key_padding_mask=edge_mask)
        x = self.norm2(x + x_prime).squeeze()
        x = self.norm3(x + self._ff_block(x))
        return x, edge, None

    def _build_memory(self, node: Tensor, edge: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        input:
            node:   (N, d_model)
            edge:   (N, N, d_edge)
        output:
            :param  (1, N, d_model)
            :param  (N, N, d_edge)
            :param  (N, N, d_model)
        """
        n_token = node.shape[0]

        # 1. build memory
        src_x = node.unsqueeze(dim=0).repeat([n_token, 1, 1])  # (N, N, d_model)
        tar_x = node.unsqueeze(dim=1).repeat([1, n_token, 1])  # (N, N, d_model)
        memory = self.proj_memory(torch.cat([edge, src_x, tar_x], dim=-1))  # (N, N, d_model)
        # 2. (optional) update edge (with residual)
        if self.update_edge:
            edge = self.norm_edge(edge + self.proj_edge(memory))  # (N, N, d_edge)

        return node.unsqueeze(dim=0), edge, memory

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        """
        input:
            x:                  [1, N, d_model]
            mem:                [N, N, d_model]
            attn_mask:          [N, N]
            key_padding_mask:   [N, N]
        output:
            :param      [1, N, d_model]
            :param      [N, N]
        """
        x, _ = self.multihead_attn(
            x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )  # return average attention weights
        return self.dropout2(x), None

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class SymmetricFusionTransformer(nn.Module):
    def __init__(
        self,
        device,
        d_model: int = 128,
        d_edge: int = 128,
        n_head: int = 8,
        n_layer: int = 6,
        dropout: float = 0.1,
        update_edge: bool = True,
    ):
        super(SymmetricFusionTransformer, self).__init__()
        self.device = device

        fusion = []
        for i in range(n_layer):
            need_update_edge = False if i == n_layer - 1 else update_edge
            fusion.append(
                SftLayer(
                    device=device,
                    d_edge=d_edge,
                    d_model=d_model,
                    d_ffn=d_model * 2,
                    n_head=n_head,
                    dropout=dropout,
                    update_edge=need_update_edge,
                )
            )
        self.fusion = nn.ModuleList(fusion)

    def forward(self, x: Tensor, edge: Tensor, edge_mask: Tensor) -> Tensor:
        """
        x: (N, d_model)
        edge: (d_model, N, N)
        edge_mask: (N, N)
        """
        for mod in self.fusion:
            x, edge, _ = mod(x, edge, edge_mask)
        return x, None


class FusionNet(nn.Module):
    def __init__(self, device, config):
        super(FusionNet, self).__init__()
        self.device = device

        d_embed = config["d_embed"]
        dropout = config["dropout"]
        update_edge = config["update_edge"]

        self.proj_actor = nn.Sequential(
            nn.Linear(config["d_actor"], d_embed), nn.LayerNorm(d_embed), nn.ReLU(inplace=True)
        )
        self.proj_lane = nn.Sequential(
            nn.Linear(config["d_lane"], d_embed), nn.LayerNorm(d_embed), nn.ReLU(inplace=True)
        )
        self.proj_rpe_scene = nn.Sequential(
            nn.Linear(config["d_rpe_in"], config["d_rpe"]),
            nn.LayerNorm(config["d_rpe"]),
            nn.ReLU(inplace=True),
        )

        self.fuse_scene = SymmetricFusionTransformer(
            self.device,
            d_model=d_embed,
            d_edge=config["d_rpe"],
            n_head=config["n_scene_head"],
            n_layer=config["n_scene_layer"],
            dropout=dropout,
            update_edge=update_edge,
        )

    def forward(
        self,
        actors: Tensor,
        actor_idcs: List[Tensor],
        lanes: Tensor,
        lane_idcs: List[Tensor],
        rpe_prep: Dict[str, Tensor],
    ):
        # projection
        actors = self.proj_actor(actors)
        lanes = self.proj_lane(lanes)

        actors_new, lanes_new = list(), list()
        for a_idcs, l_idcs, rpes in zip(actor_idcs, lane_idcs, rpe_prep):
            # * fusion - scene
            _actors = actors[a_idcs]
            _lanes = lanes[l_idcs]
            tokens = torch.cat([_actors, _lanes], dim=0)  # (N, d_model)
            rpe = self.proj_rpe_scene(rpes["scene"].permute(1, 2, 0))  # (N, N, d_rpe)
            out, _ = self.fuse_scene(tokens, rpe, rpes["scene_mask"])

            actors_new.append(out[: len(a_idcs)])
            lanes_new.append(out[len(a_idcs) :])
        actors = torch.cat(actors_new, dim=0)
        lanes = torch.cat(lanes_new, dim=0)
        return actors, lanes, None


class MLPDecoder(nn.Module):
    def __init__(self, device, config) -> None:
        super(MLPDecoder, self).__init__()
        self.device = device
        self.config = config
        self.hidden_size = config["d_embed"]
        self.future_steps = config["g_pred_len"]
        self.num_modes = config["g_num_modes"]
        self.param_out = config["param_out"]  # parametric output: bezier/monomial/none
        self.N_ORDER = config["param_order"]

        dim_mm = self.hidden_size * self.num_modes
        dim_inter = dim_mm // 2
        self.multihead_proj = nn.Sequential(
            nn.Linear(self.hidden_size, dim_inter),
            nn.LayerNorm(dim_inter),
            nn.ReLU(inplace=True),
            nn.Linear(dim_inter, dim_mm),
            nn.LayerNorm(dim_mm),
            nn.ReLU(inplace=True),
        )

        self.cls = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1),
        )

        if self.param_out == "bezier":
            self.mat_T = self._get_T_matrix_bezier(
                n_order=self.N_ORDER, n_step=self.future_steps
            ).to(self.device)
            self.mat_Tp = self._get_Tp_matrix_bezier(
                n_order=self.N_ORDER, n_step=self.future_steps
            ).to(self.device)

            self.reg = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, (self.N_ORDER + 1) * 2),
            )
        elif self.param_out == "monomial":
            self.mat_T = self._get_T_matrix_monomial(
                n_order=self.N_ORDER, n_step=self.future_steps
            ).to(self.device)
            self.mat_Tp = self._get_Tp_matrix_monomial(
                n_order=self.N_ORDER, n_step=self.future_steps
            ).to(self.device)

            self.reg = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, (self.N_ORDER + 1) * 2),
            )
        elif self.param_out == "none":
            self.reg = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, self.future_steps * 2),
            )
        else:
            raise NotImplementedError

    def _get_T_matrix_bezier(self, n_order, n_step):
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        T = []
        for i in range(n_order + 1):
            coeff = math.comb(n_order, i) * (1.0 - ts) ** (n_order - i) * ts**i
            T.append(coeff)
        return torch.Tensor(np.array(T).T)

    def _get_Tp_matrix_bezier(self, n_order, n_step):
        # ~ 1st derivatives
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        Tp = []
        for i in range(n_order):
            coeff = n_order * math.comb(n_order - 1, i) * (1.0 - ts) ** (n_order - 1 - i) * ts**i
            Tp.append(coeff)
        return torch.Tensor(np.array(Tp).T)

    def _get_T_matrix_monomial(self, n_order, n_step):
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        T = []
        for i in range(n_order + 1):
            coeff = ts**i
            T.append(coeff)
        return torch.Tensor(np.array(T).T)

    def _get_Tp_matrix_monomial(self, n_order, n_step):
        # ~ 1st derivatives
        ts = np.linspace(0.0, 1.0, n_step, endpoint=True)
        Tp = []
        for i in range(n_order):
            coeff = (i + 1) * (ts**i)
            Tp.append(coeff)
        return torch.Tensor(np.array(Tp).T)

    def forward(
        self, embed: torch.Tensor, actor_idcs: List[Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # input embed: [159, 128]
        embed = (
            self.multihead_proj(embed).view(-1, self.num_modes, self.hidden_size).permute(1, 0, 2)
        )

        cls = self.cls(embed).view(self.num_modes, -1).permute(1, 0)  # e.g., [159, 6]
        cls = F.softmax(cls * 1.0, dim=1)  # e.g., [159, 6]

        if self.param_out == "bezier":
            param = self.reg(embed).view(
                self.num_modes, -1, self.N_ORDER + 1, 2
            )  # e.g., [6, 159, N_ORDER + 1, 2]
            param = param.permute(1, 0, 2, 3)  # e.g., [159, 6, N_ORDER + 1, 2]
            reg = torch.matmul(self.mat_T, param)  # e.g., [159, 6, 30, 2]
            vel = torch.matmul(self.mat_Tp, torch.diff(param, dim=2)) / (self.future_steps * 0.1)
        elif self.param_out == "monomial":
            param = self.reg(embed).view(
                self.num_modes, -1, self.N_ORDER + 1, 2
            )  # e.g., [6, 159, N_ORDER + 1, 2]
            param = param.permute(1, 0, 2, 3)  # e.g., [159, 6, N_ORDER + 1, 2]
            reg = torch.matmul(self.mat_T, param)  # e.g., [159, 6, 30, 2]
            vel = torch.matmul(self.mat_Tp, param[:, :, 1:, :]) / (self.future_steps * 0.1)
        elif self.param_out == "none":
            reg = self.reg(embed).view(
                self.num_modes, -1, self.future_steps, 2
            )  # e.g., [6, 159, 30, 2]
            reg = reg.permute(1, 0, 2, 3)  # e.g., [159, 6, 30, 2]
            vel = torch.gradient(reg, dim=-2)[0] / 0.1  # vel is calculated from pos

        # de-batchify
        res_cls, res_reg, res_aux = [], [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            res_cls.append(cls[idcs])
            res_reg.append(reg[idcs])

            if self.param_out == "none":
                res_aux.append((vel[idcs], None))  # ! None is a placeholder
            else:
                res_aux.append((vel[idcs], param[idcs]))  # List[Tuple[Tensor,...]]

        return res_cls, res_reg, res_aux


class Simpl(nn.Module):
    # Initialization
    def __init__(self, cfg, device):
        super(Simpl, self).__init__()
        self.device = device

        self.actor_net = ActorNet(
            n_in=cfg["in_actor"], hidden_size=cfg["d_actor"], n_fpn_scale=cfg["n_fpn_scale"]
        )

        self.lane_net = LaneNet(
            device=self.device,
            in_size=cfg["in_lane"],
            hidden_size=cfg["d_lane"],
            dropout=cfg["dropout"],
        )

        self.fusion_net = FusionNet(device=self.device, config=cfg)

        self.pred_net = MLPDecoder(device=self.device, config=cfg)

        if cfg["init_weights"]:
            self.apply(init_weights)

    def forward(self, data):
        actors, actor_idcs, lanes, lane_idcs, rpe = data

        # * actors/lanes encoding
        actors = self.actor_net(actors)  # output: [N_{actor}, 128]
        lanes = self.lane_net(lanes)  # output: [N_{lane}, 128]
        # * fusion
        actors, lanes, _ = self.fusion_net(actors, actor_idcs, lanes, lane_idcs, rpe)
        # * decoding
        out = self.pred_net(actors, actor_idcs)

        return out

    def pre_process(self, data):
        actors = gpu(data["ACTORS"], self.device)
        actor_idcs = gpu(data["ACTOR_IDCS"], self.device)
        lanes = gpu(data["LANES"], self.device)
        lane_idcs = gpu(data["LANE_IDCS"], self.device)
        rpe = gpu(data["RPE"], self.device)

        return actors, actor_idcs, lanes, lane_idcs, rpe

    def post_process(self, out):
        post_out = dict()
        res_cls = out[0]
        res_reg = out[1]

        # get prediction results for target vehicles only
        reg = torch.stack([trajs[0] for trajs in res_reg], dim=0)
        cls = torch.stack([probs[0] for probs in res_cls], dim=0)

        post_out["out_raw"] = out
        post_out["traj_pred"] = reg  # batch x n_mod x pred_len x 2
        post_out["prob_pred"] = cls  # batch x n_mod

        return post_out


# ---------------------------------------------------------------------------
# menagerie staging wrapper
# ---------------------------------------------------------------------------
# config/simpl_cfg.py: AdvCfg.get_net_cfg(), verbatim values (g_num_modes/g_pred_len
# from AdvCfg.g_cfg); init_weights disabled to match the repo default (random init
# is fine for tracing either way).
_NET_CFG: Dict[str, Any] = {
    "init_weights": False,
    "in_actor": 3,
    "d_actor": 128,
    "n_fpn_scale": 3,
    "in_lane": 10,
    "d_lane": 128,
    "d_rpe_in": 5,
    "d_rpe": 128,
    "d_embed": 128,
    "n_scene_layer": 4,
    "n_scene_head": 8,
    "dropout": 0.1,
    "update_edge": True,
    "param_out": "none",  # bezier/monomial/none
    "param_order": 5,
    "g_num_modes": 6,
    "g_obs_len": 20,
    "g_pred_len": 30,
}


def build_simpl():
    return Simpl(cfg=_NET_CFG, device="cpu")


def _make_rpe(ctrs: Tensor, vecs: Tensor) -> Dict[str, Any]:
    """Faithful port of ArgoDataset._get_rpe (av1_dataset.py) for a single scene."""
    radius = 100.0
    d_pos = (ctrs.unsqueeze(0) - ctrs.unsqueeze(1)).norm(dim=-1)
    d_pos = d_pos * 2 / radius
    pos_rpe = d_pos.unsqueeze(0)

    def _cos(v1, v2):
        v1n, v2n = v1.norm(dim=-1), v2.norm(dim=-1)
        return (v1[..., 0] * v2[..., 0] + v1[..., 1] * v2[..., 1]) / (v1n * v2n + 1e-10)

    def _sin(v1, v2):
        v1n, v2n = v1.norm(dim=-1), v2.norm(dim=-1)
        return (v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]) / (v1n * v2n + 1e-10)

    cos_a1 = _cos(vecs.unsqueeze(0), vecs.unsqueeze(1))
    sin_a1 = _sin(vecs.unsqueeze(0), vecs.unsqueeze(1))
    v_pos = ctrs.unsqueeze(0) - ctrs.unsqueeze(1)
    cos_a2 = _cos(vecs.unsqueeze(0), v_pos)
    sin_a2 = _sin(vecs.unsqueeze(0), v_pos)
    ang_rpe = torch.stack([cos_a1, sin_a1, cos_a2, sin_a2])
    rpe = torch.cat([ang_rpe, pos_rpe], dim=0)  # (5, N, N)
    return {"scene": rpe, "scene_mask": None}


def example_input_simpl():
    """Single-scene batch: n_actor actors (obs_len=20 timesteps, 3 feats: vx, vy, pad-flag)
    and n_lane 10-point lane polylines (10 feats each), following
    ArgoDataset.actor_gather / graph_gather / _get_rpe from av1_dataset.py.
    """
    torch.manual_seed(0)
    n_actor = 4
    n_lane = 3
    obs_len = _NET_CFG["g_obs_len"]

    actors = torch.randn(n_actor, _NET_CFG["in_actor"], obs_len)
    actor_idcs = [torch.arange(n_actor)]

    lanes = torch.randn(n_lane, 10, _NET_CFG["in_lane"])
    lane_idcs = [torch.arange(n_lane)]

    actor_ctrs = torch.randn(n_actor, 2)
    actor_vecs = torch.randn(n_actor, 2)
    lane_ctrs = torch.randn(n_lane, 2)
    lane_vecs = torch.randn(n_lane, 2)
    scene_ctrs = torch.cat([actor_ctrs, lane_ctrs], dim=0)
    scene_vecs = torch.cat([actor_vecs, lane_vecs], dim=0)
    rpe = [_make_rpe(scene_ctrs, scene_vecs)]

    data = (actors, actor_idcs, lanes, lane_idcs, rpe)
    return (data,)


MENAGERIE_ENTRIES = [
    ("SIMPL", "build_simpl", "example_input_simpl", 2024, "vendored-pytorch"),
]
