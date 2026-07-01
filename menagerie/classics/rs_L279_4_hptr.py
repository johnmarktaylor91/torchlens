# SOURCE: vendored from zhejz/HPTR @ main
# Files combined:
#   src/models/sc_relative.py (SceneCentricRelative, IntraClassEncoder, Decoder -- the
#       heterogeneous polyline transformer's scene-centric-relative variant, matching
#       configs/model/scr_av2.yaml)
#   src/models/modules/mlp.py (MLP)
#   src/models/modules/point_net.py (PointNet -- the VectorNet-style polyline subnet)
#   src/models/modules/transformer.py (TransformerBlock, TransformerCrossAttention)
#   src/models/modules/attention.py (AttentionRPE -- "KNARPE", the KNN + relative-pose-encoded
#       attention kernel that is HPTR's core mechanism)
#   src/models/modules/decoder_ensemble.py (DecoderEnsemble, MLPHead, MLPEnsemble)
#   src/models/modules/rpe.py (get_rel_pose, get_tgt_knn_idx -- the KNN target-selection +
#       relative-pose-encoding pipeline)
#   src/models/modules/multi_modal.py (MultiModalAnchors)
#   src/models/modules/pos_emb.py (PositionalEmbedding, PositionalEmbeddingRad)
#   src/utils/pose_pe.py (PosePE)
#   src/utils/transform_utils.py (cast_rad, torch_rad2rot, torch_pos2local, torch_rad2local --
#       only the torch-tensor helpers actually used by rpe.py/pose_pe.py; the numpy-only
#       transforms3d-dependent helpers in the real file are not used on this forward path and
#       are omitted, see note below)
#
# HPTR (Zhang et al., "Real-Time Motion Prediction via Heterogeneous Polyline Transformer with
# Relative Pose Encoding", NeurIPS 2023) is a real-time (40fps) multi-agent motion forecaster.
# Its core contribution, "KNARPE" (K-Nearest-Neighbor Attention with Relative Pose Encoding,
# `AttentionRPE`), lets every token (map polyline / traffic light / agent) attend only to its
# K nearest neighbors by relative pose, with the relative pose itself injected into the
# attention keys/values -- this is what gives the model its factorized, cacheable,
# translation/rotation-invariant scene representation. `IntraClassEncoder` embeds
# map/traffic-light/agent tokens independently (each via `PointNet` polyline pooling +
# `TransformerBlock` self-attention over KNN neighbors), then `Decoder` runs staged
# cross-attention reinforcement (tl->map, agent->map+tl, multi-modal anchors->everything) to
# emit K-mode future trajectories via an ensemble MLP head.
#
# Import-only fixes applied (no architectural change):
#   - `decoder_ensemble.py`'s `DecoderEnsemble`/`MLPEnsemble` call
#     `hydra.utils.instantiate(decoder_cfg)` to build sub-decoders from an OmegaConf
#     `DictConfig` with a `_target_` dotted-path string. Vendored unmodified; this staging
#     module registers itself in `sys.modules` under a fixed name so `_target_` strings can
#     resolve real dotted paths into it (see `build_hptr()` below) -- hydra/omegaconf are both
#     already installed, so `hydra.utils.instantiate` runs for real, unmodified.
#   - `transform_utils.py`'s `_rotation33_as_yaw`/`_yaw_as_rotation33`/`get_so2_from_se2`/
#     `get_yaw_from_se2`/`transform_points`/`get_transformation_matrix` (numpy-only helpers
#     gated behind an unused `transforms3d` import, for converting between world-frame SE(2)
#     poses and yaw -- used only by the repo's raw-data preprocessing, not by any nn.Module
#     forward pass) are not needed on the traced path and are omitted; only the torch-tensor
#     helpers actually called by `rpe.py`/`pose_pe.py` (`cast_rad`, `torch_rad2rot`,
#     `torch_pos2local`, `torch_rad2local`) are vendored.
#   - The repo's `SceneCentricRelative.forward()` always builds `emb_invalid`/`rel_pose`/
#     `rel_dist` for `inference_repeat_n` iterations (a benchmarking loop that re-runs
#     identical work `inference_repeat_n` times to measure steady-state latency); the example
#     input uses the repo's own default `inference_repeat_n=1`, so this is a single real pass,
#     not a modification.
#
# MENAGERIE_ZOO = "vendored-pytorch"

import math
import sys
from typing import List, Optional, Tuple, Union

import hydra
import torch
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# src/utils/transform_utils.py (torch-tensor helpers only; see header note)
# ---------------------------------------------------------------------------
def cast_rad(angle):
    """Cast angle such that they are always in the [-pi, pi) range."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def torch_rad2rot(rad: Tensor) -> Tensor:
    _cos = torch.cos(rad)
    _sin = torch.sin(rad)
    return torch.stack(
        [torch.stack([_cos, -_sin], dim=-1), torch.stack([_sin, _cos], dim=-1)], dim=-2
    )


def torch_pos2local(in_pos: Tensor, local_pos: Tensor, local_rot: Tensor) -> Tensor:
    return torch.matmul(in_pos - local_pos, local_rot)


def torch_rad2local(in_rad: Tensor, local_rad: Tensor, cast: bool = True) -> Tensor:
    out_rad = in_rad - local_rad.unsqueeze(-1)
    if cast:
        out_rad = cast_rad(out_rad)
    return out_rad


# ---------------------------------------------------------------------------
# src/models/modules/pos_emb.py
# ---------------------------------------------------------------------------
class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        freqs = freqs.repeat_interleave(2, 0)
        self.register_buffer("freqs", freqs)

    def forward(self, x: Tensor):
        pos_enc = x.unsqueeze(-1) * self.freqs.view([1] * x.dim() + [-1])
        pos_enc = torch.cat([torch.cos(pos_enc[..., ::2]), torch.sin(pos_enc[..., 1::2])], dim=-1)
        return pos_enc


class PositionalEmbeddingRad(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        freqs = torch.arange(0, dim // 2) + 1.0
        freqs = freqs.repeat_interleave(2, 0)
        self.register_buffer("freqs", freqs)

    def forward(self, x: Tensor):
        pos_enc = x.unsqueeze(-1) * self.freqs.view([1] * x.dim() + [-1])
        pos_enc = torch.cat([torch.cos(pos_enc[..., ::2]), torch.sin(pos_enc[..., 1::2])], dim=-1)
        return pos_enc


# ---------------------------------------------------------------------------
# src/utils/pose_pe.py
# ---------------------------------------------------------------------------
class PosePE(nn.Module):
    def __init__(self, mode: str, pe_dim: int = 256, theta_xy: float = 1e3, theta_cs: float = 1e1):
        super().__init__()
        self.mode = mode
        if self.mode == "xy_dir":
            self.out_dim = 4
        elif self.mode == "mpa_pl":
            self.out_dim = 7
        elif self.mode == "pe_xy_dir":
            self.out_dim = pe_dim
            self.pe_xy = PositionalEmbedding(dim=pe_dim // 4, theta=theta_xy)
            self.pe_dir = PositionalEmbedding(dim=pe_dim // 4, theta=theta_cs)
        elif self.mode == "pe_xy_yaw":
            self.out_dim = pe_dim
            self.pe_xy = PositionalEmbedding(dim=pe_dim // 4, theta=theta_xy)
            self.pe_yaw = PositionalEmbeddingRad(dim=pe_dim // 2)
        else:
            raise NotImplementedError

    def forward(self, xy: Tensor, dir: Tensor):
        if self.mode == "xy_dir":
            if dir.shape[-1] == 1:
                dir = torch.cat([dir.cos(), dir.sin()], dim=-1)
            pos_out = torch.cat([xy, dir], dim=-1)
        elif self.mode == "mpa_pl":
            if dir.shape[-1] == 1:
                dir = torch.cat([dir.cos(), dir.sin()], dim=-1)
            pos_out = self.encode_polyline(xy, dir)
        elif self.mode == "pe_xy_dir":
            if dir.shape[-1] == 1:
                dir = torch.cat([dir.cos(), dir.sin()], dim=-1)
            pos_out = torch.cat(
                [
                    self.pe_xy(xy[..., 0]),
                    self.pe_xy(xy[..., 1]),
                    self.pe_dir(dir[..., 0]),
                    self.pe_dir(dir[..., 1]),
                ],
                dim=-1,
            )
        elif self.mode == "pe_xy_yaw":
            if dir.shape[-1] == 1:
                dir = dir.squeeze(-1)
            else:
                dir = torch.atan2(dir[..., 1], dir[..., 0])
            pos_out = torch.cat(
                [self.pe_xy(xy[..., 0]), self.pe_xy(xy[..., 1]), self.pe_yaw(dir)], dim=-1
            )
        return pos_out

    @staticmethod
    def encode_polyline(pos: Tensor, dir: Tensor) -> Tensor:
        eps = torch.finfo(pos.dtype).eps
        segments_start = pos
        segment_vec = dir
        segment_proj = (-segments_start * segment_vec).sum(-1) / (
            (segment_vec * segment_vec).sum(-1) + eps
        )
        closest_points = (
            segments_start + torch.clamp(segment_proj, min=0, max=1).unsqueeze(-1) * segment_vec
        )
        r_norm = torch.norm(closest_points, dim=-1, keepdim=True)
        segment_vec_norm = torch.norm(segment_vec, dim=-1, keepdim=True)
        pl_feature = torch.cat(
            [
                r_norm,
                closest_points / (r_norm + eps),
                segment_vec / (segment_vec_norm + eps),
                segment_vec_norm,
                torch.norm(segments_start + segment_vec - closest_points, dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        return pl_feature


# ---------------------------------------------------------------------------
# src/models/modules/rpe.py
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_rel_pose(pose: Tensor, invalid: Tensor) -> Tuple[Tensor, Tensor]:
    xy = pose[:, :, :2]
    yaw = pose[:, :, -1]
    rel_pose = torch.cat(
        [
            torch_pos2local(xy.unsqueeze(1), xy.unsqueeze(2), torch_rad2rot(yaw)),
            torch_rad2local(yaw.unsqueeze(1), yaw, cast=False).unsqueeze(-1),
        ],
        dim=-1,
    )
    rel_dist = torch.norm(rel_pose[..., :2], dim=-1)
    rel_dist.masked_fill_(invalid.unsqueeze(1) | invalid.unsqueeze(2), float("inf"))
    return rel_pose, rel_dist


@torch.no_grad()
def get_tgt_knn_idx(
    tgt_invalid: Tensor,
    rel_pose: Optional[Tensor],
    rel_dist: Tensor,
    n_tgt_knn: int,
    dist_limit: Union[float, Tensor],
) -> Tuple[Optional[Tensor], Tensor, Optional[Tensor]]:
    n_scene, n_src, _ = rel_dist.shape
    idx_scene = torch.arange(n_scene)[:, None, None]
    idx_src = torch.arange(n_src)[None, :, None]

    if 0 < n_tgt_knn < tgt_invalid.shape[1]:
        dist_knn, idx_tgt = torch.topk(rel_dist, n_tgt_knn, dim=-1, largest=False, sorted=False)
        tgt_invalid_knn = tgt_invalid.unsqueeze(1).expand(-1, n_src, -1)[
            idx_scene, idx_src, idx_tgt
        ]
        if rel_pose is None:
            rpe = None
        else:
            rpe = rel_pose[idx_scene, idx_src, idx_tgt]
    else:
        dist_knn = rel_dist
        tgt_invalid_knn = tgt_invalid.unsqueeze(1).expand(-1, n_src, -1)
        rpe = rel_pose
        idx_tgt = None

    tgt_invalid_knn = tgt_invalid_knn | (dist_knn > dist_limit)
    if rpe is not None:
        rpe = rpe.masked_fill(tgt_invalid_knn.unsqueeze(-1), 0)

    return idx_tgt, tgt_invalid_knn, rpe


# ---------------------------------------------------------------------------
# src/models/modules/mlp.py
# ---------------------------------------------------------------------------
def _get_activation(activation: str, inplace: bool) -> nn.Module:
    if activation == "relu":
        return nn.ReLU(inplace=inplace)
    elif activation == "gelu":
        return nn.GELU()
    raise RuntimeError("activation {} not implemented".format(activation))


class MLP(nn.Module):
    def __init__(
        self,
        fc_dims: Union[List, Tuple],
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
            x = x.masked_fill(~valid_mask.unsqueeze(-1), fill_invalid)
        if self.end_layer_activation is not None:
            x = self.end_layer_activation(x)
        return x


# ---------------------------------------------------------------------------
# src/models/modules/point_net.py
# ---------------------------------------------------------------------------
class PointNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layer: int = 3,
        use_layernorm: bool = False,
        use_batchnorm: bool = False,
        end_layer_activation: bool = True,
        dropout_p: Optional[float] = None,
        pool_mode: str = "max",
    ) -> None:
        super().__init__()
        self.pool_mode = pool_mode
        self.input_mlp = MLP(
            [input_dim, hidden_dim, hidden_dim],
            dropout_p=dropout_p,
            use_layernorm=use_layernorm,
            use_batchnorm=use_batchnorm,
        )

        mlp_layers: List[nn.Module] = []
        for _ in range(n_layer - 2):
            mlp_layers.append(
                MLP(
                    [hidden_dim, hidden_dim // 2],
                    dropout_p=dropout_p,
                    use_layernorm=use_layernorm,
                    use_batchnorm=use_batchnorm,
                )
            )
        mlp_layers.append(
            MLP(
                [hidden_dim, hidden_dim // 2],
                dropout_p=dropout_p,
                use_layernorm=use_layernorm,
                use_batchnorm=use_batchnorm,
                end_layer_activation=end_layer_activation,
            )
        )
        self.mlp_layers = nn.ModuleList(mlp_layers)

    def forward(self, x: Tensor, valid: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.input_mlp(x, valid)

        for mlp in self.mlp_layers:
            feature_encoded = mlp(x, valid, float("-inf"))
            feature_pooled = feature_encoded.amax(dim=2, keepdim=True)
            x = torch.cat(
                (feature_encoded, feature_pooled.expand(-1, -1, valid.shape[-1], -1)), dim=-1
            )

        if self.pool_mode == "max":
            x = x.masked_fill(~valid.unsqueeze(-1), float("-inf"))
            emb = x.amax(dim=2, keepdim=False)
        elif self.pool_mode == "first":
            emb = x[:, :, 0]
        elif self.pool_mode == "mean":
            x = x.masked_fill(~valid.unsqueeze(-1), 0)
            emb = x.sum(dim=2, keepdim=False)
            emb = emb / (valid.sum(dim=-1, keepdim=True) + torch.finfo(x.dtype).eps)

        emb_valid = valid.any(-1)
        emb = emb.masked_fill(~emb_valid.unsqueeze(-1), 0)
        return emb, emb_valid


# ---------------------------------------------------------------------------
# src/models/modules/multi_modal.py
# ---------------------------------------------------------------------------
class MultiModalAnchors(nn.Module):
    def __init__(
        self,
        mode_emb: str,
        mode_init: str,
        hidden_dim: int,
        n_pred: int,
        emb_dim: int,
        use_agent_type: bool,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_pred = n_pred
        self.use_agent_type = use_agent_type

        self.mode_init = mode_init
        n_anchors = 3 if use_agent_type else 1
        if self.mode_init == "xavier":
            self.anchors = torch.empty((n_anchors, n_pred, hidden_dim))
            nn.init.xavier_normal_(self.anchors)
            self.anchors = nn.Parameter(self.anchors * scale, requires_grad=True)
        elif self.mode_init == "uniform":
            self.anchors = torch.empty((n_anchors, n_pred, hidden_dim))
            self.anchors.uniform_(-scale, scale)
            self.anchors = nn.Parameter(self.anchors, requires_grad=True)
        elif self.mode_init == "randn":
            self.anchors = nn.Parameter(
                torch.randn([n_anchors, n_pred, hidden_dim]) * scale, requires_grad=True
            )
        else:
            raise NotImplementedError

        self.mode_emb = mode_emb
        if self.mode_emb == "linear":
            self.mlp_anchor = nn.Linear(self.anchors.shape[-1] + emb_dim, hidden_dim, bias=False)
        elif self.mode_emb == "mlp":
            self.mlp_anchor = MLP(
                [self.anchors.shape[-1] + emb_dim] + [hidden_dim] * 2, end_layer_activation=False
            )
        elif self.mode_emb == "add" or self.mode_emb == "none":
            assert emb_dim == hidden_dim
            if self.anchors.shape[-1] != hidden_dim:
                self.mlp_anchor = nn.Linear(self.anchors.shape[-1], hidden_dim, bias=False)
            else:
                self.mlp_anchor = None
        else:
            raise NotImplementedError

    def forward(self, valid: Tensor, emb: Tensor, agent_type: Tensor) -> Tensor:
        if self.use_agent_type:
            anchors = (self.anchors.unsqueeze(0) * agent_type[:, :, None, None]).sum(1)
        else:
            anchors = self.anchors.expand(valid.shape[0], -1, -1)

        if self.mode_emb == "linear" or self.mode_emb == "mlp":
            mm_emb = torch.cat([emb.unsqueeze(1).expand(-1, self.n_pred, -1), anchors], dim=-1)
            mm_emb = self.mlp_anchor(mm_emb)
        elif self.mode_emb == "add":
            if self.mlp_anchor is not None:
                anchors = self.mlp_anchor(anchors)
            mm_emb = emb.unsqueeze(1) + anchors
        elif self.mode_emb == "none":
            if self.mlp_anchor is not None:
                anchors = self.mlp_anchor(anchors)
            mm_emb = anchors
        return mm_emb.masked_fill(~valid[:, None, None], 0)


# ---------------------------------------------------------------------------
# src/models/modules/attention.py -- "KNARPE"
# ---------------------------------------------------------------------------
class AttentionRPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout_p: float = 0.0,
        bias: bool = True,
        d_rpe: int = -1,
        apply_q_rpe: bool = False,
    ) -> None:
        super(AttentionRPE, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.apply_q_rpe = apply_q_rpe
        self.d_rpe = d_rpe

        assert self.d_head * n_head == d_model, "d_model must be divisible by n_head"

        if self.d_rpe > 0:
            n_project_rpe = 3 if apply_q_rpe else 2
            self.mlp_rpe = nn.Linear(d_rpe, n_project_rpe * d_model, bias=bias)

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
        if self.out_proj_bias is not None:
            nn.init.constant_(self.out_proj_bias, 0.0)

    def forward(
        self,
        src: Tensor,
        tgt: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        rpe: Optional[Tensor] = None,
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

        if rpe is None:
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
        else:
            assert self.d_rpe > 0
            k = k.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
            v = v.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
            q = q.view(n_batch, n_src, self.n_head, self.d_head).transpose(1, 2).unsqueeze(3)

            rpe = self.mlp_rpe(rpe)
            if self.apply_q_rpe:
                rpe_q, rpe_k, rpe_v = rpe.chunk(3, dim=-1)
                rpe_q = rpe_q.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
            else:
                rpe_k, rpe_v = rpe.chunk(2, dim=-1)
            rpe_k = rpe_k.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)
            rpe_v = rpe_v.view(n_batch, n_src, n_tgt, self.n_head, self.d_head).movedim(3, 1)

            if self.apply_q_rpe:
                attn = torch.sum((q + rpe_q) * (k + rpe_k), dim=-1)
            else:
                attn = torch.sum(q * (k + rpe_k), dim=-1)

        if attn_invalid_mask is not None:
            attn = attn.masked_fill(attn_invalid_mask.unsqueeze(1), float("-inf"))

        attn = torch.softmax(attn / math.sqrt(self.d_head), dim=-1)
        if self.dropout is not None:
            attn = self.dropout(attn)

        if rpe is None:
            if v.dim() == 4:
                out = torch.matmul(attn, v)
            else:
                out = torch.sum(v * attn.unsqueeze(-1), dim=3)
        else:
            out = torch.sum((v + rpe_v) * attn.unsqueeze(-1), dim=3)

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
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


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
        d_rpe: int = -1,
        apply_q_rpe: bool = False,
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
                    d_rpe=d_rpe,
                    apply_q_rpe=apply_q_rpe,
                )
                for _ in range(n_layer)
            ]
        )

    def forward(
        self,
        src: Tensor,
        src_padding_mask: Optional[Tensor] = None,
        tgt: Optional[Tensor] = None,
        tgt_padding_mask: Optional[Tensor] = None,
        rpe: Optional[Tensor] = None,
        decoder_tgt: Optional[Tensor] = None,
        decoder_tgt_padding_mask: Optional[Tensor] = None,
        decoder_rpe: Optional[Tensor] = None,
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
                rpe=rpe,
                decoder_tgt=decoder_tgt,
                decoder_tgt_padding_mask=decoder_tgt_padding_mask,
                decoder_rpe=decoder_rpe,
                attn_mask=attn_mask,
                need_weights=need_weights,
            )
        return src, attn_weights


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
        d_rpe: int = -1,
        apply_q_rpe: bool = False,
    ) -> None:
        super(TransformerCrossAttention, self).__init__()
        self.norm_first = norm_first
        self.d_feedforward = d_feedforward
        self.decoder_self_attn = decoder_self_attn
        inplace = False

        self.dropout = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None
        self.activation = _get_activation_fn(activation)
        self.norm1 = nn.LayerNorm(d_model)

        if self.decoder_self_attn:
            self.attn_src = AttentionRPE(
                d_model=d_model,
                n_head=n_head,
                dropout_p=dropout_p,
                bias=bias,
                d_rpe=d_rpe,
                apply_q_rpe=apply_q_rpe,
            )
            self.norm_src = nn.LayerNorm(d_model)
            self.dropout_src = nn.Dropout(p=dropout_p, inplace=inplace) if dropout_p > 0 else None

        if self.norm_first:
            self.norm_tgt = nn.LayerNorm(d_model)

        self.attn = AttentionRPE(
            d_model=d_model,
            n_head=n_head,
            dropout_p=dropout_p,
            bias=bias,
            d_rpe=d_rpe,
            apply_q_rpe=apply_q_rpe,
        )
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
        rpe: Optional[Tensor] = None,
        decoder_tgt: Optional[Tensor] = None,
        decoder_tgt_padding_mask: Optional[Tensor] = None,
        decoder_rpe: Optional[Tensor] = None,
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
                    _s = self.attn_src(
                        _s, decoder_tgt, tgt_padding_mask=decoder_tgt_padding_mask, rpe=decoder_rpe
                    )[0]

                if self.dropout_src is None:
                    src = src + _s
                else:
                    src = src + self.dropout_src(_s)
            else:
                if decoder_tgt is None:
                    _s = self.attn_src(src, tgt_padding_mask=src_padding_mask)[0]
                else:
                    _s = self.attn_src(
                        src, decoder_tgt, tgt_padding_mask=decoder_tgt_padding_mask, rpe=decoder_rpe
                    )[0]

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
            rpe=rpe,
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
            src = src.masked_fill(src_padding_mask.unsqueeze(-1), 0.0)
            if need_weights:
                attn_weights = attn_weights.masked_fill(src_padding_mask.unsqueeze(-1), 0.0)
        return src, attn_weights


# ---------------------------------------------------------------------------
# src/models/modules/decoder_ensemble.py
# ---------------------------------------------------------------------------
class DecoderEnsemble(nn.Module):
    def __init__(self, n_decoders: int, decoder_cfg) -> None:
        super().__init__()
        self.use_vmap = decoder_cfg["use_vmap"]
        self.n_decoders = n_decoders
        if self.use_vmap and self.n_decoders > 1:
            from functorch import combine_state_for_ensemble, vmap

            _decoders = [hydra.utils.instantiate(decoder_cfg) for _ in range(n_decoders)]
            fmodel_decoders, params_decoders, buffers_decoders = combine_state_for_ensemble(
                _decoders
            )
            assert buffers_decoders == ()
            self.v_model = vmap(fmodel_decoders, randomness="different")
            [p.requires_grad_() for p in params_decoders]
            self.params_decoders = nn.ParameterList(params_decoders)
        else:
            self._decoders = nn.ModuleList(
                [hydra.utils.instantiate(decoder_cfg) for _ in range(n_decoders)]
            )

    def forward(self, **kwargs) -> Tuple[Tensor, Tensor]:
        if self.use_vmap and self.n_decoders > 1:
            conf, pred = self.v_model(tuple(self.params_decoders), (), **kwargs)
        else:
            conf, pred = [], []
            for decoder in self._decoders:
                c, p = decoder(**kwargs)
                conf.append(c)
                pred.append(p)
            conf = torch.stack(conf, dim=0)
            pred = torch.stack(pred, dim=0)
        return conf, pred


class MLPHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        use_vmap: bool,
        n_step_future: int,
        out_mlp_layernorm: bool,
        out_mlp_batchnorm: bool,
        use_agent_type: bool,
        predictions: List[str],
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_agent_type = use_agent_type
        self.n_step_future = n_step_future

        self.pred_dim = 0
        if "pos" in predictions:
            self.pred_dim += 2
        if "spd" in predictions:
            self.pred_dim += 1
        if "vel" in predictions:
            self.pred_dim += 2
        if "yaw_bbox" in predictions:
            self.pred_dim += 1
        if "cov1" in predictions:
            self.pred_dim += 1
        elif "cov2" in predictions:
            self.pred_dim += 2
        elif "cov3" in predictions:
            self.pred_dim += 3

        _d = hidden_dim * 2
        cfg_mlp_pred = {
            "fc_dims": [hidden_dim, _d, _d, self.n_step_future * self.pred_dim],
            "end_layer_activation": False,
            "use_layernorm": out_mlp_layernorm,
            "use_batchnorm": out_mlp_batchnorm,
        }
        cfg_mlp_conf = {
            "end_layer_activation": False,
            "use_layernorm": out_mlp_layernorm,
            "use_batchnorm": out_mlp_batchnorm,
        }
        n_mlp_head = 3 if use_agent_type else 1
        self.mlp_pred = MLPEnsemble(
            n_decoders=n_mlp_head, decoder_cfg=cfg_mlp_pred, use_vmap=use_vmap
        )

        cfg_mlp_conf["fc_dims"] = [hidden_dim, _d, _d, 1]
        self.mlp_conf = MLPEnsemble(
            n_decoders=n_mlp_head, decoder_cfg=cfg_mlp_conf, use_vmap=use_vmap
        )

    def forward(self, valid: Tensor, emb: Tensor, agent_type: Tensor) -> Tuple[Tensor, Tensor]:
        pred = self.mlp_pred(x=emb, valid_mask=valid.unsqueeze(-1))
        conf = self.mlp_conf(x=emb, valid_mask=valid.unsqueeze(-1)).squeeze(-1)

        if self.use_agent_type:
            _type = agent_type.movedim(-1, 0).unsqueeze(-1)
            pred = (pred * _type.unsqueeze(-1)).sum(0)
            conf = (conf * _type).sum(0)
        else:
            pred = pred.squeeze(0)
            conf = conf.squeeze(0)

        n_scene, n_agent, n_pred = conf.shape
        return conf, pred.view(n_scene, n_agent, n_pred, self.n_step_future, self.pred_dim)


class MLPEnsemble(nn.Module):
    def __init__(self, n_decoders: int, decoder_cfg, use_vmap: bool) -> None:
        super().__init__()
        self.use_vmap = use_vmap
        self.n_decoders = n_decoders
        if self.use_vmap and self.n_decoders > 1:
            from functorch import combine_state_for_ensemble, vmap

            _decoders = [MLP(**decoder_cfg) for _ in range(n_decoders)]
            fmodel_decoders, params_decoders, buffers_decoders = combine_state_for_ensemble(
                _decoders
            )
            assert buffers_decoders == ()
            self.v_model = vmap(fmodel_decoders, randomness="different")
            [p.requires_grad_() for p in params_decoders]
            self.params_decoders = nn.ParameterList(params_decoders)
        else:
            self._decoders = nn.ModuleList([MLP(**decoder_cfg) for _ in range(n_decoders)])

    def forward(self, **kwargs) -> Tensor:
        if self.use_vmap and self.n_decoders > 1:
            out = self.v_model(tuple(self.params_decoders), (), **kwargs)
        else:
            out = []
            for decoder in self._decoders:
                x = decoder(**kwargs)
                out.append(x)
            out = torch.stack(out, dim=0)
        return out


# ---------------------------------------------------------------------------
# src/models/sc_relative.py
# ---------------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        use_vmap: bool,
        d_rpe: int,
        n_pred: int,
        mlp_head,
        multi_modal_anchors,
        tf_n_layer: int,
        tf_cfg,
        agent_attr_dim: int,
        k_reinforce_tl: float,
        k_reinforce_agent: float,
        k_reinforce_anchor: float,
        k_reinforce_all: float,
        n_latent_query: float,
        latent_query_use_tf_decoder: bool,
        latent_query,
        use_attr_for_multi_modal: bool,
        anchor_self_attn: bool,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_pred = n_pred
        self.k_reinforce_tl = k_reinforce_tl
        self.k_reinforce_agent = k_reinforce_agent
        self.k_reinforce_anchor = k_reinforce_anchor
        self.k_reinforce_all = k_reinforce_all
        self.n_latent_query = n_latent_query
        self.use_attr_for_multi_modal = use_attr_for_multi_modal
        self.anchor_self_attn = anchor_self_attn
        self.latent_query_use_tf_decoder = latent_query_use_tf_decoder

        if self.k_reinforce_tl > 0:
            self.tf_reinforce_tl = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        d_rpe=d_rpe,
                        decoder_self_attn=True,
                        **tf_cfg,
                    )
                    for _ in range(tf_n_layer)
                ]
            )

        if self.k_reinforce_agent > 0:
            self.tf_reinforce_agent = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        d_rpe=d_rpe,
                        decoder_self_attn=True,
                        **tf_cfg,
                    )
                    for _ in range(tf_n_layer)
                ]
            )

        if self.k_reinforce_all > 0:
            self.tf_reinforce_all = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        d_rpe=d_rpe,
                        decoder_self_attn=False,
                        **tf_cfg,
                    )
                    for _ in range(tf_n_layer)
                ]
            )

        emb_dim = agent_attr_dim if self.use_attr_for_multi_modal else hidden_dim
        self.anchors = MultiModalAnchors(
            hidden_dim=hidden_dim, emb_dim=emb_dim, n_pred=n_pred, **multi_modal_anchors
        )
        self.mlp_head = MLPHead(hidden_dim=hidden_dim, use_vmap=use_vmap, n_pred=n_pred, **mlp_head)

        if self.k_reinforce_anchor > 0:
            if self.n_latent_query > 0:
                self.latent_query = MultiModalAnchors(
                    hidden_dim=hidden_dim,
                    emb_dim=emb_dim,
                    n_pred=self.n_latent_query,
                    **latent_query,
                )
                if self.latent_query_use_tf_decoder:
                    self.tf_latent_query = TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        d_rpe=d_rpe,
                        n_layer=tf_n_layer,
                        decoder_self_attn=True,
                        **tf_cfg,
                    )
                else:
                    self.tf_latent_cross = TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        d_rpe=d_rpe,
                        n_layer=1,
                        **tf_cfg,
                    )
                    self.tf_latent_self = TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        n_layer=tf_n_layer,
                        **tf_cfg,
                    )

                self.tf_reinforce_anchor = TransformerBlock(
                    d_model=hidden_dim,
                    d_feedforward=hidden_dim * 4,
                    n_layer=tf_n_layer,
                    decoder_self_attn=anchor_self_attn,
                    **tf_cfg,
                )
            else:
                self.tf_reinforce_anchor = TransformerBlock(
                    d_model=hidden_dim,
                    d_feedforward=hidden_dim * 4,
                    d_rpe=d_rpe,
                    n_layer=tf_n_layer,
                    decoder_self_attn=anchor_self_attn,
                    **tf_cfg,
                )
        else:
            if self.anchor_self_attn:
                self.tf_anchor_self = TransformerBlock(
                    d_model=hidden_dim,
                    d_feedforward=hidden_dim * 4,
                    n_layer=tf_n_layer,
                    decoder_self_attn=False,
                    **tf_cfg,
                )

    def forward(
        self,
        agent_type: Tensor,
        agent_valid: Tensor,
        agent_attr: Tensor,
        agent_emb: Tensor,
        tl_valid: Tensor,
        tl_emb: Tensor,
        map_valid: Tensor,
        map_emb: Tensor,
        knn_idx_tl2self: Optional[Tensor],
        knn_invalid_tl2self: Optional[Tensor],
        knn_rpe_tl2self: Optional[Tensor],
        knn_idx_tl2map: Optional[Tensor],
        knn_invalid_tl2map: Optional[Tensor],
        knn_rpe_tl2map: Optional[Tensor],
        knn_idx_agent2self: Optional[Tensor],
        knn_invalid_agent2self: Optional[Tensor],
        knn_rpe_agent2self: Optional[Tensor],
        knn_idx_agent2maptl: Optional[Tensor],
        knn_invalid_agent2maptl: Optional[Tensor],
        knn_rpe_agent2maptl: Optional[Tensor],
        knn_idx_agent2all: Optional[Tensor],
        knn_invalid_agent2all: Optional[Tensor],
        knn_rpe_agent2all: Optional[Tensor],
        knn_idx_all2all: Optional[Tensor],
        knn_invalid_all2all: Optional[Tensor],
        knn_rpe_all2all: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        n_scene, n_agent = agent_valid.shape
        n_tl = tl_valid.shape[1]
        n_map = map_valid.shape[1]
        _idx_scene = torch.arange(n_scene)[:, None, None]
        _idx_agent = torch.arange(n_agent)[None, :, None]

        if self.k_reinforce_tl > 0:
            _idx_tl = torch.arange(n_tl)[None, :, None]
            tl_invalid = ~tl_valid
            _tgt = map_emb.unsqueeze(1).expand(-1, n_tl, -1, -1)
            if knn_idx_tl2map is not None:
                _tgt = _tgt[_idx_scene, _idx_tl, knn_idx_tl2map]
            for mod in self.tf_reinforce_tl:
                _decoder_tgt = tl_emb.unsqueeze(1).expand(-1, n_tl, -1, -1)
                if knn_idx_tl2self is not None:
                    _decoder_tgt = _decoder_tgt[_idx_scene, _idx_tl, knn_idx_tl2self]
                tl_emb, _ = mod(
                    src=tl_emb,
                    src_padding_mask=tl_invalid,
                    tgt=_tgt,
                    tgt_padding_mask=knn_invalid_tl2map,
                    rpe=knn_rpe_tl2map,
                    decoder_tgt=_decoder_tgt,
                    decoder_tgt_padding_mask=knn_invalid_tl2self,
                    decoder_rpe=knn_rpe_tl2self,
                )

        if self.k_reinforce_agent > 0:
            agent_invalid = ~agent_valid
            _tgt = torch.cat([map_emb, tl_emb], dim=1).unsqueeze(1).expand(-1, n_agent, -1, -1)
            if knn_idx_agent2maptl is not None:
                _tgt = _tgt[_idx_scene, _idx_agent, knn_idx_agent2maptl]
            for mod in self.tf_reinforce_agent:
                _decoder_tgt = agent_emb.unsqueeze(1).expand(-1, n_agent, -1, -1)
                if knn_idx_agent2self is not None:
                    _decoder_tgt = _decoder_tgt[_idx_scene, _idx_agent, knn_idx_agent2self]
                agent_emb, _ = mod(
                    src=agent_emb,
                    src_padding_mask=agent_invalid,
                    tgt=_tgt,
                    tgt_padding_mask=knn_invalid_agent2maptl,
                    rpe=knn_rpe_agent2maptl,
                    decoder_tgt=_decoder_tgt,
                    decoder_tgt_padding_mask=knn_invalid_agent2self,
                    decoder_rpe=knn_rpe_agent2self,
                )

        if self.k_reinforce_all > 0:
            _emb = torch.cat([map_emb, tl_emb, agent_emb], dim=1)
            _emb_invalid = ~torch.cat([map_valid, tl_valid, agent_valid], dim=1)
            n_emb = n_map + n_tl + n_agent
            _idx_all = torch.arange(n_emb)[None, :, None]
            for mod in self.tf_reinforce_all:
                _tgt = _emb.unsqueeze(1).expand(-1, n_emb, -1, -1)
                if knn_idx_all2all is not None:
                    _tgt = _tgt[_idx_scene, _idx_all, knn_idx_all2all]
                _emb, _ = mod(
                    src=_emb,
                    src_padding_mask=_emb_invalid,
                    tgt=_tgt,
                    tgt_padding_mask=knn_invalid_all2all,
                    rpe=knn_rpe_all2all,
                )
            map_emb = _emb[:, :n_map]
            tl_emb = _emb[:, n_map : n_map + n_tl]
            agent_emb = _emb[:, -n_agent:]

        anchor_emb = agent_attr if self.use_attr_for_multi_modal else agent_emb
        anchor_emb = self.anchors(
            agent_valid.flatten(0, 1), anchor_emb.flatten(0, 1), agent_type.flatten(0, 1)
        )
        if self.k_reinforce_anchor > 0:
            if self.n_latent_query > 0:
                ctx_emb = agent_attr if self.use_attr_for_multi_modal else agent_emb
                ctx_emb = self.latent_query(
                    agent_valid.flatten(0, 1), ctx_emb.flatten(0, 1), agent_type.flatten(0, 1)
                )
                _tgt = (
                    torch.cat([map_emb, tl_emb, agent_emb], dim=1)
                    .unsqueeze(1)
                    .expand(-1, n_agent, -1, -1)
                )
                if knn_idx_agent2all is not None:
                    _tgt = _tgt[_idx_scene, _idx_agent, knn_idx_agent2all]

                if self.latent_query_use_tf_decoder:
                    ctx_emb, _ = self.tf_latent_query(
                        src=ctx_emb,
                        tgt=_tgt.flatten(0, 1).unsqueeze(1).expand(-1, self.n_latent_query, -1, -1),
                        tgt_padding_mask=knn_invalid_agent2all.flatten(0, 1)
                        .unsqueeze(1)
                        .expand(-1, self.n_latent_query, -1),
                        rpe=knn_rpe_agent2all.flatten(0, 1)
                        .unsqueeze(1)
                        .expand(-1, self.n_latent_query, -1, -1),
                    )
                else:
                    ctx_emb, _ = self.tf_latent_cross(
                        src=ctx_emb,
                        tgt=_tgt.flatten(0, 1).unsqueeze(1).expand(-1, self.n_latent_query, -1, -1),
                        tgt_padding_mask=knn_invalid_agent2all.flatten(0, 1)
                        .unsqueeze(1)
                        .expand(-1, self.n_latent_query, -1),
                        rpe=knn_rpe_agent2all.flatten(0, 1)
                        .unsqueeze(1)
                        .expand(-1, self.n_latent_query, -1, -1),
                    )
                    ctx_emb, _ = self.tf_latent_self(src=ctx_emb, tgt=ctx_emb)

                anchor_emb, _ = self.tf_reinforce_anchor(src=anchor_emb, tgt=ctx_emb)
            else:
                ctx_emb = (
                    torch.cat([map_emb, tl_emb, agent_emb], dim=1)
                    .unsqueeze(1)
                    .expand(-1, n_agent, -1, -1)
                )
                if knn_idx_agent2all is not None:
                    ctx_emb = ctx_emb[_idx_scene, _idx_agent, knn_idx_agent2all]
                anchor_emb, _ = self.tf_reinforce_anchor(
                    src=anchor_emb,
                    tgt=ctx_emb.flatten(0, 1).unsqueeze(1).expand(-1, self.n_pred, -1, -1),
                    tgt_padding_mask=knn_invalid_agent2all.flatten(0, 1)
                    .unsqueeze(1)
                    .expand(-1, self.n_pred, -1),
                    rpe=knn_rpe_agent2all.flatten(0, 1)
                    .unsqueeze(1)
                    .expand(-1, self.n_pred, -1, -1),
                )
        else:
            if self.anchor_self_attn:
                anchor_emb, _ = self.tf_anchor_self(src=anchor_emb, tgt=anchor_emb)

        anchor_emb = anchor_emb.view(n_scene, n_agent, self.n_pred, self.hidden_dim)
        conf, pred = self.mlp_head(agent_valid, anchor_emb, agent_type)

        return conf, pred


class IntraClassEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        agent_attr_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        pl_aggr: bool,
        pose_rpe: nn.Module,
        tf_cfg,
        n_tgt_knn: int,
        n_layer_mlp: int,
        mlp_cfg,
        n_layer_tf_map: int,
        n_layer_tf_tl: int,
        n_layer_tf_agent: int,
    ) -> None:
        super().__init__()
        self.pl_aggr = pl_aggr
        self.n_tgt_knn = n_tgt_knn
        self.pose_rpe = pose_rpe

        self.fc_tl = MLP([tl_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)
        if self.pl_aggr:
            self.fc_map = MLP([map_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)
            self.fc_agent = MLP([agent_attr_dim] + [hidden_dim] * n_layer_mlp, **mlp_cfg)
        else:
            self.point_net_map = PointNet(map_attr_dim, hidden_dim, n_layer=n_layer_mlp, **mlp_cfg)
            self.point_net_agent = PointNet(
                agent_attr_dim, hidden_dim, n_layer=n_layer_mlp, **mlp_cfg
            )

        self.tf_map = None
        self.tf_tl = None
        self.tf_agent = None
        if n_layer_tf_map > 0:
            self.tf_map = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        d_rpe=self.pose_rpe.out_dim,
                        **tf_cfg,
                    )
                    for _ in range(n_layer_tf_map)
                ]
            )
        if n_layer_tf_tl > 0:
            self.tf_tl = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        d_rpe=self.pose_rpe.out_dim,
                        **tf_cfg,
                    )
                    for _ in range(n_layer_tf_tl)
                ]
            )
        if n_layer_tf_agent > 0:
            self.tf_agent = nn.ModuleList(
                [
                    TransformerBlock(
                        d_model=hidden_dim,
                        d_feedforward=hidden_dim * 4,
                        d_rpe=self.pose_rpe.out_dim,
                        **tf_cfg,
                    )
                    for _ in range(n_layer_tf_agent)
                ]
            )

    def forward(
        self,
        inference_repeat_n: int,
        inference_cache_map: bool,
        agent_valid: Tensor,
        agent_attr: Tensor,
        map_valid: Tensor,
        map_attr: Tensor,
        tl_valid: Tensor,
        tl_attr: Tensor,
        rel_pose: Tensor,
        rel_dist: Tensor,
        dist_limit_map: float,
        dist_limit_tl: float,
        dist_limit_agent: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        for _ in range(inference_repeat_n):
            n_scene, n_tl = tl_valid.shape
            n_map = map_valid.shape[1]
            n_agent = agent_valid.shape[1]
            _idx_scene = torch.arange(n_scene)[:, None, None]

        _n_repeat_map = 1 if inference_cache_map else inference_repeat_n
        for _ in range(_n_repeat_map):
            map_emb, map_valid_reduced = self._mlp_map(map_attr, map_valid)
            if self.tf_map is not None:
                _map_invalid = ~map_valid_reduced
                _map_idx_knn, _map_invalid_knn, _map_rpe_knn = get_tgt_knn_idx(
                    _map_invalid,
                    rel_pose[:, :n_map, :n_map],
                    rel_dist[:, :n_map, :n_map],
                    self.n_tgt_knn,
                    dist_limit=dist_limit_map,
                )
                _rpe = self.pose_rpe(xy=_map_rpe_knn[..., :2], dir=_map_rpe_knn[..., [2]])
                _idx_map = torch.arange(n_map)[None, :, None]
                for mod in self.tf_map:
                    _tgt = map_emb.unsqueeze(1).expand(-1, n_map, -1, -1)
                    if _map_idx_knn is not None:
                        _tgt = _tgt[_idx_scene, _idx_map, _map_idx_knn]
                    map_emb, _ = mod(
                        src=map_emb,
                        src_padding_mask=_map_invalid,
                        tgt=_tgt,
                        tgt_padding_mask=_map_invalid_knn,
                        rpe=_rpe,
                    )

        for _ in range(inference_repeat_n):
            tl_emb = self.fc_tl(tl_attr, tl_valid)
            if self.tf_tl is not None:
                _tl_invalid = ~tl_valid
                _tl_idx_knn, _tl_invalid_knn, _tl_rpe_knn = get_tgt_knn_idx(
                    _tl_invalid,
                    rel_pose[:, n_map : n_map + n_tl, n_map : n_map + n_tl],
                    rel_dist[:, n_map : n_map + n_tl, n_map : n_map + n_tl],
                    self.n_tgt_knn,
                    dist_limit=dist_limit_tl,
                )
                _rpe = self.pose_rpe(xy=_tl_rpe_knn[..., :2], dir=_tl_rpe_knn[..., [2]])
                _idx_tl = torch.arange(n_tl)[None, :, None]
                for mod in self.tf_tl:
                    _tgt = tl_emb.unsqueeze(1).expand(-1, n_tl, -1, -1)
                    if _tl_idx_knn is not None:
                        _tgt = _tgt[_idx_scene, _idx_tl, _tl_idx_knn]
                    tl_emb, _ = mod(
                        src=tl_emb,
                        src_padding_mask=_tl_invalid,
                        tgt=_tgt,
                        tgt_padding_mask=_tl_invalid_knn,
                        rpe=_rpe,
                    )

        for _ in range(inference_repeat_n):
            agent_emb, agent_valid_reduced = self._mlp_agent(agent_attr, agent_valid)
            if self.tf_agent is not None:
                _agent_invalid = ~agent_valid_reduced
                _agent_idx_knn, _agent_invalid_knn, _agent_rpe_knn = get_tgt_knn_idx(
                    _agent_invalid,
                    rel_pose[:, -n_agent:, -n_agent:],
                    rel_dist[:, -n_agent:, -n_agent:],
                    self.n_tgt_knn,
                    dist_limit=dist_limit_agent,
                )
                _rpe = self.pose_rpe(xy=_agent_rpe_knn[..., :2], dir=_agent_rpe_knn[..., [2]])
                _idx_agent = torch.arange(n_agent)[None, :, None]
                for mod in self.tf_agent:
                    _tgt = agent_emb.unsqueeze(1).expand(-1, n_agent, -1, -1)
                    if _agent_idx_knn is not None:
                        _tgt = _tgt[_idx_scene, _idx_agent, _agent_idx_knn]
                    agent_emb, _ = mod(
                        src=agent_emb,
                        src_padding_mask=_agent_invalid,
                        tgt=_tgt,
                        tgt_padding_mask=_agent_invalid_knn,
                        rpe=_rpe,
                    )

        return map_emb, map_valid_reduced, tl_emb, tl_valid, agent_emb, agent_valid_reduced

    def _mlp_agent(self, agent_attr: Tensor, agent_valid: Tensor) -> Tuple[Tensor, Tensor]:
        if self.pl_aggr:
            agent_emb = self.fc_agent(agent_attr, agent_valid)
        else:
            agent_emb, agent_valid = self.point_net_agent(agent_attr, agent_valid)
        return agent_emb, agent_valid

    def _mlp_map(self, map_attr: Tensor, map_valid: Tensor) -> Tuple[Tensor, Tensor]:
        if self.pl_aggr:
            map_emb = self.fc_map(map_attr, map_valid)
        else:
            map_emb, map_valid = self.point_net_map(map_attr, map_valid)
        return map_emb, map_valid


class SceneCentricRelative(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        agent_attr_dim: int,
        map_attr_dim: int,
        tl_attr_dim: int,
        pl_aggr: bool,
        n_tgt_knn: int,
        tf_cfg,
        intra_class_encoder,
        decoder_remove_ego_agent: bool,
        n_decoders: int,
        decoder,
        rpe_mode: str,
        dist_limit_map: float = 2000,
        dist_limit_tl: float = 2000,
        dist_limit_agent: List[float] = [2000, 2000, 2000],
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_pred = decoder["n_pred"]
        self.n_decoders = n_decoders
        self.decoder_remove_ego_agent = decoder_remove_ego_agent
        self.decoder_k_reinforce_tl = decoder["k_reinforce_tl"]
        self.decoder_k_reinforce_agent = decoder["k_reinforce_agent"]
        self.decoder_k_reinforce_anchor = decoder["k_reinforce_anchor"]
        self.decoder_k_reinforce_all = decoder["k_reinforce_all"]
        self.n_tgt_knn = n_tgt_knn
        self.pl_aggr = pl_aggr
        self.dist_limit_map = dist_limit_map
        self.dist_limit_tl = dist_limit_tl
        self.dist_limit_agent = dist_limit_agent

        assert rpe_mode in ["xy_dir", "pe_xy_dir", "pe_xy_yaw"]
        self.pose_rpe = PosePE(rpe_mode, pe_dim=hidden_dim)

        self.intra_class_encoder = IntraClassEncoder(
            hidden_dim=hidden_dim,
            agent_attr_dim=agent_attr_dim,
            map_attr_dim=map_attr_dim,
            tl_attr_dim=tl_attr_dim,
            pl_aggr=pl_aggr,
            pose_rpe=self.pose_rpe,
            tf_cfg=tf_cfg,
            n_tgt_knn=n_tgt_knn,
            **intra_class_encoder,
        )

        decoder = dict(decoder)
        decoder["hidden_dim"] = hidden_dim
        decoder["d_rpe"] = self.pose_rpe.out_dim
        decoder["agent_attr_dim"] = agent_attr_dim
        decoder["tf_cfg"] = tf_cfg
        self.decoder = DecoderEnsemble(n_decoders, decoder_cfg=decoder)

    def forward(
        self,
        agent_valid: Tensor,
        agent_type: Tensor,
        agent_attr: Tensor,
        agent_pose: Tensor,
        map_valid: Tensor,
        map_attr: Tensor,
        map_pose: Tensor,
        tl_valid: Tensor,
        tl_attr: Tensor,
        tl_pose: Tensor,
        inference_repeat_n: int = 1,
        inference_cache_map: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        dist_limit_agent = 0
        for i in range(agent_type.shape[-1]):
            dist_limit_agent += agent_type[:, :, i] * self.dist_limit_agent[i]
        dist_limit_agent = dist_limit_agent.unsqueeze(-1)

        tl_valid = tl_valid.flatten(1, 2)
        tl_attr = tl_attr.flatten(1, 2)
        tl_pose = tl_pose.flatten(1, 2)
        for _ in range(inference_repeat_n):
            if self.pl_aggr:
                emb_invalid = ~torch.cat([map_valid, tl_valid, agent_valid], dim=1)
            else:
                emb_invalid = ~torch.cat([map_valid.any(-1), tl_valid, agent_valid.any(-1)], dim=1)
            rel_pose, rel_dist = get_rel_pose(
                torch.cat([map_pose, tl_pose, agent_pose], dim=1), emb_invalid
            )

        map_emb, map_valid, tl_emb, tl_valid, agent_emb, agent_valid = self.intra_class_encoder(
            inference_repeat_n=inference_repeat_n,
            inference_cache_map=inference_cache_map,
            agent_valid=agent_valid,
            agent_attr=agent_attr,
            map_valid=map_valid,
            map_attr=map_attr,
            tl_valid=tl_valid,
            tl_attr=tl_attr,
            rel_pose=rel_pose,
            rel_dist=rel_dist,
            dist_limit_map=self.dist_limit_map,
            dist_limit_tl=self.dist_limit_tl,
            dist_limit_agent=dist_limit_agent,
        )

        for _ in range(inference_repeat_n):
            n_map = map_valid.shape[1]
            n_tl = tl_valid.shape[1]
            n_agent = agent_valid.shape[1]

            if self.decoder_remove_ego_agent:
                rel_dist[:, range(-n_agent, 0), range(-n_agent, 0)] += float("inf")

            if self.decoder_k_reinforce_tl > 0:
                knn_idx_tl2self, knn_invalid_tl2self, knn_rpe_tl2self = get_tgt_knn_idx(
                    tgt_invalid=emb_invalid[:, n_map : n_map + n_tl],
                    rel_pose=rel_pose[:, n_map : n_map + n_tl, n_map : n_map + n_tl],
                    rel_dist=rel_dist[:, n_map : n_map + n_tl, n_map : n_map + n_tl],
                    n_tgt_knn=self.n_tgt_knn,
                    dist_limit=self.dist_limit_tl,
                )
                knn_idx_tl2map, knn_invalid_tl2map, knn_rpe_tl2map = get_tgt_knn_idx(
                    tgt_invalid=emb_invalid[:, :n_map],
                    rel_pose=rel_pose[:, n_map : n_map + n_tl, :n_map],
                    rel_dist=rel_dist[:, n_map : n_map + n_tl, :n_map],
                    n_tgt_knn=int(self.n_tgt_knn * self.decoder_k_reinforce_tl),
                    dist_limit=self.dist_limit_tl,
                )
                knn_rpe_tl2self = self.pose_rpe(
                    xy=knn_rpe_tl2self[..., :2], dir=knn_rpe_tl2self[..., [2]]
                )
                knn_rpe_tl2map = self.pose_rpe(
                    xy=knn_rpe_tl2map[..., :2], dir=knn_rpe_tl2map[..., [2]]
                )
            else:
                knn_idx_tl2self = None
                knn_invalid_tl2self = None
                knn_rpe_tl2self = None
                knn_idx_tl2map = None
                knn_invalid_tl2map = None
                knn_rpe_tl2map = None

            if self.decoder_k_reinforce_agent > 0:
                knn_idx_agent2self, knn_invalid_agent2self, knn_rpe_agent2self = get_tgt_knn_idx(
                    tgt_invalid=emb_invalid[:, -n_agent:],
                    rel_pose=rel_pose[:, -n_agent:, -n_agent:],
                    rel_dist=rel_dist[:, -n_agent:, -n_agent:],
                    n_tgt_knn=self.n_tgt_knn,
                    dist_limit=dist_limit_agent,
                )
                knn_idx_agent2maptl, knn_invalid_agent2maptl, knn_rpe_agent2maptl = get_tgt_knn_idx(
                    tgt_invalid=emb_invalid[:, : n_map + n_tl],
                    rel_pose=rel_pose[:, -n_agent:, : n_map + n_tl],
                    rel_dist=rel_dist[:, -n_agent:, : n_map + n_tl],
                    n_tgt_knn=int(self.n_tgt_knn * self.decoder_k_reinforce_agent),
                    dist_limit=dist_limit_agent,
                )
                knn_rpe_agent2self = self.pose_rpe(
                    xy=knn_rpe_agent2self[..., :2], dir=knn_rpe_agent2self[..., [2]]
                )
                knn_rpe_agent2maptl = self.pose_rpe(
                    xy=knn_rpe_agent2maptl[..., :2], dir=knn_rpe_agent2maptl[..., [2]]
                )
            else:
                knn_idx_agent2self = None
                knn_invalid_agent2self = None
                knn_rpe_agent2self = None
                knn_idx_agent2maptl = None
                knn_invalid_agent2maptl = None
                knn_rpe_agent2maptl = None

            if self.decoder_k_reinforce_anchor:
                knn_idx_agent2all, knn_invalid_agent2all, knn_rpe_agent2all = get_tgt_knn_idx(
                    tgt_invalid=emb_invalid,
                    rel_pose=rel_pose[:, -n_agent:],
                    rel_dist=rel_dist[:, -n_agent:],
                    n_tgt_knn=int(self.n_tgt_knn * self.decoder_k_reinforce_anchor),
                    dist_limit=dist_limit_agent,
                )
                knn_rpe_agent2all = self.pose_rpe(
                    xy=knn_rpe_agent2all[..., :2], dir=knn_rpe_agent2all[..., [2]]
                )
            else:
                knn_idx_agent2all = None
                knn_invalid_agent2all = None
                knn_rpe_agent2all = None

            if self.decoder_k_reinforce_all > 0:
                knn_idx_all2all, knn_invalid_all2all, knn_rpe_all2all = get_tgt_knn_idx(
                    tgt_invalid=emb_invalid,
                    rel_pose=rel_pose,
                    rel_dist=rel_dist,
                    n_tgt_knn=int(self.n_tgt_knn * self.decoder_k_reinforce_all),
                    dist_limit=self.dist_limit_map,
                )
                knn_rpe_all2all = self.pose_rpe(
                    xy=knn_rpe_all2all[..., :2], dir=knn_rpe_all2all[..., [2]]
                )
            else:
                knn_idx_all2all = None
                knn_invalid_all2all = None
                knn_rpe_all2all = None

            conf, pred = self.decoder(
                agent_type=agent_type,
                agent_valid=agent_valid,
                agent_attr=agent_attr,
                agent_emb=agent_emb,
                tl_valid=tl_valid,
                tl_emb=tl_emb,
                map_valid=map_valid,
                map_emb=map_emb,
                knn_idx_tl2self=knn_idx_tl2self,
                knn_invalid_tl2self=knn_invalid_tl2self,
                knn_rpe_tl2self=knn_rpe_tl2self,
                knn_idx_tl2map=knn_idx_tl2map,
                knn_invalid_tl2map=knn_invalid_tl2map,
                knn_rpe_tl2map=knn_rpe_tl2map,
                knn_idx_agent2self=knn_idx_agent2self,
                knn_invalid_agent2self=knn_invalid_agent2self,
                knn_rpe_agent2self=knn_rpe_agent2self,
                knn_idx_agent2maptl=knn_idx_agent2maptl,
                knn_invalid_agent2maptl=knn_invalid_agent2maptl,
                knn_rpe_agent2maptl=knn_rpe_agent2maptl,
                knn_idx_agent2all=knn_idx_agent2all,
                knn_invalid_agent2all=knn_invalid_agent2all,
                knn_rpe_agent2all=knn_rpe_agent2all,
                knn_idx_all2all=knn_idx_all2all,
                knn_invalid_all2all=knn_invalid_all2all,
                knn_rpe_all2all=knn_rpe_all2all,
            )

        assert torch.isfinite(conf).all()
        assert torch.isfinite(pred).all()
        return agent_valid, conf, pred


# ---------------------------------------------------------------------------
# menagerie staging entry points
# ---------------------------------------------------------------------------
MENAGERIE_ZOO = "vendored-pytorch"

# `hydra.utils.instantiate` needs a real importable dotted path for `_target_`; this module
# registers itself under a fixed name in sys.modules so `_target_: "_hptr_staging_mod.Decoder"`
# resolves for real (see module header "Import-only fixes applied").
_STAGING_MODULE_NAME = "_hptr_staging_mod"
sys.modules[_STAGING_MODULE_NAME] = sys.modules[__name__]


def build_hptr():
    # Mirrors configs/model/scr_av2.yaml, sized down for a fast-tracing smoke config
    # (hidden_dim 256 -> 16; n_tgt_knn 36 -> 4; n_layer_tf_map 6 -> 1; all other structural
    # flags kept as-shipped so every branch -- tl/agent/anchor reinforcement, anchor
    # self-attention, KNN-vs-full attention -- is exercised).
    tf_cfg = {"n_head": 2, "dropout_p": 0.0, "norm_first": True, "apply_q_rpe": False, "bias": True}
    mlp_cfg = {
        "end_layer_activation": True,
        "use_layernorm": False,
        "use_batchnorm": False,
        "dropout_p": None,
    }
    intra_class_encoder = {
        "n_layer_mlp": 2,
        "mlp_cfg": mlp_cfg,
        "n_layer_tf_map": 1,
        "n_layer_tf_tl": -1,
        "n_layer_tf_agent": -1,
    }
    decoder_cfg = OmegaConf.create(
        {
            "_target_": f"{_STAGING_MODULE_NAME}.Decoder",
            "n_pred": 3,
            "tf_n_layer": 1,
            "k_reinforce_tl": 1,
            "k_reinforce_agent": 1,
            "k_reinforce_all": -1,
            "k_reinforce_anchor": 2,
            "n_latent_query": -1,
            "latent_query": {
                "use_agent_type": False,
                "mode_emb": "linear",
                "mode_init": "randn",
                "scale": 5.0,
            },
            "latent_query_use_tf_decoder": False,
            "multi_modal_anchors": {
                "use_agent_type": True,
                "mode_emb": "linear",
                "mode_init": "randn",
                "scale": 5.0,
            },
            "anchor_self_attn": True,
            "mlp_head": {
                "predictions": ["pos", "cov3", "spd", "vel", "yaw_bbox"],
                "use_agent_type": False,
                "flatten_conf_head": False,
                "out_mlp_layernorm": False,
                "out_mlp_batchnorm": False,
                "n_step_future": 8,
            },
            "use_attr_for_multi_modal": False,
            "use_vmap": True,
        }
    )
    return SceneCentricRelative(
        hidden_dim=16,
        agent_attr_dim=5,
        map_attr_dim=4,
        tl_attr_dim=6,
        pl_aggr=False,
        n_tgt_knn=4,
        tf_cfg=tf_cfg,
        intra_class_encoder=intra_class_encoder,
        decoder_remove_ego_agent=False,
        n_decoders=1,
        decoder=decoder_cfg,
        rpe_mode="pe_xy_yaw",
        dist_limit_map=1500,
        dist_limit_tl=1000,
        dist_limit_agent=[1500, 500, 1000],
    )


def example_input_hptr():
    # Small synthetic scene-centric-relative batch: 2 scenes, 4 agents, 3 map polylines (5
    # nodes each), 2 traffic lights over 3 historical steps. Field layout/shapes mirror
    # SceneCentricRelative.forward()'s docstring.
    torch.manual_seed(0)
    n_scene = 2
    n_agent = 4
    n_step_hist_agent = 3
    agent_attr_dim = 5
    n_map = 3
    n_pl_node = 5
    map_attr_dim = 4
    n_step_hist_tl = 3
    n_tl = 2
    tl_attr_dim = 6

    agent_valid = torch.ones(n_scene, n_agent, n_step_hist_agent, dtype=torch.bool)
    agent_type = F.one_hot(torch.randint(0, 3, (n_scene, n_agent)), num_classes=3).bool()
    agent_attr = torch.randn(n_scene, n_agent, n_step_hist_agent, agent_attr_dim)
    agent_pose = torch.randn(n_scene, n_agent, 3)

    map_valid = torch.ones(n_scene, n_map, n_pl_node, dtype=torch.bool)
    map_attr = torch.randn(n_scene, n_map, n_pl_node, map_attr_dim)
    map_pose = torch.randn(n_scene, n_map, 3)

    tl_valid = torch.ones(n_scene, n_step_hist_tl, n_tl, dtype=torch.bool)
    tl_attr = torch.randn(n_scene, n_step_hist_tl, n_tl, tl_attr_dim)
    tl_pose = torch.randn(n_scene, n_step_hist_tl, n_tl, 3)

    return (
        agent_valid,
        agent_type,
        agent_attr,
        agent_pose,
        map_valid,
        map_attr,
        map_pose,
        tl_valid,
        tl_attr,
        tl_pose,
    )


MENAGERIE_ENTRIES = [
    ("HPTR", build_hptr, example_input_hptr, 2023, MENAGERIE_ZOO),
]
