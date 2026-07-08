# SOURCE: vendored from ZrrSkywalker/MonoDETR @ 6994b9f512400b258c6edb75f77423beb9c126f2
# (https://github.com/ZrrSkywalker/MonoDETR, "MonoDETR: Depth-aware Transformer for
# Monocular 3D Object Detection", ICCV 2023)
#
# Files vendored (near-verbatim, only import paths/dead-branch pruning adjusted):
#   - lib/models/monodetr/backbone.py             (ResNet + FrozenBatchNorm2d backbone,
#                                                    torchvision-based)
#   - lib/models/monodetr/position_encoding.py     (sine/learned position embeddings)
#   - lib/models/monodetr/depthaware_transformer.py (VisualEncoder/DepthAwareDecoder,
#                                                      the depth-aware deformable
#                                                      transformer)
#   - lib/models/monodetr/depth_predictor/depth_predictor.py + transformer.py
#     (foreground depth map + depth-bin classifier + depth positional encoder)
#   - lib/models/monodetr/ops/modules/ms_deform_attn.py (MSDeformAttn module --
#                                                          only the class actually
#                                                          instantiated by the traced
#                                                          model path)
#   - lib/models/monodetr/ops/functions/ms_deform_attn_func.py (`ms_deform_attn_core_pytorch`,
#                                                                 see substitution note below)
#   - lib/models/monodetr/monodetr.py              (top-level `MonoDETR` nn.Module +
#                                                     `MLP` head; `SetCriterion` (the
#                                                     Hungarian-matcher training loss)
#                                                     and `build()` are dropped -- they
#                                                     are training-only and not part of
#                                                     the network's forward pass)
#   - utils/misc.py                                (`NestedTensor` + distributed-safe
#                                                     helpers used by the backbone)
#
# Environment substitution (NOT an architecture change): the upstream `MSDeformAttn`
# module calls `MSDeformAttnFunction.apply(...)`, a custom CUDA extension
# (`MultiScaleDeformableAttention`, built via `ops/setup.py`) that cannot be compiled
# in this base environment. The *same upstream file*
# (`ops/functions/ms_deform_attn_func.py`) ships a documented pure-PyTorch reference
# implementation, `ms_deform_attn_core_pytorch` (the repo's own comment: "for debug
# and test only, need to use cuda version instead" -- i.e. the exact reference used
# to validate the CUDA kernel's numerics). This module calls that real function
# instead of `MSDeformAttnFunction.apply`; the deformable-attention math itself
# (sampling-offset/attention-weight computation, bilinear grid-sample per level) is
# unchanged. `MSDeformAttn_cross` and the vendored non-standard `MultiheadAttention`
# class from `ms_deform_attn.py` are dropped: neither is ever instantiated on the
# traced forward path (`depthaware_transformer.py` uses plain `nn.MultiheadAttention`
# for `cross_attn_depth`/`self_attn`; `MSDeformAttn_cross` has zero call sites in the
# repo's model-construction code).
#
# The `Backbone.__init__` call `getattr(torchvision.models, name)(..., pretrained=
# is_main_process())` is changed to `pretrained=False` to construct offline with
# random init (avoids a network weights download); the ResNet topology itself is
# untouched. `use_dn`/`prepare_for_dn` (denoising-training query augmentation,
# `dn_components.py`) and the Hungarian `matcher.py` are training-time-only and are
# not part of `MonoDETR.forward` when `dn_args=None` and `model.eval()` (the
# `self.training` branches that reference them are skipped), so they are not vendored.

import copy
import math
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter

# --------------------------------------------------------------------------- #
# utils/misc.py (subset actually used by backbone.py / monodetr.py)
# --------------------------------------------------------------------------- #


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process():
    return get_rank() == 0


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# --------------------------------------------------------------------------- #
# lib/models/monodetr/backbone.py
# --------------------------------------------------------------------------- #


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps if hasattr(self, "eps") else 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {"layer4": "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, images):
        xs = self.body(images)
        out = {}
        for name, x in xs.items():
            m = torch.zeros(x.shape[0], x.shape[2], x.shape[3]).to(torch.bool).to(x.device)
            out[name] = NestedTensor(x, m)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False,
            norm_layer=norm_layer,
        )
        assert name not in ("resnet18", "resnet34"), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, images):
        xs = self[0](images)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(cfg):
    position_embedding = build_position_encoding(cfg)
    return_interm_layers = cfg["masks"] or cfg["num_feature_levels"] > 1
    backbone = Backbone(
        cfg["backbone"], cfg["train_backbone"], return_interm_layers, cfg["dilation"]
    )
    model = Joiner(backbone, position_embedding)
    return model


# --------------------------------------------------------------------------- #
# lib/models/monodetr/position_encoding.py
# --------------------------------------------------------------------------- #


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the
    one used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def build_position_encoding(cfg):
    N_steps = cfg["hidden_dim"] // 2
    if cfg["position_embedding"] in ("v2", "sine"):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {cfg['position_embedding']}")
    return position_embedding


# --------------------------------------------------------------------------- #
# lib/models/monodetr/ops/functions/ms_deform_attn_func.py
# --------------------------------------------------------------------------- #


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only, the repo's own documented CPU/eager reference path
    # (used upstream to validate the CUDA kernel's numerics).
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(N_, M_ * D_, Lq_)
    )
    return output.transpose(1, 2).contiguous()


# --------------------------------------------------------------------------- #
# lib/models/monodetr/ops/modules/ms_deform_attn.py (MSDeformAttn only; see header)
# --------------------------------------------------------------------------- #


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, conditional=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads, but got {} and {}".format(d_model, n_heads)
            )
        _d_per_head = d_model // n_heads
        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.conditional = conditional
        if conditional:
            self.value_proj = nn.Linear(d_model // 2, d_model // 2)
            self.output_proj = nn.Linear(d_model // 2, d_model // 2)
        else:
            self.value_proj = nn.Linear(d_model, d_model)
            self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2) or
                                            (N, Length_{query}, n_levels, 4/6)
        :param input_flatten               (N, sum(H_l*W_l), C)
        :param input_spatial_shapes        (n_levels, 2)
        :param input_level_start_index     (n_levels,)
        :param input_padding_mask          (N, sum(H_l*W_l))
        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        if self.conditional:
            value = value.view(N, Len_in, self.n_heads, (self.d_model // 2) // self.n_heads)
        else:
            value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 6:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * (
                    reference_points[:, :, None, :, None, 2::2]
                    + reference_points[:, :, None, :, None, 3::2]
                )
                * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )
        # Environment substitution: real reference eager fallback from the same
        # upstream file, in place of the compiled CUDA extension (see header).
        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )
        output = self.output_proj(output)
        return output


# --------------------------------------------------------------------------- #
# lib/models/monodetr/depth_predictor/transformer.py
# --------------------------------------------------------------------------- #


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class DepthTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_key_padding_mask, pos):
        output = src

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class DepthTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_key_padding_mask, pos):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# --------------------------------------------------------------------------- #
# lib/models/monodetr/depth_predictor/depth_predictor.py
# --------------------------------------------------------------------------- #


class DepthPredictor(nn.Module):
    def __init__(self, model_cfg):
        """
        Initialize depth predictor and depth encoder
        Args:
            model_cfg: Depth classification network config
        """
        super().__init__()
        depth_num_bins = int(model_cfg["num_depth_bins"])
        depth_min = float(model_cfg["depth_min"])
        depth_max = float(model_cfg["depth_max"])
        self.depth_max = depth_max

        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
        bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False)

        d_model = model_cfg["hidden_dim"]
        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)), nn.GroupNorm(32, d_model)
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)), nn.GroupNorm(32, d_model)
        )

        self.depth_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
        )

        self.depth_classifier = nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=(1, 1))

        depth_encoder_layer = DepthTransformerEncoderLayer(
            d_model, nhead=8, dim_feedforward=256, dropout=0.1
        )

        self.depth_encoder = DepthTransformerEncoder(depth_encoder_layer, 1)

        self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, 256)

    def forward(self, feature, mask, pos):
        assert len(feature) == 4

        src_16 = self.proj(feature[1])
        src_32 = self.upsample(F.interpolate(feature[2], size=src_16.shape[-2:], mode="bilinear"))
        src_8 = self.downsample(feature[0])
        src = (src_8 + src_16 + src_32) / 3

        src = self.depth_head(src)
        depth_logits = self.depth_classifier(src)

        depth_probs = F.softmax(depth_logits, dim=1)
        weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)

        B, C, H, W = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        pos = pos.flatten(2).permute(2, 0, 1)

        depth_embed = self.depth_encoder(src, mask, pos)
        depth_embed = depth_embed.permute(1, 2, 0).reshape(B, C, H, W)
        depth_pos_embed_ip = self.interpolate_depth_embed(weighted_depth)
        depth_embed = depth_embed + depth_pos_embed_ip

        return depth_logits, depth_embed, weighted_depth, depth_pos_embed_ip

    def interpolate_depth_embed(self, depth):
        depth = depth.clamp(min=0, max=self.depth_max)
        pos = self.interpolate_1d(depth, self.depth_pos_embed)
        pos = pos.permute(0, 3, 1, 2)
        return pos

    def interpolate_1d(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=embed.num_embeddings - 1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta


# --------------------------------------------------------------------------- #
# lib/models/monodetr/depthaware_transformer.py
# --------------------------------------------------------------------------- #


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    elif pos_tensor.size(-1) == 6:
        for i in range(2, 6):
            embed = pos_tensor[:, :, i] * scale
            pos_embed = embed[:, :, None] / dim_t
            pos_embed = torch.stack(
                (pos_embed[:, :, 0::2].sin(), pos_embed[:, :, 1::2].cos()), dim=3
            ).flatten(2)
            if i == 2:
                pos = pos_embed
            else:
                pos = torch.cat((pos, pos_embed), dim=2)
        pos = torch.cat((pos_y, pos_x, pos), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class DepthAwareTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
        two_stage=False,
        two_stage_num_proposals=50,
        group_num=11,
        use_dab=False,
        two_stage_dino=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.use_dab = use_dab
        self.two_stage_dino = two_stage_dino
        self.group_num = group_num

        encoder_layer = VisualEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
        )
        self.encoder = VisualEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DepthAwareDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
            group_num=group_num,
        )
        self.decoder = DepthAwareDecoder(
            decoder_layer,
            num_decoder_layers,
            return_intermediate_dec,
            d_model,
            use_dab=use_dab,
            two_stage_dino=two_stage_dino,
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        elif two_stage_dino:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.tgt_embed = nn.Embedding(self.two_stage_num_proposals * group_num, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
            self.two_stage_wh_embedding = None
            self.enc_out_class_embed = None
            self.enc_out_bbox_embed = None
        else:
            if not self.use_dab:
                self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage and not self.use_dab and not self.two_stage_dino:
            nn.init.xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            nn.init.constant_(self.reference_points.bias.data, 0.0)
        nn.init.normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        proposals = proposals.sigmoid() * scale
        pos = proposals[:, :, :, None] / dim_t
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

            lr = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            tb = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            wh = torch.cat((lr, tb), -1)

            proposal = torch.cat((grid, wh), -1).view(N_, -1, 6)
            proposals.append(proposal)
            _cur += H_ * W_
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
            -1, keepdim=True
        )
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(
        self,
        srcs,
        masks,
        pos_embeds,
        query_embed=None,
        depth_pos_embed=None,
        depth_pos_embed_ip=None,
        attn_mask=None,
    ):
        if not self.two_stage_dino:
            assert self.two_stage or query_embed is not None

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)

            mask = mask.flatten(1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs[0].device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
        )
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = (
                self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            )
            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 6)
            )
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points

            topk_coords_unact_input = torch.cat(
                (
                    topk_coords_unact[..., 0:2],
                    topk_coords_unact[..., 2::2] + topk_coords_unact[..., 3::2],
                ),
                dim=-1,
            )

            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact_input))
            )
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        elif self.use_dab:
            reference_points = query_embed[..., self.d_model :].sigmoid()
            tgt = query_embed[..., : self.d_model]
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            init_reference_out = reference_points
        elif self.two_stage_dino:
            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            enc_outputs_coord_unselected = self.enc_out_bbox_embed(output_memory) + output_proposals
            if self.training:
                topk = self.two_stage_num_proposals * self.group_num
            else:
                topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]

            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 6)
            )
            refpoint_embed_ = refpoint_embed_undetach.detach()
            if self.training:
                tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1)
            else:
                tgt_ = self.tgt_embed.weight[: self.two_stage_num_proposals, None, :].repeat(
                    1, bs, 1
                )
            reference_points, tgt = refpoint_embed_, tgt_
            init_reference_out = reference_points
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        depth_pos_embed = depth_pos_embed.flatten(2).permute(2, 0, 1)
        depth_pos_embed_ip = depth_pos_embed_ip.flatten(2).permute(2, 0, 1)
        mask_depth = masks[1].flatten(1)

        hs, inter_references, inter_references_dim = self.decoder(
            tgt,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_embed if (not self.use_dab and not self.two_stage_dino) else None,
            mask_flatten,
            depth_pos_embed,
            mask_depth,
            bs=bs,
            depth_pos_embed_ip=depth_pos_embed_ip,
            pos_embeds=pos_embeds,
            attn_mask=attn_mask,
        )

        inter_references_out = inter_references
        inter_references_out_dim = inter_references_dim

        if self.two_stage:
            return (
                hs,
                init_reference_out,
                inter_references_out,
                inter_references_out_dim,
                enc_outputs_class,
                enc_outputs_coord_unact,
            )
        return hs, init_reference_out, inter_references_out, inter_references_out_dim, None, None


class VisualEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None
    ):
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src = self.forward_ffn(src)

        return src


class VisualEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None
    ):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )
        for _, layer in enumerate(self.layers):
            output = layer(
                output, pos, reference_points, spatial_shapes, level_start_index, padding_mask
            )

        return output


class DepthAwareDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        group_num=1,
    ):
        super().__init__()

        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn_depth = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_depth = nn.Dropout(dropout)
        self.norm_depth = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.group_num = group_num

        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.nhead = n_heads

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask,
        depth_pos_embed,
        mask_depth,
        bs,
        query_sine_embed=None,
        is_first=None,
        depth_pos_embed_ip=None,
        pos_embeds=None,
        self_attn_mask=None,
        query_pos_un=None,
    ):
        tgt2 = self.cross_attn_depth(
            tgt.transpose(0, 1), depth_pos_embed, depth_pos_embed, key_padding_mask=mask_depth
        )[0].transpose(0, 1)

        tgt = tgt + self.dropout_depth(tgt2)
        tgt = self.norm_depth(tgt)

        q = k = self.with_pos_embed(tgt, query_pos)

        q_content = self.sa_qcontent_proj(q)
        q_pos = self.sa_qpos_proj(q)
        k_content = self.sa_kcontent_proj(k)
        k_pos = self.sa_kpos_proj(k)
        v = self.sa_v_proj(tgt)
        q = q_content + q_pos
        k = k_content + k_pos

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = tgt.transpose(0, 1)
        num_queries = q.shape[0]

        if self.training:
            num_noise = num_queries - self.group_num * 50
            num_queries = self.group_num * 50
            q_noise = q[:num_noise].repeat(1, self.group_num, 1)
            k_noise = k[:num_noise].repeat(1, self.group_num, 1)
            v_noise = v[:num_noise].repeat(1, self.group_num, 1)
            q = q[num_noise:]
            k = k[num_noise:]
            v = v[num_noise:]
            q = torch.cat(q.split(num_queries // self.group_num, dim=0), dim=1)
            k = torch.cat(k.split(num_queries // self.group_num, dim=0), dim=1)
            v = torch.cat(v.split(num_queries // self.group_num, dim=0), dim=1)
            q = torch.cat([q_noise, q], dim=0)
            k = torch.cat([k_noise, k], dim=0)
            v = torch.cat([v_noise, v], dim=0)

        tgt2 = self.self_attn(q, k, v)[0]
        if self.training:
            tgt2 = torch.cat(tgt2.split(bs, dim=1), dim=0).transpose(0, 1)
        else:
            tgt2 = tgt2.transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt = self.forward_ffn(tgt)

        return tgt


class DepthAwareDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        return_intermediate=False,
        d_model=None,
        use_dab=False,
        two_stage_dino=False,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.bbox_embed = None
        self.dim_embed = None
        self.class_embed = None
        self.use_dab = use_dab
        self.two_stgae_dino = two_stage_dino
        if use_dab:
            self.query_scale = MLP(d_model, d_model, d_model, 2)
            self.query_scale_bbox = MLP(d_model, 2, 2, 2)
            self.ref_point_head = MLP(3 * d_model, d_model, d_model, 2)
        elif two_stage_dino:
            self.ref_point_head = MLP(3 * d_model, d_model, d_model, 2)
            self.query_scale = MLP(d_model, d_model, d_model, 2)
            self.query_pos_sine_scale = None
            self.ref_anchor_head = None
        else:
            self.query_scale = MLP(d_model, d_model, d_model, 2)
            self.ref_point_head = MLP(d_model, d_model, 2, 2)

    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
        depth_pos_embed=None,
        mask_depth=None,
        bs=None,
        depth_pos_embed_ip=None,
        pos_embeds=None,
        attn_mask=None,
    ):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        intermediate_reference_dims = []
        bs = src.shape[0]
        if self.use_dab:
            reference_points = reference_points[None].repeat(bs, 1, 1)
        elif self.two_stgae_dino:
            reference_points = reference_points.sigmoid()

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 6:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios, src_valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            if self.two_stgae_dino:
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])
                raw_query_pos = self.ref_point_head(query_sine_embed)
                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos
            query_pos_un = None
            if self.use_dab:
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])
                raw_query_pos = self.ref_point_head(query_sine_embed)
                pos_scale = self.query_scale(output) if lid != 0 else 1
                query_pos = pos_scale * raw_query_pos

            output = layer(
                output,
                query_pos,
                reference_points_input,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
                depth_pos_embed,
                mask_depth,
                bs,
                query_sine_embed=None,
                is_first=(lid == 0),
                depth_pos_embed_ip=depth_pos_embed_ip,
                pos_embeds=pos_embeds,
                self_attn_mask=attn_mask,
                query_pos_un=query_pos_un,
            )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 6:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.dim_embed is not None:
                reference_dims = self.dim_embed[lid](output)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_reference_dims.append(reference_dims)

        if self.return_intermediate:
            return (
                torch.stack(intermediate),
                torch.stack(intermediate_reference_points),
                torch.stack(intermediate_reference_dims),
            )

        return output, reference_points


def build_depthaware_transformer(cfg):
    return DepthAwareTransformer(
        d_model=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        activation="relu",
        nhead=cfg["nheads"],
        dim_feedforward=cfg["dim_feedforward"],
        num_encoder_layers=cfg["enc_layers"],
        num_decoder_layers=cfg["dec_layers"],
        return_intermediate_dec=cfg["return_intermediate_dec"],
        num_feature_levels=cfg["num_feature_levels"],
        dec_n_points=cfg["dec_n_points"],
        enc_n_points=cfg["enc_n_points"],
        two_stage=cfg["two_stage"],
        two_stage_num_proposals=cfg["num_queries"],
        use_dab=cfg["use_dab"],
        two_stage_dino=cfg["two_stage_dino"],
    )


# --------------------------------------------------------------------------- #
# lib/models/monodetr/monodetr.py (MonoDETR + MLP head; SetCriterion/build() dropped,
# see header)
# --------------------------------------------------------------------------- #


class MonoDETR(nn.Module):
    """This is the MonoDETR module that performs monocualr 3D object detection"""

    def __init__(
        self,
        backbone,
        depthaware_transformer,
        depth_predictor,
        num_classes,
        num_queries,
        num_feature_levels,
        aux_loss=True,
        with_box_refine=False,
        two_stage=False,
        init_box=False,
        use_dab=False,
        group_num=11,
        two_stage_dino=False,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            depthaware_transformer: depth-aware transformer architecture.
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot.
            aux_loss: True if auxiliary decoding losses are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage MonoDETR
        """
        super().__init__()

        self.num_queries = num_queries
        self.depthaware_transformer = depthaware_transformer
        self.depth_predictor = depth_predictor
        hidden_dim = depthaware_transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.two_stage_dino = two_stage_dino
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2)  # depth and deviation
        self.use_dab = use_dab

        if init_box is True:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if not two_stage:
            if two_stage_dino:
                self.query_embed = None
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries * group_num, hidden_dim * 2)
            else:
                self.tgt_embed = nn.Embedding(num_queries * group_num, hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries * group_num, 6)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.num_classes = num_classes

        if self.two_stage_dino:
            _class_embed = nn.Linear(hidden_dim, num_classes)
            _bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(num_classes) * bias_value
            nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
            self.depthaware_transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
            self.depthaware_transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        num_pred = (
            (depthaware_transformer.decoder.num_layers + 1)
            if two_stage
            else depthaware_transformer.decoder.num_layers
        )
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            self.depthaware_transformer.decoder.bbox_embed = self.bbox_embed
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.depthaware_transformer.decoder.dim_embed = self.dim_embed_3d
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.depth_embed = _get_clones(self.depth_embed, num_pred)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = nn.ModuleList([self.angle_embed for _ in range(num_pred)])
            self.depth_embed = nn.ModuleList([self.depth_embed for _ in range(num_pred)])
            self.depthaware_transformer.decoder.bbox_embed = None

        if two_stage:
            self.depthaware_transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, images, calibs, targets, img_sizes, dn_args=None):
        """The forward expects the batched images [B x 3 x H x W], the batched
        camera calibration matrices, and image sizes."""

        features, pos = self.backbone(images)

        srcs = []
        masks = []
        for l, feat in enumerate(features):  # noqa: E741
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):  # noqa: E741
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = (
                    torch.zeros(src.shape[0], src.shape[2], src.shape[3])
                    .to(torch.bool)
                    .to(src.device)
                )
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.two_stage:
            query_embeds = None
        elif self.use_dab:
            if self.training:
                tgt_embed = self.tgt_embed.weight
                refanchor = self.refpoint_embed.weight
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
            else:
                tgt_embed = self.tgt_embed.weight[: self.num_queries]
                refanchor = self.refpoint_embed.weight[: self.num_queries]
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
        elif self.two_stage_dino:
            query_embeds = None
        else:
            if self.training:
                query_embeds = self.query_embed.weight
            else:
                query_embeds = self.query_embed.weight[: self.num_queries]

        pred_depth_map_logits, depth_pos_embed, weighted_depth, depth_pos_embed_ip = (
            self.depth_predictor(srcs, masks[1], pos[1])
        )

        (
            hs,
            init_reference,
            inter_references,
            inter_references_dim,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.depthaware_transformer(
            srcs, masks, pos, query_embeds, depth_pos_embed, depth_pos_embed_ip
        )

        outputs_coords = []
        outputs_classes = []
        outputs_3d_dims = []
        outputs_depths = []
        outputs_angles = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference

            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)

            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)

            size3d = inter_references_dim[lvl]
            outputs_3d_dims.append(size3d)

            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1:2], min=1.0)
            depth_geo = size3d[:, :, 0] / box2d_height * calibs[:, 0, 0].unsqueeze(1)

            depth_reg = self.depth_embed[lvl](hs[lvl])

            outputs_center3d = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2).detach()
            depth_map = F.grid_sample(
                weighted_depth.unsqueeze(1), outputs_center3d, mode="bilinear", align_corners=True
            ).squeeze(1)

            depth_ave = torch.cat(
                [
                    (
                        (1.0 / (depth_reg[:, :, 0:1].sigmoid() + 1e-6) - 1.0)
                        + depth_geo.unsqueeze(-1)
                        + depth_map
                    )
                    / 3,
                    depth_reg[:, :, 1:2],
                ],
                -1,
            )
            outputs_depths.append(depth_ave)

            outputs_angle = self.angle_embed[lvl](hs[lvl])
            outputs_angles.append(outputs_angle)

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        outputs_angle = torch.stack(outputs_angles)

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        out["pred_3d_dim"] = outputs_3d_dim[-1]
        out["pred_depth"] = outputs_depth[-1]
        out["pred_angle"] = outputs_angle[-1]
        out["pred_depth_map_logits"] = pred_depth_map_logits

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth
            )

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {"pred_logits": enc_outputs_class, "pred_boxes": enc_outputs_coord}
        return out

    @torch.jit.unused
    def _set_aux_loss(
        self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth
    ):
        return [
            {"pred_logits": a, "pred_boxes": b, "pred_3d_dim": c, "pred_angle": d, "pred_depth": e}
            for a, b, c, d, e in zip(
                outputs_class[:-1],
                outputs_coord[:-1],
                outputs_3d_dim[:-1],
                outputs_angle[:-1],
                outputs_depth[:-1],
            )
        ]


_CFG = {
    "num_classes": 3,
    "return_intermediate_dec": True,
    "backbone": "resnet50",
    "train_backbone": True,
    "num_feature_levels": 4,
    "dilation": False,
    "position_embedding": "sine",
    "masks": False,
    "mode": "LID",
    "num_depth_bins": 80,
    "depth_min": 1e-3,
    "depth_max": 60.0,
    "with_box_refine": True,
    "two_stage": False,
    "use_dab": False,
    "use_dn": False,
    "two_stage_dino": False,
    "init_box": False,
    "enc_layers": 1,
    "dec_layers": 1,
    # hidden_dim must stay 256: DepthPredictor.depth_pos_embed is `nn.Embedding(...,
    # 256)` -- a hardcoded 256 in the real upstream code (lib/models/monodetr/
    # depth_predictor/depth_predictor.py), not tied to hidden_dim, and it's summed
    # with depth_embed (which has hidden_dim channels), so hidden_dim must equal 256
    # for the model to be constructible at all -- this is a real upstream constraint,
    # not a shrink choice.
    "hidden_dim": 256,
    "dim_feedforward": 64,
    "dropout": 0.1,
    "nheads": 4,
    "num_queries": 8,
    "enc_n_points": 4,
    "dec_n_points": 4,
    "aux_loss": True,
}


def build_monodetr():
    backbone = build_backbone(_CFG)
    depthaware_transformer = build_depthaware_transformer(_CFG)
    depth_predictor = DepthPredictor(_CFG)
    model = MonoDETR(
        backbone,
        depthaware_transformer,
        depth_predictor,
        num_classes=_CFG["num_classes"],
        num_queries=_CFG["num_queries"],
        aux_loss=_CFG["aux_loss"],
        num_feature_levels=_CFG["num_feature_levels"],
        with_box_refine=_CFG["with_box_refine"],
        two_stage=_CFG["two_stage"],
        init_box=_CFG["init_box"],
        use_dab=_CFG["use_dab"],
        two_stage_dino=_CFG["two_stage_dino"],
    )
    model.eval()
    return model


def example_input_monodetr():
    # num_feature_levels=4 backbone strides (8,16,32) + 1 extra downsampled level;
    # depth_predictor asserts len(feature)==4 and needs feature[0..2] at strides
    # 8/16/32 to be spatially compatible after F.interpolate -- 128x448 keeps every
    # stage's H,W integral through 5 stride-2 halvings (32x -> 4x14).
    images = torch.randn(1, 3, 128, 448)
    calibs = torch.eye(3, 4).unsqueeze(0)  # (1, 3, 4) camera intrinsics, batch=1
    targets = []
    img_sizes = torch.tensor([[448.0, 128.0]])  # (1, 2): (W, H) per MonoDETR convention
    return images, calibs, targets, img_sizes


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("MonoDETR", "build_monodetr", "example_input_monodetr", 2023, "vendored-pytorch"),
]
