# SOURCE: vendored from liuruijin17/LSTR @ main
#
# https://github.com/liuruijin17/LSTR
# https://raw.githubusercontent.com/liuruijin17/LSTR/main/models/LSTR.py
# https://raw.githubusercontent.com/liuruijin17/LSTR/main/models/py_utils/kp.py
# https://raw.githubusercontent.com/liuruijin17/LSTR/main/models/py_utils/transformer.py
# https://raw.githubusercontent.com/liuruijin17/LSTR/main/models/py_utils/position_encoding.py
# https://raw.githubusercontent.com/liuruijin17/LSTR/main/config/LSTR.json
#
# LSTR ("End-to-end Lane Shape Prediction with Transformers", Liu et al.
# 2021, WACV; repo name matches the queue notes' "E2ELSPTRs" = End-to-End
# Lane Shape Prediction with TRansformerS). This vendors the real `kp`
# network class from `models/py_utils/kp.py` (small dilated-stride ResNet
# stem/backbone + DETR-style Transformer encoder-decoder producing per-query
# lane-class logits and polynomial-curve parameters), the real
# `Transformer`/`TransformerEncoder(Layer)`/`TransformerDecoder(Layer)`
# classes from `models/py_utils/transformer.py`, the real
# `PositionEmbeddingSine` from `models/py_utils/position_encoding.py`, and
# the real `model(kp)` wrapper + `BasicBlock`/`Bottleneck` stem blocks from
# `models/LSTR.py` -- copied verbatim (only whitespace-preserving copy, no
# architecture changes).
#
# Dropped from the vendor: `AELoss`/`SetCriterion`/`build_matcher` (DETR
# Hungarian-matching training loss, not part of the forward network) and the
# `from .misc import *` / `from sample.vis import save_debug_images_boxes`
# imports in the real `kp.py`, neither of which `kp.forward`/`kp._train`
# actually calls (`reduce_dict` and the visualization helper are only used
# inside `AELoss.forward`, which is not vendored); `FrozenBatchNorm2d` is
# kept exactly as upstream defines it (used as the real default
# `norm_layer`).
#
# The `model(kp)` subclass in the real `models/LSTR.py` reads its
# hyperparameters from a global `system_configs` singleton
# (`config.system_configs`, populated from a JSON file via
# `system_config.py`, not vendored here). `build_lstr()` below constructs
# `kp` directly with the exact hyperparameters from the real
# `config/LSTR.json` TUSIMPLE reference config (`res_layers=[1,2,2,2]`,
# `res_dims=[16,32,64,128]`, `res_strides=[1,2,2,2]`, `attn_dim=32`,
# `dim_feedforward=128`, `num_heads=2`, `enc_layers=2`, `dec_layers=2`,
# `block=BasicBlock`, `num_queries=7`, `lsp_dim=8` (the config's
# `kps_dim`), `pos_type='sine'`, `drop_out=0.1`, `pre_norm=False`,
# `return_intermediate=True`, `num_cls=1` (`categories`), `mlp_layers=2`
# (upstream default), `aux_loss=False` for this forward-only build) -- this
# is config plumbing that reproduces the real numbers, not an architecture
# change.

import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

BN_MOMENTUM = 0.1


# ---------------------------------------------------------------------------
# models/py_utils/position_encoding.py
# ---------------------------------------------------------------------------


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
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

    def forward(self, x, mask):
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


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
        )
        return pos


def build_position_encoding(hidden_dim, type):
    N_steps = hidden_dim // 2
    if type in ("v2", "sine"):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif type in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {type}")
    return position_embedding


# ---------------------------------------------------------------------------
# models/py_utils/transformer.py
# ---------------------------------------------------------------------------


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)

        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)

        memory, weights = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        hs = self.decoder(
            tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed
        )

        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w), weights


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output, weights = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
            )

        if self.norm is not None:
            output = self.norm(output)

        return output, weights


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
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
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2, weights = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )

        src = src + self.dropout1(src2)

        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        src = src + self.dropout2(src2)

        src = self.norm2(src)

        return src, weights

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]

        tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]

        tgt = tgt + self.dropout2(tgt2)

        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        tgt = tgt + self.dropout3(tgt2)

        tgt = self.norm3(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(
    hidden_dim,
    dropout,
    nheads,
    dim_feedforward,
    enc_layers,
    dec_layers,
    pre_norm=False,
    return_intermediate_dec=False,
):
    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=return_intermediate_dec,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


# ---------------------------------------------------------------------------
# models/py_utils/kp.py
# ---------------------------------------------------------------------------


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
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
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


def kp_conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = kp_conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = kp_conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class kp(nn.Module):
    def __init__(
        self,
        flag=False,
        block=None,
        layers=None,
        res_dims=None,
        res_strides=None,
        attn_dim=None,
        num_queries=None,
        aux_loss=None,
        pos_type=None,
        drop_out=0.1,
        num_heads=None,
        dim_feedforward=None,
        enc_layers=None,
        dec_layers=None,
        pre_norm=None,
        return_intermediate=None,
        lsp_dim=None,
        mlp_layers=None,
        num_cls=None,
        norm_layer=FrozenBatchNorm2d,
    ):
        super(kp, self).__init__()
        self.flag = flag
        self.norm_layer = norm_layer

        self.inplanes = res_dims[0]
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, res_dims[0], layers[0], stride=res_strides[0])
        self.layer2 = self._make_layer(block, res_dims[1], layers[1], stride=res_strides[1])
        self.layer3 = self._make_layer(block, res_dims[2], layers[2], stride=res_strides[2])
        self.layer4 = self._make_layer(block, res_dims[3], layers[3], stride=res_strides[3])

        hidden_dim = attn_dim
        self.aux_loss = aux_loss
        self.position_embedding = build_position_encoding(hidden_dim=hidden_dim, type=pos_type)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(res_dims[-1], hidden_dim, kernel_size=1)

        self.transformer = build_transformer(
            hidden_dim=hidden_dim,
            dropout=drop_out,
            nheads=num_heads,
            dim_feedforward=dim_feedforward,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            return_intermediate_dec=return_intermediate,
        )

        self.class_embed = nn.Linear(hidden_dim, num_cls)
        self.specific_embed = MLP(hidden_dim, hidden_dim, lsp_dim - 4, mlp_layers)
        self.shared_embed = MLP(hidden_dim, hidden_dim, 4, mlp_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _train(self, *xs, **kwargs):
        images = xs[0]  # B 3 360 640
        masks = xs[1]  # B 1 360 640

        p = self.conv1(images)
        p = self.bn1(p)
        p = self.relu(p)
        p = self.maxpool(p)
        p = self.layer1(p)
        p = self.layer2(p)
        p = self.layer3(p)
        p = self.layer4(p)
        pmasks = F.interpolate(masks[:, 0, :, :][None], size=p.shape[-2:]).to(torch.bool)[0]
        pos = self.position_embedding(p, pmasks)
        hs, _, weights = self.transformer(self.input_proj(p), pmasks, self.query_embed.weight, pos)
        output_class = self.class_embed(hs)
        output_specific = self.specific_embed(hs)
        output_shared = self.shared_embed(hs)
        output_shared = torch.mean(output_shared, dim=-2, keepdim=True)
        output_shared = output_shared.repeat(1, 1, output_specific.shape[2], 1)
        output_specific = torch.cat(
            [output_specific[:, :, :, :2], output_shared, output_specific[:, :, :, 2:]], dim=-1
        )
        out = {"pred_logits": output_class[-1], "pred_curves": output_specific[-1]}
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(output_class, output_specific)
        return out, weights

    def _test(self, *xs, **kwargs):
        return self._train(*xs, **kwargs)

    def forward(self, *xs, **kwargs):
        if self.flag:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [
            {"pred_logits": a, "pred_curves": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]


# ---------------------------------------------------------------------------
# models/LSTR.py (`model(kp)` wrapper, hyperparameters from config/LSTR.json)
# ---------------------------------------------------------------------------


def build_lstr():
    # Real values transcribed from `config/LSTR.json`'s "system" block
    # (TUSIMPLE reference config); `num_cls` is the config's "categories"
    # (from the "db" block, =1) and `lsp_dim` is the config's "kps_dim" (=8).
    return kp(
        flag=False,
        block=BasicBlock,
        layers=[1, 2, 2, 2],
        res_dims=[16, 32, 64, 128],
        res_strides=[1, 2, 2, 2],
        attn_dim=32,
        num_queries=7,
        aux_loss=False,
        pos_type="sine",
        drop_out=0.1,
        num_heads=2,
        dim_feedforward=128,
        enc_layers=2,
        dec_layers=2,
        pre_norm=False,
        return_intermediate=True,
        lsp_dim=8,
        mlp_layers=2,
        num_cls=1,
    )


def example_input_lstr():
    # Real TUSIMPLE input_size is [360, 640]; shrunk to a tiny multiple of
    # the backbone's 8x total downsampling (res_strides [1,2,2,2] after the
    # stem's stride-4) while keeping the (H, W) aspect roughly matched. The
    # real dataloader supplies `masks` as a float tensor of 0./1. (matching
    # `kp._train`'s `F.interpolate(...).to(torch.bool)` cast), not a bool
    # tensor directly -- `F.interpolate` does not support bool inputs.
    images = torch.randn(1, 3, 64, 96)
    masks = torch.zeros(1, 1, 64, 96, dtype=torch.float32)
    return images, masks


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("LSTR", "build_lstr", "example_input_lstr", 2021, "vendored"),
]
