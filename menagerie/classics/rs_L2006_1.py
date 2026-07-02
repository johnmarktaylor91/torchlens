# SOURCE: vendored from https://github.com/chongzhou96/EdgeSAM @ master
#
# EdgeSAM (Zhou et al., 2023/2024) distills Segment-Anything's ViT-H image
# encoder into a pure-CNN "RepViT" backbone for on-device inference. The files
# below are the REAL model-definition classes from the official repo
# (edge_sam/modeling/{rep_vit,prompt_encoder,mask_decoder,transformer,common}.py),
# copied verbatim with only these minimal fixes:
#   - relative "from .xxx import yyy" -> flattened into this single module
#   - dropped the top-level `import loralib` in transformer.py: loralib is an
#     optional dependency only exercised when Attention(..., lora=True); our
#     construction always uses the repo's own default lora=False, so the
#     loralib code path is never entered. Not installed in the base env.
#   - dropped edge_sam/modeling/sam.py + build_sam.py's optional RPN heads
#     (CenterNetUpdateHead / RPNHead / BiFPN / EfficientDet), which require
#     mmdet + mmengine (not base libs) and are only used for the RPN-assisted
#     auto-mask-generation variant (rpn_head='centernet'/'rpn'/'efficient_det').
#     The default `build_edge_sam()` builds a plain `Sam` with `rpn_head=None`
#     (see build_sam.py's `_build_sam`), so `EdgeSam` below reproduces that
#     default path faithfully (real image_encoder=RepViT + prompt_encoder +
#     mask_decoder, no RPN) without ever needing mmdet.
#
# EdgeSAM's actual architectural contribution is the RepViT CNN image encoder
# (prompt-in-the-loop distilled from SAM's ViT-H). PromptEncoder, MaskDecoder,
# TwoWayTransformer are inherited verbatim from Segment-Anything (same as the
# original repo re-uses them) and are vendored here too since `Sam` needs them
# to trace end-to-end.

import math
from functools import partial
from typing import Any, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite
from timm.models.vision_transformer import trunc_normal_

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# edge_sam/modeling/common.py
# ---------------------------------------------------------------------------
class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def val2list(x, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def resize(
    x: torch.Tensor,
    size=None,
    scale_factor=None,
    mode: str = "bicubic",
    align_corners=False,
) -> torch.Tensor:
    if mode in ["bilinear", "bicubic"]:
        return F.interpolate(
            x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners
        )
    elif mode in ["nearest", "area"]:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


class UpSampleLayer(nn.Module):
    def __init__(self, mode="bicubic", size=None, factor=2, align_corners=False):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class OpSequential(nn.Module):
    def __init__(self, op_list):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x


# ---------------------------------------------------------------------------
# edge_sam/modeling/rep_vit.py
# ---------------------------------------------------------------------------
m1_cfgs = [
    # k, t, c, SE, HS, s
    [3, 2, 48, 1, 0, 1],
    [3, 2, 48, 0, 0, 1],
    [3, 2, 48, 0, 0, 1],
    [3, 2, 96, 0, 0, 2],
    [3, 2, 96, 1, 0, 1],
    [3, 2, 96, 0, 0, 1],
    [3, 2, 96, 0, 0, 1],
    [3, 2, 192, 0, 1, 2],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 1, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 192, 0, 1, 1],
    [3, 2, 384, 0, 1, 2],
    [3, 2, 384, 1, 1, 1],
    [3, 2, 384, 0, 1, 1],
]


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2d_BN(torch.nn.Sequential):
    def __init__(
        self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1, resolution=-10000
    ):
        super().__init__()
        self.add_module("c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module("bn", torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.0):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return (
                x
                + self.m(x)
                * torch.rand(x.size(0), 1, 1, 1, device=x.device)
                .ge_(self.drop)
                .div(1 - self.drop)
                .detach()
            )
        else:
            return x + self.m(x)


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = Conv2d_BN(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed

    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x


class RepViTBlock(nn.Module):
    def __init__(
        self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs, skip_downsample=False
    ):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert hidden_dim == 2 * inp

        if stride == 2:
            if skip_downsample:
                stride = 1
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                )
            )
        else:
            assert self.identity
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(
                nn.Sequential(
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                )
            )

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module("bn", torch.nn.BatchNorm1d(a))
        self.add_module("l", torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)


class RepViT(nn.Module):
    arch_settings = {
        "m1": m1_cfgs,
    }

    def __init__(
        self,
        arch,
        img_size=1024,
        fuse=False,
        freeze=False,
        load_from=None,
        use_rpn=False,
        out_indices=None,
        upsample_mode="bicubic",
    ):
        super(RepViT, self).__init__()
        self.cfgs = self.arch_settings[arch]
        self.img_size = img_size
        self.fuse = fuse
        self.freeze = freeze
        self.use_rpn = use_rpn
        self.out_indices = out_indices

        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(
            Conv2d_BN(3, input_channel // 2, 3, 2, 1),
            torch.nn.GELU(),
            Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1),
        )
        layers = [patch_embed]
        block = RepViTBlock
        self.stage_idx = []
        prev_c = input_channel
        for idx, (k, t, c, use_se, use_hs, s) in enumerate(self.cfgs):
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            skip_downsample = False
            if not self.fuse and c in [384, 512]:
                skip_downsample = True
            if c != prev_c:
                self.stage_idx.append(idx - 1)
                prev_c = c
            layers.append(
                block(
                    input_channel, exp_size, output_channel, k, s, use_se, use_hs, skip_downsample
                )
            )
            input_channel = output_channel
        self.stage_idx.append(idx)
        self.features = nn.ModuleList(layers)

        if self.fuse:
            stage2_channels = _make_divisible(self.cfgs[self.stage_idx[2]][2], 8)
            stage3_channels = _make_divisible(self.cfgs[self.stage_idx[3]][2], 8)
            self.fuse_stage2 = nn.Conv2d(stage2_channels, 256, kernel_size=1, bias=False)
            self.fuse_stage3 = OpSequential(
                [
                    nn.Conv2d(stage3_channels, 256, kernel_size=1, bias=False),
                    UpSampleLayer(factor=2, mode=upsample_mode),
                ]
            )
            neck_in_channels = 256
        else:
            neck_in_channels = output_channel
        self.neck = nn.Sequential(
            nn.Conv2d(neck_in_channels, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

    def forward(self, x):
        counter = 0
        output_dict = dict()
        x = self.features[0](x)
        output_dict["stem"] = x
        for idx, f in enumerate(self.features[1:]):
            x = f(x)
            if idx in self.stage_idx:
                output_dict[f"stage{counter}"] = x
                counter += 1

        if self.fuse:
            x = self.fuse_stage2(output_dict["stage2"]) + self.fuse_stage3(output_dict["stage3"])

        x = self.neck(x)
        output_dict["final"] = x

        if self.use_rpn:
            if self.out_indices is None:
                self.out_indices = [len(output_dict) - 1]
            out = []
            for i, key in enumerate(output_dict):
                if i in self.out_indices:
                    out.append(output_dict[key])
            return tuple(out)
        return x


def rep_vit_m1(img_size=1024, **kwargs):
    return RepViT("m1", img_size, **kwargs)


# ---------------------------------------------------------------------------
# edge_sam/modeling/prompt_encoder.py
# ---------------------------------------------------------------------------
class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        point_embedding[labels == -2] = torch.zeros_like(
            self.not_a_point_embed.weight, dtype=point_embedding.dtype
        )
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        return self.mask_downscaling(masks)

    def _get_batch_size(self, points, boxes, masks) -> int:
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(self, points, boxes, masks, box_labels=None) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            if box_labels is not None:
                box_embeddings[box_labels == -1, :, :] = 0.0
                box_embeddings[box_labels == -1, :, :] += self.not_a_point_embed.weight
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


# ---------------------------------------------------------------------------
# edge_sam/modeling/transformer.py
# (loralib import dropped: repo default lora=False is what we construct with)
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(
        self, embedding_dim: int, num_heads: int, downsample_rate: int = 1, lora=False
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q, k, v, kd_targets=None, target_name=None):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        if kd_targets is not None:
            kd_targets[target_name] = attn

        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        lora=False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads, lora=lora)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, lora=lora
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, lora=lora
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_pe, key_pe, kd_targets=None, layer_idx=None):
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries, kd_targets, f"t2t_{layer_idx}")
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q, q, queries, kd_targets, f"t2t_{layer_idx}")
            queries = queries + attn_out
        queries = self.norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q, k, keys, kd_targets, f"t2i_{layer_idx}")
        queries = queries + attn_out
        queries = self.norm2(queries)

        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(k, q, queries, kd_targets, f"i2t_{layer_idx}")
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        lora=False,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    lora=lora,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate, lora=lora
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding, image_pe, point_embedding, kd_targets=None):
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding

        for idx, layer in enumerate(self.layers):
            queries, keys = layer(queries, keys, point_embedding, image_pe, kd_targets, idx)

        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q, k, keys, kd_targets, "t2i_final")
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


# ---------------------------------------------------------------------------
# edge_sam/modeling/mask_decoder.py
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        yield_kd_targets=False,
    ) -> None:
        super().__init__()
        self.yield_kd_targets = yield_kd_targets
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        num_multimask_outputs: int,
        num_prompts=None,
    ):
        if self.yield_kd_targets:
            kd_targets = dict()
        else:
            kd_targets = None

        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            num_prompts=num_prompts,
            kd_targets=kd_targets,
        )

        if num_multimask_outputs == 4:
            mask_slice = slice(0, None)
        elif num_multimask_outputs == 3:
            mask_slice = slice(1, None)
        elif num_multimask_outputs == 1:
            mask_slice = slice(0, 1)
        else:
            raise ValueError

        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        if kd_targets is not None:
            kd_targets["query"] = kd_targets["query"][:, mask_slice]
        if self.yield_kd_targets:
            return masks, iou_pred, kd_targets
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        num_prompts=None,
        kd_targets=None,
    ):
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens, kd_targets)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        iou_pred = self.iou_prediction_head(iou_token_out)

        if self.yield_kd_targets:
            kd_targets["query"] = hyper_in
            kd_targets["feat"] = upscaled_embedding
        return masks, iou_pred


# ---------------------------------------------------------------------------
# edge_sam/modeling/sam.py -- trimmed to the default (rpn_head=None) path.
# The real build_sam.py's `build_edge_sam()` -> `_build_sam(image_encoder,
# checkpoint)` never passes an rpn_head, so `use_rpn` is always False for
# EdgeSAM; the mmdet-dependent FPN/RPNHead/CenterNetUpdateHead branches in the
# original class body are dead code on that path and are omitted here.
# ---------------------------------------------------------------------------
class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: RepViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.use_rpn = False

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, image: torch.Tensor, point_coords: torch.Tensor, point_labels: torch.Tensor):
        """Single-image, single-batch forward driving encoder + prompt +
        decoder end to end (a torchlens-traceable simplification of the
        original per-record `forward()` loop over `batched_input` dicts)."""
        image_embeddings = self.image_encoder(self.preprocess(image))
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            num_multimask_outputs=1,
        )
        return low_res_masks, iou_predictions

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


# ---------------------------------------------------------------------------
# edge_sam/build_sam.py -- build_edge_sam(), shrunk to a tiny image size for
# a fast trace. RepViT (fuse=True) downsamples by /16 total (patch_embed /4,
# then 2 more stride-2 stages before the stage2/stage3 fuse point), so a
# 256x256 tiny image still exercises the full real architecture at a
# manageable resolution; vit_patch_size stays the real repo's 16.
# ---------------------------------------------------------------------------
def build_edge_sam_tiny(image_size: int = 256):
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    image_encoder = RepViT(arch="m1", img_size=image_size, upsample_mode="bicubic", fuse=True)

    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )
    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
            lora=False,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        yield_kd_targets=False,
    )
    sam = Sam(
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    return sam


_EDGE_SAM_IMAGE_SIZE = 256


def build_edge_sam():
    return build_edge_sam_tiny(image_size=_EDGE_SAM_IMAGE_SIZE)


def example_edge_sam():
    image = torch.randn(1, 3, _EDGE_SAM_IMAGE_SIZE, _EDGE_SAM_IMAGE_SIZE)
    point_coords = torch.tensor([[[64.0, 64.0]]])
    point_labels = torch.tensor([[1]])
    return image, point_coords, point_labels


MENAGERIE_ENTRIES = [
    ("EdgeSAM", "build_edge_sam", "example_edge_sam", "2023", "SEG"),
]
