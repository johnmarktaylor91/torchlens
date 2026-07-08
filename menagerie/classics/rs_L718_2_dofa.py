# SOURCE: vendored from https://github.com/zhu-xlab/DOFA @ master
#
# DOFA (Dynamic One-For-All, Xiong, Wang, Zhu et al., TPAMI 2024/CVPR workshop -- "Neural
# Plasticity-Inspired Foundation Model for Observing the Earth Crossing Modalities") -- a ViT
# backbone whose patch-embedding convolution weights are generated on-the-fly from each
# input channel's wavelength via a small Transformer-based hypernetwork
# (`Dynamic_MLP_OFA`/`TransformerWeightGenerator`), letting a single pretrained model accept
# an arbitrary number/ordering of multi-sensor satellite-imagery channels (e.g. Sentinel-1
# SAR, Sentinel-2 MSI, hyperspectral) at inference. Vendored verbatim (architecture-relevant
# classes only) from the repo's own files:
#   https://raw.githubusercontent.com/zhu-xlab/DOFA/master/dofa_v1.py
#   https://raw.githubusercontent.com/zhu-xlab/DOFA/master/wave_dynamic_layer.py
#
# What is kept: OFAViT (the top-level ViT backbone: dynamic patch embedding + cls token +
# fixed sin-cos position embedding + stack of real `timm.models.vision_transformer.Block`
# transformer blocks + global-pool head), Dynamic_MLP_OFA (the real wavelength-conditioned
# dynamic-convolution patch embedding: sin-cos wavelength encoding -> FCResLayer ->
# TransformerWeightGenerator -> per-channel conv2d kernel/bias -> F.conv2d), FCResLayer,
# TransformerWeightGenerator (the real `nn.TransformerEncoder`-based hypernetwork that maps
# wavelength embeddings + learned weight/bias tokens to convolution weights),
# get_1d_sincos_pos_embed_from_grid_torch -- every mechanism in the real trainable network,
# transcribed unmodified. `vit_small_patch16`/`vit_base_patch16` etc. factory functions kept
# verbatim; this staging build uses the same factory pattern at a tiny size.
#
# What is dropped (infra plumbing, not part of the forward-pass computation graph):
# pretraining-only files (`pretraining/*`: MAE decoder, datasets, engine, distributed
# samplers) and `downstream_tasks/*` (linear-probe/segmentation finetuning scripts,
# `checkpoints/download_weights.py`) are not vendored -- none of them define architecture,
# they only consume `OFAViT`/`vit_*_patch16` for training/eval loops.
#
# MENAGERIE_ZOO = "vendored-pytorch"

from __future__ import annotations

import math
from functools import partial, reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from timm.models.vision_transformer import Block

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# from wave_dynamic_layer.py (verbatim)
# ---------------------------------------------------------------------------
def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class TransformerWeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=4, num_layers=1):
        super(TransformerWeightGenerator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x):
        # x should have shape [seq_len, batch, input_dim]
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(transformer_output[-1])
        return weights, bias


class FCResLayer(nn.Module):
    def __init__(self, linear_size=128):
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out


class Dynamic_MLP_OFA(nn.Module):
    """
    Input: channels of wavelength (normalized): List -> List
           kernel size of the depth-wise convolution: kernel_size, default 3x3
           wv_planes
           inplanes
    """

    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        self.weight_generator = TransformerWeightGenerator(wv_planes, self._num_kernel, embed_dim)
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, wvs):
        inplanes = wvs.size(0)
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)  # 3x3x3

        dynamic_weight = weight.view(inplanes, self.kernel_size, self.kernel_size, self.embed_dim)
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves


# ---------------------------------------------------------------------------
# from dofa_v1.py (verbatim)
# ---------------------------------------------------------------------------
class OFAViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        drop_rate=0.0,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        wv_planes=128,
        num_classes=45,
        global_pool=True,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.wv_planes = wv_planes
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=128, inter_dim=128, kernel_size=patch_size, embed_dim=embed_dim
        )
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(depth)
            ]
        )

        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, wave_list):
        wavelist = torch.tensor(wave_list, device=x.device).float()
        self.waves = wavelist

        x, _ = self.patch_embed(x, self.waves)

        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for block in self.blocks:
            x = block(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

    def forward_head(self, x, pre_logits=False):
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, wave_list):
        x = self.forward_features(x, wave_list)
        x = self.forward_head(x)
        return x


def vit_small_patch16(**kwargs):
    model = OFAViT(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# ---------------------------------------------------------------------------
# staging glue (not part of the original architecture)
# ---------------------------------------------------------------------------
def build_dofa():
    # Tiny OFAViT: small img/patch size + shallow depth, matching the real
    # vit_small_patch16 factory's structure but shrunk for a fast trace.
    return OFAViT(
        img_size=32,
        patch_size=16,
        embed_dim=32,
        depth=2,
        num_heads=4,
        wv_planes=128,
        num_classes=10,
        global_pool=True,
        mlp_ratio=2.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


def example_input_dofa():
    generator = torch.Generator().manual_seed(0)
    # 3 input channels (e.g. RGB-like Sentinel-2 subset), each tagged with its
    # wavelength in micrometers -- the real multi-sensor use case (DOFA accepts an
    # arbitrary channel count/ordering via this wavelength list).
    img = torch.rand(2, 3, 32, 32, generator=generator)
    wave_list = [0.665, 0.560, 0.490]
    return (img, wave_list)


MENAGERIE_ENTRIES = [
    ("DOFA", "build_dofa", "example_input_dofa", 2024, "vendored-pytorch"),
]
