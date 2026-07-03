# FAITHFUL PORT of hustvl/TopFormer @ main (original framework: mmsegmentation/mmcv)
#
# TopFormer: Token Pyramid Transformer for Mobile Semantic Segmentation (CVPR 2022).
# The real repo (https://github.com/hustvl/TopFormer) is an mmsegmentation fork; the
# actual model is `mmseg/models/backbones/topformer.py::Topformer` (a MobileNet-style
# "Token Pyramid" CNN stem feeding a lightweight self-attention "Semantics Extractor",
# then per-scale "Semantic Injection Modules" that inject the pooled global-attention
# features back into each local CNN scale) plus
# `mmseg/models/decode_heads/simple_head.py::SimpleHead` (a multi-scale-fuse + dropout +
# 1x1-conv segmentation head, inheriting mmseg's generic `BaseDecodeHead.cls_seg`).
# mmcv (`ConvModule`, `build_norm_layer`, `BaseModule`) and mmseg's `BaseDecodeHead` are
# hard dependencies of the real files; mmcv is not installed and is not a reasonably
# installable base lib for this environment (a compiled, version-pinned OpenMMLab
# package) -- so the real code cannot run as-is (RUNG 2 vendoring fails). This module
# faithfully transcribes the ACTUAL nn.Module graph -- every conv/bn/act/attention/
# injection op, in the same structure -- into self-contained base-env torch, using the
# `topformer_tiny` config (local_configs/topformer/topformer_tiny.py) for every
# architectural constant.
#
# Architectural fidelity notes (topformer.py + topformer_tiny.py, both read verbatim):
#   - `Conv2d_BN` (mmseg/models/backbones/topformer.py): plain `nn.Conv2d` (no bias) +
#     `BatchNorm2d` (`build_norm_layer(dict(type='BN'...))` resolves to `nn.BatchNorm2d`
#     for every call site in this config; the real config file's top-level `norm_cfg`
#     says `SyncBN`, which is BatchNorm2d's distributed-training variant and identical
#     in a single-process forward pass) -- copied verbatim as `Conv2dBN`.
#   - `InvertedResidual` / `TokenPyramidModule`: the MobileNetV2-style token-pyramid
#     backbone stem, built from the `topformer_tiny` `cfgs` table
#     ([k,t,c,s] = kernel_size, expand_ratio, out_channels, stride) with
#     `embed_out_indice=[2,4,6,8]` selecting which of the 9 InvertedResidual stages'
#     outputs feed the pyramid -- copied verbatim including `_make_divisible`.
#   - `Attention` / `Mlp` / `Block` / `BasicLayer`: the lightweight self-attention
#     "Semantics Extractor Transformer" (query/key/value 1x1 Conv2d_BN projections,
#     scaled dot-product attention over flattened H*W tokens, depthwise-conv MLP) --
#     copied verbatim. `depths=4`, `key_dim=16` (default), `num_heads=4`,
#     `mlp_ratios=2` (default `attn_ratios=2`), `drop_path_rate=0.1` per
#     topformer_tiny.py (DropPath is present in the real graph but reduces to identity
#     in eval() forward, so it is included as the same `DropPath` module for structural
#     fidelity).
#   - `PyramidPoolAgg`: adaptive-avg-pools every token-pyramid scale down to the
#     coarsest scale's (H,W)//c2t_stride and concatenates along channels -- copied
#     verbatim, `c2t_stride=2` per config.
#   - `InjectionMultiSum` (the `injection_type="muli_sum"` default used by every
#     topformer_{tiny,small,base} config): per selected scale, a local 1x1-conv-BN
#     embedding of the CNN feature, a global 1x1-conv-BN embedding + h-sigmoid gate of
#     the transformer output (both bilinearly upsampled to the local scale), combined
#     as `local*gate + global` -- copied verbatim including `h_sigmoid`
#     (`ReLU6(x+3)/6`). `ConvModule(..., act_cfg=None)` in the real code means
#     conv+BN with NO activation for `local_embedding`/`global_embedding`/`global_act`
#     -- reproduced with a `Conv2dBNAct(act=None)` helper mirroring mmcv's
#     `ConvModule` conv->norm->[act] order.
#   - `Topformer.forward` (`injection=True`, the default and only path used by every
#     shipped config): tpm -> ppa -> trans -> split back into per-channel-group
#     tokens -> for each `decode_out_indices=[1,2,3]` scale, `SIM[i](local, global)` ->
#     return the 3 injected feature maps. Scale index 0 (channels[0]=16, the finest,
#     1/4-resolution scale) is NOT in `decode_out_indices` and its SIM slot is
#     `nn.Identity()` per the real `else: self.SIM.append(nn.Identity())` branch --
#     reproduced identically (index 0 is simply not decoded).
#   - `SimpleHead` (decode_heads/simple_head.py + decode_head.py `cls_seg`): takes the
#     3 injected feature maps, bilinearly resizes+sums them to the finest of the three
#     (`agg_res`), a `ConvModule` "linear_fuse" (1x1 depthwise conv+BN+act since
#     `is_dw=True` per topformer_tiny.py's decode_head config) -> Dropout2d(0.1) ->
#     1x1 `conv_seg` to `num_classes=150` (ADE20K, the tiny config's target dataset) --
#     copied verbatim. The `EncoderDecoder.forward`'s final bilinear resize back to the
#     input resolution (mmseg's `resize()` in `whole_inference`) is standard
#     mmsegmentation post-processing plumbing external to the model's own forward and
#     is applied here too (the injected-feature spatial size is much smaller than the
#     input crop) so the traced output is the actual per-pixel segmentation map.
#
# Trained weights (topformer-T-224-66.2.pth / ADE20K-finetuned checkpoints) are not
# used; this module constructs the architecture at random init for tracing.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor, min_value=None):
    """topformer.py _make_divisible(), verbatim (from the TF MobileNet repo)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2dBN(nn.Sequential):
    """topformer.py Conv2d_BN: conv (no bias) -> BatchNorm2d (build_norm_layer(BN) is
    plain nn.BatchNorm2d; the real config's SyncBN is identical in a single-process
    forward)."""

    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1.0):
        super().__init__()
        self.add_module("c", nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = nn.BatchNorm2d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)


class Conv2dBNAct(nn.Module):
    """mmcv ConvModule(conv, norm, act) equivalent used by the Injection/Fuse blocks
    and SimpleHead's linear_fuse: 1x1 conv -> BN -> optional activation."""

    def __init__(self, inp, oup, kernel_size=1, stride=1, groups=1, act=True):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(inp, oup, kernel_size, stride, pad, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.act is not None:
            x = self.act(x)
        return x


def drop_path(x, drop_prob=0.0, training=False):
    """topformer.py drop_path(), verbatim."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """topformer.py Mlp: 1x1 Conv2dBN -> depthwise 3x3 conv -> act -> dropout ->
    1x1 Conv2dBN -> dropout."""

    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2dBN(in_features, hidden_features)
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features
        )
        self.act = act_layer()
        self.fc2 = Conv2dBN(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InvertedResidual(nn.Module):
    """topformer.py InvertedResidual: MobileNetV2-style inverted residual block."""

    def __init__(self, inp, oup, ks, stride, expand_ratio, activations=nn.ReLU6):
        super().__init__()
        self.stride = stride
        assert stride in (1, 2)
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(Conv2dBN(inp, hidden_dim, ks=1))
            layers.append(activations())
        layers.extend(
            [
                Conv2dBN(
                    hidden_dim, hidden_dim, ks=ks, stride=stride, pad=ks // 2, groups=hidden_dim
                ),
                activations(),
                Conv2dBN(hidden_dim, oup, ks=1),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class TokenPyramidModule(nn.Module):
    """topformer.py TokenPyramidModule: MobileNetV2-style stem + a stack of
    InvertedResidual stages, collecting outputs at `out_indices`."""

    def __init__(self, cfgs, out_indices, inp_channel=16, activation=nn.ReLU6, width_mult=1.0):
        super().__init__()
        self.out_indices = out_indices
        self.stem = nn.Sequential(Conv2dBN(3, inp_channel, 3, 2, 1), activation())

        self.layers = nn.ModuleList()
        for k, t, c, s in cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            # NOTE: the real topformer.py also computes exp_size =
            # _make_divisible(t * inp_channel * width_mult, 8) here but never uses it
            # (InvertedResidual recomputes hidden_dim internally from expand_ratio) --
            # dead code in the original, faithfully omitted rather than transcribed
            # as an unused variable.
            layer = InvertedResidual(
                inp_channel, output_channel, ks=k, stride=s, expand_ratio=t, activations=activation
            )
            self.layers.append(layer)
            inp_channel = output_channel

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


def _get_shape(t):
    return t.shape


class Attention(nn.Module):
    """topformer.py Attention: multi-head scaled dot-product attention over flattened
    H*W spatial tokens, with 1x1-conv q/k/v projections."""

    def __init__(self, dim, key_dim, num_heads, attn_ratio=4, activation=nn.ReLU6):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads

        self.to_q = Conv2dBN(dim, self.nh_kd, 1)
        self.to_k = Conv2dBN(dim, self.nh_kd, 1)
        self.to_v = Conv2dBN(dim, self.dh, 1)
        self.proj = nn.Sequential(activation(), Conv2dBN(self.dh, dim, bn_weight_init=0))

    def forward(self, x):
        B, C, H, W = _get_shape(x)
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)

        xx = torch.matmul(attn, vv)
        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class Block(nn.Module):
    """topformer.py Block: attention + MLP with residual connections (a standard
    transformer block operating on spatial feature maps)."""

    def __init__(
        self,
        dim,
        key_dim,
        num_heads,
        mlp_ratio=4.0,
        attn_ratio=2.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.ReLU6,
    ):
        super().__init__()
        self.attn = Attention(
            dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class BasicLayer(nn.Module):
    """topformer.py BasicLayer: a stack of `block_num` transformer Blocks."""

    def __init__(
        self,
        block_num,
        embedding_dim,
        key_dim,
        num_heads,
        mlp_ratio=4.0,
        attn_ratio=2.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.ReLU6,
    ):
        super().__init__()
        self.block_num = block_num
        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(
                Block(
                    embedding_dim,
                    key_dim=key_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio,
                    drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    act_layer=act_layer,
                )
            )

    def forward(self, x):
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


class PyramidPoolAgg(nn.Module):
    """topformer.py PyramidPoolAgg: adaptive-avg-pool every scale down to the
    coarsest scale's (H,W)//stride and concatenate along channels."""

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = _get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([F.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)


class HSigmoid(nn.Module):
    """topformer.py h_sigmoid: ReLU6(x+3)/6."""

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(x + 3) / 6


class InjectionMultiSum(nn.Module):
    """topformer.py InjectionMultiSum (the default injection_type="muli_sum" used by
    every shipped config): local*gate(global) + global, all combined at the local
    (finer) spatial resolution."""

    def __init__(self, inp, oup):
        super().__init__()
        self.local_embedding = Conv2dBNAct(inp, oup, kernel_size=1, act=False)
        self.global_embedding = Conv2dBNAct(inp, oup, kernel_size=1, act=False)
        self.global_act = Conv2dBNAct(inp, oup, kernel_size=1, act=False)
        self.act = HSigmoid()

    def forward(self, x_l, x_g):
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(
            self.act(global_act), size=(H, W), mode="bilinear", align_corners=False
        )

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=(H, W), mode="bilinear", align_corners=False)

        return local_feat * sig_act + global_feat


class Topformer(nn.Module):
    """topformer.py Topformer backbone, injection=True path (the only path used by
    every shipped config)."""

    def __init__(
        self,
        cfgs,
        channels,
        out_channels,
        embed_out_indice,
        decode_out_indices,
        depths,
        num_heads,
        c2t_stride,
        drop_path_rate,
        key_dim=16,
        attn_ratios=2,
        mlp_ratios=2,
    ):
        super().__init__()
        self.channels = channels
        self.decode_out_indices = decode_out_indices

        self.tpm = TokenPyramidModule(cfgs=cfgs, out_indices=embed_out_indice)
        self.ppa = PyramidPoolAgg(stride=c2t_stride)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]
        self.trans = BasicLayer(
            block_num=depths,
            embedding_dim=sum(channels),
            key_dim=key_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratios,
            attn_ratio=attn_ratios,
            drop_path=dpr,
        )

        self.SIM = nn.ModuleList()
        for i in range(len(channels)):
            if i in decode_out_indices:
                self.SIM.append(InjectionMultiSum(channels[i], out_channels[i]))
            else:
                self.SIM.append(nn.Identity())

    def forward(self, x):
        outputs = self.tpm(x)
        out = self.ppa(outputs)
        out = self.trans(out)

        xx = out.split(self.channels, dim=1)
        results = []
        for i in range(len(self.channels)):
            if i in self.decode_out_indices:
                local_tokens = outputs[i]
                global_semantics = xx[i]
                results.append(self.SIM[i](local_tokens, global_semantics))
        return results


class SimpleHead(nn.Module):
    """simple_head.py SimpleHead + decode_head.py BaseDecodeHead.cls_seg: bilinear
    resize-and-sum the 3 injected feature maps to the finest scale, a (depthwise, per
    is_dw=True) 1x1 conv-BN-act fuse, dropout, 1x1 conv to num_classes."""

    def __init__(self, in_channels, channels, num_classes, dropout_ratio=0.1, is_dw=True):
        super().__init__()
        groups = channels if is_dw else 1
        self.linear_fuse = Conv2dBNAct(channels, channels, kernel_size=1, groups=groups, act=True)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def agg_res(self, preds):
        outs = preds[0]
        for pred in preds[1:]:
            pred = F.interpolate(pred, size=outs.shape[2:], mode="bilinear", align_corners=False)
            outs = outs + pred
        return outs

    def forward(self, inputs):
        x = self.agg_res(inputs)
        feat = self.linear_fuse(x)
        if self.dropout is not None:
            feat = self.dropout(feat)
        return self.conv_seg(feat)


class TopformerSegmentor(nn.Module):
    """EncoderDecoder(backbone=Topformer, decode_head=SimpleHead), topformer_tiny
    config (local_configs/topformer/topformer_tiny.py)."""

    def __init__(self):
        super().__init__()
        cfgs = [
            [3, 1, 16, 1],
            [3, 4, 16, 2],
            [3, 3, 16, 1],
            [5, 3, 32, 2],
            [5, 3, 32, 1],
            [3, 3, 64, 2],
            [3, 3, 64, 1],
            [5, 6, 96, 2],
            [5, 6, 96, 1],
        ]
        channels = [16, 32, 64, 96]
        out_channels = [None, 128, 128, 128]
        embed_out_indice = [2, 4, 6, 8]
        decode_out_indices = [1, 2, 3]

        self.backbone = Topformer(
            cfgs=cfgs,
            channels=channels,
            out_channels=out_channels,
            embed_out_indice=embed_out_indice,
            decode_out_indices=decode_out_indices,
            depths=4,
            num_heads=4,
            c2t_stride=2,
            drop_path_rate=0.1,
        )
        self.decode_head = SimpleHead(
            in_channels=[128, 128, 128],
            channels=128,
            num_classes=150,
            dropout_ratio=0.1,
            is_dw=True,
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        feats = self.backbone(x)
        seg_logits = self.decode_head(feats)
        seg_logits = F.interpolate(
            seg_logits, size=input_size, mode="bilinear", align_corners=False
        )
        return seg_logits


def build_topformer():
    model = TopformerSegmentor()
    model.eval()
    return model


def example_input_topformer():
    torch.manual_seed(0)
    return torch.randn(1, 3, 128, 128)


MENAGERIE_ZOO = "ported-pytorch"

MENAGERIE_ENTRIES = [
    ("TopFormer", "build_topformer", "example_input_topformer", 2022, "PORT"),
]
