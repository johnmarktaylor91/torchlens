# SOURCE: vendored from TUMFTM/CamRaDepth @ ca5b9fd30d1f (src/models/CamRaDepth.py,
# src/models/simplified_attention.py, src/utils/utils.py)
"""CamRaDepth: Semantic Guided Depth Estimation Using Monocular Camera and Sparse Radar.

A camera+radar depth-completion network built around a NVIDIA "Simplified Transformer"
(DEST-style overlap-patch-embed + MaxPool-attention pyramid encoder, 4 stages) with a
dense-skip convolutional decoder that predicts depth at 3 intermediate resolutions plus a
final full-resolution depth map, and optionally supervised/unsupervised semantic-segmentation
side branches whose outputs are concatenated back into the depth decoder features.

The upstream repo wires every hyperparameter through a single global `argparse`/`easydict`
singleton (`utils/args.py`) that parses `sys.argv`, asserts a dataset split file exists on
disk, and can block on an interactive `input()` prompt -- none of that is architecture, and
none of it is base-env-importable as a library module. This file inlines the *architecture*
verbatim from `models/simplified_attention.py`, `utils/utils.py`, and `models/CamRaDepth.py`,
and replaces every `args.<field>` reference with the literal default value the real
`utils/args.py` assigns for that field (`groupnorm_divisor=16`, `num_classes=21`,
`input_channels=7`, `image_dimension=(416, 800)`) -- i.e. this reproduces the network you get
from running the upstream repo with its default CLI flags. `torchinfo.summary` (a debug-only
print utility, not part of the model) is dropped.
"""

import math
from functools import partial

import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, trunc_normal_

MENAGERIE_ZOO = "vendored-pytorch"

# Upstream default hyperparameters from utils/args.py (non-architectural config, not a
# torchlens-visible module -- kept as plain module constants instead of the global `args`
# argparse/easydict singleton).
GROUPNORM_DIVISOR = 16
NUM_CLASSES = 21


# ---------------------------------------------------------------------------
# models/simplified_attention.py (verbatim; NVIDIA DEST-derived encoder)
# ---------------------------------------------------------------------------
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, C, N = x.shape
        x = x.reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2)
        return x


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.num_groups = hidden_features // GROUPNORM_DIVISOR
        self.norm1 = nn.GroupNorm(hidden_features // GROUPNORM_DIVISOR, hidden_features)
        self.norm2 = nn.GroupNorm(out_features // GROUPNORM_DIVISOR, hidden_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.dwconv(x, H, W)
        x = self.norm2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention_MaxPool(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        output_dim=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        output_dim = output_dim if output_dim is not None else dim
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.softmax = nn.Softmax()

        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Conv1d(dim, dim, 1, bias=qkv_bias)
        self.k = nn.Conv1d(dim, dim, 1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv1d(dim, output_dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.num_groups = dim // GROUPNORM_DIVISOR
            self.norm = nn.GroupNorm(self.num_groups, dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, H, W):
        B, C, N = x.shape
        q = self.q(x)
        q = q.reshape(B, self.num_heads, C // self.num_heads, N)
        q = q.permute(0, 1, 3, 2)

        if self.sr_ratio > 1:
            x_ = x.reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1)
            x_ = self.norm(x_)
            k = self.k(x_).reshape(B, self.num_heads, C // self.num_heads, -1)
        else:
            k = self.k(x).reshape(B, self.num_heads, C // self.num_heads, -1)
        v = torch.mean(x, 2, True).repeat(1, 1, self.num_heads).transpose(-2, -1)
        attn = (q @ k) * self.scale
        attn, _ = torch.max(attn, -1)
        out = attn.transpose(-2, -1) @ v
        out = out.transpose(-2, -1)
        out = self.proj(out)
        return out


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.ReLU,
        sr_ratio=1,
        out_features=None,
    ):
        super().__init__()
        out_features = out_features if out_features is not None else dim
        self.num_groups = dim // GROUPNORM_DIVISOR
        self.norm1 = nn.GroupNorm(self.num_groups, dim)
        self.norm2 = nn.GroupNorm(self.num_groups, dim)

        self.attn = Attention_MaxPool(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            output_dim=out_features,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(
            in_features=dim,
            out_features=out_features,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x_orig, H, W):
        x = self.norm1(x_orig)
        x = x_orig + self.drop_path((self.attn(x, H, W)))
        x = x + self.drop_path(self.mlp1(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=(224, 224), patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = img_size[0] // patch_size[0] * img_size[1] // patch_size[1]
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )

        self.norm = nn.GroupNorm(embed_dim // GROUPNORM_DIVISOR, embed_dim)

        self.H = (img_size[0] - patch_size[0] + 2 * (patch_size[0] // 2)) / stride + 1
        self.W = (img_size[1] - patch_size[1] + 2 * (patch_size[1] // 2)) / stride + 1
        self.feat_shape = (int(self.H), int(self.W))
        self.N = int(self.feat_shape[0] * self.feat_shape[1])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        x = x.flatten(2)
        return x, H, W


class SimplifiedTransformer(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.num_layers = depths

        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0]
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=(img_size[0] // 4, img_size[1] // 4),
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=(img_size[0] // 8, img_size[1] // 8),
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=(img_size[0] // 16, img_size[1] // 16),
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ]
        )

        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ]
        )

        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ]
        )

        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed1", "pos_embed2", "pos_embed3", "pos_embed4", "cls_token"}

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        ref_feat = {"1": [], "2": [], "3": [], "4": []}

        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = x.reshape(B, -1, H, W).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = x.reshape(B, -1, H, W).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = x.reshape(B, -1, H, W).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = x.reshape(B, -1, H, W).contiguous()
        outs.append(x)
        return outs, ref_feat

    def forward(self, x):
        x, ref_feat = self.forward_features(x)
        return x, ref_feat


# ---------------------------------------------------------------------------
# utils/utils.py (decoder blocks actually used by CamRaDepth; verbatim)
# ---------------------------------------------------------------------------
def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class ConvLayer(nn.Module):
    """A simple convolution layer with a norm layer and a non-linear activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        maxpool_bool=False,
        activation="gelu",
        **kwargs,
    ):
        super().__init__()
        self.activation = {
            "elu": nn.ELU(inplace=True),
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
        }[activation]
        self.norm_layer = nn.GroupNorm
        n_groups = out_channels // GROUPNORM_DIVISOR
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            self.norm_layer(n_groups, out_channels),
            self.activation,
        )

        self.maxpool_bool = maxpool_bool
        if maxpool_bool:
            self.maxpool = nn.MaxPool2d(2)

        self.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.model(x)
        if self.maxpool_bool:
            x = self.maxpool(x)
        return x


class AttentionBlcok(nn.Module):
    """
    Creates an attention vector at a desired length, corresponding to an outer-scope feature
    maps block. The learned attention is between the latter different channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.average_pooling(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.sigmoid(out)
        return out


class SparaseDenseLayer(nn.Module):
    """
    Has two branches: A convolution, and an attention vector that learns the correspondences
    between the convolution's output channels.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=128,
        maxpool_bool=False,
        dense=False,
        as_final_block=False,
    ):
        super().__init__()
        self.as_final_block = as_final_block
        self.conv3x3 = ConvLayer(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.atten = AttentionBlcok(in_channels, mid_channels)

        if as_final_block:
            self.conv_combine = nn.Conv2d(
                mid_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
        else:
            self.conv_combine = ConvLayer(
                mid_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        out = self.conv3x3(x)
        atten = self.atten(x)
        out = out * atten + out
        out = self.conv_combine(out)
        return out


class ShortResBlock(nn.Module):
    """A short dense blocks, with reducing channels as it goes deeper."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=128,
        num_layers=3,
        maxpool_bool=False,
        as_final_block=False,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.maxpool_bool = maxpool_bool

        self.layers = nn.ModuleList()

        multi_factor = 0.75
        inp = in_channels
        out = int(mid_channels * multi_factor)
        for i in range(num_layers):
            self.layers.append(ConvLayer(inp, out, kernel_size=3, stride=1, padding=1))
            inp += out
            multi_factor -= 0.25
            out = out_channels if i == num_layers - 2 else int(mid_channels * multi_factor)

    def forward(self, x):
        for layer in self.layers[:-1]:
            out = layer(x)
            x = torch.cat((x, out), dim=1)
        x = self.layers[-1](x)
        return x


class Seg_Block(nn.Module):
    """Creates a segmentation map out of an input block of logits."""

    def __init__(self, num_classes=21):
        super().__init__()
        self.seg_num_classes = num_classes

    def forward(self, seg_logits):
        seg_map = torch.argmax(seg_logits, dim=1, keepdim=True)
        seg_map = seg_map / self.seg_num_classes
        return seg_map


class Decoder(nn.Module):
    """A simple upsampling layer, further processed with a convolutional block of choice."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=128,
        dense=False,
        skip_size=None,
        as_final_block=False,
        block=ShortResBlock,
        **kwargs,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.dense = dense
        self.incoming_skip = skip_size is not None

        self.upsample = nn.Upsample(scale_factor=2, mode="bicubic")
        if self.incoming_skip:
            self.conv = block(
                in_channels + skip_size,
                out_channels,
                mid_channels=mid_channels,
                dense=dense,
                maxpool_bool=False,
                as_final_block=as_final_block,
                **kwargs,
            )
        else:
            self.conv = block(
                in_channels,
                out_channels,
                mid_channels=mid_channels,
                dense=dense,
                maxpool_bool=False,
                as_final_block=as_final_block,
                **kwargs,
            )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if self.incoming_skip:
            assert skip is not None
            x = torch.cat((x, skip), dim=1)
        out = self.conv(x)
        return out


class Depth_Activation(nn.Module):
    """
    Create a depth map, by using a sigmoid activation, and then a linear convolution, for fine
    scaling and stretching.
    """

    def __init__(self, input, output, activ_fuction=nn.Sigmoid):
        super().__init__()
        iter_channel = 32
        self.acti_func = activ_fuction()
        self.conv_1 = nn.Conv2d(input, iter_channel, kernel_size=3, padding=1, bias=True)
        self.conv_2 = nn.Conv2d(iter_channel * 1, output, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x_inter = self.conv_1(x)
        x_sigmoid = self.acti_func(x_inter)
        x = self.conv_2(x_sigmoid)
        return x


# ---------------------------------------------------------------------------
# models/CamRaDepth.py (verbatim architecture; `args.*` -> literal upstream defaults)
# ---------------------------------------------------------------------------
def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


class CamRaDepth(nn.Module):
    def __init__(
        self,
        img_size=(416, 800),
        heads=(1, 2, 4, 8),
        ff_expansion=(8, 8, 4, 4),
        reduction_ratio=(8, 4, 2, 1),
        depths=(3, 10, 16, 5),
        dims=(64, 128, 160, 256),
        input_channels=None,
        supervised_seg=False,
        unsupervised_seg=False,
        num_classes=NUM_CLASSES,
        **kwargs,
    ):
        super().__init__()

        # Hyperparameters
        self.depths = depths
        self.mid_channels = 128
        self.num_classes = num_classes
        self.dense = True
        self.dims = dims
        self.as_final_block = False
        self.unsupervised_seg = unsupervised_seg
        self.supervised_seg = supervised_seg
        self.img_size = img_size
        input_channels = input_channels if input_channels is not None else 7

        dims, heads, ff_expansion, reduction_ratio, self.depths = map(
            partial(cast_tuple, depth=4), (dims, heads, ff_expansion, reduction_ratio, self.depths)
        )
        assert all(
            [*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, self.depths))]
        ), (
            "only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values"
        )
        assert input_channels > 0, "input_channels must be > 0"

        # Architecture

        # Encoder
        self.dest_encoder = SimplifiedTransformer(
            img_size=img_size,
            in_chans=input_channels,
            num_classes=self.num_classes,
            embed_dims=dims,
            num_heads=heads,
            mlp_ratios=ff_expansion,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0,
            drop_path_rate=0.1,
            attn_drop_rate=0.0,
            depths=self.depths,
            sr_ratios=reduction_ratio,
        )

        conv_layer = ConvLayer
        self.from_encoder_1 = conv_layer(dims[-1], dims[-1], 1, padding=0)
        self.from_encoder_2 = conv_layer(dims[-2], dims[-2], 1, padding=0)
        self.from_encoder_3 = conv_layer(dims[-3], dims[-3], 1, padding=0)
        self.from_encoder_4 = conv_layer(dims[-4], dims[-4], 1, padding=0)

        # Depth
        self.depth_upsample = nn.ModuleList(
            [
                Decoder(
                    dims[-1],
                    self.mid_channels,
                    skip_size=dims[-2],
                    dense=self.dense,
                    as_final_block=self.as_final_block,
                    block=ShortResBlock,
                ),
                Decoder(
                    self.mid_channels,
                    self.mid_channels,
                    skip_size=dims[-3],
                    dense=self.dense,
                    as_final_block=self.as_final_block,
                    block=ShortResBlock,
                ),
                Decoder(
                    self.mid_channels,
                    self.mid_channels,
                    skip_size=dims[-4],
                    dense=self.dense,
                    as_final_block=self.as_final_block,
                    block=ShortResBlock,
                ),
                Decoder(
                    self.mid_channels + 1,
                    self.mid_channels,
                    dense=self.dense,
                    as_final_block=self.as_final_block,
                    block=ShortResBlock,
                ),
                Decoder(
                    self.mid_channels + 1,
                    self.mid_channels,
                    skip_size=input_channels,
                    dense=self.dense,
                    as_final_block=self.as_final_block,
                    block=ShortResBlock,
                    mid_channels=128,
                ),
            ]
        )

        self.depth_activation_3 = Depth_Activation(self.mid_channels, 1)
        self.depth_activation_4 = Depth_Activation(
            self.mid_channels + 1 * self.supervised_seg + 1 * self.unsupervised_seg, 1
        )
        self.depth_activation_5 = Depth_Activation(
            self.mid_channels + 1 * self.supervised_seg + 1 * self.unsupervised_seg, 1
        )

        # Seg
        if self.supervised_seg or self.unsupervised_seg:
            self.seg_upsample = nn.ModuleList(
                [
                    Decoder(
                        self.mid_channels + 1,
                        self.mid_channels,
                        dense=self.dense,
                        as_final_block=self.as_final_block,
                        block=ShortResBlock,
                    ),
                    Decoder(
                        self.mid_channels + 1,
                        self.mid_channels,
                        skip_size=input_channels,
                        dense=self.dense,
                        as_final_block=self.as_final_block,
                        block=ShortResBlock,
                        mid_channels=128,
                    ),
                ]
            )

        if self.supervised_seg:
            self.seg_block = Seg_Block(self.num_classes)
            self.seg_conv_stage_4 = nn.Conv2d(
                self.mid_channels, self.num_classes, kernel_size=3, stride=1, padding=1
            )
            self.seg_conv_final = nn.Conv2d(
                self.mid_channels, self.num_classes, kernel_size=3, stride=1, padding=1
            )

        if self.unsupervised_seg:
            self.unsup_seg_block = Seg_Block(19)
            self.unsup_stage_4 = nn.Conv2d(
                self.mid_channels, 19, kernel_size=3, stride=1, padding=1
            )
            self.unsup_final = nn.Conv2d(self.mid_channels, 19, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout2d(0.2)

    def dest_decoder(self, lay_out, x):
        unsup_map = None
        sup_seg_map = None
        seg_logits_final = None
        seg_map = None
        seg_features = None

        # Convolve the attention blocks, to be used in skip connections.
        encoded_1 = self.from_encoder_1(lay_out[-1])
        encoded_2 = self.from_encoder_2(lay_out[-2])
        encoded_3 = self.from_encoder_3(lay_out[-3])
        encoded_4 = self.from_encoder_4(lay_out[-4])

        # Perform upscaling, concatenation with the appropriate skip connection, and further convolution.
        decoder_stage_1 = self.dropout(self.depth_upsample[0](encoded_1, encoded_2))
        decoder_stage_2 = self.dropout(self.depth_upsample[1](decoder_stage_1, encoded_3))

        decoder_stage_3 = self.dropout(self.depth_upsample[2](decoder_stage_2, encoded_4))
        inter_depth_3 = self.depth_activation_3(decoder_stage_3)
        decoder_stage_3 = torch.cat([decoder_stage_3, inter_depth_3], 1)

        decoder_stage_4 = self.dropout(self.depth_upsample[3](decoder_stage_3))

        if self.supervised_seg or self.unsupervised_seg:
            seg_features = self.dropout(self.seg_upsample[0](decoder_stage_3))

        if self.supervised_seg:
            seg_logits_inter = self.seg_conv_stage_4(seg_features)
            sup_seg_map = self.seg_block(seg_logits_inter)
            seg_map = sup_seg_map

        if self.unsupervised_seg:
            unsup_map = self.unsup_stage_4(seg_features)
            unsup_map = self.unsup_seg_block(unsup_map)
            seg_map = unsup_map if sup_seg_map is None else torch.cat([sup_seg_map, unsup_map], 1)

        if self.supervised_seg:
            seg_features = torch.cat((seg_features, sup_seg_map), dim=1)
        elif self.unsupervised_seg:
            seg_features = torch.cat((seg_features, unsup_map), dim=1)

        tmp = (
            torch.cat((decoder_stage_4, seg_map), dim=1) if seg_map is not None else decoder_stage_4
        )

        inter_depth_4 = self.depth_activation_4(tmp)
        decoder_stage_4 = torch.cat([decoder_stage_4, inter_depth_4], 1)

        # Final predictions - last stage:
        decoder_stage_5 = self.dropout(self.depth_upsample[4](decoder_stage_4, x))

        if self.supervised_seg or self.unsupervised_seg:
            seg_features = self.dropout(self.seg_upsample[1](seg_features, x))

        if self.supervised_seg:
            seg_logits_final = self.seg_conv_final(seg_features)
            sup_seg_map = self.seg_block(seg_logits_final)
            seg_map = sup_seg_map

        if self.unsupervised_seg:
            unsup_map = self.unsup_final(seg_features)
            unsup_map = self.unsup_seg_block(unsup_map)
            seg_map = unsup_map if sup_seg_map is None else torch.cat([sup_seg_map, unsup_map], 1)

        tmp = (
            torch.cat((decoder_stage_5, seg_map), dim=1) if seg_map is not None else decoder_stage_5
        )
        final_depth = self.depth_activation_5(tmp)

        return {
            "depth": {
                "intermediate_depths": (None, None, inter_depth_3, inter_depth_4),
                "final_depth": final_depth,
            },
            "seg": {
                "final_seg": seg_logits_final,
                "intermediate_seg": None,
                "unsup_map": unsup_map,
            },
        }

    def forward(self, x):
        layer_outputs, _ = self.dest_encoder(x)
        ret_dict = self.dest_decoder(layer_outputs, x)
        return ret_dict


# ---------------------------------------------------------------------------
# Menagerie build/example wiring
# ---------------------------------------------------------------------------
def build_camradepth():
    # Upstream default architecture ("base" model: no seg branches, 7 input channels =
    # RGB + sparse radar depth/velocity/uv maps), shrunk to a tiny input resolution and
    # depth schedule for a fast, faithful forward trace (real transformer_depths="5" preset
    # is (3, 10, 16, 5) per stage -- far too slow for a menagerie smoke trace).
    model = CamRaDepth(img_size=(64, 128), depths=(1, 1, 1, 1), input_channels=7)
    model.eval()
    return model


def example_input_camradepth():
    # 7 channels = RGB (3) + sparse radar depth/uv/velocity planes, per upstream default.
    return torch.randn(1, 7, 64, 128)


MENAGERIE_ENTRIES = [
    ("CamRaDepth", build_camradepth, example_input_camradepth, 2023, MENAGERIE_ZOO),
]
