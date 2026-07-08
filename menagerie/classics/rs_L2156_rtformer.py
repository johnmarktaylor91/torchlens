# FAITHFUL PORT of https://github.com/PaddlePaddle/PaddleSeg @ a7864dfb1f9b16c774507c2da0e62feca7e9cdf9
# (paddleseg/models/rtformer.py, release/2.9 branch; original framework: PaddlePaddle)
# -- RTFormer: Efficient Design for Real-Time Semantic Segmentation with Transformer
# (NeurIPS 2022, arXiv:2210.07124). This is the ONLY public reference implementation
# (PaddlePaddle-only, no PyTorch port exists in RolandGao/RegSeg-adjacent or other repos
# checked); it is not installable in the base torch env (needs the `paddle` framework, a
# separate deep-learning stack from torch), so per the build ladder this is a faithful
# line-by-line transcription of the real PaddlePaddle module graph into torch, not a
# from-scratch reimplementation from the paper text. Every mechanism in the original
# `RTFormer`/`BasicBlock`/`MLP`/`ExternalAttention`/`EABlock`/`DAPPM`/`SegHead` classes is
# preserved: the dual-resolution ResNet-style stem+stages, the low-res EABlock external
# attention (learned global key/value memory, softmax-normalize-then-renormalize "double
# normalization" `_act_dn`), the cross-resolution external attention in the high-res branch
# (`_act_sn`/cross_kv pooled key-value bank, `use_cross_kv=True`), the compression +
# bilinear-injection cross-branch fusion, and the DAPPM multi-scale pyramid-pooling neck.
# Paddle-specific op names are mapped 1:1 to torch equivalents (`nn.Conv2D`->`nn.Conv2d`,
# `nn.BatchNorm2D`->`nn.BatchNorm2d`, `nn.AdaptiveMaxPool2D`->`nn.AdaptiveMaxPool2d`,
# `paddle.shape(x)[2:]`->`x.shape[-2:]`, `paddle.concat`->`torch.cat`,
# `paddle.ParamAttr(learning_rate=...)`->dropped (torch has no per-tensor LR-multiplier
# attribute; the `lr_mult` argument is a training-schedule knob with no effect on the
# forward-pass architecture) and `DropPath`/`Identity`/kaiming-normal & trunc-normal weight
# initializers are re-implemented with their standard torch equivalents (stochastic-depth
# `drop_path` formula and `trunc_normal_/kaiming_normal_` inits match the PaddleSeg
# `transformer_utils.py`/`param_init.py` helpers). The `pretrained`-checkpoint loading path
# is dropped (always random-init here); `use_aux_head`'s auxiliary segmentation head is kept
# as a real submodule (constructed, unused in eval-mode forward, matching the original
# `if self.training and self.use_aux_head:` guard).
"""RTFormer (base config) real-time semantic segmentation transformer, faithfully ported
from PaddlePaddle/PaddleSeg's paddleseg/models/rtformer.py to torch."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample, ported from PaddleSeg transformer_utils.py."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor)
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias_attr=False, **kwargs):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, stride, padding, bias=bias_attr, **kwargs
    )


def bn2d(in_channels, bn_mom=0.1, **kwargs):
    return nn.BatchNorm2d(in_channels, momentum=bn_mom, **kwargs)


def _init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, no_relu=False):
        super().__init__()
        self.conv1 = conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = bn2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = bn2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out if self.no_relu else self.relu(out)


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, drop_rate=0.0):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = bn2d(in_channels, eps=1e-06)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class ExternalAttention(nn.Module):
    """The ExternalAttention module: learned global key/value bank attention with an
    optional cross-resolution (cross_kv) variant, faithfully ported from PaddleSeg."""

    def __init__(self, in_channels, out_channels, inter_channels, num_heads=8, use_cross_kv=False):
        super().__init__()
        assert out_channels % num_heads == 0, (
            "out_channels ({}) should be be a multiple of num_heads ({})".format(
                out_channels, num_heads
            )
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.use_cross_kv = use_cross_kv
        self.norm = bn2d(in_channels)
        self.same_in_out_chs = in_channels == out_channels

        if use_cross_kv:
            assert self.same_in_out_chs, (
                "in_channels is not equal to out_channels when use_cross_kv is True"
            )
        else:
            self.k = nn.Parameter(torch.empty(inter_channels, in_channels, 1, 1))
            self.v = nn.Parameter(torch.empty(out_channels, inter_channels, 1, 1))
            nn.init.normal_(self.k, std=0.001)
            nn.init.normal_(self.v, std=0.001)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def _act_sn(self, x):
        x = x.reshape(-1, self.inter_channels, x.shape[-2], x.shape[-1]) * (
            self.inter_channels**-0.5
        )
        x = F.softmax(x, dim=1)
        x = x.reshape(1, -1, x.shape[-2], x.shape[-1])
        return x

    def _act_dn(self, x):
        h, w = x.shape[2], x.shape[3]
        x = x.reshape(x.shape[0], self.num_heads, self.inter_channels // self.num_heads, -1)
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-06)
        x = x.reshape(x.shape[0], self.inter_channels, h, w)
        return x

    def forward(self, x, cross_k=None, cross_v=None):
        """
        Args:
            x: input tensor, shape (n, c_in, h, w).
            cross_k: optional, shape (n*144, c_in, 1, 1).
            cross_v: optional, shape (n*c_in, 144, 1, 1).
        """
        x = self.norm(x)
        if not self.use_cross_kv:
            x = F.conv2d(
                x, self.k, bias=None, stride=2 if not self.same_in_out_chs else 1, padding=0
            )
            x = self._act_dn(x)
            x = F.conv2d(x, self.v, bias=None, stride=1, padding=0)
        else:
            assert (cross_k is not None) and (cross_v is not None), (
                "cross_k and cross_v should no be None when use_cross_kv"
            )
            B = x.shape[0]
            assert B > 0, "The first dim of x ({}) should be greater than 0".format(B)
            x = x.reshape(1, -1, x.shape[-2], x.shape[-1])
            x = F.conv2d(x, cross_k, bias=None, stride=1, padding=0, groups=B)
            x = self._act_sn(x)
            x = F.conv2d(x, cross_v, bias=None, stride=1, padding=0, groups=B)
            x = x.reshape(-1, self.in_channels, x.shape[-2], x.shape[-1])
        return x


class EABlock(nn.Module):
    """External-Attention transformer block (dual-branch high-res/low-res fusion),
    faithfully ported from PaddleSeg."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads=8,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_injection=True,
        use_cross_kv=True,
        cross_size=12,
    ):
        super().__init__()
        in_channels_h, in_channels_l = in_channels
        out_channels_h, out_channels_l = out_channels
        assert in_channels_h == out_channels_h, "in_channels_h is not equal to out_channels_h"
        self.out_channels_h = out_channels_h
        self.proj_flag = in_channels_l != out_channels_l
        self.use_injection = use_injection
        self.use_cross_kv = use_cross_kv
        self.cross_size = cross_size
        # low resolution
        if self.proj_flag:
            self.attn_shortcut_l = nn.Sequential(
                bn2d(in_channels_l), conv2d(in_channels_l, out_channels_l, 1, 2, 0)
            )
            self.attn_shortcut_l.apply(_init_weights_kaiming)
        self.attn_l = ExternalAttention(
            in_channels_l,
            out_channels_l,
            inter_channels=out_channels_l,
            num_heads=num_heads,
            use_cross_kv=False,
        )
        self.mlp_l = MLP(out_channels_l, drop_rate=drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else Identity()

        # compression
        self.compression = nn.Sequential(
            bn2d(out_channels_l), nn.ReLU(), conv2d(out_channels_l, out_channels_h, kernel_size=1)
        )
        self.compression.apply(_init_weights_kaiming)

        # high resolution
        self.attn_h = ExternalAttention(
            in_channels_h,
            in_channels_h,
            inter_channels=cross_size * cross_size,
            num_heads=num_heads,
            use_cross_kv=use_cross_kv,
        )
        self.mlp_h = MLP(out_channels_h, drop_rate=drop_rate)
        if use_cross_kv:
            self.cross_kv = nn.Sequential(
                bn2d(out_channels_l),
                nn.AdaptiveMaxPool2d(output_size=(self.cross_size, self.cross_size)),
                conv2d(out_channels_l, 2 * out_channels_h, 1, 1, 0),
            )
            self.cross_kv.apply(self._init_weights)

        # injection
        if use_injection:
            self.down = nn.Sequential(
                bn2d(out_channels_h),
                nn.ReLU(),
                conv2d(out_channels_h, out_channels_l // 2, kernel_size=3, stride=2, padding=1),
                bn2d(out_channels_l // 2),
                nn.ReLU(),
                conv2d(out_channels_l // 2, out_channels_l, kernel_size=3, stride=2, padding=1),
            )
            self.down.apply(_init_weights_kaiming)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_h, x_l = x

        # low resolution
        x_l_res = self.attn_shortcut_l(x_l) if self.proj_flag else x_l
        x_l = x_l_res + self.drop_path(self.attn_l(x_l))
        x_l = x_l + self.drop_path(self.mlp_l(x_l))  # n,out_chs_l,h,w

        # compression
        x_h_shape = x_h.shape[-2:]
        x_l_cp = self.compression(x_l)
        x_h = x_h + F.interpolate(x_l_cp, size=x_h_shape, mode="bilinear")

        # high resolution
        if not self.use_cross_kv:
            x_h = x_h + self.drop_path(self.attn_h(x_h))  # n,out_chs_h,h,w
        else:
            cross_kv = self.cross_kv(x_l)  # n,2*out_channels_h,12,12
            cross_k, cross_v = torch.split(cross_kv, cross_kv.shape[1] // 2, dim=1)
            cross_k = cross_k.permute(0, 2, 3, 1).reshape(-1, self.out_channels_h, 1, 1)
            cross_v = cross_v.reshape(-1, self.cross_size * self.cross_size, 1, 1)
            x_h = x_h + self.drop_path(self.attn_h(x_h, cross_k, cross_v))  # n,out_chs_h,h,w

        x_h = x_h + self.drop_path(self.mlp_h(x_h))

        # injection
        if self.use_injection:
            x_l = x_l + self.down(x_h)

        return x_h, x_l


class DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling Module, faithfully ported from PaddleSeg."""

    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2, count_include_pad=False),
            bn2d(in_channels),
            nn.ReLU(),
            conv2d(in_channels, inter_channels, kernel_size=1),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4, count_include_pad=False),
            bn2d(in_channels),
            nn.ReLU(),
            conv2d(in_channels, inter_channels, kernel_size=1),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8, count_include_pad=False),
            bn2d(in_channels),
            nn.ReLU(),
            conv2d(in_channels, inter_channels, kernel_size=1),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            bn2d(in_channels),
            nn.ReLU(),
            conv2d(in_channels, inter_channels, kernel_size=1),
        )
        self.scale0 = nn.Sequential(
            bn2d(in_channels), nn.ReLU(), conv2d(in_channels, inter_channels, kernel_size=1)
        )
        self.process1 = nn.Sequential(
            bn2d(inter_channels),
            nn.ReLU(),
            conv2d(inter_channels, inter_channels, kernel_size=3, padding=1),
        )
        self.process2 = nn.Sequential(
            bn2d(inter_channels),
            nn.ReLU(),
            conv2d(inter_channels, inter_channels, kernel_size=3, padding=1),
        )
        self.process3 = nn.Sequential(
            bn2d(inter_channels),
            nn.ReLU(),
            conv2d(inter_channels, inter_channels, kernel_size=3, padding=1),
        )
        self.process4 = nn.Sequential(
            bn2d(inter_channels),
            nn.ReLU(),
            conv2d(inter_channels, inter_channels, kernel_size=3, padding=1),
        )
        self.compression = nn.Sequential(
            bn2d(inter_channels * 5),
            nn.ReLU(),
            conv2d(inter_channels * 5, out_channels, kernel_size=1),
        )
        self.shortcut = nn.Sequential(
            bn2d(in_channels), nn.ReLU(), conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x_shape = x.shape[-2:]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(
            self.process1(
                (F.interpolate(self.scale1(x), size=x_shape, mode="bilinear") + x_list[0])
            )
        )
        x_list.append(
            self.process2(
                (F.interpolate(self.scale2(x), size=x_shape, mode="bilinear") + x_list[1])
            )
        )
        x_list.append(
            self.process3(
                (F.interpolate(self.scale3(x), size=x_shape, mode="bilinear") + x_list[2])
            )
        )
        x_list.append(
            self.process4(
                (F.interpolate(self.scale4(x), size=x_shape, mode="bilinear") + x_list[3])
            )
        )

        out = self.compression(torch.cat(x_list, dim=1)) + self.shortcut(x)
        return out


class SegHead(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.bn1 = bn2d(in_channels)
        self.conv1 = conv2d(in_channels, inter_channels, kernel_size=3, padding=1)
        self.bn2 = bn2d(inter_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv2d(inter_channels, out_channels, kernel_size=1, padding=0, bias_attr=True)

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        return out


class RTFormer(nn.Module):
    """RTFormer: Efficient Design for Real-Time Semantic Segmentation with Transformer.
    Faithfully ported (base config) from PaddleSeg's paddleseg/models/rtformer.py."""

    def __init__(
        self,
        num_classes,
        layer_nums=(2, 2, 2, 2),
        base_channels=64,
        spp_channels=128,
        num_heads=8,
        head_channels=128,
        drop_rate=0.0,
        drop_path_rate=0.2,
        use_aux_head=True,
        use_injection=(True, True),
        cross_size=12,
        in_channels=3,
    ):
        super().__init__()
        self.base_channels = base_channels
        base_chs = base_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_chs, kernel_size=3, stride=2, padding=1),
            bn2d(base_chs),
            nn.ReLU(),
            nn.Conv2d(base_chs, base_chs, kernel_size=3, stride=2, padding=1),
            bn2d(base_chs),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(BasicBlock, base_chs, base_chs, layer_nums[0])
        self.layer2 = self._make_layer(BasicBlock, base_chs, base_chs * 2, layer_nums[1], stride=2)
        self.layer3 = self._make_layer(
            BasicBlock, base_chs * 2, base_chs * 4, layer_nums[2], stride=2
        )
        self.layer3_ = self._make_layer(BasicBlock, base_chs * 2, base_chs * 2, 1)
        self.compression3 = nn.Sequential(
            bn2d(base_chs * 4), nn.ReLU(), conv2d(base_chs * 4, base_chs * 2, kernel_size=1)
        )
        self.layer4 = EABlock(
            in_channels=[base_chs * 2, base_chs * 4],
            out_channels=[base_chs * 2, base_chs * 8],
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=use_injection[0],
            use_cross_kv=True,
            cross_size=cross_size,
        )
        self.layer5 = EABlock(
            in_channels=[base_chs * 2, base_chs * 8],
            out_channels=[base_chs * 2, base_chs * 8],
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=use_injection[1],
            use_cross_kv=True,
            cross_size=cross_size,
        )

        self.spp = DAPPM(base_chs * 8, spp_channels, base_chs * 2)
        self.seghead = SegHead(base_chs * 4, int(head_channels * 2), num_classes)
        self.use_aux_head = use_aux_head
        if self.use_aux_head:
            self.seghead_extra = SegHead(base_chs * 2, head_channels, num_classes)

        self.init_weight()

    def init_weight(self):
        self.conv1.apply(_init_weights_kaiming)
        self.layer1.apply(_init_weights_kaiming)
        self.layer2.apply(_init_weights_kaiming)
        self.layer3.apply(_init_weights_kaiming)
        self.layer3_.apply(_init_weights_kaiming)
        self.compression3.apply(_init_weights_kaiming)
        self.spp.apply(_init_weights_kaiming)
        self.seghead.apply(_init_weights_kaiming)
        if self.use_aux_head:
            self.seghead_extra.apply(_init_weights_kaiming)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                conv2d(in_channels, out_channels, kernel_size=1, stride=stride), bn2d(out_channels)
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(out_channels, out_channels, stride=1, no_relu=True))
            else:
                layers.append(block(out_channels, out_channels, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.layer1(self.conv1(x))  # c, 1/4
        x2 = self.layer2(self.relu(x1))  # 2c, 1/8
        x3 = self.layer3(self.relu(x2))  # 4c, 1/16
        x3_ = x2 + F.interpolate(self.compression3(x3), size=x2.shape[-2:], mode="bilinear")
        x3_ = self.layer3_(self.relu(x3_))  # 2c, 1/8

        x4_, x4 = self.layer4([self.relu(x3_), self.relu(x3)])  # 2c, 1/8; 8c, 1/16
        x5_, x5 = self.layer5([self.relu(x4_), self.relu(x4)])  # 2c, 1/8; 8c, 1/32

        x6 = self.spp(x5)
        x6 = F.interpolate(x6, size=x5_.shape[-2:], mode="bilinear")  # 2c, 1/8
        x_out = self.seghead(torch.cat([x5_, x6], dim=1))  # 4c, 1/8
        logit_list = [x_out]

        if self.training and self.use_aux_head:
            x_out_extra = self.seghead_extra(x3_)
            logit_list.append(x_out_extra)

        logit_list = [
            F.interpolate(logit, size=x.shape[-2:], mode="bilinear", align_corners=False)
            for logit in logit_list
        ]

        return logit_list


# ---------------------------------------------------------------------------
# Menagerie staging helpers
# ---------------------------------------------------------------------------


def build_rtformer():
    model = RTFormer(num_classes=19)
    model.eval()
    return model


def example_input_rtformer():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 128, 256),)


MENAGERIE_ENTRIES = [
    ("RTFormer", "build_rtformer", "example_input_rtformer", 2022, "ported"),
]
