# SOURCE: vendored from changlin31/BossNAS @ bb8b26a09e3ae7889dd118950b022be6b3ce99a0
# https://raw.githubusercontent.com/changlin31/BossNAS/bb8b26a09e3ae7889dd118950b022be6b3ce99a0/retraining_hytra/boss_candidates/bot_op.py
# https://raw.githubusercontent.com/changlin31/BossNAS/bb8b26a09e3ae7889dd118950b022be6b3ce99a0/retraining_hytra/boss_candidates/resnet_op.py
# https://raw.githubusercontent.com/changlin31/BossNAS/bb8b26a09e3ae7889dd118950b022be6b3ce99a0/retraining_hytra/boss_models.py
#
# Li et al. 2021 (ICCV) "BossNAS: Exploring Hybrid CNN-Transformers with Block-wisely
# Self-supervised Neural Architecture Search". This vendors the "HyTra" (hybrid
# CNN-transformer) retraining-time model family: a ResNet-D-style stem feeding four
# stages built from a per-stage 0/1 `encoding` that switches each block between a
# bottleneck-transformer self-attention block (`ResAtt`, adapted from
# lucidrains/bottleneck-transformer-pytorch) and a standard ResNet bottleneck block
# with an inserted positional-encoding-generator depthwise conv (`ResConv`, modified
# from timm's ResNet bottleneck). `bossnet_T0` is the searched T0 architecture
# reported in the paper (stem_width=32, deep stem, SiLU activation, squeeze-excite
# attention on the conv blocks).
#
# No architectural changes were made; only mechanical fixes for import isolation:
#   - The three source files (`bot_op.py`, `resnet_op.py`, `boss_models.py`) are
#     merged into one module and their cross-imports (`from boss_candidates.bot_op
#     import ResAtt`, `from .bot_op import PEG`) replaced with plain in-module
#     references since everything now lives in one file.
#   - `@register_model` (timm's global-registry decorator) is dropped from
#     `bossnet_T0`/`bossnet_T0_nose`/`bossnet_T1` -- registering into timm's global
#     model registry is irrelevant to tracing a single instance and would pollute
#     the shared timm registry on import; the functions are otherwise unchanged.
#   - `pretrained=False` handling (a no-op guard for weight loading that this repo
#     never implements) is left as-is.

import math

import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import AvgPool2dSame, DropPath, create_attn, create_classifier
from timm.models.resnet import drop_blocks
from timm.models.vision_transformer import _cfg


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=128):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).view(b, 3 * self.heads, self.dim_head, h * w).chunk(3, dim=1)

        attn = q.transpose(-2, -1) @ k * self.scale
        attn = attn.softmax(dim=-1)

        out = (v @ attn.transpose(-2, -1)).reshape(b, -1, h, w)
        return out


class PEG(nn.Module):
    def __init__(self, dim, stride):
        super(PEG, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim)

    def forward(self, x):
        # x = x + self.conv(x)
        return self.conv(x)


class ResAtt(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        attn_dim_in,
        stride=1,
        heads=4,
        dim_head=128,
        rel_pos_emb=False,
        act_layer=nn.ReLU,
        avg_down=False,
    ):
        super().__init__()
        activation = act_layer(inplace=True)
        norm_layer = nn.BatchNorm2d
        self.inc = dim
        # shortcut

        if avg_down and (stride == 2 or dim != dim_out):
            avg_stride = stride
            if stride == 1:
                pool = nn.Identity()
            else:
                avg_pool_fn = nn.AvgPool2d
                pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

            self.shortcut = nn.Sequential(
                *[
                    pool,
                    nn.Conv2d(dim, dim_out, 1, stride=1, padding=0, bias=False),
                    norm_layer(dim_out),
                ]
            )
        else:
            if stride == 2:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(dim, dim_out, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dim_out),
                    activation,
                )
            elif dim != dim_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(dim, dim_out, 1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(dim_out),
                    activation,
                )
            else:
                self.shortcut = nn.Identity()

        # contraction and expansion
        attn_dim_in = attn_dim_in
        attn_dim_out = heads * dim_head

        self.proj = nn.Sequential(
            nn.Conv2d(dim, attn_dim_in, 1, bias=False),
            PEG(attn_dim_in, stride=stride),
            nn.BatchNorm2d(attn_dim_in),
        )

        self.net = nn.Sequential(
            activation,
            Attention(
                dim=attn_dim_in,
                heads=heads,
                dim_head=dim_head,
            ),
            nn.BatchNorm2d(attn_dim_out),
            activation,
            nn.Conv2d(attn_dim_out, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out),
        )

        # init last batch norm gamma to zero
        nn.init.zeros_(self.net[-1].weight)

        # final activation
        self.activation = activation

    def zero_init_last_bn(self):
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.proj(x)
        x = self.net(x)
        x += shortcut
        return self.activation(x)


class ResConv(nn.Module):
    # modified from timm/models/resnet.py
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        cardinality=1,
        base_width=64,
        reduce_first=1,
        dilation=1,
        first_dilation=None,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        attn_layer=None,
        aa_layer=None,
        drop_block=None,
        drop_path=None,
        avg_down=False,
    ):
        super(ResConv, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        # act_layer = nn.SiLU

        if avg_down and (stride == 2 or inplanes != outplanes):
            avg_stride = stride if dilation == 1 else 1
            if stride == 1 and dilation == 1:
                pool = nn.Identity()
            else:
                avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
                pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

            self.downsample = nn.Sequential(
                *[
                    pool,
                    nn.Conv2d(inplanes, outplanes, 1, stride=1, padding=0, bias=False),
                    norm_layer(outplanes),
                ]
            )
        else:
            if stride == 2:
                self.downsample = nn.Sequential(
                    nn.Conv2d(inplanes, outplanes, 3, stride=2, padding=1, bias=False),
                    norm_layer(outplanes),
                    act_layer(inplace=True),
                )
            elif inplanes != outplanes:
                self.downsample = nn.Sequential(
                    nn.Conv2d(inplanes, outplanes, 1, stride=1, padding=0, bias=False),
                    norm_layer(outplanes),
                    act_layer(inplace=True),
                )
            else:
                self.downsample = nn.Identity()

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.peg = PEG(first_planes, stride=stride)
        self.bn1 = norm_layer(first_planes)

        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes,
            width,
            kernel_size=3,
            stride=1,
            padding=first_dilation,
            dilation=first_dilation,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)
        self.aa = aa_layer(channels=width, stride=stride) if use_aa else None

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        if attn_layer == "se":
            self.se = create_attn(attn_layer, outplanes)
        else:
            self.se = None

        self.act3 = act_layer(inplace=True)
        # self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_block = drop_block
        self.drop_path = drop_path
        self.inc = inplanes

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = self.downsample(x)
        x = self.conv1(x)
        x = self.peg(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)
        if self.aa is not None:
            x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


def make_boss_blocks(
    encoding,
    channels,
    inplanes,
    output_stride=32,
    reduce_first=1,
    avg_down=False,
    down_kernel_size=1,
    act_layer=nn.ReLU,
    norm_layer=nn.BatchNorm2d,
    drop_block_rate=0.0,
    drop_path_rate=0.0,
    attn_layer=None,
    last_stride=2,
    **kwargs,
):
    stages = []
    feature_info = []
    net_num_blocks = 16
    net_block_idx = 0
    net_stride = 4
    expansion = 4
    dilation = prev_dilation = 1
    heads = [1, 2, 4, 8]

    for stage_idx, (planes, block_encoding, db) in enumerate(
        zip(channels, encoding, drop_blocks(drop_block_rate))
    ):
        num_blocks = len(block_encoding)
        stage_name = f"layer{stage_idx + 1}"  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        stride = last_stride if stage_idx == 3 else stride
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        blocks = []
        if num_blocks == 0:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        inplanes, planes * expansion, kernel_size=3, stride=2, padding=1, bias=False
                    ),
                    norm_layer(planes * expansion),
                    act_layer(inplace=True),
                )
            )
            inplanes = planes * expansion
        else:
            for block_idx in range(num_blocks):
                # downsample = downsample if block_idx == 0 else None
                stride = stride if block_idx == 0 else 1
                block_dpr = (
                    drop_path_rate * net_block_idx / (net_num_blocks - 1)
                )  # stochastic depth linear decay rule

                if block_encoding[block_idx] == 0:
                    blocks.append(
                        ResAtt(
                            dim=inplanes,
                            dim_out=planes * expansion,
                            attn_dim_in=heads[stage_idx] * 64,
                            stride=stride,
                            heads=heads[stage_idx],
                            dim_head=64,
                            avg_down=avg_down,
                            act_layer=act_layer,
                        )
                    )
                else:
                    blocks.append(
                        ResConv(
                            inplanes,
                            planes,
                            stride,
                            first_dilation=prev_dilation,
                            drop_path=DropPath(block_dpr) if block_dpr > 0.0 else None,
                            avg_down=avg_down,
                            act_layer=act_layer,
                            attn_layer=attn_layer,
                        )
                    )
                prev_dilation = dilation
                inplanes = planes * expansion
                net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


class BossNet(nn.Module):
    """
    Modified from ResNet class in timm.
    """

    def __init__(
        self,
        encoding=None,
        num_classes=1000,
        in_chans=3,
        cardinality=1,
        base_width=64,
        stem_width=64,
        stem_type="",
        output_stride=32,
        block_reduce_first=1,
        down_kernel_size=1,
        avg_down=False,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        aa_layer=None,
        attn_layer=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=0.0,
        global_pool="avg",
        zero_init_last_bn=True,
        block_args=None,
        last_stride=2,
    ):
        if encoding is None:
            encoding = [[1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1]]
        expansion = 4
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        super(BossNet, self).__init__()

        # Stem
        deep_stem = "deep" in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs_1 = stem_chs_2 = stem_width
            if "tiered" in stem_type:
                stem_chs_1 = 3 * (stem_width // 4)
                stem_chs_2 = stem_width if "narrow" in stem_type else 6 * (stem_width // 4)
            self.conv1 = nn.Sequential(
                *[
                    nn.Conv2d(in_chans, stem_chs_1, 3, stride=2, padding=1, bias=False),
                    norm_layer(stem_chs_1),
                    act_layer(inplace=True),
                    nn.Conv2d(stem_chs_1, stem_chs_2, 3, stride=1, padding=1, bias=False),
                    norm_layer(stem_chs_2),
                    act_layer(inplace=True),
                    nn.Conv2d(stem_chs_2, inplanes, 3, stride=1, padding=1, bias=False),
                ]
            )
        else:
            self.conv1 = nn.Conv2d(
                in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module="act1")]

        # Stem Pooling
        if aa_layer is not None:
            self.maxpool = nn.Sequential(
                *[
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                    aa_layer(channels=inplanes, stride=2),
                ]
            )
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]

        stage_modules, stage_feature_info = make_boss_blocks(
            encoding,
            channels,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            avg_down=avg_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            attn_layer=attn_layer,
            last_stride=last_stride,
            **block_args,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * expansion
        self.global_pool, self.fc = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool
        )

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, "zero_init_last_bn"):
                    m.zero_init_last_bn()

    def get_classifier(self):
        return self.fc

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(
            self.num_features, self.num_classes, pool_type=global_pool
        )

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


def bossnet_T0(pretrained=False, **kwargs):
    """Constructs a ResNet-26-D model.
    stem_width=32, stem_type='deep'
    """
    model = BossNet(
        encoding=[[1], [1], [1, 1, 1, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 0]],
        avg_down=True,
        act_layer=nn.SiLU,
        attn_layer="se",
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


def build_bossnas():
    return bossnet_T0(num_classes=10, stem_width=8)


def example_input_bossnas():
    import torch

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("BossNAS", "build_bossnas", "example_input_bossnas", 2021, "vendored"),
]
