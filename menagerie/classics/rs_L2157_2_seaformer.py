# SOURCE: vendored from fudan-zvg/SeaFormer @ 9718d05b6e70d0561dda7c052b8457feed74d4d9
# (seaformer-cls/seaformer.py -- the ImageNet classification variant: MobileNetV2-style
# StackedMV2Block stages + squeeze-axial "Sea Attention" transformer stages, ICLR 2023 /
# IJCV 2025 SeaFormer / SeaFormer++)
#
# The real file `import`s `mmcv.cnn.ConvModule` and `mmcv.cnn.build_norm_layer`. `ConvModule`
# is imported but never actually called anywhere in this file (dead import in the upstream
# repo) -- dropped here. `build_norm_layer(dict(type='BN'/'BN2d', requires_grad=True), c)[1]`
# is used in exactly one place (`Conv2d_BN.__init__`) and is semantically just a config-driven
# factory dispatch: verified against mmcv's real `build_norm_layer` source
# (open-mmlab/mmcv main, mmcv/cnn/bricks/norm.py) that `'BN'` and `'BN2d'` both register to
# `nn.BatchNorm2d`, `eps` defaults to `1e-5` (already `nn.BatchNorm2d`'s own default), and
# `requires_grad` defaults to `True` (already every `nn.Parameter`'s default) -- so
# `build_norm_layer(dict(type='BN', requires_grad=True), c)[1] == nn.BatchNorm2d(c)` exactly
# for every call site in this file. Inlined as `nn.BatchNorm2d(c)` directly; no architecture
# changed, only a config-dispatch utility call resolved to its concrete, always-selected class.
# All classes below (Conv2d_BN, Mlp, InvertedResidual, StackedMV2Block,
# SqueezeAxialPositionalEmbedding, Sea_Attention, Block, BasicLayer, h_sigmoid, SeaFormer,
# and the SeaFormer_T/S/B/L constructors) are transcribed verbatim from the real repo file.

import math

import torch
from torch import nn
import torch.nn.functional as F


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class Conv2d_BN(nn.Sequential):
    def __init__(
        self,
        a,
        b,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        bias=False,
        norm_cfg=dict(type="BN", requires_grad=True),
    ):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        # self.bias = bias
        self.add_module("c", nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=bias))
        # build_norm_layer(norm_cfg, b)[1] resolved: 'BN'/'BN2d' -> nn.BatchNorm2d(b) (see header note)
        bn = nn.BatchNorm2d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU,
        drop=0.0,
        norm_cfg=dict(type="BN", requires_grad=True),
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features
        )
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
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
    def __init__(
        self,
        inp: int,
        oup: int,
        ks: int,
        stride: int,
        expand_ratio: int,
        activations=None,
        norm_cfg=dict(type="BN", requires_grad=True),
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2d_BN(inp, hidden_dim, ks=1, norm_cfg=norm_cfg))
            layers.append(activations())
        layers.extend(
            [
                # dw
                Conv2d_BN(
                    hidden_dim,
                    hidden_dim,
                    ks=ks,
                    stride=stride,
                    pad=ks // 2,
                    groups=hidden_dim,
                    norm_cfg=norm_cfg,
                ),
                activations(),
                # pw-linear
                Conv2d_BN(hidden_dim, oup, ks=1, norm_cfg=norm_cfg),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class StackedMV2Block(nn.Module):
    def __init__(
        self,
        cfgs,
        stem,
        inp_channel=16,
        activation=nn.ReLU,
        norm_cfg=dict(type="BN", requires_grad=True),
        width_mult=1.0,
    ):
        super().__init__()
        self.stem = stem
        if stem:
            self.stem_block = nn.Sequential(
                Conv2d_BN(3, inp_channel, 3, 2, 1, norm_cfg=norm_cfg), activation()
            )
        self.cfgs = cfgs

        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = t * inp_channel
            exp_size = _make_divisible(exp_size * width_mult, 8)
            layer_name = "layer{}".format(i + 1)
            layer = InvertedResidual(
                inp_channel,
                output_channel,
                ks=k,
                stride=s,
                expand_ratio=t,
                norm_cfg=norm_cfg,
                activations=activation,
            )
            self.add_module(layer_name, layer)
            inp_channel = output_channel
            self.layers.append(layer_name)

    def forward(self, x):
        if self.stem:
            x = self.stem_block(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
        return x


class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]))

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode="linear", align_corners=False)

        return x


class Sea_Attention(torch.nn.Module):
    def __init__(
        self,
        dim,
        key_dim,
        num_heads,
        attn_ratio=2,
        activation=None,
        norm_cfg=dict(type="BN", requires_grad=True),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(
            activation(), Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg)
        )
        self.proj_encode_row = torch.nn.Sequential(
            activation(), Conv2d_BN(self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg)
        )
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.proj_encode_column = torch.nn.Sequential(
            activation(), Conv2d_BN(self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg)
        )
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.dwconv = Conv2d_BN(
            self.dh + 2 * self.nh_kd,
            2 * self.nh_kd + self.dh,
            ks=3,
            stride=1,
            pad=1,
            dilation=1,
            groups=2 * self.nh_kd + self.dh,
            norm_cfg=norm_cfg,
        )
        self.act = activation()
        self.pwconv = Conv2d_BN(2 * self.nh_kd + self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # detail enhance
        qkv = torch.cat([q, k, v], dim=1)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)

        # squeeze axial attention
        ## squeeze row
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        ## squeeze column
        qcolumn = (
            self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        )
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)
        xx = self.proj(xx)

        xx = self.sigmoid(xx) * qkv
        return xx


class Block(nn.Module):
    def __init__(
        self,
        dim,
        key_dim,
        num_heads,
        mlp_ratio=4.0,
        attn_ratio=2.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.ReLU,
        norm_cfg=dict(type="BN2d", requires_grad=True),
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Sea_Attention(
            dim,
            key_dim=key_dim,
            num_heads=num_heads,
            attn_ratio=attn_ratio,
            activation=act_layer,
            norm_cfg=norm_cfg,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            norm_cfg=norm_cfg,
        )

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class BasicLayer(nn.Module):
    def __init__(
        self,
        block_num,
        embedding_dim,
        key_dim,
        num_heads,
        mlp_ratio=4.0,
        attn_ratio=2.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_cfg=dict(type="BN2d", requires_grad=True),
        act_layer=None,
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
                    norm_cfg=norm_cfg,
                    act_layer=act_layer,
                )
            )

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class SeaFormer(nn.Module):
    def __init__(
        self,
        cfgs,
        channels,
        emb_dims,
        key_dims,
        depths=[2, 2],
        num_heads=4,
        attn_ratios=2,
        mlp_ratios=[2, 4],
        drop_path_rate=0.0,
        norm_cfg=dict(type="BN", requires_grad=True),
        act_layer=nn.ReLU6,
        init_cfg=None,
        num_classes=1000,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.depths = depths
        self.cfgs = cfgs
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        if self.init_cfg is not None:
            self.pretrained = self.init_cfg["checkpoint"]

        for i in range(len(cfgs)):
            smb = StackedMV2Block(
                cfgs=cfgs[i],
                stem=True if i == 0 else False,
                inp_channel=channels[i],
                norm_cfg=norm_cfg,
            )
            setattr(self, f"smb{i + 1}", smb)

        for i in range(len(depths)):
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depths[i])
            ]  # stochastic depth decay rule
            trans = BasicLayer(
                block_num=depths[i],
                embedding_dim=emb_dims[i],
                key_dim=key_dims[i],
                num_heads=num_heads,
                mlp_ratio=mlp_ratios[i],
                attn_ratio=attn_ratios,
                drop=0,
                attn_drop=0,
                drop_path=dpr,
                norm_cfg=norm_cfg,
                act_layer=act_layer,
            )
            setattr(self, f"trans{i + 1}", trans)

        self.linear = nn.Linear(channels[-1], 1000)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        num_smb_stage = len(self.cfgs)
        num_trans_stage = len(self.depths)
        for i in range(num_smb_stage):
            smb = getattr(self, f"smb{i + 1}")
            x = smb(x)
            if num_trans_stage + i >= num_smb_stage:
                trans = getattr(self, f"trans{i + num_trans_stage - num_smb_stage + 1}")
                x = trans(x)

        out = self.avgpool(x).view(-1, x.shape[1])
        out = self.linear(out)
        return out


def SeaFormer_T(pretrained=False, **kwargs):
    model_cfgs = dict(
        cfg1=[
            # k,  t,  c, s
            [3, 1, 16, 1],
            [3, 4, 16, 2],
            [3, 3, 16, 1],
        ],
        cfg2=[[5, 3, 32, 2], [5, 3, 32, 1]],
        cfg3=[[3, 3, 64, 2], [3, 3, 64, 1]],
        cfg4=[[5, 3, 128, 2]],
        cfg5=[[3, 6, 160, 2]],
        channels=[16, 16, 32, 64, 128, 160],
        num_heads=4,
        depths=[2, 2],
        emb_dims=[128, 160],
        key_dims=[16, 24],
        drop_path_rate=0.1,
        attn_ratios=2,
        mlp_ratios=[2, 4],
    )
    return SeaFormer(
        cfgs=[
            model_cfgs["cfg1"],
            model_cfgs["cfg2"],
            model_cfgs["cfg3"],
            model_cfgs["cfg4"],
            model_cfgs["cfg5"],
        ],
        channels=model_cfgs["channels"],
        emb_dims=model_cfgs["emb_dims"],
        key_dims=model_cfgs["key_dims"],
        depths=model_cfgs["depths"],
        attn_ratios=model_cfgs["attn_ratios"],
        mlp_ratios=model_cfgs["mlp_ratios"],
        num_heads=model_cfgs["num_heads"],
        drop_path_rate=model_cfgs["drop_path_rate"],
    )


def SeaFormer_S(pretrained=False, **kwargs):
    model_cfgs = dict(
        cfg1=[
            # k,  t,  c, s
            [3, 1, 16, 1],
            [3, 4, 24, 2],
            [3, 3, 24, 1],
        ],
        cfg2=[[5, 3, 48, 2], [5, 3, 48, 1]],
        cfg3=[[3, 3, 96, 2], [3, 3, 96, 1]],
        cfg4=[[5, 4, 160, 2]],
        cfg5=[[3, 6, 192, 2]],
        channels=[16, 24, 48, 96, 160, 192],
        num_heads=6,
        depths=[3, 3],
        key_dims=[16, 24],
        emb_dims=[160, 192],
        drop_path_rate=0.1,
        attn_ratios=2,
        mlp_ratios=[2, 4],
    )
    return SeaFormer(
        cfgs=[
            model_cfgs["cfg1"],
            model_cfgs["cfg2"],
            model_cfgs["cfg3"],
            model_cfgs["cfg4"],
            model_cfgs["cfg5"],
        ],
        channels=model_cfgs["channels"],
        emb_dims=model_cfgs["emb_dims"],
        key_dims=model_cfgs["key_dims"],
        depths=model_cfgs["depths"],
        attn_ratios=model_cfgs["attn_ratios"],
        mlp_ratios=model_cfgs["mlp_ratios"],
        num_heads=model_cfgs["num_heads"],
        drop_path_rate=model_cfgs["drop_path_rate"],
    )


def SeaFormer_B(pretrained=False, **kwargs):
    model_cfgs = dict(
        cfg1=[
            # k,  t,  c, s
            [3, 1, 16, 1],
            [3, 4, 32, 2],
            [3, 3, 32, 1],
        ],
        cfg2=[[5, 3, 64, 2], [5, 3, 64, 1]],
        cfg3=[[3, 3, 128, 2], [3, 3, 128, 1]],
        cfg4=[[5, 4, 192, 2]],
        cfg5=[[3, 6, 256, 2]],
        channels=[16, 32, 64, 128, 192, 256],
        num_heads=8,
        depths=[4, 4],
        key_dims=[16, 24],
        emb_dims=[192, 256],
        drop_path_rate=0.1,
        attn_ratios=2,
        mlp_ratios=[2, 4],
    )
    return SeaFormer(
        cfgs=[
            model_cfgs["cfg1"],
            model_cfgs["cfg2"],
            model_cfgs["cfg3"],
            model_cfgs["cfg4"],
            model_cfgs["cfg5"],
        ],
        channels=model_cfgs["channels"],
        emb_dims=model_cfgs["emb_dims"],
        key_dims=model_cfgs["key_dims"],
        depths=model_cfgs["depths"],
        attn_ratios=model_cfgs["attn_ratios"],
        mlp_ratios=model_cfgs["mlp_ratios"],
        num_heads=model_cfgs["num_heads"],
        drop_path_rate=model_cfgs["drop_path_rate"],
    )


def SeaFormer_L(pretrained=False, **kwargs):
    model_cfgs = dict(
        cfg1=[
            # k,  t,  c, s
            [3, 3, 32, 1],
            [3, 4, 64, 2],
            [3, 4, 64, 1],
        ],
        cfg2=[[5, 4, 128, 2], [5, 4, 128, 1]],
        cfg3=[[3, 4, 192, 2], [3, 4, 192, 1]],
        cfg4=[[5, 4, 256, 2]],
        cfg5=[[3, 6, 320, 2]],
        channels=[32, 64, 128, 192, 256, 320],
        num_heads=8,
        depths=[3, 3, 3],
        key_dims=[16, 20, 24],
        emb_dims=[192, 256, 320],
        drop_path_rate=0.1,
        attn_ratios=2,
        mlp_ratios=[2, 4, 6],
    )
    return SeaFormer(
        cfgs=[
            model_cfgs["cfg1"],
            model_cfgs["cfg2"],
            model_cfgs["cfg3"],
            model_cfgs["cfg4"],
            model_cfgs["cfg5"],
        ],
        channels=model_cfgs["channels"],
        emb_dims=model_cfgs["emb_dims"],
        key_dims=model_cfgs["key_dims"],
        depths=model_cfgs["depths"],
        attn_ratios=model_cfgs["attn_ratios"],
        mlp_ratios=model_cfgs["mlp_ratios"],
        num_heads=model_cfgs["num_heads"],
        drop_path_rate=model_cfgs["drop_path_rate"],
    )


def build_seaformer_t():
    return SeaFormer_T()


def example_input_seaformer_t():
    return (torch.randn(1, 3, 224, 224),)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "SeaFormer_T (squeeze-axial-attention mobile transformer)",
        "build_seaformer_t",
        "example_input_seaformer_t",
        2023,
        MENAGERIE_ZOO,
    ),
]
