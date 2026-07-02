# SOURCE: vendored from SLDGroup/MobileViG @ 087df4f4fc11d04a26af2d5eea746eeb538014b9
# https://github.com/SLDGroup/MobileViG/blob/main/models/mobilevig.py
#
# MobileViG (CVPRW 2023, arXiv:2307.00395) is a mobile hybrid CNN/GNN vision backbone
# that replaces the standard ViT self-attention stage with a Sparse Vision Graph
# Attention (SVGA) block -- max-relative graph convolution (Grapher/MRConv4d) over a
# 2D feature grid, following a local inverted-residual CNN stem. The classes below are
# the real MobileViG architecture, preserved verbatim from models/mobilevig.py; only
# the timm import path for DropPath is updated from the pre-1.0 deprecated
# `timm.models.layers.DropPath` location to the current `timm.layers.DropPath` (both
# resolve to the identical object; the old path only emits a deprecation warning in the
# installed timm 1.0.26). timm's `@register_model` decorator is dropped: it exists only
# to register the constructor with timm's global `timm.create_model()` name registry
# (a lookup convenience), not part of the model architecture -- it also requires the
# defining module to be registered in `sys.modules` under its real dotted name, which a
# standalone staging file loaded via `importlib.util.module_from_spec` is not. No
# architectural change.

import torch
from timm.layers import DropPath
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


class Stem(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.GELU):
        super(Stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.GELU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.stem(x)


class MLP(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0, mid_conv=False
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        if self.mid_conv:
            self.mid = nn.Conv2d(
                hidden_features,
                hidden_features,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=hidden_features,
            )
            self.mid_norm = nn.BatchNorm2d(hidden_features)

        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)

        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = self.drop(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
    ):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1),
                requires_grad=True,
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x


class MRConv4d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type

    K is the number of superpatches, therefore hops equals res // K.
    """

    def __init__(self, in_channels, out_channels, K=2):
        super(MRConv4d, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU(),
        )
        self.K = K

    def forward(self, x):
        B, C, H, W = x.shape

        x_j = x - x
        for i in range(self.K, H, self.K):
            x_c = x - torch.cat([x[:, :, -i:, :], x[:, :, :-i, :]], dim=2)
            x_j = torch.max(x_j, x_c)
        for i in range(self.K, W, self.K):
            x_r = x - torch.cat([x[:, :, :, -i:], x[:, :, :, :-i]], dim=3)
            x_j = torch.max(x_j, x_r)

        x = torch.cat([x, x_j], dim=1)
        return self.nn(x)


class Grapher(nn.Module):
    """
    Grapher module with graph convolution and fc layers
    """

    def __init__(self, in_channels, drop_path=0.0, K=2):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.K = K

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = MRConv4d(in_channels, in_channels * 2, K=self.K)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )  # out_channels back to 1x}
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = self.drop_path(x) + _tmp

        return x


class Downsample(nn.Module):
    """Convolution-based downsample"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features  # same as input
        hidden_features = hidden_features or in_features  # x4
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = nn.GELU()
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class MobileViG(torch.nn.Module):
    def __init__(
        self,
        local_blocks,
        local_channels,
        global_blocks,
        global_channels,
        dropout=0.0,
        drop_path=0.0,
        emb_dims=512,
        K=2,
        distillation=True,
        num_classes=1000,
    ):
        super(MobileViG, self).__init__()

        self.distillation = distillation

        n_blocks = sum(global_blocks) + sum(local_blocks)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path, n_blocks)
        ]  # stochastic depth decay rule
        dpr_idx = 0

        self.stem = Stem(input_dim=3, output_dim=local_channels[0])

        # local processing with inverted residuals
        self.local_backbone = nn.ModuleList([])
        for i in range(len(local_blocks)):
            if i > 0:
                self.local_backbone.append(Downsample(local_channels[i - 1], local_channels[i]))
            for _ in range(local_blocks[i]):
                self.local_backbone.append(
                    InvertedResidual(dim=local_channels[i], mlp_ratio=4, drop_path=dpr[dpr_idx])
                )
                dpr_idx += 1
        self.local_backbone.append(
            Downsample(local_channels[-1], global_channels[0])
        )  # transition from local to global

        # global processing with svga
        self.backbone = nn.ModuleList([])
        for i in range(len(global_blocks)):
            if i > 0:
                self.backbone.append(Downsample(global_channels[i - 1], global_channels[i]))
            for j in range(global_blocks[i]):
                self.backbone += [
                    nn.Sequential(
                        Grapher(global_channels[i], drop_path=dpr[dpr_idx], K=K),
                        FFN(global_channels[i], global_channels[i] * 4, drop_path=dpr[dpr_idx]),
                    )
                ]
                dpr_idx += 1

        self.prediction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(global_channels[-1], emb_dims, 1, bias=True),
            nn.BatchNorm2d(emb_dims),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Conv2d(emb_dims, num_classes, 1, bias=True)

        if self.distillation:
            self.dist_head = nn.Conv2d(emb_dims, num_classes, 1, bias=True)

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs)
        B, C, H, W = x.shape
        for i in range(len(self.local_backbone)):
            x = self.local_backbone[i](x)
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)

        x = self.prediction(x)

        if self.distillation:
            x = self.head(x).squeeze(-1).squeeze(-1), self.dist_head(x).squeeze(-1).squeeze(-1)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x).squeeze(-1).squeeze(-1)
        return x


def mobilevig_ti(pretrained=False, **kwargs):
    model = MobileViG(
        local_blocks=[2, 2, 6],
        local_channels=[42, 84, 168],
        global_blocks=[2],
        global_channels=[256],
        dropout=0.0,
        drop_path=0.1,
        emb_dims=512,
        K=2,
        distillation=True,
        num_classes=1000,
    )
    return model


def build_mobilevig_ti():
    return mobilevig_ti().eval()


def example_input_mobilevig_ti():
    return torch.randn(1, 3, 224, 224)


MENAGERIE_ENTRIES = [
    ("MobileViG-Ti", build_mobilevig_ti, example_input_mobilevig_ti, 2023, MENAGERIE_ZOO),
]
