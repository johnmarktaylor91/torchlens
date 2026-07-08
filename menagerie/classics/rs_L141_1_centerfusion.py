# SOURCE: vendored from https://github.com/mrnabati/CenterFusion @ 3bac5c9b26daf958456a7bbbae5eb5a00a76d98a
"""CenterFusion: radar-camera fusion 3D detector (Nabati & Qi, WACV 2021).

Vendored classes: DLA / DLASeg / DLAUp / IDAUp / BaseModel (image backbone + neck +
heads, including the pointcloud-fusion secondary heads that are CenterFusion's actual
contribution over the base CenterNet/CenterTrack detector). Code is copied verbatim
from src/lib/model/networks/dla.py and src/lib/model/networks/base_model.py, with only
minimal glue changes:
  - `opt` is a lightweight plain-namespace stand-in for the repo's argparse `opts`
    object (same field names/values the repo uses at inference), not a rewrite of the
    network.
  - `dla_node='conv'` (a real, repo-supported `DLA_NODE` choice) is used instead of the
    default `'dcn'`, because the `'dcn'` path imports the repo's bespoke `DCNv2` CUDA
    extension (`from .DCNv2.dcn_v2 import DCN`), which is not installable in the base
    env. This is a legitimate existing configuration switch in the real code, not an
    architecture change.
  - `generate_pc_hm(...)` (frustum radar-to-heatmap association, in
    src/lib/utils/pointcloud.py) needs `nuscenes-devkit` + real calibration data; it is
    a data-preprocessing step, not part of the network module. We bypass it by feeding
    `example_input_centerfusion()`'s synthetic `pc_hm` tensor directly into
    `BaseModel.forward`, so every real nn.Module (including the fusion secondary heads
    `velocity`, `nuscenes_att`, `dep_sec`, `rot_sec`) still executes.
"""

import math
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"

BN_MOMENTUM = 0.1


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class BaseModel(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()
        self.opt = opt
        if opt is not None and opt.head_kernel != 3:
            print("Using head kernel:", opt.head_kernel)
            head_kernel = opt.head_kernel
        else:
            head_kernel = 3

        self.num_stacks = num_stacks
        self.heads = heads
        self.secondary_heads = opt.secondary_heads

        last_channels = {head: last_channel for head in heads}
        for head in self.secondary_heads:
            last_channels[head] = last_channel + len(opt.pc_feat_lvl)

        for head in self.heads:
            classes = self.heads[head]
            head_conv = head_convs[head]
            if len(head_conv) > 0:
                out = nn.Conv2d(
                    head_conv[-1], classes, kernel_size=1, stride=1, padding=0, bias=True
                )
                conv = nn.Conv2d(
                    last_channels[head],
                    head_conv[0],
                    kernel_size=head_kernel,
                    padding=head_kernel // 2,
                    bias=True,
                )
                convs = [conv]
                for k in range(1, len(head_conv)):
                    convs.append(
                        nn.Conv2d(head_conv[k - 1], head_conv[k], kernel_size=1, bias=True)
                    )
                if len(convs) == 1:
                    fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
                elif len(convs) == 2:
                    fc = nn.Sequential(
                        convs[0], nn.ReLU(inplace=True), convs[1], nn.ReLU(inplace=True), out
                    )
                elif len(convs) == 3:
                    fc = nn.Sequential(
                        convs[0],
                        nn.ReLU(inplace=True),
                        convs[1],
                        nn.ReLU(inplace=True),
                        convs[2],
                        nn.ReLU(inplace=True),
                        out,
                    )
                elif len(convs) == 4:
                    fc = nn.Sequential(
                        convs[0],
                        nn.ReLU(inplace=True),
                        convs[1],
                        nn.ReLU(inplace=True),
                        convs[2],
                        nn.ReLU(inplace=True),
                        convs[3],
                        nn.ReLU(inplace=True),
                        out,
                    )
                if "hm" in head:
                    fc[-1].bias.data.fill_(opt.prior_bias)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(
                    last_channels[head], classes, kernel_size=1, stride=1, padding=0, bias=True
                )
                if "hm" in head:
                    fc.bias.data.fill_(opt.prior_bias)
                else:
                    fill_fc_weights(fc)

            self.__setattr__(head, fc)

    def img2feats(self, x):
        raise NotImplementedError

    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
        raise NotImplementedError

    def forward(self, x, pc_hm=None, pc_dep=None, calib=None):
        # extract features from image
        feats = self.img2feats(x)
        out = []

        for s in range(self.num_stacks):
            z = {}

            # Run the first stage heads
            for head in self.heads:
                if head not in self.secondary_heads:
                    z[head] = self.__getattr__(head)(feats[s])

            if self.opt.pointcloud:
                # get pointcloud heatmap. In the real repo this comes from
                # utils.pointcloud.generate_pc_hm (frustum radar association, needs
                # nuscenes-devkit + calib); here pc_hm is supplied directly by the
                # caller as a synthetic tensor, matching the not-`self.training` /
                # `disable_frustum` code path's tensor contract.
                ind = self.opt.pc_feat_channels["pc_dep"]
                z["pc_hm"] = pc_hm[:, ind, :, :].unsqueeze(1)

                # Run the second stage heads
                sec_feats = [feats[s], pc_hm]
                sec_feats = torch.cat(sec_feats, 1)
                for head in self.secondary_heads:
                    z[head] = self.__getattr__(head)(sec_feats)

            out.append(z)

        return out


# ---------------------------------------------------------------------------
# DLA-34 backbone + DLAUp/IDAUp neck (src/lib/model/networks/dla.py)
# ---------------------------------------------------------------------------


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, bias=False, padding=(kernel_size - 1) // 2
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(
        self,
        levels,
        block,
        in_channels,
        out_channels,
        stride=1,
        level_root=False,
        root_dim=0,
        root_kernel_size=1,
        dilation=1,
        root_residual=False,
    ):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(
        self,
        levels,
        channels,
        num_classes=1000,
        block=BasicBlock,
        residual_root=False,
        linear_root=False,
        opt=None,
    ):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(
                opt.num_img_channels, channels[0], kernel_size=7, stride=1, padding=3, bias=False
            ),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root,
        )
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        if opt.pre_img:
            self.pre_img_layer = nn.Sequential(
                nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
                nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
        if opt.pre_hm:
            self.pre_hm_layer = nn.Sequential(
                nn.Conv2d(1, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
                nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes,
                        kernel_size=3,
                        stride=stride if i == 0 else 1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation,
                    ),
                    nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                ]
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x, pre_img=None, pre_hm=None):
        y = []
        x = self.base_layer(x)
        if pre_img is not None:
            x = x + self.pre_img_layer(pre_img)
        if pre_hm is not None:
            x = x + self.pre_hm_layer(pre_hm)
        for i in range(6):
            x = getattr(self, "level{}".format(i))(x)
            y.append(x)

        return y


def dla34(pretrained=False, **kwargs):  # DLA-34, no ImageNet download for tracing
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock, **kwargs)
    return model


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class Conv(nn.Module):
    def __init__(self, chi, cho):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(chi, cho, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class GlobalConv(nn.Module):
    def __init__(self, chi, cho, k=7, d=1):
        super(GlobalConv, self).__init__()
        gcl = nn.Sequential(
            nn.Conv2d(
                chi,
                cho,
                kernel_size=(k, 1),
                stride=1,
                bias=False,
                dilation=d,
                padding=(d * (k // 2), 0),
            ),
            nn.Conv2d(
                cho,
                cho,
                kernel_size=(1, k),
                stride=1,
                bias=False,
                dilation=d,
                padding=(0, d * (k // 2)),
            ),
        )
        gcr = nn.Sequential(
            nn.Conv2d(
                chi,
                cho,
                kernel_size=(1, k),
                stride=1,
                bias=False,
                dilation=d,
                padding=(0, d * (k // 2)),
            ),
            nn.Conv2d(
                cho,
                cho,
                kernel_size=(k, 1),
                stride=1,
                bias=False,
                dilation=d,
                padding=(d * (k // 2), 0),
            ),
        )
        fill_fc_weights(gcl)
        fill_fc_weights(gcr)
        self.gcl = gcl
        self.gcr = gcr
        self.act = nn.Sequential(nn.BatchNorm2d(cho, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.gcl(x) + self.gcr(x)
        x = self.act(x)
        return x


class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f, node_type=(Conv, Conv)):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = node_type[0](c, o)
            node = node_type[1](o, o)

            up = nn.ConvTranspose2d(
                o, o, f * 2, stride=f, padding=f // 2, output_padding=0, groups=o, bias=False
            )
            fill_up_weights(up)

            setattr(self, "proj_" + str(i), proj)
            setattr(self, "up_" + str(i), up)
            setattr(self, "node_" + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, "up_" + str(i - startp))
            project = getattr(self, "proj_" + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, "node_" + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None, node_type=Conv):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self,
                "ida_{}".format(i),
                IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j], node_type=node_type),
            )
            scales[j + 1 :] = scales[j]
            in_channels[j + 1 :] = [channels[j] for _ in channels[j + 1 :]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, "ida_{}".format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


DLA_NODE = {
    "gcn": (Conv, GlobalConv),
    "conv": (Conv, Conv),
}


class DLASeg(BaseModel):
    def __init__(self, num_layers, heads, head_convs, opt):
        super(DLASeg, self).__init__(heads, head_convs, 1, 64 if num_layers == 34 else 128, opt=opt)
        down_ratio = 4
        self.opt = opt
        self.node_type = DLA_NODE[opt.dla_node]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = 5
        self.base = dla34(pretrained=False, opt=opt)

        channels = self.base.channels
        scales = [2**i for i in range(len(channels[self.first_level :]))]
        self.dla_up = DLAUp(
            self.first_level, channels[self.first_level :], scales, node_type=self.node_type
        )
        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel,
            channels[self.first_level : self.last_level],
            [2**i for i in range(self.last_level - self.first_level)],
            node_type=self.node_type,
        )

    def img2feats(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]]


# ---------------------------------------------------------------------------
# Menagerie build/example harness
# ---------------------------------------------------------------------------


def _make_opt():
    """Lightweight stand-in for the repo's argparse-based `opts` object, populated
    with the same field values CenterFusion's nuScenes config uses at inference, minus
    dataset paths / training-only flags the traced forward pass never reads."""
    pc_feat_lvl = ["pc_dep", "pc_vx", "pc_vz"]
    return SimpleNamespace(
        head_kernel=3,
        pointcloud=True,
        secondary_heads=["velocity", "nuscenes_att", "dep_sec", "rot_sec"],
        pc_feat_lvl=pc_feat_lvl,
        pc_feat_channels={feat: i for i, feat in enumerate(pc_feat_lvl)},
        prior_bias=-4.6,
        num_img_channels=3,
        pre_img=False,
        pre_hm=False,
        dla_node="conv",  # avoid the DCNv2 CUDA extension; a real repo-supported option
    )


def build_centerfusion():
    opt = _make_opt()
    heads = {
        "hm": 10,
        "reg": 2,
        "wh": 2,
        "dep": 1,
        "rot": 8,
        "dim": 3,
        "amodel_offset": 2,
        "velocity": 3,
        "nuscenes_att": 8,
        "dep_sec": 1,
        "rot_sec": 8,
    }
    head_convs = {h: [64] for h in heads}
    model = DLASeg(34, heads=heads, head_convs=head_convs, opt=opt)
    model.eval()
    return model


def example_input_centerfusion():
    opt = _make_opt()
    x = torch.randn(1, 3, 128, 224)
    pc_hm = torch.randn(1, len(opt.pc_feat_lvl), 32, 56)
    return (x, pc_hm)


MENAGERIE_ENTRIES = [
    ("CenterFusion", build_centerfusion, example_input_centerfusion, 2021, "vendored-pytorch"),
]
