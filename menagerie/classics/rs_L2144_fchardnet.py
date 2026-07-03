# SOURCE: vendored from PingoLH/FCHarDNet @ master
# https://raw.githubusercontent.com/PingoLH/FCHarDNet/master/ptsemseg/models/hardnet.py
#
# Chao, Kao, Ruan, Huang, Lin 2019 (ICCV) "HarDNet: A Low Memory Traffic Network" -- the
# FC-HarDNet-70 semantic segmentation network (ptsemseg/models/hardnet.py::hardnet, copied
# verbatim): a U-shaped encoder-decoder built from HarDBlock (Harmonic Dense Block, sparse
# power-of-2 skip links) stages with average-pool downsampling and bilinear
# TransitionUp-based upsampling. 76.0 mIoU on Cityscapes at 70fps.
# NOTE: the original file also defines HarDBlock_v2 and imports `CatConv2d` from the
# `CatConv2d` git submodule (a fused-CUDA-kernel Concat+Conv2d op used ONLY by the optional
# `hardnet.v2_transform()` inference-acceleration path, per the repo's own Aug-2021 README
# update -- "Added a new v2_transform() method to replace torch.cat + nn.Conv2d combinations
# with CatConv2d ... an all-in-1 fused cuda kernel"). The default `hardnet` forward path
# (this staging module) never constructs HarDBlock_v2 or calls v2_transform(); HarDBlock_v2
# and the CatConv2d import are omitted here since they are dead code for the architecture
# actually being traced (pure-torch HarDBlock only), not an architectural modification.
"""FC-HarDNet-70: Harmonic DenseNet encoder-decoder for semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=kernel // 2,
                bias=False,
            ),
        )
        self.add_module("norm", nn.BatchNorm2d(out_channels))
        self.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        return super().forward(x)


class BRLayer(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()

        self.add_module("norm", nn.BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(True))

    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2**i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(
        self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.grmul = grmul
        self.n_layers = n_layers
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0  # if upsample else in_channels
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            layers_.append(ConvLayer(inch, outch))
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

    def forward(self, x, skip, concat=True):
        is_v2 = type(skip) is list
        if is_v2:
            skip_x = skip[0]
        else:
            skip_x = skip
        out = F.interpolate(
            x,
            size=(skip_x.size(2), skip_x.size(3)),
            mode="bilinear",
            align_corners=True,
        )
        if concat:
            if is_v2:
                out = [out] + skip
            else:
                out = torch.cat([out, skip], 1)

        return out


class hardnet(nn.Module):
    def __init__(self, n_classes=19):
        super(hardnet, self).__init__()

        first_ch = [16, 24, 32, 48]
        ch_list = [64, 96, 160, 224, 320]
        grmul = 1.7
        gr = [10, 16, 18, 24, 32]
        n_layers = [4, 4, 8, 8, 8]

        blks = len(n_layers)
        self.shortcut_layers = []

        self.base = nn.ModuleList([])
        self.base.append(ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=2))
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=3))
        self.base.append(ConvLayer(first_ch[1], first_ch[2], kernel=3, stride=2))
        self.base.append(ConvLayer(first_ch[2], first_ch[3], kernel=3))

        skip_connection_channel_counts = []
        ch = first_ch[3]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.get_out_ch()
            skip_connection_channel_counts.append(ch)
            self.base.append(blk)
            if i < blks - 1:
                self.shortcut_layers.append(len(self.base) - 1)

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]

            if i < blks - 1:
                self.base.append(nn.AvgPool2d(kernel_size=2, stride=2))

        cur_channels_count = ch
        prev_block_channels = ch
        n_blocks = blks - 1
        self.n_blocks = n_blocks

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        self.conv1x1_up = nn.ModuleList([])

        for i in range(n_blocks - 1, -1, -1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]
            self.conv1x1_up.append(ConvLayer(cur_channels_count, cur_channels_count // 2, kernel=1))
            cur_channels_count = cur_channels_count // 2

            blk = HarDBlock(cur_channels_count, gr[i], grmul, n_layers[i])

            self.denseBlocksUp.append(blk)
            prev_block_channels = blk.get_out_ch()
            cur_channels_count = prev_block_channels

        self.finalConv = nn.Conv2d(
            in_channels=cur_channels_count,
            out_channels=n_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        skip_connections = []
        size_in = x.size()

        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in self.shortcut_layers:
                skip_connections.append(x)
        out = x

        for i in range(self.n_blocks):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip, True)
            out = self.conv1x1_up[i](out)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)

        out = F.interpolate(out, size=(size_in[2], size_in[3]), mode="bilinear", align_corners=True)
        return out


def build_fchardnet():
    return hardnet(n_classes=19)


def example_input_fchardnet():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 128, 128),)


MENAGERIE_ENTRIES = [
    ("FC-HarDNet-70", "build_fchardnet", "example_input_fchardnet", 2019, "vendored"),
]
