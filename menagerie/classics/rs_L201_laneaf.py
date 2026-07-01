# SOURCE: vendored from sel118/LaneAF @ main
#
# https://github.com/sel118/LaneAF
# https://raw.githubusercontent.com/sel118/LaneAF/main/models/erfnet/erfnet.py
#
# LaneAF (Abualsaud et al. 2021, "LaneAF: Robust Multi-Lane Detection with
# Affinity Fields") predicts a lane heatmap plus horizontal/vertical
# "affinity fields" (haf/vaf) that are clustered into lane instances in
# post-processing. The official repo supports three interchangeable
# backbones (`dla34` via DCNv2, `erfnet`, `enet`); the DLA-34 path requires
# `models/dla/DCNv2` -- a custom-CUDA deformable-conv extension that is not
# an installed base lib and cannot be reasonably installed here, so it is
# skipped per the rung-2 non-base-dep rule. This vendors the `erfnet`
# backbone path instead: the actual `ERFNet` class (LaneAF's own variant of
# ERFNet with the paper's `heads` dict of prediction heads) from
# `models/erfnet/erfnet.py`, copied verbatim (only whitespace-preserving
# copy, no architecture changes). `models/__init__.py` was empty (no
# re-exports to preserve).
#
# The upstream `ERFNet.__init__` unconditionally does
# `torch.load('models/erfnet/erfnet_encoder_pretrained.pth.tar')` to warm-start
# the encoder from an ImageNet-pretrained ERFNet checkpoint that ships with the
# repo as a binary asset, then copies matching keys via `load_my_state_dict`.
# That call is dropped here (no checkpoint asset available in this env) so the
# encoder gets its normal random `nn.Module` init instead -- pure weight
# initialization, not an architecture change; `load_my_state_dict` is kept
# (unused by the build path here, preserved for fidelity) and the `Encoder`,
# `Decoder`, `DownsamplerBlock`, `non_bottleneck_1d`, `UpsamplerBlock` classes,
# and the LaneAF-specific per-head `Decoder()` + final 1x1 conv construction
# are otherwise identical to the upstream file.
#
# heads = {'hm': 1, 'vaf': 2, 'haf': 1} matches the real head config used in
# `infer_culane.py`/`infer_tusimple.py`/`infer_llamas.py` (heatmap + 2-channel
# vertical affinity field + 1-channel horizontal affinity field).

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(
            chann,
            chann,
            (3, 1),
            stride=1,
            padding=(1 * dilated, 0),
            bias=True,
            dilation=(dilated, 1),
        )

        self.conv1x3_2 = nn.Conv2d(
            chann,
            chann,
            (1, 3),
            stride=1,
            padding=(0, 1 * dilated),
            bias=True,
            dilation=(1, dilated),
        )

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder(nn.Module):  # modified decoder for LaneAF
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output


def load_my_state_dict(
    model, state_dict
):  # custom function to load model when not all dict elements
    own_state = model.state_dict()
    ckpt_name = []
    cnt = 0
    for name, param in state_dict.items():
        if name not in list(own_state.keys()) or "output_conv" in name:
            ckpt_name.append(name)
            continue
        own_state[name].copy_(param)
        cnt += 1
    print("#reused param: {}".format(cnt))
    return model


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# ERFNET for LaneAF
class ERFNet(nn.Module):
    def __init__(self, heads):
        super(ERFNet, self).__init__()
        self.encoder = Encoder(128)
        # NOTE (vendoring deviation): upstream loads an ImageNet-pretrained
        # ERFNet encoder checkpoint here via
        # `torch.load('models/erfnet/erfnet_encoder_pretrained.pth.tar')` +
        # `load_my_state_dict`. That binary checkpoint asset is not available
        # in this environment, so the encoder keeps its random init instead
        # (weight initialization only -- no architectural change).

        # new additions for LaneAF
        self.heads = heads
        final_kernel = 1
        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                Decoder(),
                nn.Conv2d(
                    64,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True,
                ),
            )
            if "hm" in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        out = self.encoder(x)
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(out)
        return [z]


def build_laneaf_erfnet():
    heads = {"hm": 1, "vaf": 2, "haf": 1}
    return ERFNet(heads=heads)


def example_input_laneaf_erfnet():
    # ERFNet encoder downsamples by 8x total (3 DownsamplerBlocks); use a
    # tiny CULane-aspect-ratio-ish input that stays divisible by 8.
    return torch.randn(1, 3, 64, 176)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("LaneAF", "build_laneaf_erfnet", "example_input_laneaf_erfnet", 2021, "vendored"),
]
