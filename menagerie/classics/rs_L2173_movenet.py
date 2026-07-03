# SOURCE: vendored from fire717/movenet.pytorch @ f7391969f6bbc4590922558bc129a83bad444e53
# https://raw.githubusercontent.com/fire717/movenet.pytorch/master/lib/models/movenet_mobilenetv2.py
#
# fire717 (2020) PyTorch port of Google's MoveNet (Lightning/Thunder) real-time single-person
# pose estimator. The official Google release is TFLite/TF Hub only; this repo is a full
# PyTorch training+inference re-implementation of the published architecture: a MobileNetV2
# backbone (`Backbone`, with InvertedResidual expand/dw/pw-linear blocks) feeding a top-down
# feature-pyramid-style decoder (per-scale conv1x1 lateral connections + learned bilinear
# `upsample` blocks, `f4 += f3` / `f4 += f2` / `f4 += f1` skip fusions) and a 4-head
# `Header` (person keypoint heatmaps, person-center heatmap, keypoint regression field,
# per-keypoint 2D offset field) -- the same heatmap+regression+offset decomposition described
# in Google's MoveNet blog post. All classes/functions below (conv_3x3_act/conv_1x1_act/
# conv_1x1_act2/dw_conv/dw_conv2/dw_conv3/upsample/IRBlock/InvertedResidual/Backbone/Header/
# MoveNet) are copied verbatim from the real repo's movenet_mobilenetv2.py; only the
# `if __name__ == "__main__"` ONNX-export smoke-test block (torchsummary/torch.onnx.export,
# not part of the architecture) and a few commented-out dead-code blocks left by the original
# author (`MulReshapeArgMax`, the `mode='all'` post-processing draft in `Header.forward`) are
# dropped since they are non-functional in the original file too.
"""MoveNet (PyTorch port): MobileNetV2 backbone + FPN-style decoder + 4-head
(heatmap/center/regression/offset) pose estimation, per Google's MoveNet architecture
(fire717 2020 PyTorch re-implementation; official release is TFLite/TF Hub only)."""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


def conv_3x3_act(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True)
    )


def conv_1x1_act(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True)
    )


def conv_1x1_act2(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True)
    )


def dw_conv(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


def dw_conv2(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def dw_conv3(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


def upsample(inp, oup, scale=2):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, 1, 1, groups=inp),
        nn.ReLU(inplace=True),
        conv_1x1_act2(inp, oup),
        nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=False),
    )


def IRBlock(oup, hidden_dim):
    return nn.Sequential(
        # pw
        nn.Conv2d(oup, hidden_dim, 1, 1, 0, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(inplace=False),
        # dw
        nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(inplace=False),
        # pw-linear
        nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, n):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.n = n

        self.conv1 = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

        self.conv2 = torch.nn.ModuleList()
        for i in range(n):
            self.conv2.append(IRBlock(oup, hidden_dim))

    def forward(self, x):
        x = self.conv1(x)

        for i in range(self.n):
            x = x + self.conv2[i](x)

        return x


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        # mobilenet v2

        input_channel = 32

        self.features1 = nn.Sequential(
            *[
                conv_3x3_act(3, input_channel, 2),
                dw_conv(input_channel, 16, 1),
                InvertedResidual(16, 24, 2, 6, 1),
            ]
        )

        self.features2 = InvertedResidual(24, 32, 2, 6, 2)
        self.features3 = InvertedResidual(32, 64, 2, 6, 3)

        self.features4 = nn.Sequential(
            *[
                InvertedResidual(64, 96, 1, 6, 2),
                InvertedResidual(96, 160, 2, 6, 2),
                InvertedResidual(160, 320, 1, 6, 0),
                conv_1x1_act(320, 1280),
                nn.Conv2d(1280, 64, 1, 1, 0, bias=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ]
        )

        self.upsample2 = upsample(64, 32)
        self.upsample1 = upsample(32, 24)

        self.conv3 = nn.Conv2d(64, 64, 1, 1, 0)
        self.conv2 = nn.Conv2d(32, 32, 1, 1, 0)
        self.conv1 = nn.Conv2d(24, 24, 1, 1, 0)

        self.conv4 = dw_conv3(24, 24, 1)

    def forward(self, x):
        x = x / 127.5 - 1

        f1 = self.features1(x)

        f2 = self.features2(f1)

        f3 = self.features3(f2)

        f4 = self.features4(f3)
        f3 = self.conv3(f3)
        f4 += f3

        f4 = self.upsample2(f4)
        f2 = self.conv2(f2)
        f4 += f2

        f4 = self.upsample1(f4)
        f1 = self.conv1(f1)
        f4 += f1

        f4 = self.conv4(f4)

        return f4


class Header(nn.Module):
    def __init__(self, num_classes, mode="train"):
        super(Header, self).__init__()

        self.mode = mode

        # heatmaps, centers, regs, offsets
        # Person keypoint heatmap
        self.header_heatmaps = nn.Sequential(
            *[dw_conv3(24, 96), nn.Conv2d(96, num_classes, 1, 1, 0, bias=True), nn.Sigmoid()]
        )

        # Person center heatmap
        self.header_centers = nn.Sequential(
            *[
                dw_conv3(24, 96),
                nn.Conv2d(96, 1, 1, 1, 0, bias=True),
                nn.Sigmoid(),
            ]
        )

        # Keypoint regression field:
        self.header_regs = nn.Sequential(
            *[
                dw_conv3(24, 96),
                nn.Conv2d(96, num_classes * 2, 1, 1, 0, bias=True),
            ]
        )

        # 2D per-keypoint offset field
        self.header_offsets = nn.Sequential(
            *[
                dw_conv3(24, 96),
                nn.Conv2d(96, num_classes * 2, 1, 1, 0, bias=True),
            ]
        )

    def argmax2loc(self, x, h=48, w=48):
        # n,1
        y0 = torch.div(x, w).long()
        x0 = torch.sub(x, y0 * w).long()
        return x0, y0

    def forward(self, x):
        res = []
        if self.mode == "train":
            h1 = self.header_heatmaps(x)
            h2 = self.header_centers(x)
            h3 = self.header_regs(x)
            h4 = self.header_offsets(x)
            res = [h1, h2, h3, h4]

        elif self.mode == "test":
            pass

        elif self.mode == "all":
            # NOTE: the real repo's mode='all' branch is entirely commented-out dead code
            # in the original file (an abandoned post-processing draft: argmax-based
            # center localization + per-sample keypoint gather). It never executes --
            # the branch body is just `pass` in the shipped source. Omitted verbatim here.
            pass
        else:
            print("[ERROR] wrong mode.")

        return res


class MoveNet(nn.Module):
    def __init__(self, num_classes=17, width_mult=1.0, mode="train"):
        super(MoveNet, self).__init__()

        self.backbone = Backbone()

        self.header = Header(num_classes, mode)

        self._initialize_weights()

    def forward(self, x):
        x = self.backbone(x)  # n,24,48,48

        x = self.header(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_movenet():
    model = MoveNet(num_classes=17, mode="train")
    model.eval()
    return model


def example_input_movenet():
    # Fixed 192x192 input (config.py's "img_size": 192); the backbone's skip-connection
    # additions (f4 += f3/f2/f1) require the real MobileNetV2-stride downsample ratios, so
    # this is the real deployment resolution, not an arbitrarily shrunk one.
    torch.manual_seed(0)
    return (torch.randn(1, 3, 192, 192),)


MENAGERIE_ENTRIES = [
    ("MoveNet", "build_movenet", "example_input_movenet", 2020, "vendored"),
]
