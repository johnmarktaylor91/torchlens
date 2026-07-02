# SOURCE: vendored from xxradon/IGCV3-pytorch @ master (IGCV3.py), a PyTorch re-implementation of
# IGCV3 ("Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks", Sun, Li, Liu,
# Wang, arXiv:1806.00178), whose official ARM/hellozting release (InterleavedGroupConvolutions) ships
# only MXNet symbol/params checkpoints + custom .cc/.cu ops, not runnable PyTorch. Imports/relative
# paths adjusted minimally for standalone staging; architecture (permutation-interleaved grouped
# inverted-residual MobileNetV2 variant) untouched.

import math

import torch.nn as nn


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class PermutationBlock(nn.Module):
    def __init__(self, groups):
        super(PermutationBlock, self).__init__()
        self.groups = groups

    def forward(self, input):
        n, c, h, w = input.size()
        G = self.groups
        output = input.view(n, G, c // G, h, w).permute(0, 2, 1, 3, 4).contiguous().view(n, c, h, w)
        return output


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(
                inp, inp * expand_ratio, kernel_size=1, stride=1, padding=0, groups=2, bias=False
            ),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # permutation
            PermutationBlock(groups=2),
            # dw
            nn.Conv2d(
                inp * expand_ratio,
                inp * expand_ratio,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=inp * expand_ratio,
                bias=False,
            ),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(
                inp * expand_ratio, oup, kernel_size=1, stride=1, padding=0, groups=2, bias=False
            ),
            nn.BatchNorm2d(oup),
            # permutation
            PermutationBlock(groups=int(round((oup / 2)))),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class IGCV3(nn.Module):
    def __init__(self, args):
        super(IGCV3, self).__init__()
        s1, s2 = 2, 2
        if args.downsampling == 16:
            s1, s2 = 2, 1
        elif args.downsampling == 8:
            s1, s2 = 1, 1

        # t: expand ratio, c: output channels, n: number of residual units, s: stride
        self.interverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 4, s2],
            [6, 32, 6, 2],
            [6, 64, 8, 2],
            [6, 96, 6, 1],
            [6, 160, 6, 2],
            [6, 320, 1, 1],
        ]

        assert args.img_height % 32 == 0
        input_channel = int(32 * args.width_multiplier)
        self.last_channel = (
            int(1280 * args.width_multiplier) if args.width_multiplier > 1.0 else 1280
        )
        self.features = [conv_bn(inp=3, oup=input_channel, stride=s1)]
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * args.width_multiplier)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(
            nn.AvgPool2d(
                kernel_size=(
                    args.img_height // args.downsampling,
                    args.img_width // args.downsampling,
                )
            )
        )
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, args.num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ============================================================================
# Menagerie staging entry points
# ============================================================================
#
# IGCV3's constructor takes an `args` namespace (img_height/img_width/downsampling/
# width_multiplier/num_classes), not bare hyperparameters -- matches the real repo's CLI-driven
# config object (options.py). A small local namespace stand-in keeps the constructor call
# identical to the real one without requiring the repo's argparse module.

MENAGERIE_ZOO = "vendored-pytorch"


class _IGCV3Args:
    def __init__(
        self, img_height=32, img_width=32, downsampling=32, width_multiplier=1.0, num_classes=10
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.downsampling = downsampling
        self.width_multiplier = width_multiplier
        self.num_classes = num_classes


def build_igcv3():
    """Tiny IGCV3 (interleaved group-conv MobileNetV2 variant) at reduced input resolution.

    Returned in eval() mode: at 32x32 input the deep stride-2 stack collapses to a 1x1 spatial
    map before the final AvgPool, and BatchNorm's training-mode per-channel variance requires
    >1 spatial position per channel -- a real constraint of this architecture at small inputs,
    not a staging bug. eval() uses running statistics instead, which is also how these
    classifiers are normally traced/deployed.
    """
    import torch

    torch.manual_seed(0)
    args = _IGCV3Args(
        img_height=32, img_width=32, downsampling=32, width_multiplier=0.5, num_classes=10
    )
    model = IGCV3(args)
    model.eval()
    return model


def example_input_igcv3():
    import torch

    torch.manual_seed(0)
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    ("IGCV3", build_igcv3, example_input_igcv3, 2018, "vendored-pytorch"),
]
