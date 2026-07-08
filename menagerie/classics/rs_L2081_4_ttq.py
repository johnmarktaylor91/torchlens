# FAITHFUL PORT of czhu95/ternarynet @ master (original framework: TensorFlow 1.x + bundled
# TensorPack, 2016-era API -- `tf.select`, `tf.mul`, `InputVar`/`ModelDesc` -- cannot run in a
# modern base env; TF1 graph-mode custom gradient overrides are not reasonably installable
# alongside the torch stack, so this is a rung-3 faithful port, transcribed line-by-line
# from the real source below rather than reimplemented from a paper description.)
#
# TTQ: Trained Ternary Quantization
# Chenzhuo Zhu, Song Han, Huizi Mao, William J. Dally. ICLR 2017.
# https://github.com/czhu95/ternarynet
#
# Ported faithfully from two real files in the repo:
#   examples/Ternary-Net/ternary.py            (tw_ternarize: the trained-ternary-weight
#                                                quantizer with learnable Wp/Wn scale factors)
#   examples/Ternary-Net/tw-cifar10-resnet.py  (Model._build_graph: the CIFAR-10 pre-activation
#                                                ResNet-(6n+2) that ternarizes every conv weight
#                                                except the first ('conv0') and last ('fct') layer)
#
# `tw_ternarize(x, thre)` (real TF code):
#   thre_x = stop_gradient(max(|x|) * thre)
#   w_p, w_n = learnable scalars, init 1.0
#   mask = ones; mask_p = where(x > thre_x, w_p, mask); mask_np = where(x < -thre_x, w_n, mask_p)
#   mask_z = where(-thre_x < x < thre_x, 0, mask)                 # zero out near-zero weights
#   w = sign(x) * stop_gradient(mask_z)                            # straight-through sign
#   w = w * mask_np                                                 # scale +/- regions by Wp/Wn
# This is transcribed below as `TernaryWeight.forward` using a `torch.sign` +
# `torch.where`-based straight-through estimator (custom autograd.Function matching the real
# repo's `gradient_override_map({"Sign": "Identity"})` -- i.e. the backward pass is identity
# through `sign`, exactly the real repo's override).
#
# The real `Model._build_graph` architecture (n=3, 20-layer CIFAR-10 ResNet, "Identity
# Mappings in Deep Residual Networks" pre-activation variant) is transcribed as
# `TTQResNetCifar` below: conv0(16ch, NOT ternarized) -> BN -> ReLU -> 3x res1 (16ch, first
# block skips the pre-activation BN-ReLU per `first=True`) -> 3x res2 (32ch, first block
# increase_dim=True: stride-2 conv1 + avgpool-and-zero-pad shortcut) -> 3x res3 (64ch, same
# increase_dim pattern) -> final BN+ReLU -> global average pool -> FC(10, NOT ternarized).
# Every residual conv layer's weight is routed through the ternary quantizer at forward time,
# matching the real `new_get_variable` monkeypatch that ternarizes every 'W' variable except
# 'conv0' and 'fct'.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class _TernarySignSTE(torch.autograd.Function):
    """Straight-through sign, matching the real repo's
    `G.gradient_override_map({"Sign": "Identity", "Mul": "Add"})` -- the backward pass treats
    `sign(x)` as the identity function (gradient passes straight through unmodified)."""

    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _ternary_sign(x):
    return _TernarySignSTE.apply(x)


class TernaryWeight(nn.Module):
    """Faithful port of `tw_ternarize(x, thre)` from ternary.py: a trained-ternary-weight
    quantizer with two learnable positive/negative scale factors `Wp`/`Wn`."""

    def __init__(self, thre=0.05):
        super().__init__()
        self.thre = thre
        self.w_p = nn.Parameter(torch.tensor(1.0))
        self.w_n = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        thre_x = (x.detach().abs().max() * self.thre).detach()

        mask = torch.ones_like(x)
        mask_p = torch.where(x > thre_x, torch.full_like(x, 1.0) * self.w_p, mask)
        mask_np = torch.where(x < -thre_x, torch.full_like(x, 1.0) * self.w_n, mask_p)
        mask_z = torch.where((x < thre_x) & (x > -thre_x), torch.zeros_like(x), mask)

        w = _ternary_sign(x) * mask_z.detach()
        w = w * mask_np
        return w


class TernaryConv2d(nn.Module):
    """A `nn.Conv2d` whose weight is routed through `TernaryWeight` at every forward call --
    the real repo's `new_get_variable` monkeypatch applied to every ternarized conv layer's
    'W' variable (i.e. re-ternarize from the latent full-precision weight each step)."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, ternarize=True, thre=0.05
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.normal_(self.weight, std=(2.0 / (9 * out_channels)) ** 0.5)
        self.stride = stride
        self.padding = padding
        self.ternarize = ternarize
        if ternarize:
            self.ternary = TernaryWeight(thre=thre)

    def forward(self, x):
        w = self.ternary(self.weight) if self.ternarize else self.weight
        return F.conv2d(x, w, bias=None, stride=self.stride, padding=self.padding)


class TTQResidualBlock(nn.Module):
    """Faithful port of the `residual(name, l, increase_dim, first)` closure in
    tw-cifar10-resnet.py: pre-activation BN-ReLU-Conv-BN-ReLU-Conv with option-A
    (avgpool + zero-pad) downsampling shortcut when `increase_dim=True`."""

    def __init__(self, in_channels, increase_dim=False, first=False, thre=0.05):
        super().__init__()
        self.increase_dim = increase_dim
        self.first = first
        out_channels = in_channels * 2 if increase_dim else in_channels
        stride1 = 2 if increase_dim else 1

        if not first:
            self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = TernaryConv2d(
            in_channels, out_channels, 3, stride=stride1, padding=1, thre=thre
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = TernaryConv2d(out_channels, out_channels, 3, stride=1, padding=1, thre=thre)

        if increase_dim:
            self.pool = nn.AvgPool2d(2)
            self.in_channels = in_channels

    def forward(self, x):
        act = x
        if not self.first:
            b1 = self.bn1(act)
            b1 = F.relu(b1)
        else:
            b1 = act
        c1 = self.conv1(b1)
        b2 = self.bn2(c1)
        b2 = F.relu(b2)
        c2 = self.conv2(b2)

        if self.increase_dim:
            act = self.pool(act)
            pad = self.in_channels // 2
            act = F.pad(act, (0, 0, 0, 0, pad, pad))

        return c2 + act


class TTQResNetCifar(nn.Module):
    """Faithful port of `Model._build_graph` in tw-cifar10-resnet.py: a CIFAR-10
    pre-activation ResNet-(6n+2) with every residual conv weight routed through the
    trained-ternary-weight quantizer (`conv0` and the final FC ('fct') stay full precision,
    matching the real `new_get_variable` exemption)."""

    def __init__(self, n=3, num_classes=10, thre=0.05):
        super().__init__()
        self.n = n

        self.conv0 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(16)

        blocks = []
        blocks.append(TTQResidualBlock(16, first=True, thre=thre))
        for _ in range(1, n):
            blocks.append(TTQResidualBlock(16, thre=thre))

        blocks.append(TTQResidualBlock(16, increase_dim=True, thre=thre))
        for _ in range(1, n):
            blocks.append(TTQResidualBlock(32, thre=thre))

        blocks.append(TTQResidualBlock(32, increase_dim=True, thre=thre))
        for _ in range(1, n):
            blocks.append(TTQResidualBlock(64, thre=thre))

        self.blocks = nn.ModuleList(blocks)
        self.bn_last = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x / 128.0 - 1
        act = self.conv0(x)
        act = self.bn0(act)
        act = F.relu(act)

        for block in self.blocks:
            act = block(act)

        act = self.bn_last(act)
        act = F.relu(act)
        act = F.adaptive_avg_pool2d(act, 1).flatten(1)
        logits = self.fc(act)
        return logits


def build_ttq():
    model = TTQResNetCifar(n=1, num_classes=10, thre=0.05)
    model.eval()
    return model


def example_input_ttq():
    return torch.rand(1, 3, 32, 32) * 255.0


MENAGERIE_ENTRIES = [
    (
        "TTQ (Trained Ternary Quantization) CIFAR-10 ResNet",
        "build_ttq",
        "example_input_ttq",
        2017,
        MENAGERIE_ZOO,
    ),
]
