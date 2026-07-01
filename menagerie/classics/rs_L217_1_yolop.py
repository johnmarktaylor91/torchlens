# SOURCE: vendored from hustvl/YOLOP @ main
#
# https://github.com/hustvl/YOLOP
# https://raw.githubusercontent.com/hustvl/YOLOP/main/lib/models/YOLOP.py
# https://raw.githubusercontent.com/hustvl/YOLOP/main/lib/models/common.py
# https://raw.githubusercontent.com/hustvl/YOLOP/main/lib/utils/utils.py (initialize_weights only)
# https://raw.githubusercontent.com/hustvl/YOLOP/main/lib/utils/autoanchor.py (check_anchor_order only)
#
# YOLOP ("You Only Look Once for Panoptic driving Perception", Wang et al.
# 2022, MIR). This vendors the real `Conv`/`Bottleneck`/`BottleneckCSP`/`SPP`/
# `Focus`/`Concat`/`Detect`/`SharpenConv`/`Hardswish` building blocks from
# `lib/models/common.py`, the real `MCnet` graph-executor class + the real
# `YOLOP` block-config list from `lib/models/YOLOP.py`, and the real
# `initialize_weights` / `check_anchor_order` helper functions from
# `lib/utils/utils.py` / `lib/utils/autoanchor.py` -- copied verbatim (only
# whitespace-preserving copy, no architecture changes).
#
# Dropped from the vendor: the commented-out `MCnet_SPP` / `MCnet_0` /
# `MCnet_share` / `MCnet_no_share` / `MCnet_feedback` / `MCnet_Da_feedback1` /
# `MCnet_Da_feedback2` / `MCnet_share1` block-config lists in the real
# `YOLOP.py` (they are historical variants inside a triple-quoted docstring
# in the original file and are never referenced by `get_net`, which always
# builds `MCnet(YOLOP)`); `DepthSeperabelConv2d` from `common.py` (imported
# by nothing `YOLOP.py` actually uses -- `lib/models/YOLOP.py` imports only
# `Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv`
# from `common.py`, so `DepthSeperabelConv2d`'s undefined-at-module-scope
# `BN_MOMENTUM` reference is never hit); the `lib.core.evaluate.SegmentationMetric`
# and `lib.utils.utils.time_synchronized` imports in the real `YOLOP.py`,
# which are only used in the file's `if __name__ == "__main__":` demo block,
# not by `MCnet`/`get_net`; `lib.utils.utils.py`'s other helpers (logging,
# optimizer/checkpoint plumbing, `is_parallel`, `xyxy2xywh`,
# `torch_distributed_zero_first`) and `lib.utils.autoanchor.py`'s other
# helpers (`run_anchor`, `kmean_anchors`, which pull in `scipy`/`yaml`/
# `tqdm`/a dataset), none of which the forward-only `MCnet.__init__`/
# `MCnet.forward` graph needs -- only `initialize_weights` and
# `check_anchor_order` are called from `MCnet.__init__`.
#
# `get_net()` in the real repo takes a `cfg` positional arg it never uses
# (`m_block_cfg = YOLOP` is hardcoded, ignoring the passed `cfg`); preserved
# here as `build_yolop()` with no arg needed, matching the real function's
# actual (ignoring-`cfg`) behavior.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# lib/models/common.py
# ---------------------------------------------------------------------------


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Hardswish(nn.Module):  # export-friendly version of nn.Hardswish()
    @staticmethod
    def forward(x):
        # return x * F.hardsigmoid(x)  # for torchscript and CoreML
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0  # for torchscript, CoreML and ONNX


class Conv(nn.Module):
    # Standard convolution
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        try:
            self.act = Hardswish() if act else nn.Identity()
        except:  # noqa: E722 (verbatim from upstream)
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self, c1, c2, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    # slice concat conv
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(
            torch.cat(
                [x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1
            )
        )


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class SharpenConv(nn.Module):
    # SharpenConv convolution
    def __init__(
        self, c1, c2, k=3, s=1, p=None, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(SharpenConv, self).__init__()
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype="float32")
        kenel_weight = np.vstack([sobel_kernel] * c2 * c1).reshape(c2, c1, 3, 3)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.conv.weight.data = torch.from_numpy(kenel_weight)
        self.conv.weight.requires_grad = False
        self.bn = nn.BatchNorm2d(c2)
        try:
            self.act = Hardswish() if act else nn.Identity()
        except:  # noqa: E722 (verbatim from upstream)
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Detect(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, nc=13, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor 85
        self.nl = len(anchors)  # number of detection layers 3
        self.na = len(anchors[0]) // 2  # number of anchors 3
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer(
            "anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2)
        )  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,w,w) to x(bs,3,w,w,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny * nx)
                .permute(0, 1, 3, 2)
                .view(bs, self.na, ny, nx, self.no)
                .contiguous()
            )

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                y[..., 0:2] = (
                    y[..., 0:2] * 2.0 - 0.5 + self.grid[i].to(x[i].device)
                ) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


# ---------------------------------------------------------------------------
# lib/utils/utils.py (initialize_weights only)
# ---------------------------------------------------------------------------


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


# ---------------------------------------------------------------------------
# lib/utils/autoanchor.py (check_anchor_order only)
# ---------------------------------------------------------------------------


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


# ---------------------------------------------------------------------------
# lib/models/YOLOP.py
# ---------------------------------------------------------------------------

# The lane line and the driving area segment branches without share information with each other and without link
YOLOP = [
    [24, 33, 42],  # Det_out_idx, Da_Segout_idx, LL_Segout_idx
    [-1, Focus, [3, 32, 3]],  # 0
    [-1, Conv, [32, 64, 3, 2]],  # 1
    [-1, BottleneckCSP, [64, 64, 1]],  # 2
    [-1, Conv, [64, 128, 3, 2]],  # 3
    [-1, BottleneckCSP, [128, 128, 3]],  # 4
    [-1, Conv, [128, 256, 3, 2]],  # 5
    [-1, BottleneckCSP, [256, 256, 3]],  # 6
    [-1, Conv, [256, 512, 3, 2]],  # 7
    [-1, SPP, [512, 512, [5, 9, 13]]],  # 8
    [-1, BottleneckCSP, [512, 512, 1, False]],  # 9
    [-1, Conv, [512, 256, 1, 1]],  # 10
    [-1, nn.Upsample, [None, 2, "nearest"]],  # 11
    [[-1, 6], Concat, [1]],  # 12
    [-1, BottleneckCSP, [512, 256, 1, False]],  # 13
    [-1, Conv, [256, 128, 1, 1]],  # 14
    [-1, nn.Upsample, [None, 2, "nearest"]],  # 15
    [[-1, 4], Concat, [1]],  # 16         # Encoder
    [-1, BottleneckCSP, [256, 128, 1, False]],  # 17
    [-1, Conv, [128, 128, 3, 2]],  # 18
    [[-1, 14], Concat, [1]],  # 19
    [-1, BottleneckCSP, [256, 256, 1, False]],  # 20
    [-1, Conv, [256, 256, 3, 2]],  # 21
    [[-1, 10], Concat, [1]],  # 22
    [-1, BottleneckCSP, [512, 512, 1, False]],  # 23
    [
        [17, 20, 23],
        Detect,
        [
            1,
            [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31], [19, 50, 38, 81, 68, 157]],
            [128, 256, 512],
        ],
    ],  # Detection head 24
    [16, Conv, [256, 128, 3, 1]],  # 25
    [-1, nn.Upsample, [None, 2, "nearest"]],  # 26
    [-1, BottleneckCSP, [128, 64, 1, False]],  # 27
    [-1, Conv, [64, 32, 3, 1]],  # 28
    [-1, nn.Upsample, [None, 2, "nearest"]],  # 29
    [-1, Conv, [32, 16, 3, 1]],  # 30
    [-1, BottleneckCSP, [16, 8, 1, False]],  # 31
    [-1, nn.Upsample, [None, 2, "nearest"]],  # 32
    [-1, Conv, [8, 2, 3, 1]],  # 33 Driving area segmentation head
    [16, Conv, [256, 128, 3, 1]],  # 34
    [-1, nn.Upsample, [None, 2, "nearest"]],  # 35
    [-1, BottleneckCSP, [128, 64, 1, False]],  # 36
    [-1, Conv, [64, 32, 3, 1]],  # 37
    [-1, nn.Upsample, [None, 2, "nearest"]],  # 38
    [-1, Conv, [32, 16, 3, 1]],  # 39
    [-1, BottleneckCSP, [16, 8, 1, False]],  # 40
    [-1, nn.Upsample, [None, 2, "nearest"]],  # 41
    [-1, Conv, [8, 2, 3, 1]],  # 42 Lane line segmentation head
]


class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save = [], []
        self.nc = 1
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(
                x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1
            )  # append to savelist

        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride, anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _, _ = model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            Detector.anchors /= Detector.stride.view(
                -1, 1, 1
            )  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()

        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = (
                    cache[block.from_]
                    if isinstance(block.from_, int)
                    else [x if j == -1 else cache[j] for j in block.from_]
                )  # calculate concat detect
            x = block(x)
            if i in self.seg_out_idx:  # save driving area segment result
                m = nn.Sigmoid()
                out.append(m(x))
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0, det_out)
        return out

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += (
                math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def get_net(cfg=None, **kwargs):
    m_block_cfg = YOLOP
    model = MCnet(m_block_cfg, **kwargs)
    return model


def build_yolop():
    return get_net()


def example_input_yolop():
    # Real repo demo (`lib/models/YOLOP.py __main__`) uses (1, 3, 256, 256);
    # kept at the same size here since it is already small and matches the
    # `s = 128` stride-probe the model performs internally at construction.
    return torch.randn(1, 3, 256, 256)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("YOLOP", "build_yolop", "example_input_yolop", 2022, "vendored"),
]
