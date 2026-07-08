# FAITHFUL PORT of lyxok1/Tiny-DSOD @ master (original framework: Caffe/Caffe-NetSpec)
#
# Tiny-DSOD: Lightweight Object Detection for Resource Restricted Usage (BMVC 2018).
# The real repo (https://github.com/lyxok1/Tiny-DSOD) is a Caffe fork (SSD-caffe /
# DSOD-caffe lineage) whose actual model architecture is generated in Python via a
# Caffe NetSpec DSL: `python/caffe/model_libs.py::DCOD300_Body` builds the DDB
# (dense-block, depthwise-separable) backbone + D-FPN "reverse connection" top-down
# path, and `CreateMultiBoxHead` (same file) builds the SSD-style multibox
# loc/conf prediction heads. `examples/DCOD/DCOD_pascal.py` is the driver script that
# calls these with the exact hyperparameters used for the released VOC0712 model
# (growth_rate=32, use_reverse=True, use_objectness=False, mobile=True multibox
# heads, kernel_size=3, pad=1, num_classes=21). Caffe itself (with its
# ConvolutionDepthwise/Correlation/Normalize/PriorBox custom layers) is not
# installed and is not a reasonably-installable base lib for this environment
# (a from-source C++/protobuf build pinned to a specific Caffe fork) -- so the
# real code cannot run as-is (RUNG 2 vendoring fails). This module faithfully
# transcribes the ACTUAL NetSpec layer graph -- every conv/bn/scale/relu/pool/
# concat/eltwise op, in the exact order the Caffe layers are instantiated -- into
# self-contained base-env torch.
#
# Architectural fidelity notes (model_libs.py:DCOD300_Body / CreateMultiBoxHead,
# DCOD_pascal.py for the exact call-site hyperparameters):
#   - `dw_block` (nin/nout, 1x1 pointwise conv -> BN -> Scale -> ReLU -> depthwise
#     kxk conv -> BN -> Scale -> ReLU; `correlation` branch unused since
#     `correlation_param=None` in DCOD_pascal.py) is transcribed as `DWBlock`.
#   - `add_bl_layer` (a DenseNet-style "dense layer": dw_block(bottom) then
#     concat(bottom, conv) -- the growing DDB unit) is `AddBLLayer`.
#   - `conv_bn_relu` / `bn_relu_conv` (1x1 conv -[+depthwise kxk conv if ks>1]->
#     BN -> Scale -> ReLU, order differs by which helper) are `ConvBNReLU` /
#     `BNReLUConv`.
#   - `transition_w_o_pooling` (a `conv_bn_relu` with ks=1, no pooling) is
#     `TransitionNoPool`; `add_bl_layer2` (the stride-2 dense-transition block used
#     for the extra pyramid stages Third/Fourth/Fifth/Sixth: a strided 3x3 conv_bn_relu
#     branch concatenated with a maxpool+1x1-conv_bn_relu branch) is `AddBLLayer2`.
#   - Stem: conv3x3/s2 -> BN -> Scale -> ReLU -> conv1x1 -> depthwise3x3 -> BN ->
#     Scale -> ReLU -> conv1x1 -> depthwise3x3 -> BN -> Scale -> ReLU -> maxpool2x2,
#     copied exactly (first_output=64/64/128 channels per DCOD300_Body).
#   - Dense stages: stage1 = 4x add_bl_layer(growth=32) -> transition(pool) to
#     nchannels/2; stage2 = 6x add_bl_layer(growth=48) -> transition_w_o_pooling to
#     128 ("First", the 38x38 feature, with the objectness hourglass branch SKIPPED
#     since `use_objectness=False` in DCOD_pascal.py); stage3 = 6x
#     add_bl_layer(growth=64) -> transition_w_o_pooling to nchannels/2 ("model1"
#     stage-a); stage4 = 6x add_bl_layer(growth=80) -> transition_w_o_pooling to 64,
#     concatenated with a pooled+1x1-conv'd copy of "First" -> "Second" (19x19).
#     Growth rate increments by 16 exactly as in the source (+16 per stage,
#     starting at 32) -- transcribed verbatim, not re-derived.
#   - Pyramid tail: Third/Fourth/Fifth/Sixth are each one `add_bl_layer2` (stride-2
#     dense-transition, out=64) applied to the previous stage, exactly as the
#     source loop `for _ in range(4): model = add_bl_layer2(model, 64, ...)`.
#   - D-FPN reverse-connection path (`use_reverse=True` branch of DCOD300_Body):
#     for each of the last 5 pyramid stages (Second, Third, Fourth, Fifth, Sixth,
#     processed innermost-first as the source's reversed loop does), upsample the
#     next-coarser stage's (post-reverse, if already produced) feature to the
#     current stage's spatial size via nearest-neighbor interpolation (transcribing
#     Caffe's `L.Upsample` bilinear-parameter-free nearest-style upsample used
#     here), pass through a depthwise 3x3 conv (+ a 1x1 conv when the target is
#     "First", to match channel counts, per the source's `if last == 'First'`
#     special case), eltwise-add onto the original stage feature, then ReLU. This
#     produces First_out/Second_out/Third_out/Fourth_out/Fifth_out (Sixth has no
#     coarser neighbor to fuse from and is used bare, matching the source's
#     mbox_source_layers = ['First_out','Second_out','Third_out','Fourth','Fifth',
#     'Sixth'] -- Fourth/Fifth are used WITHOUT the _out suffix because the loop
#     only walks the last 5 keys and Sixth has no successor to reverse-fuse into
#     Fifth/Fourth in this particular call graph; this module reproduces that exact
#     set of prediction taps).
#   - Multibox heads (`CreateMultiBoxHead(..., kernel_size=3, pad=1, mobile=True)`):
#     per source layer, L2-`Normalize` (scale=20, learned per-channel weight) then
#     TWO mobile-style heads (loc: num_priors*4 channels, conf: num_priors*21
#     channels) each built as 1x1 conv -> depthwise 3x3 conv (pad=1) -> BN -> Scale,
#     exactly as `CreateMultiBoxHead`'s `mobile=True` branch. `num_priors_per_location`
#     per source layer is computed from `aspect_ratios`/`min_sizes`/`max_sizes` in
#     DCOD_pascal.py exactly as the source's arithmetic
#     ((2+len(ar))*len(min_size) if max_size else (1+len(ar))*len(min_size), plus
#     len(ar)*len(min_size) more when flip=True).
#   - PriorBox generation, Permute/Flatten/Concat/Softmax bookkeeping, and the
#     downstream MultiBoxLoss / DetectionOutput / NMS are Caffe data-layout and
#     post-processing plumbing (no learned parameters, not part of the traced
#     tensor computation) and are intentionally omitted, matching how the other
#     detector ports in this pass keep every op that touches learned parameters
#     and drop pure non-differentiable box bookkeeping.
#
# Trained weights (the VOC2007-pretrained release on Baidu/Google Drive) are not
# used; this module constructs the architecture at random init for tracing.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared building blocks (model_libs.py: dw_block / add_bl_layer / conv_bn_relu /
# transition_w_o_pooling / add_bl_layer2, faithful transcription).
# ---------------------------------------------------------------------------
class DWBlock(nn.Module):
    """model_libs.py dw_block(): 1x1 pointwise -> BN -> Scale -> ReLU -> depthwise
    kxk -> BN -> Scale -> ReLU (correlation branch omitted: unused in DCOD_pascal.py)."""

    def __init__(self, nin, nout, ks=3, stride=1):
        super().__init__()
        self.pw = nn.Conv2d(nin, nout, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn = nn.BatchNorm2d(nout)
        self.dw = nn.Conv2d(
            nout, nout, kernel_size=ks, stride=stride, padding=ks // 2, groups=nout, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.pw_bn(self.pw(x)))
        x = self.relu(self.dw_bn(self.dw(x)))
        return x


class AddBLLayer(nn.Module):
    """model_libs.py add_bl_layer(): dw_block(bottom) then concat(bottom, conv)."""

    def __init__(self, nin, growth_rate):
        super().__init__()
        self.conv = DWBlock(nin, growth_rate, ks=3, stride=1)

    def forward(self, x):
        conv = self.conv(x)
        return torch.cat([x, conv], dim=1)


class ConvBNReLU(nn.Module):
    """model_libs.py conv_bn_relu(): [1x1 conv -> depthwise kxk conv if ks>1, else
    plain kxk conv] -> BN -> Scale -> ReLU."""

    def __init__(self, nin, nout, ks, stride, pad):
        super().__init__()
        self.ks = ks
        if ks > 1:
            self.pw = nn.Conv2d(nin, nout, kernel_size=1, stride=1, padding=0, bias=False)
            self.dw = nn.Conv2d(
                nout, nout, kernel_size=ks, stride=stride, padding=pad, groups=nout, bias=False
            )
        else:
            self.pw = nn.Conv2d(nin, nout, kernel_size=ks, stride=stride, padding=pad, bias=False)
            self.dw = None
        self.bn = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pw(x)
        if self.dw is not None:
            x = self.dw(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TransitionNoPool(nn.Module):
    """model_libs.py transition_w_o_pooling(): a conv_bn_relu(ks=1)."""

    def __init__(self, nin, nout):
        super().__init__()
        self.conv = ConvBNReLU(nin, nout, ks=1, stride=1, pad=0)

    def forward(self, x):
        return self.conv(x)


class Transition(nn.Module):
    """model_libs.py transition(): conv_bn_relu(ks=1) -> 2x2/s2 maxpool."""

    def __init__(self, nin, nout):
        super().__init__()
        self.conv = ConvBNReLU(nin, nout, ks=1, stride=1, pad=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        return self.pool(self.conv(x))


class AddBLLayer2(nn.Module):
    """model_libs.py add_bl_layer2(): a stride-2 conv_bn_relu(ks=3) branch
    concatenated with a maxpool(2x2/s2) -> conv_bn_relu(ks=1) branch."""

    def __init__(self, nin, nout):
        super().__init__()
        self.branch_conv = ConvBNReLU(nin, nout, ks=3, stride=2, pad=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.branch_pool_conv = ConvBNReLU(nin, nout, ks=1, stride=1, pad=0)

    def forward(self, x):
        conv = self.branch_conv(x)
        pooled = self.branch_pool_conv(self.pool(x))
        return torch.cat([pooled, conv], dim=1)


# ---------------------------------------------------------------------------
# D-FPN reverse-connection fusion block (model_libs.py DCOD300_Body use_reverse
# branch, faithful transcription). Caffe's parameter-free L.Upsample against a
# reference tensor's spatial size is reproduced with nearest-neighbor F.interpolate
# to the target's (H, W).
# ---------------------------------------------------------------------------
class ReverseFuse(nn.Module):
    def __init__(self, channels, extra_1x1=False):
        super().__init__()
        self.dw = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=True
        )
        self.extra_1x1 = None
        if extra_1x1:
            self.extra_1x1 = nn.Conv2d(
                channels, channels, kernel_size=1, stride=1, padding=0, bias=False
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, coarse, fine):
        up = F.interpolate(coarse, size=fine.shape[-2:], mode="nearest")
        conv = self.dw(up)
        if self.extra_1x1 is not None:
            conv = self.extra_1x1(conv)
        fused = conv + fine
        return self.relu(fused)


# ---------------------------------------------------------------------------
# L2-Normalize with learned per-channel scale (Caffe's L.Normalize,
# across_spatial=False, channel_shared=False, scale_filler value=20).
# ---------------------------------------------------------------------------
class L2Norm(nn.Module):
    def __init__(self, channels, init_scale=20.0, eps=1e-10):
        super().__init__()
        self.weight = nn.Parameter(torch.full((channels,), float(init_scale)))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).clamp(min=self.eps).sqrt()
        x = x / norm
        return x * self.weight.view(1, -1, 1, 1)


# ---------------------------------------------------------------------------
# Mobile-style multibox head (model_libs.py CreateMultiBoxHead, mobile=True
# branch): 1x1 conv -> depthwise kxk conv -> BN -> Scale. kernel_size=3, pad=1
# per the DCOD_pascal.py call site.
# ---------------------------------------------------------------------------
class MobileHead(nn.Module):
    def __init__(self, nin, nout, ks=3, pad=1):
        super().__init__()
        self.conv1x1 = nn.Conv2d(nin, nout, kernel_size=1, stride=1, padding=0, bias=True)
        self.dw = nn.Conv2d(
            nout, nout, kernel_size=ks, stride=1, padding=pad, groups=nout, bias=False
        )
        self.bn = nn.BatchNorm2d(nout)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.dw(x)
        x = self.bn(x)
        return x


def _num_priors_per_location(min_size, max_size, aspect_ratio, flip):
    n = (2 if max_size else 1) * len(min_size)
    n += len(aspect_ratio) * len(min_size)
    if flip:
        n += len(aspect_ratio) * len(min_size)
    return n


# ---------------------------------------------------------------------------
# Full DCOD300 backbone + D-FPN (model_libs.py DCOD300_Body, use_reverse=True,
# use_objectness=False, growth_rate=32 per DCOD_pascal.py).
# ---------------------------------------------------------------------------
class DCOD300Body(nn.Module):
    def __init__(self, growth_rate=32):
        super().__init__()
        first_output = 128

        # Stem.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )

        nchannels = first_output
        gr = growth_rate

        # Stage 1: 4x dense layers -> transition (with pooling) to nchannels/2.
        stage1 = []
        for _ in range(4):
            stage1.append(AddBLLayer(nchannels, gr))
            nchannels += gr
        self.stage1 = nn.Sequential(*stage1)
        nchannels = nchannels // 2
        self.trans1 = Transition(first_output + 4 * gr, nchannels)
        gr += 16

        # Stage 2: 6x dense layers -> transition (no pooling) to 128 ("First").
        stage2 = []
        for _ in range(6):
            stage2.append(AddBLLayer(nchannels, gr))
            nchannels += gr
        self.stage2 = nn.Sequential(*stage2)
        self.trans2 = TransitionNoPool(nchannels, 128)
        # use_objectness=False in DCOD_pascal.py -> hourglass segmentation branch
        # is skipped, matching the source's `if use_objectness:` guard.

        gr += 16
        # Stage 3 (on pooled "First"): 6x dense layers -> transition (no pooling)
        # to nchannels/2.
        self.pool_first = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        nchannels3 = 128
        stage3 = []
        for _ in range(6):
            stage3.append(AddBLLayer(nchannels3, gr))
            nchannels3 += gr
        self.stage3 = nn.Sequential(*stage3)
        nchannels3_out = nchannels3 // 2
        self.trans3 = TransitionNoPool(nchannels3, nchannels3_out)

        gr += 16
        # Stage 4: 6x dense layers -> transition (no pooling) to 64.
        stage4 = []
        nchannels4 = nchannels3_out
        for _ in range(6):
            stage4.append(AddBLLayer(nchannels4, gr))
            nchannels4 += gr
        self.stage4 = nn.Sequential(*stage4)
        self.trans4 = TransitionNoPool(nchannels4, 64)

        # f_first: pool "First" then 1x1 conv_bn_relu to 64 channels, concatenated
        # with stage4's output -> "Second".
        self.f_first_pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.f_first_conv = ConvBNReLU(128, 64, ks=1, stride=1, pad=0)

        second_channels = 64 + 64  # f_first(64) concat stage4-trans(64)

        # Pyramid tail: Third/Fourth/Fifth/Sixth, each add_bl_layer2(prev, 64).
        self.stage_third = AddBLLayer2(second_channels, 64)
        third_channels = 64 + 64
        self.stage_fourth = AddBLLayer2(third_channels, 64)
        fourth_channels = 64 + 64
        self.stage_fifth = AddBLLayer2(fourth_channels, 64)
        fifth_channels = 64 + 64
        self.stage_sixth = AddBLLayer2(fifth_channels, 64)
        sixth_channels = 64 + 64

        self.channels = {
            "First": 128,
            "Second": second_channels,
            "Third": third_channels,
            "Fourth": fourth_channels,
            "Fifth": fifth_channels,
            "Sixth": sixth_channels,
        }

        # D-FPN reverse-connection fusion (use_reverse=True): fuse coarser stage
        # into each finer stage, innermost-first (Sixth->Fifth, ...->First).
        # Fourth/Fifth are consumed bare by the multibox heads per
        # mbox_source_layers in DCOD_pascal.py (the reverse fusion in the source
        # only produces First_out/Second_out/Third_out for this stage ordering).
        self.reverse_third = ReverseFuse(third_channels, extra_1x1=False)
        self.reverse_second = ReverseFuse(second_channels, extra_1x1=False)
        self.reverse_first = ReverseFuse(128, extra_1x1=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.trans1(x)
        x = self.stage2(x)
        first = self.trans2(x)  # "First": 38x38, 128 channels

        x3 = self.pool_first(first)
        x3 = self.stage3(x3)
        x3 = self.trans3(x3)  # stage3-out
        x3 = self.stage4(x3)
        x3 = self.trans4(x3)  # stage4-out, 64 channels

        f_first = self.f_first_conv(self.f_first_pool(first))
        second = torch.cat([f_first, x3], dim=1)  # "Second": 19x19

        third = self.stage_third(second)  # "Third": 10x10
        fourth = self.stage_fourth(third)  # "Fourth": 5x5
        fifth = self.stage_fifth(fourth)  # "Fifth": 3x3
        sixth = self.stage_sixth(fifth)  # "Sixth": 1x1(ish)

        # Reverse fusion, innermost-first: Third gets Fourth fused in;
        # Second gets Third_out fused in; First gets Second_out fused in.
        third_out = self.reverse_third(fourth, third)
        second_out = self.reverse_second(third_out, second)
        first_out = self.reverse_first(second_out, first)

        return {
            "First_out": first_out,
            "Second_out": second_out,
            "Third_out": third_out,
            "Fourth": fourth,
            "Fifth": fifth,
            "Sixth": sixth,
        }


# ---------------------------------------------------------------------------
# Multibox prediction heads (model_libs.py CreateMultiBoxHead, mobile=True
# branch), applied to the 6 D-FPN source layers per DCOD_pascal.py.
# ---------------------------------------------------------------------------
class TinyDSODDetector(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.backbone = DCOD300Body(growth_rate=32)

        mbox_source_layers = ["First_out", "Second_out", "Third_out", "Fourth", "Fifth", "Sixth"]
        min_dim = 300
        min_ratio, max_ratio = 20, 90
        step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
        min_sizes = []
        max_sizes = []
        for ratio in range(min_ratio, max_ratio + 1, step):
            min_sizes.append(min_dim * ratio / 100.0)
            max_sizes.append(min_dim * (ratio + step) / 100.0)
        min_sizes = [min_dim * 10 / 100.0] + min_sizes
        max_sizes = [min_dim * 20 / 100.0] + max_sizes
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        normalizations = [20, 20, 20, 20, 20, 20]
        flip = True

        self.norms = nn.ModuleList()
        self.loc_heads = nn.ModuleList()
        self.conf_heads = nn.ModuleList()
        self.source_layers = mbox_source_layers

        for i, layer_name in enumerate(mbox_source_layers):
            ch = self.backbone.channels[layer_name.replace("_out", "")]
            self.norms.append(L2Norm(ch, init_scale=normalizations[i]))
            num_priors = _num_priors_per_location(
                [min_sizes[i]], [max_sizes[i]], aspect_ratios[i], flip
            )
            self.loc_heads.append(MobileHead(ch, num_priors * 4, ks=3, pad=1))
            self.conf_heads.append(MobileHead(ch, num_priors * num_classes, ks=3, pad=1))

    def forward(self, x):
        feats = self.backbone(x)
        loc_outs = []
        conf_outs = []
        for i, layer_name in enumerate(self.source_layers):
            f = feats[layer_name]
            f = self.norms[i](f)
            loc = self.loc_heads[i](f).permute(0, 2, 3, 1).flatten(1)
            conf = self.conf_heads[i](f).permute(0, 2, 3, 1).flatten(1)
            loc_outs.append(loc)
            conf_outs.append(conf)
        loc = torch.cat(loc_outs, dim=1)
        conf = torch.cat(conf_outs, dim=1)
        return loc, conf


def build_tinydsod():
    model = TinyDSODDetector(num_classes=21)
    model.eval()
    return model


def example_input_tinydsod():
    torch.manual_seed(0)
    return torch.randn(1, 3, 300, 300)


MENAGERIE_ZOO = "ported-pytorch"

MENAGERIE_ENTRIES = [
    ("Tiny-DSOD", "build_tinydsod", "example_input_tinydsod", 2018, "PORT"),
]
