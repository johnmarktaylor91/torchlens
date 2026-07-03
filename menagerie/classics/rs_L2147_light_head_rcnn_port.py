# FAITHFUL PORT of zengarden/light_head_rcnn @ master (original framework: TensorFlow 1.x
# + custom CUDA ops), transcribed via the real PyTorch port TreB1eN/Lighthead-RCNN-in-
# Pytorch0.4.1 @ master (branch: https://github.com/TreB1eN/Lighthead-RCNN-in-Pytorch0.4.1).
#
# Li, Zhang, Chen, Wang, Yan, Sun 2017 "Light-Head R-CNN: In Defense of Two-Stage Object
# Detector" (arxiv 1711.07264). The official repo is TF1.x with custom CUDA psalign_pooling
# ops; TreB1eN's PyTorch port also depends on custom CUDA extensions (RoIAlign_pytorch,
# psroi_pooling, nms) that are unbuildable in a modern base env (py3.6-era .so files,
# unmaintained). This module transcribes the ACTUAL PyTorch source, verbatim where
# possible:
#   - ResNet101Extractor / Conv2DBNActiv / Chainer_resnet_bottleneck / Chainer_ResBlock
#     (models/backbone.py) -- copied verbatim (real dilated-C5 ResNet101 backbone).
#   - RegionProposalNetwork conv1/score/loc (models/region_proposal_network.py) -- the
#     trainable RPN conv trunk is copied verbatim; the *proposal* step (numpy anchor
#     generation + non-differentiable NMS over ~20000 anchors, returning a
#     variable-length ROI array) is not a tensor op and is out of scope for a graph
#     capture -- this port keeps the exact RPN conv trunk (conv1/score/loc, the
#     part with parameters) and, in the head-only forward wrapper below, feeds a fixed
#     stand-in set of ROIs (as the real code would after its proposal step) into the
#     head, matching how LightHeadRCNNResNet101_Head._call__ actually receives `rois`.
#   - GlobalContextModule / LightHeadRCNNResNet101_Head (models/head.py) -- the paper's
#     actual "large separable convolution" thin-head contribution -- copied verbatim.
#   - PSRoIMaxAlignPooling2D (models/psroialign_cpu.py) -- the repo's own pure-PyTorch
#     CPU reference implementation of the custom CUDA PSRoIAlign kernel (position-
#     sensitive bilinear-interpolated max pooling), copied verbatim; this is the exact
#     algorithm the CUDA kernel implements, just without the JIT-compiled extension.
"""Light-Head R-CNN: dilated-C5 ResNet101 backbone + RPN trunk + thin/large-separable-conv
position-sensitive-ROI-align detection head (Li et al. 2017)."""

import torch
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
    init,
)
from torch.nn import functional as F
from torch.autograd import Function

MENAGERIE_ZOO = "ported-pytorch"


# --- vendored from utils/utils.py ---
def normal_init(m, mean, stddev):
    if type(m) == Linear or type(m) == Conv2d:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


# --- vendored from models/backbone.py ---
class SEModule(Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Conv2DBNActiv(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ksize=None,
        stride=1,
        pad=0,
        dilate=1,
        groups=1,
        nobias=True,
        activ=F.relu,
        bn_kwargs={},
    ):
        if ksize is None:
            out_channels, ksize, in_channels = in_channels, out_channels, None
        self.activ = activ
        super(Conv2DBNActiv, self).__init__()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            bias=(not nobias),
            dilation=dilate,
            groups=groups,
        )
        self.bn = BatchNorm2d(out_channels, **bn_kwargs)

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        if self.activ is None:
            return h
        else:
            return self.activ(h)


class Chainer_resnet_bottleneck(Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        stride=1,
        dilate=1,
        groups=1,
        bn_kwargs={},
        residual_conv=False,
        stride_first=False,
        add_seblock=False,
    ):
        if stride_first:
            first_stride = stride
            second_stride = 1
        else:
            first_stride = 1
            second_stride = stride
        super(Chainer_resnet_bottleneck, self).__init__()
        self.conv1 = Conv2DBNActiv(
            in_channels, mid_channels, 1, first_stride, 0, nobias=True, bn_kwargs=bn_kwargs
        )
        self.conv2 = Conv2DBNActiv(
            mid_channels,
            mid_channels,
            3,
            second_stride,
            dilate,
            dilate,
            groups,
            nobias=True,
            bn_kwargs=bn_kwargs,
        )
        self.conv3 = Conv2DBNActiv(
            mid_channels, out_channels, 1, 1, 0, nobias=True, activ=None, bn_kwargs=bn_kwargs
        )
        if add_seblock:
            self.se = SEModule(out_channels)
        if residual_conv:
            self.residual_conv = Conv2DBNActiv(
                in_channels,
                out_channels,
                1,
                stride,
                0,
                nobias=True,
                activ=None,
                bn_kwargs=bn_kwargs,
            )

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)

        if hasattr(self, "se"):
            h = self.se(h)

        if hasattr(self, "residual_conv"):
            residual = self.residual_conv(x)
        else:
            residual = x
        h += residual
        h = F.relu(h)
        return h


class Chainer_ResBlock(Module):
    def __init__(
        self,
        n_layer,
        in_channels,
        mid_channels,
        out_channels,
        stride,
        dilate=1,
        groups=1,
        bn_kwargs={},
        stride_first=False,
        add_seblock=False,
    ):
        super(Chainer_ResBlock, self).__init__()
        self.a = Chainer_resnet_bottleneck(
            in_channels,
            mid_channels,
            out_channels,
            stride,
            dilate,
            groups,
            bn_kwargs=bn_kwargs,
            residual_conv=True,
            stride_first=stride_first,
            add_seblock=add_seblock,
        )
        blocks = []
        for i in range(n_layer - 1):
            blocks.append(
                Chainer_resnet_bottleneck(
                    out_channels,
                    mid_channels,
                    out_channels,
                    stride=1,
                    dilate=dilate,
                    bn_kwargs=bn_kwargs,
                    residual_conv=False,
                    add_seblock=add_seblock,
                    groups=groups,
                )
            )
        self.layers = Sequential(*blocks)

    def forward(self, x):
        return self.layers(self.a(x))


class ResNet101Extractor(Module):
    """ResNet101 Extractor for LightHeadRCNN ResNet101 implementation.

    Outputs feature maps. Dilated convolution is used in the C5 stage.

    NOTE (port scope): layer depths (3, 4, 23, 3) are the real ResNet101 config from the
    source repo; the checkpoint-load / grad-freeze / bn-eval-freeze training bookkeeping
    in the original __init__ is deliberately omitted here since it is orthogonal to the
    module's forward-pass architecture (random-init trace, not a fine-tuning run).
    """

    def __init__(self):
        super(ResNet101Extractor, self).__init__()
        kwargs = {
            "stride_first": True,
            "bn_kwargs": {"eps": 2e-05},
        }

        self.conv1 = Conv2DBNActiv(3, 64, 7, 2, 3, nobias=True, bn_kwargs={"eps": 2e-05})
        self.pool1 = MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.res2 = Chainer_ResBlock(3, 64, 64, 256, 1, **kwargs)
        self.res3 = Chainer_ResBlock(4, 256, 128, 512, 2, **kwargs)
        self.res4 = Chainer_ResBlock(23, 512, 256, 1024, 2, **kwargs)
        self.res5 = Chainer_ResBlock(3, 1024, 512, 2048, 1, 2, **kwargs)

        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.pool1(self.conv1(x))
        h = self.res2(h)
        h = self.res3(h)
        res4 = self.res4(h)
        res5 = self.res5(res4)
        return res4, res5


# --- vendored from models/region_proposal_network.py (trainable conv trunk only; see
# module docstring for why the numpy proposal step is out of scope for graph capture) ---
class RegionProposalNetworkTrunk(Module):
    """Region Proposal Network conv trunk introduced in Faster R-CNN (Ren et al. 2015),
    as used by Light-Head R-CNN. Produces per-anchor location offsets and objectness
    scores from the backbone feature map; the anchor generation and 20000->300 proposal
    selection (numpy + NMS, non-differentiable) live outside this trunk in the real
    RegionProposalNetwork.forward and are not tensor ops."""

    def __init__(self, in_channels=1024, mid_channels=512, n_anchor=9):
        super(RegionProposalNetworkTrunk, self).__init__()
        self.conv1 = Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.loc = Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        self.apply(lambda x: normal_init(x, 0, 0.01))

    def forward(self, x):
        n, _, hh, ww = x.shape
        h = F.relu(self.conv1(x))
        rpn_locs = self.loc(h).permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(h).permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        return rpn_locs, rpn_scores


# --- vendored from models/psroialign_cpu.py (the repo's own pure-PyTorch CPU reference
# implementation of the custom CUDA PSRoIAlign kernel used by the head) ---
class PSROIMaxAlignPooling2D(Function):
    @staticmethod
    def forward(ctx, ps_sensitive_feature, rois, pooling_paras):
        [pooled_channels, pooled_size, spatial_scale, sampling_ratio] = pooling_paras
        channels, height, width = ps_sensitive_feature.shape[1:]
        n_roi = rois.shape[0]
        pooled_data = torch.zeros(
            [n_roi, pooled_channels, pooled_size, pooled_size], dtype=torch.float
        )

        w1w2w3w4_group = []
        xlowylowxhighyhigh_group = []
        for i in range(pooled_data.nelement()):
            pw = i % pooled_size
            ph = (i // pooled_size) % pooled_size
            ctop = (i // pooled_size // pooled_size) % pooled_channels
            n = i // pooled_size // pooled_size // pooled_channels
            roi_start_h = rois[n, 0] * spatial_scale
            roi_start_w = rois[n, 1] * spatial_scale
            roi_end_h = rois[n, 2] * spatial_scale
            roi_end_w = rois[n, 3] * spatial_scale

            roi_height = max(roi_end_h - roi_start_h, 1.0)
            roi_width = max(roi_end_w - roi_start_w, 1.0)
            bin_size_h = 1.0 * roi_height / pooled_size
            bin_size_w = 1.0 * roi_width / pooled_size

            c = (ctop * pooled_size + ph) * pooled_size + pw

            if sampling_ratio > 0:
                roi_bin_grid_h = sampling_ratio
                roi_bin_grid_w = sampling_ratio
            else:
                roi_bin_grid_h = int((roi_height / pooled_size) + 0.999999)
                roi_bin_grid_w = int((roi_width / pooled_size) + 0.999999)

            maxval = -1e20
            max_w1w2w3w4 = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
            max_xlowylowxhighyhigh = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)

            iy = 0
            while iy < roi_bin_grid_h:
                y = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
                ix = 0
                while ix < roi_bin_grid_w:
                    x = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w

                    if y <= 0:
                        y = 0
                    if x <= 0:
                        x = 0

                    y_low = int(y)
                    x_low = int(x)

                    if y_low >= height - 1:
                        y_high = y_low = height - 1
                        y = float(y_low)
                    else:
                        y_high = y_low + 1

                    if x_low >= width - 1:
                        x_high = x_low = width - 1
                        x = float(x_low)
                    else:
                        x_high = x_low + 1

                    ly = y - y_low
                    lx = x - x_low
                    hy = 1.0 - ly
                    hx = 1.0 - lx

                    v1 = ps_sensitive_feature[0, c, y_low, x_low].item()
                    v2 = ps_sensitive_feature[0, c, y_low, x_high].item()
                    v3 = ps_sensitive_feature[0, c, y_high, x_low].item()
                    v4 = ps_sensitive_feature[0, c, y_high, x_high].item()

                    w1 = hy * hx
                    w2 = hy * lx
                    w3 = ly * hx
                    w4 = ly * lx

                    tmpval = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
                    if tmpval > maxval:
                        maxval = tmpval
                        max_w1w2w3w4 = torch.tensor([[float(w1), float(w2), float(w3), float(w4)]])
                        max_xlowylowxhighyhigh = torch.tensor(
                            [[x_low, y_low, x_high, y_high]], dtype=torch.long
                        )

                    ix += 1
                iy += 1

            xlowylowxhighyhigh_group.append(max_xlowylowxhighyhigh)
            w1w2w3w4_group.append(max_w1w2w3w4)
            pooled_data[n, ctop, ph, pw] = maxval

        ctx.save_for_backward(
            torch.cat(xlowylowxhighyhigh_group),
            torch.cat(w1w2w3w4_group),
            torch.tensor([channels, height, width]),
        )
        return pooled_data

    @staticmethod
    def backward(ctx, grad_output):
        xlowylowxhighyhigh_group, w1w2w3w4_group, paras = ctx.saved_tensors
        channels, height, width = paras[0].item(), paras[1].item(), paras[2].item()
        pooled_channels, pooled_size = grad_output.shape[1], grad_output.shape[2]
        bp_grads = torch.zeros([1, channels, height, width], dtype=torch.float)

        for i in range(grad_output.nelement()):
            pw = i % pooled_size
            ph = (i // pooled_size) % pooled_size
            ctop = (i // pooled_size // pooled_size) % pooled_channels
            n = i // pooled_size // pooled_size // pooled_channels

            c = (ctop * pooled_size + ph) * pooled_size + pw

            [w1, w2, w3, w4] = w1w2w3w4_group[i].tolist()
            [x_low, y_low, x_high, y_high] = xlowylowxhighyhigh_group[i].tolist()
            grad_this_bin = grad_output[n, ctop, ph, pw]

            g1 = grad_this_bin * w1
            g2 = grad_this_bin * w2
            g3 = grad_this_bin * w3
            g4 = grad_this_bin * w4

            if x_low >= 0 and x_high >= 0 and y_low >= 0 and y_high >= 0:
                bp_grads[0, c, y_low, x_low] += g1
                bp_grads[0, c, y_low, x_high] += g2
                bp_grads[0, c, y_high, x_low] += g3
                bp_grads[0, c, y_high, x_high] += g4
        return bp_grads, None, None


class PSRoIAlign(Module):
    """Thin nn.Module wrapper around the repo's PSROIMaxAlignPooling2D CPU reference
    Function, matching the call signature used by LightHeadRCNNResNet101_Head."""

    def __init__(self, spatial_scale, pooled_size, sampling_ratio, out_channels):
        super(PSRoIAlign, self).__init__()
        self.pooling_paras = [out_channels, pooled_size, spatial_scale, sampling_ratio]

    def forward(self, ps_sensitive_feature, rois):
        return PSROIMaxAlignPooling2D.apply(ps_sensitive_feature, rois, self.pooling_paras)


# --- vendored from models/head.py ---
class GlobalContextModule(Module):
    def __init__(self, in_channels, mid_channels, out_channels, ksize):
        super(GlobalContextModule, self).__init__()
        padsize = int((ksize - 1) / 2)
        self.col_max = Conv2d(in_channels, mid_channels, (ksize, 1), 1, (padsize, 0))
        self.col = Conv2d(mid_channels, out_channels, (1, ksize), 1, (0, padsize))
        self.row_max = Conv2d(in_channels, mid_channels, (1, ksize), 1, (0, padsize))
        self.row = Conv2d(mid_channels, out_channels, (ksize, 1), 1, (padsize, 0))

    def __call__(self, x):
        h_col = self.col(self.col_max(x))
        h_row = self.row(self.row_max(x))
        return F.relu(h_col + h_row)


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class LightHeadRCNNResNet101_Head(Module):
    def __init__(
        self, n_class=81, roi_size=7, out_channels=10, spatial_scale=1 / 16.0, sampling_ratio=2
    ):
        super(LightHeadRCNNResNet101_Head, self).__init__()
        self.n_class = n_class
        self.spatial_scale = spatial_scale
        self.roi_size = roi_size
        self.out_channels = out_channels
        self.sampling_ratio = sampling_ratio
        self.c_out = self.roi_size * self.roi_size * self.out_channels
        self.global_context_module = GlobalContextModule(2048, 256, self.c_out, 15)
        self.flatten = Flatten()
        self.fc1 = Linear(self.c_out, 2048)
        self.score = Linear(2048, n_class)
        self.cls_loc = Linear(2048, 4 * n_class)
        self.apply(lambda x: normal_init(x, 0, 0.01))
        self.cls_loc.apply(lambda x: normal_init(x, 0, 0.001))
        self.pooling = PSRoIAlign(
            self.spatial_scale, self.roi_size, self.sampling_ratio, self.out_channels
        )

    def __call__(self, x, rois):
        device = x.device
        h = self.global_context_module(x)
        pool = self.pooling(h, rois.to(device))
        pool = self.flatten(pool)
        fc1 = F.relu(self.fc1(pool))
        roi_cls_locs = self.cls_loc(fc1)
        roi_scores = self.score(fc1)
        return roi_cls_locs, roi_scores


# --- port-scope forward wrapper: real backbone -> real RPN trunk -> real thin head,
# with a fixed stand-in ROI set replacing the non-differentiable numpy proposal step
# (see RegionProposalNetworkTrunk docstring above) ---
class LightHeadRCNN(Module):
    def __init__(self, n_class=21, roi_size=7, n_rois=4, out_channels=10, sampling_ratio=2):
        super(LightHeadRCNN, self).__init__()
        self.extractor = ResNet101Extractor()
        self.rpn_trunk = RegionProposalNetworkTrunk(in_channels=1024, mid_channels=512, n_anchor=9)
        self.head = LightHeadRCNNResNet101_Head(
            n_class=n_class,
            roi_size=roi_size,
            out_channels=out_channels,
            spatial_scale=1 / 16.0,
            sampling_ratio=sampling_ratio,
        )
        self.n_rois = n_rois

    def forward(self, x):
        img_h, img_w = x.shape[2], x.shape[3]
        res4, res5 = self.extractor(x)
        rpn_locs, rpn_scores = self.rpn_trunk(res4)

        # Stand-in ROIs (y1, x1, y2, x2) in input-pixel coordinates, spanning the image,
        # matching the format LightHeadRCNNResNet101_Head consumes in the real repo's
        # inference path (post-proposal `rois`, shape [R, 4]).
        rois = torch.tensor(
            [[0.0, 0.0, img_h / (i + 1), img_w / (i + 1)] for i in range(self.n_rois)],
            dtype=torch.float,
        )

        roi_cls_locs, roi_scores = self.head(res5, rois)
        return rpn_locs, rpn_scores, roi_cls_locs, roi_scores


def build_light_head_rcnn():
    # roi_size/out_channels/sampling_ratio kept small (vs. the real repo's
    # roi_size=7/out_channels=10/sampling_ratio=2 defaults) purely to bound the
    # PSRoIMaxAlignPooling2D pure-Python double loop's iteration count for a fast
    # random-init trace; this changes only pooling granularity, not the architecture.
    return LightHeadRCNN(n_class=5, roi_size=3, n_rois=1, out_channels=4, sampling_ratio=1)


def example_input_light_head_rcnn():
    torch.manual_seed(0)
    # Small spatial size: the dilated C5 stage keeps feature-map resolution close to
    # C4 (no extra downsampling), so ResNet101 compute on CPU scales steeply with input
    # size. 64x64 is enough for every layer (7x7/stride-2 conv1 + 3x3/stride-2 pool1 +
    # three stride-2 stages) to produce non-degenerate feature maps for a random-init
    # architectural trace.
    return (torch.randn(1, 3, 64, 64),)


MENAGERIE_ENTRIES = [
    ("Light-Head R-CNN", "build_light_head_rcnn", "example_input_light_head_rcnn", 2017, "ported"),
]
