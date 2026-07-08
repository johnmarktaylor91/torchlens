# FAITHFUL PORT of lisjin/dcan-tensorflow @ master (original framework: TensorFlow 1.x)
# https://raw.githubusercontent.com/lisjin/dcan-tensorflow/master/tf-dcan/bbbc006.py
# (inference())
#
# Chen, Qi, Yu, Heng 2016 (AAAI) "DCAN: Deep Contour-Aware Networks for Accurate Gland
# Segmentation" -- this repo (lisjin/dcan-tensorflow) is the confirmed official TF1.x
# implementation, applied to the BBBC006 nuclei dataset. TF1.x's `tf.layers`/
# `tf.variable_scope`/`tf.app.flags` graph-mode API is not available/installable in this
# base torch env, so this is a faithful architectural transcription of the real
# `inference()` function into base-env torch. DCAN's defining trait -- a shared VGG-style
# encoder trunk feeding TWO parallel deep-supervision decoder branches (contours + object
# segments), each branch fusing multiple resolution levels via transposed-conv upsampling
# -- is reproduced exactly:
#
#   feat_out = feat_root (32); in_layer = images
#   for layer in range(num_layers=6):
#       feat_out *= 2 if layer != 4 else 1          # 64,128,256,256,512,1024
#       conv = ReLU(Conv2d(in_layer, feat_out, 3x3, padding='same'))   # 'conv{layer+1}'
#       if training and layer > 3: conv = Dropout(conv, rate=0.5)      # layers 5,6 only
#       if layer > 0: in_layer = MaxPool2d(conv, 2, 2, padding='same')  # layer 1 has no pool
#       else: in_layer = conv
#       if layer > 2:   # layers 4,5,6 (index 3,4,5) feed the two deep-supervision decoders
#           for branch in (contours, segments):
#               w = bilinear-interpolation-initialized deconv filter, kernel=2*dc, stride=dc
#               deconv = ReLU(ConvTranspose2d(in_layer, w, bias, stride=dc) -> full image size)
#               output = ReLU(Conv2d(deconv, num_classes, 1x1, padding='same'))
#               branch_outputs.append(output)
#           dc *= 2   # deconv_root doubles each qualifying layer: 8, 16, 32
#   c_fuse = sum(contours branch outputs); s_fuse = sum(segments branch outputs)
#
# `get_deconv_filter` (bilinear-interpolation init, borrowed by the original authors from
# https://github.com/MarvinTeichmann/tensorflow-fcn) is reproduced verbatim as a torch
# tensor initializer for the `ConvTranspose2d` weight. 'same'-padding conv2d/max_pool2d in
# TF1 (odd kernel/stride=1, or stride=2 kernel=2) is exactly torch's padding=kernel//2 (conv)
# / ceil-mode pooling for the sizes used here. The real code's transposed-conv output shape
# (`ds`) is hardcoded to the dataset's full IMAGE_HEIGHT/IMAGE_WIDTH, which for a `dc`-stride
# `2*dc`-kernel 'SAME' conv2d_transpose is algebraically equivalent to exactly
# `stride * input_size` (each branch's `dc` is exactly the encoder's cumulative downsampling
# factor at that layer, so `stride * input_size == full image size`); we reproduce that with
# `ConvTranspose2d(kernel=2*dc, stride=dc, padding=dc//2)`, which yields output size exactly
# `stride * input_size` with no dynamic output-shape override needed. Dropout is disabled here
# since `build_dcan()` constructs the module in eval mode (matching `inference(images,
# train=False)`), matching the real function's own `train` flag gate. No architectural
# changes; loss/training-op/summary code (not part of the network itself) is omitted.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bilinear_deconv_weight(kernel_size, num_classes, feat_in):
    # Faithful port of get_deconv_filter(shape=[kh, kw, num_classes, feat_in]) --
    # TF conv2d_transpose kernel layout is [kh, kw, out_ch, in_ch]; torch ConvTranspose2d
    # weight layout is [in_ch, out_ch, kh, kw]. We build the same bilinear kernel values.
    width = kernel_size
    height = kernel_size
    f = math.ceil(width / 2.0)
    c = (2.0 * f - 1 - f % 2) / (2.0 * f)

    bilinear = torch.zeros(kernel_size, kernel_size)
    for x in range(width):
        for y in range(height):
            bilinear[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))

    weight = torch.zeros(feat_in, num_classes, kernel_size, kernel_size)
    for i in range(num_classes):
        weight[:, i, :, :] = bilinear
    return weight


class _ConvBlock(nn.Module):
    """One 'convN' stage: Conv2d(3x3, same) -> ReLU (+ optional dropout on layers 5/6)."""

    def __init__(self, in_channels, out_channels, use_dropout):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.relu(self.conv(x))
        if self.use_dropout and self.training:
            x = self.dropout(x)
        return x


class _DeconvBranchHead(nn.Module):
    """One deep-supervision branch head at a given encoder layer: bilinear-init
    ConvTranspose2d up to full image resolution, bias+ReLU, then a 1x1 conv2d to
    num_classes+ReLU (real 'deconv{layer+1}_{i}' + 'output{layer+1}_{i}')."""

    def __init__(self, feat_in, num_classes, dc):
        super().__init__()
        kernel_size = dc * 2
        self.dc = dc
        # TF1 'SAME'-padding conv2d_transpose(stride=dc, kernel=2*dc) with an explicit output
        # shape of exactly stride*input_size is equivalent to torch's ConvTranspose2d with
        # symmetric padding = (kernel - stride) // 2 = dc // 2, which yields output size
        # exactly stride*input_size with no need for a runtime output_size= override.
        pad = (kernel_size - dc) // 2
        self.deconv = nn.ConvTranspose2d(
            feat_in, num_classes, kernel_size=kernel_size, stride=dc, padding=pad, bias=True
        )
        with torch.no_grad():
            self.deconv.weight.copy_(_bilinear_deconv_weight(kernel_size, num_classes, feat_in))
            self.deconv.bias.fill_(0.1)
        self.deconv_relu = nn.ReLU(inplace=True)
        self.output_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1, padding=0)
        self.output_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        deconv = self.deconv(x)
        deconv = self.deconv_relu(deconv)
        out = self.output_conv(deconv)
        out = self.output_relu(out)
        return out


class DCAN(nn.Module):
    """Faithful port of bbbc006.inference()."""

    def __init__(self, num_layers=6, feat_root=32, deconv_root=8, num_classes=2, in_channels=1):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes

        conv_blocks = []
        feat_out = feat_root
        in_ch = in_channels
        channels_by_layer = []
        for layer in range(num_layers):
            feat_out = feat_out * 2 if layer != 4 else feat_out
            conv_blocks.append(_ConvBlock(in_ch, feat_out, use_dropout=(layer > 3)))
            channels_by_layer.append(feat_out)
            in_ch = feat_out
        self.conv_blocks = nn.ModuleList(conv_blocks)

        contour_heads = []
        segment_heads = []
        dc = deconv_root
        for layer in range(num_layers):
            if layer > 2:
                feat_in = channels_by_layer[layer]
                contour_heads.append(_DeconvBranchHead(feat_in, num_classes, dc))
                segment_heads.append(_DeconvBranchHead(feat_in, num_classes, dc))
                dc *= 2
        self.contour_heads = nn.ModuleList(contour_heads)
        self.segment_heads = nn.ModuleList(segment_heads)

    def forward(self, images):
        in_layer = images
        c_outputs = []
        s_outputs = []
        head_idx = 0
        for layer in range(self.num_layers):
            conv = self.conv_blocks[layer](in_layer)
            if layer > 0:
                in_layer = F.max_pool2d(conv, kernel_size=2, stride=2, ceil_mode=True)
            else:
                in_layer = conv

            if layer > 2:
                c_outputs.append(self.contour_heads[head_idx](in_layer))
                s_outputs.append(self.segment_heads[head_idx](in_layer))
                head_idx += 1

        c_fuse = torch.stack(c_outputs, dim=0).sum(dim=0)
        s_fuse = torch.stack(s_outputs, dim=0).sum(dim=0)
        return c_fuse, s_fuse


def build_dcan():
    model = DCAN(num_layers=6, feat_root=4, deconv_root=8, num_classes=2, in_channels=1)
    model.eval()
    return model


def example_input_dcan():
    # Real dataset images are 520x696 (single-channel); 32 divides both after 5 poolings
    # (layers 2-6). Use a much smaller multiple of 32 for a tiny trace.
    return (torch.randn(1, 1, 64, 96),)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("DCAN", "build_dcan", "example_input_dcan", 2016, "ported"),
]
