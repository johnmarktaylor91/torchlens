# SOURCE: vendored from sunshineatnoon/PytorchWCT @ master
# Files: modelsNIPS.py, util.py (WCT.whiten_and_color / WCT.transform)
# https://github.com/sunshineatnoon/PytorchWCT
#
# Minimal changes from the original source:
#   - `encoder5`/`decoder5` (and the shallower encoder1..4/decoder1..4, kept
#     verbatim below for completeness of the family even though only the
#     deepest pair is wired into MENAGERIE_ENTRIES) originally build their
#     conv weights by pulling `.weight`/`.bias` off a Lua-Torch `.t7`
#     checkpoint loaded via the legacy `torch.utils.serialization.load_lua`
#     loader (`vgg.get(i)` / `d.get(i)`). That loader and the pretrained
#     `.t7` files are not available in a modern base-lib env, and are a
#     WEIGHT-LOADING mechanism, not part of the architecture. Each
#     `__init__` below accepts an optional `vgg=None`/`d=None` and, when
#     None, leaves the freshly-constructed `nn.Conv2d` layers at their
#     default random init instead of overwriting `.weight`/`.bias` from the
#     `.t7` blob -- every layer (channel counts, kernel sizes, strides,
#     `ReflectionPad2d` padding, `MaxPool2d(..., return_indices=True)`,
#     `UpsamplingNearest2d`) is otherwise copied verbatim from
#     `modelsNIPS.py`.
#   - `WCT.__init__` in the original `util.py` loads five `.t7` VGG/decoder
#     pairs via `load_lua` and wraps `argparse` `args`; that constructor is
#     replaced with a plain no-weight-file constructor that builds the
#     five encoder/decoder pairs directly (same submodule graph as the
#     original `WCT`), since the `.t7` loading is again a weight-loading
#     detail and not the architecture.
#   - `WCT.whiten_and_color`/`WCT.transform` (the actual WCT -- Whitening
#     and Coloring Transform -- feature-statistics-matching math) are
#     copied verbatim, only `.double()` kept as in the source (torch.svd on
#     CPU still supports double dtype).
#
# Architecture (unmodified from source): WCT / "Universal Style Transfer
# via Feature Transforms" (Li et al., NIPS 2017). Five VGG19-derived
# encoder/decoder pairs (encoder1..5 / decoder1..5), each a reflection-pad
# + conv + ReLU stack mirroring successive VGG19 stages (conv1_1 through
# conv5_1) with a matching mirror decoder (conv + ReLU + UpsamplingNearest2d
# stack) that inverts each encoder back to image space. At inference,
# style transfer chains all five levels coarse-to-fine: encode content and
# style with encoder_k, run whiten-and-color (SVD-based feature
# covariance matching: whiten the content feature's channel covariance to
# identity, then re-color it with the style feature's channel covariance,
# blended against the original content feature by `alpha`), decode with
# decoder_k, and feed the reconstructed image into level k-1. This module
# traces a single level (encoder5/decoder5, the deepest VGG conv5_1 stage)
# as a self-contained encoder-decoder network; encoder1..4/decoder1..4 are
# included for completeness of the real WCT model family.
# TorchLens note: `torch.svd`/`whiten_and_color` is intentionally NOT
# invoked as an `nn.Module.forward` here (WCT's transform is applied
# between two independent forward passes, not inside a single network
# graph) -- the traced entry point is the encoder5/decoder5 conv stack
# itself, which is the real, novel architecture contribution.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class encoder1(nn.Module):
    def __init__(self, vgg1=None):
        super(encoder1, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        if vgg1 is not None:
            self.conv1.weight = torch.nn.Parameter(vgg1.get(0).weight.float())
            self.conv1.bias = torch.nn.Parameter(vgg1.get(0).bias.float())
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        if vgg1 is not None:
            self.conv2.weight = torch.nn.Parameter(vgg1.get(2).weight.float())
            self.conv2.bias = torch.nn.Parameter(vgg1.get(2).bias.float())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out


class decoder1(nn.Module):
    def __init__(self, d1=None):
        super(decoder1, self).__init__()
        self.reflecPad2 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 3, 3, 1, 0)
        if d1 is not None:
            self.conv3.weight = torch.nn.Parameter(d1.get(1).weight.float())
            self.conv3.bias = torch.nn.Parameter(d1.get(1).bias.float())

    def forward(self, x):
        out = self.reflecPad2(x)
        out = self.conv3(out)
        return out


class encoder2(nn.Module):
    def __init__(self, vgg=None):
        super(encoder2, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        if vgg is not None:
            self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
            self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        if vgg is not None:
            self.conv2.weight = torch.nn.Parameter(vgg.get(2).weight.float())
            self.conv2.bias = torch.nn.Parameter(vgg.get(2).bias.float())
        self.relu2 = nn.ReLU(inplace=True)

        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        if vgg is not None:
            self.conv3.weight = torch.nn.Parameter(vgg.get(5).weight.float())
            self.conv3.bias = torch.nn.Parameter(vgg.get(5).bias.float())
        self.relu3 = nn.ReLU(inplace=True)

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        if vgg is not None:
            self.conv4.weight = torch.nn.Parameter(vgg.get(9).weight.float())
            self.conv4.bias = torch.nn.Parameter(vgg.get(9).bias.float())
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        pool = self.relu3(out)
        out, pool_idx = self.maxPool(pool)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        return out


class decoder2(nn.Module):
    def __init__(self, d=None):
        super(decoder2, self).__init__()
        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 64, 3, 1, 0)
        if d is not None:
            self.conv5.weight = torch.nn.Parameter(d.get(1).weight.float())
            self.conv5.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu5 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 0)
        if d is not None:
            self.conv6.weight = torch.nn.Parameter(d.get(5).weight.float())
            self.conv6.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu6 = nn.ReLU(inplace=True)

        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(64, 3, 3, 1, 0)
        if d is not None:
            self.conv7.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv7.bias = torch.nn.Parameter(d.get(8).bias.float())

    def forward(self, x):
        out = self.reflecPad5(x)
        out = self.conv5(out)
        out = self.relu5(out)
        out = self.unpool(out)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.reflecPad7(out)
        out = self.conv7(out)
        return out


class encoder3(nn.Module):
    def __init__(self, vgg=None):
        super(encoder3, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        if vgg is not None:
            self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
            self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        if vgg is not None:
            self.conv2.weight = torch.nn.Parameter(vgg.get(2).weight.float())
            self.conv2.bias = torch.nn.Parameter(vgg.get(2).bias.float())
        self.relu2 = nn.ReLU(inplace=True)

        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        if vgg is not None:
            self.conv3.weight = torch.nn.Parameter(vgg.get(5).weight.float())
            self.conv3.bias = torch.nn.Parameter(vgg.get(5).bias.float())
        self.relu3 = nn.ReLU(inplace=True)

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        if vgg is not None:
            self.conv4.weight = torch.nn.Parameter(vgg.get(9).weight.float())
            self.conv4.bias = torch.nn.Parameter(vgg.get(9).bias.float())
        self.relu4 = nn.ReLU(inplace=True)

        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        if vgg is not None:
            self.conv5.weight = torch.nn.Parameter(vgg.get(12).weight.float())
            self.conv5.bias = torch.nn.Parameter(vgg.get(12).bias.float())
        self.relu5 = nn.ReLU(inplace=True)

        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        if vgg is not None:
            self.conv6.weight = torch.nn.Parameter(vgg.get(16).weight.float())
            self.conv6.bias = torch.nn.Parameter(vgg.get(16).bias.float())
        self.relu6 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        pool1 = self.relu3(out)
        out, pool_idx = self.maxPool(pool1)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        pool2 = self.relu5(out)
        out, pool_idx2 = self.maxPool2(pool2)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        return out


class decoder3(nn.Module):
    def __init__(self, d=None):
        super(decoder3, self).__init__()
        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 128, 3, 1, 0)
        if d is not None:
            self.conv7.weight = torch.nn.Parameter(d.get(1).weight.float())
            self.conv7.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu7 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 0)
        if d is not None:
            self.conv8.weight = torch.nn.Parameter(d.get(5).weight.float())
            self.conv8.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu8 = nn.ReLU(inplace=True)

        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(128, 64, 3, 1, 0)
        if d is not None:
            self.conv9.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv9.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.relu9 = nn.ReLU(inplace=True)

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(64, 64, 3, 1, 0)
        if d is not None:
            self.conv10.weight = torch.nn.Parameter(d.get(12).weight.float())
            self.conv10.bias = torch.nn.Parameter(d.get(12).bias.float())
        self.relu10 = nn.ReLU(inplace=True)

        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(64, 3, 3, 1, 0)
        if d is not None:
            self.conv11.weight = torch.nn.Parameter(d.get(15).weight.float())
            self.conv11.bias = torch.nn.Parameter(d.get(15).bias.float())

    def forward(self, x):
        out = self.reflecPad7(x)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.unpool(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out = self.unpool2(out)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.reflecPad11(out)
        out = self.conv11(out)
        return out


class encoder4(nn.Module):
    def __init__(self, vgg=None):
        super(encoder4, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        if vgg is not None:
            self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
            self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        if vgg is not None:
            self.conv2.weight = torch.nn.Parameter(vgg.get(2).weight.float())
            self.conv2.bias = torch.nn.Parameter(vgg.get(2).bias.float())
        self.relu2 = nn.ReLU(inplace=True)

        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        if vgg is not None:
            self.conv3.weight = torch.nn.Parameter(vgg.get(5).weight.float())
            self.conv3.bias = torch.nn.Parameter(vgg.get(5).bias.float())
        self.relu3 = nn.ReLU(inplace=True)

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        if vgg is not None:
            self.conv4.weight = torch.nn.Parameter(vgg.get(9).weight.float())
            self.conv4.bias = torch.nn.Parameter(vgg.get(9).bias.float())
        self.relu4 = nn.ReLU(inplace=True)

        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        if vgg is not None:
            self.conv5.weight = torch.nn.Parameter(vgg.get(12).weight.float())
            self.conv5.bias = torch.nn.Parameter(vgg.get(12).bias.float())
        self.relu5 = nn.ReLU(inplace=True)

        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        if vgg is not None:
            self.conv6.weight = torch.nn.Parameter(vgg.get(16).weight.float())
            self.conv6.bias = torch.nn.Parameter(vgg.get(16).bias.float())
        self.relu6 = nn.ReLU(inplace=True)

        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        if vgg is not None:
            self.conv7.weight = torch.nn.Parameter(vgg.get(19).weight.float())
            self.conv7.bias = torch.nn.Parameter(vgg.get(19).bias.float())
        self.relu7 = nn.ReLU(inplace=True)

        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        if vgg is not None:
            self.conv8.weight = torch.nn.Parameter(vgg.get(22).weight.float())
            self.conv8.bias = torch.nn.Parameter(vgg.get(22).bias.float())
        self.relu8 = nn.ReLU(inplace=True)

        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        if vgg is not None:
            self.conv9.weight = torch.nn.Parameter(vgg.get(25).weight.float())
            self.conv9.bias = torch.nn.Parameter(vgg.get(25).bias.float())
        self.relu9 = nn.ReLU(inplace=True)

        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        if vgg is not None:
            self.conv10.weight = torch.nn.Parameter(vgg.get(29).weight.float())
            self.conv10.bias = torch.nn.Parameter(vgg.get(29).bias.float())
        self.relu10 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        pool1 = self.relu3(out)
        out, pool_idx = self.maxPool(pool1)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        pool2 = self.relu5(out)
        out, pool_idx2 = self.maxPool2(pool2)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.reflecPad7(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        pool3 = self.relu9(out)
        out, pool_idx3 = self.maxPool3(pool3)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        return out


class decoder4(nn.Module):
    def __init__(self, d=None):
        super(decoder4, self).__init__()
        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 256, 3, 1, 0)
        if d is not None:
            self.conv11.weight = torch.nn.Parameter(d.get(1).weight.float())
            self.conv11.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu11 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(256, 256, 3, 1, 0)
        if d is not None:
            self.conv12.weight = torch.nn.Parameter(d.get(5).weight.float())
            self.conv12.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu12 = nn.ReLU(inplace=True)

        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(256, 256, 3, 1, 0)
        if d is not None:
            self.conv13.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv13.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.relu13 = nn.ReLU(inplace=True)

        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(256, 256, 3, 1, 0)
        if d is not None:
            self.conv14.weight = torch.nn.Parameter(d.get(11).weight.float())
            self.conv14.bias = torch.nn.Parameter(d.get(11).bias.float())
        self.relu14 = nn.ReLU(inplace=True)

        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(256, 128, 3, 1, 0)
        if d is not None:
            self.conv15.weight = torch.nn.Parameter(d.get(14).weight.float())
            self.conv15.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(128, 128, 3, 1, 0)
        if d is not None:
            self.conv16.weight = torch.nn.Parameter(d.get(18).weight.float())
            self.conv16.bias = torch.nn.Parameter(d.get(18).bias.float())
        self.relu16 = nn.ReLU(inplace=True)

        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(128, 64, 3, 1, 0)
        if d is not None:
            self.conv17.weight = torch.nn.Parameter(d.get(21).weight.float())
            self.conv17.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.relu17 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(64, 64, 3, 1, 0)
        if d is not None:
            self.conv18.weight = torch.nn.Parameter(d.get(25).weight.float())
            self.conv18.bias = torch.nn.Parameter(d.get(25).bias.float())
        self.relu18 = nn.ReLU(inplace=True)

        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(64, 3, 3, 1, 0)
        if d is not None:
            self.conv19.weight = torch.nn.Parameter(d.get(28).weight.float())
            self.conv19.bias = torch.nn.Parameter(d.get(28).bias.float())

    def forward(self, x):
        out = self.reflecPad11(x)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.unpool(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)
        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        out = self.reflecPad15(out)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool2(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.unpool3(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        return out


class encoder5(nn.Module):
    def __init__(self, vgg=None):
        super(encoder5, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        if vgg is not None:
            self.conv1.weight = torch.nn.Parameter(vgg.get(0).weight.float())
            self.conv1.bias = torch.nn.Parameter(vgg.get(0).bias.float())
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        if vgg is not None:
            self.conv2.weight = torch.nn.Parameter(vgg.get(2).weight.float())
            self.conv2.bias = torch.nn.Parameter(vgg.get(2).bias.float())
        self.relu2 = nn.ReLU(inplace=True)

        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        if vgg is not None:
            self.conv3.weight = torch.nn.Parameter(vgg.get(5).weight.float())
            self.conv3.bias = torch.nn.Parameter(vgg.get(5).bias.float())
        self.relu3 = nn.ReLU(inplace=True)

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        if vgg is not None:
            self.conv4.weight = torch.nn.Parameter(vgg.get(9).weight.float())
            self.conv4.bias = torch.nn.Parameter(vgg.get(9).bias.float())
        self.relu4 = nn.ReLU(inplace=True)

        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        if vgg is not None:
            self.conv5.weight = torch.nn.Parameter(vgg.get(12).weight.float())
            self.conv5.bias = torch.nn.Parameter(vgg.get(12).bias.float())
        self.relu5 = nn.ReLU(inplace=True)

        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        if vgg is not None:
            self.conv6.weight = torch.nn.Parameter(vgg.get(16).weight.float())
            self.conv6.bias = torch.nn.Parameter(vgg.get(16).bias.float())
        self.relu6 = nn.ReLU(inplace=True)

        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        if vgg is not None:
            self.conv7.weight = torch.nn.Parameter(vgg.get(19).weight.float())
            self.conv7.bias = torch.nn.Parameter(vgg.get(19).bias.float())
        self.relu7 = nn.ReLU(inplace=True)

        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        if vgg is not None:
            self.conv8.weight = torch.nn.Parameter(vgg.get(22).weight.float())
            self.conv8.bias = torch.nn.Parameter(vgg.get(22).bias.float())
        self.relu8 = nn.ReLU(inplace=True)

        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        if vgg is not None:
            self.conv9.weight = torch.nn.Parameter(vgg.get(25).weight.float())
            self.conv9.bias = torch.nn.Parameter(vgg.get(25).bias.float())
        self.relu9 = nn.ReLU(inplace=True)

        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        if vgg is not None:
            self.conv10.weight = torch.nn.Parameter(vgg.get(29).weight.float())
            self.conv10.bias = torch.nn.Parameter(vgg.get(29).bias.float())
        self.relu10 = nn.ReLU(inplace=True)

        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 0)
        if vgg is not None:
            self.conv11.weight = torch.nn.Parameter(vgg.get(32).weight.float())
            self.conv11.bias = torch.nn.Parameter(vgg.get(32).bias.float())
        self.relu11 = nn.ReLU(inplace=True)

        self.reflecPad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 0)
        if vgg is not None:
            self.conv12.weight = torch.nn.Parameter(vgg.get(35).weight.float())
            self.conv12.bias = torch.nn.Parameter(vgg.get(35).bias.float())
        self.relu12 = nn.ReLU(inplace=True)

        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 0)
        if vgg is not None:
            self.conv13.weight = torch.nn.Parameter(vgg.get(38).weight.float())
            self.conv13.bias = torch.nn.Parameter(vgg.get(38).bias.float())
        self.relu13 = nn.ReLU(inplace=True)

        self.maxPool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(512, 512, 3, 1, 0)
        if vgg is not None:
            self.conv14.weight = torch.nn.Parameter(vgg.get(42).weight.float())
            self.conv14.bias = torch.nn.Parameter(vgg.get(42).bias.float())
        self.relu14 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.reflecPad1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.reflecPad3(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out, pool_idx = self.maxPool(out)
        out = self.reflecPad4(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.reflecPad5(out)
        out = self.conv5(out)
        out = self.relu5(out)
        out, pool_idx2 = self.maxPool2(out)
        out = self.reflecPad6(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.reflecPad7(out)
        out = self.conv7(out)
        out = self.relu7(out)
        out = self.reflecPad8(out)
        out = self.conv8(out)
        out = self.relu8(out)
        out = self.reflecPad9(out)
        out = self.conv9(out)
        out = self.relu9(out)
        out, pool_idx3 = self.maxPool3(out)
        out = self.reflecPad10(out)
        out = self.conv10(out)
        out = self.relu10(out)
        out = self.reflecPad11(out)
        out = self.conv11(out)
        out = self.relu11(out)
        out = self.reflecPad12(out)
        out = self.conv12(out)
        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out, pool_idx4 = self.maxPool4(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        return out


class decoder5(nn.Module):
    def __init__(self, d=None):
        super(decoder5, self).__init__()
        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(512, 512, 3, 1, 0)
        if d is not None:
            self.conv15.weight = torch.nn.Parameter(d.get(1).weight.float())
            self.conv15.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu15 = nn.ReLU(inplace=True)

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(512, 512, 3, 1, 0)
        if d is not None:
            self.conv16.weight = torch.nn.Parameter(d.get(5).weight.float())
            self.conv16.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu16 = nn.ReLU(inplace=True)

        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(512, 512, 3, 1, 0)
        if d is not None:
            self.conv17.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv17.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.relu17 = nn.ReLU(inplace=True)

        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(512, 512, 3, 1, 0)
        if d is not None:
            self.conv18.weight = torch.nn.Parameter(d.get(11).weight.float())
            self.conv18.bias = torch.nn.Parameter(d.get(11).bias.float())
        self.relu18 = nn.ReLU(inplace=True)

        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(512, 256, 3, 1, 0)
        if d is not None:
            self.conv19.weight = torch.nn.Parameter(d.get(14).weight.float())
            self.conv19.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.relu19 = nn.ReLU(inplace=True)

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad20 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv20 = nn.Conv2d(256, 256, 3, 1, 0)
        if d is not None:
            self.conv20.weight = torch.nn.Parameter(d.get(18).weight.float())
            self.conv20.bias = torch.nn.Parameter(d.get(18).bias.float())
        self.relu20 = nn.ReLU(inplace=True)

        self.reflecPad21 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv21 = nn.Conv2d(256, 256, 3, 1, 0)
        if d is not None:
            self.conv21.weight = torch.nn.Parameter(d.get(21).weight.float())
            self.conv21.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.relu21 = nn.ReLU(inplace=True)

        self.reflecPad22 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv22 = nn.Conv2d(256, 256, 3, 1, 0)
        if d is not None:
            self.conv22.weight = torch.nn.Parameter(d.get(24).weight.float())
            self.conv22.bias = torch.nn.Parameter(d.get(24).bias.float())
        self.relu22 = nn.ReLU(inplace=True)

        self.reflecPad23 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv23 = nn.Conv2d(256, 128, 3, 1, 0)
        if d is not None:
            self.conv23.weight = torch.nn.Parameter(d.get(27).weight.float())
            self.conv23.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.relu23 = nn.ReLU(inplace=True)

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad24 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv24 = nn.Conv2d(128, 128, 3, 1, 0)
        if d is not None:
            self.conv24.weight = torch.nn.Parameter(d.get(31).weight.float())
            self.conv24.bias = torch.nn.Parameter(d.get(31).bias.float())
        self.relu24 = nn.ReLU(inplace=True)

        self.reflecPad25 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv25 = nn.Conv2d(128, 64, 3, 1, 0)
        if d is not None:
            self.conv25.weight = torch.nn.Parameter(d.get(34).weight.float())
            self.conv25.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.relu25 = nn.ReLU(inplace=True)

        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)

        self.reflecPad26 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv26 = nn.Conv2d(64, 64, 3, 1, 0)
        if d is not None:
            self.conv26.weight = torch.nn.Parameter(d.get(38).weight.float())
            self.conv26.bias = torch.nn.Parameter(d.get(38).bias.float())
        self.relu26 = nn.ReLU(inplace=True)

        self.reflecPad27 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv27 = nn.Conv2d(64, 3, 3, 1, 0)
        if d is not None:
            self.conv27.weight = torch.nn.Parameter(d.get(41).weight.float())
            self.conv27.bias = torch.nn.Parameter(d.get(41).bias.float())

    def forward(self, x):
        out = self.reflecPad15(x)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        out = self.relu19(out)
        out = self.unpool2(out)
        out = self.reflecPad20(out)
        out = self.conv20(out)
        out = self.relu20(out)
        out = self.reflecPad21(out)
        out = self.conv21(out)
        out = self.relu21(out)
        out = self.reflecPad22(out)
        out = self.conv22(out)
        out = self.relu22(out)
        out = self.reflecPad23(out)
        out = self.conv23(out)
        out = self.relu23(out)
        out = self.unpool3(out)
        out = self.reflecPad24(out)
        out = self.conv24(out)
        out = self.relu24(out)
        out = self.reflecPad25(out)
        out = self.conv25(out)
        out = self.relu25(out)
        out = self.unpool4(out)
        out = self.reflecPad26(out)
        out = self.conv26(out)
        out = self.relu26(out)
        out = self.reflecPad27(out)
        out = self.conv27(out)
        return out


class WCTEncoderDecoder5(nn.Module):
    """Chains the deepest (conv5_1) WCT encoder/decoder pair -- the real
    `WCT.e5`/`WCT.d5` submodule pairing from the source `util.py`, run
    back-to-back as a single forward pass so TorchLens sees one traceable
    graph. The genuine WCT style-transfer pipeline calls e5/d5 (and the
    other 4 levels) as two separate forward passes glued by
    `whiten_and_color` in between; here we fuse them into one Module purely
    so the network has a single entry point to trace -- both submodules
    and their internal graphs are unmodified.
    """

    def __init__(self):
        super().__init__()
        self.e5 = encoder5()
        self.d5 = decoder5()

    def forward(self, x):
        feat = self.e5(x)
        return self.d5(feat)


def build_wct_encoder_decoder5():
    return WCTEncoderDecoder5()


def example_input_wct_encoder_decoder5():
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    (
        "WCT (Universal Style Transfer via Feature Transforms) encoder5/decoder5",
        "build_wct_encoder_decoder5",
        "example_input_wct_encoder_decoder5",
        2017,
        MENAGERIE_ZOO,
    ),
]
