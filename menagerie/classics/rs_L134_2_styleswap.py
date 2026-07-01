# SOURCE: vendored from irasin/Pytorch_Style_Swap @ master
#   https://raw.githubusercontent.com/irasin/Pytorch_Style_Swap/master/model.py
#   https://raw.githubusercontent.com/irasin/Pytorch_Style_Swap/master/style_swap.py
#   https://raw.githubusercontent.com/irasin/Pytorch_Style_Swap/master/test.py
#
# Chen & Schmidt 2016 "Fast Patch-based Style Transfer of Arbitrary Style"
# (a.k.a. "Style Swap") originally shipped as Lua/Torch7 code
# (rtqichen/style-swap, the paper's official repo). irasin/Pytorch_Style_Swap
# is a faithful PyTorch re-implementation of the same architecture: a
# VGG19-feature encoder, a patch-swap operation performed via conv2d/
# conv_transpose2d on normalized style-feature patches, and a small
# ReflectionPad2d+Conv2d decoder that inverts the swapped feature map back
# to pixel space -- copied VERBATIM below (VGGEncoder, RC, Decoder,
# style_swap), only removing `pretrained=True` (no network weight
# download for a tiny random-init trace) and wiring VGGEncoder -> style_swap
# -> Decoder into one nn.Module exactly matching test.py's
# `cf = e(c); sf = e(s); style_swap_res = style_swap(cf, sf, p, 1); out = d(style_swap_res)`
# forward pipeline.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# model.py (verbatim, minus pretrained=True)
# ---------------------------------------------------------------------------
class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=None).features
        self.slice1 = vgg[:2]
        self.slice2 = vgg[2:7]
        self.slice3 = vgg[7:12]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        return h3


class RC(nn.Module):
    """A wrapper of ReflectionPad2d and Conv2d"""

    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1, activated=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.activated = activated

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        if self.activated:
            return F.relu(h)
        else:
            return h


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc1 = RC(256, 128, 3, 1)
        self.rc2 = RC(128, 128, 3, 1)
        self.rc3 = RC(128, 64, 3, 1)
        self.rc4 = RC(64, 64, 3, 1)
        self.rc5 = RC(64, 3, 3, 1, False)

    def forward(self, features):
        h = self.rc1(features)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc2(h)
        h = self.rc3(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc4(h)
        h = self.rc5(h)
        return h


# ---------------------------------------------------------------------------
# style_swap.py (verbatim)
# ---------------------------------------------------------------------------
def style_swap(content_feature, style_feature, kernel_size, stride=1):
    # content_feature and style_feature should have shape as (1, C, H, W)
    # kernel_size here is equivalent to extracted patch size
    kh, kw = kernel_size, kernel_size
    sh, sw = stride, stride

    patches = style_feature.unfold(2, kh, sh).unfold(3, kw, sw)

    patches = patches.permute(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(-1, *patches.shape[-3:])  # (patch_numbers, C, kh, kw)

    # calculate Frobenius norm and normalize the patches at each filter
    norm = torch.norm(patches.reshape(patches.shape[0], -1), dim=1).reshape(-1, 1, 1, 1)

    noramalized_patches = patches / norm

    conv_out = F.conv2d(content_feature, noramalized_patches)

    # calculate the argmax at each spatial location, which means at each (kh, kw),
    # there should exist a filter which provides the biggest value of the output
    one_hots = torch.zeros_like(conv_out)
    one_hots.scatter_(1, conv_out.argmax(dim=1, keepdim=True), 1)

    # deconv/transpose conv
    deconv_out = F.conv_transpose2d(one_hots, patches)

    # calculate the overlap from deconv/transpose conv
    overlap = F.conv_transpose2d(one_hots, torch.ones_like(patches))

    # average the deconv result
    res = deconv_out / overlap
    return res


# ---------------------------------------------------------------------------
# End-to-end pipeline, matching test.py's forward wiring:
#   cf = e(c); sf = e(s); style_swap_res = style_swap(cf, sf, p, 1); out = d(style_swap_res)
# ---------------------------------------------------------------------------
class StyleSwapModel(nn.Module):
    def __init__(self, patch_size=3):
        super().__init__()
        self.encoder = VGGEncoder()
        self.decoder = Decoder()
        self.patch_size = patch_size

    def forward(self, content, style):
        cf = self.encoder(content)
        sf = self.encoder(style)
        swapped = style_swap(cf, sf, self.patch_size, 1)
        out = self.decoder(swapped)
        return out


def build_styleswap():
    return StyleSwapModel(patch_size=3)


def example_input_styleswap():
    torch.manual_seed(0)
    content = torch.rand(1, 3, 64, 64)
    style = torch.rand(1, 3, 64, 64)
    return (content, style)


MENAGERIE_ENTRIES = [
    (
        "Style Swap (Fast Patch-based Style Transfer)",
        "build_styleswap",
        "example_input_styleswap",
        2016,
        "vendored-pytorch",
    ),
]
