# FAITHFUL PORT of bloc97/Anime4K @ master (original framework: GLSL fragment shaders,
# mpv/libplacebo `.glsl` custom-shader format -- no PyTorch/Python model code exists in the
# repo at all; the only Python present is `tensorflow/shaderutils.py`/`utils.py`, training
# utilities, not a model definition).
#
# The "CNN" variants (glsl/Upscale/Anime4K_Upscale_CNN_x2_*.glsl) are frozen-weight small
# conv nets whose exact topology is fully specified by consecutive `!DESC`/`!HOOK`/`!BIND`/
# `!SAVE` shader passes, each pass literally containing a `mat4`-encoded 3x3 (or 1x1)
# convolution over the previous pass's output, e.g. (Anime4K_Upscale_CNN_x2_M.glsl):
#   pass 1: Conv2d(3->4, k=3, pad=1) on MAIN (the RGB input)                 "Conv-4x3x3x3"
#   pass 2-7 (x6): Conv2d(4->4, k=3, pad=1), each applied to the *split*
#                  pos/neg-relu activation of the previous pass's 4 channels  "Conv-4x3x3x8"
#                  (go_0 = relu(prev), go_1 = relu(-prev) -> 8 effective input channels)
#   pass 8: Conv2d(8*7=56 -> 4, k=1) over the concatenated pos/neg-relu       "Conv-4x1x1x56"
#           activations of ALL 7 previous conv passes (dense skip connections)
#   pass 9: depth-to-space (PixelShuffle x2) of the final 4-channel map into
#           a 1-channel x2 upsampled map, added residually to the bicubic-
#           upsampled MAIN input                                             "Depth-to-Space"
# This module transcribes that exact topology (conv channel counts, split-relu nonlinearity,
# dense skip-concat into the last 1x1 conv, PixelShuffle head, bicubic residual) faithfully
# into a self-contained torch.nn.Module. Weights are randomly initialized (menagerie captures
# architecture at tiny random-init size, not the shipped GLSL float literals) using the
# "M" (medium) variant's channel width (4 intermediate channels) as the concrete instance;
# the S/L/UL/VL variants only change this one width constant per the shader family.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class _SplitReLUConv(nn.Module):
    """One Anime4K conv pass: takes a 4-channel input, applies split pos/neg ReLU to
    double it to 8 effective channels (matching the GLSL go_0/go_1 texture-fetch macros),
    then a Conv2d(8->out_ch, k=3, pad=1). The very first pass instead takes the raw
    3-channel RGB input directly (no preceding relu-split)."""

    def __init__(self, in_ch, out_ch, split_relu=True):
        super().__init__()
        self.split_relu = split_relu
        conv_in = in_ch * 2 if split_relu else in_ch
        self.conv = nn.Conv2d(conv_in, out_ch, kernel_size=3, stride=1, padding=1, bias=True)

    def _prep(self, x):
        if not self.split_relu:
            return x
        pos = F.relu(x)
        neg = F.relu(-x)
        return torch.cat([pos, neg], dim=1)

    def forward(self, x):
        return self.conv(self._prep(x))


class Anime4KCNNUpscale(nn.Module):
    """Faithful port of the Anime4K-v3.2 Upscale-CNN-x2 (M) GLSL shader pipeline."""

    def __init__(self, base_channels=4):
        super().__init__()
        c = base_channels
        # pass 1: RGB(3) -> c, no split-relu on the raw input
        self.conv1 = _SplitReLUConv(3, c, split_relu=False)
        # passes 2-7: c -> c, each preceded by split-relu of the previous pass output
        self.conv2 = _SplitReLUConv(c, c, split_relu=True)
        self.conv3 = _SplitReLUConv(c, c, split_relu=True)
        self.conv4 = _SplitReLUConv(c, c, split_relu=True)
        self.conv5 = _SplitReLUConv(c, c, split_relu=True)
        self.conv6 = _SplitReLUConv(c, c, split_relu=True)
        self.conv7 = _SplitReLUConv(c, c, split_relu=True)
        # pass 8: dense 1x1 conv over split-relu(pass1..pass7) concatenated (7 * 2c channels)
        self.conv_last = nn.Conv2d(7 * 2 * c, c, kernel_size=1, stride=1, padding=0, bias=True)
        # pass 9: depth-to-space x2 (c must be divisible by 4 for a x2 PixelShuffle to 1 channel;
        # the real shader packs 4 sub-pixel values per output pixel from the c=4 channel map)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        bicubic = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)

        p1 = self.conv1(x)
        p2 = self.conv2(p1)
        p3 = self.conv3(p2)
        p4 = self.conv4(p3)
        p5 = self.conv5(p4)
        p6 = self.conv6(p5)
        p7 = self.conv7(p6)

        def split(t):
            return torch.cat([F.relu(t), F.relu(-t)], dim=1)

        dense = torch.cat(
            [split(p1), split(p2), split(p3), split(p4), split(p5), split(p6), split(p7)], dim=1
        )
        last = self.conv_last(dense)

        # Depth-to-Space pass: the shader picks ONE of the 4 learned channels per output
        # sub-pixel via `i0.y * 2 + i0.x` (exactly nn.PixelShuffle(2)'s channel->space
        # mapping on a 4-channel map), yielding a single-channel x2-upsampled correction
        # map `c0`, then broadcasts that scalar to all 4 output components
        # (`vec4(c0, c0, c0, c0)`) and adds it to every channel of the bicubic-upsampled
        # MAIN input residually.
        correction = self.pixel_shuffle(last)  # (B, 1, 2H, 2W)
        correction = correction.expand(-1, bicubic.shape[1], -1, -1)
        return bicubic + correction


def build_anime4k():
    model = Anime4KCNNUpscale(base_channels=4)
    model.eval()
    return model


def example_input_anime4k():
    return torch.randn(1, 3, 16, 16)


MENAGERIE_ENTRIES = [
    (
        "Anime4K CNN Upscale (v3.2, M)",
        "build_anime4k",
        "example_input_anime4k",
        2019,
        MENAGERIE_ZOO,
    ),
]
