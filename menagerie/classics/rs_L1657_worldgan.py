# SOURCE: vendored from https://github.com/Mawiszus/World-GAN @ main
#
# Official repo (Awiszus, Trebing & Risi, IEEE CoG 2021, "World-GAN: a
# Generative Model for Minecraft Worlds"). Extends TOAD-GAN's 2D SinGAN-style
# progressive-scale patch GAN to 3D voxel worlds (block2vec token embeddings
# are a separate pretraining stage, not part of this generator/discriminator
# graph). Vendored verbatim from models/conv_block.py, models/generator.py,
# models/discriminator.py: `ConvBlock` (dim=2 or dim=3 Conv+Norm+LeakyReLU),
# `Level_WDiscriminator` (patch-based Wasserstein critic), and
# `Level_GeneratorConcatSkip2CleanAdd` (patch-based generator with a skip
# connection that concat-adds cropped conditioning input `y` to the head/body/
# tail conv-block output `x`, softmax-temperature block-token output). Only
# the unrelated Minecraft I/O (block2vec/, level_renderer.py, Blender export
# scripts) and training loop (train.py, train_single_scale.py) are dropped --
# the model files themselves import only torch.
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    """Conv block containing Conv2d, BatchNorm2d and LeakyReLU Layers."""

    def __init__(self, in_channel, out_channel, ker_size, padd, stride, dim=2):
        super().__init__()
        if dim == 2:
            self.add_module(
                "conv",
                nn.Conv2d(
                    in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd
                ),
            )
            self.add_module("norm", nn.BatchNorm2d(out_channel))
        elif dim == 3:
            self.add_module(
                "conv",
                nn.Conv3d(
                    in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd
                ),
            )
            self.add_module("norm", nn.BatchNorm3d(out_channel))
        else:
            raise NotImplementedError("Can only make 2D or 3D Conv Layers.")

        self.add_module("LeakyRelu", nn.LeakyReLU(0.2, inplace=True))


class Level_WDiscriminator(nn.Module):
    """Patch based Discriminator. Uses Namespace opt."""

    def __init__(self, opt):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        dim = len(opt.level_shape)
        kernel = tuple(opt.ker_size for _ in range(dim))
        self.head = ConvBlock(opt.nc_current, N, kernel, 0, 1, dim)  # Padding is done externally
        self.body = nn.Sequential()

        for i in range(opt.num_layer - 2):
            block = ConvBlock(N, N, kernel, 0, 1, dim)
            self.body.add_module("block%d" % (i + 1), block)

        block = ConvBlock(N, N, kernel, 0, 1, dim)
        self.body.add_module("block%d" % (opt.num_layer - 2), block)

        if dim == 2:
            self.tail = nn.Conv2d(N, 1, kernel_size=kernel, stride=1, padding=0)
        elif dim == 3:
            self.tail = nn.Conv3d(N, 1, kernel_size=kernel, stride=1, padding=0)
        else:
            raise NotImplementedError("Can only make 2D or 3D Conv Layers.")

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class Level_GeneratorConcatSkip2CleanAdd(nn.Module):
    """Patch based Generator. Uses namespace opt."""

    def __init__(self, opt, use_softmax=True):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        self.use_softmax = use_softmax
        N = int(opt.nfc)
        dim = len(opt.level_shape)
        kernel = tuple(opt.ker_size for _ in range(dim))
        self.head = ConvBlock(opt.nc_current, N, kernel, 0, 1, dim)  # Padding is done externally
        self.body = nn.Sequential()

        for i in range(opt.num_layer - 2):
            block = ConvBlock(N, N, kernel, 0, 1, dim)
            self.body.add_module("block%d" % (i + 1), block)

        block = ConvBlock(N, N, kernel, 0, 1, dim)
        self.body.add_module("block%d" % (opt.num_layer - 2), block)

        if dim == 2:
            if use_softmax:
                self.tail = nn.Sequential(
                    nn.Conv2d(N, opt.nc_current, kernel_size=kernel, stride=1, padding=0)
                )
            else:
                self.tail = nn.Sequential(
                    nn.Conv2d(N, opt.nc_current, kernel_size=kernel, stride=1, padding=0),
                    # nn.ReLU()
                )
        elif dim == 3:
            if use_softmax:
                self.tail = nn.Sequential(
                    nn.Conv3d(N, opt.nc_current, kernel_size=kernel, stride=1, padding=0)
                )
            else:
                self.tail = nn.Sequential(
                    nn.Conv3d(N, opt.nc_current, kernel_size=kernel, stride=1, padding=0),
                    # nn.ReLU()
                )
        else:
            raise NotImplementedError("Can only make 2D or 3D Conv Layers.")

    def forward(self, x, y, temperature=1):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        if self.use_softmax:
            x = F.softmax(
                x * temperature, dim=1
            )  # Softmax is added here to allow for the temperature parameter
        ind = int((y.shape[2] - x.shape[2]) / 2)
        if len(y.shape) == 4:
            y = y[:, :, ind : (y.shape[2] - ind), ind : (y.shape[3] - ind)]
        elif len(y.shape) == 5:
            y = y[
                :, :, ind : (y.shape[2] - ind), ind : (y.shape[3] - ind), ind : (y.shape[4] - ind)
            ]
        else:
            raise NotImplementedError("only supports 4D or 5D tensors")

        return x + y


# --- staging harness: build + example input ---------------------------------


class _Opt:
    """Minimal stand-in for World-GAN's argparse Namespace `opt` -- only the
    fields Level_GeneratorConcatSkip2CleanAdd/Level_WDiscriminator read.
    """

    def __init__(self, nfc, ker_size, num_layer, nc_current, level_shape):
        self.nfc = nfc
        self.ker_size = ker_size
        self.num_layer = num_layer
        self.nc_current = nc_current
        self.level_shape = level_shape


class WorldGANGeneratorTraceWrapper(nn.Module):
    """The real Level_GeneratorConcatSkip2CleanAdd already takes plain tensors
    (x, y); this wrapper only fixes `temperature=1` as a constant so the
    example_input is a flat 2-tensor tuple, exercising the identical
    forward path (softmax block-token generation + skip-add).
    """

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x, y):
        return self.generator(x, y, temperature=1)


def _make_opt():
    # Shrunk from the paper's default (nfc=64, num_layer=5) to a tiny
    # architecturally faithful 3D voxel-GAN config: 3x3x3 kernels, 3 conv
    # layers, 12 block-token channels (matches the repo's Minecraft level-1
    # default nc_current=12), 3D level_shape (dim=3 branch).
    return _Opt(nfc=8, ker_size=3, num_layer=3, nc_current=12, level_shape=(1, 1, 1))


def build_worldgan_generator():
    opt = _make_opt()
    generator = Level_GeneratorConcatSkip2CleanAdd(opt, use_softmax=True)
    return WorldGANGeneratorTraceWrapper(generator).eval()


def example_input_worldgan_generator():
    batch, channels = 1, 12
    # x: noise map at this scale's spatial resolution (pre-conv, gets
    # cropped by the 3 unpadded 3x3x3 conv blocks: -2 per block = -6 total).
    x = torch.randn(batch, channels, 14, 14, 14)
    # y: previous-scale (upsampled) block-token map, cropped internally to
    # match the generator's (unpadded) spatial output before the skip-add.
    y = torch.randn(batch, channels, 14, 14, 14)
    return (x, y)


def build_worldgan_discriminator():
    opt = _make_opt()
    return Level_WDiscriminator(opt).eval()


def example_input_worldgan_discriminator():
    batch, channels = 1, 12
    x = torch.randn(batch, channels, 14, 14, 14)
    return (x,)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "WorldGAN-Generator",
        build_worldgan_generator,
        example_input_worldgan_generator,
        2021,
        MENAGERIE_ZOO,
    ),
    (
        "WorldGAN-Discriminator",
        build_worldgan_discriminator,
        example_input_worldgan_discriminator,
        2021,
        MENAGERIE_ZOO,
    ),
]
