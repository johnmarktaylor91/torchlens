# SOURCE: vendored from Mawiszus/TOAD-GAN @ master
# (models/generator.py, models/discriminator.py, models/conv_block.py)
# https://github.com/Mawiszus/TOAD-GAN -- "TOAD-GAN: Coherent Style Level
# Generation from a Single Example" (Awiszus, Schubert, Rosenhahn, AIIDE 2020).
# The architecture (itself adapted, per the repo's own header comment, from
# tamarott/SinGAN's single-image GAN pyramid) is a multi-scale patch-based
# conditional WGAN-GP: `Level_GeneratorConcatSkip2CleanAdd` is a small fully
# convolutional generator that adds a residual correction to an upsampled
# coarser-scale level and applies a softmax over token channels (so outputs
# are valid categorical-token level maps), and `Level_WDiscriminator` is the
# matching patch critic; both are built from a shared `ConvBlock`
# (Conv2d + BatchNorm2d + LeakyReLU) stack. `ConvBlock`,
# `Level_GeneratorConcatSkip2CleanAdd`, and `Level_WDiscriminator` are
# transcribed verbatim; only the `opt` namespace (an argparse Namespace in the
# original CLI) is replaced with a plain Python object exposing the same
# attributes (`nc_current`, `nfc`, `ker_size`, `num_layer`), and generator
# forward now infers cropping directly from the head/body/tail output shape
# instead of the original CLI's externally-padded-`y` calling convention.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---- verbatim from models/conv_block.py ----
class ConvBlock(nn.Sequential):
    """Conv block containing Conv2d, BatchNorm2d and LeakyReLU Layers."""

    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super().__init__()
        (
            self.add_module(
                "conv",
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=ker_size,
                    stride=stride,
                    padding=padd,
                ),
            ),
        )
        (self.add_module("norm", nn.BatchNorm2d(out_channel)),)
        self.add_module("LeakyRelu", nn.LeakyReLU(0.2, inplace=True))


# ---- verbatim from models/generator.py ----
class Level_GeneratorConcatSkip2CleanAdd(nn.Module):
    """Patch based Generator. Uses namespace opt."""

    def __init__(self, opt):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_current, N, (opt.ker_size, opt.ker_size), 0, 1)
        self.body = nn.Sequential()

        for i in range(opt.num_layer - 2):
            block = ConvBlock(N, N, (opt.ker_size, opt.ker_size), 0, 1)
            self.body.add_module("block%d" % (i + 1), block)

        block = ConvBlock(N, N, (opt.ker_size, opt.ker_size), 0, 1)
        self.body.add_module("block%d" % (opt.num_layer - 2), block)

        self.tail = nn.Sequential(
            nn.Conv2d(
                N, opt.nc_current, kernel_size=(opt.ker_size, opt.ker_size), stride=1, padding=0
            )
        )

    def forward(self, x, y, temperature=1):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = F.softmax(x * temperature, dim=1)
        ind = int((y.shape[2] - x.shape[2]) / 2)
        y = y[:, :, ind : (y.shape[2] - ind), ind : (y.shape[3] - ind)]
        return x + y


# ---- verbatim from models/discriminator.py ----
class Level_WDiscriminator(nn.Module):
    """Patch based Discriminator. Uses Namespace opt."""

    def __init__(self, opt):
        super().__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_current, N, (opt.ker_size, opt.ker_size), 0, 1)
        self.body = nn.Sequential()

        for i in range(opt.num_layer - 2):
            block = ConvBlock(N, N, (opt.ker_size, opt.ker_size), 0, 1)
            self.body.add_module("block%d" % (i + 1), block)

        block = ConvBlock(N, N, (opt.ker_size, opt.ker_size), 0, 1)
        self.body.add_module("block%d" % (opt.num_layer - 2), block)

        self.tail = nn.Conv2d(N, 1, kernel_size=(opt.ker_size, opt.ker_size), stride=1, padding=0)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class _Opt:
    """Plain stand-in for the original argparse Namespace `opt`, exposing the
    same attributes the real generator/discriminator constructors read."""

    def __init__(self, nc_current, nfc, ker_size, num_layer):
        self.nc_current = nc_current
        self.nfc = nfc
        self.ker_size = ker_size
        self.num_layer = num_layer


# ---- staging build/example helpers (tiny sizes for fast tracing) ----
def build_toadgan_generator():
    torch.manual_seed(0)
    opt = _Opt(nc_current=12, nfc=16, ker_size=3, num_layer=3)
    model = Level_GeneratorConcatSkip2CleanAdd(opt)
    model.eval()
    return model


def example_input_toadgan_generator():
    torch.manual_seed(0)
    batch_size = 2
    n_tokens = 12
    h = w = 24
    x = torch.randn(batch_size, n_tokens, h, w)
    y = torch.randn(batch_size, n_tokens, h, w)
    return (x, y)


def build_toadgan_discriminator():
    torch.manual_seed(0)
    opt = _Opt(nc_current=12, nfc=16, ker_size=3, num_layer=3)
    model = Level_WDiscriminator(opt)
    model.eval()
    return model


def example_input_toadgan_discriminator():
    torch.manual_seed(0)
    batch_size = 2
    n_tokens = 12
    h = w = 24
    x = torch.randn(batch_size, n_tokens, h, w)
    return (x,)


MENAGERIE_ENTRIES = [
    (
        "TOAD-GAN Generator",
        build_toadgan_generator,
        example_input_toadgan_generator,
        2020,
        "vendored-pytorch",
    ),
    (
        "TOAD-GAN Discriminator",
        build_toadgan_discriminator,
        example_input_toadgan_discriminator,
        2020,
        "vendored-pytorch",
    ),
]
