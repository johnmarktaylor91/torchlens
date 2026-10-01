# SOURCE: vendored from dipjyoti92/StarGAN-Voice-Conversion-2 @ master
# https://raw.githubusercontent.com/dipjyoti92/StarGAN-Voice-Conversion-2/master/model.py
#
# StarGAN-VC2 (Interspeech 2019): Kaneko et al., "StarGAN-VC2: Rethinking
# Conditional Methods for StarGAN-Based Voice Conversion". Community PyTorch
# reference implementation (official NTT page has no GitHub release).
#
# Vendored verbatim (only trivial edits: removed the `device`-pinning default
# in ConditionalInstanceNormalisation so the module is instantiable/traceable
# on whatever device the caller uses, and dropped the `if __name__ == "__main__"`
# training-script smoke test block, which imports a data_loader module we do
# not vendor). Architecture is untouched.

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class ConditionalInstanceNormalisation(nn.Module):
    """CIN Block."""

    def __init__(self, in_channel, n_speakers):
        super(ConditionalInstanceNormalisation, self).__init__()

        self.dim_in = in_channel
        self.gamma = nn.Linear(n_speakers, in_channel)
        self.beta = nn.Linear(n_speakers, in_channel)

    def forward(self, x, c):
        u = torch.mean(x, dim=2, keepdim=True)
        var = torch.mean((x - u) * (x - u), dim=2, keepdim=True)
        std = torch.sqrt(var + 1e-8)

        gamma = self.gamma(c)
        gamma = gamma.view(-1, self.dim_in, 1)
        beta = self.beta(c)
        beta = beta.view(-1, self.dim_in, 1)

        h = (x - u) / std
        h = h * gamma + beta

        return h


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, in_channel, out_channel, n_speakers):
        super(ResidualBlock, self).__init__()
        self.conv_1 = nn.Conv1d(
            in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.cin_1 = ConditionalInstanceNormalisation(out_channel, n_speakers * 2)
        self.relu_1 = nn.GLU(dim=1)

    def forward(self, x, c):
        x = self.conv_1(x)
        x = self.cin_1(x, c)
        x = self.relu_1(x)

        return x


class Down2d_initial(nn.Module):
    """docstring for Down2d."""

    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(Down2d_initial, self).__init__()

        self.c1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding
        )
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.glu(x1)

        return x1


class Down2d(nn.Module):
    """docstring for Down2d."""

    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(Down2d, self).__init__()

        self.c1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding
        )
        self.n1 = nn.InstanceNorm2d(out_channel, affine=True, track_running_stats=True)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)
        x1 = self.glu(x1)

        return x1


class Up2d(nn.Module):
    """docstring for Up2d."""

    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(Up2d, self).__init__()
        self.c1 = nn.ConvTranspose2d(
            in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding
        )
        self.n1 = nn.InstanceNorm2d(out_channel, affine=True, track_running_stats=True)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)
        x1 = self.glu(x1)

        return x1


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, num_speakers=70, repeat_num=6):
        super(Generator, self).__init__()
        self.downsample = nn.Sequential(
            Down2d_initial(1, 128, (3, 9), (1, 1), (1, 4)),
            Down2d(64, 256, (4, 8), (2, 2), (1, 3)),
            Down2d(128, 512, (4, 8), (2, 2), (1, 3)),
        )

        # Down-conversion layers.
        self.down_conversion = nn.Sequential(
            nn.Conv1d(
                in_channels=2304,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.InstanceNorm1d(num_features=256, affine=True),
        )

        # Bottleneck layers.
        self.residual_1 = ResidualBlock(in_channel=256, out_channel=512, n_speakers=num_speakers)
        self.residual_2 = ResidualBlock(in_channel=256, out_channel=512, n_speakers=num_speakers)
        self.residual_3 = ResidualBlock(in_channel=256, out_channel=512, n_speakers=num_speakers)
        self.residual_4 = ResidualBlock(in_channel=256, out_channel=512, n_speakers=num_speakers)
        self.residual_5 = ResidualBlock(in_channel=256, out_channel=512, n_speakers=num_speakers)
        self.residual_6 = ResidualBlock(in_channel=256, out_channel=512, n_speakers=num_speakers)
        self.residual_7 = ResidualBlock(in_channel=256, out_channel=512, n_speakers=num_speakers)
        self.residual_8 = ResidualBlock(in_channel=256, out_channel=512, n_speakers=num_speakers)
        self.residual_9 = ResidualBlock(in_channel=256, out_channel=512, n_speakers=num_speakers)

        # Up-conversion layers.
        self.up_conversion = nn.Conv1d(
            in_channels=256,
            out_channels=2304,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.up1 = Up2d(256, 256, (4, 4), (2, 2), (1, 1))
        self.up2 = Up2d(128, 128, (4, 4), (2, 2), (1, 1))

        self.deconv = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False
        )

    def forward(self, x, con1, con2):
        c = torch.cat((con1, con2), dim=1)
        width_size = x.size(3)
        x = self.downsample(x)
        x = x.contiguous().view(-1, 2304, width_size // 4)
        x = self.down_conversion(x)

        x = self.residual_1(x, c)
        x = self.residual_2(x, c)
        x = self.residual_3(x, c)
        x = self.residual_4(x, c)
        x = self.residual_5(x, c)
        x = self.residual_6(x, c)
        x = self.residual_7(x, c)
        x = self.residual_8(x, c)
        x = self.residual_9(x, c)

        x = self.up_conversion(x)
        x = x.view(-1, 256, 9, width_size // 4)

        x = self.up1(x)
        x = self.up2(x)
        x = self.deconv(x)

        return x


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, input_size=(36, 512), conv_dim=64, repeat_num=5, num_speakers=70):
        super(Discriminator, self).__init__()

        self.downsample = nn.Sequential(
            Down2d_initial(1, 128, (3, 9), (1, 1), (1, 4)),
            Down2d(64, 256, (3, 3), (2, 2), (1, 1)),
            Down2d(128, 512, (3, 3), (2, 2), (1, 1)),
            Down2d(256, 1024, (3, 3), (2, 2), (1, 1)),
            Down2d(512, 1024, (1, 5), (1, 1), (0, 2)),
        )

        self.fc = nn.Linear(in_features=512, out_features=1)
        self.proj = nn.Linear(num_speakers * 2, 512)

    def forward(self, x, con1, con2):
        c = torch.cat((con1, con2), dim=1)
        x = self.downsample(x)

        h = torch.sum(x, dim=(2, 3))

        x = self.fc(h)

        p = self.proj(c)

        x += torch.sum(p * h, dim=1, keepdim=True)

        return x


# --- Menagerie staging wrappers -------------------------------------------------
#
# The real Generator/Discriminator take a mel-cepstrum-like [B, 1, 36, T] input
# (T must be a multiple of 4, per `width_size // 4` bottleneck reshape) plus two
# one-hot speaker-label conditioning vectors. num_speakers is shrunk to a tiny
# value for a fast trace; architecture is unmodified.

_NUM_SPEAKERS = 4
_MEL_BINS = 36
_TIME_STEPS = 16  # multiple of 4


def build_stargan_vc2_generator():
    return Generator(num_speakers=_NUM_SPEAKERS)


def example_input_stargan_vc2_generator():
    x = torch.randn(1, 1, _MEL_BINS, _TIME_STEPS)
    con1 = torch.eye(_NUM_SPEAKERS)[[0]]
    con2 = torch.eye(_NUM_SPEAKERS)[[1]]
    return x, con1, con2


def build_stargan_vc2_discriminator():
    return Discriminator(num_speakers=_NUM_SPEAKERS)


def example_input_stargan_vc2_discriminator():
    x = torch.randn(1, 1, _MEL_BINS, _TIME_STEPS)
    con1 = torch.eye(_NUM_SPEAKERS)[[0]]
    con2 = torch.eye(_NUM_SPEAKERS)[[1]]
    return x, con1, con2


MENAGERIE_ENTRIES = [
    (
        "StarGAN-VC2 Generator",
        build_stargan_vc2_generator,
        example_input_stargan_vc2_generator,
        2019,
        "CODE",
    ),
    (
        "StarGAN-VC2 Discriminator",
        build_stargan_vc2_discriminator,
        example_input_stargan_vc2_discriminator,
        2019,
        "CODE",
    ),
]
