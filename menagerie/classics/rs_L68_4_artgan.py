# FAITHFUL PORT of cs-chan/ArtGAN @ master (original framework: TensorFlow 1.x
# + neon DataLoader, Python 2)
# File ported: ArtGAN/Genre128GANAE.py (+ ArtGAN/nn/layers.py primitives it
# calls: conv2d, batchnorm, nnupsampling, pool, linear, flatten, gaussnoise)
# https://github.com/cs-chan/ArtGAN
#
# Why ported rather than vendored/rung-1: the real repo is TF1.x
# (tf.variable_scope/tf.get_variable/tf.contrib.layers.batch_norm,
# tf.cond-based train/eval branching) with Python-2-only `print` statements
# and a hard dependency on the (now-defunct) Nervana `neon` backend for data
# loading -- none of this can run, or be reasonably installed, in a modern
# base torch environment. This is a line-for-line architecture transcription
# from the actual `discriminator`/`generator` functions and the `nn/layers.py`
# primitives they call, translated to PyTorch with equivalent semantics:
#   - `conv2d(x, nout, kernel, strides, pad='SAME')` (TF SAME padding, no
#     bias unless use_b=True, weights ~ N(0, std^2)) -> `nn.Conv2d(...,
#     padding=kernel//2, bias=use_b)` -- TF's `SAME` padding for odd
#     kernel/stride-1-or-2 convs used here is exactly `kernel//2` symmetric
#     padding, matching every call site (kernel=3 throughout).
#   - `batchnorm(x, is_training, name)` (`tf.contrib.layers.batch_norm`,
#     decay=0.9, center=True, scale=True, NCHW) -> `nn.BatchNorm2d(nout,
#     momentum=1-0.9=0.1)` (TF `decay` and PyTorch `momentum` are
#     complementary conventions for the same EMA update).
#   - `nnupsampling(x, size)` (`tf.image.resize_nearest_neighbor`) ->
#     `nn.Upsample(size=..., mode='nearest')` (identical nearest-neighbor
#     resize semantics).
#   - `pool(x, fsize, strides, op='avg')` -> `nn.AvgPool2d(fsize, strides,
#     padding=...)` with TF SAME padding replicated via explicit padding
#     for the odd 3x3/stride-2 case used in the generator's final
#     `g6b_64 = pool(g6b, fsize=3, strides=2, op='avg')` downsample.
#   - `linear(x, nout)` -> `nn.Linear(in, nout, bias=use_b)`.
#   - `lrelu(x, leak)` -> `nn.LeakyReLU(leak)`.
#   - `gaussnoise(x, std)` (additive N(0, std^2) noise on the discriminator
#     input, train-time only) -> applied unconditionally in `forward`
#     (matches the original's unconditional call at the top of
#     `discriminator()`; the original script only ever traces the graph
#     once with noise baked in, there is no is_train-gated noise branch in
#     the source).
#   - Dropped entirely: data loading (`neon` backend, `train_loader`/
#     `validation_loader`, `OneHot`), image I/O (`imageio`, `drawblock`),
#     the training loop, loss functions (`log_sum_exp`, softmax-CE,
#     softplus GAN losses, MSE reconstruction terms), and optimizers -- none
#     of these are part of the network architecture.
#   - `discriminator()`'s classifier head (`clspred`) and pixel-decoder head
#     (`g5b`, an auto-encoder-style reconstruction branch back to image
#     space) are BOTH kept, matching the source's dual-output
#     `Opred_n, recon_n = discriminator(x_n)` -- ArtGAN's core contribution
#     (ICIP'16/TIP'19) is exactly this: a GAN discriminator that is also an
#     autoencoder+classifier ("categorical label + decoder-autoencoder
#     branch" per the queue notes), trained jointly with the generator.
#
# Architecture (faithfully transcribed from source): ArtGAN (Genre-128
# configuration). `Generator`: a DCGAN-style decoder from a
# concat[z(100), one-hot class(10)] latent through a Linear projection to a
# 4x4x512 feature map, then 6 stages of (nearest-upsample -> conv3x3 ->
# batchnorm -> leaky-relu) doubling resolution 4->8->16->32->64->128,
# terminating in a tanh conv to 3 channels at 128x128, average-pooled down to
# a 64x64 output (`recon_64`) alongside the full-res `recon_128`.
# `Discriminator/Encoder`: a mirror-image conv encoder (5 conv-stride-2
# stages 64->32->16->8->4, with dropout+batchnorm+leaky-relu, plus a bonus
# stride-1 conv at 8x8) feeding two heads off the same 4x4x1024 bottleneck:
# a linear classifier (`clspred`, n_classes logits) and a decoder-autoencoder
# branch (`g5b`, mirroring the Generator's upsample stack) that reconstructs
# a 64x64x3 image -- ArtGAN's distinguishing "decoder-autoencoder branch on
# the discriminator" design.

import torch
from torch import nn
import torch.nn.functional as F


def _same_pad(kernel):
    # TF 'SAME' padding for odd kernels / stride in {1, 2} used throughout
    # this network reduces to symmetric kernel//2 padding.
    return kernel // 2


class Conv2dSame(nn.Module):
    """conv2d(x, nout, kernel, strides, pad='SAME') from nn/layers.py."""

    def __init__(self, in_ch, nout, kernel=3, strides=1, use_b=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, nout, kernel_size=kernel, stride=strides, padding=_same_pad(kernel), bias=use_b
        )
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.02)
        if use_b:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class NNUpsampling(nn.Module):
    """nnupsampling(inp, size) -> tf.image.resize_nearest_neighbor."""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size, mode="nearest")


class Discriminator(nn.Module):
    """`discriminator()` in Genre128GANAE.py: shared Encoder producing a
    classifier head (clspred) and a decoder-autoencoder head (g5b)."""

    def __init__(self, n_classes=10, dropout=0.2, gauss_std=0.05):
        super().__init__()
        self.gauss_std = gauss_std
        self.dropout_p = dropout

        # Encoder conv stack (64 -> 32 -> 16 -> 8 -> 4)
        self.conv1 = Conv2dSame(3, 128, kernel=3, strides=2)
        self.conv2 = Conv2dSame(128, 256, kernel=3, strides=2)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.1)
        self.conv3 = Conv2dSame(256, 512, kernel=3, strides=2)
        self.bn3 = nn.BatchNorm2d(512, momentum=0.1)
        self.conv3b = Conv2dSame(512, 512, kernel=3, strides=1)
        self.bn3b = nn.BatchNorm2d(512, momentum=0.1)
        self.conv4 = Conv2dSame(512, 1024, kernel=3, strides=2)
        self.bn4 = nn.BatchNorm2d(1024, momentum=0.1)

        self.lrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        # Classifier head
        self.cpred = nn.Linear(1024 * 4 * 4, n_classes)

        # Decoder-autoencoder head (mirrors Generator's upsample stack,
        # always run in train-mode batchnorm per source: `is_training=tf.constant(True)`)
        self.dg1 = Conv2dSame(1024, 512, kernel=3, strides=1)
        self.dbn1 = nn.BatchNorm2d(512, momentum=0.1)
        self.up2 = NNUpsampling((8, 8))
        self.dg2 = Conv2dSame(512, 256, kernel=3, strides=1)
        self.dbn2 = nn.BatchNorm2d(256, momentum=0.1)
        self.up3 = NNUpsampling((16, 16))
        self.dg3 = Conv2dSame(256, 128, kernel=3, strides=1)
        self.dbn3 = nn.BatchNorm2d(128, momentum=0.1)
        self.up4 = NNUpsampling((32, 32))
        self.dg4 = Conv2dSame(128, 64, kernel=3, strides=1)
        self.dbn4 = nn.BatchNorm2d(64, momentum=0.1)
        self.up5 = NNUpsampling((64, 64))
        self.dg5 = Conv2dSame(64, 32, kernel=3, strides=1)
        self.dbn5 = nn.BatchNorm2d(32, momentum=0.1)
        self.dg5b = Conv2dSame(32, 3, kernel=3, strides=1)

    def forward(self, x):
        x = x + torch.randn_like(x) * self.gauss_std  # gaussnoise

        c1 = self.lrelu(self.conv1(x))
        c2 = self.dropout(c1)
        c2 = self.lrelu(self.bn2(self.conv2(c2)))
        c3 = self.dropout(c2)
        c3 = self.lrelu(self.bn3(self.conv3(c3)))
        c3b = self.lrelu(self.bn3b(self.conv3b(c3)))
        c4 = self.dropout(c3b)
        c4 = self.lrelu(self.bn4(self.conv4(c4)))

        flat = c4.reshape(c4.size(0), -1)
        clspred = self.cpred(flat)

        g1 = self.lrelu(self.dbn1(self.dg1(c4)))
        g2 = self.up2(g1)
        g2 = self.lrelu(self.dbn2(self.dg2(g2)))
        g3 = self.up3(g2)
        g3 = self.lrelu(self.dbn3(self.dg3(g3)))
        g4 = self.up4(g3)
        g4 = self.lrelu(self.dbn4(self.dg4(g4)))
        g5 = self.up5(g4)
        g5 = self.lrelu(self.dbn5(self.dg5(g5)))
        g5b = torch.tanh(self.dg5b(g5))

        return clspred, g5b


class Generator(nn.Module):
    """`generator()` in Genre128GANAE.py: z + one-hot class -> 128x128 image
    (+ average-pooled 64x64 version)."""

    def __init__(self, zdim=100, n_classes=10):
        super().__init__()
        sz = 4
        self.sz = sz
        self.fc = nn.Linear(zdim + n_classes, 512 * sz * sz)
        self.bn1 = nn.BatchNorm1d(512 * sz * sz, momentum=0.1)
        self.lrelu = nn.LeakyReLU(0.2)

        self.up2 = NNUpsampling((8, 8))
        self.g2 = Conv2dSame(512, 512, kernel=3, strides=1)
        self.bn2 = nn.BatchNorm2d(512, momentum=0.1)

        self.up3 = NNUpsampling((16, 16))
        self.g3 = Conv2dSame(512, 256, kernel=3, strides=1)
        self.bn3 = nn.BatchNorm2d(256, momentum=0.1)

        self.up4 = NNUpsampling((32, 32))
        self.g4 = Conv2dSame(256, 128, kernel=3, strides=1)
        self.bn4 = nn.BatchNorm2d(128, momentum=0.1)

        self.up5 = NNUpsampling((64, 64))
        self.g5 = Conv2dSame(128, 64, kernel=3, strides=1)
        self.bn5 = nn.BatchNorm2d(64, momentum=0.1)

        self.g5b = Conv2dSame(64, 64, kernel=3, strides=1)
        self.bn5b = nn.BatchNorm2d(64, momentum=0.1)

        self.up6 = NNUpsampling((128, 128))
        self.g6 = Conv2dSame(64, 32, kernel=3, strides=1)
        self.bn6 = nn.BatchNorm2d(32, momentum=0.1)
        self.g6b = Conv2dSame(32, 3, kernel=3, strides=1)

        self.avg_pool_64 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, z, y_onehot):
        inp = torch.cat([z, y_onehot], dim=1)
        g1 = self.lrelu(self.bn1(self.fc(inp)))
        g1 = g1.reshape(-1, 512, self.sz, self.sz)

        g2 = self.up2(g1)
        g2 = self.lrelu(self.bn2(self.g2(g2)))

        g3 = self.up3(g2)
        g3 = self.lrelu(self.bn3(self.g3(g3)))

        g4 = self.up4(g3)
        g4 = self.lrelu(self.bn4(self.g4(g4)))

        g5 = self.up5(g4)
        g5 = self.lrelu(self.bn5(self.g5(g5)))

        g5b = self.lrelu(self.bn5b(self.g5b(g5)))

        g6 = self.up6(g5b)
        g6 = self.lrelu(self.bn6(self.g6(g6)))
        g6b = torch.tanh(self.g6b(g6))

        g6b_64 = self.avg_pool_64(g6b)
        return g6b_64, g6b


# ---------------------------------------------------------------------------
# Menagerie harness
# ---------------------------------------------------------------------------


class ArtGAN(nn.Module):
    """Wires Generator + Discriminator matching the source's real forward
    graph construction: `Opred_n, recon_n = discriminator(x_n);
    samples, samples128 = generator(z, iny); Opred_g, recon_g =
    discriminator(samples, reuse=True)` -- both discriminator calls share
    weights (reuse=True in source), matching a single nn.Module instance
    called twice here."""

    def __init__(self, zdim=100, n_classes=10):
        super().__init__()
        self.generator = Generator(zdim=zdim, n_classes=n_classes)
        self.discriminator = Discriminator(n_classes=n_classes)

    def forward(self, x_real, z, y_onehot):
        clspred_real, recon_real = self.discriminator(x_real)
        samples_64, samples_128 = self.generator(z, y_onehot)
        # Source: `Opred_g, recon_g = discriminator(samples, reuse=True)` --
        # `samples` is the 64x64 avg-pooled generator output (matching x_n's
        # 64x64 shape, the only size the shared discriminator's im_size=[64,64]
        # encoder/cpred head is built for), not the full-res `samples128`.
        clspred_fake, recon_fake = self.discriminator(samples_64)
        return clspred_real, recon_real, samples_64, samples_128, clspred_fake, recon_fake


def build_artgan():
    return ArtGAN(zdim=16, n_classes=4).eval()


def example_input_artgan():
    batch_size = 2
    n_classes = 4
    zdim = 16
    x_real = torch.randn(batch_size, 3, 64, 64)
    z = torch.rand(batch_size, zdim) * 2 - 1
    y_onehot = torch.eye(n_classes)[torch.randint(0, n_classes, (batch_size,))]
    return (x_real, z, y_onehot)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "ArtGAN (Genre-128, GAN + decoder-autoencoder discriminator for WikiArt)",
        "build_artgan",
        "example_input_artgan",
        2017,
        "ported",
    ),
]
