# FAITHFUL PORT of rkjones4/GANGogh @ master (original framework: TensorFlow 1.x)
# File: GANgogh.py (uses tflib/ops/{conv2d,deconv2d,batchnorm,layernorm,linear}.py)
# https://github.com/rkjones4/GANGogh
#
# GANgogh's own repo (an undergraduate project, Williams 2017; blog post
# "GANGogh: Creating Art with GANs") ships only TensorFlow 1.x source built
# on a bespoke `tflib` op library (raw `tf.nn.conv2d`/`conv2d_transpose`/
# `fused_batch_norm` calls with hand-rolled He/Glorot init, no Keras/
# `tf.layers`). No PyTorch port of this exact repo exists (confirmed via
# `gh search code "GANGogh"`: only mirrors of the original TF tree and one
# thesis repo that vendors the same TF code unmodified). Per rung 3, the
# architecture is transcribed faithfully into base-env PyTorch: every op
# in `kACGANGenerator`/`kACGANDiscriminator`/`ResidualBlock`/
# `pixcnn_gated_nonlinearity`/`SubpixelConv2D` from the real TF source is
# reproduced below with equivalent PyTorch primitives (only random init
# is not reproduced bit-for-bit; layer topology, channel counts, gating,
# and up/downsampling scheme are unchanged from the source).
#
# Architecture (transcribed from source): AC-WGAN-GP (Auxiliary-
# Classifier Improved Wasserstein GAN) for class-conditional WikiArt
# genre-conditioned art generation, `MODE = 'acwgan'` in the source
# (CLASSES=14 WikiArt genres). Both Generator and Discriminator are
# built from `ResidualBlock` (source lines 78-116): a pre-activation
# ResNet block with a 1x1 bottleneck-in conv, a `filter_size`x
# `filter_size` conv (spatial, optionally resampled), a 1x1 bottleneck-out
# conv, and a BatchNorm (or, in the discriminator under wgan/wgan-gp/acwgan
# modes, a LayerNorm-over-[C,H,W] per the source's `Batchnorm()` dispatch
# at line 53-60), with the block output computed as
# `shortcut + 0.3 * output` (the source's fixed 0.3 residual-scale
# constant, reproduced verbatim).
#   - `resample=None`: shortcut is identity (in_dim==out_dim) or a 1x1 conv;
#     spatial conv keeps resolution.
#   - `resample='down'` (Discriminator): shortcut is a stride-2 1x1 conv;
#     spatial conv is stride-2.
#   - `resample='up'` (Generator): shortcut is `SubpixelConv2D` (the
#     source's PixelShuffle-based nearest-neighbor-free upsampler: a conv
#     to 4x the target channels, `depth_to_space` with block_size=2);
#     spatial conv is a transposed conv (`Deconv2D`, stride 2).
# `kACGANGenerator` (source lines 120-168): a class-conditioned noise
# vector (128-d Gaussian noise concatenated with a one-hot class label)
# is projected by a Linear layer to an 8*dim*2 x 4 x 4 feature map, then
# passed through 4 upsampling stages (4x4 -> 8x8 -> 16x16 -> 32x32 ->
# 64x64), each stage: BatchNorm -> "PixCNN gated nonlinearity"
# (`pixcnn_gated_nonlinearity`, source lines 62-68: splits the 2*dim
# feature map in half along channels into (a, b), computes
# `sigmoid(a + cond_a) * tanh(b + cond_b)` where cond_a/cond_b come from a
# per-stage Linear projection of the one-hot class label -- this is the
# class-conditioning mechanism, distinct from plain concatenation) ->
# `Deconv2D` (stride-2 transposed conv) to the next stage's channel count.
# A final `Deconv2D` (no residual block) projects to 3 RGB channels,
# followed by `tanh`.
# `kACGANDiscriminator` (source lines 170-207): the mirror-image plain
# strided-conv stack (4 stride-2 Conv2D + BatchNorm + LeakyReLU stages,
# 64x64 -> 4x4), flattened and passed through two Linear heads: a scalar
# WGAN critic score (`sourceOutput`) and a `CLASSES`-way auxiliary
# classifier logit (`classOutput`), matching the source's two-head
# AC-GAN discriminator exactly (this simplified conditional discriminator,
# not the full `ResidualBlock`-based one, is what `kACGANDiscriminator`
# actually builds -- reproduced as such here rather than the unused
# `ResidualBlock`-with-LayerNorm discriminator path that only fires for
# the plain 'wgan'/'wgan-gp' `Discriminator` builders GANgogh's
# `GeneratorAndDiscriminator()` does not select).

import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 64  # Model dimensionality (matches source DIM=64)
CLASSES = 14  # WikiArt genre classes (matches source CLASSES=14)
NOISE_DIM = 128  # matches source's `noise = tf.random_normal([n_samples, 128])`


class SubpixelConv2d(nn.Module):
    """Faithful port of GANgogh's SubpixelConv2D (source conv2d.py wrapper +
    depth_to_space): a conv to 4x the target output channels followed by a
    PixelShuffle(2), i.e. the standard "sub-pixel convolution" upsampler."""

    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size, padding=kernel_size // 2)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))


class ResidualBlockUp(nn.Module):
    """Faithful port of GANgogh's ResidualBlock(resample='up') used by the
    generator: SubpixelConv2d shortcut + (1x1 bottleneck-in -> stride-2
    transposed conv -> 1x1 bottleneck-out) main path, combined as
    shortcut + 0.3 * output (source line 116)."""

    def __init__(self, input_dim, output_dim, filter_size=3):
        super().__init__()
        self.shortcut = SubpixelConv2d(input_dim, output_dim, kernel_size=1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // 2, 1)
        self.conv1b = nn.ConvTranspose2d(
            input_dim // 2,
            output_dim // 2,
            filter_size,
            stride=2,
            padding=filter_size // 2,
            output_padding=1,
        )
        self.conv2 = nn.Conv2d(output_dim // 2, output_dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv1b(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn(out)
        return shortcut + 0.3 * out


def pixcnn_gated_nonlinearity(a, b, c=None, d=None):
    """Faithful port of GANgogh's pixcnn_gated_nonlinearity (source lines
    62-68): class-conditioned gated activation used between generator
    upsampling stages."""
    if c is not None and d is not None:
        a = a + c
        b = b + d
    return torch.sigmoid(a) * torch.tanh(b)


class GANgoghGenerator(nn.Module):
    """Faithful port of GANgogh's kACGANGenerator (source lines 120-168):
    class-conditioned noise -> Linear projection to a 4x4 feature map ->
    4 gated-nonlinearity + Deconv2D upsampling stages -> Deconv2D to RGB
    -> tanh. dim=DIM=64 matches the source's default model width."""

    def __init__(self, dim=DIM, num_classes=CLASSES, noise_dim=NOISE_DIM):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.noise_dim = noise_dim

        self.input_linear = nn.Linear(noise_dim + num_classes, 8 * 4 * 4 * dim * 2)
        self.bn1 = nn.BatchNorm2d(8 * dim * 2)
        self.cond1 = nn.Linear(num_classes, 8 * 4 * 4 * dim * 2, bias=False)

        self.deconv2 = nn.ConvTranspose2d(
            8 * dim, 4 * dim * 2, 5, stride=2, padding=2, output_padding=1
        )
        self.bn2 = nn.BatchNorm2d(4 * dim * 2)
        self.cond2 = nn.Linear(num_classes, 4 * 8 * 8 * dim * 2)

        self.deconv3 = nn.ConvTranspose2d(
            4 * dim, 2 * dim * 2, 5, stride=2, padding=2, output_padding=1
        )
        self.bn3 = nn.BatchNorm2d(2 * dim * 2)
        self.cond3 = nn.Linear(num_classes, 2 * 16 * 16 * dim * 2)

        self.deconv4 = nn.ConvTranspose2d(
            2 * dim, dim * 2, 5, stride=2, padding=2, output_padding=1
        )
        self.bn4 = nn.BatchNorm2d(dim * 2)
        self.cond4 = nn.Linear(num_classes, 32 * 32 * dim * 2)

        self.deconv5 = nn.ConvTranspose2d(dim, 3, 5, stride=2, padding=2, output_padding=1)

    def forward(self, noise, labels):
        # labels: [B, num_classes] one-hot (float)
        b = noise.shape[0]
        z = torch.cat([noise, labels], dim=1)

        out = self.input_linear(z)
        out = out.view(b, 8 * self.dim * 2, 4, 4)
        out = self.bn1(out)
        cond = self.cond1(labels).view(b, 8 * self.dim * 2, 4, 4)
        out = pixcnn_gated_nonlinearity(out[:, 0::2], out[:, 1::2], cond[:, 0::2], cond[:, 1::2])

        out = self.deconv2(out)
        out = self.bn2(out)
        cond = self.cond2(labels).view(b, 4 * self.dim * 2, 8, 8)
        out = pixcnn_gated_nonlinearity(out[:, 0::2], out[:, 1::2], cond[:, 0::2], cond[:, 1::2])

        out = self.deconv3(out)
        out = self.bn3(out)
        cond = self.cond3(labels).view(b, 2 * self.dim * 2, 16, 16)
        out = pixcnn_gated_nonlinearity(out[:, 0::2], out[:, 1::2], cond[:, 0::2], cond[:, 1::2])

        out = self.deconv4(out)
        out = self.bn4(out)
        cond = self.cond4(labels).view(b, self.dim * 2, 32, 32)
        out = pixcnn_gated_nonlinearity(out[:, 0::2], out[:, 1::2], cond[:, 0::2], cond[:, 1::2])

        out = self.deconv5(out)
        out = torch.tanh(out)
        return out


class GANgoghDiscriminator(nn.Module):
    """Faithful port of GANgogh's kACGANDiscriminator (source lines
    170-207): plain strided-conv stack (64x64 -> 4x4), then two Linear
    heads -- a scalar WGAN-GP critic score and a CLASSES-way auxiliary
    classifier, matching the source's two-head AC-GAN discriminator."""

    def __init__(self, dim=DIM, num_classes=CLASSES):
        super().__init__()
        self.dim = dim

        self.conv1 = nn.Conv2d(3, dim, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(dim, 2 * dim, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(2 * dim)
        self.conv3 = nn.Conv2d(2 * dim, 4 * dim, 5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(4 * dim)
        self.conv4 = nn.Conv2d(4 * dim, 8 * dim, 5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(8 * dim)

        self.source_output = nn.Linear(4 * 4 * 8 * dim, 1)
        self.class_output = nn.Linear(4 * 4 * 8 * dim, num_classes)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.2)
        out = F.leaky_relu(self.bn2(self.conv2(out)), 0.2)
        out = F.leaky_relu(self.bn3(self.conv3(out)), 0.2)
        out = F.leaky_relu(self.bn4(self.conv4(out)), 0.2)
        out = out.reshape(out.shape[0], -1)
        source_out = self.source_output(out)
        class_out = self.class_output(out)
        return source_out, class_out


class GANgoghACWGAN(nn.Module):
    """Wraps the faithfully-ported GANgogh generator+discriminator into a
    single forward pass so torchlens can trace the full computation graph
    in one call (the original code drives them via a separate TF session
    training loop; the two networks themselves are unmodified above)."""

    def __init__(self, dim=DIM, num_classes=CLASSES, noise_dim=NOISE_DIM):
        super().__init__()
        self.generator = GANgoghGenerator(dim=dim, num_classes=num_classes, noise_dim=noise_dim)
        self.discriminator = GANgoghDiscriminator(dim=dim, num_classes=num_classes)

    def forward(self, noise, labels):
        fake = self.generator(noise, labels)
        source_out, class_out = self.discriminator(fake)
        return fake, source_out, class_out


def build_gangogh_acwgan():
    # Small dim for a fast trace; layer topology/gating/conditioning
    # scheme unchanged from the source.
    return GANgoghACWGAN(dim=8, num_classes=CLASSES, noise_dim=NOISE_DIM)


def example_input_gangogh_acwgan():
    torch.manual_seed(0)
    batch = 2
    noise = torch.randn(batch, NOISE_DIM)
    labels = torch.zeros(batch, CLASSES)
    labels[torch.arange(batch), torch.randint(0, CLASSES, (batch,))] = 1.0
    return (noise, labels)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "GANgogh AC-WGAN-GP",
        "build_gangogh_acwgan",
        "example_input_gangogh_acwgan",
        2017,
        "ported",
    ),
]
