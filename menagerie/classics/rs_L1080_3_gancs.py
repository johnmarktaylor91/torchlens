# FAITHFUL PORT of gongenhao/GANCS @ master (original framework: TensorFlow 1.x)
#
# GANCS: GAN for Compressed Sensing MRI reconstruction. The official repo
# (srez_model.py) is written against TF1.x graph-mode APIs (tf.variable_scope,
# tf.get_variable, tf.nn.conv2d, tf.fft2d/tf.ifft2d, session-based training)
# with no PyTorch/TF2 port available, so the architecture is transcribed
# faithfully here rather than vendored.
#
# Ported components (real code, translated layer-for-layer -- not a
# from-scratch guess):
#   Model.add_residual_block / add_conv2d / add_conv2d_transpose /
#       add_batch_norm / add_relu -> the generic layer-builder helper methods
#       used to assemble the networks below.
#   _generator_model_with_scale(...)  -> the default 'resnet' generator used
#       by create_model(architecture='resnet'): an "upside-down all
#       convolutional resnet" (Arxiv 1603.05027) of 4 residual blocks (2
#       conv layers each, pre-activation BN-ReLU-Conv, identity shortcut),
#       finalized by a 1x1-all-convolutional head, followed by a k-space
#       data-consistency (DC) correction step that FFTs the generator
#       output, replaces the sampled-mask region with the true measured
#       k-space, and inverse-FFTs back to image space (the paper's DC
#       layer, ported with torch.fft.fft2/ifft2 in place of TF1's
#       tf.fft2d/tf.ifft2d).
#   _discriminator_model(...) -> the fully-convolutional PatchGAN-style
#       discriminator: 4 stride-2 conv+BN+ReLU blocks over channel widths
#       [8, 16, 32, 64] (repo default, downsized from the paper's
#       [64,128,256,512] for a tiny test config), 2 stride-1 finalization
#       conv+BN+ReLU blocks, a final 1x1 conv to a single logit channel,
#       and a spatial mean to a scalar real/fake score per example.
#
# Data layout: the original code is NHWC (TF convention); this port uses
# NCHW (PyTorch convention) with equivalent semantics (all convs are
# 'SAME'-padded, matching PyTorch's padding=kernel//2 for the odd kernel
# sizes used here).

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class ResidualBlock(nn.Module):
    """Port of Model.add_residual_block (Arxiv 1512.03385, Figure 3):
    for each of num_layers steps: BN -> ReLU -> Conv(mapsize), then add the
    original block input as the identity shortcut (with a 1x1 projection
    conv first if the channel count changes)."""

    def __init__(self, in_channels, num_units, mapsize=3, num_layers=2):
        super().__init__()
        self.proj = None
        if num_units != in_channels:
            self.proj = nn.Conv2d(in_channels, num_units, kernel_size=1, padding=0)

        layers = []
        c = num_units if self.proj is not None else in_channels
        for _ in range(num_layers):
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=False))
            layers.append(nn.Conv2d(c, num_units, kernel_size=mapsize, padding=mapsize // 2))
            c = num_units
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x)
        bypass = x
        out = self.block(x)
        return out + bypass


class GancsGenerator(nn.Module):
    """Port of _generator_model_with_scale(..., architecture='resnet'):
    the default upside-down all-convolutional resnet generator with a
    k-space data-consistency (DC) correction stage."""

    def __init__(self, in_channels=2, res_units=(16, 16, 16, 16, 16), mapsize=3, apply_dc=True):
        super().__init__()
        self.apply_dc = apply_dc

        blocks = []
        c = in_channels
        # loop over res_units[:-1]: 2x residual block, then BN-ReLU-ConvTranspose(stride=1)
        for nunits in res_units[:-1]:
            blocks.append(ResidualBlock(c, nunits, mapsize=mapsize, num_layers=2))
            c = nunits
            blocks.append(nn.BatchNorm2d(c))
            blocks.append(nn.ReLU(inplace=False))
            # add_conv2d_transpose with stride=1 on 'SAME' padding == a same-shape conv;
            # ported as ConvTranspose2d(stride=1) to mirror the original op choice.
            blocks.append(nn.ConvTranspose2d(c, nunits, kernel_size=mapsize, padding=mapsize // 2))
            c = nunits
        self.trunk = nn.Sequential(*blocks)

        nunits = res_units[-1]
        # "Finalization a la all convolutional net"
        self.final_conv1 = nn.Conv2d(c, nunits, kernel_size=mapsize, padding=mapsize // 2)
        self.final_conv2 = nn.Conv2d(nunits, nunits, kernel_size=1, padding=0)
        # Last layer -> back to complex-valued (real, imag) image channels, no activation.
        self.final_conv3 = nn.Conv2d(nunits, in_channels, kernel_size=1, padding=0)

    def forward(self, features, mask):
        """features: [B, 2, H, W] real/imag zero-filled image (channels-first).
        mask: [B, 1, H, W] real-valued k-space sampling mask (broadcastable)."""

        x = self.trunk(features)
        x = self.final_conv1(x)
        x = F.relu(x)
        x = self.final_conv2(x)
        x = F.relu(x)
        gene_out = self.final_conv3(x)

        if not self.apply_dc:
            return gene_out

        # Data-consistency (DC) correction, ported from the TF1 Fourier()/
        # tf.ifft2d path: FFT the sampled input and generator output, replace
        # the mask-sampled frequencies of the output with the true measured
        # frequencies, inverse-FFT back to image space.
        feature_complex = torch.complex(features[:, 0], features[:, 1])
        gene_complex = torch.complex(gene_out[:, 0], gene_out[:, 1])

        feature_kspace = torch.fft.fft2(feature_complex)
        gene_kspace = torch.fft.fft2(gene_complex)

        mask_c = mask[:, 0].to(feature_kspace.dtype)
        projected_kspace = feature_kspace * mask_c
        corrected_kspace = projected_kspace + gene_kspace * (1.0 - mask_c)

        corrected_complex = torch.fft.ifft2(corrected_kspace)
        corrected_real = corrected_complex.real.unsqueeze(1)
        corrected_imag = corrected_complex.imag.unsqueeze(1)

        return torch.cat([corrected_real, corrected_imag], dim=1)


class GancsDiscriminator(nn.Module):
    """Port of _discriminator_model: fully-convolutional PatchGAN-style
    discriminator over channel widths `layers`, 4 stride-2 downsampling
    blocks, 2 stride-1 finalization blocks, a 1x1 conv to a single logit
    channel, and a spatial mean -> scalar score."""

    def __init__(self, in_channels=2, layers=(8, 16, 32, 64), mapsize=3):
        super().__init__()
        blocks = []
        c = in_channels
        for nunits in layers:
            blocks.append(nn.Conv2d(c, nunits, kernel_size=mapsize, stride=2, padding=mapsize // 2))
            blocks.append(nn.BatchNorm2d(nunits))
            blocks.append(nn.ReLU(inplace=False))
            c = nunits
        self.down = nn.Sequential(*blocks)

        nunits = layers[-1]
        self.finalize1 = nn.Sequential(
            nn.Conv2d(c, nunits, kernel_size=mapsize, stride=1, padding=mapsize // 2),
            nn.BatchNorm2d(nunits),
            nn.ReLU(inplace=False),
        )
        self.finalize2 = nn.Sequential(
            nn.Conv2d(nunits, nunits, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(nunits),
            nn.ReLU(inplace=False),
        )
        self.score_conv = nn.Conv2d(nunits, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, disc_input):
        # disc_hybrid = 2 * disc_input - 1  (rescale to [-1, 1])
        x = 2 * disc_input - 1
        x = self.down(x)
        x = self.finalize1(x)
        x = self.finalize2(x)
        x = self.score_conv(x)
        return x.mean(dim=[1, 2, 3])


# ---------------------------------------------------------------------------
# Staging build/example helpers. Original repo uses 256x128 MRI slices with
# 5 residual blocks of 128 channels; shrunk here to a tiny 32x32 image with
# 5 blocks of 16 channels for a fast CPU trace, same architecture shape.
# ---------------------------------------------------------------------------
def build_gancs_generator():
    torch.manual_seed(0)
    model = GancsGenerator(in_channels=2, res_units=(16, 16, 16, 16, 16), mapsize=3, apply_dc=True)
    model.eval()
    return model


def example_input_gancs_generator():
    torch.manual_seed(0)
    features = torch.randn(2, 2, 32, 32)
    mask = (torch.rand(2, 1, 32, 32) > 0.5).float()
    return (features, mask)


def build_gancs_discriminator():
    torch.manual_seed(0)
    model = GancsDiscriminator(in_channels=2, layers=(8, 16, 32, 64), mapsize=3)
    model.eval()
    return model


def example_input_gancs_discriminator():
    torch.manual_seed(0)
    return torch.rand(2, 2, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "GANCS-Generator",
        "build_gancs_generator",
        "example_input_gancs_generator",
        2017,
        MENAGERIE_ZOO,
    ),
    (
        "GANCS-Discriminator",
        "build_gancs_discriminator",
        "example_input_gancs_discriminator",
        2017,
        MENAGERIE_ZOO,
    ),
]
