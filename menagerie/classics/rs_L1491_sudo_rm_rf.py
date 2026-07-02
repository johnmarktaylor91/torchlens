# SOURCE: vendored from etzinis/sudo_rm_rf @ master
# https://raw.githubusercontent.com/etzinis/sudo_rm_rf/master/sudo_rm_rf/dnn/models/sudormrf.py
#
# SuDoRM-RF ("Successive Downsampling and Resampling of Multi-Resolution Features"):
# an efficient universal sound-source-separation network. A learned 1D conv encoder
# produces a latent representation of the mixture waveform; a stack of `UBlock`
# separation modules (each a U-Net-style REDUCE -> SPLIT -> multi-depth dilated-conv
# TRANSFORM -> MERGE, with residual skip connections between successive downsampling
# scales) predicts per-source soft masks; a mask-generating Conv2d + softmax/sigmoid
# produces `num_sources` masks that are applied to the encoder features; a transposed
# 1D conv decoder reconstructs the separated waveforms.
#
# `ConvNormAct`, `ConvNorm`, `NormAct`, `DilatedConv`, `DilatedConvNorm`, `UBlock`, and
# `SuDORMRF` are transcribed verbatim from `sudo_rm_rf/dnn/models/sudormrf.py`. No
# architectural changes were made -- every Conv1d/Conv2d/ConvTranspose1d/GroupNorm/PReLU
# layer and its arguments, the dilated multi-depth downsample/upsample U-block topology,
# and the mask-softmax/mixture-consistency forward flow are unchanged. Only the
# module-level `if __name__ == "__main__":` smoke-test block (not part of the traced
# architecture) is dropped.

import math

import torch
import torch.nn as nn


class ConvNormAct(nn.Module):
    """This class defines the convolution layer with normalization and a PReLU
    activation"""

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups
        )
        self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    """This class defines the convolution layer with normalization and PReLU
    activation"""

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups
        )
        self.norm = nn.GroupNorm(1, nOut, eps=1e-08)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    """This class defines a normalization and PReLU activation"""

    def __init__(self, nOut):
        super().__init__()
        self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(nn.Module):
    """This class defines the dilated convolution."""

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):
    """This class defines the dilated convolution with normalized output."""

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )
        self.norm = nn.GroupNorm(1, nOut, eps=1e-08)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class UBlock(nn.Module):
    """This class defines the Upsampling block, which is based on the following
    principle: REDUCE ---> SPLIT ---> TRANSFORM --> MERGE"""

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DilatedConvNorm(in_channels, in_channels, kSize=5, stride=1, groups=in_channels, d=1)
        )

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(
                DilatedConvNorm(
                    in_channels,
                    in_channels,
                    kSize=2 * stride + 1,
                    stride=stride,
                    groups=in_channels,
                    d=1,
                )
            )
        if upsampling_depth > 1:
            self.upsampler = torch.nn.Upsample(scale_factor=2)
        self.conv_1x1_exp = ConvNorm(in_channels, out_channels, 1, 1, groups=1)
        self.final_norm = NormAct(in_channels)
        self.module_act = NormAct(out_channels)

    def forward(self, x):
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # Gather them now in reverse order
        for _ in range(self.depth - 1):
            resampled_out_k = self.upsampler(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k

        expanded = self.conv_1x1_exp(self.final_norm(output[-1]))

        return self.module_act(expanded + x)


class SuDORMRF(nn.Module):
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        enc_kernel_size=21,
        enc_num_basis=512,
        num_sources=2,
    ):
        super(SuDORMRF, self).__init__()

        # Number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(self.enc_kernel_size // 2 * 2**self.upsampling_depth) // math.gcd(
            self.enc_kernel_size // 2, 2**self.upsampling_depth
        )

        # Front end
        self.encoder = nn.Sequential(
            *[
                nn.Conv1d(
                    in_channels=1,
                    out_channels=enc_num_basis,
                    kernel_size=enc_kernel_size,
                    stride=enc_kernel_size // 2,
                    padding=enc_kernel_size // 2,
                ),
                nn.ReLU(),
            ]
        )

        # Norm before the rest, and apply one more dense layer
        self.ln = nn.GroupNorm(1, enc_num_basis, eps=1e-08)
        self.l1 = nn.Conv1d(in_channels=enc_num_basis, out_channels=out_channels, kernel_size=1)

        # Separation module
        self.sm = nn.Sequential(
            *[
                UBlock(
                    out_channels=out_channels,
                    in_channels=in_channels,
                    upsampling_depth=upsampling_depth,
                )
                for r in range(num_blocks)
            ]
        )

        if out_channels != enc_num_basis:
            self.reshape_before_masks = nn.Conv1d(
                in_channels=out_channels, out_channels=enc_num_basis, kernel_size=1
            )

        # Masks layer
        self.m = nn.Conv2d(
            in_channels=1,
            out_channels=num_sources,
            kernel_size=(enc_num_basis + 1, 1),
            padding=(enc_num_basis - enc_num_basis // 2, 0),
        )

        # Back end
        self.decoder = nn.ConvTranspose1d(
            in_channels=enc_num_basis * num_sources,
            out_channels=num_sources,
            output_padding=(enc_kernel_size // 2) - 1,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding=enc_kernel_size // 2,
            groups=num_sources,
        )
        self.ln_mask_in = nn.GroupNorm(1, enc_num_basis, eps=1e-08)

    # Forward pass
    def forward(self, input_wav):
        # Front end
        x = self.pad_to_appropriate_length(input_wav)
        x = self.encoder(x)

        # Split paths
        s = x.clone()

        # Separation module
        x = self.ln(x)
        x = self.l1(x)
        x = self.sm(x)

        if self.out_channels != self.enc_num_basis:
            x = self.reshape_before_masks(x)

        # Get masks and apply them
        x = self.m(x.unsqueeze(1))
        if self.num_sources == 1:
            x = torch.sigmoid(x)
        else:
            x = nn.functional.softmax(x, dim=1)
        x = x * s.unsqueeze(1)
        # Back end
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        return self.remove_trailing_zeros(estimated_waveforms, input_wav)

    def pad_to_appropriate_length(self, x):
        values_to_pad = int(x.shape[-1]) % self.lcm
        if values_to_pad:
            appropriate_shape = x.shape
            padded_x = torch.zeros(
                list(appropriate_shape[:-1]) + [appropriate_shape[-1] + self.lcm - values_to_pad],
                dtype=torch.float32,
            )
            padded_x[..., : x.shape[-1]] = x
            return padded_x
        return x

    @staticmethod
    def remove_trailing_zeros(padded_x, initial_x):
        return padded_x[..., : initial_x.shape[-1]]


def build_sudo_rm_rf():
    torch.manual_seed(0)
    # Repo's own __main__ convention (out_channels=128, in_channels=512, num_blocks=16,
    # upsampling_depth=4, enc_kernel_size=21, enc_num_basis=512, num_sources=2) shrunk
    # to tiny sizes for fast tracing.
    model = SuDORMRF(
        out_channels=8,
        in_channels=16,
        num_blocks=2,
        upsampling_depth=2,
        enc_kernel_size=21,
        enc_num_basis=8,
        num_sources=2,
    )
    model.eval()
    return model


def example_input_sudo_rm_rf():
    torch.manual_seed(0)
    # Raw waveform input convention from the repo's __main__ (batch, 1, samples).
    return torch.randn(1, 1, 4000)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("SuDoRM-RF", "build_sudo_rm_rf", "example_input_sudo_rm_rf", 2020, MENAGERIE_ZOO),
]
