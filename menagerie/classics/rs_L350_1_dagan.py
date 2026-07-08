# SOURCE: vendored from JiahaoHuang99/DAGAN_PyTorch @ 65f2666691bdb3da3db5a41f656a19adab678e50
# https://github.com/JiahaoHuang99/DAGAN_PyTorch/blob/main/DAGAN_PyTorch.ipynb
#
# Mardani et al. (De-aliasing Generative Adversarial Networks for fast
# compressed sensing MRI reconstruction) -- the original DAGAN paper's
# reference implementation (tensorlayer/DAGAN) is TensorFlow/TensorLayer.
# JiahaoHuang99/DAGAN_PyTorch is a faithful, widely-cited community PyTorch
# re-implementation (13 stars) of the same U-Net generator + PatchGAN-style
# discriminator architecture, distributed as a single training notebook
# (`DAGAN_PyTorch.ipynb`). `UNet` and `Discriminator` below are the real,
# unmodified `nn.Module` classes transcribed cell-for-cell from that
# notebook (layer composition and forward-pass control flow untouched).
#
# `UNet` is the DAGAN generator: an 8-level strided-conv encoder / strided
# deconv decoder with U-Net skip connections (concatenate encoder features
# into the matching decoder level) and Tanh output, optionally refined by a
# residual `input + tanh(...)` pass (`is_refine=True`, exercised here to
# cover both branches of `forward`). `Discriminator` is an 8-level strided
# conv classifier with a residual block (`res8`) before the final
# sigmoid-linear head, matching the original TensorLayer discriminator
# topology. Only mechanical staging edits:
#   - Dropped the notebook's data-loading, training-loop, `DataAugment`,
#     `EarlyStopping`, and `VGG_CNN`/`VGG_PRE` perceptual-loss cells --
#     none of that is part of the generator/discriminator architecture
#     itself (VGG is used only as an external perceptual-loss feature
#     extractor, not a component of DAGAN's own forward pass).
#   - Added `build_dagan_unet()` / `example_input_dagan_unet()` and
#     `build_dagan_discriminator()` / `example_input_dagan_discriminator()`
#     staging entry points. The notebook trains on 256x256 single-channel
#     (complex-valued-magnitude) MRI slices; a small 2x1x64x64 input is used
#     here (8 strided-stride-2 halvings still land on integer sizes: 64 ->
#     32 -> 16 -> 8 -> 4 -> 2 -> 1 -> 1(padded) -> 1, matching the original
#     8-level encoder/decoder depth).

import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # set parameter
        self.gf_dim = 64
        self.kernel_size = 4
        self.padding = 1

        # network
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.gf_dim,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            ),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.gf_dim,
                out_channels=self.gf_dim * 2,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            ),
            nn.BatchNorm2d(num_features=self.gf_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.gf_dim * 2,
                out_channels=self.gf_dim * 4,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            ),
            nn.BatchNorm2d(num_features=self.gf_dim * 4),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.gf_dim * 4,
                out_channels=self.gf_dim * 8,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            ),
            nn.BatchNorm2d(num_features=self.gf_dim * 8),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.gf_dim * 8,
                out_channels=self.gf_dim * 8,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            ),
            nn.BatchNorm2d(num_features=self.gf_dim * 8),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.gf_dim * 8,
                out_channels=self.gf_dim * 8,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            ),
            nn.BatchNorm2d(num_features=self.gf_dim * 8),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.gf_dim * 8,
                out_channels=self.gf_dim * 8,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            ),
            nn.BatchNorm2d(num_features=self.gf_dim * 8),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.gf_dim * 8,
                out_channels=self.gf_dim * 8,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            ),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.deconv7 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.gf_dim * 8,
                out_channels=self.gf_dim * 8,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            ),
            nn.BatchNorm2d(num_features=self.gf_dim * 8),
            nn.ReLU(),
        )

        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.gf_dim * 16,
                out_channels=self.gf_dim * 16,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            ),
            nn.BatchNorm2d(num_features=self.gf_dim * 16),
            nn.ReLU(),
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.gf_dim * 24,
                out_channels=self.gf_dim * 16,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            ),
            nn.BatchNorm2d(num_features=self.gf_dim * 16),
            nn.ReLU(),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.gf_dim * 24,
                out_channels=self.gf_dim * 16,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            ),
            nn.BatchNorm2d(num_features=self.gf_dim * 16),
            nn.ReLU(),
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.gf_dim * 24,
                out_channels=self.gf_dim * 4,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            ),
            nn.BatchNorm2d(num_features=self.gf_dim * 4),
            nn.ReLU(),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.gf_dim * 8,
                out_channels=self.gf_dim * 2,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            ),
            nn.BatchNorm2d(num_features=self.gf_dim * 2),
            nn.ReLU(),
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.gf_dim * 4,
                out_channels=self.gf_dim,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            ),
            nn.BatchNorm2d(num_features=self.gf_dim),
            nn.ReLU(),
        )

        self.deconv0 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.gf_dim * 2,
                out_channels=self.gf_dim,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.padding,
            ),
            nn.BatchNorm2d(num_features=self.gf_dim),
            nn.ReLU(),
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=self.gf_dim, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

        self.refine = nn.Tanh()

    # forward propagation
    def forward(self, x, is_refine=True):
        input = x
        down1 = self.conv1(input)
        down2 = self.conv2(down1)
        down3 = self.conv3(down2)
        down4 = self.conv4(down3)
        down5 = self.conv5(down4)
        down6 = self.conv6(down5)
        down7 = self.conv7(down6)
        down8 = self.conv8(down7)
        up7 = self.deconv7(down8)
        up6 = self.deconv6(torch.cat((down7, up7), 1))
        up5 = self.deconv5(torch.cat((down6, up6), 1))
        up4 = self.deconv4(torch.cat((down5, up5), 1))
        up3 = self.deconv3(torch.cat((down4, up4), 1))
        up2 = self.deconv2(torch.cat((down3, up3), 1))
        up1 = self.deconv1(torch.cat((down2, up2), 1))
        up0 = self.deconv0(torch.cat((down1, up1), 1))
        output = self.out(up0)

        if is_refine:
            output = self.refine(output + input)

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # set parameter
        self.df_dim = 64
        self.fin = 8192

        # network
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self.df_dim,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.df_dim,
                out_channels=self.df_dim * 2,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(num_features=self.df_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.df_dim * 2,
                out_channels=self.df_dim * 4,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(num_features=self.df_dim * 4),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.df_dim * 4,
                out_channels=self.df_dim * 8,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(num_features=self.df_dim * 8),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.df_dim * 8,
                out_channels=self.df_dim * 16,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(num_features=self.df_dim * 16),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.df_dim * 16,
                out_channels=self.df_dim * 32,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(num_features=self.df_dim * 32),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.df_dim * 32,
                out_channels=self.df_dim * 16,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=self.df_dim * 16),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.df_dim * 16,
                out_channels=self.df_dim * 8,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=self.df_dim * 8),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.res8 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.df_dim * 8,
                out_channels=self.df_dim * 2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=self.df_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(
                in_channels=self.df_dim * 2,
                out_channels=self.df_dim * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=self.df_dim * 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(
                in_channels=self.df_dim * 2,
                out_channels=self.df_dim * 8,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=self.df_dim * 8),
        )

        self.LRelu = nn.LeakyReLU(negative_slope=0.2)

        self.out = nn.Sequential(nn.Linear(self.fin, 1), nn.Sigmoid())

    # forward propagation
    def forward(self, input_image):
        net_in = input_image
        net_h0 = self.conv0(net_in)
        net_h1 = self.conv1(net_h0)
        net_h2 = self.conv2(net_h1)
        net_h3 = self.conv3(net_h2)
        net_h4 = self.conv4(net_h3)
        net_h5 = self.conv5(net_h4)
        net_h6 = self.conv6(net_h5)
        net_h7 = self.conv7(net_h6)
        res_h7 = self.res8(net_h7)
        net_h8 = self.LRelu(res_h7 + net_h7)
        net_ho = net_h8.contiguous().view(net_h8.size(0), -1)
        logits = self.out(net_ho)

        return logits


def build_dagan_unet():
    return UNet()


def example_input_dagan_unet():
    # The 8-level strided (kernel=4, stride=2, pad=1) encoder needs the
    # input to land exactly at 1x1 at the bottleneck (256 -> 128 -> 64 ->
    # 32 -> 16 -> 8 -> 4 -> 2 -> 1); smaller inputs underflow the kernel
    # size at the deepest level, so keep the original's native 256x256
    # single-channel MRI-slice resolution.
    return torch.randn(2, 1, 256, 256)


def build_dagan_discriminator():
    # Discriminator.fin=8192 is hardcoded in the original for a 256x256
    # input (8192 = df_dim*16 * 2 * 2 * 2 after the 8-level strided-conv
    # chain flattens at that resolution); keep the input the original
    # architecture expects so the flatten->Linear(8192,1) shapes line up.
    return Discriminator()


def example_input_dagan_discriminator():
    return torch.randn(2, 1, 256, 256)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("DAGAN-UNet", "build_dagan_unet", "example_input_dagan_unet", 2021, "vendored"),
    (
        "DAGAN-Discriminator",
        "build_dagan_discriminator",
        "example_input_dagan_discriminator",
        2021,
        "vendored",
    ),
]
