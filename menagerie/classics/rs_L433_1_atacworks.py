# SOURCE: vendored from NVIDIA-Genomics-Research/AtacWorks @ master
# Files: atacworks/dl4atac/models/models.py, atacworks/dl4atac/layers.py
# (import paths adapted for standalone staging; architecture code kept verbatim)
"""AtacWorks: ResNet/U-Net denoising models for ATAC-seq signal, vendored from
the real NVIDIA-Genomics-Research/AtacWorks repository (dl4atac package)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --- atacworks/dl4atac/layers.py (verbatim) ---


class ZeroSamePad1d(nn.Module):
    """Apply SAME zero padding to input."""

    def __init__(self, interval_size, kernel_size, stride, dilation):
        super(ZeroSamePad1d, self).__init__()

        required_total_padding = ZeroSamePad1d._get_total_same_padding(
            interval_size, kernel_size, stride, dilation
        )
        padding_left = required_total_padding // 2
        padding_right = required_total_padding - padding_left
        self.pad = nn.ConstantPad1d((padding_left, padding_right), 0)

    @staticmethod
    def _get_total_same_padding(interval_size, kernel_size, stride, dilation):
        effective_kernel_size = (kernel_size - 1) * dilation + 1
        required_total_padding = (
            (interval_size - 1) * stride + effective_kernel_size - interval_size
        )
        return required_total_padding

    def forward(self, x):
        return self.pad(x)


class Activation(nn.Module):
    """Configurable activation layer."""

    def __init__(self, afunc="relu"):
        super(Activation, self).__init__()
        self.act_layer = nn.Identity()
        if afunc == "relu":
            self.act_layer = nn.ReLU()
        elif afunc == "prelu":
            self.act_layer = nn.PReLU()
        elif afunc is not None:
            raise NotImplementedError

    def forward(self, x):
        return self.act_layer(x)


class ConvAct1d(nn.Module):
    """1D conv layer with same padding.

    Optional batch normalization and activation layer.
    """

    def __init__(
        self,
        interval_size,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        bn=False,
        afunc="relu",
    ):
        self.interval_size = interval_size
        super(ConvAct1d, self).__init__()

        self.padding_layer = ZeroSamePad1d(interval_size, kernel_size, stride, dilation)
        self.conv_layer = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, bias=bias
        )
        self.bn_layer = nn.BatchNorm1d(out_channels) if bn else None
        self.act_layer = Activation(afunc) if afunc else None

    def forward(self, x):
        x = self.padding_layer(x)
        x = self.conv_layer(x)
        if self.bn_layer:
            x = self.bn_layer(x)
        if self.act_layer:
            x = self.act_layer(x)
        return x


class ResBlock(nn.Module):
    """Residual block.

    2 conv/activation layers followed by residual connection
    and third activation.
    """

    def __init__(
        self,
        interval_size,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        bn=False,
        afunc="relu",
        conv_input=False,
    ):
        super(ResBlock, self).__init__()

        if conv_input:
            self.conv_input = ConvAct1d(
                interval_size, in_channels, out_channels, kernel_size=1, bn=bn, afunc=afunc
            )
        else:
            self.conv_input = nn.Identity()
        self.conv_act1 = ConvAct1d(
            interval_size, in_channels, out_channels, kernel_size, stride, dilation, bias, bn, afunc
        )
        self.conv_act2 = ConvAct1d(
            interval_size,
            out_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            bias,
            bn,
            afunc,
        )
        self.conv_act3 = ConvAct1d(
            interval_size,
            out_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            bias,
            bn,
            afunc=None,
        )
        self.activation = nn.PReLU() if afunc == "prelu" else nn.ReLU()

    def forward(self, input):
        x = self.conv_act1(input)
        x = self.conv_act2(x)
        x = self.conv_act3(x)
        x = x + self.conv_input(input)
        x = self.activation(x)

        return x


class DownBlock(nn.Module):
    """U-net down block - 2 conv/activation layers followed by max pool."""

    def __init__(
        self,
        interval_size,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        bn=True,
        afunc="relu",
    ):
        super(DownBlock, self).__init__()

        self.conv_act1 = ConvAct1d(
            interval_size, in_channels, out_channels, kernel_size, stride, dilation, bias, bn, afunc
        )
        self.conv_act2 = ConvAct1d(
            interval_size,
            out_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            bias,
            bn,
            afunc,
        )
        self.max_pool = nn.MaxPool1d(2)

    def forward(self, input):
        x = self.conv_act1(input)
        x = self.conv_act2(x)
        xp = self.max_pool(x)

        return x, xp


class UpBlock(nn.Module):
    """U-net up block - upsampling, merge, followed by 2 conv layers."""

    def __init__(
        self,
        interval_size,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        bn=True,
        afunc="relu",
    ):
        super(UpBlock, self).__init__()

        self.conv_act1 = ConvAct1d(
            interval_size, in_channels, out_channels, kernel_size, stride, dilation, bias, bn, afunc
        )
        self.conv_act2 = ConvAct1d(
            interval_size,
            out_channels * 2,
            out_channels,
            kernel_size,
            stride,
            dilation,
            bias,
            bn,
            afunc,
        )
        self.conv_act3 = ConvAct1d(
            interval_size,
            out_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            bias,
            bn,
            afunc,
        )

    def forward(self, x_up, x_down):
        x_up = F.interpolate(x_up, scale_factor=2, mode="nearest")
        x_up = self.conv_act1(x_up)
        x_up = torch.cat((x_down, x_up), dim=1)
        x_up = self.conv_act2(x_up)
        x_up = self.conv_act3(x_up)

        return x_up


# --- atacworks/dl4atac/models/models.py (verbatim) ---


class DenoisingResNet(nn.Module):
    """Resnet model."""

    def __init__(
        self,
        interval_size,
        in_channels=1,
        out_channels=15,
        num_blocks=5,
        kernel_size=50,
        dilation=8,
        bn=False,
        afunc="relu",
        num_blocks_class=2,
        kernel_size_class=50,
        dilation_class=8,
        out_channels_class=15,
    ):
        self.interval_size = interval_size
        super(DenoisingResNet, self).__init__()

        self.res_blocks = nn.ModuleList()
        self.res_blocks_class = nn.ModuleList()

        # Residual blocks for regression
        self.res_blocks.append(
            ResBlock(
                interval_size,
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                bn=bn,
                afunc=afunc,
                conv_input=True,
            )
        )
        for _ in range(num_blocks - 1):
            self.res_blocks.append(
                ResBlock(
                    interval_size,
                    out_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation,
                    bn=bn,
                    afunc=afunc,
                    conv_input=False,
                )
            )
        self.regressor = ConvAct1d(
            interval_size,
            in_channels=out_channels,
            out_channels=1,
            kernel_size=1,
            dilation=1,
            bn=bn,
            afunc=afunc,
        )

        # Residual blocks for classification
        self.res_blocks_class.append(
            ResBlock(
                interval_size,
                in_channels=1,
                out_channels=out_channels_class,
                kernel_size=kernel_size_class,
                dilation=dilation_class,
                bn=bn,
                afunc=afunc,
                conv_input=True,
                bias=True,
            )
        )
        for _ in range(num_blocks_class - 1):
            self.res_blocks_class.append(
                ResBlock(
                    interval_size,
                    out_channels_class,
                    out_channels_class,
                    kernel_size_class,
                    dilation=dilation_class,
                    bn=bn,
                    afunc=afunc,
                    conv_input=False,
                    bias=True,
                )
            )
        self.classifier = ConvAct1d(
            interval_size,
            in_channels=out_channels,
            out_channels=1,
            kernel_size=1,
            dilation=1,
            bn=bn,
            afunc=None,
            bias=True,
        )

    def forward(self, x):
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.regressor(x)
        out_reg = x.squeeze(1)
        for res_block in self.res_blocks_class:
            x = res_block(x)
        out_cla = torch.sigmoid(self.classifier(x).squeeze(1))

        return out_reg, out_cla


class DenoisingUNet(nn.Module):
    """U-net model."""

    def __init__(self, interval_size, in_channels=1, afunc="relu", bn=False):
        self.interval_size = interval_size
        super(DenoisingUNet, self).__init__()
        self.down1 = DownBlock(
            interval_size,
            in_channels=in_channels,
            out_channels=16,
            kernel_size=5,
            bn=bn,
            afunc=afunc,
        )
        self.down2 = DownBlock(
            interval_size, in_channels=16, out_channels=32, kernel_size=5, bn=bn, afunc=afunc
        )
        self.down3 = DownBlock(
            interval_size, in_channels=32, out_channels=64, kernel_size=25, bn=bn, afunc=afunc
        )
        self.down4 = DownBlock(
            interval_size, in_channels=64, out_channels=128, kernel_size=25, bn=bn, afunc=afunc
        )

        self.conv5 = ConvAct1d(
            interval_size,
            in_channels=128,
            out_channels=256,
            kernel_size=250,
            dilation=1,
            bn=bn,
            afunc=afunc,
        )

        self.up6 = UpBlock(
            interval_size, in_channels=256, out_channels=128, kernel_size=5, bn=bn, afunc=afunc
        )
        self.up7 = UpBlock(
            interval_size, in_channels=128, out_channels=64, kernel_size=5, bn=bn, afunc=afunc
        )
        self.up8 = UpBlock(
            interval_size, in_channels=64, out_channels=32, kernel_size=5, bn=bn, afunc=afunc
        )
        self.up9 = UpBlock(
            interval_size, in_channels=32, out_channels=16, kernel_size=5, bn=bn, afunc=afunc
        )

        self.regressor = ConvAct1d(
            interval_size,
            in_channels=16,
            out_channels=1,
            kernel_size=1,
            dilation=1,
            bn=bn,
            afunc=afunc,
        )
        self.classifier = ConvAct1d(
            interval_size,
            in_channels=16,
            out_channels=1,
            kernel_size=1,
            dilation=1,
            bn=bn,
            afunc=None,
        )

    def forward(self, input):
        # for readability, keeping itermediate p1 ~ p4 and x5 ~ x9,
        # but actually unnecessary and a waste of memory
        x1, p1 = self.down1(input)
        x2, p2 = self.down2(p1)
        x3, p3 = self.down3(p2)
        x4, p4 = self.down4(p3)

        x5 = self.conv5(p4)

        x6 = self.up6(x5, x4)
        x7 = self.up7(x6, x3)
        x8 = self.up8(x7, x2)
        x9 = self.up9(x8, x1)

        out_reg = self.regressor(x9).squeeze(1)
        out_cla = torch.sigmoid(self.classifier(x9).squeeze(1))  # (N, 1, L) => (N, L)

        return out_reg, out_cla


# --- staging build/example helpers ---


def build_atacworks_resnet():
    """Tiny DenoisingResNet (regression + classification heads)."""
    return DenoisingResNet(
        interval_size=512,
        in_channels=1,
        out_channels=8,
        num_blocks=2,
        kernel_size=9,
        dilation=2,
        bn=True,
        afunc="relu",
        num_blocks_class=2,
        kernel_size_class=9,
        dilation_class=2,
        out_channels_class=8,
    )


def example_input_atacworks_resnet():
    return torch.randn(1, 1, 512)


def build_atacworks_unet():
    """Tiny DenoisingUNet (regression + classification heads)."""
    return DenoisingUNet(interval_size=256, in_channels=1, afunc="relu", bn=True)


def example_input_atacworks_unet():
    return torch.randn(1, 1, 256)


MENAGERIE_ENTRIES = [
    (
        "AtacWorks-ResNet",
        "build_atacworks_resnet",
        "example_input_atacworks_resnet",
        2019,
        "vendored-pytorch",
    ),
    (
        "AtacWorks-UNet",
        "build_atacworks_unet",
        "example_input_atacworks_unet",
        2019,
        "vendored-pytorch",
    ),
]
