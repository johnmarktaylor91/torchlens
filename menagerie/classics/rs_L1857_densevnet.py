# SOURCE: vendored from https://github.com/yykzjh/PMFSNet @ master
# (lib/models/DenseVNet.py). The official DenseVNet repo
# (wangzhenlin123/DenseVNet3d, mirroring NiftyNet's dense_vnet.py) is
# TensorFlow-only, so no runnable PyTorch source exists there; this file is a
# self-contained, faithful independent PyTorch reimplementation of the same
# "Automatic multi-organ segmentation on abdominal CT with dense V-networks"
# (Gibson et al. 2018) architecture -- identical hyperparameters
# (kernel_size=[5,3,3], downsample/skip channel counts, dense-feature-stack
# units=[5,10,10], growth_rate=[4,8,16]) as the original NiftyNet
# `DenseVNet.__hyper_params__`. Copied verbatim (class bodies unmodified);
# only import paths and the `if __name__ == "__main__"` smoke test were
# dropped for staging.
"""Vendored DenseVNet model definition (DenseVNet + submodules)."""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class DenseVNet(nn.Module):
    def __init__(self, in_channels: int = 1, classes: int = 1):
        super().__init__()

        self.model_name = "DenseVNet"
        self.classes = classes
        kernel_size = [5, 3, 3]
        num_downsample_channels = [24, 24, 24]
        num_skip_channels = [12, 24, 24]
        units = [5, 10, 10]
        growth_rate = [4, 8, 16]

        self.dfs_blocks = torch.nn.ModuleList()
        for i in range(3):
            self.dfs_blocks.append(
                DownsampleWithDfs(
                    in_channels=in_channels,
                    downsample_channels=num_downsample_channels[i],
                    skip_channels=num_skip_channels[i],
                    kernel_size=kernel_size[i],
                    units=units[i],
                    growth_rate=growth_rate[i],
                )
            )
            in_channels = num_downsample_channels[i] + units[i] * growth_rate[i]

        self.upsample_1 = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.upsample_2 = torch.nn.Upsample(scale_factor=4, mode="trilinear")

        self.out_conv = ConvBlock(
            in_channels=sum(num_skip_channels),
            out_channels=self.classes,
            kernel_size=3,
            batch_norm=True,
            preactivation=True,
        )
        self.upsample_out = torch.nn.Upsample(scale_factor=2, mode="trilinear")

    def forward(self, x):
        x, skip_1 = self.dfs_blocks[0](x)
        x, skip_2 = self.dfs_blocks[1](x)
        _, skip_3 = self.dfs_blocks[2](x)

        skip_2 = self.upsample_1(skip_2)
        skip_3 = self.upsample_2(skip_3)

        out = self.out_conv(torch.cat([skip_1, skip_2, skip_3], 1))
        out = self.upsample_out(out)

        return out


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        batch_norm=True,
        preactivation=False,
    ):
        super().__init__()

        if dilation != 1:
            raise NotImplementedError()

        padding = kernel_size - stride
        if padding % 2 != 0:
            pad = torch.nn.ConstantPad3d(tuple([padding % 2, padding - padding % 2] * 3), 0)
        else:
            pad = torch.nn.ConstantPad3d(padding // 2, 0)

        if preactivation:
            layers = [
                torch.nn.ReLU(),
                pad,
                torch.nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            ]
            if batch_norm:
                layers = [torch.nn.BatchNorm3d(in_channels)] + layers
        else:
            layers = [
                pad,
                torch.nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
            ]
            if batch_norm:
                layers.append(torch.nn.BatchNorm3d(out_channels))
            layers.append(torch.nn.ReLU())

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class DenseFeatureStack(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        units,
        growth_rate,
        kernel_size,
        dilation=1,
        batch_norm=True,
        batchwise_spatial_dropout=False,
    ):
        super().__init__()

        self.units = torch.nn.ModuleList()
        for _ in range(units):
            if batchwise_spatial_dropout:
                raise NotImplementedError

            self.units.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=growth_rate,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=1,
                    batch_norm=batch_norm,
                    preactivation=True,
                )
            )
            in_channels += growth_rate

    def forward(self, x):
        feature_stack = [x]

        for unit in self.units:
            inputs = torch.cat(feature_stack, 1)
            out = unit(inputs)
            feature_stack.append(out)

        return torch.cat(feature_stack, 1)


class DownsampleWithDfs(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        downsample_channels,
        skip_channels,
        kernel_size,
        units,
        growth_rate,
    ):
        super().__init__()

        self.downsample = ConvBlock(
            in_channels=in_channels,
            out_channels=downsample_channels,
            kernel_size=kernel_size,
            stride=2,
            batch_norm=True,
            preactivation=True,
        )
        self.dfs = DenseFeatureStack(downsample_channels, units, growth_rate, 3, batch_norm=True)
        self.skip = ConvBlock(
            in_channels=downsample_channels + units * growth_rate,
            out_channels=skip_channels,
            kernel_size=3,
            batch_norm=True,
            preactivation=True,
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.dfs(x)
        x_skip = self.skip(x)

        return x, x_skip


# ---------------------------------------------------------------------------
# Staging build/example helpers (tiny spatial size, matches the repo's own
# smoke-test input shape family, scaled down for fast tracing).
# ---------------------------------------------------------------------------


def build_densevnet():
    return DenseVNet(in_channels=1, classes=3)


def example_input_densevnet():
    # Spatial dims must be divisible by 8 (3 downsample-by-2 stages).
    return (torch.rand(1, 1, 32, 32, 32),)


MENAGERIE_ENTRIES = [
    ("DenseVNet", build_densevnet, example_input_densevnet, 2018, "vendored-pytorch"),
]
