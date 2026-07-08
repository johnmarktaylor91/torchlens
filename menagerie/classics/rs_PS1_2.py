# SOURCE: vendored from DingXiaoH/ACNet @ master
# (https://github.com/DingXiaoH/ACNet/blob/master/acnet/acb.py: the real `ACBlock`
#  asymmetric-convolution-block class, vendored verbatim except .cuda()-free /
#  training-only bits stripped)
#
# Ding, Guo, Ding, Han 2019 (ICCV) "ACNet: Strengthening the Kernel Skeletons for
# Powerful CNN via Asymmetric Convolution Block". `ACBlock` replaces a KxK conv with
# three parallel branches (a real KxK conv, a Kx1 "vertical" conv, and a 1xK
# "horizontal" conv, each with its own BatchNorm) summed together; at deploy time the
# three branches re-parameterize (BN-fold + kernel-add) into a single equivalent KxK
# conv (`get_equivalent_kernel_bias`/`switch_to_deploy`, also vendored verbatim, though
# not exercised by the traced forward pass below).
#
# `acnet/acnet_builder.py` (`ACNetBuilder.Conv2dBNReLU`) shows how the paper's own code
# actually threads `ACBlock` into an existing conv net: for kernel_size==3 it swaps in
# `nn.Sequential(ACBlock(...), ReLU())` directly (no extra external BN -- ACBlock
# already owns its own BN internally) in place of a normal Conv2dBNReLU stage; that
# builder also references a `use_last_bn` kwarg that the current master `ACBlock.__init__`
# does not accept (repo version drift between the builder and the block -- calling it
# with that kwarg would raise a TypeError), so it is not reproduced here. `ACNetVGGStem`
# below mirrors that real Conv2dBNReLU(ACBlock+ReLU) usage pattern plus the real
# `base_model/vgg.py` VGG-stem topology (conv-conv-maxpool stages), at tiny channel
# widths (`deps`) for fast tracing. No architectural change to `ACBlock` itself.
import torch
import torch.nn as nn
import torch.nn.init as init

MENAGERIE_ZOO = "vendored-pytorch"


class ACBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        deploy=False,
        use_affine=True,
        reduce_gamma=False,
        gamma_init=None,
    ):
        super().__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )
        else:
            self.square_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, kernel_size),
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
                padding_mode=padding_mode,
            )
            self.square_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

            if padding - kernel_size // 2 >= 0:
                #   Common use case. E.g., k=3, p=1 or k=5, p=2
                self.crop = 0
                #   Compared to the KxK layer, the padding of the 1xK layer and Kx1 layer should be adjust to align the sliding windows (Fig 2 in the paper)
                hor_padding = [padding - kernel_size // 2, padding]
                ver_padding = [padding, padding - kernel_size // 2]
            else:
                #   A negative "padding" (padding - kernel_size//2 < 0, which is not a common use case) is cropping.
                #   Since nn.Conv2d does not support negative padding, we implement it manually
                self.crop = kernel_size // 2 - padding
                hor_padding = [0, padding]
                ver_padding = [padding, 0]

            self.ver_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                stride=stride,
                padding=ver_padding,
                dilation=dilation,
                groups=groups,
                bias=False,
                padding_mode=padding_mode,
            )

            self.hor_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=hor_padding,
                dilation=dilation,
                groups=groups,
                bias=False,
                padding_mode=padding_mode,
            )
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)

            if reduce_gamma:
                self.init_gamma(1.0 / 3)

            if gamma_init is not None:
                assert not reduce_gamma
                self.init_gamma(gamma_init)

    def _fuse_bn_tensor(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[
            :,
            :,
            square_h // 2 - asym_h // 2 : square_h // 2 - asym_h // 2 + asym_h,
            square_w // 2 - asym_w // 2 : square_w // 2 - asym_w // 2 + asym_w,
        ] += asym_kernel

    def get_equivalent_kernel_bias(self):
        hor_k, hor_b = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        ver_k, ver_b = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        square_k, square_b = self._fuse_bn_tensor(self.square_conv, self.square_bn)
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        return square_k, hor_b + ver_b + square_b

    def switch_to_deploy(self):
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.deploy = True
        self.fused_conv = nn.Conv2d(
            in_channels=self.square_conv.in_channels,
            out_channels=self.square_conv.out_channels,
            kernel_size=self.square_conv.kernel_size,
            stride=self.square_conv.stride,
            padding=self.square_conv.padding,
            dilation=self.square_conv.dilation,
            groups=self.square_conv.groups,
            bias=True,
            padding_mode=self.square_conv.padding_mode,
        )
        self.__delattr__("square_conv")
        self.__delattr__("square_bn")
        self.__delattr__("hor_conv")
        self.__delattr__("hor_bn")
        self.__delattr__("ver_conv")
        self.__delattr__("ver_bn")
        self.fused_conv.weight.data = deploy_k
        self.fused_conv.bias.data = deploy_b

    def init_gamma(self, gamma_value):
        init.constant_(self.square_bn.weight, gamma_value)
        init.constant_(self.ver_bn.weight, gamma_value)
        init.constant_(self.hor_bn.weight, gamma_value)

    def single_init(self):
        init.constant_(self.square_bn.weight, 1.0)
        init.constant_(self.ver_bn.weight, 0.0)
        init.constant_(self.hor_bn.weight, 0.0)

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            if self.crop > 0:
                ver_input = input[:, :, :, self.crop : -self.crop]
                hor_input = input[:, :, self.crop : -self.crop, :]
            else:
                ver_input = input
                hor_input = input
            vertical_outputs = self.ver_conv(ver_input)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv(hor_input)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            result = square_outputs + vertical_outputs + horizontal_outputs
            return result


def _acb_relu_stage(in_channels, out_channels, kernel_size=3, padding=1, deploy=False):
    """Mirrors the real acnet/acnet_builder.py ACNetBuilder.Conv2dBNReLU override for
    kernel_size==3: an ACBlock (which owns its own internal BN) followed by ReLU, with
    no extra external BatchNorm layer."""
    stage = nn.Sequential()
    stage.add_module(
        "acb",
        ACBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding, deploy=deploy),
    )
    stage.add_module("relu", nn.ReLU())
    return stage


class ACNetVGGStem(nn.Module):
    """Tiny VGG-style stem built from the real vendored `ACBlock`, mirroring the real
    base_model/vgg.py conv-conv-maxpool topology with ACNetBuilder's ACBlock+ReLU
    conv stages, at small channel widths for fast tracing."""

    def __init__(self, deps=(8, 8, 16, 16, 32), num_classes=10, deploy=False):
        super().__init__()
        self.conv1 = _acb_relu_stage(3, deps[0], deploy=deploy)
        self.conv2 = _acb_relu_stage(deps[0], deps[1], deploy=deploy)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = _acb_relu_stage(deps[1], deps[2], deploy=deploy)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = _acb_relu_stage(deps[2], deps[3], deploy=deploy)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = _acb_relu_stage(deps[3], deps[4], deploy=deploy)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(deps[4], num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.maxpool1(out)
        out = self.conv3(out)
        out = self.maxpool2(out)
        out = self.conv4(out)
        out = self.maxpool3(out)
        out = self.conv5(out)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


def build_acnet():
    return ACNetVGGStem()


def example_input_acnet():
    return torch.randn(1, 3, 32, 32)


MENAGERIE_ENTRIES = [
    (
        "ACNet (Asymmetric Convolution Block)",
        "build_acnet",
        "example_input_acnet",
        2019,
        MENAGERIE_ZOO,
    ),
]
