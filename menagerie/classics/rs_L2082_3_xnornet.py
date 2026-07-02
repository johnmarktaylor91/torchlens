# SOURCE: vendored from jiecaoyu/XNOR-Net-PyTorch @ master
#
# XNOR-Net: "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural
# Networks" (Rastegari, Ordonez, Redmon, Farhadi, ECCV 2016). This is the official
# jiecaoyu/XNOR-Net-PyTorch AlexNet reproduction
# (ImageNet/networks/model_list/alexnet.py): weights and activations are binarized
# (sign()) and convolutions/linears run on the binarized tensors with a
# straight-through estimator gradient (clip outside [-1, 1]); the conv/linear layers
# themselves stay full-precision `nn.Conv2d`/`nn.Linear` (as in the real repo -- true
# XNOR+popcount kernels are a runtime/inference-engine optimization applied to the
# binarized weights, not a distinct nn.Module here) with per-layer scaling handled by
# the paper's separate BinOp weight-binarization utility (main.py, training-time only,
# not part of the forward architecture).
#
# The only change from the real code is a minimal PyTorch API-compat fix, NOT an
# architectural change: `BinActive` used the pre-2019 legacy `torch.autograd.Function`
# calling convention (bare instance `.forward`/`.backward`, no `@staticmethod`), which
# torch 2.8 hard-rejects at call time (RuntimeError: "Legacy autograd function with
# non-static forward method is deprecated"). It is rewritten here using the modern
# `@staticmethod forward(ctx, ...)` / `@staticmethod backward(ctx, ...)` convention
# with byte-identical binarization math (sign() forward, straight-through-estimator
# gradient clipped to input.ge(1)/input.le(-1)) -- everything else (BinConv2d,
# AlexNet's exact layer sequence/channel counts/kernel sizes) is unmodified.

from __future__ import annotations

import torch
from torch import Tensor, nn


class BinActive(torch.autograd.Function):
    """Binarize the input activations (sign()) with a straight-through estimator
    gradient. Modern @staticmethod rewrite of the real repo's legacy-style
    torch.autograd.Function (see module header); math is unchanged."""

    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:  # noqa: A002 - matches upstream name
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class BinConv2d(nn.Module):
    """XNOR-Net binarized conv block: BN -> binarize activations -> conv -> ReLU."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = -1,
        stride: int = -1,
        padding: int = -1,
        groups: int = 1,
        dropout: float = 0,
        Linear: bool = False,  # noqa: N803 - matches upstream name
    ) -> None:
        super().__init__()
        self.layer_type = "BinConv2d"
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        self.Linear = Linear
        if not self.Linear:
            self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.conv = nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            )
        else:
            self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn(x)
        x = BinActive.apply(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x)
        if not self.Linear:
            x = self.conv(x)
        else:
            x = self.linear(x)
        x = self.relu(x)
        return x


class XNORAlexNet(nn.Module):
    """XNOR-Net's AlexNet reproduction (ImageNet/networks/model_list/alexnet.py::AlexNet)."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BinConv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            BinConv2d(256, 384, kernel_size=3, stride=1, padding=1),
            BinConv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=1),
            BinConv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            BinConv2d(256 * 6 * 6, 4096, Linear=True),
            BinConv2d(4096, 4096, dropout=0.5, Linear=True),
            nn.BatchNorm1d(4096, eps=1e-3, momentum=0.1, affine=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def build_xnor_alexnet() -> nn.Module:
    """Build a random-init XNOR-Net AlexNet at a reduced class count."""

    return XNORAlexNet(num_classes=10).eval()


def example_input_xnor_alexnet() -> Tensor:
    """Return the AlexNet-sized RGB input the real repo trains at (227x227 crop)."""

    return torch.randn(1, 3, 227, 227)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "XNOR-Net-AlexNet",
        "build_xnor_alexnet",
        "example_input_xnor_alexnet",
        "2016",
        "CV",
    )
]
