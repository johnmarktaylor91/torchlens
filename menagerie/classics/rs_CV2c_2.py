# SOURCE: vendored from dicarlolab/CORnet @ master (cornet_s.py)

import math
from collections import OrderedDict

import torch
from torch import nn


HASH = "1d3f7974"


class Flatten(nn.Module):
    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules.
    """

    def forward(self, x):
        """
        Flatten a batch of tensors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Flattened tensor.
        """

        return x.view(x.size(0), -1)


class Identity(nn.Module):
    """
    Helper module that stores the current tensor. Useful for accessing by name.
    """

    def forward(self, x):
        """
        Return the input unchanged.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            The same tensor.
        """

        return x


class CORblock_S(nn.Module):
    """
    CORnet-S recurrent residual block.
    """

    scale = 4

    def __init__(self, in_channels, out_channels, times=1):
        """
        Initialize the CORnet-S block.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        times : int
            Number of recurrent timesteps.
        """

        super().__init__()

        self.times = times
        self.conv_input = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.skip = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale, kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels * self.scale,
            out_channels * self.scale,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels, kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()

        for t in range(self.times):
            setattr(self, f"norm1_{t}", nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f"norm2_{t}", nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f"norm3_{t}", nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        """
        Run the recurrent CORnet-S block.

        Parameters
        ----------
        inp : torch.Tensor
            Input image features.

        Returns
        -------
        torch.Tensor
            Output features.
        """

        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.skip(x)
                skip = self.norm_skip(skip)
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f"norm1_{t}")(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f"norm2_{t}")(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f"norm3_{t}")(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)
        return output


def CORnet_S():
    """
    Build the official CORnet-S model.

    Returns
    -------
    nn.Sequential
        CORnet-S model.
    """

    model = nn.Sequential(
        OrderedDict(
            [
                (
                    "V1",
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv1",
                                    nn.Conv2d(
                                        3,
                                        64,
                                        kernel_size=7,
                                        stride=2,
                                        padding=3,
                                        bias=False,
                                    ),
                                ),
                                ("norm1", nn.BatchNorm2d(64)),
                                ("nonlin1", nn.ReLU(inplace=True)),
                                ("pool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                                (
                                    "conv2",
                                    nn.Conv2d(
                                        64,
                                        64,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False,
                                    ),
                                ),
                                ("norm2", nn.BatchNorm2d(64)),
                                ("nonlin2", nn.ReLU(inplace=True)),
                                ("output", Identity()),
                            ]
                        )
                    ),
                ),
                ("V2", CORblock_S(64, 128, times=2)),
                ("V4", CORblock_S(128, 256, times=4)),
                ("IT", CORblock_S(256, 512, times=2)),
                (
                    "decoder",
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("avgpool", nn.AdaptiveAvgPool2d(1)),
                                ("flatten", Flatten()),
                                ("linear", nn.Linear(512, 1000)),
                                ("output", Identity()),
                            ]
                        )
                    ),
                ),
            ]
        )
    )

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    model = model.module if hasattr(model, "module") else model
    model = model.cuda() if torch.cuda.is_available() else model
    return model


def build_cornet_s() -> nn.Sequential:
    """
    Build CORnet-S for tracing.

    Returns
    -------
    nn.Sequential
        CORnet-S in evaluation mode.
    """

    model = CORnet_S()
    model.eval()
    return model


def example_input_cornet_s() -> torch.Tensor:
    """
    Create an example image input.

    Returns
    -------
    torch.Tensor
        Example input tensor with shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Brain-Score optimized recurrent ResNet family",
        "build_cornet_s",
        "example_input_cornet_s",
        "2019",
        "CV2c_158",
    )
]
