# SOURCE: vendored from jangsoopark/AConvNet-pytorch @ main: src/model/network.py, src/model/_blocks.py
"""Staged real-source A-ConvNet model."""

import collections
from collections.abc import Callable

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "leaky_relu": nn.LeakyReLU,
}


class BaseBlock(nn.Module):
    """Base block wrapper from upstream AConvNet-pytorch."""

    def __init__(self) -> None:
        """Initialize the block."""
        super().__init__()
        self._layer: nn.Sequential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the wrapped layer.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        return self._layer(x)


class Conv2DBlock(BaseBlock):
    """Convolution block from upstream AConvNet-pytorch."""

    def __init__(
        self,
        shape: list[int],
        stride: int,
        padding: str = "same",
        **params: object,
    ) -> None:
        """Initialize the convolution block.

        Parameters
        ----------
        shape
            Upstream shape list ``[h, w, in_channels, out_channels]``.
        stride
            Convolution stride.
        padding
            Convolution padding.
        **params
            Optional upstream block parameters.
        """
        super().__init__()

        h, w, in_channels, out_channels = shape
        seq = collections.OrderedDict(
            [
                (
                    "conv",
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(h, w),
                        stride=stride,
                        padding=padding,
                    ),
                )
            ]
        )

        if params.get("batch_norm"):
            seq.update({"bn": nn.BatchNorm2d(out_channels)})

        act_name = params.get("activation")
        if isinstance(act_name, str):
            seq.update({act_name: _ACTIVATIONS[act_name](inplace=True)})

        if params.get("max_pool"):
            kernel_size = params.get("max_pool_size", 2)
            pool_stride = params.get("max_pool_stride", kernel_size)
            seq.update({"max_pool": nn.MaxPool2d(kernel_size=kernel_size, stride=pool_stride)})

        self._layer = nn.Sequential(seq)

        w_init = params.get("w_init")
        idx = list(dict(self._layer.named_children()).keys()).index("conv")
        if isinstance(w_init, Callable):
            w_init(self._layer[idx].weight)
        b_init = params.get("b_init")
        if isinstance(b_init, Callable):
            b_init(self._layer[idx].bias)


class Network(nn.Module):
    """A-ConvNet SAR target recognizer from upstream AConvNet-pytorch."""

    def __init__(self, **params: object) -> None:
        """Initialize A-ConvNet.

        Parameters
        ----------
        **params
            Upstream network parameters.
        """
        super().__init__()
        self.dropout_rate = params.get("dropout_rate", 0.5)
        self.classes = params.get("classes", 10)
        self.channels = params.get("channels", 1)

        w_init = params.get("w_init", lambda x: nn.init.kaiming_normal_(x, nonlinearity="relu"))
        b_init = params.get("b_init", lambda x: nn.init.constant_(x, 0.1))

        self._layer = nn.Sequential(
            Conv2DBlock(
                shape=[5, 5, self.channels, 16],
                stride=1,
                padding="valid",
                activation="relu",
                max_pool=True,
                w_init=w_init,
                b_init=b_init,
            ),
            Conv2DBlock(
                shape=[5, 5, 16, 32],
                stride=1,
                padding="valid",
                activation="relu",
                max_pool=True,
                w_init=w_init,
                b_init=b_init,
            ),
            Conv2DBlock(
                shape=[6, 6, 32, 64],
                stride=1,
                padding="valid",
                activation="relu",
                max_pool=True,
                w_init=w_init,
                b_init=b_init,
            ),
            Conv2DBlock(
                shape=[5, 5, 64, 128],
                stride=1,
                padding="valid",
                activation="relu",
                w_init=w_init,
                b_init=b_init,
            ),
            nn.Dropout(p=self.dropout_rate),
            Conv2DBlock(
                shape=[3, 3, 128, self.classes],
                stride=1,
                padding="valid",
                w_init=w_init,
                b_init=nn.init.zeros_,
            ),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run A-ConvNet.

        Parameters
        ----------
        x
            SAR image tensor.

        Returns
        -------
        torch.Tensor
            Class logits.
        """
        return self._layer(x)


def build_a_convnet() -> nn.Module:
    """Build the staged A-ConvNet model.

    Returns
    -------
    nn.Module
        Model instance.
    """
    return Network(classes=10, channels=1)


def example_input_a_convnet() -> torch.Tensor:
    """Return an example SAR image input.

    Returns
    -------
    torch.Tensor
        Example input tensor.
    """
    return torch.randn(1, 1, 88, 88)


MENAGERIE_ENTRIES = [
    (
        "A-ConvNet",
        "build_a_convnet",
        "example_input_a_convnet",
        2016,
        "vendored from jangsoopark/AConvNet-pytorch",
    ),
]
