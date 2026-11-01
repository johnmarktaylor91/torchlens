"""Layer-level capture of Paddle ``Conv1D``, ``Conv2D`` and ``Conv3D``.

``paddle.nn.Conv2D.forward`` and ``Conv3D.forward`` call Paddle's private
``functional.conv._conv_nd``, which the functional wrappers never see, so a
convolution layer's output used to reach its consumer as an untraced tensor.
The layers are captured where they are public: their ``forward``. One op is
emitted per layer call, named ``conv1d``/``conv2d``/``conv3d``, whose tensor
arguments are the input, the weight and the bias (so the graph has the input
as parent and the parameters as parameter leaves), and whose callable is
:func:`paddle_conv_layer_replay`, a public-API restatement of the layer's
forward that validation re-executes and compares bit-for-tolerance against
the captured output. Ops inside the layer forward are not recorded.

The module imports no Paddle at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Layer class name -> emitted ``func_name``.
PADDLE_CONV_LAYER_OPS: dict[str, str] = {
    "Conv1D": "conv1d",
    "Conv2D": "conv2d",
    "Conv3D": "conv3d",
}

_SPATIAL_DIMS: dict[str, int] = {"conv1d": 1, "conv2d": 2, "conv3d": 3}


@dataclass(frozen=True)
class _ConvHyperparameters:
    """The recorded keyword hyperparameters of one convolution layer call.

    Parameters
    ----------
    spatial_dims:
        1, 2 or 3.
    padding_mode:
        The layer's ``padding_mode`` (``"zeros"`` pads inside the convolution;
        the others pad the input first, as the layer does).
    reversed_padding:
        The layer's ``_reversed_padding_repeated_twice`` (used when
        ``padding_mode`` is not ``"zeros"``).
    stride, padding, dilation, groups, data_format:
        The layer's convolution hyperparameters.
    """

    spatial_dims: int
    padding_mode: str
    reversed_padding: Any
    stride: Any
    padding: Any
    dilation: Any
    groups: int
    data_format: str


def paddle_conv_layer_replay(x: Any, weight: Any, bias: Any = None, **hyperparameters: Any) -> Any:
    """Recompute a Paddle convolution layer's forward from public functionals.

    Parameters
    ----------
    x, weight, bias:
        Input, kernel and optional bias tensors.
    **hyperparameters:
        The recorded keyword hyperparameters :func:`conv_layer_call` emits
        (the fields of :class:`_ConvHyperparameters`).

    Returns
    -------
    Any
        The convolution output.
    """

    import paddle.nn.functional as paddle_functional

    layer = _ConvHyperparameters(**hyperparameters)
    functional: Any = paddle_functional
    padding = layer.padding
    if layer.padding_mode != "zeros":
        x = functional.pad(
            x, layer.reversed_padding, mode=layer.padding_mode, data_format=layer.data_format
        )
        padding = 0
    convolution = {1: functional.conv1d, 2: functional.conv2d, 3: functional.conv3d}[
        layer.spatial_dims
    ]
    return convolution(
        x,
        weight,
        bias=bias,
        stride=layer.stride,
        padding=padding,
        dilation=layer.dilation,
        groups=layer.groups,
        data_format=layer.data_format,
    )


def conv_layer_call(layer: Any, op_name: str, x: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """The recorded call of one convolution layer: tensor args and hyperparameters.

    Parameters
    ----------
    layer:
        The ``Conv1D``/``Conv2D``/``Conv3D`` instance.
    op_name:
        ``conv1d``, ``conv2d`` or ``conv3d``.
    x:
        The layer input.

    Returns
    -------
    tuple[tuple[Any, ...], dict[str, Any]]
        ``(x, weight, bias)`` and the keyword hyperparameters
        :func:`paddle_conv_layer_replay` takes.
    """

    padding_mode = str(getattr(layer, "_padding_mode", "zeros"))
    kwargs = {
        "spatial_dims": _SPATIAL_DIMS[op_name],
        "padding_mode": padding_mode,
        "reversed_padding": getattr(layer, "_reversed_padding_repeated_twice", None),
        "stride": layer._stride,
        "padding": layer._padding,
        "dilation": layer._dilation,
        "groups": layer._groups,
        "data_format": layer._data_format,
    }
    return (x, layer.weight, layer.bias), kwargs


def conv_layer_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """The input tensor of a convolution layer call (``x``, alias ``input``).

    Parameters
    ----------
    args, kwargs:
        The ``forward`` call's arguments after ``self``.

    Returns
    -------
    Any
        The input, or ``None`` when the call shape is not recognised.
    """

    if args:
        return args[0]
    for key in ("x", "input"):
        if key in kwargs:
            return kwargs[key]
    return None
