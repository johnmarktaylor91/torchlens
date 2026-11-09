"""Paddle's op spelling list (see :mod:`torchlens.backends._op_spelling`).

Keys are the ``func_name`` each wrapped op records: the wrap inventory key
from :mod:`.wrappers` (bare top-level names, ``functional.<name>``,
``tensor.<name>``) and the convolution layer ops of :mod:`.conv_layers`.
Used only for candidate preference and drawing labels; never for pairing.
"""

from __future__ import annotations

_TOP_SAME = [
    "abs",
    "add",
    "argmax",
    "argmin",
    "cos",
    "einsum",
    "exp",
    "flatten",
    "floor",
    "full",
    "linspace",
    "log",
    "matmul",
    "mean",
    "mm",
    "ones",
    "pow",
    "prod",
    "reshape",
    "sin",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "sum",
    "tanh",
    "tile",
    "unsqueeze",
    "var",
    "where",
    "zeros",
    "conv1d",
    "conv2d",
    "conv3d",
]
_FUNCTIONAL_SAME = [
    "conv1d",
    "conv2d",
    "dropout",
    "gelu",
    "hardsigmoid",
    "hardswish",
    "identity",
    "linear",
    "relu",
    "relu6",
    "sigmoid",
    "silu",
    "softmax",
    "tanh",
]
_TENSOR_SAME = [
    "abs",
    "add",
    "contiguous",
    "exp",
    "flatten",
    "log",
    "matmul",
    "mean",
    "pow",
    "prod",
    "reshape",
    "rsqrt",
    "sqrt",
    "square",
    "squeeze",
    "std",
    "sum",
    "t",
    "tile",
    "unsqueeze",
    "var",
]
_SHARED_RENAMES = {
    "cast": "to",
    "clip": "clamp",
    "divide": "div",
    "max": "amax",
    "min": "amin",
    "multiply": "mul",
    "subtract": "sub",
    # paddle.transpose takes a full permutation (torch.permute).
    "transpose": "permute",
}
_TOP_RENAMES = {
    "concat": "cat",
    "equal": "eq",
    "full_like": "fulllike",
    "greater_equal": "ge",
    "greater_than": "gt",
    "less_equal": "le",
    "less_than": "lt",
    "negative": "neg",
    "ones_like": "oneslike",
    "to_tensor": "tensor",
    "zeros_like": "zeroslike",
}
_FUNCTIONAL_RENAMES = {
    "adaptive_avg_pool1d": "adaptiveavgpool1d",
    "adaptive_avg_pool2d": "adaptiveavgpool2d",
    "adaptive_avg_pool3d": "adaptiveavgpool3d",
    "avg_pool1d": "avgpool1d",
    "avg_pool2d": "avgpool2d",
    "avg_pool3d": "avgpool3d",
    "batch_norm": "batchnorm",
    "layer_norm": "layernorm",
    "leaky_relu": "leakyrelu",
    "max_pool1d": "maxpool1d",
    "max_pool2d": "maxpool2d",
    "max_pool3d": "maxpool3d",
}
_TENSOR_RENAMES = {
    "__add__": "add",
    "__radd__": "add",
    "__sub__": "sub",
    "__rsub__": "rsub",
    "__mul__": "mul",
    "__rmul__": "mul",
    "__truediv__": "div",
    "__rtruediv__": "rtruediv",
    "__pow__": "pow",
    "__rpow__": "rpow",
    "__matmul__": "matmul",
    "__rmatmul__": "rmatmul",
    "__neg__": "neg",
    "__getitem__": "getitem",
    "__eq__": "eq",
    "__ne__": "ne",
    "__gt__": "gt",
    "__ge__": "ge",
    "__lt__": "lt",
    "__le__": "le",
    "astype": "to",
}

#: Paddle ``func_name`` to the canonical torch spelling.
SPELLINGS: dict[str, str] = (
    {name: name for name in _TOP_SAME}
    | _SHARED_RENAMES
    | _TOP_RENAMES
    | {f"functional.{name}": name for name in _FUNCTIONAL_SAME}
    | {f"functional.{name}": canonical for name, canonical in _FUNCTIONAL_RENAMES.items()}
    | {f"tensor.{name}": name for name in _TENSOR_SAME}
    | {f"tensor.{name}": canonical for name, canonical in _SHARED_RENAMES.items()}
    | {f"tensor.{name}": canonical for name, canonical in _TENSOR_RENAMES.items()}
)

#: Paddle ``func_name`` values with no canonical torch spelling, and their families.
UNMAPPED: dict[str, str] = {
    "assign": "identity",
    # Tensor.scale computes ``scale * x + bias``.
    "tensor.scale": "arith",
}
