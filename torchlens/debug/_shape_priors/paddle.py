"""PaddlePaddle shape priors: layer dimensions, ``data_format`` and closure weights.

Candidates come from the first constraining sublayers of a ``paddle.nn.Layer``
in definition order (``Linear`` weights are ``[in, out]``; ``Conv1D/2D/3D``
weights are ``[out, in / groups, *kernel]`` with the layer's ``data_format``;
``Embedding`` and the recurrent layers read their sizes), then from rank-2
closure tensors of a plain function, then a small generic ladder that is never
called determined. See :mod:`torchlens.debug._shape_priors` for the contract.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from . import (
    ExecutedConstraint,
    ProbeError,
    SearchRequest,
    ShapeCandidate,
    ValueRange,
    closure_matrix_candidates,
    constraint_from_trace,
    framework_of,
    generic_ladder,
    integer_range,
    parse_feature_mismatch,
    walk_references,
)

FRAMEWORK = "paddle"

_UNIFORM: ValueRange = ("uniform", 0.0, 1.0)
#: Paddle error text naming an expected dtype.
_DTYPE_RE = re.compile(
    r"(?:expected|must be|should be|only supports?)[^.]*?\b(int64|int32|float32|float64)\b", re.I
)
_DEVICE_RE = re.compile(r"place|device|GPU|CUDA", re.I)
_SHAPE_RE = re.compile(r"shape|dim|rank|size|InvalidArgument|mismatch", re.I)
_CONSTRAINT_KINDS = (
    ("conv", "conv"),
    ("embedding", "embedding"),
    ("linear", "linear"),
    ("matmul", "matmul"),
)
_MAX_LAYER_CANDIDATES = 4


def handles(model: Any) -> bool:
    """Whether ``model`` is a Paddle layer or a callable over Paddle objects."""

    paddle = sys.modules.get("paddle")
    if paddle is None:
        return False
    if isinstance(model, paddle.nn.Layer):
        return True
    if not callable(model):
        return False
    return any(framework_of(obj) == FRAMEWORK for _path, obj in walk_references(model)[1:])


def lazy_reason(model: Any) -> str | None:
    """Paddle dygraph layers create their weights eagerly; nothing is lazy."""

    del model
    return None


def candidates(model: Any, request: SearchRequest) -> Sequence[ShapeCandidate]:
    """Layer priors, then closure-weight priors, then the generic ladder."""

    import paddle

    dtype = _float_dtype(request.input_dtype)
    found: list[ShapeCandidate] = []
    if isinstance(model, paddle.nn.Layer):
        layers = [(name, layer) for name, layer in model.named_sublayers(include_self=True)]
    else:
        layers = [
            (path, obj) for path, obj in walk_references(model) if isinstance(obj, paddle.nn.Layer)
        ]
    for name, layer in layers:
        candidate = _layer_candidate(name, layer, request, dtype)
        if candidate is not None:
            found.append(candidate)
        if len(found) >= _MAX_LAYER_CANDIDATES:
            break
    tensors = [
        (path, tuple(int(dim) for dim in obj.shape))
        for path, obj in walk_references(model)
        if isinstance(obj, paddle.Tensor) and not isinstance(model, paddle.nn.Layer)
    ]
    found.extend(closure_matrix_candidates(tensors, request, dtype))
    found.extend(generic_ladder(request, dtype))
    return found


def _float_dtype(requested: Any) -> Any:
    """The caller's dtype, or ``paddle.float32``."""

    import paddle

    if requested is None:
        return paddle.float32
    return getattr(paddle, requested) if isinstance(requested, str) else requested


def _layer_candidate(
    name: str, layer: Any, request: SearchRequest, dtype: Any
) -> ShapeCandidate | None:
    """The candidate one Paddle layer pins, or ``None`` when it pins nothing."""

    import paddle

    nn = paddle.nn
    batch = request.batch_size
    address = name or None
    op = type(layer).__name__
    if isinstance(layer, nn.Linear):
        features = int(layer.weight.shape[0])
        return ShapeCandidate(
            "linear",
            (batch, features),
            dtype,
            _UNIFORM,
            "NC",
            determined=True,
            constraining_module=address,
            constraining_op=op,
            strategy="paddle_layer_prior",
        )
    if isinstance(layer, (nn.Conv1D, nn.Conv2D, nn.Conv3D)):
        return _conv_candidate(layer, request, dtype, address, op)
    if isinstance(layer, nn.Embedding):
        vocab = int(layer.weight.shape[0])
        seq = request.seq_len or 8
        return ShapeCandidate(
            "embedding",
            (batch, seq),
            paddle.int64,
            integer_range(vocab),
            "NL",
            flexible_dims=(1,),
            determined=True,
            constraining_module=address,
            constraining_op=op,
            strategy="paddle_layer_prior",
        )
    if isinstance(layer, (nn.LSTM, nn.GRU, nn.SimpleRNN)):
        return _rnn_candidate(layer, request, dtype, address, op)
    return None


def _conv_candidate(
    layer: Any, request: SearchRequest, dtype: Any, address: str | None, op: str
) -> ShapeCandidate:
    """A convolution candidate in the layer's own ``data_format``."""

    weight_shape = tuple(int(dim) for dim in layer.weight.shape)
    rank = len(weight_shape) - 2
    groups = int(getattr(layer, "_groups", 1) or 1)
    channels = request.channels or weight_shape[1] * groups
    kernel = max(weight_shape[2:]) if rank else 1
    side = max(32, kernel)
    data_format = str(getattr(layer, "_data_format", "") or ("NCL", "NCHW", "NCDHW")[rank - 1])
    channel_last = data_format.endswith("C")
    spatial = tuple(range(1, rank + 1)) if channel_last else tuple(range(2, rank + 2))
    shape = (
        (request.batch_size, *([side] * rank), channels)
        if channel_last
        else (request.batch_size, channels, *([side] * rank))
    )
    return ShapeCandidate(
        "conv",
        shape,
        dtype,
        _UNIFORM,
        data_format,
        spatial_axes=spatial,
        flexible_dims=spatial,
        determined=True,
        constraining_module=address,
        constraining_op=op,
        strategy="paddle_layer_prior",
        min_side=kernel,
    )


def _rnn_candidate(
    layer: Any, request: SearchRequest, dtype: Any, address: str | None, op: str
) -> ShapeCandidate:
    """A recurrent candidate, batch-major unless the layer is time-major."""

    size = int(layer.input_size)
    seq = request.seq_len or 8
    time_major = bool(getattr(layer, "time_major", False))
    shape = (seq, request.batch_size, size) if time_major else (request.batch_size, seq, size)
    flexible = (0,) if time_major else (1,)
    return ShapeCandidate(
        "sequence",
        shape,
        dtype,
        _UNIFORM,
        "LNC" if time_major else "NLC",
        flexible_dims=flexible,
        determined=True,
        constraining_module=address,
        constraining_op=op,
        strategy="paddle_layer_prior",
    )


def make_input(shape: tuple[int, ...], dtype: Any, value_range: ValueRange, seed: int) -> Any:
    """A deterministic Paddle tensor drawn with NumPy (the Paddle RNG stays untouched)."""

    import numpy as np
    import paddle

    rng = np.random.default_rng(seed)
    if value_range[0] == "randint":
        values = rng.integers(int(value_range[1]), int(value_range[2]), size=shape)
        return paddle.to_tensor(values, dtype=dtype)
    values = rng.uniform(value_range[1], value_range[2], size=shape).astype("float32")
    return paddle.to_tensor(values).astype(dtype)


def forward(model: Any, example: Any) -> Any:
    """One plain forward under ``paddle.no_grad``."""

    import paddle

    with paddle.no_grad():
        return model(example)


def classify_error(exc: Exception) -> ProbeError:
    """Sort a Paddle forward error into dtype, device, shape or blocker."""

    message = " ".join(str(exc).split())[:400]
    suggested = _suggested_dtype(message)
    if suggested is not None:
        return ProbeError("dtype", message, suggested_dtype=suggested)
    if _DEVICE_RE.search(message) and "shape" not in message.lower():
        return ProbeError("device", message)
    got, target = parse_feature_mismatch(message)
    if got is not None or _SHAPE_RE.search(message) or isinstance(exc, (ValueError, IndexError)):
        return ProbeError("shape", message, got_features=got, target_features=target)
    return ProbeError("blocker", message)


def _suggested_dtype(message: str) -> Any:
    """The Paddle dtype an error message asks for (``int64`` when it allows both ints)."""

    lowered = message.lower()
    if "dtype" not in lowered and "support" not in lowered:
        return None
    match = _DTYPE_RE.search(message)
    if match is None:
        return None
    import paddle

    name = (
        "int64"
        if "int64" in lowered and "int32" in match.group(0).lower()
        else match.group(1).lower()
    )
    return getattr(paddle, name)


def executed_constraint(trace: Any) -> ExecutedConstraint | None:
    """The first conv, embedding, linear or matmul op the trace executed."""

    return constraint_from_trace(trace, _CONSTRAINT_KINDS)


@contextmanager
def state_scope(model: Any) -> Iterator[None]:
    """Save every layer's training flag and the Paddle RNG; eval inside; restore on exit."""

    import paddle

    layers = (
        list(model.sublayers(include_self=True))
        if isinstance(model, paddle.nn.Layer)
        else [obj for _path, obj in walk_references(model) if isinstance(obj, paddle.nn.Layer)]
    )
    flags = [(layer, layer.training) for layer in layers]
    rng_state = paddle.get_rng_state()
    try:
        for layer in layers:
            layer.eval()
        yield
    finally:
        for layer, training in flags:
            layer.training = training
        paddle.set_rng_state(rng_state)
