"""tinygrad shape priors: layer weights read channel-first, plus closure weights.

tinygrad models are plain objects and functions holding ``tinygrad.nn`` layers,
so candidates come from the layers reachable from the model in definition order
(:func:`~torchlens.debug._shape_priors.walk_references`): ``Linear`` weights are
``[out, in]``, ``Conv2d`` (and the ``Conv1d`` helper, which returns one) weights
are ``[out, in / groups, *kernel]`` over channel-first inputs, ``Embedding``
weights are ``[vocab, dim]``. Rank-2 closure tensors and the generic ladder
follow. See :mod:`torchlens.debug._shape_priors` for the contract.
"""

from __future__ import annotations

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
    framework_of,
    generic_ladder,
    integer_range,
    parse_feature_mismatch,
    walk_references,
)

FRAMEWORK = "tinygrad"

_UNIFORM: ValueRange = ("uniform", 0.0, 1.0)
_MAX_LAYER_CANDIDATES = 4
#: tinygrad class attributes that carry RNG state (copied on entry, restored on exit).
_RNG_ATTRIBUTES = ("_seed", "_device_seeds", "_device_rng_counters")


def handles(model: Any) -> bool:
    """Whether ``model`` reaches a tinygrad tensor or layer."""

    if "tinygrad" not in sys.modules or not callable(model):
        return False
    return any(framework_of(obj) == FRAMEWORK for _path, obj in walk_references(model)[1:])


def lazy_reason(model: Any) -> str | None:
    """tinygrad layers allocate their weights in ``__init__``; nothing is lazy."""

    del model
    return None


def candidates(model: Any, request: SearchRequest) -> Sequence[ShapeCandidate]:
    """Layer priors, then closure-weight priors, then the generic ladder."""

    from tinygrad import Tensor, dtypes

    dtype = request.input_dtype if request.input_dtype is not None else dtypes.float32
    found: list[ShapeCandidate] = []
    references = walk_references(model)
    for path, obj in references:
        candidate = _layer_candidate(path, obj, request, dtype)
        if candidate is not None:
            found.append(candidate)
        if len(found) >= _MAX_LAYER_CANDIDATES:
            break
    layer_weights = {id(getattr(obj, "weight", None)) for _path, obj in references}
    tensors = [
        (path, tuple(int(dim) for dim in obj.shape))
        for path, obj in references
        if isinstance(obj, Tensor) and id(obj) not in layer_weights and not path.endswith(".weight")
    ]
    found.extend(closure_matrix_candidates(tensors, request, dtype))
    found.extend(generic_ladder(request, dtype))
    return found


def _layer_candidate(
    path: str, layer: Any, request: SearchRequest, dtype: Any
) -> ShapeCandidate | None:
    """The candidate one tinygrad layer pins, or ``None``."""

    from tinygrad import dtypes, nn

    weight = getattr(layer, "weight", None)
    if weight is None or not hasattr(weight, "shape"):
        return None
    shape = tuple(int(dim) for dim in weight.shape)
    address, op = path or None, type(layer).__name__
    if isinstance(layer, nn.Linear) and len(shape) == 2:
        return ShapeCandidate(
            "linear",
            (request.batch_size, shape[1]),
            dtype,
            _UNIFORM,
            "NC",
            determined=True,
            constraining_module=address,
            constraining_op=op,
            strategy="tinygrad_layer_prior",
        )
    if isinstance(layer, nn.Conv2d) and len(shape) >= 3:
        return _conv_candidate(layer, request, dtype, address, op)
    if isinstance(layer, nn.Embedding) and len(shape) == 2:
        return ShapeCandidate(
            "embedding",
            (request.batch_size, request.seq_len or 8),
            dtypes.int32,
            integer_range(shape[0]),
            "NL",
            flexible_dims=(1,),
            determined=True,
            constraining_module=address,
            constraining_op=op,
            strategy="tinygrad_layer_prior",
        )
    return None


def _conv_candidate(
    layer: Any, request: SearchRequest, dtype: Any, address: str | None, op: str
) -> ShapeCandidate:
    """A channel-first convolution candidate."""

    weight_shape = tuple(int(dim) for dim in layer.weight.shape)
    rank = len(weight_shape) - 2
    groups = int(getattr(layer, "groups", 1) or 1)
    channels = request.channels or weight_shape[1] * groups
    kernel = max(weight_shape[2:])
    side = max(32, kernel)
    spatial = tuple(range(2, rank + 2))
    return ShapeCandidate(
        "conv",
        (request.batch_size, channels, *([side] * rank)),
        dtype,
        _UNIFORM,
        ("NCL", "NCHW", "NCDHW")[min(rank, 3) - 1],
        spatial_axes=spatial,
        flexible_dims=spatial,
        determined=True,
        constraining_module=address,
        constraining_op=op,
        strategy="tinygrad_layer_prior",
        min_side=kernel,
    )


def make_input(shape: tuple[int, ...], dtype: Any, value_range: ValueRange, seed: int) -> Any:
    """A deterministic tinygrad tensor built from NumPy (tinygrad's RNG stays untouched)."""

    import numpy as np
    from tinygrad import Tensor

    rng = np.random.default_rng(seed)
    if value_range[0] == "randint":
        values = rng.integers(int(value_range[1]), int(value_range[2]), size=shape).astype("int32")
    else:
        values = rng.uniform(value_range[1], value_range[2], size=shape).astype("float32")
    return Tensor(values).cast(dtype)


def forward(model: Any, example: Any) -> Any:
    """One forward with every output tensor realized."""

    from tinygrad import Tensor

    output = model(example)
    leaves = output if isinstance(output, (tuple, list)) else (output,)
    for leaf in leaves:
        if isinstance(leaf, Tensor):
            leaf.realize()
    return output


def classify_error(exc: Exception) -> ProbeError:
    """Sort a tinygrad forward error; shape asserts dominate."""

    message = " ".join(str(exc).split())[:400] or type(exc).__name__
    got, target = parse_feature_mismatch(message)
    lowered = message.lower()
    if "dtype" in lowered and got is None:
        from tinygrad import dtypes

        suggested = dtypes.int32 if "integer dtype" in lowered else None
        return ProbeError("dtype", message, suggested_dtype=suggested)
    if "device" in lowered and "shape" not in lowered:
        return ProbeError("device", message)
    if got is not None or isinstance(exc, (AssertionError, ValueError, IndexError, RuntimeError)):
        return ProbeError("shape", message, got_features=got, target_features=target)
    return ProbeError("blocker", message)


def executed_constraint(trace: Any) -> ExecutedConstraint | None:
    """No executed constraint: tinygrad records UOps, where a matmul or conv is
    already decomposed into reshape, expand, multiply and reduce, so the
    first-layer prior names the constraining layer instead."""

    del trace
    return None


@contextmanager
def state_scope(model: Any) -> Iterator[None]:
    """Save ``Tensor.training`` and the tinygrad RNG attributes; restore on exit."""

    from tinygrad import Tensor

    del model
    training = Tensor.training
    saved = {
        name: _copy(getattr(Tensor, name)) for name in _RNG_ATTRIBUTES if hasattr(Tensor, name)
    }
    try:
        Tensor.training = False
        yield
    finally:
        Tensor.training = training
        for name, value in saved.items():
            setattr(Tensor, name, value)


def _copy(value: Any) -> Any:
    """A shallow copy of a dict or list RNG attribute; other values as they are."""

    return (
        dict(value)
        if isinstance(value, dict)
        else list(value)
        if isinstance(value, list)
        else value
    )
