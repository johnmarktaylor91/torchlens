"""MLX shape priors: module dimensions in channel-last layout, lazy probes for functions.

``mlx.nn`` layers store their weights in channel-last order (``Linear``
``(out, in)``, ``Conv2d`` ``(out, kh, kw, in)``), so the first layer's weight
shape pins the input's feature or channel axis and the input layout is
``NLC``/``NHWC``/``NDHWC``. Reading a lazy weight's ``shape`` does not evaluate
it. Function roots get bounded hypotheses; MLX checks shapes while building
the lazy graph, so a hypothesis is rejected without computing anything.
"""

from __future__ import annotations

import contextlib
import re
import sys
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from . import ExecutedConstraint, ProbeError, SearchRequest, ShapeCandidate, ValueRange
from ._fn_roots import (
    array_dims,
    executed_constraint as _neutral_constraint,
    fill_array,
    function_candidates,
    names_framework,
    parse_ints,
)

#: Module whose presence in ``sys.modules`` lets the search route a callable
#: that names no framework to this prior.
FRAMEWORK = "mlx.core"
_PREFIXES = ("mlx",)
_UNIFORM: ValueRange = ("uniform", 0.0, 1.0)
_MATMUL_MISMATCH = re.compile(
    r"first input with shape \(([\d,\s]+)\) must match second to last dimension of "
    r"second input with shape \(([\d,\s]+)\)"
)
_CONV_MISMATCH = re.compile(r"input: \(([\d,\s]+)\) and weight: \(([\d,\s]+)\)")
_SHAPE_WORDS = ("shape", "dimension", "broadcast", "reshape", "ndim", "rank")
_CONV_LAYOUTS = {1: "NLC", 2: "NHWC", 3: "NDHWC"}


def _nn() -> Any:
    """The imported ``mlx.nn`` module, or ``None``."""

    return sys.modules.get("mlx.nn")


def _is_module(model: Any) -> bool:
    """Whether ``model`` is an ``mlx.nn.Module`` (only if MLX is imported)."""

    nn = _nn()
    return nn is not None and isinstance(model, nn.Module)


def handles(model: Any) -> bool:
    """Whether ``model`` is an MLX module or a function that uses MLX.

    Parameters
    ----------
    model:
        The caller's model object.

    Returns
    -------
    bool
        ``False`` when MLX was never imported.
    """

    if "mlx.core" not in sys.modules:
        return False
    return _is_module(model) or names_framework(model, _PREFIXES)


def lazy_reason(model: Any) -> str | None:
    """MLX creates every weight at construction (lazily evaluated); nothing is unbuilt.

    Parameters
    ----------
    model:
        The caller's model object.

    Returns
    -------
    str | None
        Always ``None``.
    """

    del model
    return None


def _children(node: Any, depth: int = 0) -> Iterator[Any]:
    """MLX submodules in attribute definition order, depth first."""

    nn = _nn()
    values = list(node.values()) if isinstance(node, dict) else list(node)
    for value in values:
        if isinstance(value, nn.Module):
            yield value
            if depth < 8:
                yield from _children(value, depth + 1)
        elif isinstance(value, (list, tuple, dict)) and depth < 8:
            yield from _children(value, depth + 1)


def _weight_shape(layer: Any) -> tuple[int, ...] | None:
    """Shape of a layer's ``weight`` without evaluating it."""

    weight = layer.get("weight") if isinstance(layer, dict) else None
    shape = getattr(weight, "shape", None)
    return tuple(int(dim) for dim in shape) if shape is not None else None


def _first_side(min_side: int, request: SearchRequest) -> int:
    """First preferred side at least ``min_side`` (else ``min_side``)."""

    fits = [s for s in request.preferred_sizes if min_side <= s <= request.max_size]
    return fits[0] if fits else min_side


def _conv_candidate(
    layer: Any, weight: tuple[int, ...], request: SearchRequest, dtype: Any
) -> ShapeCandidate:
    """Channel-last candidate for ``Conv1d``/``Conv2d``/``Conv3d`` (weight ``(out, *k, in)``)."""

    kernel = weight[1:-1]
    dilation = getattr(layer, "dilation", 1)
    dilations = dilation if isinstance(dilation, (tuple, list)) else [dilation] * len(kernel)
    min_side = max(
        request.min_size, *((k - 1) * d + 1 for k, d in zip(kernel, dilations, strict=False))
    )
    side = _first_side(min_side, request)
    groups = int(getattr(layer, "groups", 1) or 1)
    channels = request.channels or weight[-1] * groups
    return ShapeCandidate(
        "conv",
        (request.batch_size, *([side] * len(kernel)), channels),
        dtype,
        _UNIFORM,
        _CONV_LAYOUTS[len(kernel)],
        spatial_axes=tuple(range(1, len(kernel) + 1)),
        constraining_op=type(layer).__name__,
        strategy="mlx_layer",
        min_side=min_side,
    )


def _embedding_candidate(layer: Any, request: SearchRequest) -> ShapeCandidate | None:
    """Token candidate for ``Embedding`` and ``QuantizedEmbedding``."""

    vocab = getattr(layer, "num_embeddings", None)
    if vocab is None:
        weight = _weight_shape(layer)
        vocab = weight[0] if weight else None
    if vocab is None:
        return None
    return ShapeCandidate(
        "embedding",
        (request.batch_size, request.seq_len or 8),
        "int32",
        ("randint", 0, int(vocab)),
        "NL",
        flexible_dims=(1,),
        constraining_op=type(layer).__name__,
        strategy="mlx_layer",
    )


def _layer_candidate(layer: Any, request: SearchRequest, dtype: Any) -> ShapeCandidate | None:
    """Candidate for one MLX layer, or ``None`` when it pins nothing readable."""

    nn, name = _nn(), type(layer).__name__
    if "Embedding" in name:
        return _embedding_candidate(layer, request)
    weight = _weight_shape(layer)
    if weight is None:
        return None
    if name.startswith("Conv") and not name.startswith("ConvTranspose") and len(weight) >= 3:
        return _conv_candidate(layer, weight, request, dtype)
    if isinstance(layer, nn.Linear) and len(weight) == 2:
        width = weight[1]
    elif name in {"LayerNorm", "RMSNorm", "BatchNorm"} and len(weight) == 1:
        width = weight[0]
    else:
        return None
    return ShapeCandidate(
        "linear",
        (request.batch_size, width),
        dtype,
        _UNIFORM,
        "NC",
        determined=True,
        constraining_op=name,
        strategy="mlx_layer",
    )


@contextlib.contextmanager
def _rng_restored() -> Iterator[None]:
    """Save MLX's global PRNG key and put it back on exit."""

    import mlx.core as mx

    saved = mx.random.state[0]
    try:
        yield
    finally:
        mx.random.state[0] = saved


def _lazy_works(fn: Any, dtype: Any) -> Callable[[tuple[int, ...]], bool]:
    """A hypothesis check that builds ``fn``'s lazy graph without evaluating it."""

    import mlx.core as mx

    def works(shape: tuple[int, ...]) -> bool:
        """Whether ``fn`` builds a graph for an input of ``shape``."""

        try:
            with _rng_restored():
                fn(mx.zeros(shape, dtype=dtype))
        except Exception:  # noqa: BLE001 - any build failure rejects the hypothesis
            return False
        return True

    return works


def candidates(model: Any, request: SearchRequest) -> Sequence[ShapeCandidate]:
    """First-layer prior for modules, lazily probed hypotheses for functions.

    Parameters
    ----------
    model:
        An MLX root this module handles.
    request:
        The validated search arguments.

    Returns
    -------
    Sequence[ShapeCandidate]
        Best first; empty when no layer has a readable weight.
    """

    import mlx.core as mx

    dtype = mx.float32 if request.input_dtype is None else request.input_dtype
    if not _is_module(model):
        return function_candidates(array_dims(model), request, dtype, _lazy_works(model, dtype))
    for layer in _children(model):
        candidate = _layer_candidate(layer, request, dtype)
        if candidate is not None:
            return [candidate]
    return []


def _np_dtype_name(dtype: Any) -> str:
    """NumPy dtype name for an MLX dtype or a dtype name."""

    if isinstance(dtype, str):
        return dtype
    return str(dtype).rsplit(".", 1)[-1]


def make_input(shape: tuple[int, ...], dtype: Any, value_range: ValueRange, seed: int) -> Any:
    """One deterministic MLX array (NumPy's private generator; MLX's key untouched).

    Parameters
    ----------
    shape, dtype, value_range, seed:
        As in :class:`~torchlens.debug._shape_priors.ShapeCandidate`.

    Returns
    -------
    Any
        An ``mlx.core.array``.
    """

    import mlx.core as mx

    name = _np_dtype_name(dtype)
    host_dtype = "float32" if name == "bfloat16" else name
    array = mx.array(fill_array(shape, value_range, seed, host_dtype))
    return array.astype(getattr(mx, name)) if name != host_dtype else array


def forward(model: Any, example: Any) -> Any:
    """One realized forward pass (``mx.eval`` on the outputs).

    Parameters
    ----------
    model:
        An MLX root this module handles.
    example:
        One input array.

    Returns
    -------
    Any
        The evaluated outputs.
    """

    import mlx.core as mx

    out = model(example)
    mx.eval(out)
    return out


def classify_error(exc: Exception) -> ProbeError:
    """Sort an MLX probe exception into shape, dtype, device or blocker.

    Parameters
    ----------
    exc:
        The exception a probe raised.

    Returns
    -------
    ProbeError
        The classification, with measured feature or channel counts when the
        message names them.
    """

    message = str(exc)
    lowered = message.lower()
    for pattern in (_MATMUL_MISMATCH, _CONV_MISMATCH):
        found = pattern.search(message)
        if found:
            first, second = parse_ints(found.group(1)), parse_ints(found.group(2))
            target = second[-2] if pattern is _MATMUL_MISMATCH and len(second) > 1 else second[-1]
            return ProbeError("shape", message, got_features=first[-1], target_features=target)
    if "integral" in lowered or ("indices" in lowered and "dtype" in lowered):
        return ProbeError("dtype", message, suggested_dtype="int32")
    if any(word in lowered for word in _SHAPE_WORDS):
        return ProbeError("shape", message)
    if "device" in lowered or "stream" in lowered:
        return ProbeError("device", message)
    return ProbeError("blocker", message)


def executed_constraint(trace: Any) -> ExecutedConstraint | None:
    """The first matmul, convolution or gather the trace executed.

    Parameters
    ----------
    trace:
        The verification trace.

    Returns
    -------
    ExecutedConstraint | None
        The first constraining op, or ``None``.
    """

    return _neutral_constraint(trace)


def _training_flags(model: Any) -> list[tuple[int, Any, bool]]:
    """Each module's training flag with its depth (root first)."""

    flags = [(0, model, bool(model.training))]
    for name, module in model.named_modules():
        if module is not model:
            flags.append((name.count(".") + 1, module, bool(module.training)))
    return sorted(flags, key=lambda item: item[0])


@contextlib.contextmanager
def state_scope(model: Any) -> Iterator[None]:
    """Switch MLX modules to ``eval()`` and restore training flags and the PRNG key.

    Parameters
    ----------
    model:
        An MLX root this module handles.

    Yields
    ------
    None
        Inside the scope modules are in inference mode.
    """

    flags = _training_flags(model) if _is_module(model) else []
    with _rng_restored():
        try:
            if flags:
                model.eval()
            yield
        finally:
            for _depth, module, training in flags:
                module.train(training)
