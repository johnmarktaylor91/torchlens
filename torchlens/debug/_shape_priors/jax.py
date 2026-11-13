"""JAX shape priors: Equinox and Flax NNX layers, raw functions by ``jax.eval_shape``.

Module roots read the first layer's attributes (``in_features``, kernel
shapes) in the library's own layout: Equinox layers take one unbatched
example with channels first, Flax NNX layers take a batch with channels last.
Raw function roots (``fn(x)`` or ``functools.partial(fn, params)``) have no
layer, so bounded hypotheses ordered by the dimensions of the arrays the
function reads are checked with ``jax.eval_shape``, which traces shapes and
computes nothing. A raw ``fn(params, x)`` with ``params`` unbound cannot be
probed with one input and yields no candidate.

Probes never mutate the caller's module: NNX probes run on a split-and-merged
copy under ``jax.eval_shape``, and :func:`state_scope` restores NNX state and
mode flags around the whole search.
"""

from __future__ import annotations

import contextlib
import inspect
import re
import sys
from collections.abc import Iterator, Sequence
from typing import Any

from . import ExecutedConstraint, ProbeError, SearchRequest, ShapeCandidate, ValueRange
from ._fn_roots import (
    array_dims,
    executed_constraint as _neutral_constraint,
    fill_array,
    function_candidates,
    names_framework,
)

#: Module whose presence in ``sys.modules`` lets the search route a callable
#: that names no framework to this prior.
FRAMEWORK = "jax"
_PREFIXES = ("jax", "jaxlib", "flax", "equinox")
_UNIFORM: ValueRange = ("uniform", 0.0, 1.0)
#: Mode attributes ``nnx.Module.eval`` flips; saved and restored per submodule.
_NNX_MODE_ATTRS = ("deterministic", "use_running_average")
_CONV_MISMATCH = re.compile(r"rhs input feature dimension size, but (\d+) // \d+ != (\d+)")
_SHAPE_WORDS = ("shape", "dimension", "incompatible", "broadcast", "reshape", "rank", "ndim")
_TRACER_ERRORS = (
    "ConcretizationTypeError",
    "TracerBoolConversionError",
    "TracerIntegerConversionError",
    "TracerArrayConversionError",
)


def _is_equinox(model: Any) -> bool:
    """Whether ``model`` is an ``eqx.Module`` (only if Equinox is imported)."""

    eqx = sys.modules.get("equinox")
    return eqx is not None and isinstance(model, eqx.Module)


def _is_nnx(model: Any) -> bool:
    """Whether ``model`` is an ``nnx.Module`` (only if Flax NNX is imported)."""

    nnx = sys.modules.get("flax.nnx")
    return nnx is not None and isinstance(model, nnx.Module)


def handles(model: Any) -> bool:
    """Whether ``model`` is a JAX root: an Equinox or NNX module or a JAX function.

    Parameters
    ----------
    model:
        The caller's model object.

    Returns
    -------
    bool
        ``False`` when JAX was never imported.
    """

    if "jax" not in sys.modules:
        return False
    if _is_equinox(model) or _is_nnx(model):
        return True
    return names_framework(model, _PREFIXES)


def lazy_reason(model: Any) -> str | None:
    """JAX layers initialize eagerly; nothing is lazy.

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


# ------------------------------------------------------------ module layers
def _nnx_children(node: Any, depth: int = 0) -> Iterator[Any]:
    """NNX submodules in attribute definition order, depth first."""

    nnx = sys.modules["flax.nnx"]
    values = list(vars(node).values()) if hasattr(node, "__dict__") else list(node)
    for value in values:
        if isinstance(value, nnx.Module):
            yield value
            if depth < 8:
                yield from _nnx_children(value, depth + 1)
        elif isinstance(value, (list, tuple)) and depth < 8:
            yield from _nnx_children(value, depth + 1)
        elif isinstance(value, dict) and depth < 8:
            yield from _nnx_children(list(value.values()), depth + 1)


def _eqx_layers(model: Any) -> Iterator[Any]:
    """Equinox submodules in field order (pytree flatten order)."""

    import equinox as eqx
    import jax

    def is_layer(value: Any) -> bool:
        """Stop flattening at known first-layer types."""

        return isinstance(value, (eqx.nn.Linear, eqx.nn.Conv, eqx.nn.Embedding, eqx.nn.LayerNorm))

    for leaf in jax.tree_util.tree_leaves(model, is_leaf=is_layer):
        if is_layer(leaf):
            yield leaf


def _side(kernel: Sequence[int], dilation: Sequence[int] | None, request: SearchRequest) -> int:
    """Smallest square side a kernel accepts, raised to ``min_size``."""

    dil = list(dilation or [1] * len(kernel))
    return max(request.min_size, max((k - 1) * d + 1 for k, d in zip(kernel, dil, strict=False)))


def _first_side(min_side: int, request: SearchRequest) -> int:
    """First preferred side at least ``min_side`` (else ``min_side``)."""

    fits = [s for s in request.preferred_sizes if min_side <= s <= request.max_size]
    return fits[0] if fits else min_side


def _eqx_flat(
    name: str, shape: tuple[int, ...], dtype: Any, kind: str = "linear"
) -> ShapeCandidate:
    """Determined unbatched candidate for an Equinox ``Linear``, ``LayerNorm`` or ``Embedding``."""

    value_range: ValueRange = _UNIFORM
    if kind == "embedding":
        value_range, dtype, shape = ("randint", 0, shape[0]), "int32", ()
    return ShapeCandidate(
        kind,
        shape,
        dtype,
        value_range,
        "C" if shape else "",
        determined=True,
        constraining_op=name,
        strategy="equinox_layer",
    )


def _eqx_conv(layer: Any, request: SearchRequest, dtype: Any) -> ShapeCandidate:
    """Unbatched channel-first candidate for an Equinox ``Conv`` (weight ``(out, in, *k)``)."""

    rank = int(layer.num_spatial_dims)
    min_side = _side(layer.kernel_size, layer.dilation, request)
    side = _first_side(min_side, request)
    channels = request.channels or int(layer.in_channels)
    return ShapeCandidate(
        "conv",
        (channels, *([side] * rank)),
        dtype,
        _UNIFORM,
        "C" + "DHW"[3 - rank :],
        spatial_axes=tuple(range(1, rank + 1)),
        constraining_op=type(layer).__name__,
        strategy="equinox_layer",
        min_side=min_side,
    )


def _eqx_candidate(layer: Any, request: SearchRequest, dtype: Any) -> ShapeCandidate | None:
    """Candidate for an Equinox first layer: one unbatched example."""

    import equinox as eqx

    name = type(layer).__name__
    if isinstance(layer, eqx.nn.Linear):
        width = layer.in_features
        return _eqx_flat(name, () if width == "scalar" else (int(width),), dtype)
    if isinstance(layer, eqx.nn.Embedding):
        return _eqx_flat(name, (int(layer.num_embeddings),), dtype, kind="embedding")
    if isinstance(layer, eqx.nn.LayerNorm) and layer.shape is not None:
        shape = (layer.shape,) if isinstance(layer.shape, int) else tuple(layer.shape)
        return _eqx_flat(name, shape, dtype)
    if isinstance(layer, eqx.nn.Conv):
        return _eqx_conv(layer, request, dtype)
    return None


def _nnx_candidate(layer: Any, request: SearchRequest, dtype: Any) -> ShapeCandidate | None:
    """Candidate for an NNX first layer: a batch with channels last."""

    nnx = sys.modules["flax.nnx"]
    batch, name = request.batch_size, type(layer).__name__
    if isinstance(layer, nnx.Embed):
        seq = request.seq_len or 8
        return ShapeCandidate(
            "embedding",
            (batch, seq),
            "int32",
            ("randint", 0, int(layer.num_embeddings)),
            "NL",
            flexible_dims=(1,),
            constraining_op=name,
            strategy="nnx_layer",
        )
    if isinstance(layer, nnx.Conv):
        kernel = tuple(int(k) for k in layer.kernel_size)
        min_side = _side(kernel, getattr(layer, "kernel_dilation", None), request)
        side = _first_side(min_side, request)
        channels = request.channels or int(layer.in_features)
        layout = "N" + "DHW"[3 - len(kernel) :] + "C"
        return ShapeCandidate(
            "conv",
            (batch, *([side] * len(kernel)), channels),
            dtype,
            _UNIFORM,
            layout,
            spatial_axes=tuple(range(1, len(kernel) + 1)),
            constraining_op=name,
            strategy="nnx_layer",
            min_side=min_side,
        )
    width = getattr(layer, "in_features", None) or getattr(layer, "num_features", None)
    if isinstance(width, int) and isinstance(
        layer, (nnx.Linear, nnx.LayerNorm, nnx.RMSNorm, nnx.BatchNorm)
    ):
        return ShapeCandidate(
            "linear",
            (batch, width),
            dtype,
            _UNIFORM,
            "NC",
            determined=True,
            constraining_op=name,
            strategy="nnx_layer",
        )
    return None


def _module_candidates(model: Any, request: SearchRequest, dtype: Any) -> list[ShapeCandidate]:
    """The first readable layer's candidate for an Equinox or NNX root."""

    if _is_equinox(model):
        layers: Iterator[Any] = _eqx_layers(model)
        build = _eqx_candidate
    else:
        layers = _nnx_children(model)
        build = _nnx_candidate
    for layer in layers:
        candidate = build(layer, request, dtype)
        if candidate is not None:
            return [candidate]
    return []


# ----------------------------------------------------------- function roots
def _unbound_positionals(fn: Any) -> int:
    """Required positional parameters a function root still needs."""

    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return 1
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return sum(1 for p in parameters if p.kind in kinds and p.default is inspect.Parameter.empty)


def _abstract_works(fn: Any, dtype: Any) -> Any:
    """A hypothesis check that traces ``fn`` with ``jax.eval_shape`` (no compute)."""

    import jax

    def works(shape: tuple[int, ...]) -> bool:
        """Whether ``fn`` traces on an abstract input of ``shape``."""

        try:
            jax.eval_shape(fn, jax.ShapeDtypeStruct(shape, dtype))
        except Exception:  # noqa: BLE001 - any trace failure rejects the hypothesis
            return False
        return True

    return works


def candidates(model: Any, request: SearchRequest) -> Sequence[ShapeCandidate]:
    """First-layer priors for module roots, abstract-probed hypotheses for functions.

    Parameters
    ----------
    model:
        A JAX root this module handles.
    request:
        The validated search arguments.

    Returns
    -------
    Sequence[ShapeCandidate]
        Best first; empty for a raw ``fn(params, x)`` whose ``params`` is
        not bound (bind it with ``functools.partial(fn, params)``).
    """

    import jax.numpy as jnp

    dtype = jnp.float32 if request.input_dtype is None else request.input_dtype
    if _is_equinox(model) or _is_nnx(model):
        return _module_candidates(model, request, dtype)
    if _unbound_positionals(model) != 1:
        return []
    return function_candidates(array_dims(model), request, dtype, _abstract_works(model, dtype))


def make_input(shape: tuple[int, ...], dtype: Any, value_range: ValueRange, seed: int) -> Any:
    """One deterministic JAX array (NumPy's private generator; JAX has no global RNG).

    Parameters
    ----------
    shape, dtype, value_range, seed:
        As in :class:`~torchlens.debug._shape_priors.ShapeCandidate`.

    Returns
    -------
    Any
        A ``jax.Array``.
    """

    import jax.numpy as jnp
    import numpy as np

    return jnp.asarray(fill_array(shape, value_range, seed, np.dtype(dtype).name))


def _abstract_call(model: Any, example: Any) -> Any:
    """Trace one call with ``jax.eval_shape``; NNX runs on a merged copy."""

    import jax

    if _is_nnx(model):
        nnx = sys.modules["flax.nnx"]
        graphdef, state = nnx.split(model)
        return jax.eval_shape(lambda s, x: nnx.merge(graphdef, s)(x), state, example)
    if _is_equinox(model):
        import equinox as eqx

        return eqx.filter_eval_shape(model, example)
    return jax.eval_shape(model, example)


def forward(model: Any, example: Any) -> Any:
    """Probe one call by ``jax.eval_shape``; run it for real only on value-dependent code.

    Parameters
    ----------
    model:
        A JAX root this module handles.
    example:
        One input array.

    Returns
    -------
    Any
        Abstract outputs, or realized outputs when the forward branches on
        values and cannot be traced abstractly.
    """

    import jax

    try:
        return _abstract_call(model, example)
    except Exception as exc:  # noqa: BLE001 - only tracer conversions fall back
        if type(exc).__name__ not in _TRACER_ERRORS:
            raise
    return jax.block_until_ready(model(example))


def classify_error(exc: Exception) -> ProbeError:
    """Sort a JAX probe exception into shape, dtype, device or blocker.

    Parameters
    ----------
    exc:
        The exception a probe raised.

    Returns
    -------
    ProbeError
        The classification, with measured channel counts for convolutions.
    """

    message = str(exc)
    lowered = message.lower()
    conv = _CONV_MISMATCH.search(message)
    if conv:
        return ProbeError(
            "shape", message, got_features=int(conv.group(1)), target_features=int(conv.group(2))
        )
    if "integer or boolean" in lowered or "indexer must" in lowered:
        return ProbeError("dtype", message, suggested_dtype="int32")
    if any(word in lowered for word in _SHAPE_WORDS):
        return ProbeError("shape", message)
    if "device" in lowered:
        return ProbeError("device", message)
    return ProbeError("blocker", message)


def executed_constraint(trace: Any) -> ExecutedConstraint | None:
    """The first ``dot_general``, convolution or gather the trace executed.

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


def _nnx_modes(model: Any) -> list[tuple[Any, str, Any]]:
    """Every NNX mode flag ``eval()`` may flip, with its current value."""

    saved = []
    for module in (model, *_nnx_children(model)):
        for attr in _NNX_MODE_ATTRS:
            if attr in vars(module):
                saved.append((module, attr, vars(module)[attr]))
    return saved


@contextlib.contextmanager
def state_scope(model: Any) -> Iterator[None]:
    """Restore NNX state (RNG counters, statistics) and mode flags on exit.

    Equinox modules are immutable and raw functions hold no state, and JAX has
    no global RNG, so only NNX roots save anything.

    Parameters
    ----------
    model:
        A JAX root this module handles.

    Yields
    ------
    None
        Inside the scope NNX roots are in ``eval()`` mode.
    """

    if not _is_nnx(model):
        yield
        return
    nnx = sys.modules["flax.nnx"]
    saved_state, saved_modes = nnx.state(model), _nnx_modes(model)
    try:
        eval_mode = getattr(model, "eval", None)
        if callable(eval_mode):
            eval_mode()
        yield
    finally:
        nnx.update(model, saved_state)
        for module, attr, value in saved_modes:
            setattr(module, attr, value)


__all__ = [
    "candidates",
    "classify_error",
    "executed_constraint",
    "forward",
    "handles",
    "lazy_reason",
    "make_input",
    "state_scope",
]
