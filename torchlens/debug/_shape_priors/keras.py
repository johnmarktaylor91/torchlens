"""TensorFlow and Keras shape priors: built inputs, ``input_spec`` and kernels.

A Keras model built from ``keras.Input`` declares its input: ``model.inputs``
gives the shape with ``None`` for free axes, so the candidate is determined
up to those axes. Without declared inputs the prior reads the model's or the
first layer's ``input_spec`` and then the first built kernel in the layer's
own ``data_format`` (``Dense`` ``(in, out)``, ``Conv2D`` ``(kh, kw, in, out)``
channels last unless the layer says ``channels_first``). An unbuilt model has
no kernels; building it would create weights in the caller's object, so the
search refuses it instead. Plain TensorFlow functions get bounded hypotheses
probed eagerly on small inputs.
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

_PREFIXES = ("tensorflow", "keras", "tf_keras")
_UNIFORM: ValueRange = ("uniform", 0.0, 1.0)
_SPEC_MISMATCH = re.compile(
    r"expected axis (-?\d+) of input shape to have value (\d+), but received input with "
    r"shape \(([^)]*)\)"
)
_MATMUL_MISMATCH = re.compile(r"In\[0\]: \[([\d,\s]+)\], In\[1\]: \[([\d,\s]+)\]")
_DEPTH_MISMATCH = re.compile(
    r"Depth of input \((\d+)\) is not a multiple of input depth of filter \((\d+)\)"
)
_SHAPE_WORDS = ("shape", "dimension", "incompatible", "ndim", "rank", "reshape", "size")
_FLOAT_WORDS = ("not in list of allowed values", "expected to be a float", "float tensor")


def _keras() -> Any:
    """The imported ``keras`` module, or ``None``."""

    return sys.modules.get("keras")


def _is_layer(model: Any) -> bool:
    """Whether ``model`` is a Keras layer or model (only if Keras is imported)."""

    keras = _keras()
    layer_type = getattr(getattr(keras, "layers", None), "Layer", None)
    return layer_type is not None and isinstance(model, layer_type)


def _is_tf_function(model: Any) -> bool:
    """Whether ``model`` is a ``tf.function`` object."""

    return str(getattr(type(model), "__module__", "")).startswith("tensorflow.") and hasattr(
        model, "python_function"
    )


def handles(model: Any) -> bool:
    """Whether ``model`` is a Keras layer, a ``tf.function`` or a function using TF.

    Parameters
    ----------
    model:
        The caller's model object.

    Returns
    -------
    bool
        ``False`` when neither TensorFlow nor Keras was imported.
    """

    if "tensorflow" not in sys.modules and "keras" not in sys.modules:
        return False
    return _is_layer(model) or _is_tf_function(model) or names_framework(model, _PREFIXES)


def _declared_inputs(model: Any) -> list[Any]:
    """``model.inputs`` when the model was built from ``keras.Input``, else ``[]``."""

    try:
        inputs = getattr(model, "inputs", None)
    except (AttributeError, ValueError):
        return []
    return list(inputs) if isinstance(inputs, (list, tuple)) else []


def lazy_reason(model: Any) -> str | None:
    """Why an unbuilt Keras model cannot be probed without building it.

    Parameters
    ----------
    model:
        The caller's model object.

    Returns
    -------
    str | None
        The reason for an unbuilt layer without declared inputs, else ``None``.
    """

    if not _is_layer(model) or getattr(model, "built", True) or _declared_inputs(model):
        return None
    return (
        f"The Keras {type(model).__name__} is not built: its weights are created on the first "
        "call, so their shapes are unknown. Call it once on a real input (or build it from "
        "keras.Input), then retry; the model was left untouched."
    )


# ------------------------------------------------------------ module priors
def _dtype_name(dtype: Any) -> str:
    """Dtype name for a Keras string, ``tf.DType`` or NumPy dtype."""

    if dtype is None:
        return "float32"
    return str(getattr(dtype, "name", dtype))


def _from_inputs(model: Any, request: SearchRequest, dtype: Any) -> ShapeCandidate | None:
    """Candidate from a single declared ``keras.Input``."""

    inputs = _declared_inputs(model)
    if len(inputs) != 1:
        return None
    dims = list(getattr(inputs[0], "shape", ()) or ())
    if not dims:
        return None
    free = [axis for axis, dim in enumerate(dims) if dim is None and axis > 0]
    filler = request.seq_len or _first_side(1, request)
    shape = (request.batch_size, *((filler if dim is None else int(dim)) for dim in dims[1:]))
    native = _dtype_name(getattr(inputs[0], "dtype", None))
    integer = native.startswith(("int", "uint"))
    return ShapeCandidate(
        "sequence" if integer else ("conv" if len(shape) >= 3 else "linear"),
        shape,
        dtype if request.input_dtype is not None else native,
        ("randint", 0, 2) if integer else _UNIFORM,
        _default_layout(len(shape), integer),
        flexible_dims=tuple(free),
        determined=not free,
        strategy="keras_inputs",
    )


def _default_layout(rank: int, integer: bool) -> str:
    """Keras's default layout for a declared input of ``rank`` (channels last)."""

    if integer:
        return "NL"
    return {2: "NC", 3: "NLC", 4: "NHWC", 5: "NDHWC"}.get(rank, "N" + "?" * (rank - 1))


def _first_side(min_side: int, request: SearchRequest) -> int:
    """First preferred side at least ``min_side`` (else ``min_side``)."""

    fits = [s for s in request.preferred_sizes if min_side <= s <= request.max_size]
    return fits[0] if fits else max(min_side, request.min_size)


def _sublayers(model: Any, depth: int = 0) -> Iterator[Any]:
    """Tracked sublayers in definition order, depth first."""

    for layer in getattr(model, "layers", None) or ():
        yield layer
        if depth < 8:
            yield from _sublayers(layer, depth + 1)


def _kernel_shape(layer: Any, attr: str = "kernel") -> tuple[int, ...] | None:
    """Shape of a built layer weight, or ``None``."""

    weight = getattr(layer, attr, None) if getattr(layer, "built", False) else None
    shape = getattr(weight, "shape", None)
    try:
        return tuple(int(dim) for dim in shape) if shape is not None else None
    except TypeError:
        return None


def _conv_candidate(
    layer: Any, kernel: tuple[int, ...], request: SearchRequest, dtype: Any
) -> ShapeCandidate:
    """Candidate for a built ``ConvND`` (kernel ``(*k, in/groups, out)``)."""

    spatial = kernel[:-2]
    dilation = getattr(layer, "dilation_rate", (1,) * len(spatial))
    min_side = max(
        request.min_size, *((k - 1) * d + 1 for k, d in zip(spatial, dilation, strict=False))
    )
    side = _first_side(min_side, request)
    channels = request.channels or kernel[-2] * int(getattr(layer, "groups", 1) or 1)
    rank = len(spatial)
    if getattr(layer, "data_format", "channels_last") == "channels_first":
        shape, layout, axes = (
            (request.batch_size, channels, *[side] * rank),
            "NC" + "DHW"[3 - rank :],
            tuple(range(2, rank + 2)),
        )
    else:
        shape, layout, axes = (
            (request.batch_size, *[side] * rank, channels),
            "N" + "DHW"[3 - rank :] + "C",
            tuple(range(1, rank + 1)),
        )
    return ShapeCandidate(
        "conv",
        shape,
        dtype,
        _UNIFORM,
        layout,
        spatial_axes=axes,
        constraining_module=getattr(layer, "name", None),
        constraining_op=type(layer).__name__,
        strategy="keras_kernel",
        min_side=min_side,
    )


def _layer_candidate(layer: Any, request: SearchRequest, dtype: Any) -> ShapeCandidate | None:
    """Candidate from one built layer's weights, or ``None``."""

    name, batch = type(layer).__name__, request.batch_size
    if name == "Embedding" and getattr(layer, "input_dim", None):
        return ShapeCandidate(
            "embedding",
            (batch, request.seq_len or 8),
            "int32",
            ("randint", 0, int(layer.input_dim)),
            "NL",
            flexible_dims=(1,),
            constraining_module=layer.name,
            constraining_op=name,
            strategy="keras_kernel",
        )
    kernel = _kernel_shape(layer)
    if (
        kernel is not None
        and name.startswith("Conv")
        and "Transpose" not in name
        and len(kernel) >= 3
    ):
        return _conv_candidate(layer, kernel, request, dtype)
    width = kernel[0] if kernel is not None and name == "Dense" and len(kernel) == 2 else None
    gamma = _kernel_shape(layer, "gamma")
    if width is None and name in {"LayerNormalization", "BatchNormalization"} and gamma:
        width = gamma[0]
    if width is None:
        return None
    return ShapeCandidate(
        "linear",
        (batch, width),
        dtype,
        _UNIFORM,
        "NC",
        determined=True,
        constraining_module=layer.name,
        constraining_op=name,
        strategy="keras_kernel",
    )


def _from_input_spec(model: Any, request: SearchRequest, dtype: Any) -> ShapeCandidate | None:
    """Candidate from a fully specified ``input_spec`` (``shape`` with only a free batch)."""

    spec = getattr(model, "input_spec", None)
    spec = spec[0] if isinstance(spec, (list, tuple)) and len(spec) == 1 else spec
    dims = getattr(spec, "shape", None)
    if not dims or any(dim is None for dim in list(dims)[1:]):
        return None
    shape = (request.batch_size, *(int(dim) for dim in list(dims)[1:]))
    return ShapeCandidate(
        "linear" if len(shape) == 2 else "conv",
        shape,
        dtype,
        _UNIFORM,
        _default_layout(len(shape), False),
        determined=True,
        strategy="keras_input_spec",
    )


def _module_candidates(model: Any, request: SearchRequest, dtype: Any) -> list[ShapeCandidate]:
    """Declared inputs, then ``input_spec``, then the first built kernel."""

    for reader in (_from_inputs, _from_input_spec):
        candidate = reader(model, request, dtype)
        if candidate is not None:
            return [candidate]
    for layer in (model, *_sublayers(model)):
        candidate = _layer_candidate(layer, request, dtype)
        if candidate is not None:
            return [candidate]
    return []


# ----------------------------------------------------------- function roots
def _eager_works(fn: Any, dtype: Any) -> Callable[[tuple[int, ...]], bool]:
    """A hypothesis check that runs ``fn`` eagerly on zeros."""

    import tensorflow as tf

    def works(shape: tuple[int, ...]) -> bool:
        """Whether ``fn`` runs on a zero input of ``shape``."""

        try:
            fn(tf.zeros(shape, dtype=dtype))
        except Exception:  # noqa: BLE001 - any failure rejects the hypothesis
            return False
        return True

    return works


def candidates(model: Any, request: SearchRequest) -> Sequence[ShapeCandidate]:
    """Keras inputs, specs and kernels; probed hypotheses for TF functions.

    Parameters
    ----------
    model:
        A TensorFlow root this module handles.
    request:
        The validated search arguments.

    Returns
    -------
    Sequence[ShapeCandidate]
        Best first; empty when nothing readable pins the input.
    """

    dtype = "float32" if request.input_dtype is None else request.input_dtype
    if _is_layer(model):
        return _module_candidates(model, request, dtype)
    source = model.python_function if _is_tf_function(model) else model
    return function_candidates(array_dims(source), request, dtype, _eager_works(model, dtype))


def make_input(shape: tuple[int, ...], dtype: Any, value_range: ValueRange, seed: int) -> Any:
    """One deterministic TF tensor (NumPy's private generator; TF RNG untouched).

    Parameters
    ----------
    shape, dtype, value_range, seed:
        As in :class:`~torchlens.debug._shape_priors.ShapeCandidate`.

    Returns
    -------
    Any
        A ``tf.Tensor``.
    """

    import tensorflow as tf

    return tf.constant(fill_array(shape, value_range, seed, _dtype_name(dtype)))


def forward(model: Any, example: Any) -> Any:
    """One eager forward pass; Keras layers run with ``training=False``.

    Parameters
    ----------
    model:
        A TensorFlow root this module handles.
    example:
        One input tensor.

    Returns
    -------
    Any
        The outputs.
    """

    if _is_layer(model):
        return model(example, training=False)
    return model(example)


def _measured(message: str) -> tuple[int | None, int | None]:
    """``(got, target)`` feature counts a TF or Keras message names."""

    spec = _SPEC_MISMATCH.search(message)
    if spec:
        axis, shape = int(spec.group(1)), parse_ints(spec.group(3))
        got = shape[axis] if shape and -len(shape) <= axis < len(shape) else None
        return got, int(spec.group(2))
    matmul = _MATMUL_MISMATCH.search(message)
    if matmul:
        return parse_ints(matmul.group(1))[-1], parse_ints(matmul.group(2))[0]
    depth = _DEPTH_MISMATCH.search(message)
    if depth:
        return int(depth.group(1)), int(depth.group(2))
    return None, None


def classify_error(exc: Exception) -> ProbeError:
    """Sort a TF or Keras probe exception into shape, dtype, device or blocker.

    Parameters
    ----------
    exc:
        The exception a probe raised.

    Returns
    -------
    ProbeError
        The classification, with measured feature counts when named.
    """

    message = str(exc)
    lowered = message.lower()
    got, target = _measured(message)
    if got is not None or target is not None:
        return ProbeError("shape", message, got_features=got, target_features=target)
    if any(word in lowered for word in _FLOAT_WORDS):
        return ProbeError("dtype", message, suggested_dtype="float32")
    if any(word in lowered for word in _SHAPE_WORDS):
        return ProbeError("shape", message)
    if "device" in lowered:
        return ProbeError("device", message)
    return ProbeError("blocker", message)


def executed_constraint(trace: Any) -> ExecutedConstraint | None:
    """The first MatMul, convolution or gather the trace executed.

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


@contextlib.contextmanager
def state_scope(model: Any) -> Iterator[None]:
    """Restore a Keras model's non-trainable state (statistics, seed states) on exit.

    Keras takes the training flag per call, so there is no mode to switch.
    Eager TF random ops advance the op-level seed counter, which TensorFlow
    gives no way to restore; the probes draw their inputs from NumPy instead.

    Parameters
    ----------
    model:
        A TensorFlow root this module handles.

    Yields
    ------
    None
        Nothing; the scope only restores state.
    """

    variables = (
        list(getattr(model, "non_trainable_variables", None) or ()) if _is_layer(model) else []
    )
    saved = [variable.numpy() for variable in variables]
    try:
        yield
    finally:
        for variable, value in zip(variables, saved, strict=True):
            variable.assign(value)


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
