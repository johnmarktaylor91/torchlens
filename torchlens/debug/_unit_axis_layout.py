"""Resolve preview-backend unit axes from recorded layouts and parent edges."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from .._utils._record_group_table import record_member
from ._common import _op_from_label, _op_label
from ._unit_axis import _MAX_INHERIT_HOPS

if TYPE_CHECKING:
    from torchlens._data_classes.op import Op
    from torchlens._data_classes.trace import Trace

_SPATIAL_NAME = re.compile(
    r"(?:conv(?:transpose)?[123]d|convgeneral|convgeneraldilated|"
    r"depthwiseconv[123]d(?:native)?|(?:adaptive)?(?:max|avg)?pool[123]?d?|"
    r"maxpoolv2|avgpoolv2)\Z"
)
_JAX_DIMENSIONS = re.compile(
    r"['\"]dimension_numbers['\"]\s*:\s*"
    r"(ConvDimensionNumbers\((?:[^()]|\([^()]*\))*\)|\((?:[^()]|\([^()]*\))*\))"
)
_JAX_OUT_SPEC = re.compile(r"\bout_spec\s*=\s*(\([^)]*\))")
#: Frameworks whose spatial ops have one fixed channel layout and record none per op
#: (MLX convolutions and pools are channels-last; tinygrad's are NCHW).
_FIXED_CHANNEL_AXIS: Final[dict[str, str]] = {"mlx": "last", "tinygrad": "first"}


def _literal(value: Any) -> Any:
    """Decode a recorded literal repr without executing expressions."""

    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _call_layout(op: Op, key: str) -> Any:
    """Read a layout argument retained in the op's call metadata."""

    for field in ("saved_kwargs", "non_tensor_kwargs"):
        kwargs = record_member(op, field, None)
        if isinstance(kwargs, Mapping) and key in kwargs:
            return kwargs[key]
    return None


def _format_axis(value: Any, ndim: int) -> int | None:
    """Resolve an explicit channel-format token for the activation rank."""

    value = _literal(value)
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="replace")
    if not isinstance(value, str):
        return None
    value = value.upper()
    if value in {"CHANNELS_FIRST", "CHANNELS_LAST"}:
        return 1 if value == "CHANNELS_FIRST" else ndim - 1
    if len(value) != ndim or value not in {"NCW", "NWC", "NCHW", "NHWC", "NCDHW", "NDHWC"}:
        return None
    return value.index("C")


def _tf_format(op: Op) -> Any:
    """Read TF eager/static attrs, including numbered callback key/value pairs."""

    annotations = getattr(op, "annotations", {}) or {}
    for field in ("tf_attrs", "tf_static_attrs"):
        attrs = annotations.get(field, {})
        if not isinstance(attrs, Mapping):
            continue
        if "data_format" in attrs:
            return attrs["data_format"]
        for key, value in attrs.items():
            if str(key).isdigit() and _literal(value) in ("data_format", b"data_format"):
                return attrs.get(str(int(key) + 1))
    return _call_layout(op, "data_format")


def _jax_dimensions(op: Op) -> Any:
    """Read JAX dimension numbers from recorded kwargs or primitive params."""

    value = _call_layout(op, "dimension_numbers")
    if value is not None:
        return value
    params = (getattr(op, "annotations", {}) or {}).get("jax_params")
    if isinstance(params, Mapping):
        return params.get("dimension_numbers")
    if isinstance(params, str):
        match = _JAX_DIMENSIONS.search(params)
        if match is not None:
            return match.group(1)
    return None


def _jax_spec_axis(spec: Any, ndim: int) -> int | None:
    """Resolve the channel in a JAX output permutation or dimension string."""

    if isinstance(spec, str):
        if len(spec) == ndim and spec.count("C") == 1 and spec.count("N") == 1:
            return spec.index("C")
        return None
    if not isinstance(spec, (tuple, list)) or len(spec) != ndim:
        return None
    if not all(type(axis) is int for axis in spec) or sorted(spec) != list(range(ndim)):
        return None
    return int(spec[1])


def _jax_axis(value: Any, ndim: int) -> int | None:
    """Resolve duck-typed, string-triple, or repr JAX output dimension numbers."""

    value = _literal(value)
    spec = getattr(value, "out_spec", None)
    if isinstance(value, str):
        match = _JAX_OUT_SPEC.search(value)
        spec = None if match is None else _literal(match.group(1))
    if isinstance(value, (tuple, list)) and len(value) == 3:
        spec = value[2]
    return _jax_spec_axis(spec, ndim)


def _spatial_name(name: str) -> bool:
    """Recognize recorded convolution/pooling names without substring guessing."""

    return _SPATIAL_NAME.fullmatch(name.rsplit(".", 1)[-1].lower().replace("_", "")) is not None


def _tinygrad_conv_return(trace: Trace, op: Op) -> bool:
    """Recognize a lowered tinygrad convolution by its recorded module return."""

    modules = getattr(trace, "modules", {})
    for address in record_member(op, "output_of_modules", None) or ():
        try:
            module = modules[address]
        except (KeyError, IndexError):
            continue
        if getattr(module, "class_qualname", "") == "tinygrad.nn.Conv2d":
            return True
    return False


def _recorded_axis(op: Op, ndim: int) -> int | None:
    """Return the channel axis named by the op's own recorded layout metadata.

    TF records ``data_format`` in its eager/static attrs, Paddle in its call
    kwargs and JAX as ``dimension_numbers`` in its primitive params; each key
    is read only where its framework records it.
    """

    axis = _format_axis(_tf_format(op), ndim)
    return axis if axis is not None else _jax_axis(_jax_dimensions(op), ndim)


def _own_layout_axis(trace: Trace, op: Op, ndim: int) -> tuple[bool, int | None]:
    """Return whether this op fixes a spatial layout, and its evidenced axis."""

    if _tinygrad_conv_return(trace, op):
        return True, 1
    if not _spatial_name(str(getattr(op, "func_name", "") or "")):
        return False, None
    fixed = _FIXED_CHANNEL_AXIS.get(str(trace._provenance_backend))
    if fixed is not None:
        return True, 1 if fixed == "first" else ndim - 1
    return True, _recorded_axis(op, ndim)


def backend_unit_axis(trace: Trace, op: Op, out: Any, memo: dict[str, int]) -> int | None:
    """Return a preview activation's evidenced unit axis, or ``None``.

    Parameters
    ----------
    trace, op, out:
        Completed trace, producer op, and activation exposing ``ndim``/``shape``.
    memo:
        Per-call cache of decided axes keyed by op label, as in ``unit_axis``.

    Returns
    -------
    int | None
        Recorded channel axis, inherited through same-shaped parents. Matrices
        retain the last-axis convention; spatial values without layout evidence
        and values below two dimensions have no inferred axis. Unknown layouts
        are never cached or replaced with backend defaults.
    """

    ndim = int(out.ndim)
    if ndim < 2:
        return None
    if ndim == 2:
        return 1
    return _inherited_axis(trace, op, (tuple(out.shape), ndim), memo, set(), _MAX_INHERIT_HOPS)


def _same_shape_parents(trace: Trace, op: Op, shape: tuple[int, ...]) -> list[Op]:
    """Return every parent op whose output has exactly ``shape``, in parent order."""

    parents = (_op_from_label(trace, str(label)) for label in op.parents or ())
    return [
        parent
        for parent in parents
        if parent is not None and tuple(getattr(parent, "shape", None) or ()) == shape
    ]


def _inherited_axis(
    trace: Trace,
    op: Op,
    geometry: tuple[tuple[int, ...], int],
    memo: dict[str, int],
    unknown: set[str],
    hops: int,
) -> int | None:
    """Return the axis evidenced by ``op`` or by any same-shaped ancestor within ``hops``.

    Every same-shaped parent is searched, not only the first: a framework that
    lowers an elementwise op (tinygrad's relu is ``cmplt`` + ``where`` over an
    expanded constant) gives the activation several same-shaped parents, and
    only one of them may lead back to the layout-fixing op. Ancestors that
    evidence different axes leave the axis unknown rather than picking one.
    ``unknown`` caches labels already found without evidence in this call.
    """

    label = _op_label(op)
    if label in memo:
        return memo[label]
    if label in unknown or hops <= 0:
        return None
    shape, ndim = geometry
    spatial, axis = _own_layout_axis(trace, op, ndim)
    unknown.add(label)  # in progress: a parent cycle back to this op proves nothing
    if not spatial:
        found = {
            parent_axis
            for parent in _same_shape_parents(trace, op, shape)
            if (parent_axis := _inherited_axis(trace, parent, geometry, memo, unknown, hops - 1))
            is not None
        }
        axis = next(iter(found)) if len(found) == 1 else None
    if axis is not None:
        unknown.discard(label)
        memo[label] = axis
    return axis
