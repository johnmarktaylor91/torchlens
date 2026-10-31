"""Checked suppression of redundant constructor-arg label rows (L5 M4).

DEFAULT-ON: node labels omit a constructor arg (``in_features=4``) exactly
when the CHECK licenses it — the arg value provably equals the captured
shape dimension it claims to duplicate ON THIS TRACE. A mismatch or an
unavailable shape keeps the arg VISIBLE (the mismatch case is exactly the
interesting one — the rule is self-honest, it can only REVEAL more, never
hide a discrepancy). ``draw(show_redundant_args=True)`` shows everything.

ELIGIBILITY IS A CLOSED TABLE keyed on the torch nn module family, whose
axis semantics are the PyTorch module contract (conv/norm channel = logical
axis 1 regardless of memory format; linear in_features = last dim;
layernorm = trailing dims). Non-torch capture backends, custom or
unrecognized modules, and any func_config lacking the exact arg are NOT
candidates — ambiguity never suppresses. NEVER candidates (not recoverable
from I/O shapes): kernel_size, stride, padding, dilation, groups,
num_embeddings, num_heads, dropout p.

The check runs at the TRACE-BEARING PREPASS (the render pipeline owns the
trace there) and produces a per-record suppressed-key set that travels into
the row build as plain data; render paths that reach the row builder
without the prepass (detached/standalone records) get an empty set — all
args visible, and a detached record can never crash a default-ON label
build (no ``source_trace`` weakref read exists anywhere in this module).

The check is data equality on records, never a render-back loop.

NAMING: ``show_redundant_args`` is DOCUMENTED-UNSTABLE (slate 8.7) pending
naming-session ratification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..backends.registry import TORCH_BACKEND_NAME

if TYPE_CHECKING:  # typing only
    from .._data_classes.trace import Trace

#: side "output": check against the record's own captured output shape.
#: side "input": check against the SINGLE parent's captured output shape.
#: axis "last": shape[-1]; axis "channel": shape[1] (rank >= 2 required);
#: axis "trailing": tuple equality against the trailing dims.
_LINEAR_ROWS = {"in_features": ("input", "last"), "out_features": ("output", "last")}
_CONV_ROWS = {"in_channels": ("input", "channel"), "out_channels": ("output", "channel")}
_NORM_ROWS = {"num_features": ("output", "channel")}

#: The closed candidate table: normalized layer_type -> {arg: (side, axis)}.
SUPPRESSION_CANDIDATE_TABLE: dict[str, dict[str, tuple[str, str]]] = {
    "linear": _LINEAR_ROWS,
    "conv1d": _CONV_ROWS,
    "conv2d": _CONV_ROWS,
    "conv3d": _CONV_ROWS,
    "convolution": _CONV_ROWS,
    "convtranspose1d": _CONV_ROWS,
    "convtranspose2d": _CONV_ROWS,
    "convtranspose3d": _CONV_ROWS,
    "batchnorm": _NORM_ROWS,
    "batchnorm1d": _NORM_ROWS,
    "batchnorm2d": _NORM_ROWS,
    "batchnorm3d": _NORM_ROWS,
    "instancenorm": _NORM_ROWS,
    "instancenorm1d": _NORM_ROWS,
    "instancenorm2d": _NORM_ROWS,
    "instancenorm3d": _NORM_ROWS,
    "layernorm": {"normalized_shape": ("output", "trailing")},
    "embedding": {"embedding_dim": ("output", "last")},
    "multiheadattention": {"embed_dim": ("output", "last")},
}


def _usable_shape(value: Any) -> tuple[int, ...] | None:
    """Return ``value`` as an all-int shape tuple, or ``None``."""

    if value is None:
        return None
    try:
        entries = tuple(value)
    except TypeError:
        return None
    for entry in entries:
        if isinstance(entry, bool) or not isinstance(entry, int):
            return None
    return entries


def _record_output_shape(node: Any) -> tuple[int, ...] | None:
    """Return a record's captured output shape when unambiguous.

    A rolled multi-pass aggregate whose passes disagree on shape has NO
    single output shape: the reconciler's variation marker gates the read
    (shape unavailable -> arg stays visible; the deliberate
    rolled-vs-unrolled divergence).
    """

    ops = getattr(node, "ops", None)
    if isinstance(ops, dict) and len(ops) > 1:
        annotations = getattr(node, "annotations", None)
        if isinstance(annotations, dict):
            marker = annotations.get("varying_across_passes")
            if isinstance(marker, dict) and "shape" in marker:
                return None
    from .._utils._multipass_access import get_multipass_attr

    return _usable_shape(get_multipass_attr(node, "shape", None, multipass=None))


def _parent_shape(trace: Trace, node: Any, op_by_label: dict[str, Any]) -> tuple[int, ...] | None:
    """Return the single data parent's output shape, or ``None``.

    Only a node with EXACTLY ONE incoming data edge whose source shape is
    available is input-side checkable; multi-input or shape-less parents
    keep the arg visible.
    """

    parents = tuple(getattr(node, "parents", ()) or ())
    if len(parents) != 1:
        return None
    parent_label = str(parents[0])
    parent = op_by_label.get(parent_label)
    if parent is not None:
        return _record_output_shape(parent)
    layer_logs = getattr(trace, "layer_logs", None)
    if layer_logs is not None and parent_label in layer_logs:
        return _record_output_shape(layer_logs[parent_label])
    return None


def _axis_matches(arg_value: Any, shape: tuple[int, ...], axis: str) -> bool:
    """Return whether ``arg_value`` equals the claimed dimension of ``shape``."""

    if axis == "last":
        return _scalar_axis_matches(arg_value, shape, -1)
    if axis == "channel":
        # PyTorch module contract: channel = LOGICAL axis 1 (channels_last is
        # a memory format that permutes strides, not the logical shape).
        return _scalar_axis_matches(arg_value, shape, 1)
    if axis == "trailing":
        dims = _usable_shape(arg_value if not isinstance(arg_value, int) else (arg_value,))
        if dims is None or len(dims) == 0 or len(dims) > len(shape):
            return False
        return tuple(shape[-len(dims) :]) == dims
    return False


def _scalar_axis_matches(arg_value: Any, shape: tuple[int, ...], axis: int) -> bool:
    """Return whether an integer argument equals one available shape axis."""

    if not (-len(shape) <= axis < len(shape)):
        return False
    return (
        isinstance(arg_value, int) and not isinstance(arg_value, bool) and arg_value == shape[axis]
    )


def suppressed_arg_keys_for_record(
    trace: Trace, node: Any, op_by_label: dict[str, Any]
) -> frozenset[str]:
    """Return the constructor-arg keys provably redundant for one record."""

    config = getattr(node, "func_config", None)
    if not config:
        return frozenset()
    normalized_type = str(getattr(node, "layer_type", "")).lower().replace("_", "")
    rows = SUPPRESSION_CANDIDATE_TABLE.get(normalized_type)
    if not rows:
        return frozenset()
    suppressed: set[str] = set()
    output_shape = _record_output_shape(node)
    input_shape: tuple[int, ...] | None = None
    input_shape_resolved = False
    for arg_name, (side, axis) in rows.items():
        if arg_name not in config:
            continue
        if side == "output":
            shape = output_shape
        else:
            if not input_shape_resolved:
                input_shape = _parent_shape(trace, node, op_by_label)
                input_shape_resolved = True
            shape = input_shape
        if shape is None:
            continue
        if _axis_matches(config[arg_name], shape, axis):
            suppressed.add(arg_name)
    return frozenset(suppressed)


def compute_suppressed_arg_keys(trace: Trace, universe: Any) -> dict[int, frozenset[str]]:
    """TRACE-BEARING PREPASS: per-record suppressed-key sets for one draw.

    Keys are ``id(record)`` — the prepass and the row build see the SAME
    record objects within one draw, and records outlive the render (they
    live on the trace).
    """

    if str(getattr(trace, "_provenance_backend", "")) != TORCH_BACKEND_NAME:
        # Non-torch capture backends are never candidates (the axis
        # semantics in the table are the PyTorch module contract).
        return {}
    op_by_label: dict[str, Any] = {}
    for op in getattr(trace, "layer_list", ()) or ():
        label = getattr(op, "label", None)
        if isinstance(label, str):
            op_by_label.setdefault(label, op)
    result: dict[int, frozenset[str]] = {}
    for unit in universe.units:
        emission = unit.emission
        if emission.kind != "raw_op" or emission.node is None:
            continue
        node = emission.node
        keys = suppressed_arg_keys_for_record(trace, node, op_by_label)
        if keys:
            result[id(node)] = keys
    return result
