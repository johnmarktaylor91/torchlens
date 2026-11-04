"""Shared helpers for TorchLens debug utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .._data_classes._lookup_kinds import op_at
from .._data_classes.func_call_location import display_frame
from .._utils._multipass_access import is_multipass_layer

if TYPE_CHECKING:
    from torchlens._data_classes.op import Op
    from torchlens._data_classes.trace import Trace


def _ordered_ops(trace: Trace) -> list[Op]:
    """Return trace ops in forward execution order.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.

    Returns
    -------
    list[Op]
        Ops sorted by ``step_index`` with boundary ops kept after compute ops
        at the same step.
    """

    return sorted(
        trace._layer_list,
        key=lambda op: (
            int(getattr(op, "step_index", 0) or 0),
            bool(getattr(op, "is_input", False)),
            bool(getattr(op, "is_output", False)),
            str(getattr(op, "layer_label", "")),
        ),
    )


def _source_line(op: Op) -> str | None:
    """Return the source location of the line that ran an op.

    The location is the innermost recorded frame outside torch's own code
    (:func:`torchlens._data_classes.func_call_location.display_frame`), the
    same line an op's card shows.

    Parameters
    ----------
    op:
        TorchLens op.

    Returns
    -------
    str | None
        ``"file:line"`` or ``None`` when no source context is present.
    """

    location = display_frame(getattr(op, "code_context", None))
    if location is None:
        return None
    file_name = getattr(location, "file", None)
    line_number = getattr(location, "line_number", None)
    if file_name is None or line_number is None:
        return None
    return f"{file_name}:{line_number}"


def _require_pandas() -> Any:
    """Import pandas with the TorchLens tabular-extra error message.

    Returns
    -------
    Any
        Imported pandas module.
    """

    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
        ) from e
    return pd


def _op_label(op: Op) -> str:
    """Return the stable pass-qualified label for an op or aggregate Layer.

    Reading the per-pass ``label`` on an aggregate recurrent ``Layer`` trips the
    deliberate ``ValueError`` multi-pass tripwire, so route through the shared
    :func:`~torchlens._utils._multipass_access.get_multipass_attr` helper (the one
    documented pattern for the whole sprint) and fall back to the aggregate
    ``layer_label``, which is the honest identifier for a multi-pass Layer.

    Parameters
    ----------
    op:
        TorchLens op or aggregate Layer.

    Returns
    -------
    str
        Op label, or the aggregate ``layer_label`` for a multi-pass Layer.
    """

    from .._utils._multipass_access import get_multipass_attr

    label = get_multipass_attr(op, "label", None, multipass=None)
    if isinstance(label, str) and label:
        return label
    layer_label = get_multipass_attr(op, "layer_label", "", multipass=None)
    return str(layer_label or "")


def _compute_ops(trace: Trace) -> list[Op]:
    """Return pass-qualified compute ops by the debug module's local convention.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.

    Returns
    -------
    list[Op]
        Ops whose ``step_index`` is positive.
    """

    return [op for op in _ordered_ops(trace) if int(getattr(op, "step_index", 0) or 0) > 0]


def _multipass_layer_or_none(trace: Trace, label: Any) -> Any | None:
    """Return the aggregate Layer a bare multi-pass label names, else ``None``.

    A bare multi-pass label names a Layer, not one op (every pass is a distinct
    op); the lineage contract still owes the honest "select a pass" message,
    which needs the aggregate record's pass count.
    """

    if not isinstance(label, str):
        return None
    try:
        layer = trace.layers[label]
    except Exception:  # noqa: BLE001 - an unresolvable label keeps the door's own message.
        return None
    return layer if is_multipass_layer(layer) else None


def _resolve_op(trace: Trace, op_or_label: Any) -> tuple[Op | None, str | None]:
    """Resolve an op-like object or lookup key without raising.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    op_or_label:
        Op object or key accepted by ``trace.__getitem__``.

    Returns
    -------
    tuple[Op | None, str | None]
        Resolved op and error message.
    """

    candidate = op_or_label
    if not (hasattr(candidate, "parents") and hasattr(candidate, "children")):
        try:
            candidate = op_at(trace, op_or_label)  # a module name names its output op
        except Exception as exc:  # noqa: BLE001 - debug helpers report odd inputs instead of raising.
            candidate = _multipass_layer_or_none(trace, op_or_label)
            if candidate is None:
                return None, f"unavailable: {exc}"
    # An aggregate recurrent Layer is per-pass ambiguous: its graph edges are
    # bare labels that re-resolve to aggregates and its per-pass fields trip the
    # multi-pass ValueError tripwire. Refuse it honestly (non-raising, per the
    # lineage contract) and direct the caller to a specific pass instead of
    # leaking a bare ValueError or walking a bare-label graph.
    if is_multipass_layer(candidate):
        base = str(getattr(candidate, "layer_label", op_or_label))
        num_passes = int(getattr(candidate, "num_passes", 0) or 0)
        return None, (
            f"unavailable: {base!r} is recurrent ({num_passes} passes); "
            f"select a pass, e.g. {base}:1"
        )
    # A single-pass Layer -> drill to its one Op so traversal follows the
    # pass-qualified graph edges (a Layer exposes bare-label edges instead).
    if type(candidate).__name__ == "Layer":
        ops = getattr(candidate, "ops", None)
        if ops is None:
            return None, f"unavailable: {op_or_label!r} did not resolve to an op"
        try:
            first_op = next(iter(ops.values())) if hasattr(ops, "values") else next(iter(ops))
        except (AttributeError, TypeError, StopIteration):
            return None, f"unavailable: {op_or_label!r} did not resolve to an op"
        return first_op, None
    if hasattr(candidate, "parents") and hasattr(candidate, "children"):
        return candidate, None
    return None, f"unavailable: {op_or_label!r} did not resolve to an op"


def _safe_out(op: Op) -> tuple[Any | None, str | None]:
    """Read ``op.out`` and convert known unavailable cases into a reason.

    Parameters
    ----------
    op:
        TorchLens op.

    Returns
    -------
    tuple[Any | None, str | None]
        Payload and unavailable reason.
    """

    if not bool(getattr(op, "has_saved_activation", False)):
        return None, "unsaved"
    try:
        return op.out, None
    except ValueError as exc:
        return None, str(exc)


def _tensor_unavailable_reason(value: Any) -> str | None:
    """Return why a value is not a usable dense floating tensor.

    Parameters
    ----------
    value:
        Candidate activation or gradient payload.

    Returns
    -------
    str | None
        Reason string, or ``None`` when the tensor is usable.
    """

    if not isinstance(value, torch.Tensor):
        return "non-tensor/container"
    if bool(getattr(value, "is_meta", False)):
        return "meta"
    if bool(getattr(value, "is_sparse", False)):
        return "sparse"
    if bool(getattr(value, "is_quantized", False)):
        return "quantized"
    if torch.is_complex(value):
        return "complex"
    if not torch.is_floating_point(value):
        return "non-floating"
    return None


def _shape_dtype(op: Op) -> tuple[tuple[int, ...] | None, str | None]:
    """Return shape and dtype metadata without materializing payloads.

    Parameters
    ----------
    op:
        TorchLens op.

    Returns
    -------
    tuple[tuple[int, ...] | None, str | None]
        Shape tuple and dtype string when available.
    """

    shape = getattr(op, "shape", None)
    dtype = getattr(op, "dtype", None)
    return shape, None if dtype is None else str(dtype)


def _op_from_label(trace: Trace, label: str) -> Op | None:
    """Resolve an edge label to an op, returning ``None`` when unavailable.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.
    label:
        Edge label from ``Op.parents`` or ``Op.children``.

    Returns
    -------
    Op | None
        Resolved op.
    """

    op, _ = _resolve_op(trace, label)
    return op


def _nonfinite_kind(tensor: torch.Tensor) -> str:
    """Classify non-finite values in a tensor.

    Parameters
    ----------
    tensor:
        Tensor to inspect.

    Returns
    -------
    str
        ``"nan"``, ``"inf"``, ``"nan+inf"``, or ``"none"``.
    """

    if not torch.is_floating_point(tensor) and not torch.is_complex(tensor):
        return "none"
    has_nan = bool(torch.isnan(tensor).any().item())
    has_inf = bool(torch.isinf(tensor).any().item())
    if has_nan and has_inf:
        return "nan+inf"
    if has_nan:
        return "nan"
    if has_inf:
        return "inf"
    return "none"


def _module_address(op: Op) -> str | None:
    """Return the containing module address recorded for an op.

    Parameters
    ----------
    op:
        Operation whose module metadata should be read.

    Returns
    -------
    str | None
        Atomic module address or the innermost containing module address.
    """

    atomic_address = getattr(op, "atomic_module_address", None)
    if isinstance(atomic_address, str):
        return atomic_address
    # Live (mid-capture) ops have no atomic_module_address yet; the fallback
    # reads the innermost module CALL label ("0:1"), which carries the pass
    # qualifier -- strip it so the field keeps its ADDRESS contract.
    module = getattr(op, "module", None)
    if isinstance(module, tuple) and module and isinstance(module[0], str):
        return module[0].rsplit(":", 1)[0]
    if isinstance(module, str):
        return module.rsplit(":", 1)[0]
    return None
