"""NaN and Inf bisection helpers for TorchLens traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..errors import CaptureError

if TYPE_CHECKING:
    from torchlens.data_classes.op import Op
    from torchlens.data_classes.trace import Trace

from ._common import _ordered_ops, _source_line


@dataclass(frozen=True)
class BisectNanResult:
    """Result returned by :func:`bisect_nan`.

    Parameters
    ----------
    found:
        Whether a saved activation containing NaN or Inf was found.
    op:
        First offending op, or ``None`` when no saved offender was found.
    label:
        Offending op label, when available.
    source_line:
        Source location formatted as ``"file:line"``, when available.
    kind:
        ``"nan"``, ``"inf"``, ``"nan+inf"``, or ``"none"``.
    message:
        Actionable human-readable result summary.
    """

    found: bool
    op: Op | None
    label: str | None
    source_line: str | None
    kind: str
    message: str


@dataclass(frozen=True)
class FindNanResult:
    """Structured first-non-finite diagnostic returned by ``find_nan``.

    Parameters
    ----------
    found:
        Whether a NaN or Inf was found.
    op:
        First offending op when available.
    label:
        Offending operation label.
    module_address:
        Atomic module address containing the op, when available.
    func_name:
        Torch function name for the op.
    dtype:
        Offending output dtype.
    shape:
        Offending output shape.
    bad_tensors:
        Tensor positions found non-finite. Current eager capture detects op
        outputs, so this is ``("output",)`` for a finding.
    source_line:
        ``"file:line"`` source location when available.
    kind:
        ``"nan"``, ``"inf"``, ``"nan+inf"``, or ``"none"``.
    scope:
        Scope of the conclusion. Sparse traces say ``"first among saved tensors"``.
    uncertainty_zone:
        Unsaved ancestor labels of a sparse finding. They may contain the
        actual birth operation and therefore prevent a stronger claim.
    message:
        Human-readable diagnostic summary.
    """

    found: bool
    op: Op | None
    label: str | None
    module_address: str | None
    func_name: str | None
    dtype: str | None
    shape: tuple[int, ...] | None
    bad_tensors: tuple[str, ...]
    source_line: str | None
    kind: str
    scope: str
    uncertainty_zone: tuple[str, ...]
    message: str

    def __repr__(self) -> str:
        """Return a compact notebook-oriented diagnostic representation.

        Returns
        -------
        str
            One-line diagnostic summary.
        """

        if not self.found:
            return f"FindNanResult(found=False, scope={self.scope!r})"
        return (
            "FindNanResult("
            f"found=True, kind={self.kind!r}, label={self.label!r}, "
            f"source_line={self.source_line!r}, scope={self.scope!r})"
        )


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


def _op_by_any_label(trace: Trace) -> dict[str, Op]:
    """Build a lookup of final and raw labels to trace ops.

    Parameters
    ----------
    trace:
        Trace whose graph labels should be indexed.

    Returns
    -------
    dict[str, Op]
        Label lookup for parent traversal.
    """

    lookup: dict[str, Op] = {}
    for op in _ordered_ops(trace):
        for label in (
            getattr(op, "label", None),
            getattr(op, "layer_label", None),
            getattr(op, "_label_raw", None),
        ):
            if isinstance(label, str):
                lookup[label] = op
    return lookup


def _op_label(op: Op) -> str:
    """Return the best available final or raw label for an op.

    Parameters
    ----------
    op:
        Operation whose label should be rendered.

    Returns
    -------
    str
        Final label for completed traces, otherwise the live raw label.
    """

    for attribute in ("layer_label", "label", "_label_raw"):
        value = getattr(op, attribute, None)
        if isinstance(value, str) and value:
            return value
    return "unknown"


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
    module = getattr(op, "module", None)
    if isinstance(module, tuple) and module and isinstance(module[0], str):
        return module[0]
    if isinstance(module, str):
        return module
    return None


def _unsaved_ancestors(trace: Trace, op: Op) -> tuple[str, ...]:
    """Return unsaved computational ancestors of an op in execution order.

    Parameters
    ----------
    trace:
        Trace containing ``op``.
    op:
        First saved non-finite op.

    Returns
    -------
    tuple[str, ...]
        Labels of ancestor ops that do not retain output payloads.
    """

    by_label = _op_by_any_label(trace)
    pending = list(getattr(op, "parents", ()) or ())
    visited: set[str] = set()
    unsaved: set[str] = set()
    while pending:
        parent_label = pending.pop()
        if parent_label in visited:
            continue
        visited.add(parent_label)
        parent = by_label.get(parent_label)
        if parent is None:
            continue
        pending.extend(getattr(parent, "parents", ()) or ())
        if not bool(getattr(parent, "is_input", False)) and not bool(
            getattr(parent, "has_saved_activation", False)
        ):
            unsaved.add(_op_label(parent))
    return tuple(
        _op_label(candidate) for candidate in _ordered_ops(trace) if _op_label(candidate) in unsaved
    )


def _result_from_op(
    op: Op,
    *,
    kind: str,
    source_line: str | None = None,
    scope: str = "first non-finite tensor",
    uncertainty_zone: tuple[str, ...] = (),
) -> FindNanResult:
    """Build a public diagnostic result from an offending op.

    Parameters
    ----------
    op:
        Offending operation.
    kind:
        Non-finite value kind.
    source_line:
        Optional source location override.
    scope:
        Claim scope for the result.
    uncertainty_zone:
        Unsaved ancestor labels that bound the conclusion.

    Returns
    -------
    FindNanResult
        Structured public diagnostic.
    """

    label = _op_label(op)
    location = source_line or _source_line(op)
    prefix = scope[:1].upper() + scope[1:] if scope.startswith("first ") else f"First {scope}"
    message = f"{prefix} is {kind} at {label}."
    if uncertainty_zone:
        message += " Unsaved upstream uncertainty zone: " + ", ".join(uncertainty_zone) + "."
    return FindNanResult(
        found=True,
        op=op,
        label=label,
        module_address=_module_address(op),
        func_name=str(getattr(op, "func_name", "unknown")),
        dtype=str(getattr(op, "dtype", "unknown")),
        shape=tuple(getattr(op, "shape", ()) or ()),
        bad_tensors=("output",),
        source_line=location,
        kind=kind,
        scope=scope,
        uncertainty_zone=uncertainty_zone,
        message=message,
    )


def _no_finding_result(*, scope: str, uncertainty_zone: tuple[str, ...] = ()) -> FindNanResult:
    """Build a no-finding diagnostic result.

    Parameters
    ----------
    scope:
        Scope searched for non-finite values.
    uncertainty_zone:
        Unsaved ops that make a clean result incomplete.

    Returns
    -------
    FindNanResult
        No-finding diagnostic.
    """

    message = f"No NaN/Inf found in {scope}."
    if uncertainty_zone:
        message += " Unsaved upstream uncertainty zone: " + ", ".join(uncertainty_zone) + "."
    return FindNanResult(
        False,
        None,
        None,
        None,
        None,
        None,
        None,
        (),
        None,
        "none",
        scope,
        uncertainty_zone,
        message,
    )


def find_nan(model: Any, x: Any, **trace_kwargs: Any) -> FindNanResult:
    """Run a memory-light live capture and return its first NaN or Inf finding.

    The capture stops at the first non-finite op output, rather than retaining
    all activations. ``trace_kwargs`` accepts the same options as :func:`tl.trace`.

    Parameters
    ----------
    model:
        PyTorch model to execute.
    x:
        Positional model input accepted by :func:`torchlens.trace`.
    **trace_kwargs:
        Additional :func:`torchlens.trace` keyword arguments. ``raise_on_nan``
        is always enabled by this diagnostic.

    Returns
    -------
    FindNanResult
        First live non-finite output, or a clean result.
    """

    from ..options import CaptureOptions
    from ..user_funcs import trace

    capture = trace_kwargs.pop("capture", None)
    trace_kwargs.pop("raise_on_nan", None)
    if capture is not None:
        explicit_values = {
            field_name: value
            for field_name, value in capture.as_dict().items()
            if capture.is_field_explicit(field_name)
        }
        trace_kwargs["capture"] = CaptureOptions(**explicit_values, raise_on_nan=True)
    else:
        trace_kwargs["capture"] = CaptureOptions(raise_on_nan=True)
    try:
        trace(model, x, **trace_kwargs)
    except CaptureError as exc:
        fields = exc.fields
        if "op" not in fields or "layer" not in fields:
            raise
        partial = getattr(exc, "partial_log", None)
        raw_layers = getattr(partial, "raw_layers", ())
        op = next(
            (
                entry
                for entry in reversed(raw_layers)
                if getattr(entry, "_label_raw", None) == fields["layer"]
            ),
            None,
        )
        if op is None:
            shape = fields["shape"]
            if not isinstance(shape, tuple) or not all(
                isinstance(dimension, int) for dimension in shape
            ):
                raise ValueError("CaptureError non-finite payload has an invalid shape field.")
            return FindNanResult(
                True,
                None,
                str(fields["layer"]),
                None,
                str(fields["op"]),
                str(fields["dtype"]),
                tuple(int(dimension) for dimension in shape),
                ("output",),
                _exception_source_line(exc),
                "nan",
                "first non-finite tensor",
                (),
                str(exc),
            )
        op_out = getattr(op, "out", None)
        kind = _nonfinite_kind(op_out) if isinstance(op_out, torch.Tensor) else "nan"
        return _result_from_op(op, kind=kind, source_line=_exception_source_line(exc))
    return _no_finding_result(scope="live operation outputs")


def _exception_source_line(exc: CaptureError) -> str | None:
    """Format the live source location attached to a capture error.

    Parameters
    ----------
    exc:
        Capture error raised at a non-finite op boundary.

    Returns
    -------
    str | None
        ``"file:line"`` when the live stack identified a user frame.
    """

    if exc.file_path is None or exc.line_no is None:
        return None
    return f"{exc.file_path}:{exc.line_no}"


def find_nan_in_trace(trace: Trace) -> FindNanResult:
    """Find the first NaN or Inf among saved trace outputs.

    Parameters
    ----------
    trace:
        Completed trace to inspect in execution order.

    Returns
    -------
    FindNanResult
        First saved non-finite output. Selectively saved traces explicitly
        report unsaved ancestor ops as an uncertainty zone.
    """

    any_unsaved = False
    for op in _ordered_ops(trace):
        if bool(getattr(op, "is_input", False)):
            continue
        if not bool(getattr(op, "has_saved_activation", False)):
            any_unsaved = True
            continue
        try:
            out = op.out
        except ValueError:
            any_unsaved = True
            continue
        if isinstance(out, torch.Tensor):
            kind = _nonfinite_kind(out)
            if kind != "none":
                uncertainty_zone = _unsaved_ancestors(trace, op)
                scope = "first among saved tensors" if any_unsaved else "first non-finite tensor"
                return _result_from_op(
                    op, kind=kind, scope=scope, uncertainty_zone=uncertainty_zone
                )
    return _no_finding_result(
        scope="saved tensors",
        uncertainty_zone=tuple(
            _op_label(op)
            for op in _ordered_ops(trace)
            if not bool(getattr(op, "is_input", False))
            and not bool(getattr(op, "has_saved_activation", False))
        )
        if any_unsaved
        else (),
    )


def bisect_nan(trace: Trace) -> BisectNanResult:
    """Locate the first saved op whose output contains NaN or Inf.

    Parameters
    ----------
    trace:
        Completed TorchLens trace with saved activations.

    Returns
    -------
    BisectNanResult
        Clear result object instead of raising when no non-finite saved
        activation is found.
    """

    unsaved_compute_ops = 0
    for op in _ordered_ops(trace):
        if bool(getattr(op, "is_input", False)):
            continue
        if not bool(getattr(op, "has_saved_activation", False)):
            if int(getattr(op, "step_index", 0) or 0) > 0:
                unsaved_compute_ops += 1
            continue
        try:
            out = op.out
        except ValueError:
            unsaved_compute_ops += 1
            continue
        if not isinstance(out, torch.Tensor):
            continue
        kind = _nonfinite_kind(out)
        if kind != "none":
            label = str(getattr(op, "layer_label", ""))
            source_line = _source_line(op)
            return BisectNanResult(
                found=True,
                op=op,
                label=label,
                source_line=source_line,
                kind=kind,
                message=f"First non-finite saved activation is {kind} at {label}.",
            )

    if unsaved_compute_ops:
        return BisectNanResult(
            found=False,
            op=None,
            label=None,
            source_line=None,
            kind="none",
            message=(
                "No NaN/Inf found in saved activations; no saved activation for the suspect "
                "region may be available. Re-trace with save=tl.func(...) for the suspect op, "
                "a wider predicate, or no selective save."
            ),
        )
    return BisectNanResult(
        found=False,
        op=None,
        label=None,
        source_line=None,
        kind="none",
        message="No NaN/Inf found in saved activations.",
    )
