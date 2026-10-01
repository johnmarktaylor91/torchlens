"""Unified performance and resource profile tables for completed traces."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..utils._multipass_access import is_multipass_layer

if TYPE_CHECKING:
    import pandas as pd

    from torchlens.data_classes.trace import Trace


ProfileLevel = Literal["op", "module", "call"]
ProfileSort = Literal["time", "flops", "activation_memory", "param_count"]


@dataclass(frozen=True)
class TraceProfile:
    """A printable, tabular resource profile assembled from one trace.

    Parameters
    ----------
    frame:
        Profile rows in the requested granularity.
    level:
        Granularity used to construct ``frame``.
    _honesty_frame:
        Index-aligned provenance labels for numeric resource columns.
    _tree_text:
        Module-call tree rendered from recorded call-parent relationships.

    Notes
    -----
    Operation wall-times are collected while TorchLens instrumentation is
    active. Use them to identify relative hotspots, not as clean benchmarks.
    """

    frame: pd.DataFrame
    level: ProfileLevel
    _honesty_frame: pd.DataFrame | None = None
    _tree_text: str = ""
    # Capture-level verification facts (round-7 R67/R88): the report honesty
    # contract requires a rescued/ceilinged capture to stay visible in profile
    # output, so these are stamped from the source trace at build time.
    capture_status: str = "unknown"
    capture_verified: bool | None = None
    capture_verification_reason: str | None = None
    rescue_rerun: bool = False
    structure_only: bool = False

    def to_pandas(self) -> pd.DataFrame:
        """Return a copy of the underlying profile dataframe.

        The stamped capture-honesty facts ride ``DataFrame.attrs`` so the
        exported table carries the same disclosure the repr banner shows.

        Returns
        -------
        pandas.DataFrame
            Resource profile rows.
        """

        frame = self.frame.copy()
        frame.attrs["torchlens_capture_honesty"] = {
            "schema": "torchlens.capture_honesty.v1",
            "capture_status": self.capture_status,
            "capture_verified": self.capture_verified,
            "capture_verification_reason": self.capture_verification_reason,
            "rescue_rerun": self.rescue_rerun,
            "structure_only": self.structure_only,
        }
        return frame

    def __repr__(self) -> str:
        """Render the profile as a compact notebook-friendly table.

        Returns
        -------
        str
            Human-readable table representation.
        """

        display_frame = self.frame.copy()
        display_frame["time"] = display_frame["time"].map(_format_duration)
        table = display_frame.to_string(index=False)
        banner = self._verification_banner()
        return f"{banner}\n{table}" if banner else table

    def _verification_banner(self) -> str:
        """Return the mandatory disclosure line for a non-clean capture.

        Returns
        -------
        str
            One-line disclosure when the capture is unverified, rescued, or
            settled non-complete; empty for a clean complete capture.
        """

        notes = []
        # L7a G3 render honesty: hypothesis shapes must never present as
        # measurements (memo sec 3.4).
        if self.structure_only:
            notes.append("structure-only capture -- shapes/dtypes are HYPOTHESES, not measurements")
        if self.capture_status not in ("complete", "unknown"):
            notes.append(f"capture outcome: {self.capture_status}")
        if self.capture_verified is False:
            reason = self.capture_verification_reason or "unrecorded reason"
            notes.append(f"capture UNVERIFIED ({reason})")
        if self.rescue_rerun:
            notes.append("rescue re-run result (mode_rescue_rerun)")
        if not notes:
            return ""
        return "! " + "; ".join(notes) + " -- rows below may undercount what ran"

    def honesty(self) -> pd.DataFrame:
        """Return index-aligned evidence labels for resource quantities.

        Returns
        -------
        pandas.DataFrame
            Labels from ``{"measured", "estimated", "unknown"}`` for timing,
            FLOP, activation-memory, and parameter-count columns. The existing
            profile values and dtypes are not changed.
        """

        if self._honesty_frame is not None:
            return self._honesty_frame.copy()
        pd = _require_pandas()
        rows = [_honesty_row(row) for row in self.frame.to_dict(orient="records")]
        return pd.DataFrame(
            rows,
            columns=["name", "time", "flops", "activation_memory", "param_count"],
        )

    def tree(self) -> str:
        """Return a torchinfo-style tree following recorded call nesting.

        The indentation follows ``ModuleCall.call_parent`` relationships, not
        module address depth. A module invoked from inside another call is
        therefore nested under that invocation even when its attribute address
        suggests a different hierarchy.

        Returns
        -------
        str
            Indented module-call labels in execution order within each parent.
        """

        return self._tree_text


def _require_pandas() -> Any:
    """Import pandas with the standard TorchLens tabular-extra guidance.

    Returns
    -------
    Any
        Imported pandas module.
    """

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for this feature. Install with `pip install torchlens[tabular]`."
        ) from exc
    return pd


def _ops_for_labels(trace: Trace, labels: list[str]) -> list[Any]:
    """Resolve pass-qualified labels to operation records.

    Parameters
    ----------
    trace:
        Source trace.
    labels:
        Pass-qualified operation labels.

    Returns
    -------
    list[Any]
        Resolved per-pass operation records; stale labels are ignored.

    Notes
    -----
    A bare (non-pass-qualified) layer label resolves to an aggregate ``Layer``,
    not a single ``Op``. The root ``ModuleCall`` stores such bare labels while
    submodule calls store pass-qualified op labels. For a multi-pass (recurrent)
    layer that aggregate cannot answer per-pass reads (``has_saved_activation``,
    ``func_duration``, ...): ``Layer._single_pass_or_error`` raises the deliberate
    multi-pass ``ValueError`` tripwire, which every downstream ``getattr(op, ...,
    default)`` in :func:`_row`/:func:`_sum_optional`/:func:`_values` would leak
    (``getattr`` shields only ``AttributeError``), crashing ``profile("call")``
    and ``profile("module")`` on ANY recurrent model. Expanding the aggregate to
    its concrete per-pass Ops surfaces the true per-pass values and also repairs
    the silent undercount (N passes would otherwise collapse into one row).
    """

    ops: list[Any] = []
    for label in labels:
        try:
            resolved = trace[label]
        except (KeyError, ValueError):
            continue
        if is_multipass_layer(resolved):
            ops.extend(resolved.ops.values())
        else:
            ops.append(resolved)
    return ops


def _sum_optional(ops: list[Any], field: str) -> int | float | None:
    """Sum a metric without turning wholly unavailable metadata into zero.

    Parameters
    ----------
    ops:
        Operations to aggregate.
    field:
        Numeric metadata field.

    Returns
    -------
    int | float | None
        Sum when at least one value is available, otherwise ``None``.
    """

    values = [getattr(op, field, None) for op in ops]
    if field == "func_duration":
        available = [float(value) for value in values if value is not None]
    else:
        available = [int(value) for value in values if value is not None]
    return sum(available) if available else None


def _format_duration(seconds: float | None) -> str:
    """Format an operation duration using a compact human-readable unit.

    Parameters
    ----------
    seconds:
        Duration in seconds, if instrumentation captured one.

    Returns
    -------
    str
        Duration rendered in seconds, milliseconds, or microseconds.
    """

    if seconds is None:
        return ""
    if seconds >= 1:
        return f"{seconds:.3f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds * 1e6:.3f} us"


def _values(ops: list[Any], field: str) -> str | None:
    """Format the distinct non-null values of one operation metadata field.

    Parameters
    ----------
    ops:
        Operations to inspect.
    field:
        Metadata field name.

    Returns
    -------
    str | None
        One value or a comma-separated stable set, if available.
    """

    values = sorted({str(value) for op in ops if (value := getattr(op, field, None)) is not None})
    return ", ".join(values) if values else None


def _row(name: str, kind: str, ops: list[Any], *, param_count: int | None) -> dict[str, Any]:
    """Build one consistent profile row from operation metadata.

    Parameters
    ----------
    name:
        Stable row label.
    kind:
        Row granularity label.
    ops:
        Operations represented by the row.
    param_count:
        Parameter count owned by the corresponding module scope.

    Returns
    -------
    dict[str, Any]
        Profile row.
    """

    saved = [bool(getattr(op, "has_saved_activation", False)) for op in ops]
    saved_activation: bool | str | None
    if not saved:
        saved_activation = None
    elif all(saved):
        saved_activation = True
    elif not any(saved):
        saved_activation = False
    else:
        saved_activation = "partial"
    return {
        "name": name,
        "kind": kind,
        "op_count": len(ops),
        "time": _sum_optional(ops, "func_duration"),
        "flops": _sum_optional(ops, "flops_forward"),
        "activation_memory": _sum_optional(ops, "activation_memory"),
        "saved_activation": saved_activation,
        "param_count": param_count,
        "dtype": _values(ops, "dtype"),
        # ``Op`` has no ``device`` attribute; the canonical field is ``device_ref``
        # (a ``DeviceRef`` whose ``str`` is the device name, e.g. ``"cpu"``).
        # Reading the nonexistent ``device`` left this column permanently ``None``
        # on every model via the ``getattr(op, field, None)`` default.
        "device": _values(ops, "device_ref"),
    }


def _honesty_row(row: Mapping[Hashable, Any]) -> dict[str, str]:
    """Return evidence-source labels for one numeric profile row.

    Parameters
    ----------
    row:
        Profile row produced by :func:`_row`.

    Returns
    -------
    dict[str, str]
        Index-aligned source labels. Instrumented wall time is measured;
        analytic FLOPs, shape-derived memory, and metadata parameter counts are
        estimated; absent values are unknown.
    """

    if row.get("kind") == "boundary":
        # A boundary pseudo-row executed nothing: its absent cells are NOT
        # APPLICABLE, never "unknown" -- and never a fabricated evidence label
        # (costreport D2).
        return {
            "name": str(row["name"]),
            "time": "not_applicable",
            "flops": "not_applicable",
            "activation_memory": (
                "estimated" if row["activation_memory"] is not None else "not_applicable"
            ),
            "param_count": "not_applicable",
        }
    return {
        "name": str(row["name"]),
        "time": "measured" if row["time"] is not None else "unknown",
        "flops": "estimated" if row["flops"] is not None else "unknown",
        "activation_memory": ("estimated" if row["activation_memory"] is not None else "unknown"),
        "param_count": "estimated" if row["param_count"] is not None else "unknown",
    }


def _build_call_tree(trace: Trace) -> str:
    """Render module calls from recorded call-parent relationships.

    Parameters
    ----------
    trace:
        Completed trace containing module-call records.

    Returns
    -------
    str
        Indented call tree, or an empty string when no calls are recorded.
    """

    calls = sorted(trace.module_calls.values(), key=lambda call: int(call.ordinal_index))
    if not calls:
        return ""
    calls_by_label = {str(call.call_label): call for call in calls}
    children: dict[str, list[Any]] = {label: [] for label in calls_by_label}
    roots: list[Any] = []
    for call in calls:
        parent = getattr(call, "call_parent", None)
        if parent is None or str(parent) not in calls_by_label:
            roots.append(call)
        else:
            children[str(parent)].append(call)
    for siblings in children.values():
        siblings.sort(key=lambda call: int(call.ordinal_index))

    lines: list[str] = []
    visited: set[str] = set()

    def visit(call: Any, prefix: str, is_last: bool, is_root: bool) -> None:
        """Append one call and its recorded descendants."""

        label = str(call.call_label)
        connector = "" if is_root else ("└── " if is_last else "├── ")
        lines.append(f"{prefix}{connector}{label}")
        if label in visited:
            return
        visited.add(label)
        descendants = children[label]
        child_prefix = prefix if is_root else prefix + ("    " if is_last else "│   ")
        for index, child in enumerate(descendants):
            visit(child, child_prefix, index == len(descendants) - 1, False)

    for root_index, root in enumerate(roots):
        visit(root, "", root_index == len(roots) - 1, True)
    for call in calls:
        if str(call.call_label) not in visited:
            visit(call, "", True, True)
    return "\n".join(lines)


def _op_level_rows(trace: Trace) -> list[dict[str, Any]]:
    """Build op-level profile rows; boundary pseudo-rows own nothing additive.

    Identity partition (A1 / costreport D7): a non-null additive cell on a
    boundary row is a CI failure, never a display choice. Input rows keep
    their owned external-input bytes; output alias rows own no bytes either.
    """

    rows: list[dict[str, Any]] = []
    for op in trace.layer_list:
        is_boundary = bool(getattr(op, "is_input", False)) or bool(getattr(op, "is_output", False))
        row = _row(
            str(getattr(op, "label", None) or getattr(op, "layer_label", "")),
            "boundary" if is_boundary else "op",
            [op],
            param_count=None if is_boundary else getattr(op, "num_params", None),
        )
        if is_boundary:
            row["time"] = None
            row["flops"] = None
            if getattr(op, "is_output", False):
                row["activation_memory"] = None
                row["saved_activation"] = None
        rows.append(row)
    return rows


def build_profile(
    trace: Trace,
    *,
    level: ProfileLevel = "op",
    sort_by: ProfileSort = "time",
    ascending: bool = False,
    top_k: int | None = None,
) -> TraceProfile:
    """Build a unified op, module, or invocation resource profile.

    Parameters
    ----------
    trace:
        Completed trace whose existing metadata is reported.
    level:
        ``"op"`` for one row per operation, ``"module"`` for one row per
        module address, or ``"call"`` for one row per module invocation.
    sort_by:
        Column used for sorting. ``"time"`` sorts descending by default.
    ascending:
        Whether to reverse the default descending order.
    top_k:
        Number of sorted bottleneck rows to retain. ``None`` retains all rows.

    Returns
    -------
    TraceProfile
        Printable profile object exposing :meth:`TraceProfile.to_pandas`.

    Notes
    -----
    Operation wall-times are captured under instrumentation. They are relative
    hotspot guidance, not clean benchmark timings.
    """

    if level not in {"op", "module", "call"}:
        raise ValueError("level must be 'op', 'module', or 'call'.")
    if sort_by not in {"time", "flops", "activation_memory", "param_count"}:
        raise ValueError("sort_by must be 'time', 'flops', 'activation_memory', or 'param_count'.")
    if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0):
        raise ValueError("top_k must be a non-negative integer or None.")

    pd = _require_pandas()
    rows: list[dict[str, Any]] = []
    if level == "op":
        rows.extend(_op_level_rows(trace))
    elif level == "call":
        for call in trace.module_calls.values():
            ops = _ops_for_labels(trace, list(getattr(call, "ops", ())))
            rows.append(
                _row(
                    str(getattr(call, "call_label", "")),
                    "call",
                    ops,
                    param_count=getattr(call, "num_params", None),
                )
            )
    else:
        for module in trace.modules.values():
            labels = [label for call in module.calls.values() for label in call.ops]
            ops = _ops_for_labels(trace, labels)
            rows.append(
                _row(
                    str(getattr(module, "address", "")),
                    "module",
                    ops,
                    param_count=getattr(module, "num_params", None),
                )
            )

    columns = [
        "name",
        "kind",
        "op_count",
        "time",
        "flops",
        "activation_memory",
        "saved_activation",
        "param_count",
        "dtype",
        "device",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    for index, row in enumerate(rows):
        row["_profile_row_index"] = index
    frame["_profile_row_index"] = list(range(len(frame)))
    frame = frame.sort_values(sort_by, ascending=ascending, na_position="last", kind="stable")
    if top_k is not None:
        frame = frame.head(top_k)
    sorted_row_indices = frame["_profile_row_index"].tolist()
    frame = frame.drop(columns=["_profile_row_index"])
    frame = frame.reset_index(drop=True)
    frame.attrs.update({"level": level, "sort_by": sort_by, "ascending": ascending})
    honesty_rows = [_honesty_row(rows[index]) for index in sorted_row_indices]
    honesty_frame = pd.DataFrame(
        honesty_rows,
        columns=["name", "time", "flops", "activation_memory", "param_count"],
    )
    if top_k is not None:
        frame.attrs["top_k"] = top_k
    outcome = getattr(trace, "outcome", None)
    status_value = getattr(getattr(outcome, "status", None), "value", None)
    capture_status = str(status_value) if status_value is not None else "unknown"
    capture_verified = getattr(trace, "capture_verified", None)
    capture_verification_reason = getattr(trace, "capture_verification_reason", None)
    rescue_rerun = bool(getattr(trace, "rescue_rerun", None) or False)
    structure_only = bool(getattr(trace, "structure_only", False))
    verification = {
        "capture_status": capture_status,
        "capture_verified": capture_verified,
        "capture_verification_reason": capture_verification_reason,
        "rescue_rerun": rescue_rerun,
        "structure_only": structure_only,
    }
    frame.attrs.update(verification)
    honesty_frame.attrs.update(verification)
    return TraceProfile(
        frame=frame,
        level=level,
        _honesty_frame=honesty_frame,
        _tree_text=_build_call_tree(trace),
        capture_status=capture_status,
        capture_verified=capture_verified,
        capture_verification_reason=capture_verification_reason,
        rescue_rerun=rescue_rerun,
        structure_only=structure_only,
    )
