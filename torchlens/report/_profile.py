"""Unified performance and resource profile tables for completed traces.

F09 re-base (sumfam item 12; costreport items 6-7): compute figures come
from the ONE canonical aggregation (``compute_aggregation``), never a
private re-sum; every export carries row roles and column additivity
(D5); percents are of the whole-capture KNOWN partition total (D4/D6);
``honesty()`` is a projection of per-cell evidence facts, never a null
check (D2/D9); instrumented wall time is labeled as such everywhere it
appears.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ..utils._multipass_access import is_multipass_layer

if TYPE_CHECKING:
    import pandas as pd

    from torchlens.data_classes.trace import Trace


ProfileLevel = Literal["op", "module", "call"]
ProfileSort = Literal["time", "flops", "activation_memory", "param_count"]

#: Column additivity metadata (costreport D5): stamped on every export so
#: no consumer sums an inclusive column. At op level the self and subtree
#: families coincide, so ``flops`` is additive there and ``flops_self``
#: is only emitted at module/call level.
PROFILE_COLUMN_ADDITIVITY: dict[str, bool] = {
    "time": True,
    "flops": True,  # op level: exclusive == inclusive; module/call: see flops_self
    "flops_self": True,
    "flops_pct": True,
    "activation_memory": True,
    "param_count": False,  # module rollups share containing params; never sum
}

#: The evidence rendering vocabulary (D2/D9): a closed enum with the
#: mandatory qualifier spellings. Host time is never bare "measured";
#: parameter counts are a declared inventory (never "measured" or
#: "estimated"); structure-only shape-derived cells are hypotheses.
HONESTY_LABELS: tuple[str, ...] = (
    "measured+instrumentation_inclusive",
    "formula_exact",
    "formula_exact+shape_derived",
    "estimated",
    "hypothesis",
    "unknown",
    "not_applicable",
)

#: Evidence strength order (weakest first) for D2 aggregation: a mixed
#: subtotal never upgrades to exact.
_EVIDENCE_STRENGTH: tuple[str, ...] = (
    "unknown",
    "hypothesis",
    "estimated",
    "formula_exact",
    "measured",
)


def _weakest_evidence(labels: list[str]) -> str:
    """The weakest evidence among members (D2); unknown when empty."""

    if not labels:
        return "unknown"
    return min(labels, key=lambda label: _EVIDENCE_STRENGTH.index(label))


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
    # F09 (costreport D4/D6): the whole-capture KNOWN forward partition
    # total -- the invariant percent denominator -- plus the unknown EVENT
    # count disclosed beside it (never a FLOP percentage).
    partition_total: int | None = None
    unknown_events: int = 0
    column_additivity: dict[str, bool] = field(default_factory=dict)

    def to_pandas(self, *, include_totals: bool = False, evidence: bool = False) -> pd.DataFrame:
        """Return a copy of the underlying profile dataframe.

        The stamped capture-honesty facts and the column-additivity map
        (D5) ride ``DataFrame.attrs`` so the exported table carries the
        same disclosures the repr shows.

        Parameters
        ----------
        include_totals:
            Append the explicit ``TOTAL`` row (``kind="total"``): the
            additive families carry the whole-capture partition totals --
            the sort/top_k view above it never changes these denominators
            (D6).
        evidence:
            Interleave per-cell evidence columns (``<column>_evidence``)
            into the frame (costreport D9's inline form). The labels are
            the same :meth:`honesty` projection; the ``TOTAL`` row carries
            no evidence labels.

        Returns
        -------
        pandas.DataFrame
            Resource profile rows.
        """

        frame = self.frame.copy()
        if evidence:
            honesty_frame = self._honesty_table()
            for column in ("time", "flops", "activation_memory", "param_count"):
                frame.insert(
                    frame.columns.get_loc(column) + 1,
                    f"{column}_evidence",
                    list(honesty_frame[column]),
                )
        if include_totals:
            pd = _require_pandas()
            records = frame.to_dict(orient="records")
            records.append(self._totals_row())
            frame = pd.DataFrame(records, columns=frame.columns)
        frame.attrs["torchlens_capture_honesty"] = {
            "schema": "torchlens.capture_honesty.v1",
            "capture_status": self.capture_status,
            "capture_verified": self.capture_verified,
            "capture_verification_reason": self.capture_verification_reason,
            "rescue_rerun": self.rescue_rerun,
            "structure_only": self.structure_only,
        }
        frame.attrs["column_additivity"] = dict(self.column_additivity)
        frame.attrs["partition_total"] = self.partition_total
        frame.attrs["time_basis"] = "instrumented"
        return frame

    def _totals_row(self) -> dict[str, Any]:
        """The explicit totals row (sumfam item 12): partition denominators."""

        row: dict[str, Any] = dict.fromkeys(self.frame.columns)
        row["name"] = "TOTAL"
        row["kind"] = "total"
        if "role" in row:
            row["role"] = None
        if "flops" in row and self.level == "op":
            row["flops"] = self.partition_total
        if "flops_self" in row:
            row["flops_self"] = self.partition_total
        if "flops_pct" in row and self.partition_total:
            row["flops_pct"] = 100.0
        if "time" in self.frame.columns:
            time_values = [
                value
                for value, kind in zip(self.frame["time"], self.frame["kind"], strict=True)
                if value == value and value is not None and kind == "op"
            ]
            row["time"] = sum(time_values) if time_values else None
        return row

    def __repr__(self) -> str:
        """Render the profile as a compact notebook-friendly table.

        The time column header names its basis (instrumented wall time --
        relative hotspot guidance, not a clean benchmark), and the footer
        prints the invariant partition total plus the unknown EVENT count
        (D4: never a FLOP percentage).

        Returns
        -------
        str
            Human-readable table representation.
        """

        display_frame = self.frame.copy()
        display_frame["time"] = display_frame["time"].map(_format_duration)
        display_frame = display_frame.rename(columns={"time": "time (instrumented)"})
        table = display_frame.to_string(index=False)
        banner = self._verification_banner()
        footer_parts = []
        if self.partition_total is not None:
            footer_parts.append(
                f"known forward FLOPs partition total: {self.partition_total} "
                "(whole capture; invariant under sort/top_k)"
            )
        if self.unknown_events:
            footer_parts.append(
                f"unknown-cost ops: {self.unknown_events} (event count; see unknown_flop_ops)"
            )
        footer = "\n".join(footer_parts)
        parts = [part for part in (banner, table, footer) if part]
        return "\n".join(parts)

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

    def honesty(self, *, as_summary: bool = False) -> pd.DataFrame | str:
        """Return per-cell evidence labels for resource quantities (D2/D9).

        A projection of cell facts, never a null check: FLOP labels come
        from each op's compute-record evidence (``formula_exact`` /
        ``estimated`` / ``unknown``) with weakest-member aggregation on
        rollup rows; host time is ``measured+instrumentation_inclusive``,
        never bare "measured"; parameter counts are ``formula_exact``
        (a declared inventory); structure-only shape-derived cells are
        ``hypothesis``.

        Parameters
        ----------
        as_summary:
            Return the one-line disclosure string that feeds the report's
            evidence line instead of the frame.

        Returns
        -------
        pandas.DataFrame | str
            Index-aligned evidence labels, or the disclosure line.
        """

        frame = self._honesty_table()
        if not as_summary:
            return frame
        flops_counts: dict[str, int] = {}
        for label in frame["flops"]:
            flops_counts[str(label)] = flops_counts.get(str(label), 0) + 1
        mix = ", ".join(
            f"{count} {label}" for label, count in sorted(flops_counts.items()) if count
        )
        return (
            f"evidence: flops [{mix}]; time measured on the host wall clock "
            "(time.time around wrapped calls, instrumentation-inclusive; "
            "never device kernel time); params formula_exact (declared "
            "inventory)"
        )

    def _honesty_table(self) -> pd.DataFrame:
        """The index-aligned per-cell evidence frame behind :meth:`honesty`."""

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


def _honesty_row(
    row: Mapping[Hashable, Any],
    *,
    flops_evidence: str | None = None,
    structure_only: bool = False,
) -> dict[str, str]:
    """Return per-cell evidence labels for one profile row (D2/D9).

    Parameters
    ----------
    row:
        Profile row produced by :func:`_row`.
    flops_evidence:
        The compute-record evidence for this row's FLOPs cell (op rows:
        the op's own record; rollup rows: the weakest member evidence).
        ``None`` falls back to presence-based ``formula_exact``/``unknown``
        for callers without an aggregation in hand.
    structure_only:
        Shape-derived cells on a structure-only capture are HYPOTHESES,
        and no execution happened, so time is not applicable.

    Returns
    -------
    dict[str, str]
        Index-aligned per-cell evidence labels from
        :data:`HONESTY_LABELS`.
    """

    if row.get("kind") == "boundary":
        # A boundary pseudo-row executed nothing: its absent cells are NOT
        # APPLICABLE, never "unknown" -- and never a fabricated evidence label
        # (costreport D2). Its retained input bytes are shape-derived facts.
        memory_label = "formula_exact+shape_derived" if structure_only is False else "hypothesis"
        return {
            "name": str(row["name"]),
            "time": "not_applicable",
            "flops": "not_applicable",
            "activation_memory": (
                memory_label if row["activation_memory"] is not None else "not_applicable"
            ),
            "param_count": "not_applicable",
        }
    if flops_evidence is None:
        flops_evidence = "formula_exact" if row["flops"] is not None else "unknown"
    if structure_only:
        return {
            "name": str(row["name"]),
            "time": "not_applicable",
            "flops": "hypothesis" if row["flops"] is not None else "unknown",
            "activation_memory": (
                "hypothesis" if row["activation_memory"] is not None else "unknown"
            ),
            # The parameter inventory is declared structure, not a value
            # hypothesis -- it stays formula_exact on structure-only captures.
            "param_count": (
                "formula_exact" if row["param_count"] is not None else "not_applicable"
            ),
        }
    return {
        "name": str(row["name"]),
        # D9: host time is never bare "measured" -- instrumentation overhead
        # is inside the number.
        "time": ("measured+instrumentation_inclusive" if row["time"] is not None else "unknown"),
        "flops": flops_evidence if row["flops"] is not None else "unknown",
        "activation_memory": (
            "formula_exact+shape_derived" if row["activation_memory"] is not None else "unknown"
        ),
        # D9: a parameter count is a declared inventory -- never "measured",
        # never "estimated"; an op that owns no parameters has no cell.
        "param_count": ("formula_exact" if row["param_count"] is not None else "not_applicable"),
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
        # ASCII rails: returned report strings are ASCII-canonical (lovely
        # bug 10 / summary-memo string contract).
        connector = "" if is_root else ("`-- " if is_last else "|-- ")
        lines.append(f"{prefix}{connector}{label}")
        if label in visited:
            return
        visited.add(label)
        descendants = children[label]
        child_prefix = prefix if is_root else prefix + ("    " if is_last else "|   ")
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


def _register_spellings(
    mapping: dict[str, Any],
    op_label: str,
    layer_label: str,
    value: Any,
    bare_counts: dict[str, int],
) -> None:
    """Register a per-op value under its pass-qualified label, plus the bare
    layer-label spelling when that spelling is unambiguous (single-pass).

    Member lists arriving from ``ModuleCall.ops`` use bare layer labels for
    single-pass layers and pass-qualified ``Op.label`` for expanded
    multi-pass members; both must resolve, and a multi-pass bare label must
    never silently pick one pass.
    """

    mapping[op_label] = value
    if bare_counts.get(layer_label, 0) == 1:
        mapping[layer_label] = value


def _flops_evidence_by_label(aggregation: Any) -> dict[str, str]:
    """Per-label FLOPs evidence from the canonical aggregation rows."""

    op_rows = [row for row in aggregation.rows if row.kind == "op"]
    bare_counts: dict[str, int] = {}
    for row in op_rows:
        bare_counts[row.label] = bare_counts.get(row.label, 0) + 1
    mapping: dict[str, str] = {}
    for row in op_rows:
        _register_spellings(
            mapping, row.op_label or row.label, row.label, row.evidence, bare_counts
        )
    return mapping


def _flops_values_by_label(aggregation: Any) -> dict[str, int | None]:
    """Per-label FLOPs values from the canonical aggregation rows."""

    op_rows = [row for row in aggregation.rows if row.kind == "op"]
    bare_counts: dict[str, int] = {}
    for row in op_rows:
        bare_counts[row.label] = bare_counts.get(row.label, 0) + 1
    mapping: dict[str, int | None] = {}
    for row in op_rows:
        _register_spellings(
            mapping, row.op_label or row.label, row.label, row.flops_fma2, bare_counts
        )
    return mapping


def _innermost_call_by_label(trace: Trace) -> dict[str, str | None]:
    """Map each op label to its innermost recorded module call (or None)."""

    ops = list(trace.layer_list)
    bare_counts: dict[str, int] = {}
    for op in ops:
        layer_label = str(getattr(op, "layer_label", ""))
        bare_counts[layer_label] = bare_counts.get(layer_label, 0) + 1
    owners: dict[str, str | None] = {}
    for op in ops:
        label = str(getattr(op, "label", None) or getattr(op, "layer_label", ""))
        layer_label = str(getattr(op, "layer_label", ""))
        stack = tuple(getattr(op, "module_call_stack", ()) or ())
        _register_spellings(
            owners, label, layer_label, str(stack[-1]) if stack else None, bare_counts
        )
    return owners


def _self_flops_of_labels(
    labels: list[str],
    owner_scope: set[str],
    owners: dict[str, str | None],
    flops_by_label: dict[str, int | None],
) -> int:
    """Sum known FLOPs of ops whose innermost call is inside ``owner_scope``."""

    total = 0
    for label in labels:
        if owners.get(label) in owner_scope:
            value = flops_by_label.get(label)
            if value is not None:
                total += value
    return total


def _member_flops(labels: list[str], flops_by_label: dict[str, int | None]) -> int | None:
    """Inclusive member FLOPs from the aggregation rows (never a raw re-sum)."""

    values = [
        value
        for label in labels
        if label in flops_by_label and (value := flops_by_label[label]) is not None
    ]
    return sum(values) if values else None


def _rollup_row(
    identity: tuple[str, str, int | None],
    ops: list[Any],
    scope: set[str],
    context: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """One module/call SUBTOTAL row with both compute families (D5).

    ``identity`` is the (name, kind, param_count) triple of the rollup.
    """

    name, kind, param_count = identity
    row = _row(name, kind, ops, param_count=param_count)
    row["role"] = "SUBTOTAL"
    member_labels = [str(getattr(op, "label", "")) for op in ops]
    flops_by_label = context["flops_by_label"]
    row["flops"] = _member_flops(member_labels, flops_by_label)
    row["flops_self"] = _self_flops_of_labels(
        member_labels, scope, context["owners"], flops_by_label
    )
    partition_total = context["partition_total"]
    row["flops_pct"] = 100.0 * row["flops_self"] / partition_total if partition_total > 0 else None
    flops_evidence = context["flops_evidence"]
    row_evidence = (
        _weakest_evidence(
            [flops_evidence[label] for label in member_labels if label in flops_evidence]
        )
        if member_labels
        else None
    )
    return row, row_evidence


def _validate_profile_args(level: str, sort_by: str, top_k: int | None) -> None:
    """Refuse invalid view arguments with the historical messages."""

    if level not in {"op", "module", "call"}:
        raise ValueError("level must be 'op', 'module', or 'call'.")
    if sort_by not in {"time", "flops", "activation_memory", "param_count"}:
        raise ValueError("sort_by must be 'time', 'flops', 'activation_memory', or 'param_count'.")
    if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0):
        raise ValueError("top_k must be a non-negative integer or None.")


def _assemble_level_rows(
    trace: Trace,
    level: str,
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str | None]]:
    """Build the requested granularity's rows plus per-row FLOPs evidence.

    ``context`` carries the shared aggregation-derived maps
    (partition_total, flops_evidence, flops_by_label, owners).
    """

    partition_total = context["partition_total"]
    flops_evidence = context["flops_evidence"]
    flops_by_label = context["flops_by_label"]
    rows: list[dict[str, Any]] = []
    row_evidence: list[str | None] = []
    if level == "op":
        rows.extend(_op_level_rows(trace))
        for row, op in zip(rows, trace.layer_list, strict=True):
            row["role"] = "OWNER" if row["kind"] == "op" else None
            row["flops_pct"] = (
                100.0 * row["flops"] / partition_total
                if row["flops"] is not None and partition_total > 0
                else None
            )
            op_label = str(getattr(op, "label", None) or getattr(op, "layer_label", ""))
            if row["kind"] == "op":
                # One aggregation for ALL compute facts (sumfam item 12):
                # the op row's FLOPs cell is the service row's value.
                row["flops"] = flops_by_label.get(op_label)
            row_evidence.append(flops_evidence.get(op_label))
    elif level == "call":
        for call in trace.module_calls.values():
            ops = _ops_for_labels(trace, list(getattr(call, "ops", ())))
            row, one_evidence = _rollup_row(
                (str(getattr(call, "call_label", "")), "call", getattr(call, "num_params", None)),
                ops,
                {str(call.call_label)},
                context,
            )
            rows.append(row)
            row_evidence.append(one_evidence)
    else:
        for module in trace.modules.values():
            labels = [label for call in module.calls.values() for label in call.ops]
            ops = _ops_for_labels(trace, labels)
            # Module.calls is keyed by call ordinal; ownership scopes need
            # the recorded call LABELS.
            row, one_evidence = _rollup_row(
                (
                    str(getattr(module, "address", "")),
                    "module",
                    getattr(module, "num_params", None),
                ),
                ops,
                {str(call.call_label) for call in module.calls.values()},
                context,
            )
            rows.append(row)
            row_evidence.append(one_evidence)
    return rows, row_evidence


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
        Percent denominators are UNAFFECTED (D6: a view never changes the
        partition total).

    Returns
    -------
    TraceProfile
        Printable profile object exposing :meth:`TraceProfile.to_pandas`.

    Notes
    -----
    Operation wall-times are captured under instrumentation. They are relative
    hotspot guidance, not clean benchmark timings. Compute figures come from
    the ONE canonical aggregation; module/call rows carry BOTH families --
    ``flops`` (inclusive subtree, NON-additive there) and ``flops_self``
    (exclusive, additive, sums to the partition total) -- with roles and
    column additivity stamped on every export (D5).
    """

    _validate_profile_args(level, sort_by, top_k)

    from ._compute_truth import compute_aggregation

    pd = _require_pandas()
    aggregation = compute_aggregation(trace)
    partition_total = int(aggregation.partition_total)
    flops_evidence = _flops_evidence_by_label(aggregation)
    flops_by_label = _flops_values_by_label(aggregation)
    owners = _innermost_call_by_label(trace)
    structure_only = bool(getattr(trace, "structure_only", False))

    rows, row_evidence = _assemble_level_rows(
        trace,
        level,
        {
            "partition_total": partition_total,
            "flops_evidence": flops_evidence,
            "flops_by_label": flops_by_label,
            "owners": owners,
        },
    )

    columns = [
        "name",
        "kind",
        "role",
        "op_count",
        "time",
        "flops",
        "flops_pct",
        "activation_memory",
        "saved_activation",
        "param_count",
        "dtype",
        "device",
    ]
    if level in ("module", "call"):
        columns.insert(columns.index("flops") + 1, "flops_self")
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
    honesty_rows = [
        _honesty_row(
            rows[index],
            flops_evidence=row_evidence[index],
            structure_only=structure_only,
        )
        for index in sorted_row_indices
    ]
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
    verification = {
        "capture_status": capture_status,
        "capture_verified": capture_verified,
        "capture_verification_reason": capture_verification_reason,
        "rescue_rerun": rescue_rerun,
        "structure_only": structure_only,
    }
    frame.attrs.update(verification)
    honesty_frame.attrs.update(verification)
    additivity = {
        name: PROFILE_COLUMN_ADDITIVITY[name]
        for name in frame.columns
        if name in PROFILE_COLUMN_ADDITIVITY
    }
    if level in ("module", "call"):
        # Inclusive rollups: the ``flops`` column is the subtree family
        # there and must never be summed (D5).
        additivity["flops"] = False
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
        partition_total=partition_total,
        unknown_events=aggregation.coverage.unknown,
        column_additivity=additivity,
    )
