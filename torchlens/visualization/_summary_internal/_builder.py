"""Build compact text summaries for ``Trace`` objects."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)

from ..._errors import InvalidArgumentError
from ..._source_links import file_line_text
from ...quantities import Macs
from ...utils.display import format_flops, human_readable_size
from ._discoverability import format_discoverability_summary

if TYPE_CHECKING:
    from ..data_classes.layer import Layer
    from ..data_classes.module import Module
    from ..data_classes.op import Op
    from ..data_classes.trace import ConditionalEvent, Trace


SummaryLevel = Literal[
    "overview", "graph", "memory", "control_flow", "compute", "cost", "waterfall", "output"
]
SummaryMode = Literal["auto", "rolled", "unrolled"]

_LEVEL_ALIASES: dict[str, str] = {"cost": "compute"}

_COLUMN_LABELS: dict[str, str] = {
    "name": "Layer",
    "shape": "Output Shape",
    "params": "Params",
    "train": "Train",
    "class": "Class",
    "parents": "Connected To",
    "dtype": "Dtype",
    "tensor_mb": "Tensor MB",
    "running_mb": "Cum Tensor MB",
    "flops": "Fwd FLOPs",
    "macs": "MACs",
    "time_ms": "Time (ms)",
    "start_ms": "Start (ms)",
    "end_ms": "End (ms)",
    "memory": "Memory",
    "site": "Site",
    "source": "Source",
    "taken": "Taken",
    "bool_layer": "Bool Layer",
    "branch_ops": "Branch Ops",
    "notes": "Notes",
    "batch_item": "Batch",
    "rank": "Rank",
    "label": "Label",
    "prob": "Prob",
}

_LEVEL_DEFAULT_FIELDS: dict[str, list[str]] = {
    "overview": ["name", "shape", "params", "train"],
    "graph": ["name", "shape", "params", "parents"],
    "memory": ["name", "shape", "dtype", "tensor_mb", "running_mb"],
    "control_flow": ["site", "source", "taken", "bool_layer", "branch_ops", "notes"],
    "compute": ["name", "params", "flops", "macs", "time_ms", "dtype"],
    "waterfall": ["name", "start_ms", "time_ms", "end_ms", "memory"],
    "output": ["batch_item", "rank", "label", "prob"],
}


def render_model_summary(
    trace: Trace,
    *,
    level: SummaryLevel = "overview",
    preset: SummaryLevel | None = None,
    fields: list[str] | None = None,
    columns: list[str] | None = None,
    mode: SummaryMode = "auto",
    show_ops: bool = False,
    include_ops: bool | None = None,
    max_rows: int | None = 200,
    print_to: Callable[[str], None] | None = None,
    count_fma_as_two: bool | None = None,
    show_input_preprocessing_details: bool = False,
) -> str:
    """Render a textual summary for a ``Trace``.

    Parameters
    ----------
    trace:
        Logged model metadata to summarize.
    level:
        Primary summary level. This is the public selector used by the sprint
        prompt and maps to the design doc's preset concept.
    preset:
        Alias for ``level`` retained for compatibility with the design wording.
    fields:
        Explicit column selection for the primary table.
    columns:
        Alias for ``fields``.
    mode:
        Operation aggregation mode. ``"rolled"`` uses ``Layer`` rows,
        ``"unrolled"`` uses ``Op`` rows, and ``"auto"`` chooses
        based on recurrence.
    show_ops:
        Whether to append an operation table after the primary summary.
    include_ops:
        Alias for ``show_ops`` retained for compatibility with the design wording.
    max_rows:
        Maximum number of rows to render per table. ``None`` disables truncation.
    print_to:
        Optional callable that receives the rendered summary.
    show_input_preprocessing_details:
        Whether to include input preprocessing verification/source detail.

    Returns
    -------
    str
        Rendered summary text.
    """
    resolved_level = _resolve_level(level=level, preset=preset)
    resolved_show_ops = _resolve_show_ops(show_ops=show_ops, include_ops=include_ops)
    resolved_fields = _resolve_fields(resolved_level, fields=fields, columns=columns)

    if not trace._tracing_finished:
        legacy_text = _render_in_progress_summary(
            trace=trace,
            fields=resolved_fields,
            mode=mode,
            show_ops=resolved_show_ops,
            max_rows=max_rows,
        )
    else:
        legacy_text = _render_finished_summary(
            trace=trace,
            level=resolved_level,
            fields=resolved_fields,
            mode=mode,
            show_ops=resolved_show_ops,
            max_rows=max_rows,
            count_fma_as_two=count_fma_as_two,
        )
    text = (
        f"{format_discoverability_summary(trace, show_input_preprocessing_details=show_input_preprocessing_details)}"
        f"\n\n{legacy_text}"
    )
    banner = _capture_verification_banner(trace)
    if banner:
        text = f"{banner}\n{text}"
    if print_to is not None:
        print_to(text)
    return text


def _capture_verification_banner(trace: Trace) -> str:
    """Return the disclosure line for a non-clean capture, or ``""``.

    Round-7 R67/R88: the report honesty contract requires a rescued or
    ceilinged capture (``capture_verified=False``) and any non-COMPLETE
    settled outcome to stay VISIBLE in summary output rather than rendering
    indistinguishably from a clean complete capture.

    Parameters
    ----------
    trace:
        Trace being summarized.

    Returns
    -------
    str
        One-line disclosure, or empty for a clean complete capture.
    """

    notes = []
    # L7a G3 render honesty: a structure-only capture's shapes/dtypes are
    # HYPOTHESES; every human surface says so (memo sec 3.4).
    if bool(getattr(trace, "structure_only", False)):
        notes.append("structure-only capture -- shapes/dtypes are HYPOTHESES, not measurements")
    status_value = getattr(getattr(getattr(trace, "outcome", None), "status", None), "value", None)
    if status_value not in (None, "complete"):
        notes.append(f"capture outcome: {status_value}")
    if getattr(trace, "capture_verified", None) is False:
        reason = getattr(trace, "capture_verification_reason", None) or "unrecorded reason"
        notes.append(f"capture UNVERIFIED ({reason})")
    if bool(getattr(trace, "rescue_rerun", None) or False):
        notes.append("rescue re-run result (mode_rescue_rerun)")
    if not notes:
        return ""
    return "! " + "; ".join(notes) + " -- this summary may undercount what ran"


def format_model_repr(trace: Trace) -> str:
    """Return a short ``repr`` string for a ``Trace``.

    Parameters
    ----------
    trace:
        Logged model metadata to summarize.

    Returns
    -------
    str
        Short two-line representation.
    """
    state = getattr(getattr(trace, "state", None), "name", "UNKNOWN")
    model_class_name = getattr(trace, "model_class_name", None)
    tracing_finished = getattr(trace, "_tracing_finished", True)
    if not tracing_finished:
        return (
            f"Trace(name={getattr(trace, 'trace_label', None)!r}, "
            f"model_class_qualname={model_class_name!r}, layers={_live_op_count(trace)}, "
            f"state={state})"
        )

    layer_logs = getattr(trace, "layer_logs", {}) or {}
    # Weightsfree memo L7: the repr was an unmarked channel — a slice of a
    # hypothesis is a hypothesis, and so is the identity card. The claim
    # ladder rides the session discharge state (HYPOTHESIS / CORROBORATED /
    # REFUTED), never a bare flag.
    structure_note = ""
    if bool(getattr(trace, "structure_only", False)):
        from ...capture.structure_only import claim_status_for

        structure_note = f", structure_only={claim_status_for(trace).value.upper()}"
    return (
        f"Trace(name={getattr(trace, 'trace_label', None)!r}, "
        f"model_class_qualname={model_class_name!r}, layers={len(layer_logs)}, "
        f"state={state}{structure_note})"
    )


def _live_op_count(trace: Trace) -> int:
    """Return live op-event count when capture events are present.

    Parameters
    ----------
    trace:
        Trace to inspect.

    Returns
    -------
    int
        Number of live operation events or raw logs.
    """

    events = getattr(trace, "capture_events", None)
    if events is not None and getattr(events, "op_events", None) is not None:
        return len(events.op_events)
    return len(trace._raw_graph_ws.raw_layer_dict)


def _resolve_level(*, level: SummaryLevel, preset: SummaryLevel | None) -> str:
    """Resolve the level/preset selector to a canonical level name.

    Parameters
    ----------
    level:
        User-supplied level selector.
    preset:
        Optional alias supplied via the design-doc name.

    Returns
    -------
    str
        Canonical level name.

    Raises
    ------
    ValueError
        If conflicting selectors are provided.
    """
    if preset is not None and preset != level:
        raise InvalidArgumentError(
            "Pass either `level` or `preset`, not both with different values",
            code="summary_option_conflict",
            remedy="pass level= or preset=, or give both the same value",
        )
    selected_name: str = preset or level
    selected_name = _LEVEL_ALIASES.get(selected_name, selected_name)
    if selected_name not in _LEVEL_DEFAULT_FIELDS:
        raise InvalidArgumentError(
            f"Unsupported summary level: {selected_name!r}",
            code="summary_level_invalid",
            remedy="pass a documented summary level",
            argument="level",
        )
    return selected_name


def _resolve_show_ops(*, show_ops: bool, include_ops: bool | None) -> bool:
    """Resolve the operation-dump toggle.

    Parameters
    ----------
    show_ops:
        Public operation toggle used by the sprint prompt.
    include_ops:
        Optional alias supplied via the design-doc name.

    Returns
    -------
    bool
        Final operation-dump toggle.

    Raises
    ------
    ValueError
        If conflicting values are provided.
    """
    # NOTE: The prompt names `show_ops` while the design doc names `include_ops`.
    # This implementation treats them as strict aliases and rejects conflicting
    # values rather than silently guessing precedence.
    if include_ops is not None and include_ops != show_ops:
        raise InvalidArgumentError(
            "Pass either `show_ops` or `include_ops`, not both with different values",
            code="summary_option_conflict",
            remedy="pass show_ops= or include_ops=, or give both the same value",
        )
    return include_ops if include_ops is not None else show_ops


def _resolve_fields(
    level: str,
    *,
    fields: list[str] | None,
    columns: list[str] | None,
) -> list[str]:
    """Resolve the primary-table field selection.

    Parameters
    ----------
    level:
        Canonical summary level.
    fields:
        Primary field selector.
    columns:
        Alias for ``fields``.

    Returns
    -------
    list[str]
        Resolved field list.

    Raises
    ------
    ValueError
        If conflicting values are provided.
    """
    if fields is not None and columns is not None and fields != columns:
        raise InvalidArgumentError(
            "Pass either `fields` or `columns`, not both with different values",
            code="summary_option_conflict",
            remedy="pass fields= or columns=, or give both the same value",
        )
    selected = columns if columns is not None else fields
    if selected is None:
        return list(_LEVEL_DEFAULT_FIELDS[level])
    unknown = [field for field in selected if field not in _COLUMN_LABELS]
    if unknown:
        raise InvalidArgumentError(
            f"Unsupported summary fields: {unknown}",
            code="summary_fields_invalid",
            remedy="pass documented summary field names",
            fields=unknown,
        )
    return list(selected)


def _render_in_progress_summary(
    *,
    trace: Trace,
    fields: Sequence[str],
    mode: SummaryMode,
    show_ops: bool,
    max_rows: int | None,
) -> str:
    """Render a truthful summary while the pass is still in progress.

    Parameters
    ----------
    trace:
        Log still being populated.
    fields:
        Requested primary fields.
    mode:
        Requested aggregation mode.
    show_ops:
        Whether to include raw operation rows.
    max_rows:
        Maximum number of rows to render.

    Returns
    -------
    str
        Rendered in-progress summary.
    """
    lines = [
        f"Model: {trace.model_class_name}",
        "Status: pass in progress; postprocessing has not finished yet.",
        f"Ops logged so far: {_live_op_count(trace)}",
    ]
    rows = _live_op_rows(trace)
    if show_ops and rows:
        display_fields = [field for field in fields if field in {"name", "shape", "dtype"}] or [
            "name",
            "shape",
            "dtype",
        ]
        lines.extend(
            [
                "",
                "Raw Operations:",
                _render_table(
                    display_fields, rows, max_rows=max_rows, label_overrides={"name": "Op"}
                ),
            ]
        )
    return "\n".join(lines)


def _live_op_rows(trace: Trace) -> list[dict[str, str]]:
    """Return display rows for live operation records.

    Parameters
    ----------
    trace:
        Trace to inspect.

    Returns
    -------
    list[dict[str, str]]
        Rows containing operation name, shape, and dtype strings.
    """

    events = getattr(trace, "capture_events", None)
    if events is not None and getattr(events, "op_events", None) is not None:
        rows = []
        for event in events.amended_op_records():
            rows.append(
                {
                    "name": str(event.layer_label_raw or event.label_raw),
                    "shape": _shape_str(event.output.tensor.shape),
                    "dtype": _dtype_str(event.output.tensor.dtype),
                }
            )
        return rows

    raw_graph_ws = trace.__dict__.get("_raw_graph_ws")
    if raw_graph_ws is not None and raw_graph_ws.raw_layer_dict:
        rows = []
        for raw_label in trace._raw_graph_ws.raw_layer_labels_list:
            entry = trace._raw_graph_ws.raw_layer_dict[raw_label]
            rows.append(
                {
                    "name": str(
                        getattr(entry, "_layer_label_raw", None)
                        or getattr(entry, "_label_raw", raw_label)
                    ),
                    "shape": _shape_str(getattr(entry, "shape", None)),
                    "dtype": _dtype_str(getattr(entry, "dtype", None)),
                }
            )
        return rows
    return []


def _render_finished_summary(
    *,
    trace: Trace,
    level: str,
    fields: Sequence[str],
    mode: SummaryMode,
    show_ops: bool,
    max_rows: int | None,
    count_fma_as_two: bool | None = None,
) -> str:
    """Render a summary for a finalized ``Trace``.

    Parameters
    ----------
    trace:
        Finalized log object.
    level:
        Canonical summary level.
    fields:
        Primary field selection.
    mode:
        Operation aggregation mode.
    show_ops:
        Whether to append an operation table.
    max_rows:
        Maximum number of rows to render per table.

    Returns
    -------
    str
        Rendered summary text.
    """
    primary_rows, footer_lines = _build_level_rows(
        trace=trace, level=level, mode=mode, count_fma_as_two=count_fma_as_two
    )
    lines = [
        _level_title(trace=trace, level=level),
        _render_table(
            fields,
            primary_rows,
            max_rows=max_rows,
            label_overrides=_name_label_override(level),
        ),
    ]
    if footer_lines:
        lines.extend(footer_lines)
    if show_ops and level != "control_flow":
        op_fields = _default_op_fields(level)
        op_rows, op_footer_lines = _build_operation_rows(trace=trace, mode=mode, level=level)
        lines.extend(
            [
                "",
                "Operations:",
                _render_table(
                    op_fields, op_rows, max_rows=max_rows, label_overrides={"name": "Op"}
                ),
            ]
        )
        if op_footer_lines:
            lines.extend(op_footer_lines)
    return "\n".join(lines)


def _name_label_override(level: str) -> dict[str, str] | None:
    """Return the row-kind-aware heading for the ``name`` column (A10).

    Module-row tables head their name column "Module (type)"; op-row tables
    head it "Op". The one-size-fits-all "Layer" heading mislabeled both.
    """

    if level in {"overview", "graph", "compute"}:
        return {"name": "Module (type)"}
    if level in {"memory", "waterfall"}:
        return {"name": "Op"}
    return None


def _level_title(*, trace: Trace, level: str) -> str:
    """Return the section title for a summary level.

    Parameters
    ----------
    trace:
        Finalized log object.
    level:
        Canonical level name.

    Returns
    -------
    str
        Section title.
    """
    title_map = {
        "overview": f"Model: {trace.model_class_name}",
        "graph": f"Graph Summary: {trace.model_class_name}",
        "memory": f"Memory Summary: {trace.model_class_name}",
        "control_flow": f"Control-Flow Summary: {trace.model_class_name}",
        "compute": f"Compute Summary: {trace.model_class_name}",
        "waterfall": f"Waterfall Summary: {trace.model_class_name}",
        "output": f"Output Summary: {trace.model_class_name}",
    }
    return title_map[level]


def _build_level_rows(
    *,
    trace: Trace,
    level: str,
    mode: SummaryMode,
    count_fma_as_two: bool | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    """Build rows and footer lines for one summary level.

    Parameters
    ----------
    trace:
        Finalized log object.
    level:
        Canonical level name.
    mode:
        Operation aggregation mode.

    Returns
    -------
    tuple[list[dict[str, str]], list[str]]
        Primary table rows and footer lines.
    """
    if level == "overview":
        return _build_overview_rows(trace, count_fma_as_two=count_fma_as_two)
    if level == "graph":
        return _build_graph_rows(trace)
    if level == "memory":
        return _build_memory_rows(trace, mode=mode)
    if level == "control_flow":
        return _build_control_flow_rows(trace)
    if level == "waterfall":
        return _build_waterfall_rows(trace, mode=mode)
    if level == "output":
        return _build_output_rows(trace)
    return _build_compute_rows(trace, count_fma_as_two=count_fma_as_two)


def _build_overview_rows(
    trace: Trace, count_fma_as_two: bool | None = None
) -> tuple[list[dict[str, str]], list[str]]:
    """Build the default overview rows.

    Parameters
    ----------
    trace:
        Finalized log object.

    Returns
    -------
    tuple[list[dict[str, str]], list[str]]
        Overview rows and footer lines.
    """
    rows: list[dict[str, str]] = [
        {
            "name": "input",
            "shape": _combined_shape_str(trace, trace.input_layers),
            # Boundary rows own nothing additive: every additive cell is "-"
            # (identity partition, A1), never a number.
            "params": "-",
            "train": "-",
        }
    ]
    origin_by_label, input_labels = _module_dataflow_origins(trace)
    for module in _iter_summary_modules(trace):
        rows.append(_module_overview_row(trace, module, origin_by_label, input_labels))
    rows.append(
        {
            "name": "output",
            "shape": _combined_shape_str(trace, trace.output_layers),
            "params": "-",
            "train": "-",
        }
    )
    footer_lines = [
        *_param_footer_lines(trace),
        f"Ops: {trace.num_ops} total",
        f"Edges: {trace.num_edges} total",
        f"Branching factor: {trace.branching_factor:.2f}",
        _saved_outs_footer_line(trace),
        *_compute_footer_lines(trace, count_fma_as_two),
        *_health_footer_lines(trace),
    ]
    return rows, footer_lines


def _health_footer_lines(trace: Trace) -> list[str]:
    """One conditional health line from the three states (sumfam D5/F3).

    Silent when CHECKED-AND-CLEAN so clean goldens stay byte-stable;
    FOUND and NOT-CHECKED render with the basis and the follow-up
    spelling. Never scans payloads beyond the memoized health basis.
    """

    from ...report._health import health_facts

    facts = health_facts(trace)
    if facts.verdict == "checked_and_clean":
        return []
    if facts.verdict == "found":
        total = facts.nonfinite_count + len(facts.alias_nonfinite_labels)
        return [
            f"Health: NaN/Inf FOUND in {total} op output(s) "
            f"(basis: {facts.source_basis or facts.basis}; see trace.health_facts)"
        ]
    return [
        f"Health: NOT-CHECKED ({facts.unexamined} op output(s) unexamined; see trace.health_facts)"
    ]


def _saved_outs_footer_line(trace: Trace) -> str:
    """The saved-outs footer under the payload-scope law (C02; sumfam D8).

    The footer prints retained_now (what THIS object holds) and names
    at_capture only when the two differ -- a payload-stripped artifact
    stops claiming bytes it does not contain, and clean live goldens stay
    byte-stable.
    """

    from ...report._factcore import _memory

    memory = _memory(trace)
    if memory.retained_now_bytes == memory.at_capture_bytes:
        return f"Saved outs: {human_readable_size(memory.retained_now_bytes)}"
    return (
        f"Saved outs: {human_readable_size(memory.retained_now_bytes)} retained now "
        f"({human_readable_size(memory.at_capture_bytes)} at capture)"
    )


def _compute_footer_lines(trace: Trace, count_fma_as_two: bool | None) -> list[str]:
    """Return the compute footer block (A5/A6) from the canonical aggregation.

    True MACs come from each op's two-term compute record, never flops//2;
    the active FMA convention is always named; an explicit fma=1 request on a
    trace with underivable MAC splits refuses typed
    (``flop_convention_unavailable``) inside the aggregation -- never
    accepted-and-ignored.
    """

    from ...report._compute_truth import aggregate_forward_compute, forward_flops_total

    totals = aggregate_forward_compute(trace)
    macs_part = f"MACs: {_human_macs(int(totals.macs))}"
    if totals.macs_unknown_split:
        count = len(totals.macs_unknown_split)
        macs_part += f" (lower bound; MAC split unknown for {count} op{'s' if count > 1 else ''})"
    if count_fma_as_two is False:
        fma1_total = forward_flops_total(trace, fma=1)
        flops_line = f"Forward FLOPs (fma=1): {_human_flops(int(fma1_total))}  {macs_part}"
        convention_line = (
            "FLOP convention: fma=1 (explicit; one multiply-accumulate = 1 FLOP); "
            "MACs are true multiply-accumulate counts."
        )
    else:
        marker = " (explicit)" if count_fma_as_two is True else ""
        flops_line = f"Forward FLOPs: {_human_flops(int(totals.flops_fma2))}  {macs_part}"
        convention_line = (
            f"FLOP convention: fma=2{marker} (one multiply-accumulate = 2 FLOPs); "
            "MACs are true multiply-accumulate counts."
        )
    return [flops_line, _unknown_flops_footer(trace), convention_line]


def _param_footer_lines(trace: Trace) -> list[str]:
    """Return the parameter footer block (A2/A3/A12).

    Headline = declared parameters under torch's own identity rule (tie-
    deduplicated). When ties exist the per-module-path total is printed
    BESIDE it with the tie named -- so a torchinfo switcher sees why the two
    tools disagree instead of filing a bug. Declared-but-never-executed
    parameters are named, never silently dropped.
    """

    total = int(trace.num_params)
    trainable = int(trace.num_params_trainable)
    frozen = int(trace.num_params_frozen)
    if total > 0:
        pct = 100.0 * trainable / total
        headline = (
            f"Params: {_int_with_commas(total)} unique (parameter identity); "
            f"trainable: {_int_with_commas(trainable)} ({pct:.1f}%); "
            f"frozen: {_int_with_commas(frozen)}"
        )
    else:
        headline = "Params: 0"
    lines = [headline]

    tied_groups = trace.tied_param_groups
    if tied_groups:
        by_path = trace.num_params_by_path
        tie_names = "; ".join(" = ".join(group) for group in tied_groups[:3])
        extra = "" if len(tied_groups) <= 3 else f" (+{len(tied_groups) - 3} more ties)"
        lines.append(
            f"Shared params: {tie_names}{extra} -- counted once; "
            f"per-module-path total: {_int_with_commas(by_path)}"
        )

    unexecuted = trace.num_params_unexecuted
    if unexecuted:
        names = trace.unexecuted_param_names
        shown = ", ".join(names[:3])
        extra = "" if len(names) <= 3 else f" (+{len(names) - 3} more)"
        lines.append(f"Never executed: {_int_with_commas(unexecuted)} params ({shown}{extra})")
    return lines


def _build_graph_rows(trace: Trace) -> tuple[list[dict[str, str]], list[str]]:
    """Build graph-summary rows.

    Parameters
    ----------
    trace:
        Finalized log object.

    Returns
    -------
    tuple[list[dict[str, str]], list[str]]
        Graph rows and footer lines.
    """
    rows = []
    origin_by_label, input_labels = _module_dataflow_origins(trace)
    for module in _iter_summary_modules(trace):
        rows.append(
            {
                "name": f"{module.address} ({module.class_name})",
                "shape": _module_shape(trace, module),
                "params": _human_count(module.num_params),
                "parents": _module_parent_summary(module, origin_by_label, input_labels),
            }
        )
    footer_lines = [
        f"Modules shown: {len(rows)}",
        f"Ops tracked: {trace.num_ops}",
        f"Edges tracked: {trace.num_edges}",
        f"Branching factor: {trace.branching_factor:.2f}",
    ]
    return rows, footer_lines


def _build_memory_rows(
    trace: Trace,
    *,
    mode: SummaryMode,
) -> tuple[list[dict[str, str]], list[str]]:
    """Build memory-summary rows.

    Parameters
    ----------
    trace:
        Finalized log object.
    mode:
        Operation aggregation mode.

    Returns
    -------
    tuple[list[dict[str, str]], list[str]]
        Memory rows and footer lines.
    """
    running_total = 0
    rows: list[dict[str, str]] = []
    for entry in _iter_operation_entries(trace, mode=mode):
        # Output alias rows display their shape but OWN no bytes (identity
        # partition, A1): the producing op owns the returned tensor.
        alias = _is_output_alias_row(entry)
        memory = int(getattr(entry, "activation_memory", 0) or 0)
        if not alias:
            running_total += memory
        rows.append(
            {
                "name": _entry_name(entry),
                "shape": _shape_str(getattr(entry, "shape", None)),
                "dtype": _dtype_str(getattr(entry, "dtype", None)),
                "tensor_mb": "-" if alias else _mb_str(memory),
                "running_mb": "-" if alias else _mb_str(running_total),
            }
        )
    footer_lines = [
        f"Tracked tensor volume: {human_readable_size(trace.total_activation_memory)}",
        f"Saved outs: {human_readable_size(trace.saved_activation_memory)}",
        _forward_peak_memory_line(trace),
    ]
    return rows, footer_lines


def _forward_peak_memory_line(trace: Trace) -> str:
    """Return the honest forward-peak-memory footer line.

    Routed through the ONE pass-level-peak publication gate (observe item 4):
    the basis is always named, cpu/mps render under their real meaning
    (process RSS growth / allocator delta, never a bare "peak"), and a CUDA
    ``0`` renders as a high-water fact instead of "used 0 bytes".
    """

    from torchlens.observe._peaks import format_pass_peak

    return format_pass_peak(trace, "forward")


def _build_control_flow_rows(trace: Trace) -> tuple[list[dict[str, str]], list[str]]:
    """Build control-flow rows or an empty state.

    Parameters
    ----------
    trace:
        Finalized log object.

    Returns
    -------
    tuple[list[dict[str, str]], list[str]]
        Control-flow rows and footer lines.
    """
    rows: list[dict[str, str]] = []
    for event in trace.conditional_records:
        branch_kinds = _event_branch_kinds(trace, event)
        rows.append(
            {
                "site": f"cond#{event.id}",
                "source": _event_source(event),
                "taken": ",".join(branch_kinds) if branch_kinds else "unknown",
                "bool_layer": _event_bool_layer(event),
                "branch_ops": str(_event_branch_op_count(trace, event)),
                "notes": event.function_qualname,
            }
        )
    loop_groups = _recurrent_loop_groups(trace)
    recurrent = bool(loop_groups) or bool(getattr(trace, "is_recurrent", False))
    if not rows:
        if not recurrent:
            return rows, [
                "No conditional branches or recurrent loop groups were detected "
                "in this forward pass."
            ]
        footer_lines = ["No conditional branches were detected in this forward pass."]
    else:
        footer_lines = [f"Conditionals: {len(rows)}"]
    footer_lines.extend(_recurrent_loop_group_lines(loop_groups, recurrent))
    return rows, footer_lines


def _recurrent_loop_groups(trace: Trace) -> list[tuple[str, int]]:
    """Return ``(layer_label, num_passes)`` for every recurrent (multi-pass) layer.

    These are the loop groups the control-flow summary claims to detect. Driven
    off the concrete per-layer pass counts rather than only ``trace.is_recurrent``
    so the disclosure names the exact layers that replay.
    """
    groups: list[tuple[str, int]] = []
    for layer in trace.layer_logs.values():
        num_passes = int(getattr(layer, "num_passes", 1) or 1)
        if num_passes > 1:
            groups.append((str(layer.layer_label), num_passes))
    return groups


def _recurrent_loop_group_lines(
    loop_groups: list[tuple[str, int]],
    recurrent: bool,
) -> list[str]:
    """Return honest footer disclosure lines for recurrent loop groups."""
    if not recurrent:
        return []
    if not loop_groups:
        return ["Recurrent execution detected (rolled layers replay across passes)."]
    detail = ", ".join(f"{label} (x{passes})" for label, passes in loop_groups)
    return [f"Recurrent loop groups ({len(loop_groups)}): {detail}"]


def _build_compute_rows(
    trace: Trace, count_fma_as_two: bool | None = None
) -> tuple[list[dict[str, str]], list[str]]:
    """Build compute-summary rows.

    Parameters
    ----------
    trace:
        Finalized log object.

    Returns
    -------
    tuple[list[dict[str, str]], list[str]]
        Compute rows and footer lines.
    """
    rows = []
    for module in _iter_summary_modules(trace):
        rows.append(
            {
                "name": module.address,
                "params": _human_count(module.num_params),
                "flops": _human_flops(module.total_flops_forward),
                "macs": _human_macs(int(module.total_macs_forward)),
                "time_ms": f"{_module_time_ms(trace, module):.2f}",
                "dtype": _module_dtype(trace, module),
            }
        )
    accumulated_ms = (
        sum(
            _entry_func_duration(entry) for entry in _iter_operation_entries(trace, mode="unrolled")
        )
        * 1000.0
    )
    wall_ms = float(getattr(trace, "forward_duration", 0.0) or 0.0) * 1000.0
    footer_lines = [
        *_param_footer_lines(trace),
        *_compute_footer_lines(trace, count_fma_as_two),
        # Report the compute-relevant accumulated op time (matching the waterfall
        # level) as the headline number, and disclose the raw capture wall time
        # separately as overhead-inclusive. Previously a single "Forward time"
        # line reported trace.forward_duration -- capture wall time that INCLUDES
        # all TorchLens instrumentation overhead (~100x+ the real op time) -- and
        # sitting next to FLOPs/MACs it read as the model's forward compute cost.
        f"Accumulated op time: {accumulated_ms:.2f} ms",
        f"Capture wall time (includes TorchLens overhead): {wall_ms:.2f} ms",
    ]
    return rows, footer_lines


def _build_output_rows(trace: Trace) -> tuple[list[dict[str, str]], list[str]]:
    """Build decoded output rows for the output summary level.

    Parameters
    ----------
    trace:
        Finalized log object.

    Returns
    -------
    tuple[list[dict[str, str]], list[str]]
        Output table rows and footer lines.
    """

    try:
        table = trace.output_table()
    except ValueError as exc:
        return [], [str(exc)]
    rows = [
        {
            "batch_item": str(row["batch_item"]),
            "rank": str(row["rank"]),
            "label": str(row["label"]),
            "prob": f"{float(row['prob']):.1%}",
        }
        for row in table.to_dict(orient="records")
    ]
    return rows, [f"Decoded output rows: {len(rows)}"]


def _build_waterfall_rows(
    trace: Trace,
    *,
    mode: SummaryMode,
) -> tuple[list[dict[str, str]], list[str]]:
    """Build timing and memory waterfall rows.

    Parameters
    ----------
    trace:
        Finalized log object.
    mode:
        Operation aggregation mode.

    Returns
    -------
    tuple[list[dict[str, str]], list[str]]
        Waterfall rows and footer lines.
    """

    elapsed = 0.0
    peak_memory = 0
    rows: list[dict[str, str]] = []
    for entry in _iter_operation_entries(trace, mode=mode):
        duration = _entry_func_duration(entry)
        memory = int(getattr(entry, "activation_memory", 0) or 0)
        peak_memory = max(peak_memory, memory)
        rows.append(
            {
                "name": _entry_name(entry),
                "start_ms": f"{elapsed * 1000:.2f}",
                "time_ms": f"{duration * 1000:.2f}",
                "end_ms": f"{(elapsed + duration) * 1000:.2f}",
                "memory": human_readable_size(memory),
            }
        )
        elapsed += duration
    footer_lines = [
        f"Accumulated op time: {elapsed * 1000:.2f} ms",
        f"Max single tensor memory: {human_readable_size(peak_memory)}",
    ]
    return rows, footer_lines


def _build_operation_rows(
    *,
    trace: Trace,
    mode: SummaryMode,
    level: str,
) -> tuple[list[dict[str, str]], list[str]]:
    """Build operation rows for the optional op dump.

    Parameters
    ----------
    trace:
        Finalized log object.
    mode:
        Operation aggregation mode.
    level:
        Active summary level, used to pick the most relevant row shape.

    Returns
    -------
    tuple[list[dict[str, str]], list[str]]
        Operation rows and footer lines.
    """
    rows: list[dict[str, str]] = []
    running_total = 0
    for entry in _iter_operation_entries(trace, mode=mode):
        alias = _is_output_alias_row(entry)
        memory = int(getattr(entry, "activation_memory", 0) or 0)
        if not alias:
            running_total += memory
        rows.append(
            {
                "name": _entry_name(entry),
                "shape": _shape_str(getattr(entry, "shape", None)),
                "params": "-"
                if _is_boundary_row(entry)
                else _human_count(int(getattr(entry, "num_params", 0) or 0)),
                "parents": _parent_summary(getattr(entry, "parents", [])),
                "dtype": _dtype_str(getattr(entry, "dtype", None)),
                "tensor_mb": "-" if alias else _mb_str(memory),
                "running_mb": "-" if alias else _mb_str(running_total),
                "flops": _flops_cell(entry),
                "macs": _macs_cell(entry),
                "time_ms": "-"
                if _is_boundary_row(entry)
                else (f"{_entry_func_duration(entry) * 1000:.2f}"),
            }
        )
    footer_lines = [
        f"Operation rows shown: {len(rows)} ({mode if mode != 'auto' else _effective_mode(trace, mode)})"
    ]
    return rows, footer_lines


def _is_boundary_row(entry: Any) -> bool:
    """Return whether ``entry`` is an input/output boundary pseudo-row.

    Boundary rows render but OWN nothing additive (identity partition, A1):
    every additive cell renders "-", never a number.
    """

    return bool(getattr(entry, "is_input", False)) or bool(getattr(entry, "is_output", False))


def _is_output_alias_row(entry: Any) -> bool:
    """Return whether ``entry`` is an output alias row (owns no bytes)."""

    return bool(getattr(entry, "is_output", False))


def _flops_cell(entry: Any) -> str:
    """Typed FLOPs cell: "-" not applicable, "?" unknown, else the value."""

    if _is_boundary_row(entry):
        return "-"
    flops = getattr(entry, "flops_forward", None)
    if flops is None:
        return "?"
    return _human_flops(int(flops))


def _macs_cell(entry: Any) -> str:
    """Typed MACs cell: "-" not applicable, "?" unknown, else the value."""

    if _is_boundary_row(entry):
        return "-"
    macs = getattr(entry, "macs_forward", None)
    if macs is None:
        return "?"
    return _human_macs(int(macs))


def _default_op_fields(level: str) -> list[str]:
    """Return the default operation columns for the active level.

    Parameters
    ----------
    level:
        Canonical level name.

    Returns
    -------
    list[str]
        Default operation fields.
    """
    if level == "memory":
        return ["name", "shape", "dtype", "tensor_mb", "running_mb"]
    if level == "compute":
        return ["name", "params", "flops", "macs", "time_ms", "dtype"]
    return ["name", "shape", "params", "parents"]


def _iter_summary_modules(trace: Trace) -> list[Module]:
    """Return top-level module rows for summary tables.

    Parameters
    ----------
    trace:
        Finalized log object.

    Returns
    -------
    list[Module]
        Top-level modules in accessor order.
    """
    modules = []
    for module in trace.modules:
        if module.address == "self":
            continue
        if module.address_depth == 1:
            modules.append(module)
    return modules


def _module_overview_row(
    trace: Trace,
    module: Module,
    origin_by_label: dict[str, str],
    input_labels: set[str],
) -> dict[str, str]:
    """Build one overview row for a module.

    Parameters
    ----------
    trace:
        Finalized log object.
    module:
        Module to summarize.
    origin_by_label:
        Reverse index mapping op labels to owning top-level module addresses.
    input_labels:
        Set of graph-input op labels.

    Returns
    -------
    dict[str, str]
        Renderable overview row.
    """
    # Trainability is a tri-state (A12): "yes" only when EVERY parameter
    # element is trainable, "partial" for a mixed module, "no" for none --
    # the old boolean OR read "yes" with a single unfrozen tensor.
    train = "-"
    if module.num_params > 0:
        if module.num_params_trainable == module.num_params:
            train = "yes"
        elif module.num_params_trainable == 0:
            train = "no"
        else:
            train = "partial"
    return {
        "name": f"{module.address} ({module.class_name})",
        "shape": _module_shape(trace, module),
        "params": _human_count(module.num_params),
        "train": train,
        "parents": _module_parent_summary(module, origin_by_label, input_labels),
        "class": module.class_name,
    }


def _module_shape(trace: Trace, module: Module) -> str:
    """Return a representative output shape for a module.

    Parameters
    ----------
    trace:
        Finalized log object.
    module:
        Module to summarize.

    Returns
    -------
    str
        Representative output shape.
    """
    layer = _module_output_layer(trace, module)
    if layer is None:
        return "-"
    return _shape_str(getattr(layer, "shape", None))


def _strip_pass_suffix(label: str) -> str:
    """Return the aggregate layer label for a possibly pass-qualified op label.

    ``relu_1_1:2`` -> ``relu_1_1``; a label without a pass suffix is returned
    unchanged.
    """
    return str(label).split(":", 1)[0]


def _module_dataflow_origins(trace: Trace) -> tuple[dict[str, str], set[str]]:
    """Build a reverse index for module-level dataflow connectivity.

    Returns ``(origin_by_label, input_labels)`` where ``origin_by_label`` maps
    every (aggregate) op label to the address of the top-level summary module
    that owns it, and ``input_labels`` is the set of graph-input op labels. Both
    are keyed by pass-stripped labels so pass-qualified producers resolve.
    """
    origin_by_label: dict[str, str] = {}
    for module in _iter_summary_modules(trace):
        for label in module.layer_labels:
            origin_by_label[_strip_pass_suffix(label)] = module.address
    input_labels = {_strip_pass_suffix(op.label) for op in trace.input_ops}
    return origin_by_label, input_labels


def _module_parent_summary(
    module: Module,
    origin_by_label: dict[str, str],
    input_labels: set[str],
) -> str:
    """Return the REAL upstream dataflow producers feeding a module.

    The graph/overview "Connected To" column is a dataflow claim. Previously it
    returned ``module.address_parent`` -- the containment-tree parent -- and
    hard-coded ``"input"`` for every top-level module, so a chain ``a -> b``
    falsely reported both ``a`` and ``b`` as connected to ``input``. This
    fabricated topology from the wrong graph entirely. Now each of the module's
    recorded input ops is mapped to its producing top-level module (or ``input``
    for a graph-input producer, or the bare op label when the producer is not
    inside any summary module). Producers are de-duplicated in first-seen order;
    a module with no recorded upstream reports ``-`` rather than inventing one.
    """
    input_ops = getattr(module, "input_ops", None)
    if not input_ops:
        return "-"
    upstream: list[str] = []
    for op_label in input_ops:
        normalized = _strip_pass_suffix(op_label)
        if normalized in input_labels:
            origin = "input"
        else:
            origin = origin_by_label.get(normalized, normalized)
        if origin not in upstream:
            upstream.append(origin)
    return ", ".join(upstream) if upstream else "-"


def _module_dtype(trace: Trace, module: Module) -> str:
    """Return a representative dtype for a module.

    Parameters
    ----------
    trace:
        Finalized log object.
    module:
        Module to summarize.

    Returns
    -------
    str
        Representative dtype.
    """
    layer = _module_output_layer(trace, module)
    if layer is None:
        return "-"
    return _dtype_str(getattr(layer, "dtype", None))


def _module_output_layer(trace: Trace, module: Module) -> Any | None:
    """Return the representative output layer for a module.

    Parameters
    ----------
    trace:
        Finalized log object.
    module:
        Module to summarize.

    Returns
    -------
    Any or None
        Last explicit module-call output, with the historical layer-list tail
        used only when module-call output metadata is unavailable.
    """

    try:
        module_call = trace.module_calls[f"{module.address}:{module.num_calls}"]
        if module_call.output_ops:
            return trace[module_call.output_ops[-1]]
    except (KeyError, IndexError):
        pass
    if not module.layer_labels:
        return None
    try:
        return trace[module.layer_labels[-1]]
    except KeyError:
        return None


def _module_time_ms(trace: Trace, module: Module) -> float:
    """Return the summed forward time for a module.

    Parameters
    ----------
    trace:
        Finalized log object.
    module:
        Module to summarize.

    Returns
    -------
    float
        Summed execution time in milliseconds.
    """
    total = 0.0
    for layer_label in module.layer_labels:
        try:
            layer = trace[layer_label]
        except KeyError:
            continue
        duration = getattr(layer, "total_func_duration", None)
        if duration is None:
            duration = getattr(layer, "func_duration", 0.0)
        total += float(duration or 0.0)
    return total * 1000.0


def _entry_func_duration(entry: Any) -> float:
    """Return an entry's forward duration in seconds without tripping the tripwire.

    In ``rolled`` mode ``_iter_operation_entries`` yields aggregate ``Layer``
    objects; a recurrent (multi-pass) ``Layer`` deliberately RAISES ``ValueError``
    on the per-pass ``func_duration`` accessor (the locked multi-pass tripwire) and
    exposes the documented aggregate ``total_func_duration`` (sum over passes)
    instead. In ``unrolled`` mode the entries are per-pass ``Op`` objects, which
    expose their own ``func_duration`` and do not define ``total_func_duration``.
    Prefer the aggregate accessor when present, else the per-pass value; this is
    the same safe idiom already used by ``_module_time_ms`` and never lets the
    tripwire ``ValueError`` leak nor silently substitutes a wrong default.
    """
    duration = getattr(entry, "total_func_duration", None)
    if duration is None:
        duration = getattr(entry, "func_duration", 0.0)
    return float(duration or 0.0)


def _iter_operation_entries(
    trace: Trace,
    *,
    mode: SummaryMode,
) -> Iterable[Layer | Op]:
    """Iterate operation-like entries according to the requested mode.

    Parameters
    ----------
    trace:
        Finalized log object.
    mode:
        Requested aggregation mode.

    Returns
    -------
    Iterable[Layer | Op]
        Operation entries in display order.
    """
    effective_mode = _effective_mode(trace, mode)
    if effective_mode == "rolled":
        return cast(Iterable["Layer | Op"], trace.layer_logs.values())
    return cast(Iterable["Layer | Op"], trace.layer_list)


def _effective_mode(trace: Trace, mode: SummaryMode) -> Literal["rolled", "unrolled"]:
    """Resolve the effective operation mode.

    Parameters
    ----------
    trace:
        Finalized log object.
    mode:
        Requested mode.

    Returns
    -------
    Literal["rolled", "unrolled"]
        Effective operation mode.
    """
    if mode == "auto":
        return "unrolled" if trace.is_recurrent else "rolled"
    return mode


def _entry_name(entry: Any) -> str:
    """Return a display name for a layer or layer-pass entry.

    Parameters
    ----------
    entry:
        Layer-like object.

    Returns
    -------
    str
        Display name.
    """
    base_name = getattr(entry, "layer_label", None) or getattr(entry, "layer_label_short", None)
    if base_name is None:
        base_name = getattr(entry, "label", None) or getattr(entry, "layer_label", "?")
    num_passes = int(getattr(entry, "num_passes", 1) or 1)
    if num_passes > 1 and _is_pass_op(entry):
        # Unrolled tables emit one row PER PASS; each such row is a per-pass Op,
        # not the aggregate Layer. Name it with its pass-qualified identity
        # (relu_1_1:2) rather than the aggregate "xN" multiplicity, which on a
        # per-pass row would imply N calls per row (a false 9-call reading for a
        # 3-pass layer). An Op exposes a safe pass-qualified label; only the
        # aggregate Layer would raise the multi-pass tripwire here.
        pass_label = getattr(entry, "label", None)
        if isinstance(pass_label, str):
            return pass_label
    if num_passes > 1 and hasattr(entry, "ops"):
        # One pass/op vocabulary (A10): the aggregate rolled row spells its
        # multiplicity as passes, so a 3-pass layer can never read as 3 ops.
        return f"{base_name} (x{num_passes} passes)"
    if getattr(entry, "call_index", 1) > 1:
        return str(getattr(entry, "layer_label", base_name))
    return str(base_name)


def _is_pass_op(entry: Any) -> bool:
    """Return True if ``entry`` is a per-pass ``Op`` (vs an aggregate ``Layer``).

    Unrolled summaries iterate per-pass ``Op`` objects while rolled summaries
    iterate aggregate ``Layer`` objects, but an ``Op`` proxies its parent's
    ``num_passes``/``ops`` so those attributes cannot tell them apart. Use the
    concrete type as the discriminator (imported lazily to avoid any import
    cycle at module load).
    """
    from ...data_classes.op import Op

    return isinstance(entry, Op)


def _combined_shape_str(trace: Trace, labels: Sequence[str]) -> str:
    """Return a compact combined shape string for one or more labels.

    Parameters
    ----------
    trace:
        Finalized log object.
    labels:
        Layer labels whose shapes should be summarized.

    Returns
    -------
    str
        Shape summary string.
    """
    if not labels:
        return "-"
    shapes = []
    for label in labels:
        try:
            shapes.append(_shape_str(getattr(trace[label], "shape", None)))
        except KeyError:
            continue
    if not shapes:
        return "-"
    if len(shapes) == 1:
        return shapes[0]
    return f"{len(shapes)} tensors"


def _event_branch_kinds(trace: Trace, event: ConditionalEvent) -> list[str]:
    """Return the taken branch kinds for one conditional event.

    Parameters
    ----------
    trace:
        Finalized log object.
    event:
        Conditional event to inspect.

    Returns
    -------
    list[str]
        Branch kinds observed for the event.
    """
    branch_kinds = {
        branch_kind
        for (cond_id, branch_kind) in trace.conditional_arm_entry_edges
        if cond_id == event.id
    }
    return sorted(branch_kinds)


def _event_source(event: ConditionalEvent) -> str:
    """Return a short source locator for a conditional event.

    Parameters
    ----------
    event:
        Conditional event to summarize.

    Returns
    -------
    str
        Source locator.
    """
    # Returned summary strings never carry escape bytes (OSC-8 included):
    # they must be safe to log, diff, snapshot, and paste into issues on any
    # sink. Hyperlinks are an HTML-renderer concern, never a str() concern.
    return file_line_text(event.source_file, event.if_stmt_span[0])


def _event_bool_layer(event: ConditionalEvent) -> str:
    """Return a compact bool-layer summary for a conditional event.

    Parameters
    ----------
    event:
        Conditional event to summarize.

    Returns
    -------
    str
        Bool-layer summary.
    """
    if not event.bool_layers:
        return "-"
    if len(event.bool_layers) == 1:
        return str(event.bool_layers[0])
    return f"{event.bool_layers[0]} +{len(event.bool_layers) - 1}"


def _event_branch_op_count(trace: Trace, event: ConditionalEvent) -> int:
    """Return the number of operation edges attributed to a conditional event.

    Parameters
    ----------
    trace:
        Finalized log object.
    event:
        Conditional event to inspect.

    Returns
    -------
    int
        Number of attributed branch edges.
    """
    return sum(
        len(edges)
        for (cond_id, _branch_kind), edges in trace.conditional_arm_entry_edges.items()
        if cond_id == event.id
    )


def _parent_summary(parents: Sequence[str]) -> str:
    """Return a compact parent-layer summary.

    Parameters
    ----------
    parents:
        Parent layer labels.

    Returns
    -------
    str
        Compact parent summary.
    """
    if not parents:
        return "-"
    if len(parents) == 1:
        return str(parents[0])
    return f"{parents[0]} +{len(parents) - 1}"


def _shape_str(shape: Any) -> str:
    """Format a tensor shape using ASCII-only list syntax.

    Parameters
    ----------
    shape:
        Shape-like object.

    Returns
    -------
    str
        ASCII shape string.
    """
    if shape is None:
        return "-"
    return str(list(shape)).replace(" ", "")


def _dtype_str(dtype: Any) -> str:
    """Format a dtype name.

    Parameters
    ----------
    dtype:
        Dtype-like object.

    Returns
    -------
    str
        Short dtype string.
    """
    if dtype is None:
        return "-"
    text = str(dtype)
    return text.replace("torch.", "")


def _mb_str(num_bytes: int) -> str:
    """Format a byte count in megabytes.

    Parameters
    ----------
    num_bytes:
        Number of bytes.

    Returns
    -------
    str
        Megabyte string.
    """
    return f"{num_bytes / (1024.0 * 1024.0):.2f}"


def _human_count(value: int) -> str:
    """Format an integer count compactly.

    Parameters
    ----------
    value:
        Integer count.

    Returns
    -------
    str
        Compact count string.
    """
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f} B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f} M"
    if value >= 1_000:
        return f"{value / 1_000:.1f} K"
    return str(value)


def _unknown_flops_footer(trace: Trace) -> str:
    """Return the summary disclosure for operations with unknown FLOPs.

    Parameters
    ----------
    trace
        Finalized trace whose compute operations are summarized.

    Returns
    -------
    str
        Unknown-operation count and, when nonzero, the total-exclusion warning.
    """

    from ...report._compute_truth import unknown_op_ledger

    ledger = unknown_op_ledger(trace)
    if not ledger:
        return "Unknown-FLOPs ops: 0"
    count = sum(group.count for group in ledger)
    named = ", ".join(f"{group.func_name} x{group.count}" for group in ledger[:4])
    extra = "" if len(ledger) <= 4 else f", +{len(ledger) - 4} more names"
    return (
        f"Unknown-FLOPs ops: {count} ({named}{extra}; excluded from FLOP/MAC "
        "totals; remedy: torchlens.capture.flops.register_op_rule)"
    )


def _human_flops(value: int) -> str:
    """Format a FLOP count compactly.

    Parameters
    ----------
    value:
        FLOP integer.

    Returns
    -------
    str
        Compact FLOP string.
    """
    return format_flops(value)


def _human_macs(value: int) -> str:
    """Format a MAC count compactly, in MAC units.

    A MACs value must NEVER route through the FLOPs formatter (listA row 13:
    "718.9 MFLOPs" printed for a MACs quantity) -- formatting dispatches on
    the semantic type.
    """

    return str(Macs(value))


def _int_with_commas(value: int) -> str:
    """Format an integer with comma separators.

    Parameters
    ----------
    value:
        Integer value.

    Returns
    -------
    str
        Comma-separated string.
    """
    return f"{value:,}"


def _render_table(
    fields: Sequence[str],
    rows: Sequence[dict[str, str]],
    *,
    max_rows: int | None,
    label_overrides: dict[str, str] | None = None,
) -> str:
    """Render an ASCII table.

    Parameters
    ----------
    fields:
        Ordered field names to render.
    rows:
        Row dictionaries.
    max_rows:
        Maximum number of rows to render.
    label_overrides:
        Per-field header overrides so a heading can follow the row kind
        (module rows vs op rows) instead of the one-size-fits-all "Layer".

    Returns
    -------
    str
        ASCII table string.
    """
    if not rows:
        return "(no rows)"

    labels = dict(_COLUMN_LABELS)
    if label_overrides:
        labels.update(label_overrides)

    display_rows = list(rows)
    truncated_count = 0
    if max_rows is not None and len(display_rows) > max_rows:
        truncated_count = len(display_rows) - max_rows
        display_rows = display_rows[:max_rows]

    widths = []
    for field in fields:
        header = labels[field]
        cell_width = max(len(str(row.get(field, "-"))) for row in display_rows)
        widths.append(max(len(header), cell_width))

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header_row = (
        "| " + " | ".join(labels[field].ljust(width) for field, width in zip(fields, widths)) + " |"
    )
    body_rows = [
        "| "
        + " | ".join(str(row.get(field, "-")).ljust(width) for field, width in zip(fields, widths))
        + " |"
        for row in display_rows
    ]
    lines = [border, header_row, border, *body_rows, border]
    if truncated_count:
        lines.append(
            f"... showing first {len(display_rows)} of {len(rows)} rows; "
            "try a narrower level or show_ops=False"
        )
    return "\n".join(lines)
