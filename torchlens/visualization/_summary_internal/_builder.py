"""Build compact text summaries for ``Trace`` objects."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    cast,
)

from ...utils.display import format_flops, human_readable_size
from ..._source_links import terminal_file_line_link

if TYPE_CHECKING:
    from ..data_classes.layer import Layer
    from ..data_classes.op import Op
    from ..data_classes.trace import ConditionalEvent, Trace
    from ..data_classes.module import Module


SummaryLevel = Literal[
    "overview", "graph", "memory", "control_flow", "compute", "cost", "waterfall"
]
SummaryMode = Literal["auto", "rolled", "unrolled"]

_LEVEL_ALIASES: Dict[str, str] = {"cost": "compute"}

_COLUMN_LABELS: Dict[str, str] = {
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
}

_LEVEL_DEFAULT_FIELDS: Dict[str, List[str]] = {
    "overview": ["name", "shape", "params", "train"],
    "graph": ["name", "shape", "params", "parents"],
    "memory": ["name", "shape", "dtype", "tensor_mb", "running_mb"],
    "control_flow": ["site", "source", "taken", "bool_layer", "branch_ops", "notes"],
    "compute": ["name", "params", "flops", "macs", "time_ms", "dtype"],
    "waterfall": ["name", "start_ms", "time_ms", "end_ms", "memory"],
}


def render_model_summary(
    trace: "Trace",
    *,
    level: SummaryLevel = "overview",
    preset: SummaryLevel | None = None,
    fields: Optional[List[str]] = None,
    columns: Optional[List[str]] = None,
    mode: SummaryMode = "auto",
    show_ops: bool = False,
    include_ops: Optional[bool] = None,
    max_rows: Optional[int] = 200,
    print_to: Optional[Callable[[str], None]] = None,
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
        )
    text = f"{format_discoverability_summary(trace)}\n\n{legacy_text}"
    if print_to is not None:
        print_to(text)
    return text


def format_model_repr(trace: "Trace") -> str:
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
    if not trace._tracing_finished:
        return (
            f"Trace(name={getattr(trace, 'trace_label', None)!r}, "
            f"model_class_qualname={trace.model_class_name!r}, layers={_live_op_count(trace)}, "
            f"state={state})"
        )

    return (
        f"Trace(name={getattr(trace, 'trace_label', None)!r}, "
        f"model_class_qualname={trace.model_class_name!r}, layers={len(trace.layer_logs)}, "
        f"state={state})"
    )


def _live_op_count(trace: "Trace") -> int:
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
    return len(trace._raw_layer_dict)


def format_discoverability_summary(trace: "Trace") -> str:
    """Render the Phase 13 user-facing discoverability summary.

    Parameters
    ----------
    trace:
        Model log to summarize.

    Returns
    -------
    str
        Multi-section notebook-friendly summary.
    """

    spec = getattr(trace, "_intervention_spec", None)
    target_specs = tuple(getattr(spec, "target_value_specs", ()) or ())
    hook_specs = tuple(getattr(spec, "hook_specs", ()) or ())
    lines = [
        "TorchLens Discoverability Summary",
        "Capture:",
        f"  name: {getattr(trace, 'trace_label', None)!r}",
        f"  model_class_qualname: {getattr(trace, 'model_class_name', None)}",
        f"  input_shape: {_input_shape_summary(trace)}",
        *_input_preprocessing_lines(trace),
        f"  capture_timestamp: {_capture_timestamp(trace)}",
        f"  intervention_ready: {bool(getattr(trace, 'intervention_ready', False))}",
        f"  save_arg_templates: {bool(getattr(trace, 'save_arg_templates', False))}",
        "Run state:",
        f"  state: {_run_state_name(trace)}",
        f"  direct_write_dirty: {bool(getattr(trace, '_has_direct_writes', False))}",
        f"  append: is_appended={bool(getattr(trace, 'is_appended', False))}, "
        f"sequence_id={getattr(trace, '_append_sequence_id', 0)}",
        f"  stale_spec: {_stale_spec_status(trace)}",
        f"  last_run: {_last_run_summary(trace)}",
        "Active recipe:",
        f"  target_value_specs: {len(target_specs)}{_spec_sample(target_specs)}",
        f"  hook_specs: {len(hook_specs)}{_spec_sample(hook_specs)}",
        f"  portability: {_portability_status(target_specs, hook_specs)}",
        "Recent operations:",
        *_recent_operation_lines(trace),
        "Lineage:",
        f"  parent_run: {_parent_run_summary(trace)}",
        f"  fork_chain: {_fork_chain_summary(trace)}",
        "Graph and relationship evidence:",
        f"  graph_shape_hash: {_truncated(getattr(trace, 'graph_shape_hash', None))}",
        f"  model_class_qualname: {getattr(trace, 'model_class_qualname', None)}",
        f"  weight_fingerprint: {_truncated(getattr(trace, 'param_hash_quick', None))}",
        f"  relationship_evidence: {_relationship_evidence_summary(trace)}",
        "Next operations:",
        f"  {_next_operation_hint(trace)}",
        "RNG and helper notes:",
        f"  {_rng_note_summary(trace)}",
    ]
    return "\n".join(lines)


def _input_preprocessing_lines(trace: "Trace") -> list[str]:
    """Return optional input-preprocessing summary lines.

    Parameters
    ----------
    trace:
        Model log to inspect.

    Returns
    -------
    list[str]
        Empty list when no automatic preprocessing was applied, otherwise a
        two-line summary block.
    """

    record = getattr(trace, "input_preprocessor", None)
    if record is None:
        return []
    return ["Input preprocessing:", f"  {record.description}"]


def _input_shape_summary(trace: "Trace") -> str:
    """Return a compact input-shape summary.

    Parameters
    ----------
    trace:
        Model log to inspect.

    Returns
    -------
    str
        Shape summary or ``"unknown"``.
    """

    layers = getattr(trace, "input_layers", []) or []
    shape = _combined_shape_str(trace, layers)
    if shape and shape != "-":
        return shape
    metadata = getattr(trace, "input_annotations", {}) or {}
    if metadata:
        return _shorten(repr(metadata), limit=80)
    return "unknown"


def _capture_timestamp(trace: "Trace") -> str:
    """Return a readable capture timestamp surrogate.

    Parameters
    ----------
    trace:
        Model log to inspect.

    Returns
    -------
    str
        Pass start/end timing information.
    """

    pass_start = float(getattr(trace, "capture_start_time", 0.0) or 0.0)
    pass_end = float(getattr(trace, "capture_end_time", 0.0) or 0.0)
    if pass_start <= 0:
        return "unknown"
    if pass_end > 0:
        return f"start={pass_start:.6f}, end={pass_end:.6f}"
    return f"start={pass_start:.6f}"


def _run_state_name(trace: "Trace") -> str:
    """Return the run-state enum name.

    Parameters
    ----------
    trace:
        Model log to inspect.

    Returns
    -------
    str
        Run-state name or repr.
    """

    state = getattr(trace, "state", None)
    return str(getattr(state, "name", state))


def _stale_spec_status(trace: "Trace") -> str:
    """Return whether the out recipe is stale.

    Parameters
    ----------
    trace:
        Model log to inspect.

    Returns
    -------
    str
        Staleness summary.
    """

    spec_revision = int(getattr(trace, "_spec_revision", 0) or 0)
    recipe_revision = int(getattr(trace, "_out_recipe_revision", 0) or 0)
    stale = spec_revision != recipe_revision
    return f"{stale} (spec={spec_revision}, out_recipe={recipe_revision})"


def _last_run_summary(trace: "Trace") -> str:
    """Return a compact last-run context summary.

    Parameters
    ----------
    trace:
        Model log to inspect.

    Returns
    -------
    str
        Last-run status.
    """

    ctx = getattr(trace, "last_run", None)
    if not isinstance(ctx, dict) or not ctx:
        return "none"
    engine = ctx.get("engine", "unknown")
    revision = ctx.get("spec_revision", getattr(trace, "_spec_revision", 0))
    duration = ctx.get("duration_s")
    duration_text = (
        f", duration={float(duration):.4f}s" if isinstance(duration, (int, float)) else ""
    )
    return f"engine={engine}, spec_revision={revision}{duration_text}"


def _spec_sample(specs: Sequence[Any]) -> str:
    """Return a short sample of recipe specs.

    Parameters
    ----------
    specs:
        Sequence of recipe spec objects.

    Returns
    -------
    str
        Empty string or parenthesized summary.
    """

    if not specs:
        return ""
    labels = [_site_target_repr(getattr(spec, "site_target", None)) for spec in specs[:3]]
    if len(specs) > 3:
        labels.append("...")
    return f" ({', '.join(labels)})"


def _site_target_repr(site_target: Any) -> str:
    """Return a compact site-target representation.

    Parameters
    ----------
    site_target:
        Target spec-like object.

    Returns
    -------
    str
        Compact representation.
    """

    if site_target is None:
        return "unknown"
    kind = getattr(site_target, "selector_kind", getattr(site_target, "kind", None))
    value = getattr(site_target, "selector_value", getattr(site_target, "value", None))
    if kind is not None:
        return f"{kind}:{value}"
    return _shorten(repr(site_target), limit=48)


def _portability_status(target_specs: Sequence[Any], hook_specs: Sequence[Any]) -> str:
    """Return recipe save/load portability status.

    Parameters
    ----------
    target_specs:
        Target value specs.
    hook_specs:
        Hook specs.

    Returns
    -------
    str
        Portability summary.
    """

    helpers = []
    for spec in tuple(target_specs) + tuple(hook_specs):
        helper = getattr(spec, "helper", None)
        if helper is not None:
            helpers.append(helper)
        value = getattr(spec, "value", None)
        if getattr(value, "portability", None) is not None:
            helpers.append(value)
    opaque = sum(1 for helper in helpers if getattr(helper, "portability", None) == "opaque_audit")
    import_ref = sum(
        1 for helper in helpers if getattr(helper, "portability", None) == "import_ref"
    )
    if opaque:
        return f"{opaque} opaque -> audit-only"
    if import_ref:
        return f"{import_ref} import-ref helper(s) -> environment-dependent"
    return "all helpers builtin -> portable"


def _recent_operation_lines(trace: "Trace") -> list[str]:
    """Return recent operation-history lines.

    Parameters
    ----------
    trace:
        Model log to inspect.

    Returns
    -------
    list[str]
        Indented operation lines.
    """

    history = list(getattr(trace, "state_history", []) or [])
    if not history:
        return ["  none"]
    lines = []
    for record in history[-8:]:
        if isinstance(record, dict):
            op = record.get("op", "unknown")
            revision = record.get("spec_revision", "?")
            detail = _operation_detail(record)
            lines.append(f"  - {op} (spec={revision}){detail}")
        else:
            lines.append(f"  - {_shorten(repr(record), limit=96)}")
    return lines


def _operation_detail(record: Mapping[str, Any]) -> str:
    """Return selected details from one operation record.

    Parameters
    ----------
    record:
        Operation-history record.

    Returns
    -------
    str
        Optional details string.
    """

    detail_keys = ("site", "engine", "name", "origins", "hooks", "append_sequence_id")
    parts = []
    for key in detail_keys:
        if key in record and record[key] not in (None, (), []):
            parts.append(f"{key}={_shorten(repr(record[key]), limit=36)}")
    return f": {', '.join(parts)}" if parts else ""


def _parent_run_summary(trace: "Trace") -> str:
    """Return parent-run status.

    Parameters
    ----------
    trace:
        Model log to inspect.

    Returns
    -------
    str
        Parent summary.
    """

    parent_ref = getattr(trace, "parent_run", None)
    if parent_ref is None:
        return "none"
    parent = parent_ref()
    if parent is None:
        return "collected"
    return f"{getattr(parent, 'trace_label', None)!r} ({getattr(parent, 'model_class_name', None)})"


def _fork_chain_summary(trace: "Trace") -> str:
    """Return a compact fork lineage chain.

    Parameters
    ----------
    trace:
        Model log to inspect.

    Returns
    -------
    str
        Fork chain from root to current log.
    """

    names = [str(getattr(trace, "trace_label", None))]
    seen = {id(trace)}
    current = trace
    while True:
        parent_ref = getattr(current, "parent_run", None)
        if parent_ref is None:
            break
        parent = parent_ref()
        if parent is None or id(parent) in seen:
            break
        names.append(str(getattr(parent, "trace_label", None)))
        seen.add(id(parent))
        current = parent
    return " <- ".join(reversed(names))


def _truncated(value: Any, *, length: int = 8) -> str:
    """Return a truncated hash-like value.

    Parameters
    ----------
    value:
        Value to display.
    length:
        Maximum prefix length.

    Returns
    -------
    str
        Truncated string or ``"unknown"``.
    """

    if value is None:
        return "unknown"
    text = str(value)
    return text[:length]


def _relationship_evidence_summary(trace: "Trace") -> str:
    """Return relationship evidence enum names.

    Parameters
    ----------
    trace:
        Model log to inspect.

    Returns
    -------
    str
        Compact relationship summary.
    """

    evidence = getattr(trace, "relationship_evidence", {}) or {}
    if not evidence:
        return "unknown"
    parts = []
    for key in ("model", "weights", "input", "graph"):
        value = evidence.get(key)
        parts.append(f"{key}={getattr(value, 'name', value)}")
    return ", ".join(parts)


def _next_operation_hint(trace: "Trace") -> str:
    """Return available next-operation guidance.

    Parameters
    ----------
    trace:
        Model log to inspect.

    Returns
    -------
    str
        User-facing next-step hint.
    """

    if getattr(trace, "_has_direct_writes", False):
        return "direct writes present; replay() or rerun() will overlay recipe state"
    if getattr(trace, "_spec_revision", 0) != getattr(trace, "_out_recipe_revision", 0):
        return "spec stale; call replay() or rerun() to propagate"
    if not getattr(trace, "intervention_ready", False):
        return "not intervention-ready; recapture with intervention_ready=True for replay templates"
    return "ready for set(), attach_hooks(), do(), replay(), rerun(), or fork()"


def _rng_note_summary(trace: "Trace") -> str:
    """Return helper RNG and non-determinism notes.

    Parameters
    ----------
    trace:
        Model log to inspect.

    Returns
    -------
    str
        RNG summary.
    """

    notes = []
    for layer in getattr(trace, "layer_list", []) or []:
        for record in getattr(layer, "interventions", []) or []:
            note = getattr(record, "determinism_note", None)
            if note:
                notes.append(str(note))
    if notes:
        return _shorten("; ".join(notes[:3]), limit=140)
    if getattr(trace, "save_rng_states", False):
        return "per-operation RNG states captured"
    return "no unseeded helper RNG notes"


def _shorten(text: str, *, limit: int) -> str:
    """Shorten text to a fixed display limit.

    Parameters
    ----------
    text:
        Text to shorten.
    limit:
        Maximum returned length.

    Returns
    -------
    str
        Shortened text.
    """

    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


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
        raise ValueError("Pass either `level` or `preset`, not both with different values.")
    selected_name: str = preset or level
    selected_name = _LEVEL_ALIASES.get(selected_name, selected_name)
    if selected_name not in _LEVEL_DEFAULT_FIELDS:
        raise ValueError(f"Unsupported summary level: {selected_name!r}.")
    return selected_name


def _resolve_show_ops(*, show_ops: bool, include_ops: Optional[bool]) -> bool:
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
        raise ValueError("Pass either `show_ops` or `include_ops`, not both with different values.")
    return include_ops if include_ops is not None else show_ops


def _resolve_fields(
    level: str,
    *,
    fields: Optional[List[str]],
    columns: Optional[List[str]],
) -> List[str]:
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
        raise ValueError("Pass either `fields` or `columns`, not both with different values.")
    selected = columns if columns is not None else fields
    if selected is None:
        return list(_LEVEL_DEFAULT_FIELDS[level])
    unknown = [field for field in selected if field not in _COLUMN_LABELS]
    if unknown:
        raise ValueError(f"Unsupported summary fields: {unknown}.")
    return list(selected)


def _render_in_progress_summary(
    *,
    trace: "Trace",
    fields: Sequence[str],
    mode: SummaryMode,
    show_ops: bool,
    max_rows: Optional[int],
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
                _render_table(display_fields, rows, max_rows=max_rows),
            ]
        )
    return "\n".join(lines)


def _live_op_rows(trace: "Trace") -> list[dict[str, str]]:
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
        for event in events.op_events:
            rows.append(
                {
                    "name": str(event.layer_label_raw or event.label_raw),
                    "shape": _shape_str(event.output.tensor.shape),
                    "dtype": _dtype_str(event.output.tensor.dtype),
                }
            )
        return rows

    if getattr(trace, "_raw_layer_dict", None):
        rows = []
        for raw_label in trace._raw_layer_labels_list:
            entry = trace._raw_layer_dict[raw_label]
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
    trace: "Trace",
    level: str,
    fields: Sequence[str],
    mode: SummaryMode,
    show_ops: bool,
    max_rows: Optional[int],
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
    primary_rows, footer_lines = _build_level_rows(trace=trace, level=level, mode=mode)
    lines = [
        _level_title(trace=trace, level=level),
        _render_table(fields, primary_rows, max_rows=max_rows),
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
                _render_table(op_fields, op_rows, max_rows=max_rows),
            ]
        )
        if op_footer_lines:
            lines.extend(op_footer_lines)
    return "\n".join(lines)


def _level_title(*, trace: "Trace", level: str) -> str:
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
    }
    return title_map[level]


def _build_level_rows(
    *,
    trace: "Trace",
    level: str,
    mode: SummaryMode,
) -> tuple[List[Dict[str, str]], List[str]]:
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
        return _build_overview_rows(trace)
    if level == "graph":
        return _build_graph_rows(trace)
    if level == "memory":
        return _build_memory_rows(trace, mode=mode)
    if level == "control_flow":
        return _build_control_flow_rows(trace)
    if level == "waterfall":
        return _build_waterfall_rows(trace, mode=mode)
    return _build_compute_rows(trace)


def _build_overview_rows(trace: "Trace") -> tuple[List[Dict[str, str]], List[str]]:
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
    rows: List[Dict[str, str]] = [
        {
            "name": "input",
            "shape": _combined_shape_str(trace, trace.input_layers),
            "params": "0",
            "train": "-",
        }
    ]
    for module in _iter_summary_modules(trace):
        rows.append(_module_overview_row(trace, module))
    rows.append(
        {
            "name": "output",
            "shape": _combined_shape_str(trace, trace.output_layers),
            "params": "-",
            "train": "-",
        }
    )
    footer_lines = [
        f"Params: {_int_with_commas(trace.num_params)} unique; trainable: "
        f"{_int_with_commas(trace.num_params_trainable)}",
        f"Ops: {trace.num_ops} total",
        f"Edges: {trace.num_edges} total",
        f"Branching factor: {trace.branching_factor:.2f}",
        f"Saved outs: {human_readable_size(trace.saved_activation_memory)}",
        f"Forward FLOPs: {_human_flops(trace.total_flops_forward)}  "
        f"MACs: {_human_flops(trace.total_macs_forward)}",
        "FLOP convention: counts use the captured TorchLens convention; "
        "MACs are reported as FLOPs // 2.",
    ]
    return rows, footer_lines


def _build_graph_rows(trace: "Trace") -> tuple[List[Dict[str, str]], List[str]]:
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
    for module in _iter_summary_modules(trace):
        rows.append(
            {
                "name": f"{module.address} ({module.class_name})",
                "shape": _module_shape(trace, module),
                "params": _human_count(module.num_params),
                "parents": _module_parent_summary(module),
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
    trace: "Trace",
    *,
    mode: SummaryMode,
) -> tuple[List[Dict[str, str]], List[str]]:
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
    rows: List[Dict[str, str]] = []
    for entry in _iter_operation_entries(trace, mode=mode):
        memory = int(getattr(entry, "activation_memory", 0) or 0)
        running_total += memory
        rows.append(
            {
                "name": _entry_name(entry),
                "shape": _shape_str(getattr(entry, "shape", None)),
                "dtype": _dtype_str(getattr(entry, "dtype", None)),
                "tensor_mb": _mb_str(memory),
                "running_mb": _mb_str(running_total),
            }
        )
    footer_lines = [
        f"Tracked tensor volume: {human_readable_size(trace.total_activation_memory)}",
        f"Saved outs: {human_readable_size(trace.saved_activation_memory)}",
        "Live forward-memory peak: not tracked in Trace",
    ]
    return rows, footer_lines


def _build_control_flow_rows(trace: "Trace") -> tuple[List[Dict[str, str]], List[str]]:
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
    rows: List[Dict[str, str]] = []
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
    if not rows:
        footer_lines = [
            "No conditional branches or recurrent loop groups were detected in this forward pass."
        ]
        return rows, footer_lines
    footer_lines = [f"Conditionals: {len(rows)}"]
    return rows, footer_lines


def _build_compute_rows(trace: "Trace") -> tuple[List[Dict[str, str]], List[str]]:
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
                "macs": _human_flops(module.total_macs_forward),
                "time_ms": f"{_module_time_ms(trace, module):.2f}",
                "dtype": _module_dtype(trace, module),
            }
        )
    footer_lines = [
        f"Params: {_int_with_commas(trace.num_params)} unique",
        f"Forward FLOPs: {_human_flops(trace.total_flops_forward)}",
        f"MACs: {_human_flops(trace.total_macs_forward)}",
        f"Forward time: {trace.forward_duration * 1000:.2f} ms",
    ]
    return rows, footer_lines


def _build_waterfall_rows(
    trace: "Trace",
    *,
    mode: SummaryMode,
) -> tuple[List[Dict[str, str]], List[str]]:
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
    rows: List[Dict[str, str]] = []
    for entry in _iter_operation_entries(trace, mode=mode):
        duration = float(getattr(entry, "func_duration", 0.0) or 0.0)
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
    trace: "Trace",
    mode: SummaryMode,
    level: str,
) -> tuple[List[Dict[str, str]], List[str]]:
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
    rows: List[Dict[str, str]] = []
    running_total = 0
    for entry in _iter_operation_entries(trace, mode=mode):
        memory = int(getattr(entry, "activation_memory", 0) or 0)
        running_total += memory
        rows.append(
            {
                "name": _entry_name(entry),
                "shape": _shape_str(getattr(entry, "shape", None)),
                "params": _human_count(int(getattr(entry, "num_params", 0) or 0)),
                "parents": _parent_summary(getattr(entry, "parents", [])),
                "dtype": _dtype_str(getattr(entry, "dtype", None)),
                "tensor_mb": _mb_str(memory),
                "running_mb": _mb_str(running_total),
                "flops": _human_flops(int(getattr(entry, "flops_forward", 0) or 0)),
                "macs": _human_flops(int(getattr(entry, "macs_forward", 0) or 0)),
                "time_ms": f"{float(getattr(entry, 'func_duration', 0.0) or 0.0) * 1000:.2f}",
            }
        )
    footer_lines = [
        f"Operation rows shown: {len(rows)} ({mode if mode != 'auto' else _effective_mode(trace, mode)})"
    ]
    return rows, footer_lines


def _default_op_fields(level: str) -> List[str]:
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


def _iter_summary_modules(trace: "Trace") -> List["Module"]:
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


def _module_overview_row(trace: "Trace", module: "Module") -> Dict[str, str]:
    """Build one overview row for a module.

    Parameters
    ----------
    trace:
        Finalized log object.
    module:
        Module to summarize.

    Returns
    -------
    dict[str, str]
        Renderable overview row.
    """
    train = "-"
    if module.num_params > 0:
        train = "yes" if module.num_params_trainable > 0 else "no"
    return {
        "name": f"{module.address} ({module.class_name})",
        "shape": _module_shape(trace, module),
        "params": _human_count(module.num_params),
        "train": train,
        "parents": _module_parent_summary(module),
        "class": module.class_name,
    }


def _module_shape(trace: "Trace", module: "Module") -> str:
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
    if not module.layer_labels:
        return "-"
    try:
        layer = trace[module.layer_labels[-1]]
    except KeyError:
        return "-"
    return _shape_str(getattr(layer, "shape", None))


def _module_parent_summary(module: "Module") -> str:
    """Return a short parent summary for a module row.

    Parameters
    ----------
    module:
        Module to summarize.

    Returns
    -------
    str
        Parent summary text.
    """
    if module.address_parent in (None, "self"):
        return "input"
    return str(module.address_parent)


def _module_dtype(trace: "Trace", module: "Module") -> str:
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
    if not module.layer_labels:
        return "-"
    try:
        layer = trace[module.layer_labels[-1]]
    except KeyError:
        return "-"
    return _dtype_str(getattr(layer, "dtype", None))


def _module_time_ms(trace: "Trace", module: "Module") -> float:
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
        total += float(getattr(layer, "func_duration", 0.0) or 0.0)
    return total * 1000.0


def _iter_operation_entries(
    trace: "Trace",
    *,
    mode: SummaryMode,
) -> Iterable["Layer | Op"]:
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


def _effective_mode(trace: "Trace", mode: SummaryMode) -> Literal["rolled", "unrolled"]:
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
    base_name = getattr(entry, "layer_label", None) or getattr(entry, "layer_label", None)
    if base_name is None:
        base_name = getattr(entry, "label", None) or getattr(entry, "layer_label", "?")
    if (
        getattr(entry, "num_calls", 1)
        and getattr(entry, "num_calls", 1) > 1
        and hasattr(entry, "ops")
    ):
        return f"{base_name} x{getattr(entry, 'num_calls', 1)}"
    if getattr(entry, "call_index", 1) > 1:
        return str(getattr(entry, "layer_label", base_name))
    return str(base_name)


def _combined_shape_str(trace: "Trace", labels: Sequence[str]) -> str:
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


def _event_branch_kinds(trace: "Trace", event: "ConditionalEvent") -> List[str]:
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


def _event_source(event: "ConditionalEvent") -> str:
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
    return terminal_file_line_link(event.source_file, event.if_stmt_span[0])


def _event_bool_layer(event: "ConditionalEvent") -> str:
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


def _event_branch_op_count(trace: "Trace", event: "ConditionalEvent") -> int:
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


def _human_flops(value: int) -> str:
    """Format FLOPs or MACs compactly.

    Parameters
    ----------
    value:
        FLOP-like integer.

    Returns
    -------
    str
        Compact FLOP string.
    """
    return format_flops(value)


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
    rows: Sequence[Dict[str, str]],
    *,
    max_rows: Optional[int],
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

    Returns
    -------
    str
        ASCII table string.
    """
    if not rows:
        return "(no rows)"

    display_rows = list(rows)
    truncated_count = 0
    if max_rows is not None and len(display_rows) > max_rows:
        truncated_count = len(display_rows) - max_rows
        display_rows = display_rows[:max_rows]

    widths = []
    for field in fields:
        header = _COLUMN_LABELS[field]
        cell_width = max(len(str(row.get(field, "-"))) for row in display_rows)
        widths.append(max(len(header), cell_width))

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header_row = (
        "| "
        + " | ".join(_COLUMN_LABELS[field].ljust(width) for field, width in zip(fields, widths))
        + " |"
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
