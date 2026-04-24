"""Build compact text summaries for ``ModelLog`` objects."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence

from ..utils.display import human_readable_size

if TYPE_CHECKING:
    from ..data_classes.layer_log import LayerLog
    from ..data_classes.layer_pass_log import LayerPassLog
    from ..data_classes.model_log import ConditionalEvent, ModelLog
    from ..data_classes.module_log import ModuleLog


SummaryLevel = Literal["overview", "graph", "memory", "control_flow", "compute", "cost"]
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
}


def render_model_summary(
    model_log: "ModelLog",
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
    """Render a textual summary for a ``ModelLog``.

    Parameters
    ----------
    model_log:
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
        Operation aggregation mode. ``"rolled"`` uses ``LayerLog`` rows,
        ``"unrolled"`` uses ``LayerPassLog`` rows, and ``"auto"`` chooses
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

    if not model_log._pass_finished:
        text = _render_in_progress_summary(
            model_log=model_log,
            fields=resolved_fields,
            mode=mode,
            show_ops=resolved_show_ops,
            max_rows=max_rows,
        )
    else:
        text = _render_finished_summary(
            model_log=model_log,
            level=resolved_level,
            fields=resolved_fields,
            mode=mode,
            show_ops=resolved_show_ops,
            max_rows=max_rows,
        )
    if print_to is not None:
        print_to(text)
    return text


def format_model_repr(model_log: "ModelLog") -> str:
    """Return a short ``repr`` string for a ``ModelLog``.

    Parameters
    ----------
    model_log:
        Logged model metadata to summarize.

    Returns
    -------
    str
        Short two-line representation.
    """
    if not model_log._pass_finished:
        lines = [
            f"<ModelLog {model_log.model_name}>",
            f"pass_in_progress ops_logged={len(model_log._raw_layer_dict)} finalized=False",
        ]
        return "\n".join(lines)

    lines = [
        f"<ModelLog {model_log.model_name}>",
        " ".join(
            [
                f"layers={len(model_log.layer_logs)}",
                f"params={_human_count(model_log.total_params)}",
                f"ops={model_log.num_operations}",
                f"saved={human_readable_size(model_log.saved_activation_memory)}",
                f"forward={model_log.time_forward_pass:.3f}s",
            ]
        ),
    ]
    return "\n".join(lines)


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
    model_log: "ModelLog",
    fields: Sequence[str],
    mode: SummaryMode,
    show_ops: bool,
    max_rows: Optional[int],
) -> str:
    """Render a truthful summary while the pass is still in progress.

    Parameters
    ----------
    model_log:
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
        f"Model: {model_log.model_name}",
        "Status: pass in progress; postprocessing has not finished yet.",
        f"Ops logged so far: {len(model_log._raw_layer_dict)}",
    ]
    if show_ops and model_log._raw_layer_dict:
        rows = []
        for raw_label in model_log._raw_layer_labels_list:
            entry = model_log._raw_layer_dict[raw_label]
            row = {
                "name": str(
                    getattr(entry, "layer_label_raw", None)
                    or getattr(entry, "tensor_label_raw", raw_label)
                ),
                "shape": _shape_str(getattr(entry, "tensor_shape", None)),
                "dtype": _dtype_str(getattr(entry, "tensor_dtype", None)),
            }
            rows.append(row)
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


def _render_finished_summary(
    *,
    model_log: "ModelLog",
    level: str,
    fields: Sequence[str],
    mode: SummaryMode,
    show_ops: bool,
    max_rows: Optional[int],
) -> str:
    """Render a summary for a finalized ``ModelLog``.

    Parameters
    ----------
    model_log:
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
    primary_rows, footer_lines = _build_level_rows(model_log=model_log, level=level, mode=mode)
    lines = [
        _level_title(model_log=model_log, level=level),
        _render_table(fields, primary_rows, max_rows=max_rows),
    ]
    if footer_lines:
        lines.extend(footer_lines)
    if show_ops and level != "control_flow":
        op_fields = _default_op_fields(level)
        op_rows, op_footer_lines = _build_operation_rows(
            model_log=model_log, mode=mode, level=level
        )
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


def _level_title(*, model_log: "ModelLog", level: str) -> str:
    """Return the section title for a summary level.

    Parameters
    ----------
    model_log:
        Finalized log object.
    level:
        Canonical level name.

    Returns
    -------
    str
        Section title.
    """
    title_map = {
        "overview": f"Model: {model_log.model_name}",
        "graph": f"Graph Summary: {model_log.model_name}",
        "memory": f"Memory Summary: {model_log.model_name}",
        "control_flow": f"Control-Flow Summary: {model_log.model_name}",
        "compute": f"Compute Summary: {model_log.model_name}",
    }
    return title_map[level]


def _build_level_rows(
    *,
    model_log: "ModelLog",
    level: str,
    mode: SummaryMode,
) -> tuple[List[Dict[str, str]], List[str]]:
    """Build rows and footer lines for one summary level.

    Parameters
    ----------
    model_log:
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
        return _build_overview_rows(model_log)
    if level == "graph":
        return _build_graph_rows(model_log)
    if level == "memory":
        return _build_memory_rows(model_log, mode=mode)
    if level == "control_flow":
        return _build_control_flow_rows(model_log)
    return _build_compute_rows(model_log)


def _build_overview_rows(model_log: "ModelLog") -> tuple[List[Dict[str, str]], List[str]]:
    """Build the default overview rows.

    Parameters
    ----------
    model_log:
        Finalized log object.

    Returns
    -------
    tuple[list[dict[str, str]], list[str]]
        Overview rows and footer lines.
    """
    rows: List[Dict[str, str]] = [
        {
            "name": "input",
            "shape": _combined_shape_str(model_log, model_log.input_layers),
            "params": "0",
            "train": "-",
        }
    ]
    for module in _iter_summary_modules(model_log):
        rows.append(_module_overview_row(model_log, module))
    rows.append(
        {
            "name": "output",
            "shape": _combined_shape_str(model_log, model_log.output_layers),
            "params": "-",
            "train": "-",
        }
    )
    footer_lines = [
        f"Params: {_int_with_commas(model_log.total_params)} unique; trainable: "
        f"{_int_with_commas(model_log.total_params_trainable)}",
        f"Ops: {model_log.num_operations} total",
        f"Saved activations: {human_readable_size(model_log.saved_activation_memory)}",
        f"Forward FLOPs: {_human_flops(model_log.total_flops_forward)}  "
        f"MACs: {_human_flops(model_log.total_macs_forward)}",
    ]
    return rows, footer_lines


def _build_graph_rows(model_log: "ModelLog") -> tuple[List[Dict[str, str]], List[str]]:
    """Build graph-summary rows.

    Parameters
    ----------
    model_log:
        Finalized log object.

    Returns
    -------
    tuple[list[dict[str, str]], list[str]]
        Graph rows and footer lines.
    """
    rows = []
    for module in _iter_summary_modules(model_log):
        rows.append(
            {
                "name": f"{module.address} ({module.module_class_name})",
                "shape": _module_shape(model_log, module),
                "params": _human_count(module.num_params),
                "parents": _module_parent_summary(module),
            }
        )
    footer_lines = [
        f"Modules shown: {len(rows)}",
        f"Ops tracked: {model_log.num_operations}",
    ]
    return rows, footer_lines


def _build_memory_rows(
    model_log: "ModelLog",
    *,
    mode: SummaryMode,
) -> tuple[List[Dict[str, str]], List[str]]:
    """Build memory-summary rows.

    Parameters
    ----------
    model_log:
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
    for entry in _iter_operation_entries(model_log, mode=mode):
        tensor_memory = int(getattr(entry, "tensor_memory", 0) or 0)
        running_total += tensor_memory
        rows.append(
            {
                "name": _entry_name(entry),
                "shape": _shape_str(getattr(entry, "tensor_shape", None)),
                "dtype": _dtype_str(getattr(entry, "tensor_dtype", None)),
                "tensor_mb": _mb_str(tensor_memory),
                "running_mb": _mb_str(running_total),
            }
        )
    footer_lines = [
        f"Tracked tensor volume: {human_readable_size(model_log.total_activation_memory)}",
        f"Saved activations: {human_readable_size(model_log.saved_activation_memory)}",
        "Live forward-memory peak: not tracked in ModelLog",
    ]
    return rows, footer_lines


def _build_control_flow_rows(model_log: "ModelLog") -> tuple[List[Dict[str, str]], List[str]]:
    """Build control-flow rows or an empty state.

    Parameters
    ----------
    model_log:
        Finalized log object.

    Returns
    -------
    tuple[list[dict[str, str]], list[str]]
        Control-flow rows and footer lines.
    """
    rows: List[Dict[str, str]] = []
    for event in model_log.conditional_events:
        branch_kinds = _event_branch_kinds(model_log, event)
        rows.append(
            {
                "site": f"cond#{event.id}",
                "source": _event_source(event),
                "taken": ",".join(branch_kinds) if branch_kinds else "unknown",
                "bool_layer": _event_bool_layer(event),
                "branch_ops": str(_event_branch_op_count(model_log, event)),
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


def _build_compute_rows(model_log: "ModelLog") -> tuple[List[Dict[str, str]], List[str]]:
    """Build compute-summary rows.

    Parameters
    ----------
    model_log:
        Finalized log object.

    Returns
    -------
    tuple[list[dict[str, str]], list[str]]
        Compute rows and footer lines.
    """
    rows = []
    for module in _iter_summary_modules(model_log):
        rows.append(
            {
                "name": module.address,
                "params": _human_count(module.num_params),
                "flops": _human_flops(module.flops_forward),
                "macs": _human_flops(module.macs_forward),
                "time_ms": f"{_module_time_ms(model_log, module):.2f}",
                "dtype": _module_dtype(model_log, module),
            }
        )
    footer_lines = [
        f"Params: {_int_with_commas(model_log.total_params)} unique",
        f"Forward FLOPs: {_human_flops(model_log.total_flops_forward)}",
        f"MACs: {_human_flops(model_log.total_macs_forward)}",
        f"Forward time: {model_log.time_forward_pass * 1000:.2f} ms",
    ]
    return rows, footer_lines


def _build_operation_rows(
    *,
    model_log: "ModelLog",
    mode: SummaryMode,
    level: str,
) -> tuple[List[Dict[str, str]], List[str]]:
    """Build operation rows for the optional op dump.

    Parameters
    ----------
    model_log:
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
    for entry in _iter_operation_entries(model_log, mode=mode):
        tensor_memory = int(getattr(entry, "tensor_memory", 0) or 0)
        running_total += tensor_memory
        rows.append(
            {
                "name": _entry_name(entry),
                "shape": _shape_str(getattr(entry, "tensor_shape", None)),
                "params": _human_count(int(getattr(entry, "num_params_total", 0) or 0)),
                "parents": _parent_summary(getattr(entry, "parent_layers", [])),
                "dtype": _dtype_str(getattr(entry, "tensor_dtype", None)),
                "tensor_mb": _mb_str(tensor_memory),
                "running_mb": _mb_str(running_total),
                "flops": _human_flops(int(getattr(entry, "flops_forward", 0) or 0)),
                "macs": _human_flops(int(getattr(entry, "macs_forward", 0) or 0)),
                "time_ms": f"{float(getattr(entry, 'func_time', 0.0) or 0.0) * 1000:.2f}",
            }
        )
    footer_lines = [
        f"Operation rows shown: {len(rows)} ({mode if mode != 'auto' else _effective_mode(model_log, mode)})"
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


def _iter_summary_modules(model_log: "ModelLog") -> List["ModuleLog"]:
    """Return top-level module rows for summary tables.

    Parameters
    ----------
    model_log:
        Finalized log object.

    Returns
    -------
    list[ModuleLog]
        Top-level modules in accessor order.
    """
    modules = []
    for module in model_log.modules:
        if module.address == "self":
            continue
        if module.address_depth == 1:
            modules.append(module)
    return modules


def _module_overview_row(model_log: "ModelLog", module: "ModuleLog") -> Dict[str, str]:
    """Build one overview row for a module.

    Parameters
    ----------
    model_log:
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
        "name": f"{module.address} ({module.module_class_name})",
        "shape": _module_shape(model_log, module),
        "params": _human_count(module.num_params),
        "train": train,
        "parents": _module_parent_summary(module),
        "class": module.module_class_name,
    }


def _module_shape(model_log: "ModelLog", module: "ModuleLog") -> str:
    """Return a representative output shape for a module.

    Parameters
    ----------
    model_log:
        Finalized log object.
    module:
        Module to summarize.

    Returns
    -------
    str
        Representative output shape.
    """
    if not module.all_layers:
        return "-"
    try:
        layer = model_log[module.all_layers[-1]]
    except KeyError:
        return "-"
    return _shape_str(getattr(layer, "tensor_shape", None))


def _module_parent_summary(module: "ModuleLog") -> str:
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


def _module_dtype(model_log: "ModelLog", module: "ModuleLog") -> str:
    """Return a representative dtype for a module.

    Parameters
    ----------
    model_log:
        Finalized log object.
    module:
        Module to summarize.

    Returns
    -------
    str
        Representative dtype.
    """
    if not module.all_layers:
        return "-"
    try:
        layer = model_log[module.all_layers[-1]]
    except KeyError:
        return "-"
    return _dtype_str(getattr(layer, "tensor_dtype", None))


def _module_time_ms(model_log: "ModelLog", module: "ModuleLog") -> float:
    """Return the summed forward time for a module.

    Parameters
    ----------
    model_log:
        Finalized log object.
    module:
        Module to summarize.

    Returns
    -------
    float
        Summed execution time in milliseconds.
    """
    total = 0.0
    for layer_label in module.all_layers:
        try:
            layer = model_log[layer_label]
        except KeyError:
            continue
        total += float(getattr(layer, "func_time", 0.0) or 0.0)
    return total * 1000.0


def _iter_operation_entries(
    model_log: "ModelLog",
    *,
    mode: SummaryMode,
) -> Iterable["LayerLog | LayerPassLog"]:
    """Iterate operation-like entries according to the requested mode.

    Parameters
    ----------
    model_log:
        Finalized log object.
    mode:
        Requested aggregation mode.

    Returns
    -------
    Iterable[LayerLog | LayerPassLog]
        Operation entries in display order.
    """
    effective_mode = _effective_mode(model_log, mode)
    if effective_mode == "rolled":
        return model_log.layer_logs.values()
    return model_log.layer_list


def _effective_mode(model_log: "ModelLog", mode: SummaryMode) -> Literal["rolled", "unrolled"]:
    """Resolve the effective operation mode.

    Parameters
    ----------
    model_log:
        Finalized log object.
    mode:
        Requested mode.

    Returns
    -------
    Literal["rolled", "unrolled"]
        Effective operation mode.
    """
    if mode == "auto":
        return "unrolled" if model_log.is_recurrent else "rolled"
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
    base_name = getattr(entry, "layer_label_no_pass", None) or getattr(entry, "layer_label", None)
    if base_name is None:
        base_name = getattr(entry, "layer_label_w_pass", None) or getattr(entry, "layer_label", "?")
    if (
        getattr(entry, "num_passes", 1)
        and getattr(entry, "num_passes", 1) > 1
        and hasattr(entry, "passes")
    ):
        return f"{base_name} x{getattr(entry, 'num_passes', 1)}"
    if getattr(entry, "pass_num", 1) > 1:
        return str(getattr(entry, "layer_label", base_name))
    return str(base_name)


def _combined_shape_str(model_log: "ModelLog", labels: Sequence[str]) -> str:
    """Return a compact combined shape string for one or more labels.

    Parameters
    ----------
    model_log:
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
            shapes.append(_shape_str(getattr(model_log[label], "tensor_shape", None)))
        except KeyError:
            continue
    if not shapes:
        return "-"
    if len(shapes) == 1:
        return shapes[0]
    return f"{len(shapes)} tensors"


def _event_branch_kinds(model_log: "ModelLog", event: "ConditionalEvent") -> List[str]:
    """Return the taken branch kinds for one conditional event.

    Parameters
    ----------
    model_log:
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
        for (cond_id, branch_kind) in model_log.conditional_arm_edges
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
    return f"{Path(event.source_file).name}:{event.if_stmt_span[0]}"


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
        return event.bool_layers[0]
    return f"{event.bool_layers[0]} +{len(event.bool_layers) - 1}"


def _event_branch_op_count(model_log: "ModelLog", event: "ConditionalEvent") -> int:
    """Return the number of operation edges attributed to a conditional event.

    Parameters
    ----------
    model_log:
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
        for (cond_id, _branch_kind), edges in model_log.conditional_arm_edges.items()
        if cond_id == event.id
    )


def _parent_summary(parent_layers: Sequence[str]) -> str:
    """Return a compact parent-layer summary.

    Parameters
    ----------
    parent_layers:
        Parent layer labels.

    Returns
    -------
    str
        Compact parent summary.
    """
    if not parent_layers:
        return "-"
    if len(parent_layers) == 1:
        return str(parent_layers[0])
    return f"{parent_layers[0]} +{len(parent_layers) - 1}"


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
    return _human_count(value)


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
