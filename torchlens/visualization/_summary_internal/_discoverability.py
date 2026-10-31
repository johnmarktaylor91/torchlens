"""The capture-provenance block served by ``trace.provenance()``.

The ~35-line block and its exclusive helpers (formerly the summary
preamble; the F08 rebuild moved it to ``trace.provenance()`` byte-identically).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

if TYPE_CHECKING:
    from ...data_classes.trace import Trace


def format_discoverability_summary(
    trace: Trace,
    *,
    show_input_preprocessing_details: bool = False,
) -> str:
    """Render the Phase 13 user-facing discoverability summary.

    Parameters
    ----------
    trace:
        Model log to summarize.
    show_input_preprocessing_details:
        Whether to include verification/source detail for input preprocessing.

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
        *_input_preprocessing_lines(
            trace,
            show_details=show_input_preprocessing_details,
        ),
        *_output_postprocessing_lines(trace),
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


def _input_preprocessing_lines(
    trace: Trace,
    *,
    show_details: bool = False,
) -> list[str]:
    """Return optional input-preprocessing summary lines.

    Parameters
    ----------
    trace:
        Model log to inspect.
    show_details:
        Whether to include verification/source detail.

    Returns
    -------
    list[str]
        Empty list when no automatic preprocessing was applied, otherwise a
        two-line summary block.
    """

    record = getattr(trace, "input_preprocessor", None)
    if record is None:
        return []
    lines = ["Input preprocessing:", f"  {record.description}"]
    if not show_details:
        return lines
    verified = bool(getattr(record, "verified", False))
    status = "verified" if verified else "UNVERIFIED"
    lines.append(
        f"  status: {status}; source={getattr(record, 'source', None)}; "
        f"identifier={getattr(record, 'identifier', None)}"
    )
    if not verified:
        lines.append("  WARNING: input preprocessing is UNVERIFIED; inspect transform assumptions.")
    return lines


def _output_postprocessing_lines(trace: Trace) -> list[str]:
    """Return output-postprocessing summary lines.

    Parameters
    ----------
    trace:
        Model log to inspect.

    Returns
    -------
    list[str]
        Compact output decode provenance and preview lines.
    """

    record = getattr(trace, "output_postprocessor", None)
    if record is None:
        return [
            "Output postprocessing:",
            "  undetected; pass output_style= to decode.",
        ]
    verified = "verified" if bool(getattr(record, "verified", False)) else "unverified"
    confidence = getattr(record, "confidence", None)
    confidence_text = "" if confidence is None else f", confidence={float(confidence):.2f}"
    lines = [
        "Output postprocessing:",
        f"  style={getattr(record, 'style', None) or 'unknown'}; {verified}{confidence_text}; "
        f"{getattr(record, 'description', '')}",
    ]
    preview = _decoded_output_preview(trace)
    if preview:
        lines.append(f"  preview: {preview}")
    return lines


def _decoded_output_preview(trace: Trace) -> str | None:
    """Return a compact decoded-output preview for discoverability summary.

    Parameters
    ----------
    trace:
        Model log to inspect.

    Returns
    -------
    str | None
        Preview text, if a batch top-k table is available.
    """

    rows = _decoded_batch_topk_rows(getattr(trace, "decoded_output", None))
    if not rows:
        return None
    parts: list[str] = []
    for batch_item in sorted({int(row.get("batch_item", 0)) for row in rows})[:2]:
        item_rows = [row for row in rows if int(row.get("batch_item", -1)) == batch_item][:3]
        labels = ", ".join(
            f"{row.get('label')} {float(row.get('prob', 0.0)):.0%}" for row in item_rows
        )
        parts.append(f"item {batch_item}: {labels}")
    return " | ".join(parts)


def _decoded_batch_topk_rows(value: Any) -> list[Mapping[str, Any]] | None:
    """Return batch top-k rows from a decoded output value.

    Parameters
    ----------
    value:
        Decoded output candidate.

    Returns
    -------
    list[Mapping[str, Any]] | None
        Rows if the value is a batch top-k table.
    """

    if isinstance(value, Mapping) and value.get("kind") == "batch_topk":
        rows = value.get("rows")
    elif isinstance(value, list):
        rows = value
    else:
        return None
    if isinstance(rows, list) and all(
        isinstance(row, Mapping) and {"batch_item", "rank", "label", "prob"} <= set(row)
        for row in rows
    ):
        return cast(list[Mapping[str, Any]], rows)
    return None


def _input_shape_summary(trace: Trace) -> str:
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


def _capture_timestamp(trace: Trace) -> str:
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


def _run_state_name(trace: Trace) -> str:
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


def _stale_spec_status(trace: Trace) -> str:
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


def _last_run_summary(trace: Trace) -> str:
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


def _recent_operation_lines(trace: Trace) -> list[str]:
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


def _parent_run_summary(trace: Trace) -> str:
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


def _fork_chain_summary(trace: Trace) -> str:
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


def _relationship_evidence_summary(trace: Trace) -> str:
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


def _next_operation_hint(trace: Trace) -> str:
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


def _rng_note_summary(trace: Trace) -> str:
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
