"""Rebuilt summary result assembly (F08; summary memo item 12, 3.7/3.8).

``build_rebuilt_summary`` is the ONE report builder both entry points
share: resolve the view ladder, gather detached facts, render the
canonical ASCII payload (plus its unicode sibling and the HTML fragment),
and hand back a ``SummaryReport`` -- a ``str`` subclass whose text IS the
ASCII contract and whose data survives model/Trace cleanup and GC.

The raw-numbers pin holds throughout: every numeric field on the result
is a plain int or None; formatting exists only in renderers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError
from ._summary_config import SummaryConfig
from ._summary_ladder import ResolvedView, ViewRow, resolve_view
from ._summary_render import SummaryFacts, render_summary_pair

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from ._summary_report import SummaryReport


@dataclass(frozen=True)
class RebuiltPayload:
    """The detached render payload carried by a rebuilt summary result."""

    view: ResolvedView
    facts: SummaryFacts
    config: SummaryConfig
    ascii_text: str
    unicode_text: str
    visible_rows: tuple[ViewRow, ...]
    filter_note: str | None


def _tensor_leaf_shapes(trace: Trace) -> tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]:
    """Input shapes/dtypes from the capture's input boundary rows."""

    shapes: list[tuple[int, ...]] = []
    dtypes: list[str] = []
    for op in getattr(trace, "input_ops", ()) or ():
        shape = getattr(op, "shape", None)
        if shape is None:
            continue
        shapes.append(tuple(int(dim) for dim in shape))
        dtype = getattr(op, "dtype", None)
        dtypes.append(str(dtype).replace("torch.", "") if dtype is not None else "?")
    return tuple(shapes), tuple(dtypes)


def collect_summary_facts(
    trace: Trace,
    *,
    execution_note: str | None = None,
    input_synthesis: str | None = None,
) -> SummaryFacts:
    """Gather every trace-side fact the renderer prints, detached."""

    from .._capture_honesty import honesty_banner_lines
    from ._compute_truth import unknown_op_ledger
    from ._factcore import factcore
    from ._health import health_facts

    core = factcore(trace)
    health = health_facts(trace)
    input_shapes, input_dtypes = _tensor_leaf_shapes(trace)
    outcome = getattr(trace, "outcome", None)
    # Pass-level peaks publish ONLY through the observe gate (F24 item 4):
    # the basis is named, cpu/mps render under their real meaning, and a
    # CUDA 0 is a high-water fact -- never a raw forward-peak read here.
    from ..observe._peaks import pass_peak_facts

    peak_facts = pass_peak_facts(trace, "forward")
    unexecuted_names: tuple[str, ...] = tuple(
        str(name) for name in (getattr(trace, "uncalled_modules", ()) or ())[:5]
    )
    unknown_groups = unknown_op_ledger(trace)
    unknown_names = tuple(group.func_name or "?" for group in unknown_groups)
    param_bytes = None
    try:
        param_bytes = sum(int(getattr(param, "num_params", 0) or 0) * 4 for param in trace.params)
    except Exception:  # noqa: BLE001 - fact gathering degrades to unknown, never breaks summary()
        param_bytes = None
    buffer_count = None
    try:
        buffer_count = len(list(getattr(trace, "buffer_layers", ()) or ()))
    except Exception:  # noqa: BLE001 - fact gathering degrades to unknown, never breaks summary()
        buffer_count = None
    passes_max = 1
    for op in getattr(trace, "layer_list", ()) or ():
        passes_max = max(passes_max, int(getattr(op, "num_passes", 1) or 1))
    return SummaryFacts(
        model_class_name=getattr(trace, "model_class_name", None),
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        backend=getattr(trace, "backend", None),
        outcome_status=getattr(getattr(outcome, "status", None), "value", None),
        capture_verified=getattr(trace, "capture_verified", None),
        structure_only=bool(getattr(trace, "structure_only", False)),
        health_verdict=health.verdict,
        params_total=core.params.total,
        params_per_path_total=core.params.per_path_total,
        params_trainable=core.params.trainable,
        params_frozen=core.params.frozen,
        params_executed=core.params.executed,
        params_unexecuted=core.params.unexecuted,
        unexecuted_names=unexecuted_names,
        tied_groups=core.params.tied_groups,
        param_bytes=param_bytes,
        buffer_count=buffer_count,
        flops_forward_fma2=int(core.compute.partition_total),
        macs_forward=int(core.compute.macs_total),
        coverage_known=core.compute.coverage.known,
        coverage_zero=core.compute.coverage.zero_by_rule,
        coverage_unknown=core.compute.coverage.unknown,
        unknown_op_names=unknown_names,
        peak_value_bytes=peak_facts.value_bytes,
        peak_meaning=peak_facts.meaning,
        at_capture_bytes=core.memory.at_capture_bytes,
        retained_now_bytes=core.memory.retained_now_bytes,
        compute_ops=core.counts.compute_ops,
        tracked_tensor_rows=core.counts.tracked_tensor_rows,
        passes_max=passes_max,
        execution_note=execution_note,
        input_synthesis=input_synthesis,
        honesty_lines=tuple(honesty_banner_lines(trace)),
    )


def _apply_filter(
    view: ResolvedView, facts: SummaryFacts, config: SummaryConfig
) -> tuple[tuple[ViewRow, ...], str | None]:
    """Presentation-only filtering with the mandatory coverage disclosure."""

    if config.filter is None:
        return view.rows, None
    predicate = config.filter
    if isinstance(predicate, str):
        pattern = re.compile(predicate)

        def match(row: ViewRow) -> bool:
            """Regex filter over the row name and address."""

            return bool(pattern.search(row.name) or (row.address and pattern.search(row.address)))

    elif callable(predicate):

        def match(row: ViewRow) -> bool:
            """User-callable filter over the typed row."""

            return bool(predicate(row))

    else:
        raise InvalidArgumentError(
            f"filter= accepts a regex string or a callable over rows; got "
            f"{type(predicate).__name__}.",
            code="summary_option_invalid",
            remedy="pass a regex string, e.g. filter='conv', or a callable row -> bool",
        )
    visible = tuple(row for row in view.rows if match(row))
    params_total = facts.params_total or 0
    flops_total = facts.flops_forward_fma2 or 0
    visible_params = sum(row.params_owned for row in visible)
    visible_flops = sum(row.flops_owned for row in visible)
    param_pct = 100.0 * visible_params / params_total if params_total else 0.0
    flop_pct = 100.0 * visible_flops / flops_total if flops_total else 0.0
    note = (
        f"showing {len(visible)}/{len(view.rows)} rows; visible coverage "
        f"{param_pct:.0f}% params, {flop_pct:.0f}% known FLOPs; totals are whole-model"
    )
    return visible, note


def build_rebuilt_summary(
    trace: Trace,
    config: SummaryConfig,
    *,
    execution_note: str | None = None,
    input_synthesis: str | None = None,
) -> SummaryReport:
    """Build one rebuilt summary result from a finished trace."""

    from ._summary_report import SummaryReport, build_summary_data

    if config.flop_convention == "fma1":
        from ._factcore import factcore

        for row in factcore(trace).compute.rows:
            if row.kind == "op" and row.flops_fma2 is not None and row.fma_macs is None:
                raise InvalidArgumentError(
                    "flop_convention='fma1' needs each op's two-term compute record; "
                    f"op {row.label!r} has FLOPs with no derivable MAC split -- the "
                    "request is never accepted-and-ignored.",
                    code="flop_convention_unavailable",
                    remedy="re-capture with a torchlens build that records the "
                    "two-term compute record, or keep the stored fma2 convention",
                )
    view = resolve_view(
        trace,
        level=config.level,
        depth=config.depth,
        max_rows=config.max_rows,
        fold_repeats=config.fold_repeats,
    )
    facts = collect_summary_facts(
        trace, execution_note=execution_note, input_synthesis=input_synthesis
    )
    visible_rows, filter_note = _apply_filter(view, facts, config)
    ascii_text, unicode_text = render_summary_pair(
        view, facts, config, visible_rows=visible_rows, filter_note=filter_note
    )
    rows, totals, capture = build_summary_data(trace)
    payload = RebuiltPayload(
        view=view,
        facts=facts,
        config=config,
        ascii_text=ascii_text,
        unicode_text=unicode_text,
        visible_rows=visible_rows,
        filter_note=filter_note,
    )
    return SummaryReport(ascii_text, rows=rows, totals=totals, capture=capture, rebuilt=payload)


def render_result_html(payload: RebuiltPayload) -> str:
    """The HTML fragment for one rebuilt payload (shared by both doors)."""

    from ._summary_html import render_html
    from ._summary_render import _banner_lines

    facts = payload.facts
    header = facts.model_class_name or "model"
    if facts.input_shapes:
        bits = ", ".join(
            "(" + ", ".join(str(d) for d in shape) + f") {dtype}"
            for shape, dtype in zip(facts.input_shapes, facts.input_dtypes, strict=False)
        )
        header += f" | input {bits}"
    if facts.execution_note:
        header += f" | {facts.execution_note}"
    footer_lines = tuple(
        line
        for line in payload.ascii_text.rsplit("\n", 8)[-8:]
        if line.startswith(("Params", "Compute", "Memory", "Graph", "Capture", "More:", "*"))
    )
    return render_html(
        payload.view,
        facts,
        payload.config,
        visible_rows=payload.visible_rows,
        footer_lines=footer_lines,
        banner_lines=tuple(_banner_lines(facts)),
        header_line=header,
    )


def result_to_pandas(report: SummaryReport, scope: str = "display") -> Any:
    """Typed pandas projection (display rows or the full op grain)."""

    import pandas

    if scope not in ("display", "all"):
        raise InvalidArgumentError(
            f"to_pandas scope must be 'display' or 'all'; got {scope!r}.",
            code="summary_option_invalid",
            remedy="use scope='display' (rendered rows) or scope='all' (op grain)",
        )
    payload = getattr(report, "_rebuilt", None)
    if scope == "display" and payload is not None:
        records = [
            {
                "row_id": row.row_id,
                "kind": row.kind,
                "name": row.name,
                "class_name": row.class_name,
                "address": row.address,
                "depth": row.depth,
                "output_shape": row.output_shape,
                "passes": row.passes,
                "params": row.params_display,
                "params_owned": row.params_owned,
                "flops": row.flops_display,
                "flops_owned": row.flops_owned,
                "macs_owned": row.macs_owned,
                "trainable": row.trainable,
                "fold_count": row.fold_count,
                "evidence": row.evidence,
            }
            for row in payload.visible_rows
        ]
        frame = pandas.DataFrame.from_records(records)
        for column in ("params", "params_owned", "flops", "flops_owned", "macs_owned"):
            frame[column] = frame[column].astype("Int64")
        return frame
    records = [dict(row.__dict__) for row in report.rows]
    return pandas.DataFrame.from_records(records)


def result_to_markdown(report: SummaryReport) -> str:
    """A GitHub-flavored markdown table of the rendered rows."""

    payload = getattr(report, "_rebuilt", None)
    if payload is None:
        raise InvalidArgumentError(
            "to_markdown() serves rebuilt-grammar summaries; this report "
            "carries no rebuilt-grammar payload (it was assembled from text).",
            code="summary_result_legacy",
            remedy="call summary() with the rebuilt grammar (bare call, level=, view=, ...)",
        )
    from ._summary_render import _cell

    columns = payload.config.resolved_columns(payload.view.mixed_trainability)
    params_total = payload.facts.params_total or 0
    lines = [
        "| " + " | ".join(column.header for column in columns) + " |",
        "| " + " | ".join("---:" if c.align == "right" else ":--" for c in columns) + " |",
    ]
    for row in payload.visible_rows:
        cells = [
            _cell(row, column, payload.config, params_total).replace("|", "\\|")
            for column in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def result_details(report: SummaryReport) -> str:
    """The relocated capture-facts block (memo 3.6): result-side details."""

    capture = report.capture
    payload = getattr(report, "_rebuilt", None)
    lines = [
        "capture facts:",
        f"  backend: {capture.backend}",
        f"  model: {capture.model_class_name}",
        f"  outcome: {capture.outcome_status}",
        f"  verified: {capture.capture_verified}",
        f"  structure_only: {capture.structure_only}",
        f"  health: {capture.health_verdict}",
        f"  fingerprint: {capture.capture_fingerprint}",
        f"  advisories: {capture.advisories}",
    ]
    if payload is not None:
        facts = payload.facts
        lines += [
            f"  input: {', '.join(map(str, facts.input_shapes)) or 'n/a'}",
            f"  execution: {facts.execution_note or 'as-captured'}",
            f"  synthesis: {facts.input_synthesis or 'real input'}",
            f"  view: {payload.view.disclosure}",
        ]
    return "\n".join(lines)
