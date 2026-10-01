"""The rebuilt summary renderer (F08; summary memo 3.5/3.6/3.13).

The taste reference is the HAIRLINE TABLE: one or two short header lines,
then the table immediately; ONE thin rule under the header and above the
footer; NO vertical borders; merged "name (Class)" column with the
attribute name primary; text left-aligned, numerics right-aligned;
indentation for hierarchy; a labeled five-subject footer block; one
contextual "More:" line. Designed IN ASCII (the modal rendering);
unicode is an upgrade applied strictly through the declared glyph table.

The renderer is a pure function of typed data -- it never reads the trace
and never mints a number (raw ints arrive from FactCore projections).
No ANSI or OSC-8 byte can appear in any returned string (A11).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ._summary_charset import assert_no_escape_bytes
from ._summary_config import ColumnSpec, SummaryConfig
from ._summary_ladder import ResolvedView, ViewRow

#: Micro-bar width in characters (%params emphasis; degrade-map covered).
_BAR_WIDTH = 3


@dataclass(frozen=True)
class SummaryFacts:
    """Trace-side facts the renderer prints (gathered once, detached)."""

    model_class_name: str | None
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    backend: str | None
    outcome_status: str | None
    capture_verified: bool | None
    structure_only: bool
    health_verdict: str
    params_total: int | None
    params_per_path_total: int | None
    params_trainable: int | None
    params_frozen: int | None
    params_executed: int | None
    params_unexecuted: int | None
    unexecuted_names: tuple[str, ...]
    tied_groups: tuple[tuple[str, ...], ...]
    param_bytes: int | None
    buffer_count: int | None
    flops_forward_fma2: int
    macs_forward: int
    coverage_known: int
    coverage_zero: int
    coverage_unknown: int
    unknown_op_names: tuple[str, ...]
    # Pass-level peak facts arrive pre-gated through the ONE publication gate
    # (torchlens.observe._peaks.pass_peak_facts); the basis is always named
    # and a CUDA 0 reads as a high-water fact, never "used 0 bytes".
    peak_value_bytes: int | None
    peak_meaning: str
    at_capture_bytes: int
    retained_now_bytes: int
    compute_ops: int
    tracked_tensor_rows: int
    passes_max: int
    execution_note: str | None
    input_synthesis: str | None
    honesty_lines: tuple[str, ...] = ()


def _human(value: int | None) -> str:
    """Human-unit magnitude (K/M/G/T) for table cells."""

    if value is None:
        return "?"
    if value == 0:
        return "0"
    magnitude = float(value)
    for suffix in ("", "K", "M", "G", "T", "P"):
        if abs(magnitude) < 1000:
            if suffix == "":
                return str(value)
            return f"{magnitude:.2f}".rstrip("0").rstrip(".") + suffix
        magnitude /= 1000.0
    return f"{magnitude:.2f}E"


def _commas(value: int | None) -> str:
    """Full digits with thousands separators."""

    return "?" if value is None else f"{value:,}"


def _bytes_human(value: int | None) -> str:
    """Binary-ish byte formatting (decimal units, B suffix)."""

    if value is None:
        return "unavailable"
    if value == 0:
        return "0 B (measured)"
    magnitude = float(value)
    for suffix in ("B", "KB", "MB", "GB", "TB"):
        if abs(magnitude) < 1000:
            return f"{magnitude:.1f}".rstrip("0").rstrip(".") + f" {suffix}"
        magnitude /= 1000.0
    return f"{magnitude:.1f} PB"


def _format_number(value: int | None, units: str) -> str:
    """Route a count/flops cell through the configured unit style."""

    if value is None:
        return "?"
    return _human(value) if units == "human" else _commas(value)


def _micro_bar(fraction: float) -> str:
    """The %params micro-bar (unicode glyphs; ASCII via the degrade map)."""

    filled = round(max(0.0, min(1.0, fraction)) * _BAR_WIDTH)
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


def _name_cell(row: ViewRow) -> str:
    """The merged "name (Class)" cell with hierarchy indentation."""

    indent = "  " * max(row.depth - 1, 0)
    if row.kind == "fold":
        label = f"{row.name} ({row.class_name} ×{row.fold_count})"
    elif row.kind == "elision":
        label = f"{row.name} ({row.elided_count} rows elided)"
    elif row.kind == "op":
        label = row.name
    elif row.class_name:
        label = f"{row.name} ({row.class_name})"
    else:
        label = row.name
    if row.passes > 1 and row.kind in ("module", "coalesced", "op"):
        label += f" ×{row.passes} passes"
    if row.kind == "unexecuted":
        label += " [never ran]"
    if row.tied:
        label += " *"
    return indent + label


def _output_cell(row: ViewRow, config: SummaryConfig, params_total: int) -> str:
    """The output-shape cell; '-' when the shape is unknown."""

    del config, params_total
    if row.output_shape is None:
        return "-"
    return "(" + ", ".join(str(dim) for dim in row.output_shape) + ")"


def _params_cell(row: ViewRow, config: SummaryConfig, params_total: int) -> str:
    """The subtree-params cell; '-' when the row displays no params."""

    del params_total
    if row.params_display is None:
        return "-"
    return _format_number(row.params_display, config.units)


def _params_pct_cell(row: ViewRow, config: SummaryConfig, params_total: int) -> str:
    """The %params micro-bar cell over the whole-model total."""

    del config
    if row.params_display is None or not params_total:
        return "-"
    fraction = row.params_display / params_total
    return f"{_micro_bar(fraction)} {fraction * 100:.1f}%"


def _flops_cell(row: ViewRow, config: SummaryConfig, params_total: int) -> str:
    """The flops cell under the configured convention; honesty markers kept."""

    del params_total
    if row.evidence == "unknown" and row.flops_display is None:
        return "?"
    if row.flops_display is None:
        return "-"
    value = row.flops_display
    if config.flop_convention == "fma1":
        if row.macs_display is None:
            return "?"
        value = row.flops_display - row.macs_display
    return _format_number(value, config.units)


#: Per-column cell formatters (mirrors COLUMN_REGISTRY; closed dispatch).
_CELL_FORMATTERS: dict[str, Callable[[ViewRow, SummaryConfig, int], str]] = {
    "name": lambda row, config, params_total: _name_cell(row),
    "output": _output_cell,
    "params": _params_cell,
    "params_pct": _params_pct_cell,
    "flops": _flops_cell,
    "macs": lambda row, config, params_total: _format_number(row.macs_owned, config.units),
    "train": lambda row, config, params_total: {
        "all": "yes",
        "none": "no",
        "some": "partial",
    }.get(row.trainable or "", "-"),
    "passes": lambda row, config, params_total: str(row.passes),
    "evidence": lambda row, config, params_total: row.evidence,
}


def _cell(row: ViewRow, column: ColumnSpec, config: SummaryConfig, params_total: int) -> str:
    """One typed table cell: 0 is measured, '-' inapplicable, '?' unknown."""

    formatter = _CELL_FORMATTERS.get(column.name)
    if formatter is None:
        return "-"
    return formatter(row, config, params_total)


def _render_table(
    rows: tuple[ViewRow, ...],
    columns: tuple[ColumnSpec, ...],
    config: SummaryConfig,
    params_total: int,
) -> list[str]:
    """The hairline table: header row, one rule, aligned body rows."""

    header = [column.header for column in columns]
    body = [[_cell(row, column, config, params_total) for column in columns] for row in rows]
    widths = [
        max(len(header[i]), *(len(line[i]) for line in body)) if body else len(header[i])
        for i in range(len(columns))
    ]
    lines: list[str] = []
    cells = [
        header[i].ljust(widths[i]) if columns[i].align == "left" else header[i].rjust(widths[i])
        for i in range(len(columns))
    ]
    lines.append("  ".join(cells).rstrip())
    lines.append("─" * min(sum(widths) + 2 * (len(widths) - 1), 100))
    for line in body:
        cells = [
            line[i].ljust(widths[i]) if columns[i].align == "left" else line[i].rjust(widths[i])
            for i in range(len(columns))
        ]
        lines.append("  ".join(cells).rstrip())
    return lines


def _banner_lines(facts: SummaryFacts) -> list[str]:
    """Mandatory negative-status honesty banner, above the table.

    Consumes the ONE shared honesty chokepoint's lines (A09's
    ``honesty_banner_lines``), upgrading only the structure-only line to
    the pinned house phrasing every human surface uses.
    """

    lines: list[str] = []
    for line in facts.honesty_lines:
        if line.startswith("structure-only capture"):
            lines.append(
                "!! structure-only capture -- shapes/dtypes are HYPOTHESES, not measurements"
            )
        else:
            lines.append(f"!! {line}")
    if facts.input_synthesis:
        lines.append(f"synthetic input: {facts.input_synthesis}")
    return lines


def _params_footer(facts: SummaryFacts) -> str:
    """The Params footer subject (declared unique; tie named; A3 split)."""

    parts: list[str] = []
    if facts.params_total is not None and facts.params_per_path_total not in (
        None,
        facts.params_total,
    ):
        tie = ""
        if facts.tied_groups:
            tie = "; tied: " + " = ".join(facts.tied_groups[0])
            if len(facts.tied_groups) > 1:
                tie += f" (+{len(facts.tied_groups) - 1} more groups)"
        parts.append(
            f"{_commas(facts.params_total)} declared unique "
            f"({_commas(facts.params_per_path_total)} by module path{tie})"
        )
    else:
        parts.append(f"{_commas(facts.params_total)} declared")
    if facts.params_trainable is not None and facts.params_total:
        share = 100.0 * facts.params_trainable / facts.params_total
        parts.append(f"trainable {_commas(facts.params_trainable)} ({share:.0f}%)")
    if facts.params_unexecuted:
        names = ", ".join(facts.unexecuted_names[:3]) or "unknown"
        parts.append(f"{_commas(facts.params_unexecuted)} never ran: {names}")
    return " | ".join(parts)


def _compute_footer(facts: SummaryFacts, flop_convention: str = "fma2") -> str:
    """The Compute footer subject (two-term truth; lower-bound wording)."""

    if flop_convention == "fma1":
        total = facts.flops_forward_fma2 - facts.macs_forward
        flops = f"{_human(total)} FLOPs fwd (fma=1)"
    else:
        flops = f"{_human(facts.flops_forward_fma2)} FLOPs fwd (fma=2)"
    macs = f"{_human(facts.macs_forward)} MACs"
    known = facts.coverage_known + facts.coverage_zero
    denominator = known + facts.coverage_unknown
    coverage = f"{known}/{denominator} ops known (formula-exact)"
    if facts.coverage_unknown:
        names = ", ".join(facts.unknown_op_names[:3])
        more = max(facts.coverage_unknown - len(facts.unknown_op_names[:3]), 0)
        suffix = f" +{more} more" if more else ""
        coverage += (
            f"; totals are lower bounds ({names}{suffix} unknown; "
            "remedy: torchlens.capture.flops.register_op_rule)"
        )
    return " | ".join([flops, macs, coverage])


def render_summary_pair(
    view: ResolvedView,
    facts: SummaryFacts,
    config: SummaryConfig,
    *,
    visible_rows: tuple[ViewRow, ...] | None = None,
    filter_note: str | None = None,
) -> tuple[str, str]:
    """Render one resolved view as ``(ascii_text, unicode_text)``.

    ONE template: the unicode form is rendered and the canonical ASCII
    payload is its exact glyph-table degrade, so the two charsets cannot
    drift by construction (the CI degrade-map check re-asserts it).
    """

    from ._summary_charset import degrade

    rows = visible_rows if visible_rows is not None else view.rows
    lines: list[str] = []
    input_bits = ", ".join(
        f"{_shape_str(shape)} {dtype}"
        for shape, dtype in zip(facts.input_shapes, facts.input_dtypes, strict=False)
    )
    header = facts.model_class_name or "model"
    if input_bits:
        header += f" | input {input_bits}"
    if facts.execution_note:
        header += f" | {facts.execution_note}"
    lines.append(header)
    lines.extend(_banner_lines(facts))
    lines.append(view.disclosure + (f" | {filter_note}" if filter_note else ""))
    params_total = facts.params_total or 0
    lines.extend(
        _render_table(rows, config.resolved_columns(view.mixed_trainability), config, params_total)
    )
    lines.append("─" * min(len(lines[-1]) if lines else 80, 100))
    lines.append(f"Params   {_params_footer(facts)}")
    lines.append(f"Compute  {_compute_footer(facts, config.flop_convention)}")
    if facts.peak_value_bytes is None:
        memory = f"forward peak unavailable ({facts.peak_meaning})"
    else:
        memory = f"forward peak {_bytes_human(facts.peak_value_bytes)} ({facts.peak_meaning})"
    memory += (
        f" | activations at capture {_bytes_human(facts.at_capture_bytes)}"
        f", retained now {_bytes_human(facts.retained_now_bytes)}"
    )
    lines.append(f"Memory   {memory}")
    graph = (
        f"{facts.compute_ops} ops, {len(rows)} rows shown"
        f" | {facts.tracked_tensor_rows} tracked tensor rows"
    )
    if view.root_owned_ops:
        graph += f" | {len(view.root_owned_ops)} root-level ops in footer totals"
    if facts.passes_max > 1:
        graph += f" | up to {facts.passes_max} passes"
    lines.append(f"Graph    {graph}")
    capture_bits = [facts.backend or "?"]
    if facts.outcome_status:
        capture_bits.append(facts.outcome_status)
    if facts.capture_verified is True:
        capture_bits.append("verified")
    capture_bits.append(f"health {facts.health_verdict.upper().replace('_', '-')}")
    lines.append(f"Capture  {' | '.join(capture_bits)}")
    if any(row.tied for row in rows):
        lines.append("*        tied parameters (identity counted once; see Params)")
    lines.append(
        "More:    result.details() | view='compute' for per-row MACs | docs/reference/summary.md"
    )
    unicode_text = assert_no_escape_bytes("\n".join(lines))
    return degrade(unicode_text), unicode_text


def _shape_str(shape: tuple[int, ...]) -> str:
    """Render one shape tuple."""

    return "(" + ", ".join(str(dim) for dim in shape) + ")"
