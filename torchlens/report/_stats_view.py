"""The ``summary(view="stats")`` payload: the per-site observation table as a report.

The per-op payload statistics are one view of :meth:`Trace.summary`: the
rows come from :mod:`._stats_table` (typed row states, per-family evidence,
the scan budget), the text is a fixed-width table whose header echoes the
basis, the scan-cost policy, the row order and the budget, and the result is
a :class:`SummaryReport` whose ``str()`` is that table and whose
``to_pandas()`` serves the rows. Honesty rows (``unsaved``, ``disk_backed``,
``unsupported``, ``unscanned``) are never dropped by a column choice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from ._stats_table import StatsTable, StatsTableRow
    from ._summary_config import SummaryConfig
    from ._summary_report import SummaryReport

#: Columns of the stats view in render order; ``name`` and ``state`` always render.
STATS_COLUMNS: tuple[str, ...] = (
    "name",
    "state",
    "n",
    "mean",
    "sd",
    "min",
    "max",
    "zero_count",
    "nan_count",
    "inf_count",
    "same_as_parent",
    "reason",
)

#: The default column bundle of the stats view.
STATS_BUNDLE: tuple[str, ...] = (
    "name",
    "state",
    "n",
    "mean",
    "sd",
    "min",
    "max",
    "zero_count",
    "nan_count",
    "inf_count",
    "same_as_parent",
)

_HEADINGS: dict[str, str] = {
    "name": "name",
    "state": "state",
    "n": "n",
    "mean": "mean",
    "sd": "sd",
    "min": "min",
    "max": "max",
    "zero_count": "zeros",
    "nan_count": "nan",
    "inf_count": "inf",
    "same_as_parent": "= parent",
    "reason": "reason",
}

_RIGHT_ALIGNED = frozenset(
    {"n", "mean", "sd", "min", "max", "zero_count", "nan_count", "inf_count"}
)


def resolve_stats_columns(columns: Any) -> tuple[str, ...]:
    """Resolve ``columns=`` for the stats view: a list or ``+name``/``-name`` deltas."""

    if columns is None:
        return STATS_BUNDLE
    if isinstance(columns, str):
        if columns == "stats":
            return STATS_BUNDLE
        _refuse_column(columns)
    names = list(columns)
    if names and all(isinstance(item, str) and item[:1] in "+-" for item in names):
        resolved = list(STATS_BUNDLE)
        for item in names:
            name = item[1:]
            if name not in STATS_COLUMNS:
                _refuse_column(name)
            if item[0] == "+" and name not in resolved:
                resolved.append(name)
            if item[0] == "-" and name in resolved:
                resolved.remove(name)
        return _with_identity(tuple(resolved))
    for item in names:
        if not isinstance(item, str) or item not in STATS_COLUMNS:
            _refuse_column(item)
    return _with_identity(tuple(names))


def _with_identity(columns: tuple[str, ...]) -> tuple[str, ...]:
    """``name`` and ``state`` always render: a row without them hides its honesty."""

    head = tuple(name for name in ("name", "state") if name not in columns)
    return head + columns


def _refuse_column(item: Any) -> None:
    raise InvalidArgumentError(
        f"summary(view='stats') has no column {item!r}; the stats columns are "
        f"{', '.join(STATS_COLUMNS)}.",
        code="summary_option_invalid",
        remedy="pass columns= as a list of stats column names or +name/-name deltas",
    )


def _number(value: Any, *, sampled: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        text = f"{value:.4g}"
    else:
        text = str(value)
    return f"~{text}" if sampled else text


def _cell(row: StatsTableRow, column: str) -> str:
    """One rendered cell; sampled families carry the ``~`` mark, never a clearance."""

    stats = row.stats
    if column == "name":
        return row.label
    if column == "state":
        return row.state
    if column == "same_as_parent":
        return row.parent_mark or ""
    if column == "reason":
        return row.reason or ""
    if stats is None:
        return ""
    if column == "n":
        return _number(stats.numel)
    if column == "mean":
        return _number(stats.mean, sampled=stats.mean_evidence.is_sampled)
    if column == "sd":
        return _number(stats.std, sampled=stats.std_evidence.is_sampled)
    if column == "min":
        return _number(stats.finite_min)
    if column == "max":
        return _number(stats.finite_max)
    if column == "zero_count":
        return _number(stats.num_zero)
    if column == "nan_count":
        return _number(stats.num_nan)
    inf = _inf_count(stats)
    return "" if inf is None else _number(inf)


def _inf_count(stats: Any) -> int | None:
    """``posinf + neginf`` when both counts are claimed, else no claim."""

    posinf = getattr(stats, "num_posinf", None)
    neginf = getattr(stats, "num_neginf", None)
    if posinf is None or neginf is None:
        return None
    return int(posinf) + int(neginf)


def render_stats_text(table: StatsTable, columns: tuple[str, ...], *, grain: str) -> str:
    """The fixed-width stats table with its disclosure header."""

    header = [
        f"Payload statistics ({grain} grain; {table.ordering})",
        f"basis: {table.basis}",
        f"scan policy: {table.scan_cost_policy}",
    ]
    for note in (table.budget_note, table.sort_disclosure, table.fold_note):
        if note:
            header.append(note)
    states: dict[str, int] = {}
    for row in table.rows:
        states[row.state] = states.get(row.state, 0) + 1
    header.append(
        f"rows: {len(table.rows)} ("
        + ", ".join(f"{state}={count}" for state, count in sorted(states.items()))
        + ")"
    )
    grid = [[_HEADINGS[column] for column in columns]]
    grid.extend([_cell(row, column) for column in columns] for row in table.rows)
    widths = [max(len(line[index]) for line in grid) for index in range(len(columns))]
    lines = []
    for line in grid:
        cells = []
        for index, column in enumerate(columns):
            text = line[index]
            width = widths[index]
            cells.append(text.rjust(width) if column in _RIGHT_ALIGNED else text.ljust(width))
        lines.append("  ".join(cells).rstrip())
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(header) + "\n\n" + "\n".join(lines)


def stats_rows_markdown(table: StatsTable, columns: tuple[str, ...]) -> str:
    """GitHub-flavored markdown of the rendered stats rows."""

    head = "| " + " | ".join(_HEADINGS[column] for column in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(_cell(row, column).replace("|", "\\|") for column in columns) + " |"
        for row in table.rows
    ]
    return "\n".join([head, rule, *body])


def build_stats_summary(trace: Trace, config: SummaryConfig) -> SummaryReport:
    """Build the ``view="stats"`` report over one finished trace."""

    from ._stats_table import build_stats_table
    from ._summary_report import SummaryReport, build_summary_data

    grain = "module" if config.level == "module" else "op"
    table = build_stats_table(trace, max_scan_elements=config.max_scan_elements, grain=grain)
    columns = resolve_stats_columns(config.columns)
    text = render_stats_text(table, columns, grain=grain)
    rows, totals, capture = build_summary_data(trace)
    return SummaryReport(
        text,
        rows=rows,
        totals=totals,
        capture=capture,
        stats=table,
        stats_columns=columns,
    )


__all__ = [
    "STATS_BUNDLE",
    "STATS_COLUMNS",
    "build_stats_summary",
    "render_stats_text",
    "resolve_stats_columns",
    "stats_rows_markdown",
]
