"""Dependency-free summary HTML renderer (F08; summary memo item 15).

One escaped, self-contained fragment: sticky header, numeric right
alignment, dark-mode-safe colors via ``prefers-color-scheme``, zero
JavaScript, zero CDN. Interactivity plumbing for the deferred notebook
wave rides as ``data-*`` attributes (stable row IDs, depth, kind, fold
and ownership metadata) so wave-2 sort/expand needs no schema change.

All values arrive from the typed report (raw ints); malicious content in
model/module/op names is neutralized by escaping at this one boundary.
"""

from __future__ import annotations

import html

from ._summary_config import SummaryConfig
from ._summary_ladder import ResolvedView, ViewRow
from ._summary_render import SummaryFacts, _cell

_CSS = """
<style>
.tl-summary {font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12.5px; line-height: 1.45; color: #1a1a2e;}
.tl-summary table {border-collapse: collapse; width: auto;}
.tl-summary thead th {position: sticky; top: 0; background: #f4f4f8;
  text-align: left; padding: 3px 10px; border-bottom: 1px solid #888;}
.tl-summary td {padding: 1px 10px; white-space: pre;}
.tl-summary td.num, .tl-summary th.num {text-align: right;}
.tl-summary tr[data-kind="fold"] td {font-weight: 600;}
.tl-summary tr[data-kind="elision"] td {font-style: italic; opacity: 0.85;}
.tl-summary tr[data-kind="unexecuted"] td {opacity: 0.6;}
.tl-summary .tl-footer {margin-top: 6px; border-top: 1px solid #888;
  padding-top: 4px; white-space: pre;}
.tl-summary .tl-banner {color: #8a2d0b; font-weight: 600; white-space: pre;}
.tl-summary .tl-disclosure {opacity: 0.75;}
@media (prefers-color-scheme: dark) {
  .tl-summary {color: #e6e6ef;}
  .tl-summary thead th {background: #26263a; border-bottom-color: #666;}
  .tl-summary .tl-banner {color: #ffb454;}
}
</style>
"""


def _row_html(row: ViewRow, config: SummaryConfig, columns: tuple, params_total: int) -> str:
    """One escaped <tr> with data-* metadata for the deferred interactivity."""

    cells = []
    for column in columns:
        text = html.escape(_cell(row, column, config, params_total))
        css = ' class="num"' if column.align == "right" else ""
        cells.append(f"<td{css}>{text}</td>")
    attributes = (
        f' data-row-id="{html.escape(row.row_id, quote=True)}"'
        f' data-kind="{html.escape(row.kind, quote=True)}"'
        f' data-depth="{row.depth}"'
        f' data-fold-count="{row.fold_count}"'
        f' data-owned-ops="{len(row.owned_ops)}"'
    )
    return f"<tr{attributes}>{''.join(cells)}</tr>"


def render_html(  # noqa: PLR0913 - the four presentation slices are distinct render inputs
    view: ResolvedView,
    facts: SummaryFacts,
    config: SummaryConfig,
    *,
    visible_rows: tuple[ViewRow, ...] | None = None,
    footer_lines: tuple[str, ...] = (),
    banner_lines: tuple[str, ...] = (),
    header_line: str = "",
) -> str:
    """Render the summary as one self-contained escaped HTML fragment."""

    rows = visible_rows if visible_rows is not None else view.rows
    columns = config.resolved_columns(view.mixed_trainability)
    params_total = facts.params_total or 0
    numeric_class = ' class="num"'
    head_cells = "".join(
        f"<th{numeric_class if column.align == 'right' else ''}>{html.escape(column.header)}</th>"
        for column in columns
    )
    body = "".join(_row_html(row, config, columns, params_total) for row in rows)
    banner = "".join(f'<div class="tl-banner">{html.escape(line)}</div>' for line in banner_lines)
    footer = "".join(f'<div class="tl-footer">{html.escape(line)}</div>' for line in footer_lines)
    return (
        f'{_CSS}<div class="tl-summary">'
        f"<div><b>{html.escape(header_line)}</b></div>{banner}"
        f'<div class="tl-disclosure">{html.escape(view.disclosure)}</div>'
        f"<table><thead><tr>{head_cells}</tr></thead>"
        f"<tbody>{body}</tbody></table>{footer}</div>"
    )
