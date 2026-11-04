"""Metadata-only cards for selections and their resolved sites."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._utils._display_card import (
    bounded_lines,
    inline_names,
    more_row,
    one_line,
    render_card,
)
from .._utils._display_constructor import clipped, display_guard
from .._utils._display_keys import display_key
from .._utils._display_numbers import format_count, format_exact_count, plural
from .._utils._display_text import shape_text
from .._utils.fail_open import fail_open

if TYPE_CHECKING:
    from .selection import Selection, Site


def _selected(site: Site) -> int | None:
    """Count only structural masks; a dense mask would require a tensor read."""

    return None if site._mask.form == "dense" else site.num_selected


def _safe_selected(site: Site) -> int | None:
    """Guard metadata access without reading a dense mask."""

    return fail_open(lambda: _selected(site), lambda _error: None)


def _site_key(site: Site) -> str:
    """Find a cached operation by its stored key, without walking the trace."""

    if site.kind == "ACT" and len(site.key) == 2:
        trace = site._trace_ref() if site._trace_ref is not None else None
        index = trace._layer_dict_all_keys if trace is not None else {}
        label, pass_number = site.key
        record = index.get(f"{label}:{pass_number}")
        if record is None:
            record = index.get(label)
        if record is not None:
            return display_key(record)
        return "unresolved"
    if site.kind == "EDGE":
        return "edge occurrence"
    return ":".join(str(part) for part in site.key)


@display_guard
def site_repr(site: Site) -> str:
    """Keep site identity, selected count and provenance on one line."""

    def render() -> str:
        count = _selected(site)
        values = "values not computed" if count is None else _value_count(count)
        return clipped(
            f"Site {site.kind} {_site_key(site)}: {values}, relation {site.provenance.relation}"
        )

    return fail_open(render, lambda _error: f"Site {site.kind}: values not computed")


@display_guard
def site_card(site: Site) -> str:
    """Show exact structural counts and the stored provenance without materializing a mask."""

    count = fail_open(lambda: _selected(site), lambda _error: None)
    values = "not computed" if count is None else f"{format_exact_count(count)} selected"
    return render_card(
        site_repr(site),
        [
            ("shape", shape_text(site.shape)),
            ("values", values),
            ("relation", site.provenance.relation),
            ("source", site.provenance.source),
        ],
        (".key", ".mask", ".provenance", ".record"),
    )


@display_guard
def selection_repr(selection: Selection) -> str:
    """Display the concrete selection's kind, size and producing selector."""

    total = _total_selected(selection)
    n = len(selection)
    values = "values not computed" if total is None else _value_count(total)
    source = () if selection.selector is None else (f"from {selection.selector!r}",)
    return one_line(f"Selection {selection.kind}", [plural(n, "site"), values], source)


def _total_selected(selection: Selection) -> int | None:
    """Sum structural mask metadata without reading any dense mask."""

    counts = [_safe_selected(entry) for entry in selection._entries]
    return None if None in counts else sum(count for count in counts if count is not None)


def _value_count(count: int) -> str:
    """Use compact counts while retaining singular value grammar."""

    return f"{format_count(count)} {'value' if count == 1 else 'values'}"


@display_guard
def selection_card(selection: Selection) -> str:
    """Summarize three stored site names inline, without repeating Site cards."""

    total = _total_selected(selection)
    return render_card(
        selection_repr(selection),
        [
            ("sites", inline_names([_site_key(site) for site in selection._entries], limit=3)),
            ("values", "not computed" if total is None else format_exact_count(total)),
        ],
        (".labels", ".selector", ".to_pandas()"),
    )


@display_guard
def site_table_repr(table: Any) -> str:
    """Summarize an internal table without generating all of its labels."""

    n = len(table)
    return f"_SiteTable: {plural(n, 'site')}"


@display_guard
def site_table_card(table: Any) -> str:
    """Render only the bounded prefix of resolved records."""

    return "\n".join(
        [
            site_table_repr(table),
            *bounded_lines(table._sites, display_key),
            more_row((".labels()", ".to_pandas()")),
        ]
    )
