"""Card and one-line layout shared by every printout (house style v1).

``repr(x)`` is one line of at most :data:`REPR_COLS` columns that starts with
what the object is. ``str(x)`` is a card: a first line, then labelled rows
``  label     value`` (label padded to :data:`LABEL_WIDTH`), then a ``more``
row naming the attributes that hold the rest. Collections show at most
:data:`LIST_LIMIT` members then ``... and N more``. Pure string functions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

T = TypeVar("T")

#: One-line repr width target (hard).
REPR_COLS = 100
#: Width of the label column in card rows.
LABEL_WIDTH = 9
#: Most body rows on a record card (the Trace card may use 12).
CARD_MAX_ROWS = 10
TRACE_CARD_MAX_ROWS = 12
#: Members a collection's print lists before ``... and N more``.
LIST_LIMIT = 5
#: Names an inline list inside a card row shows before ``and N more``.
INLINE_LIMIT = 3

_INDENT = "  "


def card_row(label: str, value: str, *, width: int = LABEL_WIDTH) -> str:
    """One labelled card row: ``  output    15,360 values, 60 KiB``."""

    return f"{_INDENT}{label:<{width}} {value}".rstrip()


def continuation_row(value: str, *, width: int = LABEL_WIDTH) -> str:
    """A row continuing the previous label's value, aligned under it."""

    return f"{_INDENT}{'':<{width}} {value}".rstrip()


def more_row(attributes: Sequence[str], *, width: int = LABEL_WIDTH) -> str:
    """The closing ``more`` row: ``  more      .out  .parents``."""

    return card_row("more", "  ".join(attributes), width=width)


def render_card(
    first_line: str,
    rows: Iterable[tuple[str, str]],
    more: Sequence[str] = (),
    *,
    max_rows: int = CARD_MAX_ROWS,
) -> str:
    """Assemble a card: first line, at most ``max_rows`` labelled rows, ``more`` row.

    Rows with an empty value are skipped. When rows are cut, the last kept row
    is replaced by ``... and N more rows`` so the cut is visible.
    """

    kept = [(label, value) for label, value in rows if value]
    width = max((len(label) for label, _ in kept), default=LABEL_WIDTH)
    width = max(LABEL_WIDTH, width)
    body_budget = max_rows - (1 if more else 0)
    lines = [first_line]
    if len(kept) > body_budget:
        hidden = len(kept) - body_budget + 1
        kept = kept[: body_budget - 1]
        lines.extend(card_row(label, value, width=width) for label, value in kept)
        lines.append(continuation_row(f"... and {hidden} more rows", width=width))
    else:
        lines.extend(card_row(label, value, width=width) for label, value in kept)
    if more:
        lines.append(more_row(more, width=width))
    return "\n".join(lines)


def one_line(head: str, facts: Sequence[str], optional: Sequence[str] = ()) -> str:
    """``Head: fact, fact, optional...`` within :data:`REPR_COLS` columns.

    Required ``facts`` always print; ``optional`` facts are dropped from the
    end until the line fits.
    """

    kept_optional = list(optional)
    while True:
        parts = [part for part in (*facts, *kept_optional) if part]
        line = f"{head}: {', '.join(parts)}" if parts else head
        if len(line) <= REPR_COLS or not kept_optional:
            return line
        kept_optional.pop()


def inline_names(names: Sequence[str], limit: int = INLINE_LIMIT) -> str:
    """``tanh_1, tanh_2, tanh_3, and 9 more`` (all names when at most ``limit``)."""

    shown = ", ".join(names[:limit])
    remainder = len(names) - limit
    return f"{shown}, and {remainder} more" if remainder > 0 else shown


def bounded_lines(
    items: Sequence[T],
    render: Callable[[T], str],
    *,
    limit: int = LIST_LIMIT,
    indent: str = _INDENT,
) -> list[str]:
    """Render at most ``limit`` items, then ``... and N more``.

    Only the shown items are rendered: the cut happens before formatting.
    """

    lines = [f"{indent}{render(item)}" for item in items[:limit]]
    remainder = len(items) - limit
    if remainder > 0:
        lines.append(f"{indent}... and {remainder} more")
    return lines
