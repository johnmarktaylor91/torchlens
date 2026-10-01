"""The envelope grammar and card assembly (F10; lovely memo 4.2 / D15).

The CORE (rendered by :mod:`._stats_render`) is byte-identical on every
surface; the ENVELOPE is the provenance affix supplied by whatever
surrounds it::

    <trace>/<address>:<pass> [<kind> <position> <function>] -> <core>

``repr(x)`` is ONE nestable line; ``str(x)`` is a bounded card whose FIRST
line IS the repr (D15). Cards never exceed :data:`CARD_MAX_LINES` lines,
every card names its exits on a ``More:`` line (voice rule 6), and
collections point instead of dumping (voice rule 1). Pure string
functions: nothing here touches a tensor or a record.

Spellings are DOCUMENTED-UNSTABLE pending naming-session ratification.
"""

from __future__ import annotations

#: Voice rule 9 budgets: repr one line <= 120 columns (target 100); record
#: cards <= 8 lines; collection views <= 24 lines; collections show at most
#: 5 children plus an exact remainder.
REPR_MAX_COLS = 120
REPR_TARGET_COLS = 100
CARD_MAX_LINES = 8
COLLECTION_MAX_LINES = 24
COLLECTION_MAX_CHILDREN = 5


def envelope_line(
    core: str,
    *,
    trace_token: str | None = None,
    address: str | None = None,
    pass_token: str | None = None,
    bracket: tuple[str | None, ...] = (),
) -> str:
    """Compose one envelope+core line (memo 4.2 grammar).

    Parameters
    ----------
    core:
        The rendered core stats line (or an honesty token standing in for
        it, e.g. ``(not saved)``).
    trace_token:
        Short owning-trace identity (model class name); omitted when the
        record is detached.
    address:
        Site address (layer label / module address / param address).
    pass_token:
        Pass tag ``k/n``; rendered as ``(pass k/n)`` after the address
        (the A10 vocabulary law: multiplicity spells passes, never ops).
    bracket:
        The ``[<kind> <position> <function>]`` affix of the memo 4.2
        grammar, as one triple in that order (trailing entries optional;
        ``None``/empty parts are omitted). Width overflow degrades through
        the core's own retention order by re-rendering upstream, never
        here.

    Returns
    -------
    str
        One line, no newlines, ASCII-clean if the core is.
    """

    head = ""
    if trace_token and address:
        head = f"{trace_token}/{address}"
    elif address:
        head = address
    elif trace_token:
        head = trace_token
    if pass_token:
        head = f"{head} (pass {pass_token})"
    bracket_parts = [part for part in bracket if part]
    bracket_text = f"[{' '.join(bracket_parts)}]" if bracket_parts else ""
    parts = [part for part in (head, bracket_text) if part]
    prefix = " ".join(parts)
    line = f"{prefix} -> {core}" if prefix else core
    return line


def record_card(
    repr_line: str,
    body_lines: list[str],
    *,
    more: tuple[str, ...] = (),
    max_lines: int = CARD_MAX_LINES,
) -> str:
    """Assemble a bounded record card (D15: line 1 IS the repr).

    Body lines are indented two spaces; when the budget would overflow,
    body lines are dropped from the END with an exact remainder line
    (never a silent truncation), and the ``More:`` exits line is always
    retained (state precedes detail, voice rule 7 -- callers put state
    lines FIRST in ``body_lines``).

    Parameters
    ----------
    repr_line:
        The one-line repr (card line 1, un-indented).
    body_lines:
        Detail lines in priority order (kept from the front).
    more:
        Exit spellings for the ``More:`` line (2-4 accessors).
    max_lines:
        Total line budget including the repr and ``More:`` lines.

    Returns
    -------
    str
        The assembled card.
    """

    more_line = f"  More: {'  '.join(more)}" if more else None
    budget = max_lines - 1 - (1 if more_line else 0)
    kept = list(body_lines)
    if len(kept) > budget:
        shown = max(budget - 1, 0)
        dropped = len(kept) - shown
        kept = kept[:shown] + [f"... {dropped} more lines (see More: exits)"]
    lines = [repr_line] + [f"  {line}" for line in kept]
    if more_line:
        lines.append(more_line)
    return "\n".join(lines)


def collection_line(
    type_name: str,
    count: int,
    noun: str,
    *,
    composition: str | None = None,
    exits: tuple[str, ...] = (),
) -> str:
    """One-line composition card for a collection (voice rule 1).

    Parameters
    ----------
    type_name:
        Accessor/collection class name.
    count:
        Exact member count.
    noun:
        Member noun, already pluralized by the caller when needed.
    composition:
        Optional composition breakdown (``539 ops, 14 buffers``).
    exits:
        Accessor exits named on the line.

    Returns
    -------
    str
        ``LayerAccessor(553 layers (539 ops, 14 buffers) | .head() ...)``.
    """

    inner = f"{count} {noun}"
    if composition:
        inner = f"{inner} ({composition})"
    if exits:
        inner = f"{inner} | {' '.join(exits)}"
    return f"{type_name}({inner})"


def bounded_children(
    items: list[str],
    *,
    max_children: int = COLLECTION_MAX_CHILDREN,
) -> list[str]:
    """Bound a child-line list with an exact remainder (voice rule 9).

    Parameters
    ----------
    items:
        Child one-liners.
    max_children:
        Maximum children shown.

    Returns
    -------
    list[str]
        At most ``max_children`` lines plus one exact remainder line.
    """

    if len(items) <= max_children:
        return items
    remainder = len(items) - max_children
    return items[:max_children] + [f"... {remainder} more"]
