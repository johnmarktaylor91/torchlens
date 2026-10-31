"""Post-hoc narration: one renderer, three carriers (snoop D5, tail 1).

``Trace.narrate()``, ``PartialTrace.narrate()``, and ``Recording.narrate()``
all render the same grammar from the same formatter. This is the headline
crash workflow: it works when echo was NEVER turned on, because crashes are
not scheduled -- the partial the library already attaches to the exception
carries everything a line needs. Finished traces render FINAL labels
natively; partials render the raw labels that resolve on the partial.
"""

from __future__ import annotations

from typing import Any

from ._event import NarrationEvent
from ._format import compact_device, compact_dtype, render_line

__tl_layer__ = "L5"


def _op_row(op: Any, ordinal: int, depth: int) -> NarrationEvent:
    """Build one narration record from a finished/raw op-shaped entry."""

    shape = getattr(op, "shape", None)
    # Raw (pre-postprocess) entries carry only the raw label; finished traces
    # carry the final one. Tier-honest either way: print the spelling that
    # resolves on THIS carrier.
    label = (
        getattr(op, "label", None)
        or getattr(op, "layer_label", None)
        or getattr(op, "_label_raw", None)
        or getattr(op, "_layer_label_raw", None)
        or "?"
    )
    return NarrationEvent(
        kind="op",
        ordinal=ordinal,
        label=str(label),
        pass_index=int(getattr(op, "pass_index", 1) or 1),
        layer_type=getattr(op, "layer_type", None),
        func_name=getattr(op, "func_name", None),
        address=getattr(op, "address", None) or None,
        module_depth=depth,
        shape=tuple(shape) if shape is not None else None,
        dtype=compact_dtype(getattr(op, "dtype_ref", None) or getattr(op, "dtype", None)),
        device=compact_device(getattr(op, "device_ref", None)),
        tier="post_hoc",
    )


def _module_depth(op: Any) -> int:
    """Module nesting depth for indentation, tolerant of raw entries."""

    stack = getattr(op, "module_call_stack", None) or getattr(op, "modules", None) or ()
    try:
        return len(stack)
    except TypeError:
        return 0


def _select_rows(rows: list[NarrationEvent], select: Any) -> list[NarrationEvent]:
    """Filter narration rows with a selector/callable over row attributes."""

    if select is None:
        return rows

    def _matches(row: NarrationEvent) -> bool:
        """Evaluate the post-hoc filter on one rendered row."""

        try:
            return bool(select(row))
        except Exception:  # noqa: BLE001 - a non-evaluable row is a non-match, never a crash
            # Post-hoc selectors built for live RecordContexts may reach for
            # fields a row lacks; a non-evaluable row is a non-match, and the
            # substring convenience below stays available.
            return False

    if isinstance(select, str):
        return [row for row in rows if select in row.label or select == (row.address or "")]
    return [row for row in rows if _matches(row)]


def _render_block(
    rows: list[NarrationEvent],
    *,
    last: int | None,
    header: str | None,
    footer_lines: list[str],
) -> str:
    """Render a bounded, ordered block of narration rows."""

    if last is not None and last >= 0:
        rows = rows[-last:]
    lines: list[str] = []
    if header:
        lines.append(header)
    lines.extend(render_line(row) for row in rows)
    lines.extend(footer_lines)
    return "\n".join(lines)


def narrate_trace(
    trace: Any,
    *,
    last: int | None = None,
    select: Any = None,
) -> str:
    """Render a finished trace's events in the narration grammar.

    Parameters
    ----------
    trace:
        Finished ``Trace``. Final labels render natively.
    last:
        Keep only the last N rows (the tail idiom).
    select:
        Optional filter: a substring, or a callable over
        :class:`~torchlens.snoop.NarrationEvent` rows.

    Returns
    -------
    str
        Rendered narration block (no trailing newline).
    """

    rows = [
        _op_row(op, ordinal, _module_depth(op))
        for ordinal, op in enumerate(getattr(trace, "layer_list", ()) or (), start=1)
    ]
    footer: list[str] = []
    nonfinite = getattr(trace, "nonfinite_ops", None)
    if nonfinite:
        footer.append(f"-- first_nonfinite: {tuple(nonfinite)[0]} --")
    return _render_block(_select_rows(rows, select), last=last, header=None, footer_lines=footer)


def narrate_partial(
    partial: Any,
    *,
    last: int | None = None,
    select: Any = None,
) -> str:
    """Render a failed capture's recorded frontier in the narration grammar.

    Works with echo OFF: the partial the library attaches to the exception
    carries the recorded ops; the op that raised is ABSENT from the record
    (it never produced an output), and the footer says so rather than
    presenting the last recorded op as the culprit.
    """

    ops = tuple(getattr(partial, "raw_layers", ()) or ())
    rows = [_op_row(op, ordinal, _module_depth(op)) for ordinal, op in enumerate(ops, start=1)]
    exc = getattr(partial, "original_exception", None)
    footer = ["-- the raising call is not in the record: it produced no output to log --"]
    header = None
    if exc is not None:
        header = f"!! forward failed: {type(exc).__name__}: {exc}"
    try:
        first_nonfinite = partial.first_nonfinite()
    except Exception:  # noqa: BLE001 - footer evidence is best-effort on a failed partial
        first_nonfinite = None
    if first_nonfinite and "No non-finite" not in str(first_nonfinite):
        footer.append(f"-- first_nonfinite: {first_nonfinite} --")
    return _render_block(_select_rows(rows, select), last=last, header=header, footer_lines=footer)


def narrate_recording(
    recording: Any,
    *,
    last: int | None = None,
    select: Any = None,
) -> str:
    """Render a fastlog Recording's captured events in the narration grammar.

    Failed partial Recordings (``on_forward_error="attach_partial"`` /
    ``"return_partial"``) render their captured frontier with the failure
    header; the record-tier labels ARE the recording's own lookup keys.
    """

    rows: list[NarrationEvent] = []
    for ordinal, record in enumerate(getattr(recording, "records", ()) or (), start=1):
        ctx = getattr(record, "ctx", None)
        if ctx is None:
            continue
        kind = str(getattr(ctx, "kind", "op"))
        if kind not in ("op", "input", "buffer"):
            continue
        rows.append(
            NarrationEvent(
                kind=kind,
                ordinal=ordinal,
                label=str(ctx.label),
                pass_index=int(getattr(ctx, "pass_index", 1) or 1),
                step_index=getattr(ctx, "step_index", None),
                layer_type=getattr(ctx, "layer_type", None),
                func_name=getattr(ctx, "func_name", None),
                address=getattr(ctx, "address", None),
                module_depth=len(getattr(ctx, "module_stack", ()) or ()),
                shape=getattr(ctx, "shape", None),
                dtype=compact_dtype(getattr(ctx, "dtype", None)),
                device=compact_device(getattr(ctx, "tensor_device", None)),
                output_index=getattr(ctx, "output_index", None),
                tier="post_hoc",
            )
        )
    header = None
    footer: list[str] = []
    if getattr(recording, "failed", False):
        header = f"!! forward failed: {recording.error_repr}"
        footer.append("-- the raising call is not in the record: it produced no output to log --")
    elif getattr(recording, "halted", False):
        footer.append(f"-- capture halted (not failed): {recording.halt_reason} --")
    return _render_block(_select_rows(rows, select), last=last, header=header, footer_lines=footer)


__all__ = ["narrate_partial", "narrate_recording", "narrate_trace"]
