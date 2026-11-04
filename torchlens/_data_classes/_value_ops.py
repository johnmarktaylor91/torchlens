"""Operation and pass-aware layer text cards, built from captured facts."""

from __future__ import annotations

import math
from itertools import islice
from typing import Any

from .._utils._display_card import render_card
from .._utils._display_keys import display_key
from .._utils._display_numbers import format_duration, format_exact_count, format_iec_bytes
from .._utils.fail_open import fail_open
from ._value_state import (
    METADATA_ONLY,
    _count_words,
    edited_token,
    metadata_text,
    payload_facts,
    resolve_record,
    trace_of,
    value_line,
    value_words,
)
from .func_call_location import display_frame, short_source_path

_OP_MORE = (".out", ".parents", ".children", ".func_config", ".code_context", ".lookup_keys")
_LAYER_MORE = (".ops", ".parents", ".children", ".to_pandas()")


def _key(record: Any, label: str) -> str:
    """Resolve a neighbour's stored display key, never parse its full label."""

    neighbour = resolve_record(record, label)
    return display_key(neighbour) if neighbour is not None else str(label)


def _names(record: Any, labels: Any) -> str:
    """Render only the first three graph keys with the exact remainder."""

    shown = [_key(record, label) for label in islice(labels, 3)]
    remainder = len(labels) - len(shown)
    return ", ".join(shown) + (f", and {remainder} more" if remainder else "")


def _graph(op: Any) -> str:
    """Graph neighbours and the captured count of reachable model outputs."""

    parents = getattr(op, "parents", ()) or ()
    children = getattr(op, "children", ()) or ()
    parts = [_names(op, parents)] if parents else []
    parts.append(display_key(op))
    if children:
        parts.append(_names(op, children))
    graph = " -> ".join(parts)
    downstream = getattr(op, "output_descendants", None)
    if downstream is not None:
        count = len(downstream)
        outputs = getattr(trace_of(op), "output_layers", None)
        if outputs is not None:
            graph += f" (reaches {format_exact_count(count)} of {_count_words(len(outputs), 'model output')})"
        else:
            graph += f" ({_count_words(count, 'model output')} downstream)"
    return graph


def _function(op: Any, *, duration: bool = True) -> str:
    """Function, gradient class and duration in one fact row."""

    if getattr(op, "is_input", False):
        return ""
    parts = [getattr(op, "func_name", "") or ""]
    grad = getattr(op, "grad_fn_class_name", None)
    if grad and grad != "none":
        parts.append(f"gradient {grad}")
    elapsed = getattr(op, "func_duration", None)
    if duration and elapsed is not None:
        parts.append(format_duration(float(elapsed)))
    return ", ".join(part for part in parts if part)


def _source(op: Any) -> str:
    """Captured source location only; never consult source files or linecache."""

    frame = display_frame(getattr(op, "code_context", None))
    if frame is None:
        return ""
    return f"{short_source_path(str(frame.file))}:{frame.line_number} in {frame.func_qualname or frame.func_name}"


def _code(op: Any) -> str:
    """Use the fingerprint-checked source helper, retaining missing/edited reasons."""

    frame = display_frame(getattr(op, "code_context", None))
    if frame is None:
        return ""
    view, reason = frame._source_text_state()
    return str(view[3]).strip() if view is not None else f"({reason})"


def _module(op: Any) -> str:
    """Stored module address without a redundant single-call suffix."""

    module = getattr(op, "atomic_module_address", None)
    if module:
        return str(module)
    modules = getattr(op, "modules", ()) or ()
    if modules and isinstance(modules[-1], tuple):
        return str(modules[-1][0])
    label = modules[-1] if modules else getattr(op, "module", None)
    trace = trace_of(op)
    call = fail_open(lambda: trace.module_calls[label], lambda _error: None)
    return str(getattr(call, "address", None) or label or "")


def _output(op: Any) -> str:
    """Captured shape count and byte quantity without a payload read."""

    shape = getattr(op, "shape", None)
    parts = [_count_words(math.prod(shape), "value")] if shape is not None else []
    memory = getattr(op, "activation_memory", None)
    if memory is not None:
        parts.append(format_iec_bytes(int(memory)))
    return ", ".join(parts)


def _position(op: Any, kind: str) -> str:
    """Position belongs to the card, not the one-line identity."""

    trace = trace_of(op)
    total = trace._num_ops if trace is not None else None
    index = getattr(op, "step_index", None)
    if index is None or total is None:
        return ""
    return f"{kind} {format_exact_count(index)} of {format_exact_count(total)}"


def _op_repr_line(op: Any, *, kind: str = "Op", key: str | None = None) -> str:
    """Bounded value-bearing identity with no model prefix or positional suffix."""

    metadata, stats, honesty = payload_facts(op)
    honesty = " ".join(part for part in (honesty, edited_token(op)) if part)
    return value_line(f"{kind} {key or display_key(op)}", metadata, stats, honesty)


def op_rows(op: Any, *, kind: str = "operation", source_text: bool = True) -> list[tuple[str, str]]:
    """Shared text/HTML facts in the house-style row order."""

    _, stats, honesty = payload_facts(op)
    rows = [
        ("position", _position(op, kind)),
        ("module", _module(op)),
        ("output", _output(op)),
    ]
    if not honesty:
        rows.append(("values", value_words(stats)))
    rows.extend(
        (
            ("graph", _graph(op)),
            ("function", _function(op)),
            ("source", _source(op)),
            ("code", _code(op) if source_text else ""),
        )
    )
    return rows


def _op_card(op: Any) -> str:
    """At most ten fact rows, ending in the named exits."""

    return render_card(op_repr_line(op), op_rows(op), _OP_MORE)


def _first_ops(layer: Any) -> list[Any]:
    """Bound before rendering even on a many-pass Layer accessor."""

    return list(islice(getattr(layer, "ops", ()) or (), 3))


def _layer_repr_line(layer: Any) -> str:
    """Single-pass value or multi-pass count; never a pooled statistic."""

    ops = _first_ops(layer)
    passes = getattr(layer, "num_passes", 1)
    key = display_key(layer)
    if passes > 1:
        metadata = metadata_text(ops[0]) if ops else METADATA_ONLY
        return f"Layer {key}: {format_exact_count(passes)} passes, {metadata}"
    if ops:
        return op_repr_line(ops[0], kind="Layer", key=key)
    return f"Layer {key}: {METADATA_ONLY}"


def layer_rows(layer: Any) -> list[tuple[str, str]]:
    """Per-pass value rows for reused layers, otherwise the operation facts."""

    ops = _first_ops(layer)
    passes = getattr(layer, "num_passes", 1)
    if passes <= 1:
        # Source text belongs to the Op card; Layer keeps its eight-line contract.
        return op_rows(ops[0], source_text=False) if ops else []
    rows = []
    for op in ops:
        _, stats, honesty = payload_facts(op)
        value = honesty or value_words(stats)
        edited = edited_token(op)
        rows.append((f"pass {getattr(op, 'pass_number', '?')}", f"{value} {edited}".strip()))
    if passes > len(ops):
        rows.append(("", f"... and {format_exact_count(passes - len(ops))} more passes"))
    if ops:
        rows.extend((("module", _module(ops[0])), ("function", _function(ops[0], duration=False))))
    return rows


def _layer_card(layer: Any) -> str:
    """A bounded Layer card, with no pooled summary for repeated passes."""

    return render_card(layer_repr_line(layer), layer_rows(layer), _LAYER_MORE)


def _placeholder(record: Any, kind: str) -> str:
    """Metadata survives missing trace bindings without printing a positional label."""

    key = fail_open(lambda: display_key(record), lambda _error: "?")
    metadata = fail_open(lambda: metadata_text(record), lambda _error: "metadata unavailable")
    state = "(detached)" if trace_of(record) is None else "(capture not finished)"
    return f"{kind} {key}: {metadata} {state}"


def _ready(record: Any) -> bool:
    """A finished binding is required for graph facts and layer pass iteration."""

    return trace_of(record) is not None and bool(getattr(record, "_tracing_finished", False))


def op_repr_line(op: Any, *, kind: str = "Op", key: str | None = None) -> str:
    """Never-raise Op identity, retaining detached and unfinished states."""

    return fail_open(
        lambda: _op_repr_line(op, kind=kind, key=key) if _ready(op) else _placeholder(op, kind),
        lambda _error: _placeholder(op, kind),
    )


def op_card(op: Any) -> str:
    """Never-raise Op card; detached records retain the repr as their whole card."""

    return fail_open(
        lambda: _op_card(op) if _ready(op) else op_repr_line(op), lambda _error: op_repr_line(op)
    )


def layer_repr_line(layer: Any) -> str:
    """Never-raise Layer identity, without reading ambiguous per-pass fields."""

    return fail_open(
        lambda: _layer_repr_line(layer) if _ready(layer) else _placeholder(layer, "Layer"),
        lambda _error: _placeholder(layer, "Layer"),
    )


def layer_card(layer: Any) -> str:
    """Never-raise Layer card with a visible detached fallback."""

    return fail_open(
        lambda: _layer_card(layer) if _ready(layer) else layer_repr_line(layer),
        lambda _error: layer_repr_line(layer),
    )
