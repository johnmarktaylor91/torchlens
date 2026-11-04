"""Constructor-style displays for grouped options, without default-field noise."""

from __future__ import annotations

from dataclasses import MISSING, fields
from typing import Any

from ._display_card import REPR_COLS, render_card
from ._display_constructor import display_guard, value_text
from .fail_open import fail_open


class _OptionsDisplay:
    """Display mixin; no option values or explicit-field receipts are changed."""

    def __repr__(self) -> str:
        """Show only non-default fields in a bounded constructor call."""

        return fail_open(lambda: _options_repr(self), lambda _error: f"{type(self).__name__}(...)")

    @display_guard
    def __str__(self) -> str:
        """List every public option field beneath the constructor summary."""

        rows = [
            (name, _option_card_value(value))
            for name, value, _ in _option_fields(self)
            if value is not None
        ]
        exports = tuple(f".{name}()" for name in ("to_dict", "as_dict") if hasattr(self, name))
        more = exports or tuple(f".{name}" for name, _, _ in _option_fields(self)[:3])
        return render_card(repr(self), rows, more, max_rows=len(rows) + 1)


def _option_card_value(value: Any) -> str:
    """Use words for switches; constructor reprs retain Python literals."""

    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    return value_text(value)


def _option_fields(obj: Any) -> list[tuple[str, Any, Any]]:
    """Read declared defaults without constructing a second options object."""

    result = []
    for item in fields(obj):
        if item.name.startswith("_") or not item.repr:
            continue
        default = item.default
        if default is MISSING and item.default_factory is not MISSING:
            default = item.default_factory()
        result.append((item.name, getattr(obj, item.name), default))
    return result


def _same_default(value: Any, default: Any) -> bool:
    """Compare ordinary defaults without tensor equality or user-defined comparisons."""

    if value is default:
        return True
    if type(value) is not type(default):
        return False
    if isinstance(value, (str, int, float, bool, type(None))):
        return bool(value == default)
    if isinstance(value, (tuple, list, dict, set, frozenset)):
        return len(value) == 0 and len(default) == 0
    return False


def _options_repr(obj: Any) -> str:
    """Add only fields that fit, disclosing any omitted arguments."""

    head = type(obj).__name__
    parts: list[str] = []
    for name, value, default in _option_fields(obj):
        if _same_default(value, default):
            continue
        part = f"{name}={value_text(value)}"
        if len(f"{head}({', '.join([*parts, part])})") > REPR_COLS - 5:
            parts.append("...")
            break
        parts.append(part)
    return f"{head}({', '.join(parts)})"
