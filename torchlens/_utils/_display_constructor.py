"""Bounded, metadata-only arguments for constructor-style displays."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from itertools import islice
from typing import Any, ParamSpec

import torch

from ._display_card import INLINE_LIMIT, REPR_COLS
from ._display_text import tensor_stand_in
from .fail_open import fail_open

P = ParamSpec("P")


def clipped(text: str, width: int = REPR_COLS) -> str:
    """Keep a display on one line within its column budget."""

    text = text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 3] + "..."


def value_text(value: Any, depth: int = 0) -> str:
    """Render arguments without invoking arbitrary reprs or reading tensor values."""

    return fail_open(lambda: _value_text(value, depth), lambda _error: f"<{type(value).__name__}>")


def _value_text(value: Any, depth: int) -> str:
    """Format scalars and a bounded prefix of ordinary containers."""

    if type(value).__module__ == "torchlens.intervention.types":
        kind = type(value).__name__
        if kind in {"LiteralTensor", "LiteralValue"}:
            return value_text(value.value, depth)
        if kind == "ParentRef":
            return clipped(getattr(value, "_display_key", value.parent_label), 80)
    if isinstance(value, torch.Tensor):
        return tensor_stand_in(value)
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return clipped(repr(value), 80)
    if depth >= INLINE_LIMIT:
        return f"<{type(value).__name__}>"
    if isinstance(value, (tuple, list, dict)):
        return _container_text(value, depth)
    if callable(value):
        name = getattr(value, "__qualname__", type(value).__name__)
        return f"<function {clipped(name, 60)}>"
    return f"<{type(value).__name__}>"


def _container_text(value: tuple[Any, ...] | list[Any] | dict[Any, Any], depth: int) -> str:
    """Bound before rendering container members, including recursive containers."""

    if isinstance(value, dict):
        parts = [
            f"{value_text(key, depth + 1)}: {value_text(item, depth + 1)}"
            for key, item in islice(value.items(), INLINE_LIMIT)
        ]
        left, right = "{", "}"
    else:
        parts = [value_text(item, depth + 1) for item in value[:INLINE_LIMIT]]
        left, right = ("(", ")") if isinstance(value, tuple) else ("[", "]")
    if len(value) > INLINE_LIMIT:
        parts.append(f"... and {len(value) - INLINE_LIMIT} more")
    inner = ", ".join(parts)
    if isinstance(value, tuple) and len(value) == 1:
        inner += ","
    return f"{left}{inner}{right}"


def display_guard(render: Callable[P, str]) -> Callable[P, str]:
    """Keep a display-only failure from escaping into a notebook or traceback."""

    @wraps(render)
    def guarded(*args: P.args, **kwargs: P.kwargs) -> str:
        return fail_open(
            lambda: render(*args, **kwargs),
            lambda _error: f"{type(args[0]).__name__}: display unavailable",
        )

    return guarded
