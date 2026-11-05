"""Human-readable summaries of an ``Op``'s captured call arguments (``call_args``, ``call_kwargs``)."""

from __future__ import annotations

from typing import Any

import torch


def _summarize_value(value: Any) -> str:
    """Return a compact, human-readable string for one argument value."""

    if isinstance(value, torch.Tensor):
        shape = "x".join(str(int(d)) for d in value.shape) or "scalar"
        return f"Tensor({shape}, {str(value.dtype).replace('torch.', '')})"
    text = repr(value)
    if len(text) > 60:
        text = text[:57] + "..."
    return text


def _summarize_call_args(saved_args: Any, non_tensor_pos_args: Any) -> str | None:
    """Build a human-readable summary of an Op's positional arguments.

    Prefers the fully-saved positional args when available; otherwise falls
    back to the captured non-tensor positional args. Returns ``None`` when no
    positional-argument information was captured.
    """

    source = saved_args if saved_args is not None else non_tensor_pos_args
    if source is None:
        return None
    try:
        items = list(source)
    except TypeError:
        return _summarize_value(source)
    if not items:
        return ""
    return ", ".join(_summarize_value(item) for item in items)


def _summarize_call_kwargs(saved_kwargs: Any, non_tensor_kwargs: Any) -> str | None:
    """Build a human-readable summary of an Op's keyword arguments.

    Prefers the fully-saved keyword args when available; otherwise falls back to
    the captured non-tensor keyword args. Returns ``None`` when no
    keyword-argument information was captured.
    """

    source = saved_kwargs if saved_kwargs is not None else non_tensor_kwargs
    if source is None:
        return None
    if isinstance(source, dict):
        pairs = source.items()
    else:
        try:
            pairs = list(source)  # type: ignore[assignment]
        except TypeError:
            return _summarize_value(source)
    rendered = []
    for pair in pairs:
        try:
            key, value = pair
        except (TypeError, ValueError):
            rendered.append(_summarize_value(pair))
            continue
        rendered.append(f"{key}={_summarize_value(value)}")
    if not rendered:
        return ""
    return ", ".join(rendered)
