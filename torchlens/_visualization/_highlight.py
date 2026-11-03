"""``Trace.draw(highlight=...)``: named highlights routed through the lens doors.

Each allowed value names one existing lens door; this module only checks the
value and forwards the caller's explicit draw keywords. It never renders.
The allowed values are spelled once, in
:data:`torchlens._literals.DrawHighlightLiteral`.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, get_args

from .._errors import InvalidArgumentError
from .._literals import DrawHighlightLiteral

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

#: Allowed ``highlight=`` values, in the order the refusal lists them.
HIGHLIGHT_VALUES: tuple[str, ...] = get_args(DrawHighlightLiteral)

#: ``Trace.draw`` parameters that never forward to the lens door.
_NOT_FORWARDED = frozenset({"self", "trace", "highlight"})


def check_highlight(highlight: object) -> str:
    """Return ``highlight`` when it is an allowed value, else refuse typed.

    Raises
    ------
    InvalidArgumentError
        ``draw_highlight_invalid`` naming every allowed value.
    """

    if isinstance(highlight, str) and highlight in HIGHLIGHT_VALUES:
        return highlight
    allowed = ", ".join(repr(value) for value in HIGHLIGHT_VALUES)
    raise InvalidArgumentError(
        f"highlight must be one of {allowed}; received {highlight!r}",
        code="draw_highlight_invalid",
        remedy=f"pass highlight={HIGHLIGHT_VALUES[0]!r}, or leave highlight unset",
        argument="highlight",
    )


def _is_default(value: object, default: object) -> bool:
    """Whether ``value`` is the parameter's default (identity, or equal scalar)."""

    if value is default:
        return True
    if type(value) is type(default) and isinstance(default, (str, int, float, bool)):
        return value == default
    return False


def explicit_draw_kwargs(
    draw_fn: Callable[..., Any], call_locals: Mapping[str, Any]
) -> dict[str, Any]:
    """Collect the draw keywords whose value differs from the signature default.

    A lens door gives explicit keywords precedence over its members, so only
    the values the caller actually chose may forward; a default must not
    override a lens member.

    Parameters
    ----------
    draw_fn:
        The draw function whose signature names the parameters.
    call_locals:
        ``locals()`` captured at the top of that function, before any
        parameter is reassigned.

    Returns
    -------
    dict[str, Any]
        Parameter name to value, for every non-default parameter.
    """

    explicit: dict[str, Any] = {}
    for name, parameter in inspect.signature(draw_fn).parameters.items():
        if name in _NOT_FORWARDED or name not in call_locals:
            continue
        value = call_locals[name]
        if not _is_default(value, parameter.default):
            explicit[name] = value
    return explicit


def draw_highlighted(trace: Trace, highlight: object, user_kwargs: Mapping[str, Any]) -> Any:
    """Draw ``trace`` with the named highlight.

    Parameters
    ----------
    trace:
        The trace to draw.
    highlight:
        The requested highlight; checked against :data:`HIGHLIGHT_VALUES`.
    user_kwargs:
        The caller's explicit ``Trace.draw`` keywords.

    Returns
    -------
    Any
        What the highlight's lens door returns (the ``Trace.draw`` result).
    """

    check_highlight(highlight)
    # Only "interventions" exists today; a new value adds its door here.
    from .surgery_visuals import draw_interventions

    return draw_interventions(trace, user_kwargs)
