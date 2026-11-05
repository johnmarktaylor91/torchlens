"""``Trace.draw(lens=...)``: the public door to the premade drawing styles.

Each allowed value names one registered lens row; this module checks the
value and forwards the caller's explicit draw keywords to the lens resolver.
It never renders. The allowed values are spelled once, in
:data:`torchlens._literals.DrawLensLiteral`.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, get_args

from .._errors import InvalidArgumentError
from .._literals import DrawLensLiteral

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

#: Allowed ``lens=`` values, in the order the refusal lists them.
LENS_VALUES: tuple[str, ...] = get_args(DrawLensLiteral)

#: ``Trace.draw`` parameters that never forward to the lens resolver.
_NOT_FORWARDED = frozenset({"self", "trace", "lens"})


def check_lens(lens: object) -> str:
    """Return ``lens`` when it names a public lens, else refuse typed.

    Parameters
    ----------
    lens:
        The requested lens name.

    Returns
    -------
    str
        The lens name, unchanged.

    Raises
    ------
    InvalidArgumentError
        ``draw_lens_invalid`` naming every allowed value.

    Examples
    --------
    >>> from torchlens._visualization._lens_door import check_lens
    >>> check_lens("speed")
    'speed'
    """

    if isinstance(lens, str) and lens in LENS_VALUES:
        return lens
    allowed = ", ".join(repr(value) for value in LENS_VALUES)
    raise InvalidArgumentError(
        f"lens must be one of {allowed}; received {lens!r}",
        code="draw_lens_invalid",
        remedy=f"pass one of {allowed}, or leave lens unset for the plain drawing",
        argument="lens",
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

    The lens resolver gives explicit keywords precedence over lens members,
    so only the values the caller actually chose may forward; a default must
    not override a lens member.

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

    Examples
    --------
    >>> from torchlens._visualization._lens_door import explicit_draw_kwargs
    >>> def draw(path=None, *, theme="torchlens", lens=None):
    ...     return explicit_draw_kwargs(draw, locals())
    >>> draw(theme="paper", lens="speed")
    {'theme': 'paper'}
    """

    explicit: dict[str, Any] = {}
    for name, parameter in inspect.signature(draw_fn).parameters.items():
        if name in _NOT_FORWARDED or name not in call_locals:
            continue
        value = call_locals[name]
        if not _is_default(value, parameter.default):
            explicit[name] = value
    return explicit


def draw_lens(trace: Trace, lens: object, user_kwargs: Mapping[str, Any]) -> Any:
    """Draw ``trace`` through the named lens.

    Parameters
    ----------
    trace:
        The trace to draw.
    lens:
        The requested lens; checked against :data:`LENS_VALUES`.
    user_kwargs:
        The caller's explicit ``Trace.draw`` keywords; they win over the
        lens's own settings.

    Returns
    -------
    Any
        What ``Trace.draw`` returns for the resolved keywords.

    Examples
    --------
    >>> drawing = trace.draw(lens="speed", file_format="svg")  # doctest: +SKIP
    """

    name = check_lens(lens)
    if name == "interventions":
        # The intervention lens writes its census beside the figure.
        from .surgery_visuals import draw_interventions

        return draw_interventions(trace, user_kwargs)
    from .lenses._resolve import draw_with_lens

    return draw_with_lens(trace, name, **user_kwargs)
