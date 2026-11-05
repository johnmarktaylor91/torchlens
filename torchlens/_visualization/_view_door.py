"""``Trace.draw(view=...)``: the public door to the premade views.

Each allowed value names one registered view row (the private ``lenses``
registry); this module checks the value and forwards the caller's explicit
draw keywords to the resolver. It never renders. The allowed values are
spelled once, in :data:`torchlens._literals.DrawViewLiteral`.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, get_args

from .._errors import InvalidArgumentError
from .._literals import DrawViewLiteral
from .lenses._resolve import draw_with_lens
from .surgery_visuals import draw_interventions

if TYPE_CHECKING:
    from .._data_classes.trace import Trace

#: Allowed ``view=`` values, in the order the refusal lists them.
VIEW_VALUES: tuple[str, ...] = get_args(DrawViewLiteral)

#: ``Trace.draw`` parameters that never forward to the view resolver.
_NOT_FORWARDED = frozenset({"self", "trace", "view"})


def check_view(view: object) -> str:
    """Return ``view`` when it names a public view, else refuse typed.

    Parameters
    ----------
    view:
        The requested view name.

    Returns
    -------
    str
        The view name, unchanged.

    Raises
    ------
    InvalidArgumentError
        ``draw_view_invalid`` naming every allowed value.

    Examples
    --------
    >>> from torchlens._visualization._view_door import check_view
    >>> check_view("speed")
    'speed'
    """

    if isinstance(view, str) and view in VIEW_VALUES:
        return view
    allowed = ", ".join(repr(value) for value in VIEW_VALUES)
    raise InvalidArgumentError(
        f"view must be one of {allowed}; received {view!r}",
        code="draw_view_invalid",
        remedy=f"pass one of {allowed}, or leave view unset for the plain drawing",
        argument="view",
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

    The view resolver gives explicit keywords precedence over view members,
    so only the values the caller actually chose may forward; a default must
    not override a view member.

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
    >>> from torchlens._visualization._view_door import explicit_draw_kwargs
    >>> def draw(path=None, *, theme="torchlens", view=None):
    ...     return explicit_draw_kwargs(draw, locals())
    >>> draw(theme="paper", view="speed")
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


def draw_view(
    trace: Trace,
    view: object,
    user_kwargs: Mapping[str, Any],
    banner_lines: Callable[[Trace], Iterable[str]],
) -> Any:
    """Draw ``trace`` as the named premade view.

    Parameters
    ----------
    trace:
        The trace to draw.
    view:
        The requested view; checked against :data:`VIEW_VALUES`.
    user_kwargs:
        The caller's explicit ``Trace.draw`` keywords; they win over the
        view's own settings.
    banner_lines:
        Returns the capture's honesty banner lines (the interventions view
        records the ones its header leaves out beside the figure).

    Returns
    -------
    Any
        What ``Trace.draw`` returns for the resolved keywords.

    Examples
    --------
    >>> drawing = trace.draw(view="speed", file_format="svg")  # doctest: +SKIP
    """

    name = check_view(view)
    if name == "interventions":
        # The interventions view writes its census beside the figure.
        return draw_interventions(trace, user_kwargs, banner_lines)
    return draw_with_lens(trace, name, **user_kwargs)
