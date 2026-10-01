"""``echo=`` normalization and the entry-time typed refusals (snoop D1/D3).

One normalizer serves every mount (``tl.trace``, ``tl.record``,
``fastlog.Recorder``): the accepted spellings are ``True`` (everything plus
structure lines), ``"modules"`` (structure-only flight recorder), any
LIVE-EVALUABLE selector or callable, a grouped ``EchoOptions``, and
``False``/``None`` (off). String/label/index selections are REFUSED with the
remedy naming post-hoc ``.narrate()``: final labels do not exist during a
live forward, and the panel deleted the two-pass fallback (narrating a
replay is not narrating the requested forward).
"""

from __future__ import annotations

from typing import Any

from ..options import EchoOptions
from ._errors import EchoConfigError

__tl_layer__ = "L5"

#: Selector kinds that require finalized graph facts and therefore cannot be
#: narrated live; they refuse BEFORE execution with the post-hoc remedy.
_FINALIZED_SELECTOR_KINDS = frozenset({"output", "identity"})


def _refuse_finalized_selector(selector: Any) -> None:
    """Refuse selectors whose evaluation needs finalized graph facts."""

    from ..ir.selector_eval import walk_selector

    try:
        kinds = {
            getattr(node, "selector_kind", None)
            for node in walk_selector(selector)
            if getattr(node, "selector_kind", None) is not None
        }
    except Exception:  # noqa: BLE001 - an unwalkable selector defers to live evaluation
        return
    blocked = kinds & _FINALIZED_SELECTOR_KINDS
    if blocked:
        blocked_names = sorted(str(kind) for kind in blocked)
        raise EchoConfigError(
            f"echo= selector needs finalized graph facts ({blocked_names}) that do not "
            "exist during a live forward; TorchLens will not silently run a second pass "
            "and narrate the replay as though it were live. Remedy: narrate post-hoc via "
            "trace.narrate(select=...), or scope live echo with tl.func/tl.in_module.",
            code="echo_selector_not_live",
            remedy="use trace.narrate(select=...) post-hoc, or a live selector",
        )


def normalize_echo(echo: Any) -> EchoOptions | None:
    """Normalize one ``echo=`` value to a grouped ``EchoOptions`` (or off).

    Parameters
    ----------
    echo:
        The user's ``echo=`` value on any mount.

    Returns
    -------
    EchoOptions | None
        Normalized options, or ``None`` when narration is off.

    Raises
    ------
    EchoConfigError
        For unsupported spellings (strings other than ``"modules"``, lists,
        ints) and for selectors that need finalized labels.
    """

    if echo is None or echo is False:
        return None
    normalized = _coerce_echo_value(echo)
    select = normalized.select
    if select is None or select is False:
        return None
    _validate_select(select)
    if normalized.stats not in ("off", "reuse", "sampled", "exact"):
        raise EchoConfigError(
            f"EchoOptions.stats={normalized.stats!r} is not a rung. Remedy: use 'off', "
            "'reuse', 'sampled', or 'exact' (the four measured cost classes).",
            code="echo_argument_invalid",
            remedy="use stats='off'|'reuse'|'sampled'|'exact'",
        )
    return normalized


def _coerce_echo_value(echo: Any) -> EchoOptions:
    """Coerce one armed ``echo=`` spelling to ``EchoOptions``, or refuse typed."""

    if isinstance(echo, EchoOptions):
        return echo
    if echo is True:
        return EchoOptions(select=True)
    if isinstance(echo, str):
        if echo == "modules":
            return EchoOptions(select="modules")
        raise EchoConfigError(
            f"echo={echo!r} is not a live narration scope: label/substring selections "
            "resolve against FINAL labels, which do not exist during a live forward. "
            "Remedy: pass echo=True, echo='modules', a live selector such as "
            "tl.in_module(...), or narrate post-hoc via trace.narrate(select=...).",
            code="echo_argument_invalid",
            remedy="pass True, 'modules', a live selector, or EchoOptions",
        )
    if callable(echo):
        return EchoOptions(select=echo)
    raise EchoConfigError(
        f"echo= received unsupported type {type(echo).__name__}. Remedy: pass True, "
        "'modules', a live selector/callable, or tl.options.EchoOptions.",
        code="echo_argument_invalid",
        remedy="pass True, 'modules', a live selector, or EchoOptions",
    )


def _validate_select(select: Any) -> None:
    """Refuse non-live ``EchoOptions.select`` spellings typed."""

    if isinstance(select, str) and select != "modules":
        raise EchoConfigError(
            f"EchoOptions.select={select!r} is not a live narration scope. Remedy: use "
            "True, 'modules', or a live selector; post-hoc narrate() accepts substrings.",
            code="echo_argument_invalid",
            remedy="use True, 'modules', or a live selector",
        )
    from ..intervention.selectors import BaseSelector

    if isinstance(select, BaseSelector):
        _refuse_finalized_selector(select)
    elif select is not True and not callable(select) and select != "modules":
        raise EchoConfigError(
            f"EchoOptions.select received unsupported type {type(select).__name__}. "
            "Remedy: pass True, 'modules', or a live selector/callable.",
            code="echo_argument_invalid",
            remedy="pass True, 'modules', or a live selector/callable",
        )


def refuse_echo_shaped_predicate(value: Any, *, slot: str) -> None:
    """Teaching refusal: an ``EchoOptions`` routed into ``save=``/``hooks=``.

    Narration through the save-predicate slot is evaluated TWICE per
    declined op (measured: 554 evaluations for 277 events -- the alias
    retry), and a narrator that saves nothing declines everything; narration
    through ``hooks=`` doubles trace cost (+101% measured) and renders
    weaker lines. One mechanism exists: the echo observer slot.

    Parameters
    ----------
    value:
        The user-supplied slot value.
    slot:
        ``"save"`` or ``"hooks"`` -- the slot being policed.

    Raises
    ------
    EchoConfigError
        When ``value`` is (or contains) an ``EchoOptions``.
    """

    values: tuple[Any, ...] = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    if not any(isinstance(item, EchoOptions) for item in values):
        return
    if slot == "save":
        raise EchoConfigError(
            "EchoOptions is not a save predicate: the save slot is evaluated twice per "
            "declined op (the measured 554-evaluations-for-277-events alias retry), and a "
            "narrator that saves nothing declines everything. Remedy: pass echo=EchoOptions "
            "(narration) and keep save= for retention -- the two scopes are independent.",
            code="echo_as_save_predicate",
            remedy="pass it via echo=; keep save= for retention",
        )
    raise EchoConfigError(
        "EchoOptions is not a hook: narration through hooks= pays the intervention "
        "runtime's content probes and provenance records (+101% measured trace cost) and "
        "cannot render module path, depth, or pass. Remedy: pass echo=EchoOptions -- the "
        "read-only observer slot is the one narration mechanism.",
        code="echo_via_hooks_unsupported",
        remedy="pass it via echo=",
    )


__all__ = ["normalize_echo", "refuse_echo_shaped_predicate"]
