"""Foreign wrappers over torchlens wrappers: detection, rebasing, reporting.

``unwrap_torch`` restores a namespace slot only when the CURRENT attribute is
a known torchlens wrapper; a third-party wrapper installed on top of ours is
(correctly) left in place. Left alone, that keeps the torchlens wrapper LIVE
inside the foreign wrapper's ``__wrapped__``, closure cells or module globals
(B8-6): the slot still runs torchlens code after "unwrap", and the user learns
nothing. A peer that patches torch at import after a capture (``nnsight``
re-wraps ``torch.Tensor.backward`` with ``functools.wraps`` around whatever is
installed) produces exactly this shape.

This module makes teardown honest instead of merely loud. Every reference a
foreign wrapper holds to a live torchlens wrapper is REBASED onto the original
that wrapper stood for (the callable the foreign code believed it was
wrapping), and the markers ``functools.wraps`` copied from our wrapper's
``__dict__`` are stripped, so the foreign wrapper neither runs nor impersonates
torchlens code. Only a burial the walk cannot rebase (a read-only
``__wrapped__``, a reference held somewhere the walk does not reach) is
reported, by name, through :class:`~torchlens._errors.TorchLensWarning`.
The walk is bounded and runs only for drifted slots, never on the hot path.
"""

from __future__ import annotations

import warnings
from typing import Any

from ... import _state
from ..._errors import TorchLensWarning

# Nodes visited per drifted slot; nnsight-shaped burials resolve in two.
_WALK_LIMIT = 32

# ``functools.wraps`` copies the wrapped callable's ``__dict__``; these are the
# entries ``torch_func_decorator`` stamps, so a foreign wrapper built over a
# torchlens wrapper carries them and satisfies attribute-based "is this ours"
# probes (identity_shims.py documents the anti-pattern).
_LEAKED_MARKERS = (
    "_tl",
    "__tl_original_id__",
    "__tl_wrapper_name__",
    "__tl_detector_excluded__",
    "__prepare_scriptable__",
)

# ``(holder, kind, key, wrapper)``: how a foreign object reaches a live wrapper.
_Reference = tuple[Any, str, Any, Any]


def buried_wrapper_references(current: Any) -> list[_Reference]:
    """Return every reference through which ``current`` reaches a live torchlens wrapper.

    ``kind`` is ``"wrapped"`` (``holder.__wrapped__`` is the wrapper), ``"cell"``
    (``holder`` is a closure cell holding it) or ``"global"`` (``holder`` is a
    function's ``__globals__`` dict and ``key`` the name bound to it). Torchlens
    wrappers are recognised by LEDGER IDENTITY (``_state._decorated_to_orig``),
    never by attribute, and are recorded rather than walked.
    """

    found: list[_Reference] = []
    ledger = _state._decorated_to_orig
    seen: set[int] = set()
    stack: list[Any] = [current]
    visited = 0
    while stack and visited < _WALK_LIMIT:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        visited += 1
        wrapped = getattr(obj, "__wrapped__", None)
        if callable(wrapped):
            if id(wrapped) in ledger:
                found.append((obj, "wrapped", None, wrapped))
            else:
                stack.append(wrapped)
        for cell in getattr(obj, "__closure__", None) or ():
            try:
                content = cell.cell_contents
            except ValueError:
                continue
            if not callable(content):
                continue
            if id(content) in ledger:
                found.append((cell, "cell", None, content))
            else:
                stack.append(content)
        module_globals = getattr(obj, "__globals__", None)
        if isinstance(module_globals, dict):
            for key, value in list(module_globals.items()):
                if callable(value) and id(value) in ledger:
                    found.append((module_globals, "global", key, value))
    return found


def buries_live_wrapper(current: Any) -> bool:
    """Return whether a foreign callable buries a live torchlens wrapper."""

    return bool(buried_wrapper_references(current))


def _strip_leaked_markers(holder: Any, wrapper: Any) -> None:
    """Drop the ``__dict__`` entries ``functools.wraps`` copied from ``wrapper``."""

    try:
        holder_dict = vars(holder)
        wrapper_dict = vars(wrapper)
    except TypeError:
        return
    for name in _LEAKED_MARKERS:
        if name in holder_dict and holder_dict[name] is wrapper_dict.get(name):
            try:
                delattr(holder, name)
            except (AttributeError, TypeError):
                continue


def rebase_buried_wrapper(current: Any) -> bool:
    """Point a foreign wrapper's torchlens references at the restored originals.

    Each reference found by :func:`buried_wrapper_references` is rewritten to
    the original the buried wrapper stood for, which is the callable the
    foreign code saw installed had it looked before the first capture; a
    torchlens wrapper outside a capture forwards to that original, so the
    rewrite preserves the foreign wrapper's behaviour and removes torchlens
    from its call path. Returns True when ``current`` STILL buries a live
    wrapper afterwards (a reference the walk could not rewrite), which is the
    case teardown reports.
    """

    references = buried_wrapper_references(current)
    if not references:
        return False
    for holder, kind, key, wrapper in references:
        orig = _state._decorated_to_orig[id(wrapper)]
        try:
            if kind == "wrapped":
                holder.__wrapped__ = orig
                _strip_leaked_markers(holder, wrapper)
            elif kind == "cell":
                holder.cell_contents = orig
            else:
                holder[key] = orig
        except (AttributeError, TypeError, ValueError):
            continue
    return bool(buried_wrapper_references(current))


def warn_buried_sites(buried_sites: list[str]) -> None:
    """Report the slots whose foreign wrapper still runs a torchlens wrapper.

    Called LAST in teardown: a warning filter may raise this (pytest's
    ``error::`` rules, a user's ``simplefilter("error")``), and raising before
    the decorated flag flipped left torch restored under a flag that said
    wrapped, so every later capture skipped ``wrap_torch`` and logged only
    module boundaries.
    """

    if not buried_sites:
        return
    shown = ", ".join(buried_sites[:5])
    more = f" (+{len(buried_sites) - 5} more)" if len(buried_sites) > 5 else ""
    warnings.warn(
        f"unwrap_torch() left {len(buried_sites)} torch callable(s) with a "
        f"torchlens wrapper buried under a third-party wrapper: {shown}{more}. "
        "TorchLens never clobbers foreign patches and could not rebase these onto "
        "the restored originals, so those slots still run the torchlens wrapper "
        "underneath. Remove or reinstall the outer wrapper around the restored "
        "original to fully unwrap.",
        TorchLensWarning,
        stacklevel=4,
    )
