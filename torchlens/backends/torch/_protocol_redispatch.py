"""Declined-protocol redispatch: offer a handler the WRAPPER identity it keyed on.

Torch's ``__torch_function__`` protocol hands every handler the ORIGINAL torch
callable (C builtins directly; pure-Python functionals through the site-8
protocol-identity shim in ``identity_shims``). A third-party library imported
while TorchLens wrappers are installed builds its handler table from the
namespace it reads at import time -- the WRAPPER -- so its
``func not in TABLE`` check declines the original. tensordict is the observed
case (``TD_HANDLED_FUNCTIONS[torch.stack] = ...`` at import, ``func not in
TD_HANDLED_FUNCTIONS`` at dispatch): every TensorDict ``torch.stack`` /
``torch.cat`` / ``torch.zeros_like`` raised "Multiple dispatch failed ... all
__torch_function__ handlers returned NotImplemented" once the library was
imported after ``wrap_torch()``. A per-library table shim does not scale and
the table cannot exist yet at wrap time, so the fix is at the protocol edge.

Strategy: the wrapper calls the original exactly as before; only when torch
reports that EVERY handler declined this very function does the wrapper offer
the same handlers the same arguments once more under the wrapper identity, in
torch's own handler order. A handler whose basis is the original declines
again and the original ``TypeError`` re-raises unchanged; a handler keyed on
the wrapper (or comparing ``func is torch.stack`` against a call-time
namespace read) now answers exactly as it would in an unwrapped eager process.
Nothing changes on the success path, and a declined dispatch is side-effect
free by the protocol's contract, so the second offer cannot double any work.

Unwrapped torch has no wrapper on the call path to make that second offer --
``unwrap_torch()`` and, above all, validation's pristine-torch ground truth
(``validation/_pristine.py`` unwraps for one forward). So unwrap first gives
each wrapper-keyed table built during a wrap epoch the ORIGINAL key beside
the wrapper key (:func:`alias_wrapper_keyed_tables`): module-level and
class-level ``dict``/``set`` tables of modules imported after this one,
torch and TorchLens excluded. That is the table an unwrapped eager import
builds. The next wrap removes exactly the keys it added
(:func:`undo_table_aliases`), so foreign tables differ from their import-time
contents only while torch is unwrapped. Residual: a table hidden deeper (a
closure, an instance attribute, a frozenset) is not aliased; its wrapped calls
still redispatch.
"""

from __future__ import annotations

import re
import sys
from typing import Any

import torch

from ... import _state
from ..._utils import _torch_compat

__all__ = [
    "STILL_DECLINED",
    "alias_wrapper_keyed_tables",
    "redispatch_declined",
    "undo_table_aliases",
]

#: Returned by :func:`redispatch_declined` when the caller must re-raise its
#: ``TypeError`` with a bare ``raise``, so the traceback stays exactly torch's.
STILL_DECLINED = object()

# The two "every handler declined" messages: torch's C argument parser
# (C builtins and Tensor methods) and ``torch.overrides.handle_torch_function``
# (pure-Python functionals). Both quote the dispatched function's name first.
_DECLINED_DISPATCH_RE = re.compile(
    r"^(?:Multiple dispatch failed for|no implementation found for) '([^']+)'"
)
_REDISPATCHED_MARKER = "_torchlens_protocol_redispatched"
_EXCLUDED_PACKAGES = ("torch", "torchlens")
# Modules already imported when this module loads (never later than the first
# wrap) keyed their tables on originals; scanning a later pre-wrap import too
# only costs time, since a table without wrapper keys gains nothing.
_PRE_WRAP_MODULES: frozenset[str] = frozenset(sys.modules)
# Original keys added by the last unwrap, as (table, original, added value);
# the next wrap removes them, so a foreign table is touched only while unwrapped.
_ALIAS_JOURNAL: list[tuple[Any, Any, Any]] = []


def _declined_this_function(exc: TypeError, func_name: str) -> bool:
    """Return whether ``exc`` reports that every handler declined ``func_name``.

    Parameters
    ----------
    exc:
        The ``TypeError`` raised by the original torch callable.
    func_name:
        The wrapped callable's attribute name (``"stack"``, ``"pad"``).

    Returns
    -------
    bool
        ``True`` only for torch's declined-dispatch message naming this very
        function, and never for an error a nested wrapper already redispatched,
        so an enclosing call never re-runs a handler that did real work.
    """

    if getattr(exc, _REDISPATCHED_MARKER, False) or not exc.args:
        return False
    match = _DECLINED_DISPATCH_RE.match(str(exc.args[0]))
    if match is None:
        return False
    dispatched = match.group(1)
    return dispatched == func_name or dispatched.endswith("." + func_name)


def _overloaded_args(args: tuple[Any, ...], kwargs: dict[str, Any], disabled: Any) -> list[Any]:
    """Collect the arguments torch's protocol would dispatch on, in its order.

    Top-level arguments, keyword values, and the elements of a top-level
    list/tuple (``torch.stack``'s tensor sequence) are candidates. Exact
    ``torch.Tensor`` instances and handlers equal to torch's disabled sentinel
    (``torch.nn.Parameter``, opted-out subclasses) are skipped, as torch skips
    them. One argument per type, a subclass ahead of its superclass -- the
    ordering ``torch.overrides`` documents for the protocol.

    Parameters
    ----------
    args, kwargs:
        The call's arguments.
    disabled:
        ``torch._C._disabled_torch_function_impl``.

    Returns
    -------
    list[Any]
        One representative argument per overloading type, in dispatch order.
    """

    candidates: list[Any] = []
    for value in (*args, *kwargs.values()):
        if isinstance(value, (list, tuple)):
            candidates.extend(value)
        else:
            candidates.append(value)
    ordered: list[Any] = []
    seen_types: set[type] = set()
    for value in candidates:
        value_type = type(value)
        if value_type in seen_types or value_type is torch.Tensor:
            continue
        handler = getattr(value_type, "__torch_function__", None)
        if handler is None or handler == disabled:
            continue
        seen_types.add(value_type)
        index = next(
            (i for i, prior in enumerate(ordered) if issubclass(value_type, type(prior))),
            len(ordered),
        )
        ordered.insert(index, value)
    return ordered


def redispatch_declined(
    exc: TypeError,
    wrapper: Any,
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Re-offer a fully declined ``__torch_function__`` dispatch under ``wrapper``.

    Called from the wrapper's ``except TypeError`` around its original call;
    the wrapper re-raises with a bare ``raise`` on :data:`STILL_DECLINED`, so
    every ``func`` call and every re-raise stays in the wrapper's own frame
    (the escape detector's immediate-caller check and failure-origin
    classification both read that frame).

    Parameters
    ----------
    exc:
        The ``TypeError`` the original torch callable raised.
    wrapper:
        The TorchLens wrapper of that callable (the identity a post-wrap
        import keyed its handler table on).
    func_name:
        The wrapped callable's attribute name.
    args, kwargs:
        The call's arguments, exactly as the original received them.

    Returns
    -------
    Any
        The first non-``NotImplemented`` handler result, or
        :data:`STILL_DECLINED` when ``exc`` is not a declined dispatch of this
        function or every handler declines the wrapper identity too.
    """

    disabled = _torch_compat.get_disabled_torch_function_impl()
    if disabled is None or not _declined_this_function(exc, func_name):
        return STILL_DECLINED
    overloaded = _overloaded_args(args, kwargs, disabled)
    types = tuple(type(value) for value in overloaded)
    for value in overloaded:
        result = value.__torch_function__(wrapper, types, args, kwargs)
        if result is not NotImplemented:
            return result
    setattr(exc, _REDISPATCHED_MARKER, True)
    return STILL_DECLINED


def _alias_container(container: Any, ledger: dict[int, Any]) -> int:
    """Give every wrapper key in one ``dict``/``set`` its original beside it.

    Each addition is journaled for :func:`undo_table_aliases`.

    Parameters
    ----------
    container:
        A ``dict`` or ``set`` (subclasses included).
    ledger:
        The decorated-to-original unwrap ledger (wrapper id to original).

    Returns
    -------
    int
        The number of original keys added.
    """

    added = 0
    for key in list(container):
        original = ledger.get(id(key))
        if original is None or original in container:
            continue
        if isinstance(container, dict):
            value = container[key]
            container[original] = value
        else:
            value = original
            container.add(original)
        _ALIAS_JOURNAL.append((container, original, value))
        added += 1
    return added


def _module_tables(name: str, module: Any) -> list[Any]:
    """Return the module- and class-level ``dict``/``set`` tables of one module."""

    tables: list[Any] = []
    for attr, value in list(vars(module).items()):
        if attr.startswith("__"):
            continue
        if isinstance(value, (dict, set)):
            tables.append(value)
        elif isinstance(value, type) and getattr(value, "__module__", None) == name:
            tables.extend(v for v in vars(value).values() if isinstance(v, (dict, set)))
    return tables


def alias_wrapper_keyed_tables() -> int:
    """Key post-wrap third-party tables on originals too, for the unwrapped window.

    Called from ``identity_shims.remove_identity_shims`` at the head of every
    ``unwrap_torch()``, under the wrapper install lock, ledger populated; the
    next wrap undoes it (:func:`undo_table_aliases`). A ``sys.modules`` entry
    without a namespace, or a table subclass that refuses iteration or
    insertion, is skipped: aliasing serves unwrapped dispatch only and is never
    a capture fact.

    Returns
    -------
    int
        The number of original keys added across all tables.
    """

    ledger, _ = _state.wrap_epoch_ledgers()
    if not ledger:
        return 0
    added = 0
    for name, module in list(sys.modules.items()):
        if module is None or name in _PRE_WRAP_MODULES:
            continue
        if name.split(".", 1)[0] in _EXCLUDED_PACKAGES:
            continue
        try:
            tables = _module_tables(name, module)
        except TypeError:  # a sys.modules entry with no __dict__
            continue
        for table in tables:
            try:
                added += _alias_container(table, ledger)
            except (TypeError, RuntimeError, KeyError):  # a refusing table subclass
                continue
    return added


def undo_table_aliases() -> None:
    """Remove the original keys the last unwrap added (called at every wrap).

    A key is removed only while it still holds exactly the value the alias
    added, so an entry the library re-registered since is left alone.
    """

    while _ALIAS_JOURNAL:
        container, original, value = _ALIAS_JOURNAL.pop()
        try:
            if isinstance(container, dict):
                if original in container and container[original] is value:
                    del container[original]
            else:
                container.discard(original)
        except (TypeError, RuntimeError, KeyError):  # a refusing table subclass
            continue
