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

Residual: a table keyed on wrappers is stale after ``unwrap_torch()`` (no
wrapper is left on the call path to make the second offer); wrappers stay
installed for the process lifetime unless the user unwraps explicitly.
"""

from __future__ import annotations

import re
from typing import Any

import torch

from ...utils import _torch_compat

__all__ = ["redispatch_declined_protocol"]

# The two "every handler declined" messages: torch's C argument parser
# (C builtins and Tensor methods) and ``torch.overrides.handle_torch_function``
# (pure-Python functionals). Both quote the dispatched function's name first.
_DECLINED_DISPATCH_RE = re.compile(
    r"^(?:Multiple dispatch failed for|no implementation found for) '([^']+)'"
)
_REDISPATCHED_MARKER = "_torchlens_protocol_redispatched"


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


def redispatch_declined_protocol(
    exc: TypeError,
    wrapper: Any,
    func_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Re-offer a fully declined ``__torch_function__`` dispatch under ``wrapper``.

    Called from the wrapper's ``except TypeError`` around its original call.

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
        The first non-``NotImplemented`` handler result.

    Raises
    ------
    TypeError
        ``exc`` itself, unchanged, when it is not a declined dispatch of this
        function or when every handler declines the wrapper identity too.
    """

    disabled = _torch_compat.get_disabled_torch_function_impl()
    if disabled is None or not _declined_this_function(exc, func_name):
        raise exc
    overloaded = _overloaded_args(args, kwargs, disabled)
    types = tuple(type(value) for value in overloaded)
    for value in overloaded:
        result = value.__torch_function__(wrapper, types, args, kwargs)
        if result is not NotImplemented:
            return result
    setattr(exc, _REDISPATCHED_MARKER, True)
    raise exc
