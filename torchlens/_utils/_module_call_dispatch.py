"""Tell ``module(x)`` from a direct ``module.forward(x)`` by the caller frame.

Torch module hooks fire only inside ``Module.__call__``; a direct
``module.forward(x)`` call skips every hook. Capture's session ``forward``
wrapper records the modules a forward reached directly, and the fast live run
routes only those modules around its hooks; both classify a call by comparing
the frame that called ``forward`` with the code objects of torch's
call-dispatch functions, which ``_torch_compat`` looks up (the private-API
chokepoint) and hands to :func:`probe_module_call_dispatch_frame`.

The probe is behavioral, never a version parse: a module called both ways must
be told apart. Its result is the ``HAS_MODULE_CALL_DISPATCH_FRAME`` flag in
``_torch_compat`` (visible in the capability snapshot and ``doctor()``); when it
is False no call is ever classed direct, so the fast run keeps refusing a
hook-invisible module call as ``fast_live_module_hook_unobserved``.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterable
from typing import Any

import torch

#: The probed call-dispatch code objects; empty until (or unless) the probe passes.
_DISPATCH_CODES: set[types.CodeType] = set()


def _run_dispatch_probe(codes: frozenset[types.CodeType]) -> bool:
    """Return whether one probe module's two call spellings classify correctly."""

    if not codes:
        return False
    seen: list[bool] = []

    class _DispatchProbe(torch.nn.Module):
        """Module recording whether each forward came through ``__call__``."""

        def forward(self) -> None:
            """Record the caller frame's classification."""

            seen.append(sys._getframe(1).f_code in codes)

    try:
        probe = _DispatchProbe()
        probe()
        probe.forward()
    except Exception:  # noqa: BLE001 -- any probe failure means "cannot tell"
        return False
    return seen == [True, False]


def probe_module_call_dispatch_frame(forward_callers: Iterable[Any]) -> bool:
    """Probe and arm the caller-frame classification.

    Parameters
    ----------
    forward_callers:
        torch's ``Module.__call__`` functions that invoke ``forward``
        (``None`` for one this torch lacks).

    Returns
    -------
    bool
        ``True`` when ``module(x)`` and ``module.forward(x)`` are told apart; only
        then does :func:`forward_entered_through_module_call` ever answer False.
    """

    codes = frozenset(
        code
        for code in (getattr(caller, "__code__", None) for caller in forward_callers)
        if isinstance(code, types.CodeType)
    )
    _DISPATCH_CODES.clear()
    if not _run_dispatch_probe(codes):
        return False
    _DISPATCH_CODES.update(codes)
    return True


def forward_entered_through_module_call(caller: types.FrameType | None) -> bool:
    """Return whether a module ``forward`` callable was invoked by ``Module.__call__``.

    Parameters
    ----------
    caller:
        The frame that called the ``forward`` callable.

    Returns
    -------
    bool
        ``True`` for ``module(x)`` (torch module hooks run), ``False`` for a
        direct ``module.forward(x)``. Always ``True`` when the probe failed, so
        no call is classed direct on a torch where the frame cannot tell.
    """

    if caller is None or not _DISPATCH_CODES:
        return True
    return caller.f_code in _DISPATCH_CODES
