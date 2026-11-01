"""Route a fast live run around hook-invisible direct ``module.forward(x)`` calls.

The fast live session collects module sites, applies staged steering and folds
module-entry fingerprint tokens through torch module hooks, which fire only
inside ``Module.__call__``. A model that reaches a submodule by calling its
``forward`` directly (``return self.embedding.forward(indices)``) skips them,
while capture, whose ``forward`` decoration sees every call, records the module
call. Capture also marks each module it entered that way
(``ModuleMeta.direct_forward_called`` on the live module).

For exactly those modules, and only for the duration of one native forward, the
session installs an instance ``forward`` wrapper. A call through
``Module.__call__`` passes straight to the module's own ``forward`` (the hooks
already ran around it); a direct call runs the session's own hook work around
the forward in torch's hook order: the module-entry fingerprint token (the
token pre-hook), the forward, every staged steer hook for the module (the steer
forward hooks), then the site collection (the collection forward hook). User
hooks are skipped on a direct call, as torch and capture both skip them. Every
other module stays on hooks, and each run restores the prior ``forward`` by
identity.
"""

from __future__ import annotations

import sys
import weakref
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from torch import nn

from . import _state
from ._call_fingerprint import module_fingerprint_address
from ._fast_live_steer import apply_steer_paused, session_is_active
from ._utils._module_call_dispatch import forward_entered_through_module_call
from .backends.torch._session_forwards import _NO_INSTANCE_FORWARD, _SessionForward
from .backends.torch._tl import get_module_meta


@dataclass(frozen=True)
class DirectForwardRoute:
    """The session hook work one directly called module needs on a direct call."""

    module_ref: weakref.ReferenceType[nn.Module]
    token: int
    steer_addresses: tuple[str, ...]
    collect_address: str | None


def _was_called_directly(module: nn.Module) -> bool:
    """Return whether a capture marked ``module`` as entered by a direct call."""

    meta = get_module_meta(module)
    return meta is not None and meta.direct_forward_called


def build_direct_forward_routes(
    model: nn.Module,
    modules: Mapping[str, nn.Module],
    plan_addresses: tuple[str, ...],
    steer_modules: Mapping[str, nn.Module],
) -> tuple[DirectForwardRoute, ...]:
    """Build one route per live module a capture entered by a direct ``forward()`` call.

    Parameters
    ----------
    model:
        Live root model (never routed: capture leaves the root undecorated).
    modules:
        Every live address, alias spellings included.
    plan_addresses:
        Addresses the session collects through a forward hook.
    steer_modules:
        Steered address to live module, as the steer plan resolved them.
    """

    routes: list[DirectForwardRoute] = []
    for name, module in model.named_modules():
        if module is model or not _was_called_directly(module):
            continue
        collect = [
            planned
            for planned in plan_addresses
            if modules.get("" if planned == "self" else planned) is module
        ]
        routes.append(
            DirectForwardRoute(
                module_ref=weakref.ref(module),
                token=_state.module_token(module_fingerprint_address(module, name)),
                steer_addresses=tuple(
                    steered for steered, target in steer_modules.items() if target is module
                ),
                collect_address=collect[0] if collect else None,
            )
        )
    return tuple(routes)


class _RoutedForward(_SessionForward):
    """A per-run ``forward`` that runs the session hook work on a direct call only."""

    def __init__(
        self,
        direct: Callable[..., Any],
        module: nn.Module,
        original: Callable[..., Any],
        prior: Any,
    ) -> None:
        """Hold the direct-call delegate and the module's own ``forward``."""

        super().__init__(direct, module, original, prior)
        self._original = original

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Pass a ``Module.__call__`` call through; route a direct call."""

        if forward_entered_through_module_call(sys._getframe(1)):
            return self._original(*args, **kwargs)
        return self._delegate(*args, **kwargs)


def _direct_call(
    route: DirectForwardRoute,
    module: nn.Module,
    original: Callable[..., Any],
    session_ref: weakref.ReferenceType[Any],
) -> Callable[..., Any]:
    """Build the direct-call body: token, forward, steer hooks, site collection."""

    active = session_is_active(session_ref)

    def run_direct(*args: Any, **kwargs: Any) -> Any:
        """Do on a direct call what the session's hooks do around ``module(x)``."""

        session = session_ref()
        if session is None or not active():
            return original(*args, **kwargs)
        _state.note_fingerprint_token(route.token)
        output = original(*args, **kwargs)
        for address in route.steer_addresses:
            replaced = apply_steer_paused(session, address, module, args, output)
            if replaced is not None:
                output = replaced
        if route.collect_address is not None:
            session.capture_module(route.collect_address, output)
        return output

    return run_direct


@contextmanager
def direct_forward_routes_installed(
    routes: tuple[DirectForwardRoute, ...], session_ref: weakref.ReferenceType[Any]
) -> Iterator[None]:
    """Install the routes for one native forward, then restore every ``forward``.

    A module whose ``forward`` no longer holds the route at exit (the model
    reassigned it mid-forward) keeps the newer value, as session cleanup does.
    """

    installed: list[tuple[nn.Module, Any, _RoutedForward]] = []
    try:
        for route in routes:
            module = route.module_ref()
            if module is None:
                continue
            prior = module.__dict__.get("forward", _NO_INSTANCE_FORWARD)
            original = module.forward
            wrapper = _RoutedForward(
                _direct_call(route, module, original, session_ref), module, original, prior
            )
            module.__dict__["forward"] = wrapper
            installed.append((module, prior, wrapper))
        yield
    finally:
        for module, prior, wrapper in reversed(installed):
            if module.__dict__.get("forward") is not wrapper:
                continue
            if prior is _NO_INSTANCE_FORWARD:
                module.__dict__.pop("forward", None)
            else:
                module.__dict__["forward"] = prior
