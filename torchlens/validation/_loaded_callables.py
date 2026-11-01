"""Resolve a loaded trace's op callables from their recorded function identities.

A saved bundle never carries callables (``Op.func`` is dropped at every level;
executable callables are not portable). It does carry each op's
``func_id``, the portable :class:`~torchlens.intervention.types.FunctionRegistryKey`
a capture that saves arguments records. Replay validation is an execution
boundary, so it resolves those keys there through the same purity-gated
resolver intervention replay uses: only pure torch, ``torch.Tensor``,
``torch.nn.functional`` and ``operator`` callables resolve. A ``custom`` key
(foreign code) is never imported here, so its op stays unresolved and the
entry check refuses the bundle by name exactly as before.
"""

from __future__ import annotations

from typing import Any

#: Registry namespaces resolved for replay; ``custom`` (foreign code) never is.
RESOLVABLE_NAMESPACES: frozenset[str] = frozenset(
    {"torch", "torch.Tensor", "torch.nn.functional", "operator"}
)


def resolve_loaded_callables(trace: Any) -> int:
    """Set ``func`` on each loaded op whose recorded ``func_id`` resolves.

    Parameters
    ----------
    trace:
        A trace loaded from a bundle (``_loaded_from_bundle`` is true).

    Returns
    -------
    int
        Number of ops whose callable was resolved. Ops whose key is absent,
        ``custom``, or refused by the resolver keep ``func=None``; the caller's
        unresolved-callable check names them.
    """

    from .._errors import InvalidArgumentError
    from ..intervention.errors import ReplayPreconditionError, UntrustedCallableError
    from ..intervention.resolver import resolve_function_registry_key
    from ..intervention.types import FunctionRegistryKey

    resolved = 0
    for layer in getattr(trace, "layer_list", ()) or ():
        if getattr(layer, "func", None) is not None:
            continue
        key = getattr(layer, "func_id", None)
        if not isinstance(key, FunctionRegistryKey) or key.namespace not in RESOLVABLE_NAMESPACES:
            continue
        try:
            func = resolve_function_registry_key(key)
        except (
            AttributeError,
            InvalidArgumentError,
            ReplayPreconditionError,
            UntrustedCallableError,
        ):
            # Not resolvable on this torch, or refused as impure: the op stays
            # functionless and the entry check refuses the bundle naming it.
            continue
        layer.func = func
        resolved += 1
    return resolved
