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


#: ``func_name`` sentinels of structural nodes (inputs, outputs, buffers,
#: functionless sources): never resolved, whatever ``func_id`` they inherited
#: (an output node copies its producer's fields, ``func_id`` included).
STRUCTURAL_FUNC_NAMES: frozenset[str] = frozenset({"none", "input", "output", "buffer", "constant"})


def _resolution_candidate(layer: Any) -> Any:
    """The op's resolvable key, or ``None`` when the op must not be resolved."""

    from ..intervention.types import FunctionRegistryKey

    if getattr(layer, "func", None) is not None:
        return None
    if (
        getattr(layer, "is_input", False)
        or getattr(layer, "is_output", False)
        or getattr(layer, "is_buffer", False)
        or getattr(layer, "is_internal_source", False)
    ):
        return None
    func_name = str(getattr(layer, "func_name", "none"))
    if func_name in STRUCTURAL_FUNC_NAMES:
        return None
    key = getattr(layer, "func_id", None)
    if not isinstance(key, FunctionRegistryKey) or key.namespace not in RESOLVABLE_NAMESPACES:
        return None
    # The key must name the op's recorded function; a key that names another
    # function (an inherited or edited ``func_id``) fails closed: unresolved.
    if key.qualname.rsplit(".", maxsplit=1)[-1] != func_name:
        return None
    return key


def resolve_loaded_callables(trace: Any) -> list[Any]:
    """Set ``func`` on each loaded computational op whose recorded ``func_id`` resolves.

    Parameters
    ----------
    trace:
        A trace loaded from a bundle (``_loaded_from_bundle`` is true).

    Returns
    -------
    list[Any]
        The ops whose callable was resolved, for :func:`release_loaded_callables`.
        Ops whose key is absent, ``custom``, names another function, or is
        refused by the resolver keep ``func=None``; the validation entry check
        names them.
    """

    from .._errors import InvalidArgumentError
    from ..intervention.errors import ReplayPreconditionError, UntrustedCallableError
    from ..intervention.resolver import resolve_function_registry_key

    resolved: list[Any] = []
    for layer in getattr(trace, "layer_list", ()) or ():
        key = _resolution_candidate(layer)
        if key is None:
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
        resolved.append(layer)
    return resolved


def release_loaded_callables(layers: list[Any]) -> None:
    """Return resolved ops to the loaded artifact's shape (``func`` is never persisted)."""

    for layer in layers:
        layer.func = None
