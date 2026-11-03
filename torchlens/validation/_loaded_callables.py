"""Resolve a loaded trace's op callables from their recorded function identities.

A saved bundle never carries callables (``Op.func`` is dropped at every level;
executable callables are not portable). It does carry each op's
``func_id``, the portable :class:`~torchlens.intervention._types.FunctionRegistryKey`
a capture that saves arguments records. Replay validation is an execution
boundary, so it resolves those keys there through the same purity-gated
resolver intervention replay uses: only pure torch, ``torch.Tensor``,
``torch.nn.functional`` and ``operator`` callables resolve. A ``custom`` key
(foreign code) is never imported here, so its op stays unresolved and the
entry check refuses the bundle by name exactly as before.
"""

from __future__ import annotations

from typing import Any

from .._errors import InvalidArgumentError
from .._utils._callable_safety import _unwrap_capture_wrapper

#: Registry namespaces resolved for replay; ``custom`` (foreign code) never is.
RESOLVABLE_NAMESPACES: frozenset[str] = frozenset(
    {"torch", "torch.Tensor", "torch.nn.functional", "operator"}
)


#: ``func_name`` sentinels of structural nodes (inputs, outputs, buffers,
#: functionless sources): never resolved, whatever ``func_id`` they inherited
#: (an output node copies its producer's fields, ``func_id`` included).
STRUCTURAL_FUNC_NAMES: frozenset[str] = frozenset({"none", "input", "output", "buffer", "constant"})


def _is_resolution_exempt(layer: Any) -> bool:
    """Whether the op must never be resolved: it has a live function or is structural."""

    if getattr(layer, "func", None) is not None:
        return True
    # A parentless literal producer (``ones``, ``full``) is an internal source
    # that still has a function: SPEC 2 rule 3 replays it. Functionless sources
    # carry a structural ``func_name`` and are excluded by name below.
    if (
        getattr(layer, "is_input", False)
        or getattr(layer, "is_output", False)
        or getattr(layer, "is_buffer", False)
    ):
        return True
    return str(getattr(layer, "func_name", "none")) in STRUCTURAL_FUNC_NAMES


def _resolution_candidate(layer: Any) -> Any:
    """The op's resolvable key, or ``None`` when the op must not be resolved."""

    from ..intervention._types import FunctionRegistryKey

    if _is_resolution_exempt(layer):
        return None
    func_name = str(getattr(layer, "func_name", "none"))
    key = getattr(layer, "func_id", None)
    if not isinstance(key, FunctionRegistryKey):
        return None
    if key.namespace == "custom":
        key = _public_key_for_torch_c_function(key)
        if key is None:
            return None
    if key.namespace not in RESOLVABLE_NAMESPACES:
        return None
    # The key must name the op's recorded function; a key that names another
    # function (an inherited or edited ``func_id``) fails closed: unresolved.
    if key.qualname.rsplit(".", maxsplit=1)[-1] != func_name:
        return None
    return key


#: torch's own C function modules and the public namespace that exposes the
#: same functions (``torch.nn.functional.gelu`` is ``torch._C._nn.gelu``).
_TORCH_C_PUBLIC_NAMESPACES: dict[str, str] = {
    "torch._C._nn": "torch.nn.functional",
    "torch._C._VariableFunctions": "torch",
}


def _public_key_for_torch_c_function(key: Any) -> Any:
    """The public torch key for a ``custom`` key naming a torch C function, else ``None``.

    Capture keys a few torch C builtins (``F.gelu``, whose ``__module__`` is
    ``torch._C._nn``) as ``custom``. The mapping is exact or nothing: the
    public namespace's attribute of that name, unwrapped from any TorchLens
    capture wrapper, must be the very C function. The bundle's import path is
    only parsed, never imported; the result still passes the resolver's
    purity gate.
    """

    import torch

    from ..intervention._types import FunctionRegistryKey

    module_name, _, name = str(key.import_path or "").partition(":")
    public_name = _TORCH_C_PUBLIC_NAMESPACES.get(module_name)
    if public_name is None or not name.isidentifier():
        return None
    c_module = torch._C._nn if module_name == "torch._C._nn" else torch._C._VariableFunctions
    public = torch.nn.functional if public_name == "torch.nn.functional" else torch
    target = getattr(c_module, name, None)
    exposed = getattr(public, name, None)
    if target is None or exposed is None or _unwrap_capture_wrapper(exposed) is not target:
        return None
    return FunctionRegistryKey(public_name, name, key.dispatch_kind)  # type: ignore[arg-type]


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

    from ..intervention._resolver_registry import resolve_function_registry_key
    from ..intervention.errors import ReplayPreconditionError, UntrustedCallableError

    resolved: list[Any] = []
    for layer in trace._layer_list or ():
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
