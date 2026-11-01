"""Session-end tensor metadata cleanup for torch capture.

Every tensor a capture session touched carries TorchLens metadata until the
session ends. The walkers here clear it from the model tree, from the
containers and callables reachable through module attributes, and from the
globals of decorated forwards, so no tensor outlives its session carrying
provenance a later capture could mistake for its own (split out of
``model_prep.py`` under the file-size ratchet; this is the teardown seam,
not preparation). ``model_prep`` re-exports the three ``_clear_*`` walkers
under their original names.
"""

from collections import deque
from collections.abc import Callable, Iterable
from types import ModuleType
from typing import Any

import torch
from torch import nn

from ._tl import clear_meta, is_forward_call_decorated

__all__ = [
    "_clear_callable_session_tensor_metadata",
    "_clear_container_tree_tensor_metadata",
    "_clear_session_tensor_metadata",
]


def _is_isinstance_hostile_deprecation_shim(value: Any) -> bool:
    """Return whether ``value`` is torch's ``reduce_op`` deprecation singleton.

    ``torch.distributed.reduce_op`` is a ``_reduce_op`` instance whose
    ``__getattribute__`` emits a ``FutureWarning`` on ANY attribute access --
    including the ``__class__`` read every ``isinstance()`` performs -- so an
    initialized distributed process running under ``-W error`` had its capture
    CLEANUP aborted by the namespace walks below (SF-45). ``type()`` bypasses
    the instance's ``__getattribute__``, so this exact-name check is silent;
    structural matching (not an import of the private class) follows the
    ``_distributed.py`` fallback convention.
    """

    value_type = type(value)
    return (
        value_type.__name__ == "_reduce_op"
        and value_type.__module__ == "torch.distributed.distributed_c10d"
    )


def _clear_session_tensor_metadata(
    value: Any,
    seen: set[int],
    depth: int = 0,
    visit: Callable[[torch.Tensor], None] | None = None,
) -> None:
    """Clear TorchLens tensor metadata from a model-owned object graph.

    Parameters
    ----------
    value
        Candidate object reachable from a prepared model.
    seen
        Object ids already visited during this cleanup scan.
    depth
        Current recursion depth, used to bound traversal through arbitrary
        third-party helper objects.
    visit
        Optional action applied to each reachable non-Parameter tensor instead
        of the default ``clear_meta`` -- the pre-forward ownership snapshot
        (:func:`_collect_model_owned_tensor_ids`) reuses this exact traversal
        so the "model-owned" surfaces of the snapshot and the session-end
        clear can never drift apart.

    Returns
    -------
    None
        Mutates reachable tensors in place by removing TorchLens metadata
        (or applies ``visit`` when given).
    """

    action = clear_meta if visit is None else visit
    if value is None or _is_isinstance_hostile_deprecation_shim(value):
        return
    if isinstance(value, (str, bytes, int, float, bool)):
        return
    if isinstance(value, ModuleType):
        _clear_module_namespace_tensor_metadata(value, seen, depth, action, visit)
        return
    if isinstance(value, torch.Tensor):
        if not isinstance(value, torch.nn.Parameter):
            action(value)
        return
    obj_id = id(value)
    if obj_id in seen or depth >= 12:
        return
    seen.add(obj_id)
    _clear_reachable_tensor_metadata(value, seen, depth, visit)


def _clear_module_namespace_tensor_metadata(
    value: ModuleType,
    seen: set[int],
    depth: int,
    action: Callable[[torch.Tensor], None],
    visit: Callable[[torch.Tensor], None] | None,
) -> None:
    """Clear stamps on tensors and plain containers stored directly in a module namespace."""

    # r81 (r80 F1 root B): a stamped tensor stashed as a DIRECT attribute of
    # a ``types.ModuleType`` escaped the belt entirely (this walk returned
    # immediately for modules). r33/r83: a stamp nested inside a plain
    # CONTAINER in the module namespace escaped the shallow sweep too and
    # laundered provenance across sessions, so namespace containers now get
    # a dedicated PURE container-tree descent (tensors + plain containers
    # only). The descent deliberately never enters objects or nested
    # modules: this walk reaches broadly-imported modules (``torch``,
    # ``math``) through model-owned helper objects, and generic recursion
    # from a module namespace would explode through ``sys.modules`` into
    # every imported module in the process. Object stashes inside module
    # namespaces stay covered by the session identity belt, which never
    # trusts an unregistered stamp anyway.
    obj_id = id(value)
    if obj_id in seen or depth >= 12:
        return
    seen.add(obj_id)
    namespace = getattr(value, "__dict__", None)
    if isinstance(namespace, dict):
        # No slot memo (r8 b4 R39, sol probe): the memo invalidated on
        # ``len(namespace)`` only, so REPLACING a value at a constant
        # length (``mod.stash = 5`` becoming ``mod.stash = tensor``) kept
        # the stale slot list and the stamped tensor escaped the session
        # clear -- provenance laundering across sessions. No cheaper
        # invalidation is sound: detecting a new tensor/container slot
        # requires type-inspecting every item, which IS the rebuild, so
        # the filter runs fresh each walk (one isinstance pass per
        # reachable module namespace per session end).
        slot_names = tuple(
            name
            for name, item in namespace.items()
            if not _is_isinstance_hostile_deprecation_shim(item)
            and isinstance(
                item,
                (torch.Tensor, dict, list, tuple, set, frozenset, deque),
            )
        )
        for name in slot_names:
            item = namespace.get(name)
            if isinstance(item, torch.Tensor) and not isinstance(item, torch.nn.Parameter):
                action(item)
            elif isinstance(item, (dict, list, tuple, set, frozenset, deque)):
                _clear_container_tree_tensor_metadata(item, seen, depth + 1, visit)


def _clear_reachable_tensor_metadata(
    value: Any,
    seen: set[int],
    depth: int,
    visit: Callable[[torch.Tensor], None] | None,
) -> None:
    """Recurse into the children of a container or plain object, skipping scalar leaves."""

    # Scalar leaves cannot carry tl_* stamps; skipping them INLINE (instead of
    # paying a full call that immediately returns) halves the cost of walking
    # broadly-imported module namespaces like ``torch`` (round-3 b4 F1: the
    # pre-forward ownership snapshot made every capture pay this walk twice).
    # EXACT type() membership on purpose: isinstance() falls back to reading
    # ``__class__`` on a non-matching type, which the reduce_op deprecation
    # shim answers with a FutureWarning (SF-45) -- the recursion's entry guard
    # never runs for inline-skipped items, so this check must stay hostile-safe.
    # Non-exact scalar subclasses fall through to the guarded recursive call.
    _scalar_leaves = (str, bytes, int, float, bool)
    if isinstance(value, dict):
        for key, item in value.items():
            if key is not None and type(key) not in _scalar_leaves:
                _clear_session_tensor_metadata(key, seen, depth + 1, visit)
            if item is not None and type(item) not in _scalar_leaves:
                _clear_session_tensor_metadata(item, seen, depth + 1, visit)
        return
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        for item in value:
            if item is not None and type(item) not in _scalar_leaves:
                _clear_session_tensor_metadata(item, seen, depth + 1, visit)
        return
    if isinstance(value, nn.Module):
        return
    namespace = getattr(value, "__dict__", None)
    if namespace is None:
        return
    for item in namespace.values():
        if item is not None and type(item) not in _scalar_leaves:
            _clear_session_tensor_metadata(item, seen, depth + 1, visit)


def _clear_container_tree_tensor_metadata(
    value: Any,
    seen: set[int],
    depth: int,
    visit: Callable[[torch.Tensor], None] | None = None,
) -> None:
    """Clear tensor metadata from a pure container tree (no object descent).

    Restricted companion to ``_clear_session_tensor_metadata`` for
    ``types.ModuleType`` namespaces: walks plain containers and clears
    non-Parameter tensors, but never descends into objects, ``nn.Module``
    instances, or nested modules, so sweeping a broadly-imported module's
    namespace cannot fan out through the whole process object graph.

    Parameters
    ----------
    value
        Container member reached from a module-namespace container.
    seen
        Object ids already visited during this cleanup scan.
    depth
        Current recursion depth, shared with the main walk's bound.

    Returns
    -------
    None
        Mutates reachable tensors in place by removing TorchLens metadata.
    """

    if _is_isinstance_hostile_deprecation_shim(value):
        return
    if isinstance(value, torch.Tensor):
        if not isinstance(value, torch.nn.Parameter):
            (clear_meta if visit is None else visit)(value)
        return
    if not isinstance(value, (dict, list, tuple, set, frozenset, deque)):
        return
    obj_id = id(value)
    if obj_id in seen or depth >= 12:
        return
    seen.add(obj_id)
    items: Iterable[Any]
    if isinstance(value, dict):
        items = [item for pair in value.items() for item in pair]
    else:
        items = list(value)
    # Inline scalar-leaf skip: module-namespace container trees are dominated
    # by strings (``torch.__all__`` alone is ~1400), and each full call here
    # costs more than the check (round-3 b4 F1 walk-cost finding). EXACT
    # type() membership on purpose: isinstance() reads ``__class__`` on a
    # non-matching type, which the reduce_op deprecation shim answers with a
    # FutureWarning (SF-45); subclasses fall through to the guarded recursion.
    for item in items:
        if item is None or type(item) in (str, bytes, int, float, bool):
            continue
        _clear_container_tree_tensor_metadata(item, seen, depth + 1, visit)


def _forward_wrapper_chain(raw_callable: Any) -> list[Any]:
    """Follow TorchLens forward wrappers inward, returning every link ending at user code."""

    chain: list[Any] = []
    chain_ids: set[int] = set()
    current = raw_callable
    while current is not None and id(current) not in chain_ids:
        chain_ids.add(id(current))
        chain.append(current)
        if not is_forward_call_decorated(current):
            break
        wrapped = getattr(current, "__wrapped__", None)
        current = None if wrapped is None else getattr(wrapped, "__func__", wrapped)
    return chain


def _clear_callable_link_tensor_metadata(
    link: Any,
    seen: set[int],
    visit: Callable[[torch.Tensor], None] | None,
) -> None:
    """Sweep one callable's defaults, keyword defaults and closure cells."""

    defaults = getattr(link, "__defaults__", None) or ()
    _clear_session_tensor_metadata(defaults, seen, visit=visit)
    kwdefaults = getattr(link, "__kwdefaults__", None) or {}
    _clear_session_tensor_metadata(kwdefaults, seen, visit=visit)
    closure = getattr(link, "__closure__", None) or ()
    for cell in closure:
        try:
            cell_value = cell.cell_contents
        except ValueError:
            continue
        _clear_session_tensor_metadata(cell_value, seen, visit=visit)


def _clear_callable_session_tensor_metadata(
    callable_obj: Any,
    seen: set[int],
    visit: Callable[[torch.Tensor], None] | None = None,
) -> None:
    """Clear TorchLens tensor metadata captured by a callable object.

    The globals scan targets the callable that actually runs USER code, so
    TorchLens's own ``module_forward_decorator`` wrapper is unwrapped first (via
    ``functools.wraps``' ``__wrapped__``, the same link
    :func:`_restore_undecorated_forward` uses). The wrapper is defined in THIS
    module, so scanning ITS ``__code__.co_names`` against ITS ``__globals__``
    describes TorchLens internals -- ``_state``'s decoration registries and
    ``sys.modules`` -- and never the user's forward. That was both a coverage gap
    (every decorated submodule's real globals went unscanned) and the whole cost
    of the per-capture namespace sweep: those two roots alone accounted for
    ~33.7 k of the 33.7 k container visits on a 3-op model and ~35.1 k of 35.1 k
    on resnet50, reaching zero tensors. Only the globals scan moves inward;
    defaults, keyword defaults and closures are still swept at EVERY link of the
    chain, so nothing a wrapper legitimately captures is skipped. A callable that
    is not a TorchLens forward wrapper is inspected exactly as before.

    Parameters
    ----------
    callable_obj
        Candidate callable, such as a model's bound ``forward`` method.
    seen
        Object ids already visited during this cleanup scan.

    Returns
    -------
    None
        Mutates reachable tensor metadata in place.
    """

    chain = _forward_wrapper_chain(getattr(callable_obj, "__func__", callable_obj))
    for link in chain:
        _clear_callable_link_tensor_metadata(link, seen, visit)

    user_callable = chain[-1]
    globals_dict = getattr(user_callable, "__globals__", None)
    if not isinstance(globals_dict, dict):
        return
    code = getattr(user_callable, "__code__", None)
    if code is None:
        return
    for name in code.co_names:
        if name in globals_dict:
            _clear_session_tensor_metadata(globals_dict[name], seen, visit=visit)
