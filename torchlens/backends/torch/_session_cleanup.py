"""Session-end tensor metadata cleanup for torch capture.

Every tensor a capture session touched carries TorchLens metadata until the
session ends. The walkers here clear it from the model tree, from the
containers and callables reachable through module attributes, and from the
globals of decorated forwards, so no tensor outlives its session carrying
provenance a later capture could mistake for its own (split out of
``model_prep.py`` under the file-size ratchet; this is the teardown seam,
not preparation). ``model_prep`` re-exports the ``_clear_*`` walkers
under their original names.
"""

import functools
import gc
import logging
import sys
import types
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from types import ModuleType
from typing import Any, TypeAlias

import torch
from torch import nn

from ..._utils._model_walk import model_modules
from ..._utils._tensor_label import clear_meta, is_forward_call_decorated

__all__ = [
    "_clear_callable_session_tensor_metadata",
    "_clear_container_tree_tensor_metadata",
    "_clear_module_attribute_tensor_metadata",
    "_clear_model_tree_tensor_metadata",
    "_clear_module_callables_tensor_metadata",
    "_clear_session_tensor_metadata",
    "method_index",
]

#: The model's user methods by attribute name (see :func:`method_index`).
MethodIndex: TypeAlias = Mapping[str, tuple[types.FunctionType, ...]]

#: Exact leaf types that can never carry TorchLens tensor metadata.
_SCALAR_LEAF_TYPES = frozenset({str, bytes, int, float, bool})
#: Exact plain container types whose emptiness is a sound inline skip.
_PLAIN_CONTAINER_TYPES = frozenset({dict, OrderedDict, list, tuple, set, frozenset, deque})
_SCALAR_SEQUENCE_TYPES = frozenset({list, tuple})
#: The C ``functools.lru_cache`` wrapper type (``None`` where only the pure-Python
#: fallback exists; its cache lives in a closure cell the callable walk already reads).
_LRU_CACHE_WRAPPER_TYPE: type | None = getattr(functools, "_lru_cache_wrapper", None)
#: Descriptor wrappers whose ``__func__`` may be a cached function (``@staticmethod @lru_cache``).
_METHOD_DESCRIPTOR_TYPES = frozenset({staticmethod, classmethod})
#: Referents of a cache wrapper the sweep never walks: its type and the wrapped function.
_LRU_REFERENT_SKIP_TYPES = frozenset({type, types.FunctionType})
#: The process logging registry. A reachable ``Logger`` or ``Handler`` leads through
#: ``manager.loggerDict`` to every logger and handler in the process, so the walk's size
#: followed which packages had been imported (``import accelerate`` alone added container
#: builds to every capture of a transformers model). Logging objects are process plumbing,
#: not model state: the walk stops at them, as it never recurses out of a module namespace.
#: ``nn.Module`` instances stop the object walk too, as before.
_OBJECT_WALK_STOP_TYPES = (nn.Module, logging.Filterer, logging.Manager, logging.PlaceHolder)
#: Function hops followed from a forward into the user helpers it names (the MLX
#: closure-constant census follows the same number).
_HELPER_HOP_BUDGET = 3
#: ``functools.wraps`` links followed inward from one function (decorator stacks).
_WRAPPED_LINK_LIMIT = 8
#: Top-level packages whose functions are library code, never a user helper keeping
#: model state: torch (and TorchLens's own wrappers, which copy torch's
#: ``__module__``), TorchLens and the standard library.
_LIBRARY_PACKAGE_ROOTS = frozenset(
    {"torch", "torchlens", "builtins", *getattr(sys, "stdlib_module_names", ())}
)
#: Exact scalar leaf types the walk never descends into, and the sequence types it
#: iterates. Module constants, not tuple literals at the call site: a literal of
#: names is built again on every visit, and the walk visits every reachable object
#: of every forward and helper twice per capture (snapshot and clear).
_WALK_SCALAR_TYPES = (str, bytes, int, float, bool)
_WALK_SEQUENCE_TYPES = (list, tuple, set, frozenset, deque)
# Exact types a module namespace holds that can never be a tensor or a
# container: skipping them by ``type()`` avoids the ``isinstance`` (and the
# reduce_op shim check) for the bulk of ``torch``'s ~2k names on every sweep.
_NAMESPACE_INERT_TYPES = frozenset(
    {
        types.BuiltinFunctionType,
        types.FunctionType,
        types.ModuleType,
        type,
        str,
        bytes,
        int,
        float,
        bool,
        type(None),
    }
)


def _is_isinstance_hostile_deprecation_shim(value: Any) -> bool:
    """Return whether ``value`` is torch's ``reduce_op`` deprecation singleton.

    ``torch.distributed.reduce_op`` is a ``_reduce_op`` instance whose
    ``__getattribute__`` emits a ``FutureWarning`` on ANY attribute access --
    including the ``__class__`` read every ``isinstance()`` performs -- so an
    initialized distributed process running under ``-W error`` had its capture
    CLEANUP aborted by the namespace walks below. ``type`` bypasses
    the instance's ``__getattribute__``, so this exact-name check is silent;
    structural matching (not an import of the private class) follows the
    ``_distributed.py`` fallback convention.
    """

    value_type = type(value)
    return (
        value_type.__name__ == "_reduce_op"
        and value_type.__module__ == "torch.distributed.distributed_c10d"
    )


def _clear_module_attribute_tensor_metadata(
    module: nn.Module,
    seen: set[int],
    visit: Callable[[torch.Tensor], None] | None = None,
) -> None:
    """Walk one module's own ``__dict__`` values through the session walker.

    The shared entry of the pre-forward ownership snapshot
    (:func:`_module_boundary_adoption.collect_pre_forward_tensor_ids`) and the
    session-end clear (``model_prep._undecorate_model_tensors``), so both sweep
    the same surfaces. Leaves that cannot carry a stamp are skipped INLINE:
    exact scalar types, and EMPTY plain containers (the hook ``OrderedDict``
    slots every ``nn.Module`` owns are the bulk of a module namespace, and a
    full walker call per empty slot cost a one-module ``tl.record`` more than
    the sweep found). An empty container is also never memoised in ``seen``:
    revisiting it is the same no-op. EXACT ``type()`` membership on purpose,
    as in the walker (the reduce_op shim answers ``isinstance`` with a
    FutureWarning); subclasses take the guarded walker call.

    Parameters
    ----------
    module
        One module of the prepared model tree.
    seen
        Object ids already visited during this sweep.
    visit
        Optional per-tensor action replacing the default ``clear_meta``.

    Returns
    -------
    None
        Mutates reachable tensors in place (or applies ``visit``).
    """

    for attr_val in module.__dict__.values():
        if attr_val is None:
            continue
        value_type = type(attr_val)
        if value_type in _SCALAR_LEAF_TYPES:
            continue
        if value_type in _PLAIN_CONTAINER_TYPES and not attr_val:
            continue
        if value_type in _SCALAR_SEQUENCE_TYPES and all(
            type(item) in _SCALAR_LEAF_TYPES for item in attr_val
        ):
            continue  # shape tuples such as ``kernel_size``: no stamp can hide in scalars
        _clear_session_tensor_metadata(attr_val, seen, visit=visit)


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
        (:func:`_module_boundary_adoption._collect_model_owned_tensor_ids`) reuses this exact traversal
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
    if isinstance(value, _WALK_SCALAR_TYPES):
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

    # A stamped tensor stashed as a DIRECT attribute of a ``types.ModuleType``
    # must be reached by this walk, and a stamp nested inside a plain
    # CONTAINER in the module namespace would escape a shallow sweep and
    # launder provenance across sessions, so namespace containers get
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
        # No slot memo: a memo invalidated on ``len(namespace)`` only would
        # miss REPLACING a value at a constant length (``mod.stash = 5``
        # becoming ``mod.stash = tensor``), keep the stale slot list, and let
        # the stamped tensor escape the session clear -- provenance laundering across
        # sessions. No cheaper
        # invalidation is sound: detecting a new tensor/container slot
        # requires type-inspecting every item, which IS the rebuild, so
        # the filter runs fresh each walk (one isinstance pass per
        # reachable module namespace per session end).
        slot_names = tuple(
            name
            for name, item in namespace.items()
            if type(item) not in _NAMESPACE_INERT_TYPES
            and not _is_isinstance_hostile_deprecation_shim(item)
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
    # broadly-imported module namespaces like ``torch`` (the pre-forward
    # ownership snapshot makes every capture pay this walk twice).
    # EXACT type() membership on purpose: isinstance() falls back to reading
    # ``__class__`` on a non-matching type, which the reduce_op deprecation
    # shim answers with a FutureWarning -- the recursion's entry guard
    # never runs for inline-skipped items, so this check must stay hostile-safe.
    # Non-exact scalar subclasses fall through to the guarded recursive call.
    _scalar_leaves = _WALK_SCALAR_TYPES
    if isinstance(value, dict):
        for key, item in value.items():
            if key is not None and type(key) not in _scalar_leaves:
                _clear_session_tensor_metadata(key, seen, depth + 1, visit)
            if item is not None and type(item) not in _scalar_leaves:
                _clear_session_tensor_metadata(item, seen, depth + 1, visit)
        return
    if isinstance(value, _WALK_SEQUENCE_TYPES):
        for item in value:
            if item is not None and type(item) not in _scalar_leaves:
                _clear_session_tensor_metadata(item, seen, depth + 1, visit)
        return
    if isinstance(value, _OBJECT_WALK_STOP_TYPES):
        return
    if _LRU_CACHE_WRAPPER_TYPE is not None and type(value) is _LRU_CACHE_WRAPPER_TYPE:
        for entry in _lru_cache_entries(value):
            _clear_session_tensor_metadata(entry, seen, depth + 1, visit)
        return
    namespace = getattr(value, "__dict__", None)
    if namespace is None:
        return
    for name, item in namespace.items():
        # ``__annotations__`` holds classes, strings and typing forms, never a
        # stamped tensor; descending into it walked the whole ``torch.Tensor``
        # class namespace (every wrapped method) from each module ``forward``.
        if name == "__annotations__":
            continue
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
    # costs more than the check (a measured walk cost). EXACT
    # type() membership on purpose: isinstance() reads ``__class__`` on a
    # non-matching type, which the reduce_op deprecation shim answers with a
    # FutureWarning; subclasses fall through to the guarded recursion.
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
    kwdefaults = getattr(link, "__kwdefaults__", None)
    if kwdefaults:
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
    methods: MethodIndex | None = None,
    swept: dict[int, Any] | None = None,
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
    visit
        Optional per-tensor action replacing the default ``clear_meta``.
    methods
        The model's user methods by name (:func:`method_index`), so a
        ``self.<method>`` or ``self.child.<method>`` the forward names is followed
        as a helper (see :func:`_walk_named_globals`).
    swept
        This sweep's memo (:func:`_clear_module_callables_tensor_metadata`), or ``None``.

    Returns
    -------
    None
        Mutates reachable tensor metadata in place.
    """

    chain = _forward_wrapper_chain(getattr(callable_obj, "__func__", callable_obj))
    if swept is not None and id(chain[-1]) in swept:
        # TorchLens's own wrapper links differ per module and are still swept.
        for link in chain[:-1]:
            _clear_callable_link_tensor_metadata(link, seen, visit)
        return
    for link in chain:
        _clear_callable_link_tensor_metadata(link, seen, visit)
    # A user decorator on the forward (``@beartype``, a ``functools.wraps`` gradient
    # switch) is transparent: its names are read at the full budget, not one hop down.
    for link in _wrapped_links(chain[-1]):
        _clear_callable_link_tensor_metadata(link, seen, visit)
        _walk_named_globals(link, methods, seen, visit, _HELPER_HOP_BUDGET)
    if swept is not None:
        swept[id(chain[-1])] = chain[-1]  # recorded only once the whole walk has run


def _code_names(code: types.CodeType) -> Iterator[str]:
    """Yield the names a code object and its nested code (comprehensions, lambdas) load.

    Depth first, the forward's own names first and nested code last-to-first. A
    code object's constants are an acyclic compile-time tree, so the walk needs no
    stack or visited set; it builds no container because it runs for every forward
    and helper the walk follows, twice per capture (snapshot and clear).
    """

    yield from code.co_names
    for const in reversed(code.co_consts):
        if type(const) is types.CodeType:
            yield from _code_names(const)


def _user_helper(value: Any) -> types.FunctionType | None:
    """Return ``value`` when it is a plain function defined outside library packages."""

    if type(value) is not types.FunctionType:
        return None
    module = getattr(value, "__module__", None)
    root = module.partition(".")[0] if isinstance(module, str) else ""
    return None if root in _LIBRARY_PACKAGE_ROOTS else value


def _wrapped_links(func: Any) -> list[Any]:
    """Return ``func`` and the user functions its ``functools.wraps`` chain wraps, outermost first.

    A decorator built with ``functools.wraps`` (``torch.no_grad()``, ``@beartype``, an
    autocast pin) holds the decorated function in a closure cell the wrapper's code
    never names, and records it as ``__wrapped__``. Following that link keeps a
    decorated helper as visible as an undecorated one. ``__wrapped__`` is read from
    the function's own ``__dict__`` so no user ``__getattr__`` runs; library links
    are not followed (see :func:`_user_helper`).
    """

    links = [func]
    current = (
        getattr(func, "__dict__", {}).get("__wrapped__")
        if type(func) is types.FunctionType
        else None
    )
    while current is not None and len(links) < _WRAPPED_LINK_LIMIT:
        if type(current) is types.MethodType:
            current = current.__func__
        if _user_helper(current) is None or any(current is link for link in links):
            break
        links.append(current)
        current = current.__dict__.get("__wrapped__")
    return links


#: The ``torch.autograd.Function`` methods ``apply`` runs during the forward.
_AUTOGRAD_FUNCTION_STEPS = ("forward", "setup_context")


def _autograd_function_steps(value: Any) -> tuple[types.FunctionType, ...]:
    """Return the user ``forward``/``setup_context`` a named ``autograd.Function`` class runs.

    ``MyFunction.apply(x)`` runs the class's static ``forward`` from inside torch, so
    the module-level state that ``forward`` reads (L-GATr's sparse geometric product
    reads an ``lru_cache`` of index and sign tables) is reached through no function
    the forward names. Only user subclasses count; torch's own are library code.
    """

    if not (
        isinstance(value, type)
        and issubclass(value, torch.autograd.Function)
        and _user_class(value)
    ):
        return ()
    steps = (_class_helper(value, name) for name in _AUTOGRAD_FUNCTION_STEPS)
    return tuple(step for step in steps if step is not None)


def _class_helper(owner: type, name: str) -> types.FunctionType | None:
    """Return the user function ``owner`` resolves ``name`` to, ``self.<name>``'s method."""

    for klass in owner.__mro__:
        attr = vars(klass).get(name)
        if attr is None:
            continue
        if type(attr) in _METHOD_DESCRIPTOR_TYPES:
            attr = attr.__func__
        return _user_helper(attr)
    return None


def method_index(classes: Iterable[type]) -> MethodIndex:
    """Index the user methods of the model's module classes by attribute name.

    A forward reaches module-level state through methods of its own class
    (``self.forward_once``) and of its children's classes
    (``self.SO3_rotation[i].set_wigner`` calling a module-level ``wigner_D``).
    Each class resolves a name through its MRO as an instance would; library
    classes (``nn.Module`` and the rest of torch) contribute nothing.

    Parameters
    ----------
    classes
        The module classes of the prepared model, in any order (duplicates ignored).

    Returns
    -------
    MethodIndex
        ``name -> user functions`` that some class in ``classes`` resolves ``name`` to.
    """

    index: dict[str, list[types.FunctionType]] = {}
    for owner in dict.fromkeys(classes):
        user_classes = [klass for klass in owner.__mro__ if _user_class(klass)]
        for name in dict.fromkeys(name for klass in user_classes for name in vars(klass)):
            helper = _class_helper(owner, name)
            if helper is not None and not any(helper is fn for fn in index.get(name, ())):
                index.setdefault(name, []).append(helper)
    return {name: tuple(functions) for name, functions in index.items()}


def _user_class(klass: type) -> bool:
    """Return whether ``klass`` is defined outside the library packages."""

    module = getattr(klass, "__module__", None)
    root = module.partition(".")[0] if isinstance(module, str) else ""
    return root not in _LIBRARY_PACKAGE_ROOTS


def _walk_named_globals(
    func: Any,
    methods: MethodIndex | None,
    seen: set[int],
    visit: Callable[[torch.Tensor], None] | None,
    hops: int,
) -> None:
    """Sweep the globals ``func``'s code names, following user helpers for ``hops`` hops.

    A forward often reaches module-level state only through the helpers it calls:
    ``forward -> hyp_up -> disp_up`` reading a module-level ``lru_cache`` function,
    or ``forward -> flow_warp -> torch_warp`` reading a module-global dict of
    cached grids. Reading only the forward's own names left that state outside
    the snapshot (a warm cache's tensor became an unattributed argument) and
    outside the session clear (a cold cache filled in one capture kept a stale
    label into the next). A named user function, or a model-class method the
    code names (``self.forward_once``, ``self.child.set_wigner``; see
    :func:`method_index`), is swept like the forward itself: defaults, closure
    cells, its ``__dict__`` and the globals its code names, one hop less each
    time. A ``functools.wraps`` decorator costs no hop (:func:`_wrapped_links`).
    Library functions (torch, TorchLens, the standard library) are not
    followed; at zero hops a function is swept as an object.
    """

    globals_dict = getattr(func, "__globals__", None)
    code = getattr(func, "__code__", None)
    if not isinstance(globals_dict, dict) or code is None:
        return
    for name in _code_names(code):
        if methods is not None and hops > 0:
            for method in methods.get(name, ()):
                _walk_user_helper(method, methods, seen, visit, hops)
        if name not in globals_dict:
            continue
        value = globals_dict[name]
        helper = _user_helper(value) if hops > 0 else None
        if helper is None:
            _clear_session_tensor_metadata(value, seen, visit=visit)
            for apply_step in _autograd_function_steps(value) if hops > 0 else ():
                _walk_user_helper(apply_step, methods, seen, visit, hops)
        else:
            _walk_user_helper(helper, methods, seen, visit, hops)


def _walk_user_helper(
    func: types.FunctionType,
    methods: MethodIndex | None,
    seen: set[int],
    visit: Callable[[torch.Tensor], None] | None,
    hops: int,
) -> None:
    """Sweep one user helper function once per budget, its wrapped links, then their names."""

    # The inverted key keeps the helper walk apart from the plain object visit of the
    # same function (``seen`` holds object ids, all non-negative), so a function met
    # first as a module attribute is still followed when a forward names it. The
    # budget is part of the key: a helper first met deep in one chain (few hops
    # left) is walked again when a shorter chain reaches it with more hops.
    walk_key = ~(id(func) * (_HELPER_HOP_BUDGET + 1) + hops)
    if walk_key in seen:
        return
    seen.add(walk_key)
    for link in _wrapped_links(func):
        _clear_callable_link_tensor_metadata(link, seen, visit)
        _clear_session_tensor_metadata(link, seen, visit=visit)
        _walk_named_globals(link, methods, seen, visit, hops - 1)


def _lru_cache_entries(wrapper: Any) -> list[Any]:
    """Return what a C ``lru_cache`` wrapper holds: cached keys, results and its ``__dict__``.

    The C wrapper keeps its cache off every attribute: a warm cache returns a
    tensor built in an earlier pass with no op and no attribute read, so neither
    the ownership snapshot nor the session-end clear saw it. Its ``tp_traverse``
    visits every cache link's key and result, the cache dict and the wrapper's
    ``__dict__`` without running user code. The wrapped function (its globals are
    walked from the forward that names it) and the wrapper's type are dropped.
    """

    return [
        referent
        for referent in gc.get_referents(wrapper)
        if type(referent) not in _LRU_REFERENT_SKIP_TYPES
    ]


def _class_lru_caches(owner: type) -> Iterator[Any]:
    """Yield the C ``lru_cache`` wrappers the user classes of ``owner``'s MRO hold.

    A cached class attribute (``@staticmethod @lru_cache``, ``@lru_cache`` on a
    method) is the wrapper itself; a decorator that wraps the cache in a Python
    function (transformers' ``compile_compatible_method_lru_cache``) holds it in a
    closure cell. Classes defined by torch hold no model cache and are skipped.
    Runs for every module of every sweep, so it builds no container: it yields
    instead of collecting, and its type tests read module-level frozensets
    rather than tuple literals built per attribute.
    """

    if _LRU_CACHE_WRAPPER_TYPE is None:
        return
    for klass in owner.__mro__:
        if klass is object or str(getattr(klass, "__module__", "")).startswith("torch."):
            continue
        for attr in vars(klass).values():
            if type(attr) in _METHOD_DESCRIPTOR_TYPES:
                attr = attr.__func__
            if type(attr) is _LRU_CACHE_WRAPPER_TYPE:
                yield attr
            elif type(attr) is types.FunctionType:
                for cell in attr.__closure__ or ():
                    try:
                        held = cell.cell_contents
                    except ValueError:
                        continue
                    if type(held) is _LRU_CACHE_WRAPPER_TYPE:
                        yield held


def _clear_module_callables_tensor_metadata(
    module: nn.Module,
    seen: set[int],
    visit: Callable[[torch.Tensor], None] | None = None,
    methods: MethodIndex | None = None,
    swept: dict[int, Any] | None = None,
) -> None:
    """Walk one module's forward callable and its helpers, then its class's ``lru_cache`` contents.

    The callable half of the pre-forward ownership snapshot and of the session-end
    clear, so both reach the same cached tensors: the snapshot records a warm
    cache's tensor as a pre-forward constant, and the clear strips the label a
    capture left on a tensor it computed into a cold cache.

    Parameters
    ----------
    module
        One module of the prepared model tree.
    seen
        Object ids already visited during this sweep.
    visit
        Optional per-tensor action replacing the default ``clear_meta``.
    methods
        The whole model's :func:`method_index`, built once per sweep; ``None``
        indexes ``module``'s own class only.
    swept
        The user callables this sweep has already walked, by id (the callable is
        the value, so its id cannot be reused while the sweep runs). A model's
        modules mostly share one class ``forward``; walking it again with the
        same ``seen``, ``visit`` and ``methods`` reaches nothing new (containers
        and helpers are in ``seen``, and a tensor's action is idempotent), so
        the repeat is skipped. Pass one dict per sweep, with that sweep's one
        ``seen``, ``visit`` and whole-model ``methods``; it is ignored when
        ``methods`` is ``None`` (a per-class index differs between modules).
    """

    _clear_callable_session_tensor_metadata(
        getattr(module, "forward", None),
        seen,
        visit=visit,
        methods=method_index((type(module),)) if methods is None else methods,
        swept=None if methods is None else swept,
    )
    for cache in _class_lru_caches(type(module)):
        _clear_session_tensor_metadata(cache, seen, visit=visit)


def _clear_model_tree_tensor_metadata(model: nn.Module, seen: set[int]) -> None:
    """Clear the session metadata on every submodule's attributes and forward callables.

    The session-end half of the sweep :func:`._module_boundary_adoption.
    collect_pre_forward_tensor_ids` runs before the forward, over the same
    surfaces with the same :func:`method_index`, so the clear reaches every
    cached tensor the snapshot pinned.
    """

    submodules = list(model_modules(model))
    methods = method_index(type(submodule) for submodule in submodules)
    swept: dict[int, Any] = {}
    for submodule in submodules:
        _clear_module_attribute_tensor_metadata(submodule, seen)
        _clear_module_callables_tensor_metadata(submodule, seen, methods=methods, swept=swept)
