"""Argument copying, normalization, and input validation for model forward calls.

Provides safe argument copying that avoids ``copy.deepcopy`` (which can
trigger infinite loops on complex tensor wrappers with circular references,
e.g. ESCNN GeometricTensor) and normalizes user-supplied ``input_args`` into
the ``list`` form expected by ``model(*input_args)``.
"""

import copy
import inspect
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast

import torch
from torch import nn

from .._input_walk import INPUT_TREE_MAX_DEPTH, _inspect_instance_state_items
from .._state import ReentrantTraceError
from .alias_footprint import (
    TensorByteFootprint,
    tensor_byte_footprint,
    touched_bytes_relation,
)
from .tensor_utils import (
    _clone_tensor_payload,
    _copy_tensor_payload,
)

INPUT_WAS_PARAMETER_ATTR = "_torchlens_input_was_parameter"

_PRESERVE_LEAF_INPUT_COPIES: ContextVar[bool] = ContextVar(
    "torchlens_preserve_leaf_input_copies", default=False
)
"""Whether :func:`safe_copy_input_tree` keeps grad-tracked leaf inputs as leaves by default."""


@contextmanager
def leaf_preserving_input_copies() -> Iterator[None]:
    """Make every model-input copy inside the block keep leaf inputs as leaves.

    Validation copies the caller's inputs for its private forwards (ground truth,
    validation capture, re-trace). A grad-tracked leaf input copied through the
    graph-connected clone becomes a NON-leaf, so a model that sets
    ``x.requires_grad = True`` on its input raises inside validation although its
    plain forward runs. Validation never promises that gradients reach the
    caller's input, so inside this block a leaf input is copied as a new leaf with
    the same ``requires_grad``. A plain ``tl.trace`` keeps the graph-connected
    clone, whose contract is that gradients still reach the caller's input.
    """

    token = _PRESERVE_LEAF_INPUT_COPIES.set(True)
    try:
        yield
    finally:
        _PRESERVE_LEAF_INPUT_COPIES.reset(token)


def leaf_preserving_input_copies_active() -> bool:
    """Return whether the current block copies and moves leaf inputs as leaves.

    Returns
    -------
    bool
        True inside :func:`leaf_preserving_input_copies`.
    """

    return _PRESERVE_LEAF_INPUT_COPIES.get()


def _clone_input_tensor_payload(arg: torch.Tensor, *, preserve_leaf: bool = False) -> torch.Tensor:
    """Clone a forward-input tensor without preserving ``nn.Parameter`` type.

    Parameters
    ----------
    arg
        Forward input tensor or parameter to clone.
    preserve_leaf
        Whether a grad-tracked leaf ``nn.Parameter`` input is copied as a new
        leaf with the same ``requires_grad`` (validation's private copies; see
        :func:`leaf_preserving_input_copies`) instead of the graph-connected clone.

    Returns
    -------
    torch.Tensor
        Plain tensor clone for model input replay. Parameter inputs intentionally
        become tensors so input provenance wins over Python wrapper type.
    """

    if isinstance(arg, torch.nn.Parameter):
        from .._state import pause_logging

        detach_leaf = preserve_leaf and arg.is_leaf and arg.requires_grad
        with pause_logging():
            cloned = _copy_tensor_payload(arg, detach_tensor=detach_leaf, save_mode="copy")
            if detach_leaf:
                cloned.requires_grad_(True)
        setattr(cloned, INPUT_WAS_PARAMETER_ATTR, True)
        return cloned
    return cast(torch.Tensor, _clone_tensor_payload(arg, detach_tensor=False, save_mode="copy"))


def _is_structseq_type(arg_type: type[Any]) -> bool:
    """Return whether ``arg_type`` is a C ``PyStructSequence`` class.

    Structseq classes (``torch.return_types.*``, ``os.stat_result``, ...)
    carry the C-level ``n_fields``/``n_sequence_fields`` layout counts that
    ordinary namedtuples and Python tuple subclasses never define. A Python
    subclass faking these attributes merely gets the single-iterable probe
    first; the exact-identity verification still decides correctness.

    Parameters
    ----------
    arg_type:
        Exact tuple subclass under reconstruction.

    Returns
    -------
    bool
        ``True`` when the class declares the structseq field-count layout.
    """

    return isinstance(getattr(arg_type, "n_fields", None), int) and isinstance(
        getattr(arg_type, "n_sequence_fields", None), int
    )


def rebuild_tuple_like(arg_type: type[Any], items: list[Any]) -> Any:
    """Rebuild a ``tuple`` subclass from ``items``, or ``None`` if impossible.

    ``_fields`` presence is NOT proof of a namedtuple positional constructor
    (T11.7): a tuple subclass exposing ``_fields`` as a list or property
    crashed the capture entry normalizers with an untyped ``TypeError``, and
    a subclass without ``_fields`` whose ``__new__`` is not single-iterable
    crashed the other arm. Mirror the ``_move_tensors_to_device`` ladder --
    try the namedtuple positional constructor, then the single-iterable
    ``__new__`` shape (structseq / ``torch.Size``) -- and additionally verify
    the rebuilt instance carries the EXACT item identities in order (read
    inertly through ``tuple.__getitem__``): ``arg_type(*items)`` reaching a
    single-iterable constructor with one iterable item would otherwise
    silently EXPAND that item into its elements.

    Structseq classes skip the positional probe entirely: the C constructor
    shape is ``(sequence, dict)``, so ``arg_type(*items)`` on a tensor-bearing
    structseq (``torch.return_types.sort``) put the values TENSOR in the
    sequence slot and ITERATED it -- dispatching real ``dim``/``unbind`` calls
    -- before raising. Under active logging that probe side effect was
    CAPTURED as a spurious ``unbind`` op, making otherwise-identical traces
    structurally diverge (the structseq non-reproducibility regression). The
    probes additionally run under ``pause_logging()``: rebuilding a snapshot
    container is TorchLens-internal bookkeeping the user's program never
    executed, so no constructor side effect may ever enter the captured graph.

    Parameters
    ----------
    arg_type:
        Exact tuple subclass to rebuild.
    items:
        Child values, in physical tuple order.

    Returns
    -------
    Any
        Verified rebuilt instance, or ``None`` when no constructor shape
        reproduces the items (callers fall back without crashing).
    """

    from .._state import pause_logging

    if _is_structseq_type(arg_type):
        builds: tuple[Any, ...] = (lambda: arg_type(items),)
    else:
        builds = (lambda: arg_type(*items), lambda: arg_type(items))
    for build in builds:
        try:
            with pause_logging():
                candidate = build()
        except ReentrantTraceError:
            # ``build()`` runs the USER's subclass constructor: a nested
            # public capture attempted there is the R55 typed refusal, not a
            # rebuild failure -- absorbing it into the fallback ladder would
            # hide the program error the refusal exists to surface.
            raise
        except Exception:
            continue
        if type(candidate) is not arg_type:
            continue
        try:
            length = tuple.__len__(candidate)
        except Exception:
            continue
        if length == len(items) and all(
            tuple.__getitem__(candidate, index) is items[index] for index in range(length)
        ):
            return candidate
    return None


_PY_TPFLAGS_HEAPTYPE = 1 << 9
"""``Py_TPFLAGS_HEAPTYPE``: set on Python-defined classes, clear on static C types."""

_TRUSTED_MAPPING_BASES = (dict, OrderedDict, defaultdict, Counter)
"""Stock dict-backed bases whose C-level extras the rebuild ladder handles by name."""


def _inert_state_enumeration_total(cls: type[Any], trusted_bases: tuple[type[Any], ...]) -> bool:
    """Return whether ``__dict__``/slots enumeration provably covers ``cls`` state.

    A static (C-extension) base outside ``trusted_bases`` can carry C-level
    instance state (a ``defaultdict.default_factory`` analogue) that the inert
    inspector cannot enumerate; rebuilding such a class from ``__dict__`` +
    slots would silently RESET that state -- the exact substitution class the
    inert-rebuild contract exists to prevent. Python-defined classes (heap
    types) keep all their state in ``__dict__``/slots by construction.
    """

    for base in cls.__mro__:
        if base is object or base in trusted_bases:
            continue
        if not (base.__flags__ & _PY_TPFLAGS_HEAPTYPE):
            return False
    return True


def _copy_instance_state_inertly(original: Any, rebuilt: Any) -> bool:
    """Copy ``original``'s enumerable instance state onto ``rebuilt`` verbatim.

    Mirrors the dataclass device-move arm: raw-channel enumeration through
    :func:`torchlens._input_walk._inspect_instance_state_items` and
    ``object.__setattr__`` writes, never a live attribute protocol. Also copies
    the one trusted C-level extra the ladder knows by name
    (``defaultdict.default_factory``, via its member descriptor). Returns
    ``False`` when the enumeration cannot be inertly proven total.
    """

    state_items = _inspect_instance_state_items(original)
    if state_items is None:
        return False
    try:
        if isinstance(original, defaultdict):
            # Member-descriptor channel: ``default_factory`` is C-level state
            # the ``__dict__``/slots enumeration cannot see.
            descriptor = cast(Any, defaultdict).__dict__["default_factory"]
            descriptor.__set__(rebuilt, descriptor.__get__(original, type(original)))
        for name, value in state_items.items():
            object.__setattr__(rebuilt, name, value)
    except Exception:
        return False
    return True


def allocate_mapping_like(cls: type[Any]) -> Any | None:
    """Allocate an EMPTY instance of a ``dict``-backed mapping class inertly.

    Uses the trusted base-type allocator (``OrderedDict.__new__`` for od-backed
    classes so the C linked list exists, else ``dict.__new__``) -- never the
    user's ``__new__``/``__init__``. Returns ``None`` when the class is not
    ``dict``-backed or carries static C bases the inert ladder cannot prove
    state-total (callers fall back without crashing).
    """

    if not issubclass(cls, dict) or not _inert_state_enumeration_total(cls, _TRUSTED_MAPPING_BASES):
        return None
    try:
        if issubclass(cls, OrderedDict):
            shell = OrderedDict.__new__(cast(Any, cls))
        else:
            shell = dict.__new__(cast(Any, cls))
    except Exception:
        return None
    return shell if type(shell) is cls else None


def mapping_like_set_item(shell: Any, key: Any, value: Any) -> None:
    """Write one item into an :func:`allocate_mapping_like` shell physically.

    ``OrderedDict.__setitem__`` maintains both the dict storage and the od
    linked list; every other dict-backed shell writes through
    ``dict.__setitem__``. Never dispatches a user override.
    """

    if isinstance(shell, OrderedDict):
        OrderedDict.__setitem__(shell, key, value)
    else:
        dict.__setitem__(shell, key, value)


def rebuild_mapping_like(original: Any, pairs: list[tuple[Any, Any]]) -> Any | None:
    """Inertly rebuild a ``dict``-backed mapping subclass carrying ``pairs``.

    The mapping sibling of :func:`rebuild_tuple_like` and the dataclass
    device-move arm's INERT rebuild: trusted base-type allocation, physical
    item writes, then verbatim instance-state copy -- the user's
    ``__new__``/``__init__`` never runs, so ctor side effects cannot enter the
    captured program and same-class instance state is never RESET
    (grind-p5 b3-opus-R12-2). Returns ``None`` when the rebuild cannot be
    proven faithful (callers keep the original and fail loudly downstream).
    """

    cls = type(original)
    shell = allocate_mapping_like(cls)
    if shell is None:
        return None
    try:
        for key, value in pairs:
            mapping_like_set_item(shell, key, value)
    except ReentrantTraceError:
        # Key ``__hash__``/``__eq__`` is user code; the R55 refusal
        # propagates (see rebuild_tuple_like).
        raise
    except Exception:
        return None
    if not _copy_instance_state_inertly(original, shell):
        return None
    return shell


def allocate_list_like(cls: type[Any]) -> Any | None:
    """Allocate an EMPTY instance of a ``list`` subclass inertly, or ``None``.

    Same contract as :func:`allocate_mapping_like`, over ``list.__new__``.
    """

    if not issubclass(cls, list) or not _inert_state_enumeration_total(cls, (list,)):
        return None
    try:
        shell = list.__new__(cls)
    except Exception:
        return None
    return shell if type(shell) is cls else None


def rebuild_list_like(original: Any, items: list[Any]) -> Any | None:
    """Inertly rebuild a ``list`` subclass carrying ``items``.

    The sequence sibling of :func:`rebuild_mapping_like`: ``list.__new__``
    allocation, ``list.extend`` population, verbatim instance-state copy;
    the user's ``__new__``/``__init__`` never runs (grind-p5 b3-opus-R12-2).
    Returns ``None`` when the rebuild cannot be proven faithful.
    """

    shell = allocate_list_like(type(original))
    if shell is None:
        return None
    try:
        list.extend(shell, items)
    except Exception:
        return None
    if not _copy_instance_state_inertly(original, shell):
        return None
    return shell


def copy_arg_tree(arg: Any, _in_progress: dict[int, Any] | None = None, _depth: int = 0) -> Any:
    """Copy an input argument tree, cloning tensors and recursing built-in containers.

    Why not ``copy.deepcopy``?  Many third-party tensor wrappers hold
    circular references back to their parent modules.  ``deepcopy`` follows
    these references recursively and either hangs or blows the stack.
    Instead we clone tensors (which copies storage without following Python
    object references) and recurse only into standard containers.

    Clones tensors, recurses into standard containers (list, tuple, dict),
    and leaves everything else as-is.  This prevents infinite loops that
    copy.deepcopy can trigger on complex tensor wrappers (e.g. ESCNN
    GeometricTensor) while still protecting user inputs from in-place
    mutations like device moves.

    Standard containers that refer back to themselves (directly or through a
    cycle of mutable containers) are handled without a ``RecursionError``: a
    mutable container being built is registered in ``_in_progress`` before its
    elements are recursed, so a self-reference resolves to the same in-progress
    copy and the cycle is reproduced in the copy.  The memo is CALL-scoped
    (retained across siblings, r-b4 R29-3): a DAG-shaped input that reuses one
    sub-container under several paths is copied ONCE and stays aliased in the
    copy -- which both matches the aliasing topology the model itself would see
    and makes the copy O(nodes).  The historical path-scoped memo copied a
    shared node once per PATH, i.e. exponentially in shared-substructure depth
    (measured x2 per level; depth 25 hung capture entry for ~4 minutes).
    Tensors remain leaves cloned per DISTINCT container occurrence and are
    never memoized.

    Note: custom objects containing tensors are passed by reference.  If the
    model is on a different device, _fetch_label_move_input_tensors may
    mutate the wrapper's tensor attribute in-place.  This is acceptable
    because pre-fix such inputs caused an infinite hang.

    Parameters
    ----------
    arg
        Argument value to copy.
    _in_progress
        Internal recursion state mapping ``id()`` of a mutable container being
        built to its (partially populated) copy, used to terminate reference
        cycles. Callers should not supply this.
    _depth
        Internal recursion depth used to enforce the shared input-boundary
        nesting ceiling (r-b4 R27-1). Callers should not supply this.

    Returns
    -------
    Any
        Argument copy with nested tensors cloned and custom objects preserved
        by reference.
    """
    if _in_progress is None:
        # Root entry: convert stack-budget exhaustion below the depth ceiling
        # into the typed refusal shared by every input-boundary walker (T11.4).
        try:
            return copy_arg_tree(arg, {}, _depth)
        except RecursionError as exc:
            from .._input_walk import raise_input_tree_stack_refusal

            raise_input_tree_stack_refusal(exc)
            raise  # unreachable: the refusal always raises (narrows the memo type)
    if isinstance(arg, torch.Tensor):
        # Tensors are leaves and are cloned per occurrence (never memoized), so a
        # structure that reuses the same tensor keeps its historical per-slot
        # clone semantics.
        return _clone_input_tensor_payload(arg)
    arg_id = id(arg)
    existing = _in_progress.get(arg_id)
    if existing is not None:
        # Cycle (a container reachable from itself) or DAG reuse (one container
        # under several paths): both resolve to the one memoized copy.
        return existing
    if isinstance(arg, (defaultdict, dict, list, tuple)) and _depth >= INPUT_TREE_MAX_DEPTH:
        # r-b4 R27-1: the canonical per-capture input copier is depth-bounded with the
        # SAME shared ceiling as every other input-boundary walker -- a deeper tree
        # refuses typed at capture entry instead of dying in a raw RecursionError.
        from .._input_walk import raise_input_tree_depth_refusal

        raise_input_tree_depth_refusal(depth=_depth)
    if isinstance(arg, dict):
        # INERT rebuild ladder (grind-p5 b3-opus-R12-2 sibling): the historical
        # ``type(arg)()`` re-ran the user's constructor (resetting same-class
        # instance state), read children through the overridable ``items()``
        # protocol (a lying override shrank the copy refusal-free), and the
        # ``defaultdict`` arm substituted the exact stock class for any
        # subclass. Population happens after registering so a cyclic value can
        # point back at this copy; ``default_factory`` is preserved through the
        # member-descriptor channel (#127).
        arg_type = type(arg)
        copied: Any
        if arg_type is dict:
            copied = {}
        else:
            copied = allocate_mapping_like(arg_type)
            if copied is None:
                # Unreconstructable subclass: pass by reference like other
                # custom wrappers rather than substituting a different program.
                return arg
        _in_progress[arg_id] = copied
        for key, value in dict.items(arg):
            mapping_like_set_item(copied, key, copy_arg_tree(value, _in_progress, _depth + 1))
        if arg_type is not dict and not _copy_instance_state_inertly(arg, copied):
            del _in_progress[arg_id]
            return arg
        return copied
    elif isinstance(arg, list):
        list_type = type(arg)
        if list_type is list:
            copied = []
        else:
            copied = allocate_list_like(list_type)
            if copied is None:
                return arg
        _in_progress[arg_id] = copied
        for index in range(list.__len__(arg)):
            list.append(
                copied, copy_arg_tree(list.__getitem__(arg, index), _in_progress, _depth + 1)
            )
        if list_type is not list and not _copy_instance_state_inertly(arg, copied):
            del _in_progress[arg_id]
            return arg
        return copied
    elif isinstance(arg, tuple):
        # Tuples are immutable and cannot self-reference directly; any cycle
        # through a tuple passes through a mutable container that is already
        # registered above, so recursing eagerly here is safe. Children read
        # through the concrete builtin slots (inert descent).
        items = [
            copy_arg_tree(tuple.__getitem__(arg, index), _in_progress, _depth + 1)
            for index in range(tuple.__len__(arg))
        ]
        # Memoized after construction (immutable, so no cycle can pass through
        # the tuple itself) so tuple-shaped DAGs are O(nodes). Subclass
        # reconstruction goes through the verified ladder (T11.7): _fields
        # presence used to be taken as proof of an *args constructor, so a
        # tuple subclass with a malformed _fields crashed the capture untyped.
        if type(arg) is tuple:
            copied = tuple(items)
        else:
            copied = rebuild_tuple_like(type(arg), items)
            if copied is None:
                # Unreconstructable subclass: pass by reference like other
                # custom wrappers (pre-fix this path crashed the capture).
                return arg
        _in_progress[arg_id] = copied
        return copied
    else:
        # Non-container, non-tensor objects (ints, strings, custom wrappers)
        # are returned by reference — shallow enough to avoid circular ref issues.
        return arg


def _safe_copy_arg(arg: Any) -> Any:
    """Compatibility alias for :func:`copy_arg_tree`.

    Parameters
    ----------
    arg
        Argument value to copy.

    Returns
    -------
    Any
        Recursive input-argument copy result.
    """

    return copy_arg_tree(arg)


def safe_copy_args(args: list[Any]) -> list[Any]:
    """Safely copy a list of positional arguments.

    Each element is copied via :func:`copy_arg_tree`: tensors are cloned,
    containers are recursed into, and everything else is passed by reference.
    """
    return [copy_arg_tree(arg) for arg in args]


def safe_copy_kwargs(kwargs: dict[Any, Any]) -> dict[Any, Any]:
    """Safely copy a dict of keyword arguments.

    Same semantics as :func:`safe_copy_args` but for ``**kwargs``.
    """
    return {key: copy_arg_tree(val) for key, val in kwargs.items()}


def _copy_input_tree_node(
    value: Any,
    *,
    path: str,
    depth: int,
    memo: dict[int, Any],
    in_progress: set[int],
    semantic_gaps: list[str],
    tensor_records: list[tuple[str, torch.Tensor, bool]],
    preserve_leaf_inputs: bool = False,
) -> Any:
    """Copy one input-tree node INERTLY, returning the copy (b3-opus-R12-1).

    The historical implementation prepared a memo and handed the tree to
    ``copy.deepcopy``, whose protocol runs USER code on container subclasses: a
    ``__deepcopy__``/``__reduce_ex__`` override could return a DIFFERENT tree
    (executed: forward captured over [5.0, 5.0] instead of [-3, -4]) and every
    witness honestly described the SUBSTITUTED tree with zero refusals. This
    walker copies the tree itself through the ``_input_walk`` inertness
    contract: children read through concrete builtin slots, containers rebuilt
    through the trusted base-type ladders (:func:`rebuild_tuple_like` identity
    verification, :func:`allocate_mapping_like`/:func:`allocate_list_like`
    allocation + verbatim state), and the ONLY third-party protocol invoked is
    torch's own exact-``Tensor`` deepcopy (which preserves cross-tensor storage
    topology through the shared ``memo``). Unknown wrappers keep the
    established pass-by-reference contract; a container that cannot be copied
    faithfully passes by reference WITH a semantic gap, so verification fails
    closed instead of describing a substituted program.

    Parameters
    ----------
    value:
        Input-tree node to copy.
    path:
        Human-readable location used for fail-closed diagnostics.
    depth:
        Current nesting depth, bounded by the shared input-boundary ceiling.
    memo:
        Shared identity memo: repeated objects copy ONCE and stay aliased.
    in_progress:
        Ancestor identities whose copies cannot be pre-registered (tuples);
        a cycle closing through one refuses typed.
    semantic_gaps:
        Accumulator for copies that cannot preserve the captured semantics.
    tensor_records:
        Tensor paths, originals, and whether storage-preserving deepcopy
        remains available, for cross-tensor alias checks (one per PATH
        occurrence, so repeated tensors keep every alias-pair site).
    preserve_leaf_inputs:
        Whether a grad-tracked exact-``Tensor`` leaf is copied as a new leaf
        (storage-preserving deepcopy keeps ``requires_grad``) instead of the
        graph-connected clone; see :func:`leaf_preserving_input_copies`.

    Returns
    -------
    Any
        The copied node (or the original, for by-reference kinds).
    """

    if depth >= INPUT_TREE_MAX_DEPTH:
        from .._input_walk import raise_input_tree_depth_refusal

        raise_input_tree_depth_refusal(depth=depth)
    if isinstance(value, torch.nn.Parameter):
        tensor_records.append((path, value, False))
        if id(value) in memo:
            return memo[id(value)]
        cloned = _clone_input_tensor_payload(value, preserve_leaf=preserve_leaf_inputs)
        memo[id(value)] = cloned
        return cloned
    if isinstance(value, torch.Tensor):
        deepcopy_preserves_contract = (
            value.is_leaf
            and (preserve_leaf_inputs or not value.requires_grad)
            and type(value) is torch.Tensor
        )
        tensor_records.append((path, value, deepcopy_preserves_contract))
        if id(value) in memo:
            return memo[id(value)]
        if deepcopy_preserves_contract:
            # torch's own deepcopy protocol on an EXACT Tensor (trusted, not
            # user-overridable): copies each underlying storage once through
            # the shared memo, so views keep size/stride/offset and
            # cross-tensor storage sharing.
            return copy.deepcopy(value, memo)
        cloned = _clone_input_tensor_payload(value)
        memo[id(value)] = cloned
        try:
            physical_metadata_changed = (
                tuple(cloned.shape) != tuple(value.shape)
                or tuple(cloned.stride()) != tuple(value.stride())
                or cloned.storage_offset() != value.storage_offset()
            )
        except (RuntimeError, TypeError, NotImplementedError):
            physical_metadata_changed = True
        if physical_metadata_changed:
            semantic_gaps.append(
                f"{path}: grad-preserving tensor clone changed physical view metadata"
            )
        return cloned
    value_id = id(value)
    if value_id in memo:
        return memo[value_id]
    if value_id in in_progress:
        from .._input_walk import raise_input_tree_cycle_refusal

        # Only the tuple arm registers in ``in_progress`` (dict/list cycles
        # resolve through their memo shells), so the re-encountered node is
        # always a tuple: the historical "else mapping" guess was dead code
        # reporting a false closed-vocabulary kind (B3R4-R12-2 sibling).
        raise_input_tree_cycle_refusal(kind="sequence")

    def _child(child: Any, child_path: str) -> Any:
        """Recurse into one child with the shared walk state."""

        return _copy_input_tree_node(
            child,
            path=child_path,
            depth=depth + 1,
            memo=memo,
            in_progress=in_progress,
            semantic_gaps=semantic_gaps,
            tensor_records=tensor_records,
            preserve_leaf_inputs=preserve_leaf_inputs,
        )

    def _reference_with_gap(reason: str) -> Any:
        """Disclose an uncopyable container and pass it by reference."""

        semantic_gaps.append(f"{path}: {reason}")
        memo[value_id] = value
        return value

    if isinstance(value, dict):
        cls = type(value)
        state_items: dict[str, Any] | None = None
        if cls is dict:
            shell: Any = {}
        else:
            state_items = _inspect_instance_state_items(value)
            shell = None if state_items is None else allocate_mapping_like(cls)
            if shell is None:
                return _reference_with_gap(
                    f"mapping subclass {cls.__name__} cannot be copied inertly; passed by reference"
                )
        memo[value_id] = shell
        for index, (key, child) in enumerate(dict.items(value)):
            mapping_like_set_item(
                shell,
                _child(key, f"{path}.<key:{index}>"),
                _child(child, f"{path}.<value:{index}>"),
            )
        if state_items:
            copied_state = {
                name: _child(state_items[name], f"{path}.<state:{name}>") for name in state_items
            }
        else:
            copied_state = {}
        if cls is not dict:
            try:
                if isinstance(value, defaultdict):
                    descriptor = cast(Any, defaultdict).__dict__["default_factory"]
                    descriptor.__set__(shell, descriptor.__get__(value, cls))
                for name, copied_value in copied_state.items():
                    object.__setattr__(shell, name, copied_value)
            except Exception:
                del memo[value_id]
                return _reference_with_gap(
                    f"mapping subclass {cls.__name__} instance state cannot be "
                    "copied inertly; passed by reference"
                )
        return shell
    if isinstance(value, list):
        list_cls = type(value)
        state_items = None
        if list_cls is list:
            shell = []
        else:
            state_items = _inspect_instance_state_items(value)
            shell = None if state_items is None else allocate_list_like(list_cls)
            if shell is None:
                return _reference_with_gap(
                    f"sequence subclass {list_cls.__name__} cannot be copied inertly; "
                    "passed by reference"
                )
        memo[value_id] = shell
        for index in range(list.__len__(value)):
            list.append(shell, _child(list.__getitem__(value, index), f"{path}.{index}"))
        if list_cls is not list and state_items is not None:
            copied_state = {
                name: _child(state_items[name], f"{path}.<state:{name}>") for name in state_items
            }
            try:
                for name, copied_value in copied_state.items():
                    object.__setattr__(shell, name, copied_value)
            except Exception:
                del memo[value_id]
                return _reference_with_gap(
                    f"sequence subclass {list_cls.__name__} instance state cannot be "
                    "copied inertly; passed by reference"
                )
        return shell
    if isinstance(value, tuple):
        # Immutable: children first (a cycle strictly through tuples cannot be
        # constructed; one through a mutable ancestor resolves via its memo
        # shell, and an unresolvable close refuses typed via ``in_progress``).
        in_progress.add(value_id)
        try:
            items = [
                _child(tuple.__getitem__(value, index), f"{path}.{index}")
                for index in range(tuple.__len__(value))
            ]
        finally:
            in_progress.discard(value_id)
        if type(value) is tuple:
            copied: Any = tuple(items)
        else:
            copied = rebuild_tuple_like(type(value), items)
            if copied is None:
                return _reference_with_gap(
                    f"tuple subclass {type(value).__name__} cannot be rebuilt "
                    "faithfully; passed by reference"
                )
        memo[value_id] = copied
        return copied
    # Preserve the established contract for custom wrappers: pass them by
    # reference rather than following arbitrary attributes.
    memo[value_id] = value
    return value


def _record_unpreserved_tensor_aliases(
    tensor_records: list[tuple[str, torch.Tensor, bool]],
    semantic_gaps: list[str],
    *,
    require_distinct_tensor_sites: bool,
) -> None:
    """Record aliases involving tensors that require grad-preserving clones.

    Parameters
    ----------
    tensor_records:
        Tensor input paths, original tensors, and storage-deepcopy eligibility.
    semantic_gaps:
        Accumulator receiving fail-closed alias-topology diagnostics.
    require_distinct_tensor_sites:
        Whether the downstream runnable descriptor requires every model-input
        site to have distinct identity and storage.

    Returns
    -------
    None
        Appends one diagnostic for every potentially overlapping pair whose
        topology cannot be preserved without severing autograd linkage.

    Notes
    -----
    r-b4 R29-2: the historical implementation ran ``touched_bytes_relation`` on
    every pair -- O(T^2) in tensor-leaf count (measured 0.72 s at 4k leaves even
    with every storage disjoint). Candidate pairs are now pre-filtered by a
    device-scoped byte-interval sweep (:func:`_alias_candidate_pairs`) that drops
    ONLY pairs the relation ladder provably answers ``disjoint`` from bounding
    intervals alone; every surviving pair still runs the exact same per-pair
    ladder, so verdicts and diagnostic text are byte-identical.
    """

    if len(tensor_records) < 2:
        return
    footprints: list[TensorByteFootprint | None] = []
    for _left_path, tensor, _transparent in tensor_records:
        try:
            footprints.append(tensor_byte_footprint(tensor))
        except (RuntimeError, TypeError, NotImplementedError):
            footprints.append(None)
    for left_index, right_index in sorted(_alias_candidate_pairs(tensor_records, footprints)):
        left_path, left, left_transparent = tensor_records[left_index]
        right_path, right, right_transparent = tensor_records[right_index]
        if left is right:
            if require_distinct_tensor_sites:
                semantic_gaps.append(
                    f"{left_path} <-> {right_path}: runnable input sites share "
                    "one tensor identity, which the sparse descriptor cannot encode"
                )
            continue
        if left_transparent and right_transparent and not require_distinct_tensor_sites:
            continue
        try:
            relation = touched_bytes_relation(left, right)
        except (RuntimeError, TypeError, NotImplementedError):
            relation = "unknown"
        if relation == "disjoint":
            continue
        if require_distinct_tensor_sites:
            semantic_gaps.append(
                f"{left_path} <-> {right_path}: runnable input sites have "
                f"{relation} storage, which the sparse descriptor cannot encode"
            )
        else:
            semantic_gaps.append(
                f"{left_path} <-> {right_path}: grad-preserving clones cannot prove "
                f"the original tensor alias topology ({relation})"
            )


def _alias_candidate_pairs(
    tensor_records: list[tuple[str, torch.Tensor, bool]],
    footprints: list["TensorByteFootprint | None"],
) -> set[tuple[int, int]]:
    """Return the record-index pairs the alias scan cannot silently skip (r-b4 R29-2).

    A pair is EXCLUDED only when ``touched_bytes_relation`` provably answers
    ``disjoint`` without a per-pair proof: both footprints known, distinct
    objects, and either (a) an empty view on one side, (b) distinct device
    types, (c) same device type with two CONCRETE, different indexes, or
    (d) the same device key with non-overlapping absolute byte intervals.
    Everything else -- identity pairs, unprovable footprints, same-type
    None-vs-concrete device indexes, interval overlaps -- stays a candidate and
    runs the unchanged exact ladder.

    Parameters
    ----------
    tensor_records:
        Tensor input paths, original tensors, and storage-deepcopy eligibility.
    footprints:
        Pre-computed ``tensor_byte_footprint`` per record (``None`` = unprovable).

    Returns
    -------
    set[tuple[int, int]]
        Candidate ``(left_index, right_index)`` pairs with ``left < right``.
    """

    total = len(tensor_records)
    candidates: set[tuple[int, int]] = set()

    # Identity pairs (one object at several sites) always reach the loop body:
    # the caller's identity branch decides them before any footprint logic.
    by_identity: dict[int, list[int]] = {}
    for index, (_path, tensor, _transparent) in enumerate(tensor_records):
        by_identity.setdefault(id(tensor), []).append(index)
    for indices in by_identity.values():
        for position, left_index in enumerate(indices):
            for right_index in indices[position + 1 :]:
                candidates.add((left_index, right_index))

    # An unprovable footprint relates ``unknown`` to every partner (checked
    # before the empty-view rule in the ladder, so zero-numel partners count).
    for index, footprint in enumerate(footprints):
        if footprint is not None:
            continue
        for other in range(total):
            if other != index:
                candidates.add((min(index, other), max(index, other)))

    # Provable footprints with a nonzero span, grouped by exact device key as
    # (start_byte, end_byte, record_index) interval entries.
    by_device_key: dict[tuple[str, int | None], list[tuple[int, int, int]]] = {}
    for index, footprint in enumerate(footprints):
        if footprint is None or footprint.numel == 0:
            continue
        by_device_key.setdefault(footprint.device_key, []).append(
            (footprint.start_byte, footprint.end_byte, index)
        )

    # Same device TYPE under two different keys is provably disjoint only when
    # both indexes are concrete; a None-vs-concrete index answers ``unknown``.
    keys_by_type: dict[str, list[tuple[str, int | None]]] = {}
    for device_key in by_device_key:
        keys_by_type.setdefault(device_key[0], []).append(device_key)
    for device_keys in keys_by_type.values():
        for key_position, left_key in enumerate(device_keys):
            for right_key in device_keys[key_position + 1 :]:
                if left_key[1] is not None and right_key[1] is not None:
                    continue
                for _, _, left_index in by_device_key[left_key]:
                    for _, _, right_index in by_device_key[right_key]:
                        candidates.add((min(left_index, right_index), max(left_index, right_index)))

    # Interval sweep inside one exact device key: only pairs whose absolute
    # byte spans overlap survive (disjoint spans are the ladder's own verdict).
    for entries in by_device_key.values():
        if len(entries) < 2:
            continue
        entries.sort()
        active: list[tuple[int, int]] = []  # (end_byte, record_index)
        for start_byte, end_byte, index in entries:
            active = [entry for entry in active if entry[0] > start_byte]
            for _, other in active:
                candidates.add((min(index, other), max(index, other)))
            active.append((end_byte, index))

    return candidates


def safe_copy_input_tree(
    args: list[Any],
    kwargs: dict[Any, Any],
    *,
    require_distinct_tensor_sites: bool = False,
    preserve_leaf_inputs: bool | None = None,
) -> tuple[list[Any], dict[Any, Any], tuple[str, ...]]:
    """Copy one complete model-input graph while preserving tensor topology.

    Positional and keyword inputs share one identity memo. For ordinary leaf
    tensors without autograd history, PyTorch's deepcopy protocol copies each
    underlying storage once and rebuilds every view with its original size,
    stride, and storage offset. Grad-tracked tensors use the historical clone
    path so gradients still reach the caller's input; with ``preserve_leaf_inputs``
    (validation's private copies) a grad-tracked LEAF is deepcopied as a new leaf
    instead, so a model may still set ``requires_grad`` on it. Every path shares one memo,
    so the same tensor repeated at multiple call sites remains one object.

    The tree walk itself is INERT (b3-opus-R12-1): containers are copied by
    :func:`_copy_input_tree_node` through the ``_input_walk`` contract, never
    by handing the tree to ``copy.deepcopy`` -- whose protocol let a user
    ``__deepcopy__``/``__reduce_ex__`` SUBSTITUTE the tree the capture then
    honestly witnessed. Torch's exact-``Tensor`` deepcopy is the one trusted
    protocol still invoked, for storage-topology preservation.

    Parameters
    ----------
    args:
        Normalized positional model inputs.
    kwargs:
        Keyword model inputs.
    require_distinct_tensor_sites:
        Whether aliases between distinct input sites must fail closed because a
        downstream sparse runnable descriptor cannot encode them.
    preserve_leaf_inputs:
        Whether grad-tracked leaf inputs are copied as new leaves with the same
        ``requires_grad`` (so gradients do NOT reach the caller's input).
        ``None`` follows :func:`leaf_preserving_input_copies`.

    Returns
    -------
    tuple[list[Any], dict[Any, Any], tuple[str, ...]]
        Copied positional inputs, copied keyword inputs, and semantic gaps that
        require capture verification to fail closed. Distinct overlapping
        grad-tracked views, containers that cannot be copied inertly, and
        unexpected copy failures use the historical clone/by-reference
        fallbacks but are explicitly reported as unverifiable.
    """

    from .._errors import InvalidArgumentError
    from .._state import pause_logging

    memo: dict[int, Any] = {}
    in_progress: set[int] = set()
    semantic_gaps: list[str] = []
    tensor_records: list[tuple[str, torch.Tensor, bool]] = []
    if preserve_leaf_inputs is None:
        preserve_leaf_inputs = _PRESERVE_LEAF_INPUT_COPIES.get()
    try:
        with pause_logging():
            copied_args = _copy_input_tree_node(
                args,
                path="input.args",
                depth=0,
                memo=memo,
                in_progress=in_progress,
                semantic_gaps=semantic_gaps,
                tensor_records=tensor_records,
                preserve_leaf_inputs=preserve_leaf_inputs,
            )
            copied_kwargs = _copy_input_tree_node(
                kwargs,
                path="input.kwargs",
                depth=0,
                memo=memo,
                in_progress=in_progress,
                semantic_gaps=semantic_gaps,
                tensor_records=tensor_records,
                preserve_leaf_inputs=preserve_leaf_inputs,
            )
    except RecursionError as exc:
        from .._input_walk import raise_input_tree_stack_refusal

        raise_input_tree_stack_refusal(exc)
        raise  # unreachable: the refusal always raises
    except InvalidArgumentError:
        # Typed depth/cycle refusals from the shared input-boundary contract
        # propagate; a clone fallback would just re-walk the same tree.
        raise
    except ReentrantTraceError:
        # R55: since the r8 R54 admission reorder the copy runs INSIDE the
        # reserved window, so a tensor-subclass ``__torch_function__`` (or any
        # input protocol hook) attempting a nested public capture refuses
        # HERE. That refusal is a typed program error, not a copy failure --
        # degrading it to a semantic gap would settle the outer capture
        # COMPLETE while silencing the documented ReentrantTraceError.
        raise
    except Exception as exc:
        semantic_gaps.append(
            f"input: topology-preserving copy failed with {type(exc).__name__}: {exc}"
        )
        copied_args = safe_copy_args(args)
        copied_kwargs = safe_copy_kwargs(kwargs)
    _record_unpreserved_tensor_aliases(
        tensor_records,
        semantic_gaps,
        require_distinct_tensor_sites=require_distinct_tensor_sites,
    )
    return copied_args, copied_kwargs, tuple(semantic_gaps)


def _model_expects_single_arg(model: nn.Module) -> bool | None:
    """Check if the model's forward expects exactly 1 positional arg (excluding self).

    Used by :func:`normalize_input_args` to disambiguate whether a user-supplied
    tuple is "multiple positional args" or "one arg that is a tuple."

    Returns:
        True if exactly 1 positional param, False if more or uses ``*args``,
        None if introspection fails (e.g. C-extension forward).
    """
    try:
        spec = inspect.getfullargspec(model.forward)
    except (TypeError, ValueError):
        # Introspection can fail on C-extension or dynamically-generated forward custom_methods.
        return None
    named_args = [a for a in spec.args if a != "self"]
    if spec.varargs is not None:
        # Has *args — could accept any number of positional args; can't tell.
        return False
    return len(named_args) == 1


def normalize_input_args(input_args: Any, model: nn.Module) -> list[Any]:
    """Normalize ``input_args`` into a list suitable for ``model(*input_args)``.

    Handles the ambiguity when the user ops a tuple or list: it could be
    multiple positional args, or a single arg that happens to be a
    tuple/list (issue #43).  Resolves the ambiguity by inspecting the
    model's ``forward`` signature via :func:`_model_expects_single_arg`:

    * If the model expects exactly one positional param and the user passed
      a multi-element tuple/list, wrap it in a list so it arrives as a
      single argument: ``model(the_tuple)``.
    * Otherwise, treat each element of the tuple/list as a separate
      positional argument: ``model(arg0, arg1, ...)``.
    """
    if type(input_args) in (tuple, list):
        single = _model_expects_single_arg(model)
        if single and len(input_args) != 1:
            # Model expects 1 arg but user passed a multi-element tuple/list.
            # The tuple/list itself IS the single argument.
            input_args = [input_args]
        elif type(input_args) is tuple:
            # Multiple args — convert tuple to mutable list for internal use.
            input_args = list(input_args)
        # If already a list and not wrapping, leave as-is.
    elif input_args is not None:
        # Bare value (single tensor, etc.) — wrap in a list.
        input_args = [input_args]
    if not input_args:
        input_args = []
    return cast(list[Any], input_args)
