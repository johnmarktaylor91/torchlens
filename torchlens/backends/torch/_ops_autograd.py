"""Autograd saved-tensor statistics and container snapshots."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any

import torch

from ... import _state as _st
from ..._ir.container import (
    ContainerSpec,
    OutputPathComponent,
)
from ..._ir.container_registry import (
    ContainerLeafOccurrence,
    FuncSite,
    Phase,
    Role,
    walk_container,
)
from ..._state import pause_logging
from ..._utils._tensor_label import (
    get_tensor_label,
)
from ..._utils._torch_compat import (
    get_functorch_wrapped_tensor_checker,
    saved_tensors_default_hooks_active,
)
from ..._utils._torch_symbols import tensor_read, tensor_reads
from ..._utils.collections import ensure_iterable
from ..._utils.tensor_utils import (
    is_functorch_wrapped_tensor,
)
from ._completeness_origins import internal_scalar_read
from ._user_grad_fn import raw_grad_fn

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace
    from .ops import _OutputTensorEntry  # annotations only; runtime binding at the bottom


__all__ = (
    "_add_autograd_saved_tensor",
    "_get_autograd_saved_stats_by_output",
    "_partition_output_entries_with_autograd_stats",
    "_output_entries_with_autograd_stats",
    "_register_call_output_container_snapshot",
    "register_call_input_container_snapshots",
    "_container_leaf_occurrences_from_entries",
    "_get_autograd_saved_stats_for_tensor",
)


def _add_autograd_saved_tensor(
    tensor: torch.Tensor,
    seen_data_ptrs: set[int],
) -> tuple[int, int]:
    """Return byte/count contribution for a deduped autograd-saved tensor.

    Parameters
    ----------
    tensor
        Saved tensor to measure.
    seen_data_ptrs
        Data pointers already counted for this operation.

    Returns
    -------
    tuple of int
        ``(bytes, tensor_count)`` contribution for this tensor.
    """
    if is_functorch_wrapped_tensor(tensor):
        return 0, 0
    try:
        # TorchLens-internal autograd-saved-tensor dedup read: mark it as an internal read so
        # the storage-bridge host-write watch (which patches ``Tensor.data_ptr``) does not
        # mistake this bookkeeping pointer read for a user storage-alias exposure. Also
        # pause logging -- ``data_ptr`` is a wrapped tensor method and an unpaused read pays the
        # full per-op dispatch while emitting no op. The bypassed wrapped call still consumes
        # its ``next_func_call_id()`` so the id sequence stamped on real ops is unchanged.
        logging_enabled = _st._logging_enabled
        with pause_logging(), internal_scalar_read():
            if logging_enabled:
                _st.next_func_call_id()
            data_ptr = tensor_read(tensor, "data_ptr")(tensor)
    except Exception:
        return 0, 0
    if data_ptr in seen_data_ptrs:
        return 0, 0
    seen_data_ptrs.add(data_ptr)
    with pause_logging():
        numel, element_size = tensor_reads(tensor, "numel", "element_size")
        return numel(tensor) * element_size(tensor), 1


def _get_autograd_saved_stats_by_output(
    output: Any,
) -> dict[int, tuple[int | None, int | None]]:
    """Measure autograd-saved tensor bytes/counts for each tensor output.

    Parameters
    ----------
    output
        Raw function output from a decorated torch operation.

    Returns
    -------
    dict
        Mapping from output index to ``(autograd_memory,
        num_autograd_tensors)``. Non-tensor outputs and tensors without
        ``grad_fn_handle`` are omitted and handled by callers as ``None`` values.
    """
    stats_by_index: dict[int, tuple[int | None, int | None]] = {}
    seen_grad_fns: set[int] = set()
    seen_data_ptrs: set[int] = set()

    for output_index, maybe_tensor in enumerate(ensure_iterable(output)):
        # TorchLens bookkeeping ``grad_fn`` read (same receiver aliasing
        # note as _partition_output_entries_with_autograd_stats).
        with internal_scalar_read():
            grad_fn_handle = (
                maybe_tensor.grad_fn if isinstance(maybe_tensor, torch.Tensor) else None
            )
        if grad_fn_handle is None:
            continue
        grad_fn_object_id = id(grad_fn_handle)
        if grad_fn_object_id in seen_grad_fns:
            stats_by_index[output_index] = (0, 0)
            continue
        seen_grad_fns.add(grad_fn_object_id)

        total_bytes = 0
        tensor_count = 0
        for saved_value in _iter_autograd_saved_candidates(grad_fn_handle):
            for saved_tensor in _collect_tensor_values(saved_value):
                bytes_added, count_added = _add_autograd_saved_tensor(saved_tensor, seen_data_ptrs)
                total_bytes += bytes_added
                tensor_count += count_added
        stats_by_index[output_index] = (total_bytes, tensor_count)

    return stats_by_index


def _classify_saved_storage(trace: Any, tensor: torch.Tensor) -> tuple[str, int | None, int]:
    """Classify one autograd-saved tensor by its STORAGE identity.

    Keyed on ``untyped_storage().data_ptr()`` on BOTH sides (the parameter and
    buffer maps are already storage-keyed): keying on ``tensor.data_ptr()``
    would misclassify an offset VIEW of a weight -- routine in attention -- as
    an activation.

    Parameters
    ----------
    trace
        Active trace carrying the session storage-address maps.
    tensor
        Saved tensor accepted by the gross-band dedup.

    Returns
    -------
    tuple[str, int | None, int]
        ``(storage_class, storage_ptr, storage_nbytes)`` where the class is
        ``"saved_parameter"`` / ``"saved_buffer"`` / ``"saved_activation"``;
        an unreadable storage classifies as an activation with ``None`` ptr.
    """

    try:
        with pause_logging():
            storage = tensor_read(tensor, "untyped_storage")(tensor)
            storage_ptr = storage.data_ptr()
            storage_nbytes = int(storage.nbytes())
    except (RuntimeError, TypeError, NotImplementedError):
        return "saved_activation", None, 0
    param_map = trace.__dict__.get("_param_storage_addresses") or {}
    if storage_ptr in param_map:
        return "saved_parameter", storage_ptr, storage_nbytes
    buffer_map = trace.__dict__.get("_buffer_storage_addresses") or {}
    if storage_ptr in buffer_map:
        return "saved_buffer", storage_ptr, storage_nbytes
    return "saved_activation", storage_ptr, storage_nbytes


_TENSOR = torch.Tensor
_PARAMETER = torch.nn.Parameter
_HEAPTYPE_FLAG = 1 << 9
"""``Py_TPFLAGS_HEAPTYPE``: a class without it is defined in C, immortal and immutable."""
_NO_TENSOR_CLASSES = frozenset({type(None), bool, int, float, complex, str})
"""Exact classes whose saved value holds no tensor (``_collect_tensor_values`` gives ``[]``)."""

_SavedReader = tuple[bool, Any, tuple[str, ...]]
_SAVED_READER_BY_CLASS: dict[type, _SavedReader] = {}
"""Saved-attribute reader per C-defined gradient-function class.

Only classes without ``Py_TPFLAGS_HEAPTYPE`` are cached: they live as long as the
interpreter and their ``__dict__`` cannot change, so the cache holds no gradient
function, tensor or trace and never goes stale. A Python-defined class (a custom
``autograd.Function``'s backward node) is scanned on every op, as before.

The cached names are only those that can hold a tensor. Each ``_saved_*`` attribute of a
C class is a getter over one C++ field of fixed type, so a value read once as a number, a
string or a non-empty tuple of numbers (sizes, strides, scalars, flags) shows a field that
never holds a tensor; such names are dropped at the class's first read. A ``None``, an
empty tuple, a tensor or a read that raises keeps the name.
"""
_NUMBER_CLASSES = frozenset({bool, int, float, complex})
_FIXED_NON_TENSOR_CLASSES = _NUMBER_CLASSES | {str}


def _saved_reader(cls: type) -> _SavedReader:
    """Return how to read the saved values of a gradient function of class ``cls``.

    Parameters
    ----------
    cls
        The gradient function's class.

    Returns
    -------
    tuple
        ``(reads_saved_tensors, getter, names)``: whether a ``saved_tensors`` read can
        produce anything, an ``attrgetter`` over the ``_saved_*`` names of the class's own
        ``__dict__`` (``None`` when there are none), and those names in dict order.
    """

    names = tuple(name for name in cls.__dict__ if name.startswith(_AUTOGRAD_SAVED_ATTR_PREFIX))
    # A class with no ``saved_tensors`` anywhere, no instance dict and the generic
    # attribute lookup answers every instance read with the ``()`` default.
    reads_saved_tensors = (
        hasattr(cls, "saved_tensors")
        or cls.__dictoffset__ != 0
        or getattr(cls, "__getattribute__") is not object.__getattribute__
    )
    getter = operator.attrgetter(*names) if names else None
    return reads_saved_tensors, getter, names


def _read_saved(grad_fn: Any, reader: _SavedReader) -> Any:
    """Read ``grad_fn``'s saved values through ``reader``, each read best-effort.

    Same values and order as ``_iter_autograd_saved_candidates``'s reads: ``saved_tensors``
    first, then each ``_saved_*`` attribute; an attribute that raises is skipped.
    """

    reads_saved_tensors, getter, names = reader
    values: list[Any] = []
    if reads_saved_tensors:
        try:
            values.extend(getattr(grad_fn, "saved_tensors", ()))
        except Exception:
            pass
    if getter is None:
        return values
    try:
        got = getter(grad_fn)
    except Exception:
        # One attribute refused: read them one by one (reads are idempotent).
        for name in names:
            try:
                values.append(getattr(grad_fn, name))
            except Exception:
                continue
        return values
    if len(names) == 1:
        values.append(got)
    elif values:
        values.extend(got)
    else:
        return got
    return values


def _saved_values(grad_fn: Any) -> Any:
    """Return ``grad_fn``'s readable saved values (hook-packed values are skipped).

    The cheaper form of ``_iter_autograd_saved_candidates``: the ``_saved_*`` names come
    from :data:`_SAVED_READER_BY_CLASS` and are read in one ``attrgetter`` call. The hook
    rule is the same: with default saved-tensors hooks installed nothing is read, and when
    the runtime cannot tell, the reads run under ``pause_logging``.
    """

    cls = grad_fn.__class__
    reader = _SAVED_READER_BY_CLASS.get(cls)
    hooks_active = saved_tensors_default_hooks_active()
    if hooks_active:
        return ()
    if reader is None:
        reader = _saved_reader(cls)
        if not cls.__flags__ & _HEAPTYPE_FLAG:
            if hooks_active is None:
                with pause_logging():
                    return _learn_reader(cls, grad_fn, reader)
            return _learn_reader(cls, grad_fn, reader)
    if hooks_active is None:
        with pause_logging():
            return _read_saved(grad_fn, reader)
    return _read_saved(grad_fn, reader)


def _learn_reader(cls: type, grad_fn: Any, reader: _SavedReader) -> list[Any]:
    """Read ``grad_fn``'s saved values name by name and cache the reader of its tensor names.

    Same values and order as :func:`_read_saved` for the tensors among them; the names whose
    value shows a field that never holds a tensor are left out of the cached reader.
    """

    reads_saved_tensors, _getter, names = reader
    values: list[Any] = []
    if reads_saved_tensors:
        try:
            values.extend(getattr(grad_fn, "saved_tensors", ()))
        except Exception:
            pass
    kept: list[str] = []
    for name in names:
        try:
            value = getattr(grad_fn, name)
        except Exception:
            kept.append(name)
            continue
        if not _never_holds_tensor(value):
            kept.append(name)
            values.append(value)
    getter = operator.attrgetter(*kept) if kept else None
    _SAVED_READER_BY_CLASS[cls] = (reads_saved_tensors, getter, tuple(kept))
    return values


def _never_holds_tensor(value: Any) -> bool:
    """Return whether ``value`` shows a C++ saved field of a type that holds no tensor."""

    cls = value.__class__
    if cls in _FIXED_NON_TENSOR_CLASSES:
        return True
    if cls is not tuple and cls is not list:
        return False
    return bool(value) and all(item.__class__ in _NUMBER_CLASSES for item in value)


def _flatten_saved_tensors(values: Any, trace: Any = None) -> list[torch.Tensor]:
    """Return the tensors among saved ``values``, in order, as ``_collect_tensor_values`` would.

    Hot path: a plain tensor or a scalar is decided by its exact class, with no call.
    ``trace`` keys the functorch checker's resolution (once per trace; ``None`` resolves it).
    """

    # Cognitive complexity 17 on purpose: per-op hot path of the autograd section; the
    # exact-class checks stay inline because a helper per value would cost a call each.
    tensors: list[torch.Tensor] = []
    for value in values:
        cls = value.__class__
        if cls is _TENSOR or cls is _PARAMETER:
            tensors += (value,)
            continue
        if cls in _NO_TENSOR_CLASSES:
            continue
        group = value if cls is tuple or cls is list else _collect_tensor_values(value)
        for item in group:
            cls = item.__class__
            if (
                cls is _TENSOR
                or cls is _PARAMETER
                or (cls not in _NO_TENSOR_CLASSES and isinstance(item, _TENSOR))
            ):
                tensors += (item,)
    return _drop_functorch_wrapped(tensors, trace) if tensors else tensors


_FUNCTORCH_RESOLVED: tuple[int | None, Any] = (None, None)
"""``(id of the trace that resolved last, checker)``; swapped as one tuple."""


def _functorch_checker(trace: Any) -> Any:
    """Return torch's functorch wrapped-tensor checker, resolved once per trace.

    The same resolution as ``get_functorch_wrapped_tensor_checker`` (which runs the
    capability probe on every call), reused for the rest of the trace that resolved it,
    as ``_functorch_check.functorch_wrapped_check`` does per capture run.
    """

    global _FUNCTORCH_RESOLVED
    key, checker = _FUNCTORCH_RESOLVED
    if trace is not None and key == id(trace):
        return checker
    checker = get_functorch_wrapped_tensor_checker()
    _FUNCTORCH_RESOLVED = (None if trace is None else id(trace), checker)
    return checker


def _drop_functorch_wrapped(tensors: list[torch.Tensor], trace: Any = None) -> list[torch.Tensor]:
    """Return ``tensors`` without functorch wrapper tensors (``is_functorch_wrapped_tensor``).

    The checker is fetched once per trace; a check that raises ``RuntimeError`` keeps the
    tensor.
    """

    checker = _functorch_checker(trace)
    if checker is None:
        return tensors
    kept: list[torch.Tensor] = []
    for tensor in tensors:
        try:
            if checker(tensor):
                continue
        except RuntimeError:
            pass
        kept += (tensor,)
    return kept


def _saved_tensor_stats(
    trace: Any, grad_fn: Any, seen_data_ptrs: set[int], seen_saved_storages: set[int]
) -> tuple[tuple[int | None, int | None], dict[str, int]]:
    """Return one gradient function's saved-tensor ``(bytes, count)`` and saved band.

    Per saved tensor, in order, the values of ``_add_autograd_saved_tensor`` and
    ``_classify_saved_storage``, read directly under one ``pause_logging`` and
    ``internal_scalar_read`` entry per op. Each ``data_ptr`` read still consumes its
    ``next_func_call_id()`` while logging is on, so real ops keep their ids.
    """

    # Cognitive complexity 21 on purpose: per-op hot path of the autograd section; one
    # context entry per op means the per-tensor reads stay inline in this loop.
    band = dict.fromkeys(_BAND_KEYS, 0)
    tensors = _flatten_saved_tensors(_saved_values(grad_fn), trace)
    total_bytes = tensor_count = 0
    if not tensors:
        return (total_bytes, tensor_count), band
    param_map = trace.__dict__.get("_param_storage_addresses") or {}
    buffer_map = trace.__dict__.get("_buffer_storage_addresses") or {}
    logging_enabled = _st._logging_enabled
    reads_cls: Any = None
    reads: tuple[Any, ...] = ()
    with pause_logging(), internal_scalar_read():
        for tensor in tensors:
            try:
                if logging_enabled:
                    _st.next_func_call_id()
                if tensor.__class__ is not reads_cls:
                    reads = tensor_reads(tensor, *_SAVED_TENSOR_READS)
                    reads_cls = tensor.__class__
                data_ptr = reads[0](tensor)
            except Exception:
                continue
            if data_ptr in seen_data_ptrs:
                continue
            seen_data_ptrs.add(data_ptr)
            bytes_added = reads[1](tensor) * reads[2](tensor)
            total_bytes += bytes_added
            tensor_count += 1
            try:
                storage = reads[3](tensor)
                storage_ptr = storage.data_ptr()
                storage_nbytes = int(storage.nbytes())
            except (RuntimeError, TypeError, NotImplementedError):
                band["saved_activation"] += bytes_added
                continue
            storage_class = (
                "saved_parameter"
                if storage_ptr in param_map
                else "saved_buffer"
                if storage_ptr in buffer_map
                else "saved_activation"
            )
            band[storage_class] += bytes_added
            if storage_ptr not in seen_saved_storages:
                seen_saved_storages.add(storage_ptr)
                band["newly_saved_bytes"] += storage_nbytes
                band[_NEWLY_KEY[storage_class]] += storage_nbytes
    return (total_bytes, tensor_count), band


_BAND_KEYS = (
    "saved_parameter",
    "saved_buffer",
    "saved_activation",
    "newly_saved_bytes",
    "newly_saved_parameter",
    "newly_saved_buffer",
    "newly_saved_activation",
)
_NEWLY_KEY = {
    name: f"newly_{name}" for name in ("saved_parameter", "saved_buffer", "saved_activation")
}
_SAVED_TENSOR_READS = ("data_ptr", "numel", "element_size", "untyped_storage")


def _partition_output_entries_with_autograd_stats(
    trace: Any, output: Any
) -> list[_OutputTensorEntry]:
    """Collect output entries, autograd stats, and the saved-band decomposition.

    The gross per-op band (``autograd_memory``) keeps its exact historical
    dedup (by ``tensor.data_ptr()`` within one call; views each count in
    full -- the band is GROSS even within one op). The decomposition (observe
    items 7-8) rides beside it: every accepted contribution is classified by
    STORAGE identity into ``saved_parameter`` / ``saved_buffer`` /
    ``saved_activation`` sub-counters, and a TRACE-GLOBAL first-save counter
    (``newly_saved_bytes``) charges each unique storage's bytes exactly once
    at the op that saved it first -- the timeline's one honestly stackable
    saved band. Session-time bookkeeping, never persisted.

    Parameters
    ----------
    trace
        Active trace (session storage maps + the trace-global first-save set).
    output
        Raw function output from a decorated torch operation.

    Returns
    -------
    list[_OutputTensorEntry]
        Output entries in logging order, each paired with its autograd saved
        tensor byte/count stats and saved-band decomposition when available.
    """

    raw_entries: list[tuple[Any, tuple[OutputPathComponent, ...], ContainerSpec | None]] = list(
        _walk_output_tensors_with_paths(output)
    )
    if not raw_entries:
        raw_entries = [(out, (), None) for out in ensure_iterable(output)]

    seen_saved_storages: set[int] = trace.__dict__.setdefault(
        "_autograd_seen_saved_storages", set()
    )
    partitioned_entries: list[_OutputTensorEntry] = []
    seen_grad_fns: set[int] = set()
    seen_data_ptrs: set[int] = set()
    for maybe_tensor, container_path, container_spec in raw_entries:
        autograd_stats: tuple[int | None, int | None] = (None, None)
        autograd_band: dict[str, int] | None = None
        # TorchLens's own bookkeeping read: an in-place op's output IS its
        # receiver, so an unmarked ``grad_fn`` read on a registered buffer
        # (BN ``num_batches_tracked.add_(1)``) would record a phantom
        # declared-state fact (the unread-bit contract).
        with internal_scalar_read():
            grad_fn_handle = (
                maybe_tensor.grad_fn if isinstance(maybe_tensor, torch.Tensor) else None
            )
        if grad_fn_handle is not None:
            grad_fn_object_id = id(grad_fn_handle)
            if grad_fn_object_id in seen_grad_fns:
                autograd_stats = (0, 0)
            else:
                seen_grad_fns.add(grad_fn_object_id)
                autograd_stats, autograd_band = _saved_tensor_stats(
                    trace, grad_fn_handle, seen_data_ptrs, seen_saved_storages
                )
        partitioned_entries.append(
            _OutputTensorEntry(
                value=maybe_tensor,
                container_path=container_path,
                container_spec=container_spec,
                autograd_stats=autograd_stats,
                autograd_band=autograd_band,
            )
        )

    return partitioned_entries


def _output_entries_with_autograd_stats(trace: Any, output: Any) -> list[_OutputTensorEntry]:
    """Return the output entries of one call, as the partition below builds them.

    A single plain tensor gets its one entry without the generic output walk (the walk
    yields exactly ``(output, (), None)`` for it), with the statistics and band of its
    autograd node when it has one; every other output goes through
    :func:`_partition_output_entries_with_autograd_stats`. The ``grad_fn`` read is
    the C getset original: unwitnessed and unfingerprinted, as the partition's
    marked read arranges.
    """

    if type(output) is not torch.Tensor:
        return _partition_output_entries_with_autograd_stats(trace, output)
    grad_fn = raw_grad_fn(output)
    trace_fields = trace.__dict__
    if grad_fn is None:
        if "_autograd_seen_saved_storages" not in trace_fields:
            trace_fields["_autograd_seen_saved_storages"] = set()
        return [
            _OutputTensorEntry(
                value=output, container_path=(), container_spec=None, autograd_stats=(None, None)
            )
        ]
    seen_saved_storages = trace_fields.get("_autograd_seen_saved_storages")
    if seen_saved_storages is None:
        seen_saved_storages = trace_fields["_autograd_seen_saved_storages"] = set()
    stats, band = _saved_tensor_stats(trace, grad_fn, set(), seen_saved_storages)
    return [
        _OutputTensorEntry(
            value=output,
            container_path=(),
            container_spec=None,
            autograd_stats=stats,
            autograd_band=band,
        )
    ]


def _register_call_output_container_snapshot(
    trace: Trace,
    output: Any,
    *,
    output_entries: list[_OutputTensorEntry],
    func_call_id: int,
    event_index: int,
) -> None:
    """Register a function-call output container snapshot when opt-in capture is active.

    Parameters
    ----------
    trace
        Active trace.
    output
        Raw function output object.
    output_entries
        Path-aware tensor output entries from the existing output walker.
    func_call_id
        Function call identifier for the output boundary.
    event_index
        Capture event index used for ordering.
    """

    if not (
        getattr(trace, "is_replayable", False)
        or getattr(trace, "_capture_container_structure", False)
    ):
        return
    spec = next((entry.container_spec for entry in output_entries if entry.container_spec), None)
    if spec is None:
        return
    registry = trace._wrapper_runtime_ws.container_registry
    registry.register_snapshot(
        output,
        site=FuncSite(func_call_id=func_call_id, position="return"),
        role=Role.CALL_OUTPUT,
        phase=Phase.POST_CALL,
        observed_at_event_index=event_index,
        spec=spec,
        leaf_occurrences=_container_leaf_occurrences_from_entries(output_entries),
        reconstructable=True,
    )


def register_call_input_container_snapshots(
    trace: Trace,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    func_call_id: int,
    event_index: int,
) -> None:
    """Register tensor-bearing function-call input containers.

    Parameters
    ----------
    trace:
        Active trace.
    args:
        Function positional arguments at call entry.
    kwargs:
        Function keyword arguments at call entry.
    func_call_id:
        Function-call identifier for this boundary.
    event_index:
        Monotonic ordering value for the pre-call snapshot.
    """

    if not getattr(trace, "_capture_container_structure", False):
        return
    registry = trace._wrapper_runtime_ws.container_registry
    for index, arg in enumerate(args):
        result = walk_container(arg, role=Role.CALL_INPUT, capability="full_spec")
        if result is None:
            continue
        registry.register_snapshot(
            arg,
            site=FuncSite(func_call_id=func_call_id, position=("arg", index)),
            role=Role.CALL_INPUT,
            phase=Phase.PRE_CALL,
            observed_at_event_index=event_index,
            spec=result.spec,
            leaf_occurrences=result.leaf_occurrences,
            reconstructable=result.reconstructable,
        )
    for key, value in kwargs.items():
        result = walk_container(value, role=Role.CALL_INPUT, capability="full_spec")
        if result is None:
            continue
        registry.register_snapshot(
            value,
            site=FuncSite(func_call_id=func_call_id, position=("kwarg", key)),
            role=Role.CALL_INPUT,
            phase=Phase.PRE_CALL,
            observed_at_event_index=event_index,
            spec=result.spec,
            leaf_occurrences=result.leaf_occurrences,
            reconstructable=result.reconstructable,
        )


def _container_leaf_occurrences_from_entries(
    output_entries: list[_OutputTensorEntry],
) -> tuple[ContainerLeafOccurrence, ...]:
    """Build ordered leaf occurrences from path-aware output entries.

    Parameters
    ----------
    output_entries
        Path-aware tensor output entries.

    Returns
    -------
    tuple[ContainerLeafOccurrence, ...]
        Occurrence records preserving repeated tensors at multiple paths.
    """

    occurrences: list[ContainerLeafOccurrence] = []
    for occ_index, entry in enumerate(output_entries):
        producer_label = (
            get_tensor_label(entry.value) if isinstance(entry.value, torch.Tensor) else None
        )
        occurrences.append(
            ContainerLeafOccurrence(
                path=entry.container_path,
                producer_op_label=producer_label,
                tensor_identity=producer_label,
                occ_index=occ_index,
            )
        )
    return tuple(occurrences)


def _get_autograd_saved_stats_for_tensor(
    tensor: torch.Tensor,
) -> tuple[int | None, int | None]:
    """Measure autograd-saved tensor bytes/counts for a single tensor output.

    Parameters
    ----------
    tensor
        Tensor output from a decorated torch operation.

    Returns
    -------
    tuple
        ``(autograd_memory, num_autograd_tensors)``. Both values
        are ``None`` when no grad_fn_handle exists.
    """
    if tensor.grad_fn is None:
        return None, None
    stats_by_index = _get_autograd_saved_stats_by_output(tensor)
    return stats_by_index.get(0, (None, None))


# Late imports: the split modules of this package import one another (cycles), so each
# imports its siblings once every definition above exists.
from ._ops_containers import (  # noqa: E402
    _walk_output_tensors_with_paths,
)
from ._ops_exhaustive import (  # noqa: E402
    _collect_tensor_values,
    _iter_autograd_saved_candidates,
)
from .ops import (  # noqa: E402
    _AUTOGRAD_SAVED_ATTR_PREFIX,
    _OutputTensorEntry,
)
