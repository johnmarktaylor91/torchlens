"""Dispatch census and runnable-ledger processing."""

from __future__ import annotations

import threading
import weakref
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

import torch
import torch.utils.dlpack  # noqa: F401  (ensure torch.utils.dlpack.to_dlpack is importable to patch)

from ... import _state
from ...utils._torch_compat import tensor_version_or_none
from .buffer_writes import peek_buffer_write_tracker
from .escape_detection import (
    ExpectedOriginalToken,
)

if TYPE_CHECKING:
    from .completeness_witness import (
        _HOST_ESCAPE_MUTABLE_WRITEBACK,
        _HOST_ESCAPE_RAW_POINTER,
        _ORIG_TENSORBASE_UNTYPED_STORAGE,
        _ORIG_UNTYPED_STORAGE_DATA_PTR,
        _PLAIN_PARAM_CLS,
        _PLAIN_TENSOR_CLS,
        _PURE_VIEW_DISPATCH_OPERATORS,
        _RUNNABLE_LEDGER_FACTS,
        HOST_ESCAPE_OPERATORS,
        MUTABLE_ALIAS_ESCAPE_FUNCS,
        STATE_METADATA_MIRROR,
        STORAGE_BRIDGE_ESCAPE_FUNCS,
        _dispatch_callsite,
        _DispatchEvent,
        _internal_read_active,
        _is_expected_opaque_dispatch,
        _nonowner_escape_observe,
        _observe_state_metadata_read,
        _operator_base_name,
        _record_escape_source_tensor,
        _register_storage_origin,
        _WitnessState,
        internal_scalar_read,
    )

__all__ = (
    "record_uncaptured_owner_callsite",
    "_dispatch_result_holds_tensor",
    "_dispatch_credit_refs",
    "_event_builds_boundary_output",
    "_event_is_capture_accounted",
    "runnable_ledger_facts",
    "_tensor_abs_byte_span",
    "_as_strided_result_contained",
    "_finalize_runnable_ledger",
    "_whole_storage_uint8",
    "_snapshot_writeback_source",
    "_iter_dispatch_tensors",
    "_sample_writeback_at_consumption",
    "_has_state_toctou_watch",
    "_sample_state_toctou_at_consumption",
    "_split_consumed_state_items",
    "_sample_param_toctou_at_consumption",
    "_param_baseline_differs",
    "_sample_buffer_toctou_at_consumption",
    "_buffer_expected_differs",
    "_make_invisible_escape_wrapper",
)


def record_uncaptured_owner_callsite(token: ExpectedOriginalToken | None) -> None:
    """Attach a user callsite only when an owned interval emitted no op.

    Parameters
    ----------
    token:
        Completed exact wrapper token, if diagnostics were armed.

    Returns
    -------
    None
        The token receives a stable file, line, and function tuple in place.
    """

    if token is None:
        return
    callsite = _dispatch_callsite()
    token.capture_callsite = (callsite.file, callsite.line, callsite.function)


def _dispatch_result_holds_tensor(result: Any) -> bool:
    """Return whether an aten dispatch result contains any tensor (one level deep)."""

    if isinstance(result, torch.Tensor):
        return True
    if isinstance(result, (list, tuple)):
        return any(isinstance(item, torch.Tensor) for item in result)
    return False


def _dispatch_credit_refs(
    owner: ExpectedOriginalToken | None, func: Any, args: tuple[Any, ...], result: Any
) -> tuple[weakref.ref[torch.Tensor], ...]:
    """Return weak references to the tensors that can tie this dispatch to a boundary.

    Those are the tensor results (one level deep) and, for a pure alias operator
    (``aten.detach`` / ``aten.alias``), its operand. Recorded only for owners whose
    boundary credit is output-scoped, so other dispatches pay one attribute test and
    allocate nothing.
    """

    if owner is None or not owner.boundary_credit_is_output_scoped:
        return ()
    refs: list[weakref.ref[torch.Tensor]] = []
    if isinstance(result, torch.Tensor):
        refs.append(weakref.ref(result))
    elif isinstance(result, (list, tuple)):
        refs.extend(weakref.ref(item) for item in result if isinstance(item, torch.Tensor))
    if (
        args
        and isinstance(args[0], torch.Tensor)
        and _operator_base_name(func) in _PURE_VIEW_DISPATCH_OPERATORS
    ):
        refs.append(weakref.ref(args[0]))
    return tuple(refs)


def _event_builds_boundary_output(
    event: _DispatchEvent, boundary_outputs: Mapping[int, tuple[torch.Tensor, str]]
) -> bool:
    """Return whether the event built, or purely aliased, an exact boundary tensor object.

    Identity is object identity through a weak reference: ``id`` alone is unsound because
    a freed intermediate's address can be reused by a later boundary tensor, and storage
    geometry would equate a view or detach with its base. The rule for aliases:

    * a dispatch that returned the boundary object built it and is credited, including a
      view or detach whose result is the module output;
    * the op that produced that view's base did not build a boundary object and is not
      credited (it is a separate computation the boundary does not attest);
    * a pure alias (``aten.detach`` / ``aten.alias``) taken OF a boundary tensor, such as
      the detached copy autograd saves for an op whose backward reads its output, computes
      no value; any use of the alias is its own dispatch and stays visible, so it is
      credited. Other views of a boundary tensor are not.
    """

    for ref in event.credit_refs:
        tensor = ref()
        if tensor is None:
            continue
        entry = boundary_outputs.get(id(tensor))
        if entry is not None and entry[0] is tensor:
            return True
    return False


def _event_is_capture_accounted(event: _DispatchEvent) -> bool:
    """Return whether the event is represented by its owner's captured artifact.

    Ordinary wrapped operations account for their complete aten decomposition. A token
    credited by synthesized boundary Ops is narrower: it accounts for the non-mutating
    opaque output construction represented by those exact boundary tensors, but never
    for a mutating dispatch. Mutations always remain visible because a functionless
    boundary cannot attest their side effects on existing graph values. For a
    module-forward token (output-scoped credit) only a dispatch that returned one of the
    exact boundary tensor objects (or purely aliased one) is accounted; every other dispatch in the module body,
    such as a stale raw op whose value flowed elsewhere, stays visible. A raw
    replacement-hook token keeps whole-interval credit for its non-mutating dispatches.

    Parameters
    ----------
    event:
        Dispatcher event being finalized.

    Returns
    -------
    bool
        Whether this exact event has a captured representation.
    """

    owner = event.owner
    if owner is None or owner.capture_accounted is not True:
        return False
    boundary_outputs = owner.capture_accounted_outputs
    if not boundary_outputs:
        return True
    if event.mutates:
        return False
    if not owner.boundary_credit_is_output_scoped:
        return True
    return _event_builds_boundary_output(event, boundary_outputs)


def runnable_ledger_facts(trace: Any) -> tuple[Mapping[str, Any], ...]:
    """Return the recorded undischarged lifecycle facts for one trace."""

    return tuple(_RUNNABLE_LEDGER_FACTS.get(trace, ()))


def _tensor_abs_byte_span(value: torch.Tensor) -> tuple[int, int] | None:
    """Absolute (start, end] byte span a strided tensor's elements touch, or ``None``.

    Local minimal span math (min/max stride contributions on absolute addresses);
    the full shared relation engine lives in ``utils.tensor_utils`` -- this helper
    only answers CONTAINMENT for the as_strided audited row and fails ``None``-closed.
    """

    try:
        with internal_scalar_read():
            base = int(value.untyped_storage().data_ptr())
        esize = int(value.element_size())
        if base == 0 and value.numel() > 0:
            return None
        origin = base + int(value.storage_offset()) * esize
        if value.numel() == 0:
            return (origin, origin)
        low = 0
        high = 0
        for size, stride in zip(value.shape, value.stride()):
            contribution = (int(size) - 1) * int(stride)
            if contribution < 0:
                low += contribution
            else:
                high += contribution
        return (origin + low * esize, origin + high * esize + esize)
    except (RuntimeError, AttributeError, TypeError, ValueError, NotImplementedError):
        return None


def _as_strided_result_contained(args: tuple[Any, ...], result: Any) -> bool:
    """Return whether an ``aten.as_strided`` result's byte span sits inside its operand's."""

    if not args or not isinstance(args[0], torch.Tensor) or not isinstance(result, torch.Tensor):
        return False
    with _state.pause_logging():
        operand_span = _tensor_abs_byte_span(args[0])
        result_span = _tensor_abs_byte_span(result)
    if operand_span is None or result_span is None:
        return False
    return operand_span[0] <= result_span[0] and result_span[1] <= operand_span[1]


def _finalize_runnable_ledger(state: _WitnessState) -> None:
    """Discharge every observed dispatch event or record an incomplete fact (r35 I2, r37 INV-1).

    EXHAUSTIVE over the outcome vocabulary: every event terminates in exactly one
    explicit disposition -- accounted modeled call, exact audited opaque boundary,
    replacement-hook construction, escape-net witness, ``.data``-accessor state view,
    audited pure-view row, or an explicit incomplete fact. Discharge rules
    (owner-accounted; no exception-type or framework-file exemptions): a subevent
    whose enclosing wrapper owner became an accounted modeled call is discharged
    (replaying the owner replays its internals); a host-return witnessed by the exact
    escape net (``HOST_ESCAPE_OPERATORS``) is discharged (post-hon2_1 the net is
    total: every operand records a positive attribution or a fail-closed flag). A
    ``returned_tensor`` event -- the corr2-1 class -- is NEVER implicitly discharged:
    an unowned mutating dispatch records ``opaque_side_effect`` and an unowned
    non-mutating value-producing dispatch records ``unmodeled_tensor_return`` (its
    product can bake into a later traced call as an unwitnessed constant). An
    unhandled outcome value is a hard internal error, never a silent pass. This
    finalize runs only when the forward COMPLETED -- an undischarged raise means the
    exception was caught before forward completion: exception-driven control flow the
    sparse replay cannot witness.
    """

    facts: list[dict[str, Any]] = []
    for event in state.events:
        owner = event.owner
        owner_accounted = _event_is_capture_accounted(event)
        audited_opaque = owner is not None and _is_expected_opaque_dispatch(event.operator, owner)
        # ``_operator_name`` yields overload-qualified names (``aten.equal.default``);
        # the allowlists hold overload-stripped base names.
        base_operator = (
            event.operator.rsplit(".", 1)[0] if event.operator.count(".") >= 2 else event.operator
        )
        if event.outcome == "raised":
            if owner_accounted or audited_opaque or event.in_replacement_hook:
                continue
            facts.append(
                {
                    "kind": "caught_exception_control",
                    "operator": event.operator,
                    "owner_wrapper": owner.wrapper_name if owner is not None else None,
                    "owner_func_name": owner.func_name if owner is not None else None,
                    "exception_type": event.exception_type,
                    "mutates": bool(event.mutates),
                }
            )
        elif event.outcome == "returned_host_or_none":
            if owner_accounted or audited_opaque or event.in_replacement_hook:
                continue
            if base_operator in HOST_ESCAPE_OPERATORS:
                # Witnessed exactly by the tensor->host escape net.
                continue
            if event.state_view_accessor:
                continue
            if event.metadata_witnessed and not event.mutates:
                # r67 C3: witnessed exactly by the placement metadata net -- the wrapper
                # recorded the receiver's observed value (state observation ledger) or
                # input fact, so the producer gate owns the honesty decision. An
                # unattributed receiver never sets the flag and stays an incomplete fact.
                continue
            facts.append(
                {
                    "kind": "opaque_side_effect" if event.mutates else "unmodeled_host_return",
                    "operator": event.operator,
                    "owner_wrapper": owner.wrapper_name if owner is not None else None,
                    "owner_func_name": owner.func_name if owner is not None else None,
                    "exception_type": None,
                    "mutates": bool(event.mutates),
                }
            )
        elif event.outcome == "returned_tensor":
            if owner_accounted or audited_opaque or event.in_replacement_hook:
                continue
            if event.state_view_accessor:
                continue
            if not event.mutates and base_operator in _PURE_VIEW_DISPATCH_OPERATORS:
                continue
            if not event.mutates and event.contained_view:
                # Audited span-contained ``as_strided`` (DLPack/array-interop restride).
                continue
            facts.append(
                {
                    "kind": "opaque_side_effect" if event.mutates else "unmodeled_tensor_return",
                    "operator": event.operator,
                    "owner_wrapper": owner.wrapper_name if owner is not None else None,
                    "owner_func_name": owner.func_name if owner is not None else None,
                    "exception_type": None,
                    "mutates": bool(event.mutates),
                }
            )
        elif event.outcome == "started":
            # A dispatch that neither returned nor raised cannot exist on a completed
            # forward; record fail-closed rather than silently passing (INV-1).
            facts.append(
                {
                    "kind": "unclassified_event",
                    "operator": event.operator,
                    "owner_wrapper": owner.wrapper_name if owner is not None else None,
                    "owner_func_name": owner.func_name if owner is not None else None,
                    "exception_type": None,
                    "mutates": bool(event.mutates),
                }
            )
        else:  # pragma: no cover - unreachable by construction
            raise AssertionError(
                f"Internal invariant violation: unhandled dispatch outcome {event.outcome!r}; "
                "every outcome value must have an explicit ledger disposition (INV-1)."
            )
    if facts:
        _RUNNABLE_LEDGER_FACTS.setdefault(state.trace, []).extend(facts)


def _whole_storage_uint8(source: torch.Tensor) -> torch.Tensor:
    """Return a ``uint8`` tensor viewing ``source``'s ENTIRE untyped storage (all bytes).

    A zero-copy alias (``source.numpy()`` / ``source.untyped_storage()`` / a raw ``data_ptr``)
    shares the WHOLE storage, not just ``source``'s element extent: ``np.as_strided`` and a
    storage ``__setitem__`` can write bytes OUTSIDE ``source``'s view window (r15-H2). Comparing
    the whole aliased storage -- not ``source.detach().clone()`` (its own extent only) -- makes ANY
    host write anywhere in the shared storage detectable at forward end.
    """

    untyped = source.untyped_storage()
    view = torch.empty(0, dtype=torch.uint8, device=source.device)
    view.set_(untyped, 0, (untyped.nbytes(),), (1,))
    return view


def _snapshot_writeback_source(state: _WitnessState, source: torch.Tensor) -> None:
    """Record a before-image of a mutable zero-copy alias source for later write-back detection.

    Snapshots ``source``'s version and a detached byte clone of its WHOLE untyped storage under
    ``pause_logging`` (so the clone is not itself captured or censused) and holds a strong ref to
    ``source`` so the shared storage stays alive until the forward-end comparison. Snapshotting the
    full aliased storage (not just ``source``'s element extent) closes the r15-H2 out-of-extent
    gap: a host write through the alias's storage handle to bytes OUTSIDE ``source``'s view window
    (storage ``__setitem__`` / ``np.as_strided``) is still caught. A source that cannot be
    snapshotted (e.g. a meta tensor with no storage) fails closed immediately.
    """

    try:
        with _state.pause_logging():
            version = tensor_version_or_none(source)
            before = _whole_storage_uint8(source).clone()
    except (RuntimeError, TypeError, NotImplementedError):
        _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
        return
    state.writeback_watch.append((source, version, before))


def _iter_dispatch_tensors(
    args: tuple[Any, ...], kwargs: dict[str, Any] | None
) -> Iterator[torch.Tensor]:
    """Yield every ``torch.Tensor`` operand of a dispatch, flattening list/tuple containers.

    Aten operands are tensors, scalars, or (for ``cat`` / ``stack`` / ``_foreach_*``) lists/tuples
    of tensors; this walks those one-deep-or-more without importing a pytree so the per-consumption
    watch can see every tensor an op reads.
    """

    stack: list[Any] = list(args)
    if kwargs:
        stack.extend(kwargs.values())
    while stack:
        item = stack.pop()
        if isinstance(item, torch.Tensor):
            yield item
        elif isinstance(item, (list, tuple)):
            stack.extend(item)


def _sample_writeback_at_consumption(
    state: _WitnessState, args: tuple[Any, ...], kwargs: dict[str, Any] | None
) -> None:
    """Detect a transient host write-back that is LIVE when a traced op CONSUMES a watched source.

    A mutable zero-copy alias (``numpy`` / ``__array__`` / a storage handle) can be written,
    consumed by a downstream traced op, then byte-exactly RESTORED before forward end -- so the
    single end-of-forward compare (:func:`_check_writeback_watch`) sees ``before == after`` and
    would falsely VERIFY (r16-H1 TOCTOU). Sampling the watched source's WHOLE-STORAGE bytes at each
    CONSUMPTION catches the write while it is live in a traced op's input, then rolls it back.

    Soundness (no over-trigger). Only a byte difference with the source's version UNCHANGED since
    the exposure is flagged:

    * version UNCHANGED + bytes differ -> NO tracked in-place op touched the storage, so the diff is
      an opaque host write-back that is LIVE for this traced consumer -> UNVERIFIABLE (the TOCTOU);
    * version BUMPED -> a TRACKED, replayable in-place op is responsible for the diff, so the
      per-consumption sample defers to the end-of-forward compare (which conservatively handles a
      version-bumped byte diff). Flagging it here would falsely trip a legitimate tracked-op
      sequence that later restores the bytes (e.g. ``arr=y.numpy(); y.add_(1); z=y*2; y.sub_(1)``).

    A read-only ``.numpy().sum()`` never changes the bytes, so it is never flagged. The scan runs
    only while a mutable alias is live (``writeback_watch`` non-empty) and only compares a watched
    source when THIS op actually consumes its storage, so a transient write that never reaches a
    traced consumer of the source (restored before it is read) stays honestly VERIFIED.
    """

    if not state.writeback_watch and not _has_state_toctou_watch(state.trace):
        return
    # INV-2 annotation (r37): the ``data_ptr`` matching below is ATTRIBUTION-ONLY
    # identity -- it decides which watched source a consuming op MIGHT touch, and a
    # missed match merely defers to the end-of-forward WHOLE-STORAGE content compare
    # (which needs no pointer reasoning at all). Pointer identity is never used as a
    # disjointness proof here, so the absolute-interval engine is not required.
    try:
        with _state.pause_logging():
            consumed_ptrs: set[int] = set()
            for operand in _iter_dispatch_tensors(args, kwargs):
                if operand.is_meta:
                    # Every meta storage reports ``data_ptr() == 0`` (no identity) and holds
                    # no bytes, so no write is observable here; matching on 0 compared every
                    # meta buffer and fabricated the opaque host-write flag (W1-FAB: only an
                    # observed divergence may raise it). Reached once plane-P (distributed
                    # arming) installs the dispatch mode around a weights-free capture.
                    continue
                try:
                    consumed_ptrs.add(operand.untyped_storage().data_ptr())
                except (RuntimeError, TypeError, NotImplementedError):
                    continue
            if not consumed_ptrs:
                return
            if _sample_state_toctou_at_consumption(state, consumed_ptrs):
                return
            for source, version, before in state.writeback_watch:
                try:
                    if source.untyped_storage().data_ptr() not in consumed_ptrs:
                        continue
                    if tensor_version_or_none(source) != version:
                        continue
                    if not torch.equal(
                        _whole_storage_uint8(source), before
                    ):  # byte-exact uint8 view
                        _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
                        return
                except (RuntimeError, TypeError, NotImplementedError):
                    _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
                    return
    except (RuntimeError, TypeError, NotImplementedError):
        _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
        return


def _has_state_toctou_watch(trace: Any) -> bool:
    """Return whether the active trace has registered state byte watches.

    Returns
    -------
    bool
        ``True`` when buffer/parameter write tracking has live state snapshots.
    """

    tracker = peek_buffer_write_tracker(trace)
    if tracker is None:
        return False
    param_snapshots = getattr(tracker, "address_to_param_snapshot", None)
    buffer_snapshots = getattr(tracker, "address_to_expected_storage_snapshot", None)
    return bool(param_snapshots) or bool(buffer_snapshots)


def _sample_state_toctou_at_consumption(state: _WitnessState, consumed_ptrs: set[int]) -> bool:
    """Detect transient registered-state mutations when a traced op consumes them.

    Parameters
    ----------
    state:
        Active completeness-witness state.
    consumed_ptrs:
        Storage data pointers consumed by the current dispatcher operation.

    Returns
    -------
    bool
        ``True`` when an opaque state write-back was detected and recorded.
    """

    tracker = peek_buffer_write_tracker(state.trace)
    if tracker is None:
        return False
    if _sample_param_toctou_at_consumption(state, tracker, consumed_ptrs):
        return True
    return _sample_buffer_toctou_at_consumption(state, tracker, consumed_ptrs)


def _split_consumed_state_items(
    items: tuple[tuple[str, Any], ...], consumed_ptrs: set[int]
) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
    """Split registered state items into consumed-storage HITS and verbatim-path leftovers.

    PERF (w15 F1): the per-consumption TOCTOU scans read every registered param/buffer's
    CURRENT storage pointer on EVERY dispatched op -- O(ops x params), and each per-item
    wrapped ``untyped_storage().data_ptr()`` call additionally pays the armed numpy-RNG
    setprofile classifier's per-event toll, the multiplicative structure behind the
    quadratic runnable-producer capture. This helper keeps the FULL per-op scan (the scan
    itself is the r16-H1/r18 coverage: a registration-time ptr->address index goes STALE
    under a mid-forward ``p.data = other`` rebind and would falsely VERIFY a restored
    transient write) but batches the pointer reads through ``map`` over the true-original
    C accessors: C-to-C calls never enter the interpreter loop, so no profile event fires
    per item and no wrapper frame is paid. ``sys.setprofile`` hooks by contract only see
    interpreter-level calls, so nothing the RNG monitor could ever have classified is
    hidden -- user draw-sites always run through the interpreter and remain fully visible.

    Coverage is byte-identical to the verbatim loop: the raw read is only used for exact
    ``torch.Tensor`` / ``nn.Parameter`` instances (where the wrapped, paused spelling is a
    pure call-through to the same originals -- raw ``0`` pointers included), any exotic
    class falls to the leftover list for the verbatim wrapped per-item path, and ANY
    batch-read exception routes EVERY item to that verbatim path (which reproduces the
    original per-item skip semantics exactly).

    Returns
    -------
    tuple[list, list]
        ``(hits, leftovers)``: items whose current storage pointer is in
        ``consumed_ptrs`` (dict order), and items that must take the verbatim per-item
        path (exotic classes, or all items on a batch-read failure).
    """

    fast = [
        pair
        for pair in items
        if pair[1].__class__ is _PLAIN_TENSOR_CLS or pair[1].__class__ is _PLAIN_PARAM_CLS
    ]
    if len(fast) != len(items):
        leftovers = [
            pair
            for pair in items
            if not (pair[1].__class__ is _PLAIN_TENSOR_CLS or pair[1].__class__ is _PLAIN_PARAM_CLS)
        ]
    else:
        leftovers = []
    if not fast:
        return [], leftovers
    try:
        ptrs = list(
            map(
                _ORIG_UNTYPED_STORAGE_DATA_PTR,
                map(_ORIG_TENSORBASE_UNTYPED_STORAGE, [pair[1] for pair in fast]),
            )
        )
    except (RuntimeError, TypeError, NotImplementedError, AttributeError):
        # Fail SAFE, never fast: any batch failure sends every item through the verbatim
        # wrapped per-item path, which reproduces the original skip semantics per tensor.
        return [], list(items)
    hits = [pair for pair, ptr in zip(fast, ptrs) if ptr in consumed_ptrs]
    return hits, leftovers


def _sample_param_toctou_at_consumption(
    state: _WitnessState, tracker: Any, consumed_ptrs: set[int]
) -> bool:
    """Compare consumed parameters against their pre-forward byte snapshots.

    Parameters
    ----------
    state:
        Active completeness-witness state.
    tracker:
        Buffer/parameter write tracker attached to the active trace.
    consumed_ptrs:
        Storage data pointers consumed by the current dispatcher operation.

    Returns
    -------
    bool
        ``True`` when a consumed parameter differs from its pre-forward bytes.
    """

    tensors = getattr(tracker, "address_to_param_tensor", None)
    snapshots = getattr(tracker, "address_to_param_snapshot", None)
    if not isinstance(tensors, dict) or not isinstance(snapshots, dict):
        return False
    hits, leftovers = _split_consumed_state_items(tuple(tensors.items()), consumed_ptrs)
    for address, source in hits:
        if _param_baseline_differs(state, snapshots, address, source):
            return True
    for address, source in leftovers:
        if not isinstance(source, torch.Tensor):
            continue
        try:
            if source.untyped_storage().data_ptr() not in consumed_ptrs:
                continue
        except (RuntimeError, TypeError, NotImplementedError):
            continue
        if _param_baseline_differs(state, snapshots, address, source):
            return True
    return False


def _param_baseline_differs(
    state: _WitnessState, snapshots: dict[str, Any], address: str, source: torch.Tensor
) -> bool:
    """Compare one consumed param's whole-storage bytes against its pre-forward baseline."""

    baseline = snapshots.get(address)
    if not isinstance(baseline, tuple) or not baseline:
        return False
    before = baseline[0]
    if not isinstance(before, torch.Tensor):
        return False
    try:
        if not torch.equal(_whole_storage_uint8(source), before):  # byte-exact uint8 view
            _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
            return True
    except (RuntimeError, TypeError, NotImplementedError):
        _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
        return True
    return False


def _sample_buffer_toctou_at_consumption(
    state: _WitnessState, tracker: Any, consumed_ptrs: set[int]
) -> bool:
    """Compare consumed buffers against the journal-advanced expected bytes.

    Parameters
    ----------
    state:
        Active completeness-witness state.
    tracker:
        Buffer/parameter write tracker attached to the active trace.
    consumed_ptrs:
        Storage data pointers consumed by the current dispatcher operation.

    Returns
    -------
    bool
        ``True`` when a consumed buffer differs from its journal-advanced bytes.
    """

    tensors = getattr(tracker, "address_to_tensor", None)
    snapshots = getattr(tracker, "address_to_expected_storage_snapshot", None)
    if not isinstance(tensors, dict) or not isinstance(snapshots, dict):
        return False
    hits, leftovers = _split_consumed_state_items(tuple(tensors.items()), consumed_ptrs)
    for address, source in hits:
        if _buffer_expected_differs(state, snapshots, address, source):
            return True
    for address, source in leftovers:
        if not isinstance(source, torch.Tensor):
            continue
        try:
            if source.untyped_storage().data_ptr() not in consumed_ptrs:
                continue
        except (RuntimeError, TypeError, NotImplementedError):
            continue
        if _buffer_expected_differs(state, snapshots, address, source):
            return True
    return False


def _buffer_expected_differs(
    state: _WitnessState, snapshots: dict[str, Any], address: str, source: torch.Tensor
) -> bool:
    """Compare one consumed buffer's whole-storage bytes against its journal-advanced bytes."""

    expected = snapshots.get(address)
    if not isinstance(expected, torch.Tensor):
        return False
    try:
        if not torch.equal(_whole_storage_uint8(source), expected):  # byte-exact uint8 view
            _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
            return True
    except (RuntimeError, TypeError, NotImplementedError):
        _HOST_ESCAPE_MUTABLE_WRITEBACK.add(state.trace)
        return True
    return False


def _make_invisible_escape_wrapper(original: Any, state: _WitnessState, name: str) -> Any:
    """Wrap a tensor->host conversion method to record its SOURCE, then call through.

    The wrapper records the receiver tensor (the escape SOURCE) into the shared escape
    tables, gated to the active trace so a TorchLens-internal conversion (run under
    ``pause_logging``) is never mistaken for a user escape. Fires on every thread under
    the r43 owner-vs-non-owner rule: the OWNER thread (gated on ``_logging_enabled``)
    records through the full precise ladder plus the blanket flags/watches; a NON-owner
    thread (gated on ``belt_armed``) ceilings only when it touches a captured tensor and
    otherwise records nothing (a benign background thread touching its OWN tensors never
    ceilings the capture). For a
    mutable zero-copy alias conversion (``numpy`` / ``__array__``) it also records a
    before-image so a subsequent host write-back through the alias is detected at
    forward end. It always calls the original method unchanged, so values, goldens, and
    outputs are byte-identical. r43: the OWNER thread keeps the precise ladder; a NON-owner
    thread's captured-tensor touch ceilings via :func:`_nonowner_escape_observe`.
    """

    # Storage-pointer bridges (untyped_storage / storage / data_ptr) are watched for host
    # write-back but are WATCH-ONLY: a read-only pointer/identity check exposes no scalar value,
    # so recording them as value-escape sources would over-trigger and is deliberately skipped.
    is_storage_bridge = name in STORAGE_BRIDGE_ESCAPE_FUNCS
    watch_writeback = name in MUTABLE_ALIAS_ESCAPE_FUNCS or is_storage_bridge
    record_source = not is_storage_bridge
    # ``data_ptr()`` alone leaks a RAW pointer no forward-end byte watch can re-inspect (r15-H1);
    # a genuine user call fails closed to UNVERIFIABLE. ``untyped_storage()`` / ``storage()`` keep
    # the watch-only write-back treatment (their value reads are already UNVERIFIABLE).
    is_raw_pointer = name == "data_ptr"
    # r67 C3/C6: storage ACQUISITION registers the returned handle's ORIGIN (input site /
    # full state alias group / other) in the capture-scoped weak origin map -- and NOTHING
    # else. The former exposure-time geometry stamp ("``.nbytes()`` is one attribute away")
    # violated actual-read gating (corr1-4: a discarded handle on a larger-base slot false-
    # refused the save); the real accessor call on the handle now records through
    # ``STORAGE_METADATA_ACCESSOR_DISPOSITIONS``.
    registers_storage_origin = name in {"untyped_storage", "storage", "_typed_storage"}
    # r65 (closes r64 F2): a zero-copy VIEW export pins the receiver's full layout with no
    # accessor call at all -- ``numpy()``/``__array__`` expose ndarray ``.strides``/``.flags``
    # and DLPack capsules carry strides + byte offset. On a STATE-derived receiver that
    # geometry is a pure function of the slot's physical form (a view-of-state receiver
    # attributes to the slot exactly as r63), so the export records the exact-layout read
    # kinds. ``tolist()`` COPIES (layout-safe) and is deliberately excluded.
    records_state_view_geometry = name in {"numpy", "__array__", "__dlpack__"}

    def wrapper(self: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        """Record one tensor host-export access and then call through.

        Parameters
        ----------
        self:
            Tensor receiver for the export.
        *args:
            Positional arguments passed to ``original``.
        **kwargs:
            Keyword arguments passed to ``original``.

        Returns
        -------
        Any
            Result from ``original`` or the cached storage bridge result.
        """
        result_holder: dict[str, Any] = {}
        if isinstance(self, torch.Tensor) and _state._active_trace is state.trace:
            if threading.get_ident() == state.owner_thread_id:
                if _state._logging_enabled:
                    if record_source:
                        _record_escape_source_tensor(state.trace, self)
                    if records_state_view_geometry:
                        _observe_state_metadata_read(
                            state.trace, self, STATE_METADATA_MIRROR["stride"][1]
                        )
                        _observe_state_metadata_read(
                            state.trace, self, STATE_METADATA_MIRROR["storage_offset"][1]
                        )
                    if registers_storage_origin and not _internal_read_active():
                        storage = original(self, *args, **kwargs)
                        result_holder["value"] = storage
                        _register_storage_origin(state, self, storage)
                    # A raw ``data_ptr()`` pointer is unobservable; only a genuine USER call
                    # (internal marker inactive -- TorchLens's own bookkeeping ``data_ptr``
                    # reads run under it) fails closed so the tensor's subsequent value cannot
                    # be silently VERIFIED.
                    if is_raw_pointer and not _internal_read_active():
                        _HOST_ESCAPE_RAW_POINTER.add(state.trace)
                    # TorchLens's OWN capture-internal aliasing / version bookkeeping reads
                    # storage pointers (``aliasing._tensors_alias`` ->
                    # ``untyped_storage().data_ptr()``) under the explicit ``internal_scalar_read``
                    # marker. Those are NOT user exposures: snapshotting them and byte-comparing
                    # under the r14-H1 gate would falsely trip on a later legitimate TRACKED
                    # in-place op. Only watch a storage bridge when the marker is inactive -- a
                    # genuine user ``data_ptr()`` / ``storage()`` call. (The numpy / __array__
                    # mutable alias is never called internally, so it is always watched, as r13.)
                    if watch_writeback and not (is_storage_bridge and _internal_read_active()):
                        _snapshot_writeback_source(state, self)
            elif state.belt_armed:
                # r43: a non-owner touch of a captured tensor (this receiver, or a captured-
                # derived alias by storage identity) ceilings the capture. A benign OWN-tensor
                # conversion records nothing.
                _nonowner_escape_observe(state, self)
        if "value" in result_holder:
            return result_holder["value"]
        return original(self, *args, **kwargs)

    return wrapper
