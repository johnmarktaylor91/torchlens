"""Tensor-origin classification and propagation."""

from __future__ import annotations

import weakref
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
import torch.utils.dlpack  # noqa: F401  (ensure torch.utils.dlpack.to_dlpack is importable to patch)

from ... import _state
from ._completeness_types import _WitnessState
from ._tl import (
    get_tensor_label,
)
from .buffer_writes import session_validated_buffer_address

if TYPE_CHECKING:
    from .completeness_witness import (
        _CAPTURED_STORAGE_PTRS,
        _DISPATCH_TENSOR_ORIGINS,
        _ORIGIN_FLATTEN_DEPTH_LIMIT,
        _ORIGIN_LABEL_PREFIX,
        _ORIGIN_RNG,
        _ORIGIN_STATE_PREFIX,
        _ORIGIN_UNINIT,
        _ORIGIN_UNKNOWN,
        _STORAGE_REBIND_BARRIER_LABELS,
        HOST_ESCAPE_OPERATORS,
        _as_strided_result_contained,
        _internal_read_active,
        _internal_read_state,
        _is_aten_operator,
        _operator_base_name,
        _raw_storage_ptr_no_observe,
        _record_escape_source_tensor,
        _TensorOriginRegistry,
    )

__all__ = (
    "internal_scalar_read",
    "_escape_source_is_torchlens_internal",
    "_output_is_host_value",
    "_iter_tensor_operands",
    "_record_host_escape_source",
    "_escape_storage_ptr",
    "_param_derived_addresses",
    "_iter_tensors_deep",
    "_operand_origins",
    "_operand_leaf_origins",
    "_operator_is_seeded_rng",
    "_operator_uninit_family_tail",
    "_python_tensor_method_uninit_family_tail",
    "_operator_is_growth_resize",
    "_operator_total_writer_destination",
    "_live_deterministic_fill_governs",
    "_register_dispatch_result_origins",
)


@contextmanager
def internal_scalar_read() -> Iterator[None]:
    """Mark a region as a genuine TorchLens capture-internal scalar/comparison read.

    TorchLens itself reads a scalar/bool from a tensor during capture -- extracting a
    scalar-``bool`` op-output value (``ops._log_output_tensor_info``), comparing a
    pre-call input copy against its post-call value for mutation/alias detection
    (``tensor_nanequal`` via ``detect_torch_alias_contract`` and the child-version
    snapshot), and similar bookkeeping reads. Those reads lower to
    ``aten._local_scalar_dense`` (or, for content comparisons, ``aten.equal`` /
    ``aten.allclose`` returning a Python ``bool``) and would otherwise be misrecorded as
    USER host escapes and falsely trip the fail-closed INCOMPLETE gates. This context
    manager marks them EXPLICITLY so the internal-vs-user classifier is an
    allowlist-BY-CONSTRUCTION: an escape is "internal" iff this marker is active, never
    because a stack frame's filename happens to resolve inside the ``torchlens`` package
    (which is spoofable by an ``exec``-compiled user helper carrying a torchlens
    ``co_filename``, and fails for a frameless C-callable). A genuine user escape runs
    with the marker inactive and is always recorded.
    """

    depth = getattr(_internal_read_state, "depth", 0)
    _internal_read_state.depth = depth + 1
    # An internal read is not a model call: keep it out of the ordered call
    # fingerprint exactly as ``pause_logging()`` does (``_state._pause_depth``).
    _state._pause_depth += 1
    try:
        try:
            yield
        finally:
            _state._pause_depth -= 1
    finally:
        _internal_read_state.depth = depth


def _escape_source_is_torchlens_internal() -> bool:
    """Return whether an escape dispatch is a marked TorchLens capture-internal read.

    Classification is allowlist-BY-CONSTRUCTION: it is ``True`` iff an explicit
    :func:`internal_scalar_read` marker is live on this thread, set only around
    TorchLens's own genuine internal scalar/comparison reads. It NEVER infers "internal"
    from a stack frame's ``co_filename`` (spoofable via an ``exec``-compiled user helper
    given a torchlens filename, and undefined for a frameless C-callable such as
    ``operator.methodcaller("item")``). A USER escape therefore can never be classified
    internal, while TorchLens's own reads never trip the fail-closed gates.
    """

    return _internal_read_active()


def _output_is_host_value(result: Any) -> bool:
    """Return whether a dispatch output is a pure Python host value carrying no tensor.

    A host value is a Python scalar (``bool`` / ``int`` / ``float`` / ``complex``) or a
    ``list`` / ``tuple`` recursively of host values. A ``torch.Tensor`` output (or any
    container carrying a tensor) is an ordinary op result, NOT a host escape.
    """

    if isinstance(result, torch.Tensor):
        return False
    if isinstance(result, (bool, int, float, complex)):
        return True
    if isinstance(result, (list, tuple)):
        return len(result) > 0 and all(_output_is_host_value(value) for value in result)
    return False


def _iter_tensor_operands(args: tuple[Any, ...]) -> Iterator[torch.Tensor]:
    """Yield each tensor operand of a dispatch, including tensors nested one list/tuple deep."""

    for value in args:
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, torch.Tensor):
                    yield item


def _record_host_escape_source(trace: Any, func: Any, args: tuple[Any, ...], result: Any) -> None:
    """Record the raw producing-op label of every tensor->host VALUE-escape SOURCE.

    NARROW value-escape rule: the dispatch's operator (overload-stripped) must be one of
    the ``HOST_ESCAPE_OPERATORS`` VALUE reads (``aten._local_scalar_dense`` from
    ``.item()`` / ``int()`` / ``float()`` / ``__index__`` / ``bool()``, or the direct
    tensor->``bool`` predicates ``aten.equal`` / ``aten.allclose`` / ``aten.is_nonzero``)
    AND its output must be a pure Python host value. That host value can be baked into a
    downstream op literal (verbatim OR after arbitrary Python arithmetic) or steer
    pure-Python control flow -- neither of which the sparse DAG can recompute.

    The rule is DELIBERATELY not a general "any non-tensor output" census: tensor
    STRUCTURE/METADATA ops (``size`` / ``sym_size`` / ``numel`` / ``dim`` / ``stride`` /
    ``is_contiguous`` / ``dtype`` / ``device`` / ``storage_offset`` / ...) also return
    non-tensor host values, but those derive from shape/layout, are input-VALUE-
    independent (already covered by the separate input-shape-mismatch check), and a real
    model reads them constantly -- witnessing them over-triggers a false UNVERIFIABLE on
    escape-free models and pathologically slows per-op capture. Restricting to the value
    allowlist keeps genuine escapes witnessed without either regression.

    Every tensor operand of a value escape is recorded as a source; its raw capture label
    lets the runnable descriptor witness that source slot and, at run time, refuse a false
    VERIFIED when the slot recomputes different bytes for a changed input.
    """

    if not args or not _is_aten_operator(func):
        return
    if _operator_base_name(func) not in HOST_ESCAPE_OPERATORS:
        return
    if not _output_is_host_value(result):
        return
    for source in _iter_tensor_operands(args):
        _record_escape_source_tensor(trace, source)


def _escape_storage_ptr(source: torch.Tensor) -> int | None:
    """Return ``source``'s untyped-storage data pointer, read under the internal marker.

    The internal-read marker keeps this OWN resolution read from being mistaken for a user
    ``data_ptr()`` raw-pointer escape (the storage ``data_ptr`` accessor is patched for the forward).
    """

    try:
        with internal_scalar_read():
            return source.untyped_storage().data_ptr()
    except (RuntimeError, TypeError, NotImplementedError):
        return None


def _param_derived_addresses(trace: Any, source: torch.Tensor) -> set[str]:
    """Return the state address of the registered PARAMETER whose storage ``source`` aliases.

    DIRECT alias rung only (r18): ``source`` shares a param's storage (``self.w.detach()``,
    ``self.w[0]``, ``self.w.tolist()``, ``self.w.detach().numpy()``) -- its storage pointer
    is in the forward-start param index. Resolves for frozen params too (no autograd needed).

    r37 INV-1: the former DERIVED autograd rung (r19-C -- walk ``grad_fn`` back to
    ``AccumulateGrad`` leaves and declare purity when every leaf is a registered param) is
    REMOVED as an attribution mechanism. Measured on torch 2.8 (exp1, hon2_3): a DETACHED
    or non-differentiable-dtype operand (``x.data``, ``x.detach()``, a bool mask from
    ``x > 0``, a long index, ``where``'s condition) contributes NO autograd slot at all, so
    "every leaf is a param" never proves operand totality -- the walk blessed
    input-contaminated chains as pure-param (false VERIFIED, hon2_3). Pure param-derived
    reads are recovered ONLY through positive dispatch-origin propagation
    (:func:`_resolved_dispatch_origins`); no autograd-graph structural argument may ever
    serve as an operand-totality proof again (INV-1 banned mechanism).
    """

    param_storage_addresses = getattr(trace, "_param_storage_addresses", None)
    if not param_storage_addresses:
        return set()
    direct = _escape_storage_ptr(source)
    if direct is not None and direct in param_storage_addresses:
        return {str(param_storage_addresses[direct])}
    return set()


def _iter_tensors_deep(value: Any, depth: int = 0) -> Iterator[torch.Tensor]:
    """Yield every tensor in a dispatch argument/result container, bounded-depth."""

    if isinstance(value, torch.Tensor):
        yield value
        return
    if depth >= _ORIGIN_FLATTEN_DEPTH_LIMIT:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors_deep(item, depth + 1)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors_deep(item, depth + 1)


def _operand_origins(trace: Any, operand: torch.Tensor) -> frozenset[str]:
    """Resolve ONE dispatch operand to its positive value origins (or ``unknown``).

    Resolution ladder (first hit wins): capture label (a tagged input/op output is
    witnessable by its own slot digest); registered-state meta/buffer address; the
    dispatch-origin ledger (a previously registered unlabelled alias/product); direct
    registered-param storage identity. Anything unresolved is ``unknown`` -- an
    explicit taint, never an omission (INV-1).
    """

    label = get_tensor_label(operand)
    if isinstance(label, str):
        return frozenset({f"{_ORIGIN_LABEL_PREFIX}{label}"})
    # r81 (r80 F2): a ``state:`` origin requires the SESSION-VALIDATED stamp
    # (current-session object + live storage identity), never the raw static
    # stamp -- a stamped plain-attr buffer whose storage was ``.data``-rebound
    # to input data mid-forward must resolve as ``unknown`` (explicit taint),
    # not launder input-derived values into the residual-(3) state exemption.
    buffer_address = session_validated_buffer_address(trace, operand)
    if buffer_address is not None:
        return frozenset({f"{_ORIGIN_STATE_PREFIX}{buffer_address}"})
    registry = _DISPATCH_TENSOR_ORIGINS.get(trace)
    if registry is not None:
        entry = registry.get(operand)
        if entry is not None:
            return entry[0]
    param_storage_addresses = getattr(trace, "_param_storage_addresses", None)
    if param_storage_addresses:
        ptr = _escape_storage_ptr(operand)
        if ptr is not None and ptr in param_storage_addresses:
            return frozenset({f"{_ORIGIN_STATE_PREFIX}{param_storage_addresses[ptr]}"})
    return frozenset({_ORIGIN_UNKNOWN})


def _operand_leaf_origins(trace: Any, operand: torch.Tensor) -> frozenset[str]:
    """Resolve ONE operand to its TERMINAL leaf origins (state / input-or-boundary labels).

    Unlike :func:`_operand_origins` (where an interior op's own label wins -- the best,
    finest witness), leaf resolution propagates THROUGH interior labeled results down to
    a basis that survives orphan-pruning: registered state addresses and the labels of
    tensors that were never produced by an in-scope dispatch (model inputs and other
    boundary tensors). The producer consumes this basis as the fail-closed FALLBACK
    witness set for an escape whose direct source label was orphan-pruned (r37
    mechanism A: "falls back to propagated leaf origins for pruned labels").
    """

    # r81 (r80 F2): the leaf ``state:`` rung requires the SESSION-VALIDATED
    # stamp -- this ledger rung was the actual suppression vehicle for the
    # plain-attr ``.data=`` input-layout launder: the rebound receiver's raw
    # stamp resolved every downstream product to a pure ``state:`` leaf basis,
    # so the layout ladder recorded nothing (residual (3)) and the twin
    # false-VERIFIED. The belt makes the rebound receiver ``unknown`` instead.
    buffer_address = session_validated_buffer_address(trace, operand)
    if buffer_address is not None:
        return frozenset({f"{_ORIGIN_STATE_PREFIX}{buffer_address}"})
    registry = _DISPATCH_TENSOR_ORIGINS.get(trace)
    if registry is not None:
        entry = registry.get(operand)
        if entry is not None:
            # r29 F1: a ledger entry is identity-keyed, so it survives a
            # storage-SWAPPING ``.data=`` rebind of the SAME object and then
            # describes the PRE-rebind value -- the exact staleness that let a
            # rebound receiver's leaf basis resolve pure-``state:`` and launder
            # an input-layout read (sec1 false VERIFIED). A receiver currently
            # labeled by a barrier op resolves ``unknown`` (explicit taint):
            # the taint then propagates through every downstream registration,
            # so products of the rebound tensor fail closed too. Pointer-
            # preserving rebinds never register a barrier (r85 siblings keep
            # their attribution).
            rebind_barriers = _STORAGE_REBIND_BARRIER_LABELS.get(trace)
            if rebind_barriers:
                label = get_tensor_label(operand)
                if isinstance(label, str) and label in rebind_barriers:
                    return frozenset({_ORIGIN_UNKNOWN})
            return entry[1]
    param_storage_addresses = getattr(trace, "_param_storage_addresses", None)
    if param_storage_addresses:
        ptr = _escape_storage_ptr(operand)
        if ptr is not None and ptr in param_storage_addresses:
            return frozenset({f"{_ORIGIN_STATE_PREFIX}{param_storage_addresses[ptr]}"})
    label = get_tensor_label(operand)
    if isinstance(label, str):
        # Not a dispatch product: an input / boundary tensor whose label is terminal.
        return frozenset({f"{_ORIGIN_LABEL_PREFIX}{label}"})
    return frozenset({_ORIGIN_UNKNOWN})


def _operator_is_seeded_rng(func: Any) -> bool:
    """Return whether a dispatcher overload is torch-tagged nondeterministic-seeded."""

    try:
        return torch.Tag.nondeterministic_seeded in getattr(func, "tags", ())
    except (TypeError, RuntimeError):
        return True  # unreadable tags: treat as RNG (fail closed)


def _operator_uninit_family_tail(func: Any) -> str | None:
    """Return a dispatcher overload's base op name IF it is in the uninit family.

    r53 hon_2: the shared closed table lives in ``utils/rng.py`` (one predicate,
    three layers). At this layer the spelling is the overload-independent aten
    base name (``aten.empty_like`` -> ``empty_like``).
    """

    from ...utils.rng import _UNINIT_ALLOC_FACTORY_TAILS, _UNINIT_ALLOC_RESIZE_TAILS

    base = _operator_base_name(func)
    if not base.startswith("aten."):
        return None
    tail = base[len("aten.") :]
    if tail in _UNINIT_ALLOC_FACTORY_TAILS or tail in _UNINIT_ALLOC_RESIZE_TAILS:
        return tail
    return None


def _python_tensor_method_uninit_family_tail(
    namespace: str | None, qualname: str | None, args: tuple[Any, ...] = ()
) -> str | None:
    """Return a PYTHON-``torch.Tensor``-method spelling's uninit family tail (r55 hon_1).

    The dispatch-level origin ledger above keys the family off aten base names,
    which is complete for every spelling that REDISPATCHES an aten family op --
    including the legacy ``Tensor.new(sizes)``, whose size form redispatches
    ``aten.empty.memory_format`` (probed), so LIVE capture tainting already
    covers it transitively. This function is the DECLARED recognition of the
    family over the Python-``torch.Tensor``-method surface -- the spellings the
    load-side qualname classifier and the family-drift meta-test see (``new``
    has NO aten spelling: ``hasattr(torch.ops.aten, "new")`` is ``False``).
    It consults the single ``utils/rng.py`` table block (never re-derives):

    - a plain factory/resize tail (``empty_like``, ``resize_``) matches by
      qualname alone, exactly like the load-side classifier;
    - a SIZE-GATED tail (``new``) additionally requires the size-argument form;
      an UNDECIDABLE form (``uninit_new_call_is_size_form`` returning ``None``)
      fails closed to recognized-as-family, mirroring the grow-gate posture.

    The Python-Tensor-method drift meta-test
    (``tests/test_tlspec_runnable_r53_uninit_alloc.py``) enumerates
    ``torch.Tensor`` allocation-pattern methods against this recognition so a
    FUTURE python-only uninit factory with no aten spelling is a FAILING test,
    never a silent gap slipping both the aten drift test and this surface.
    """

    from ...utils.rng import (
        qualname_is_uninit_growth_resize,
        qualname_is_uninit_size_gated_alloc,
        qualname_is_uninitialized_alloc,
        uninit_new_call_is_size_form,
    )

    if not qualname:
        return None
    tail = qualname.rsplit(".", 1)[-1]
    if qualname_is_uninitialized_alloc(namespace, qualname) or qualname_is_uninit_growth_resize(
        namespace, qualname
    ):
        return tail
    if qualname_is_uninit_size_gated_alloc(namespace, qualname):
        if uninit_new_call_is_size_form(args) is False:
            return None  # data-form ``new([values])``/``new(tensor)``: deterministic copy
        return tail  # size form, or undecidable -> fail closed to tainted
    return None


def _operator_is_growth_resize(func: Any) -> bool:
    """Return whether a dispatcher overload is a resize spelling (grow-gated family)."""

    from ...utils.rng import _UNINIT_ALLOC_RESIZE_TAILS

    base = _operator_base_name(func)
    return base.startswith("aten.") and base[len("aten.") :] in _UNINIT_ALLOC_RESIZE_TAILS


def _operator_total_writer_destination(
    func: Any, args: tuple[Any, ...], kwargs: dict[str, Any] | None
) -> Any | None:
    """Return the tensor whose bytes this dispatch TOTALLY overwrites, or ``None``.

    Total writers per the shared r53 hon_2 sanitizer table: the ``out=`` kwarg
    destination (torch's ``out=`` convention IS a full overwrite; only an exact
    single-tensor ``out`` sanitizes) and the in-place
    ``copy_``/``zero_``/``fill_``/RNG-fill receivers. Partial or unprovable
    in-place writers return ``None`` (taint propagates, fail closed).
    """

    from ...utils.rng import _UNINIT_RNG_FILL_TAILS, _UNINIT_TOTAL_WRITER_TAILS

    if kwargs:
        out = kwargs.get("out")
        if isinstance(out, torch.Tensor):
            return out
    base = _operator_base_name(func)
    if base.startswith("aten."):
        tail = base[len("aten.") :]
        if tail in _UNINIT_TOTAL_WRITER_TAILS or tail in _UNINIT_RNG_FILL_TAILS:
            if args and isinstance(args[0], torch.Tensor):
                return args[0]
    return None


def _live_deterministic_fill_governs() -> bool:
    """Return whether the LIVE capture context proves deterministic uninit fill."""

    from ...utils._torch_compat import read_fill_uninitialized_memory
    from ...utils.rng import deterministic_fill_governs

    deterministic = bool(torch.are_deterministic_algorithms_enabled())
    return deterministic_fill_governs(deterministic, read_fill_uninitialized_memory())


def _register_dispatch_result_origins(
    state: _WitnessState,
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    result: Any,
    pre_dispatch_receiver_numel: int | None = None,
) -> None:
    """Propagate the union of operand origins onto every tensor result (mechanism A).

    A seeded-RNG operator additionally taints its results with the ``rng`` origin so a
    pruned host read of raw RNG output can never be attributed as a deterministic
    function of its operands. An in-place operator's mutated unlabelled receiver is
    covered because the receiver IS a result-aliasing operand: re-registering results
    updates its entry with the union (its VALUE now depends on all operands).

    r53 hon_2: an uninitialized-memory family op (``empty`` factories; a GROWING
    ``resize_``, decided against ``pre_dispatch_receiver_numel`` read at the
    interpose BEFORE the receiver was resized, failing closed to tainted when
    unavailable) additionally taints its results with the distinct ``uninit``
    origin -- closing the empty-operand-set hole where allocator garbage
    registered an EMPTY origin set and was later attributed as a "literal-only
    deterministic chain". A total writer (``out=`` destination,
    ``copy_``/``zero_``/``fill_``/RNG fill receiver) EXCLUDES the destination
    operand's prior origins from the union: the post-write value derives from
    the value-source operands only (exact value semantics -- strictly more
    precise, never less safe).
    """

    trace = state.trace
    display_union: set[str] = set()
    leaf_union: set[str] = set()
    # ``pause_logging`` + the internal marker: origin resolution reads storage
    # pointers/labels through torch-function-wrapped accessors; unpaused reads
    # would be logged as spurious ops mid-forward (shifting raw counters and
    # staling every label recorded after them) and would trip the escape census.
    # The r53 hon_2 metadata reads (``numel`` on the result, the ``out=`` kwarg
    # probe) live INSIDE the same paused scope for exactly that reason.
    with _state.pause_logging(), internal_scalar_read():
        total_write_destination = _operator_total_writer_destination(func, args, kwargs)
        for operand in _iter_tensors_deep(args):
            if operand is total_write_destination:
                continue
            display_union |= _operand_origins(trace, operand)
            leaf_union |= _operand_leaf_origins(trace, operand)
        if kwargs:
            for operand in _iter_tensors_deep(kwargs):
                if operand is total_write_destination:
                    continue
                display_union |= _operand_origins(trace, operand)
                leaf_union |= _operand_leaf_origins(trace, operand)
        exposes_uninit = False
        if _operator_uninit_family_tail(func) is not None:
            exposes_uninit = True
            result_numel = result.numel() if isinstance(result, torch.Tensor) else None
            if _operator_is_growth_resize(func):
                # Shrink/same-size preserves the element prefix (probed clean);
                # an unreadable pre-call size fails closed to tainted.
                exposes_uninit = (
                    pre_dispatch_receiver_numel is None
                    or result_numel is None
                    or result_numel > pre_dispatch_receiver_numel
                )
            elif result_numel == 0:
                exposes_uninit = False  # zero elements: no bytes to expose
            if exposes_uninit and _live_deterministic_fill_governs():
                exposes_uninit = False  # torch fills deterministically (probed NaN)
    if _operator_is_seeded_rng(func):
        display_union.add(_ORIGIN_RNG)
        leaf_union.add(_ORIGIN_RNG)
    if exposes_uninit:
        display_union.add(_ORIGIN_UNINIT)
        leaf_union.add(_ORIGIN_UNINIT)
    if _operator_base_name(func) == "aten.as_strided" and not _as_strided_result_contained(
        args, result
    ):
        # An out-of-span restride can address storage bytes outside every operand's
        # witnessed span: its value is NOT a function of the operands (fail closed).
        display_union.add(_ORIGIN_UNKNOWN)
        leaf_union.add(_ORIGIN_UNKNOWN)
    display = frozenset(display_union)
    leaf = frozenset(leaf_union)
    registry = _DISPATCH_TENSOR_ORIGINS.get(trace)
    if registry is None:
        registry = _TensorOriginRegistry()
        _DISPATCH_TENSOR_ORIGINS[trace] = registry
    # r43 CLASS 2: index every owner-produced activation's storage pointer so the non-owner
    # storage-identity catch-all recognizes a ``.data`` / view / detach alias of a captured
    # activation touched off-owner. The true-original accessor is wrapper-free.
    captured_ptrs = _CAPTURED_STORAGE_PTRS.get(trace)
    if captured_ptrs is None:
        captured_ptrs = {}
        _CAPTURED_STORAGE_PTRS[trace] = captured_ptrs
    for produced in _iter_tensors_deep(result):
        # Labeled results register too: the display ladder still prefers their own
        # label (finest witness), but the LEAF set must flow through them so a later
        # orphan-pruned chain can fall back to a surviving witness basis.
        registry.set(produced, display, leaf)
        produced_ptr = _raw_storage_ptr_no_observe(produced)
        if produced_ptr is None:
            continue
        try:
            produced_ref = weakref.ref(produced)
        except TypeError:
            continue  # non-weakref-able exotic subclass: storage identity not indexed
        # Copy-on-write, prune-dead on append (bounds a reused address to its LIVE aliases,
        # keeping the per-ptr tuple tiny and the worker-side read race-free against an
        # atomic dict-value reassignment).
        live = tuple(ref for ref in captured_ptrs.get(produced_ptr, ()) if ref() is not None)
        captured_ptrs[produced_ptr] = (*live, produced_ref)
