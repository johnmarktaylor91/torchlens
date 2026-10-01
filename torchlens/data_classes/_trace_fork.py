"""COW Trace fork (M11): fresh facades over shared frozen storage.

``build_fork`` replaces the object-graph forkcopier: instead of one
structural ``copy.deepcopy`` of the whole record graph, the fork receives

* a forked ``TraceCore`` whose ``OpStoreView``s share the sealed base
  columns and isolate every mutation (fork writes land in per-view
  overlays; reads of mutable containers are isolated copy-on-first-read;
  group cells translate to cloned group tables),
* two-word ``Op`` shells and record-facade shells bound to those views at
  the SAME rows (no field copies at fork time),
* per-record isolated copies only for what genuinely lives outside the
  store: ``Layer`` shadow dicts, record-facade instance extras, and the
  policy-driven Trace-level fields (still one shared-memo deepcopy pass,
  now over the small trace-side remainder rather than the record graph).

Traces without a sealed core-backed op store (loaded analysis traces,
failed partials) take the detached fallback: every record shell binds a
fresh single-row ``DetachedOpStore`` holding isolated copies of its public
state — the same storage class those records already use after pickle
restore.

Cost disclosure (measured 2026-08-14, small linear/BN/relu stack, 83 ops,
gc-tracked-object census with no-op and known-allocation calibration
guards; a prior uncalibrated probe over-counted 20x, and gc UNTRACKS
atomic-only dicts/tuples, so calibrate before trusting any count): the COW
fork is cheap in PAYLOAD BYTES (tensors and sealed columns are shared, and
those dominate real models) but NOT near-free in objects — it retained ~61
gc-tracked objects and ~13.5 KB of new small allocations per op (~68% of a
steady-state capture's tracked-object retention). The bulk is the eager
fork-time isolation of mutable container cells, the per-record shells, and
the Layer shadow dicts — all load-bearing for the bidirectional isolation
contract, so the object count scales O(graph), not O(1).
"""

from __future__ import annotations

import copy
import gc
import weakref
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import fields
from itertools import islice
from typing import TYPE_CHECKING, Any

import torch

from .. import _state
from .._trace_core.op_store import _MISSING, DetachedOpStore, PooledCell, cow_copy_value
from .._trace_core.record_rows import CORE_KEY, ROW_KEY
from ..capture.outcome import stamp_forked
from ..intervention.types import MODEL_LOG_FIELD_FORK_POLICY, ForkFieldPolicy
from ..utils.env_flags import closed_bool_env
from ._accessor_base import Accessor
from ._state_adapter import state_items, state_new, state_restore
from .func_call_location import FuncCallLocation
from .layer import OpAccessor
from .op import _OP_STORE_LAYOUT, Op

if TYPE_CHECKING:
    from .trace import Trace

# Session-only backward projection guard/fold fields derived from the event
# stream. Any operation that replaces a trace's stream (pickle restore, fork)
# must drop these so the next projection access materializes from the new
# stream instead of trusting a guard computed over the old one.
_STREAM_DERIVED_GUARD_FIELDS = frozenset(
    {
        "_backward_projection_event_count",
        "_backward_projection_revision",
        "_backward_projection_fold_state",
        "_tl_materializing_backward_projection",
    }
)

#: Op session-cache fields never carried across a fork (lazily recomputed).
_OP_SESSION_CACHE_FIELDS = (
    "_facets_cache",
    "_receptive_field_cache",
    "_projective_field_cache",
)

# Instance-memoized Trace accessor caches (the R37 GC fix moved them onto the
# instance). Each cached value holds SOURCE-trace record facades — and the
# op/layer entries are tuple-wrapped, which the conservative default policy
# would SHARE by reference, so a fork's `.ops`/`.layers` would hand out the
# PARENT's facades and every COW isolation guarantee would silently vanish.
# Forks drop them and rebuild lazily over their own shells on first access.
_TRACE_ACCESSOR_CACHE_FIELDS = frozenset(
    {
        "_op_accessor_cache",
        "_layer_accessor_cache",
        "_module_call_accessor",
    }
)

_OP_FID_BY_NAME = _OP_STORE_LAYOUT.fid_by_name
_SOURCE_TRACE_REF_FID = _OP_FID_BY_NAME["_source_trace_ref"]
_OP_SESSION_CACHE_FIDS = tuple(
    _OP_FID_BY_NAME[name] for name in _OP_SESSION_CACHE_FIELDS if name in _OP_FID_BY_NAME
)

_OBJECT_SETATTR = object.__setattr__
_OBJECT_GETATTRIBUTE = object.__getattribute__

_PROPAGATE = object()


class _ForkMemo(dict):
    """The single ``copy.deepcopy`` memo shared by every field copy of ONE fork.

    Rollback rides the dict's own insertion order: nothing ever deletes memo
    entries except ``rollback`` itself (which only pops the tail), so the keys
    inserted since a ``mark`` are exactly the keys past the marked length. A
    field copy that raises partway through can roll its partially built
    entries back out — a half-constructed object must never survive in the
    memo to be reused by a later field.
    """

    __slots__ = ()

    def mark(self) -> int:
        """Return a rollback token for the current memo contents."""

        return len(self)

    def rollback(self, mark: int) -> None:
        """Drop every entry inserted since ``mark``.

        The record-shell identity seeds are permanent: they are installed
        before the first field copy takes a mark, so their positions always
        precede every rollback token and no rollback can remove them.
        """

        for key in list(islice(self.keys(), mark, None)):
            del self[key]


def _memoized_deep_copy(
    value: Any,
    memo: dict[Any, Any] | None,
    *,
    on_failure: Any,
    fallback: Any = _PROPAGATE,
) -> Any:
    """Deep-copy ``value`` under the shared fork memo, rolling back on failure.

    The degradation to ``on_failure`` is intentional for opaque field values,
    but it also silences genuine copy bugs, so the opt-in
    ``TORCHLENS_DEBUG_FORK_COPY`` channel re-raises instead.
    """

    if memo is None:
        memo = _ForkMemo()
    fork_memo = memo if isinstance(memo, _ForkMemo) else None
    mark = fork_memo.mark() if fork_memo is not None else None
    try:
        return copy.deepcopy(value, memo)
    except Exception:
        # A raised deepcopy can leave half-populated objects behind in the
        # memo (``copy._reconstruct`` memoizes before it restores state), so
        # those entries are discarded rather than handed to the next field.
        if fork_memo is not None and mark is not None:
            fork_memo.rollback(mark)
        # Closed parse (round-7 b7 R47): the raw-truthiness spelling made
        # ``TORCHLENS_DEBUG_FORK_COPY=0`` ENABLE the debug re-raise it names
        # off, and a typo silently selected a state.
        if closed_bool_env("TORCHLENS_DEBUG_FORK_COPY"):
            raise
        if fallback is _PROPAGATE:
            return on_failure(value)
        try:
            return on_failure(value)
        except Exception:
            return fallback


class _RecordTranslator:
    """Map parent record facades to their fork shells by identity.

    Installed as ``OpStoreView.record_translator``: consulted on every COW
    cell read so record references inside stored containers and hydrated
    fact blocks resolve to the FORK's facades. ``Accessor`` instances stored
    in cells (``Module.ops``/``Module.params``) are rebuilt lazily with their
    members translated, memoized by identity. The keepalive list pins every
    keyed parent object so a recycled ``id`` can never mistranslate.
    """

    __slots__ = (
        "__weakref__",
        "_fork_ref",
        "_guards",
        "_keepalive",
        "_miss_classes",
        "_parent_cls",
        "_parent_ref",
        "map",
    )

    def __init__(self, fork_ref: Any = None) -> None:
        """Create an empty translator owned by the fork behind ``fork_ref``."""

        self.map: dict[int, Any] = {}
        # Classes whose instances can never translate (tensors, dtypes,
        # plain metadata): one set lookup short-circuits the miss path. Safe
        # because the id-map lookup runs FIRST (record facades hit there even
        # if their class is cached) and the parent Trace's class is excluded
        # (parent translation is identity-based, not class-based).
        self._miss_classes: set[type] = set()
        self._parent_cls: type | None = None
        # id-recycling protection WITHOUT pinning the parent graph (records
        # like ModuleCall hold a STRONG trace reference, so a blanket
        # keepalive would keep the parent Trace alive for the fork's whole
        # lifetime — a GC parity break). Weak-referenceable parents get a
        # weakref guard validated on lookup; only non-weakref-able parents
        # (Op facades, which hold no strong trace reference) are pinned.
        self._guards: dict[int, Any] = {}
        self._keepalive: list[Any] = []
        self._fork_ref = fork_ref
        # Weak parent-trace reference: a bare STRONG parent-trace value in
        # copied state (ModuleCall._source_trace_strong) translates to the
        # fork WITHOUT the translator pinning the parent (GC parity).
        self._parent_ref: Any = None

    def add(self, parent_record: Any, shell: Any) -> None:
        """Register one parent record -> fork shell mapping."""

        key = id(parent_record)
        self.map[key] = shell
        try:
            self._guards[key] = weakref.ref(parent_record)
        except TypeError:
            self._guards[key] = None
            self._keepalive.append(parent_record)

    def _rebuild_accessor(self, accessor: Any) -> Any:
        """Rebuild one cell-stored accessor with translated members."""

        accessor_cls: Any = type(accessor)
        clone = accessor_cls.__new__(accessor_cls)
        clone.__dict__.update(accessor.__dict__)
        remap = self.map
        clone._dict = {key: remap.get(id(item), item) for key, item in accessor._dict.items()}
        clone._list = [remap.get(id(item), item) for item in accessor._list]
        if "_source_ref" in clone.__dict__ and self._fork_ref is not None:
            clone._source_ref = self._fork_ref
        self.add(accessor, clone)
        return clone

    def __call__(self, value: Any) -> Any:
        """Return the fork-side object for ``value``, or ``None``."""

        key = id(value)
        mapped = self.map.get(key)
        if mapped is not None:
            guard = self._guards.get(key)
            if guard is None or guard() is value:
                return mapped
            # The original parent object died and its id was recycled by
            # ``value``: drop the stale entry and fall through.
            del self.map[key]
            del self._guards[key]
        cls = value.__class__
        if cls in self._miss_classes:
            return None
        parent_ref = self._parent_ref
        if parent_ref is not None and value is parent_ref():
            return self._fork_ref() if self._fork_ref is not None else None
        if isinstance(value, Accessor):
            return self._rebuild_accessor(value)
        if cls is not self._parent_cls:
            # ``isinstance`` returned False, so no instance of ``cls`` can
            # ever rebuild as an accessor; only the parent's own class must
            # keep taking the identity check above.
            self._miss_classes.add(cls)
        return None


def _copy_fork_value(value: Any, memo: dict[Any, Any] | None = None) -> Any:
    """Copy a fork field while preserving tensor and callable identity."""

    if isinstance(value, torch.Tensor) or callable(value):
        return value
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    return _memoized_deep_copy(value, memo, on_failure=copy.copy)


def _default_fork_policy(value: Any) -> ForkFieldPolicy:
    """Choose a conservative fork policy for fields outside policy tables."""

    if isinstance(value, (str, bytes, int, float, bool, type(None), tuple)):
        return ForkFieldPolicy.FORK_SHARE
    if isinstance(value, torch.Tensor) or callable(value):
        return ForkFieldPolicy.FORK_SHARE
    return ForkFieldPolicy.FORK_COPY


def _fork_model_field(parent: Trace, field_name: str, value: Any, memo: dict[Any, Any]) -> Any:
    """Apply the Trace fork policy to a single field."""

    if field_name == "_runnable":
        # Only the three immutable state bindings are shared; every other
        # runnable value follows the ordinary fork-copy path. In particular,
        # verdict-steering witness dictionaries must never alias the parent.
        shared_fields = {"staged_user_state", "embedded_state", "capture_state"}
        forked = copy.copy(value)
        for runnable_field in fields(value):
            runnable_value = getattr(value, runnable_field.name)
            if runnable_field.name not in shared_fields:
                runnable_value = _copy_fork_value(runnable_value, memo)
            setattr(forked, runnable_field.name, runnable_value)
        return forked
    if field_name in ("capture_events", "_capture_events"):
        # Event streams never fork by copy: ``build_fork`` installs ONE fresh
        # detached stream after the field pass (a shared stream would let a
        # fork backward corrupt the parent's projection).
        return None
    if field_name in _STREAM_DERIVED_GUARD_FIELDS:
        # Derived from the stream that was just replaced; carrying the
        # parent's values would make the fork's guard silently skip its
        # first materialize. ``build_fork`` pops these after restore.
        return None
    if field_name in _TRACE_ACCESSOR_CACHE_FIELDS:
        # Memoized accessors hold parent facades (tuple-wrapped, so the
        # default policy would share them); the fork rebuilds lazily.
        # ``build_fork`` pops these after restore.
        return None
    if field_name == "annotations":
        # A plain fork is NOT a fresh execution: its records still hold the
        # capture's own values, so every annotations sub-key (episode ledger,
        # observer values) travels here verbatim. The per-sub-key travel
        # policy (capture/_annotations_travel.py, foldA D6) fires at the
        # provider settlement finalizer the moment an engine RE-EXECUTES on
        # a fork; the fork builder never decides per key.
        return _copy_fork_value(value, memo)
    if field_name == "_annotation_blobs":
        # Render-time annotation payloads (feature maps / RDM / MDS / scree)
        # are derived from the CAPTURED activations of one specific run. A
        # fork is a mutation target: carrying them lets draw-time hooks
        # render the parent's activations (and old-batch overlays) as if
        # they were the fork's. Re-derive with the evolution helpers.
        return None
    policy = MODEL_LOG_FIELD_FORK_POLICY.get(field_name)
    if policy is None:
        policy = _default_fork_policy(value)
    if policy is ForkFieldPolicy.FORK_SHARE:
        return value
    if policy is ForkFieldPolicy.FORK_RECONSTRUCT:
        return None
    return _copy_fork_value(value, memo)


def _iter_parent_ops(parent: Trace) -> Iterator[Op]:
    """Yield every distinct Op reachable from the parent's lookup containers."""

    seen: set[int] = set()
    containers: list[Any] = [parent.__dict__.get("layer_list") or ()]
    for name in ("layer_dict_main_keys", "layer_dict_all_keys"):
        mapping = parent.__dict__.get(name)
        if mapping:
            containers.append(mapping.values())
    for layer in (parent.__dict__.get("layer_logs") or {}).values():
        ops = layer.__dict__.get("ops")
        if ops is not None:
            containers.append(ops.values())
    for container in containers:
        for op in container:
            if id(op) not in seen and isinstance(op, Op):
                seen.add(id(op))
                yield op


def _iter_parent_records(parent: Trace) -> Iterator[Any]:
    """Yield the parent's non-Op record facades (modules/calls/params/...).

    Enumerates from the owning containers (never the kind tables), so the
    fork maps exactly the objects the parent hands out; ``Layer`` shells are
    seeded separately because their state lives trace-side.
    """

    module_accessor = parent.__dict__.get("_module_logs")
    if module_accessor is not None:
        for module in module_accessor.values():
            yield module
            calls = getattr(module, "calls", None)
            if calls:
                yield from calls.values()
    param_accessor = parent.__dict__.get("param_logs")
    if param_accessor is not None:
        yield from param_accessor.values()
    buffer_accessor = parent.__dict__.get("_buffer_accessor")
    if buffer_accessor is not None:
        yield from buffer_accessor.values()
    code_context_cache = parent.__dict__.get("_code_context_cache")
    if code_context_cache:
        seen: set[int] = set()
        for cached in code_context_cache.values():
            if isinstance(cached, (tuple, list)):
                for item in cached:
                    if isinstance(item, FuncCallLocation) and id(item) not in seen:
                        seen.add(id(item))
                        yield item


def _fill_detached_op(
    parent_op: Op, shell: Op, translator: _RecordTranslator, fork_ref: Any
) -> None:
    """Fill one detached-fallback Op shell with isolated public state."""

    state: dict[str, Any] = {}
    for field_name, value in parent_op.__tl_state_items__():
        if field_name in _OP_SESSION_CACHE_FIELDS or field_name == "_source_trace_ref":
            continue
        state[field_name] = cow_copy_value(value, translator)
    shell.__tl_state_restore__(state)
    _OBJECT_SETATTR(shell, "_source_trace_ref", fork_ref)


def _fill_detached_record(parent_record: Any, shell: Any, translator: _RecordTranslator) -> None:
    """Bind one detached-fallback record shell to an isolated row copy."""

    parent_dict = parent_record.__dict__
    store = parent_dict[CORE_KEY]
    row = parent_dict[ROW_KEY]
    layout = type(parent_record)._TL_LAYOUT
    detached = DetachedOpStore(layout)
    for fid in range(layout.n_fields):
        value = store.cell_get(row, fid)
        # Decode BOTH internal cell encodings (M14) while copying: a detached
        # shell has no pool/registry to decode them later, and
        # ``DetachedOpStore.items`` streams cells verbatim — a leaked
        # ``PooledCell`` would poison the record's pickle state and make every
        # fork serialization (pickle AND tl.save/tl.load) fail downstream.
        if value.__class__ is PooledCell:
            value = value.hydrate()
        elif value.__class__ is str and store.compacted_singleton(row, fid, value):
            value = [value]
        detached.cell_set(0, fid, cow_copy_value(value, translator))
    shell.__dict__[CORE_KEY] = detached
    shell.__dict__[ROW_KEY] = 0


#: Cell-stored owning-Trace weakref fields per record kind (``Param`` uses
#: ``_source_trace_ref``, ``Buffer`` uses ``_source_ref``).
_RECORD_TRACE_REF_FIELDS = ("_source_trace_ref", "_source_ref")


def _copy_record_extras(
    parent_record: Any, shell: Any, translator: _RecordTranslator, fork_ref: Any
) -> None:
    """Copy a record facade's instance extras onto its fork shell.

    Declared stored names are excluded: a ``__dict__`` entry shadowing a
    declared field is unreachable dead weight (the cell descriptor wins),
    and copying it would propagate the inert shadow into every fork.
    """

    declared = getattr(type(parent_record), "_TL_LAYOUT", None)
    declared_names = declared.fid_by_name if declared is not None else ()
    for key, value in parent_record.__dict__.items():
        if key in (CORE_KEY, ROW_KEY) or key == "_facets_cache":
            continue
        if key in declared_names:
            continue
        if key == "_source_trace_ref":
            shell.__dict__[key] = fork_ref
            continue
        shell.__dict__[key] = cow_copy_value(value, translator)


def _rebind_record_trace_refs(shell: Any, fork_ref: Any) -> None:
    """Rebind cell-stored owning-Trace weakrefs on one fork record shell.

    The deepcopy forkcopier severed these (fork params/buffers came back
    with dead owner refs); the COW fork rebinds a PRESENT non-``None`` ref
    to the fork, keeping absent/``None`` cells untouched.
    """

    layout = getattr(type(shell), "_TL_LAYOUT", None)
    store = shell.__dict__.get(CORE_KEY)
    if layout is None or store is None:
        return
    row = shell.__dict__[ROW_KEY]
    for ref_name in _RECORD_TRACE_REF_FIELDS:
        fid = layout.fid_by_name.get(ref_name)
        if fid is None:
            continue
        current = store.cell_get(row, fid)
        if current is not _MISSING and current is not None:
            store.cell_set(row, fid, fork_ref)


def _fill_layer_shells(
    parent: Trace,
    fork: Trace,
    layer_shells: dict[int, Any],
    translator: _RecordTranslator,
    fork_ref: Any,
) -> None:
    """Fill the fork's Layer shells from the parent's per-layer shadow dicts.

    Only the per-layer ``__dict__`` is copied (shadows, tombstones, staging
    snapshots); the M8 mirror fields keep reading through ``ops[0]``, which on
    the fork resolves through the COW view — so the fork preserves the mirror
    memory win the deepcopy fork used to lose.
    """

    fork_layer_logs: OrderedDict[str, Any] = OrderedDict()
    consumed_shells: set[int] = set()
    fork_equivalent_ops = fork.op_equivalence_classes
    # One normalized relation view per equivalence class: the finished-Layer
    # write normalization (F5) converts each assigned staging set to an
    # immutable view, and materializing a fresh one PER MEMBER layer is
    # quadratic on single-class traces (32 MB/fork measured on a 2002-op
    # stack). Equal views are documented as shareable across records.
    equivalent_view_memo: dict[Any, Any] = {}
    remap = translator.map
    for label, parent_layer in (parent.__dict__.get("layer_logs") or {}).items():
        parent_layer_id = id(parent_layer)
        if parent_layer_id in consumed_shells:
            # A parent Layer registered under two labels must become two fork
            # Layers (one shell is consumed at most once).
            shell = state_new(type(parent_layer))
        else:
            shell = layer_shells[parent_layer_id]
            consumed_shells.add(parent_layer_id)
        shell_dict = shell.__dict__
        for key, value in parent_layer.__dict__.items():
            if key == "ops" or key == "_source_trace_ref" or key == "_facets_cache":
                continue
            shell_dict[key] = cow_copy_value(value, translator)
        shell.source_trace = fork
        parent_ops = parent_layer.__dict__.get("ops")
        if parent_ops is not None:
            shell.ops = OpAccessor(
                OrderedDict(
                    (call_index, remap.get(id(layer_pass), layer_pass))
                    for call_index, layer_pass in parent_ops.items()
                )
            )
        equivalence_class = getattr(shell, "equivalence_class", None)
        if equivalence_class in fork_equivalent_ops:
            view = equivalent_view_memo.get(equivalence_class)
            if view is None:
                shell.equivalent_ops = fork_equivalent_ops[equivalence_class]
                view = shell.__dict__.get("equivalent_ops", fork_equivalent_ops[equivalence_class])
                equivalent_view_memo[equivalence_class] = view
            else:
                shell.equivalent_ops = view
        fork_layer_logs[label] = shell
    fork.layer_logs = fork_layer_logs


def build_fork(parent: Trace, *, name: str | None) -> Trace:
    """Build a copy-on-write fork of ``parent``.

    Parameters
    ----------
    parent:
        Finished source Trace.
    name:
        Optional fork name (defaults to the deterministic fork counter).

    Returns
    -------
    Trace
        Fork whose record graph shares the parent's frozen storage and whose
        every mutation surface is isolated.
    """

    from ..ir.capture_events import CaptureEvents

    core = parent.__dict__.get("_trace_core")
    fork_core = (
        core.fork() if core is not None and core.ops is not None and core.ops.frozen else None
    )
    if fork_core is not None:
        # The module kind's cells embed accessor objects (``Module.ops``/
        # ``Module.params``) whose members hold STRONG trace references
        # (``ModuleCall._source_trace_strong``); a view over that table
        # would keep the parent Trace reachable for the fork's lifetime — a
        # GC parity break vs the deepcopy fork. Modules are few, so they
        # fork as detached duplicates with translated cells instead.
        fork_core.kind_rows.pop("module", None)
    fork = state_new(type(parent))
    fork_ref = weakref.ref(fork)
    memo = _ForkMemo()
    dict.update(memo, {id(parent): fork})
    translator = _RecordTranslator(fork_ref)
    translator._parent_ref = weakref.ref(parent)
    translator._parent_cls = type(parent)

    ops_view = fork_core.ops if fork_core is not None else None
    base_store = core.ops if core is not None else None
    # Parent store -> fork view, keyed by the PARENT'S store objects (which
    # are themselves views when the parent is a fork), so fork chains bind
    # COW shells instead of falling back to detached duplication.
    store_view_map: dict[int, Any] = {}
    if fork_core is not None and core is not None:
        for kind, parent_store in core.kind_rows.items():
            fork_view = fork_core.kind_rows.get(kind)
            if fork_view is not None:
                store_view_map[id(parent_store)] = fork_view

    # Phase 1: shells. Ops bound to the sealed base store become two-word
    # COW shells; everything else (detached-backed records, and every record
    # of a coreless trace) falls back to detached single-row duplication.
    cow_ops: list[tuple[Op, Op]] = []
    detached_ops: list[tuple[Op, Op]] = []
    for parent_op in _iter_parent_ops(parent):
        shell = state_new(Op)
        try:
            op_store = _OBJECT_GETATTRIBUTE(parent_op, "_core")
        except AttributeError:
            op_store = None
        if ops_view is not None and op_store is base_store:
            _OBJECT_SETATTR(shell, "_core", ops_view)
            _OBJECT_SETATTR(shell, "_row", _OBJECT_GETATTRIBUTE(parent_op, "_row"))
            cow_ops.append((parent_op, shell))
        else:
            detached_ops.append((parent_op, shell))
        translator.add(parent_op, shell)

    layer_shells: dict[int, Any] = {}
    for parent_layer in (parent.__dict__.get("layer_logs") or {}).values():
        if id(parent_layer) not in layer_shells:
            layer_shells[id(parent_layer)] = state_new(type(parent_layer))

    record_shells: list[tuple[Any, Any]] = []
    detached_records: list[tuple[Any, Any]] = []
    for parent_record in _iter_parent_records(parent):
        if id(parent_record) in translator.map:
            continue
        shell = state_new(type(parent_record))
        record_store = parent_record.__dict__.get(CORE_KEY)
        view = store_view_map.get(id(record_store)) if record_store is not None else None
        if view is not None:
            shell.__dict__[CORE_KEY] = view
            shell.__dict__[ROW_KEY] = parent_record.__dict__[ROW_KEY]
        elif record_store is not None:
            detached_records.append((parent_record, shell))
        record_shells.append((parent_record, shell))
        translator.add(parent_record, shell)

    dict.update(memo, translator.map)
    dict.update(memo, layer_shells)
    if fork_core is not None:
        # The views hold the translator WEAKLY (a retained fork Op reaching
        # the fork Trace through translator-mapped shells would root the
        # whole fork graph and disarm last-owner payload eviction); the fork
        # core is its strong anchor, so translation lives exactly as long as
        # the fork itself.
        fork_core._record_translator = translator
        for view in fork_core.store_views():
            view.record_translator = translator

    # Phase 2: the policy-driven Trace field pass (one shared-memo deepcopy
    # over the trace-side remainder; every record reference resolves to its
    # shell through the memo).
    fork_state = {
        field_name: _fork_model_field(parent, field_name, value, memo)
        for field_name, value in state_items(parent)
    }
    state_restore(fork, fork_state)
    for stale_guard_field in _STREAM_DERIVED_GUARD_FIELDS:
        fork.__dict__.pop(stale_guard_field, None)
    for accessor_cache_field in _TRACE_ACCESSOR_CACHE_FIELDS:
        fork.__dict__.pop(accessor_cache_field, None)
    # The fork gets exactly ONE fresh DETACHED stream — never the parent's
    # (shared lists let a fork backward corrupt the parent's projection),
    # and never none (a fork remains a supported backward-capture target).
    fork.__dict__.pop("capture_events", None)
    fork.__dict__["_capture_events"] = CaptureEvents.detached_from(parent)
    if fork_core is not None:
        fork.__dict__["_trace_core"] = fork_core
    fork.parent_run = weakref.ref(parent)
    fork.trace_label = name or parent._next_fork_name()
    fork._intervention_spec = copy.deepcopy(parent._ensure_intervention_spec(), memo)
    fork.state_history = copy.deepcopy(parent.state_history, memo)
    fork.relationship_evidence = copy.deepcopy(parent.relationship_evidence, memo)
    fork._out_recipe_revision = parent._out_recipe_revision
    fork._spec_revision = parent._spec_revision
    fork.state = parent.state
    fork._warned_mutate_in_place = False
    fork._warned_direct_write = False
    fork.__dict__.pop("_validation_replay_status", None)
    # The fork is the sanctioned mutation surface, so it settles a DERIVED
    # outcome (UNATTESTED for a complete parent) instead of inheriting the
    # parent's blessed attestation by identity -- a hand-edited fork must
    # never save as a bit-identical attested COMPLETE.
    stamp_forked(fork, parent)

    # Phase 3: fill. COW ops rebind their owner weakref and drop session
    # caches in the fork view; detached fallbacks duplicate isolated state.
    if ops_view is not None:
        for _parent_op, shell in cow_ops:
            row = _OBJECT_GETATTRIBUTE(shell, "_row")
            ops_view.cell_set(row, _SOURCE_TRACE_REF_FID, fork_ref)
            for fid in _OP_SESSION_CACHE_FIDS:
                ops_view.cell_del(row, fid)
    for parent_op, shell in detached_ops:
        _fill_detached_op(parent_op, shell, translator, fork_ref)
    for parent_record, shell in detached_records:
        _fill_detached_record(parent_record, shell, translator)
    for parent_record, shell in record_shells:
        _copy_record_extras(parent_record, shell, translator, fork_ref)
        _rebind_record_trace_refs(shell, fork_ref)
    _fill_layer_shells(parent, fork, layer_shells, translator, fork_ref)
    # Fork-time snapshot: eagerly isolate every mutable-container cell so a
    # PARENT's in-place mutation after the fork can never leak into the fork
    # through the copy-on-first-read window (both isolation directions now
    # hold at fork time, matching the deepcopy fork's snapshot semantics).
    # The sweep is sparse (the base's cached mutable-cell index plus the
    # overlay snapshot) but still allocates tens of thousands of small
    # containers on large traces; pausing collection for the burst avoids
    # redundant young-gen scans (fork build allocates, never frees).
    if fork_core is not None:
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        try:
            for view in fork_core.store_views():
                view.isolate_mutable_cells()
        finally:
            if gc_was_enabled:
                gc.enable()
    fork._rebind_fork_owner_refs()
    _state.register_log(fork)
    return fork
