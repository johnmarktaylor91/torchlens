"""Load-time rehydration of a coreless trace into the columnar store (F9).

``_trace_core`` is ``FieldPolicy.DROP``, so every unpickled Trace (plain
pickle AND ``.tlspec`` bundle loads, which route through the same
``__setstate__``) used to arrive as a dict-backed ISLAND: one detached
single-row store per record, no shared edge table, no group sharing —
outside the single-truth architecture entirely.

``rehydrate_trace_core`` runs at the end of ``Trace.__setstate__``: it
adopts every restored ``Op`` into one fresh per-trace ``OpRowStore``, the
non-Op record facades into their kind tables, then replays the standard
relation freeze (dataflow -> edge-occurrence CSR, interned views, shared
``GroupRef`` group tables — whose conversion explicitly coalesces the
equal-but-distinct members a pickle round-trip produces — and fact
pooling) and seals the store exactly like the capture-time freeze seam
(columnar transpose above the row threshold).

Boundaries (documented, tested):

* PARTIAL / FAILED captures are exempt: their staging surface is
  documented as staying raw, so a load must not run the relation freeze
  on them. The ``_tracing_finished`` master switch (``FieldPolicy.KEEP``)
  gates rehydration.
* Module / ModuleCall facades do not exist at ``__setstate__`` time
  (pickle strips ``_module_logs``; ``.tlspec`` loads rebuild it later
  through ``_io.accessor_rebuild``, which adopts them into the kind
  tables at that point). Plain pickle loads legitimately have none.
* Backward records (``GradFn``/``GradFnCall``/``BackwardPass``) stay
  detached-backed — loaded traces have no event stream, so there is no
  epoch to rebuild; a NEW backward on the restored trace binds fresh
  epochs through the normal projection path.
* Rehydration is strictly best-effort AND all-or-nothing: the op set is
  validated BEFORE anything is re-bound (a mixed-ownership or
  layout-drifted op set aborts with zero mutations), adoption copies each
  detached row's cells (the relation freeze mutates adopted rows in
  place, so the detached stores must never share the lists), and any
  unexpected failure later rolls every adopted binding back to its
  detached store — a load never fails because of it, no partial-adoption
  island can survive, and a failed rehydration leaves the detached
  topology byte-identical.
* Compaction passes are not re-run: pickle already preserves shared
  identity within one artifact, so pooled metadata stays pooled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .trace import Trace

#: Record kind table names keyed by facade class name (M8 vocabulary).
_KIND_BY_CLASS_NAME = {
    "Param": "param",
    "Buffer": "buffer",
    "Module": "module",
    "ModuleCall": "module_call",
    "FuncCallLocation": "func_call_location",
}


def _iter_op_cell_func_call_locations(store: Any, layout: Any) -> Any:
    """Yield FuncCallLocation records held in op ``code_context`` cells.

    Pickle empties ``_code_context_cache`` (the live discovery container),
    but the restored op rows still hold the actual ``FuncCallLocation``
    facades in their ``code_context`` cells — the same records the
    capture-time freeze seam adopts from the cache. Scanning the adopted
    rows keeps their kind table populated on BOTH load paths.
    """

    from .func_call_location import FuncCallLocation

    fid = layout.fid_by_name.get("code_context")
    if fid is None:
        return
    seen: set[int] = set()
    for row in range(len(store)):
        cell = store.cell_get(row, fid)
        if isinstance(cell, (tuple, list)):
            for item in cell:
                if isinstance(item, FuncCallLocation) and id(item) not in seen:
                    seen.add(id(item))
                    yield item


def _collect_forward_primitive_rows(
    trace: Trace, records_by_kind: dict[str, dict[int, Any]]
) -> None:
    """Stage the pre-release profile's forward primitive rows for adoption."""

    primitive_profile = trace.__dict__.get("_primitive_op_profile")
    if primitive_profile is None:
        return
    forward_primitive_rows = [
        row for row in primitive_profile.primitive_ops if row.capture_phase != "backward"
    ]
    if forward_primitive_rows:
        records_by_kind["primitive_op"] = {id(record): record for record in forward_primitive_rows}


def _adopt_backward_primitive_rows(trace: Trace, core: Any) -> None:
    """Adopt the profile's backward primitive rows into one backward epoch."""

    primitive_profile = trace.__dict__.get("_primitive_op_profile")
    if primitive_profile is None:
        return
    backward_primitive_rows = [
        row for row in primitive_profile.primitive_ops if row.capture_phase == "backward"
    ]
    if backward_primitive_rows:
        from .._trace_core.record_rows import BackwardEpoch, adopt_rows

        epoch = BackwardEpoch()
        adopt_rows(epoch.stores, "primitive_op", backward_primitive_rows)
        core.backward_epochs = [epoch]


def rehydrate_trace_core(trace: Trace) -> bool:
    """Adopt a coreless trace's detached records into a fresh sealed core.

    Returns ``True`` when a core was built and sealed, ``False`` when the
    trace already has a core, is a partial/failed capture, has no ops, or
    rehydration aborted (the coreless-island behavior is preserved on
    abort, with every op still bound to its own detached store).
    """

    if trace.__dict__.get("_trace_core") is not None:
        return False
    if not trace.__dict__.get("_tracing_finished", False):
        # Partial/failed captures keep their documented staging surface:
        # no relation freeze, no seal, no core.
        return False
    from .._trace_core.core import TraceCore
    from .._trace_core.op_store import DetachedOpStore, OpRowStore
    from .._trace_core.record_rows import CORE_KEY, adopt_records, detach_record
    from .._trace_core.relation_views import freeze_trace_relation_views
    from ._trace_fork import _iter_parent_ops, _iter_parent_records
    from .op import _OP_STORE_LAYOUT, _object_setattr

    ops = list(_iter_parent_ops(trace))
    if not ops:
        return False

    # Validation pre-scan: every op must be a plain detached-backed restore
    # BEFORE anything is re-bound. The former in-loop guards returned early
    # AFTER earlier ops were already adopted, bypassing the rollback and
    # leaving a mixed-ownership island (closure review, blocking item 2).
    adoptable: list[tuple[Any, Any]] = []
    for op in ops:
        try:
            bound = object.__getattribute__(op, "_core")
        except AttributeError:
            return False
        if not isinstance(bound, DetachedOpStore):
            # Mixed/foreign ownership: not a plain coreless load.
            return False
        if bound.layout is not _OP_STORE_LAYOUT:
            return False
        adoptable.append((op, bound))

    store = OpRowStore(_OP_STORE_LAYOUT)
    adopted: list[tuple[Any, Any, int]] = []
    records_by_kind: dict[str, dict[int, Any]] = {}
    try:
        for op, bound in adoptable:
            # Adopt a SNAPSHOT of the detached cells, never the live list:
            # ``adopt_row`` takes ownership, and the relation freeze below
            # mutates adopted rows in place (dataflow ``_CSR`` sentinel,
            # ``GroupRef``/fact-block slot writes). An identity-shared list
            # corrupted the detached stores before the rollback handler
            # could rebind them (closure round 2, blocking item 1).
            row = store.adopt_row(list(bound._cells))
            adopted.append((op, bound, row))
            _object_setattr(op, "_core", store)
            _object_setattr(op, "_row", row)

        core = TraceCore()
        core.ops = store
        for record in _iter_parent_records(trace):
            kind = _KIND_BY_CLASS_NAME.get(type(record).__name__)
            if kind is None:
                continue
            records_by_kind.setdefault(kind, {}).setdefault(id(record), record)
        for record in _iter_op_cell_func_call_locations(store, _OP_STORE_LAYOUT):
            records_by_kind.setdefault("func_call_location", {}).setdefault(id(record), record)
        _collect_forward_primitive_rows(trace, records_by_kind)
        for kind, records in records_by_kind.items():
            adopt_records(core, kind, records.values())
        _adopt_backward_primitive_rows(trace, core)

        trace.__dict__["_trace_core"] = core
        freeze_trace_relation_views(trace)
        store.freeze()
        # Register the loaded core as a payload-lifetime owner, exactly like the
        # capture freeze seam (postprocess/__init__.py) and TraceCore.fork().
        # Without this the owner count stays 0, so forking the loaded trace
        # (which DOES adopt: count -> 1) and then dropping that fork drops the
        # count back to 0 and evicts the SEALED base's tensor payloads --
        # silently NULLing the live loaded parent's saved activations (R37-1).
        store.adopt_payload_owner(core)
        for kind_store in core.kind_rows.values():
            kind_store.freeze()
        for epoch in core.backward_epochs:
            for kind_store in epoch.stores.values():
                kind_store.freeze()
    except Exception:
        # Best-effort: restore the detached bindings and stay coreless.
        for op, bound, _row in adopted:
            _object_setattr(op, "_core", bound)
            _object_setattr(op, "_row", 0)
        for records in records_by_kind.values():
            for record in records.values():
                record_store = record.__dict__.get(CORE_KEY)
                if record_store is not None and not isinstance(record_store, DetachedOpStore):
                    detach_record(record)
        trace.__dict__.pop("_trace_core", None)
        return False
    return True
