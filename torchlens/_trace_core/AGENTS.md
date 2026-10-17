# _trace_core/ - Implementation Guide

Private per-trace columnar store substrate. Nothing in this package is public API; public
classes (`Trace`, `Op`, ...) present rows from here as facades. The architecture of record is
`docs/reference/trace_core_design.md` - read it first; this file only names entry points.

## core.py
- `TraceCore` is the one per-trace store: the op store, per-kind row tables (`kind_rows`),
  the canonical label-to-row map (`label_rows`), and the backward epoch list
  (`backward_epochs`).
- `freeze()` seals the core; `transaction()` returns a `Transaction` checkpointing every
  mutation surface atomically. `KindTable` is the generic per-kind row table.

## op_store.py
- `OpRowStore` holds op rows, described by `OpStoreLayout`. Rows are row-major while
  building; `freeze()` class-swaps in `_SealedRowMajorOpRowStore` (the sealed row-major
  variant used under the columnar-transpose threshold).
- `DetachedOpStore` backs single-row detached records (`Op.copy()`, pickle restore, preview
  backends); `OpStoreView` is the per-fork copy-on-write view used by `Trace.fork()`.
- Audit instrumentation for the postprocess step contracts: `begin_cell_write_audit()` /
  `end_cell_write_audit()` (class-swap to `_AuditedOpRowStore` / `_CombinedAuditOpRowStore`,
  results as `StepAuditResult`).
- `mark_op_row_released()`, `row_clone_scope()`, and `cow_copy_value()` support removal scrub
  and fork copying.

## Relations and views
- `relations.py`: `EdgeTable` is the canonical dataflow edge-occurrence table (`Edge` rows,
  CSR indexes).
- `relation_views.py`: freeze-time conversion to immutable relation views -
  `freeze_op_relation_views()`, `freeze_trace_relation_views()`, and
  `materialize_dataflow_view()` (lazy `parents`/`children` rematerialization from the edge
  table); `tuple_view()` / `frozenset_view()` intern shared views.
- `groups.py`: `MembershipGroups` + `GroupRef` give `equivalent_ops`/`recurrent_ops` one
  shared cached view per membership group.
- `fact_blocks.py`: `FactBlocks` / `FactFamily` store call-level container facts once per
  group; member cells hold the `_FACT` sentinel and `convert_fact_cells()` installs them.

## Storage primitives
- `columns.py`: `ColumnBuilder` builds typed `FrozenColumn`s.
- `pools.py`: `InternPool` and `ClosurePool` per-trace interning.
- `payloads.py`: `PayloadArena` identity-preserving payload storage.
- `overlays.py`: `RowOverlay` sparse versioned mutation overlay; `Transaction`
  snapshot/rollback.

## record_rows.py
- Non-op record kinds as row facades: `install_record_facade()` installs `RecordCellField`
  descriptors on a record class; `adopt_records()` / `adopt_rows()` move built records into
  kind tables; `detach_record()` re-detaches one.
- `BackwardEpoch` is the atomic per-backward-projection row-store binding kept on
  `TraceCore.backward_epochs`.
- `record_state_items()` / `record_state_restore()` handle facade pickle state.

## Local invariants
- Substrate-only imports: modules here import each other and numpy/stdlib, never
  `data_classes/` (facade classes reach down, not the reverse).
- Sealing does NOT make rows read-only: post-seal `cell_set` writes still mutate rows in
  place but must register in the fork-isolation index so later forks' eager sweeps see
  containers written after the index was built (see `_SealedRowMajorOpRowStore`).
- The label-to-row binding (`TraceCore.label_rows`) is established at the freeze; do not
  read it from unfrozen cores.
- Dedicated unit suite: `tests/test_trace_core_substrate.py` (per the package docstring).
- The payload eviction walk (`_evict_payloads_in_cell` and its list, mapping, tuple and set
  helpers) dispatches on exact type first; its result must stay equal to the plain
  `isinstance` chain it replaced, which `tests/test_perf7_eviction_walk.py` keeps as the
  reference. Fork reads (`OpStoreView.cell_get`, `cow_copy_value`) return atomic values
  without the isolation call; `tests/test_perf7_fork_reads.py` keeps the former code as the
  reference.
