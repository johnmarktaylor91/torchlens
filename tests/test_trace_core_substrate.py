"""M4: standalone unit suite for the _trace_core substrate.

Zero production consumers yet — these tests prove the substrate honors the
converged semantic contracts (docs/internal/reference/trace_core_design.md section
3.2/3.5) before the M5 Op seam, including executable prototypes of every
hard seam: pool injectivity, parallel-edge order, payload identity, overlay
COW + transactional rollback, core-level fork isolation, and facade
identity caching.
"""

from __future__ import annotations

import gc

import pytest
import torch

from torchlens._data_classes.op import Bytes
from torchlens._trace_core import (
    ColumnBuilder,
    EdgeTable,
    InternPool,
    PayloadArena,
    RowOverlay,
    TraceCore,
    Transaction,
)
from torchlens._trace_core.pools import ClosurePool


@pytest.mark.smoke
def test_column_build_freeze_round_trip() -> None:
    """Values and explicit missings survive build -> freeze byte-exactly."""

    column = ColumnBuilder("int64")
    column.append(7)
    column.append(None)
    column.append(-3)
    column.set(0, 9)
    frozen = column.freeze()
    assert list(frozen) == [9, None, -3]
    assert column.get(1) is None
    with pytest.raises(RuntimeError):
        column.set(0, 1)

    objects = ColumnBuilder("object")
    objects.append("label")
    objects.append(None)
    frozen_objects = objects.freeze()
    assert list(frozen_objects) == ["label", None]


@pytest.mark.smoke
def test_pool_exact_type_injectivity() -> None:
    """True / 1 / 1.0 / Bytes(1) never collapse (shipped _pool_key contract)."""

    pool = InternPool()
    ids = {pool.intern(value) for value in (True, 1, 1.0, Bytes(1))}
    assert len(ids) == 4
    assert pool.intern(True) == pool.intern(True)
    assert pool.value(pool.intern("x")) == "x"


@pytest.mark.smoke
def test_closure_pool_shares_and_materializes_fresh() -> None:
    """One entry per distinct closure; materialization is a fresh set."""

    closures = ClosurePool()
    first = closures.intern(["a", "b"])
    second = closures.intern({"b", "a"})
    assert first == second
    assert len(closures) == 1
    materialized = closures.materialize(first)
    materialized.add("c")
    assert closures.materialize(first) == {"a", "b"}


@pytest.mark.smoke
def test_edge_table_parallel_edges_and_order() -> None:
    """Parallel edges stay distinct; CSR preserves insertion order."""

    edges = EdgeTable()
    edges.add(0, 1, arg_position=0)
    edges.add(0, 1, arg_position=1)  # parallel edge, second arg slot
    edges.add(0, 2, arg_position=2)
    edges.add(3, 1)
    edges.freeze(n_source_rows=4, n_target_rows=4)

    out = list(edges.out_edges(0))
    assert [(e.target, e.arg_position) for e in out] == [(1, 0), (1, 1), (2, 2)]
    assert edges.sources_of(1) == [0, 0, 3]
    with pytest.raises(RuntimeError):
        EdgeTable().out_edges(0).__next__()


@pytest.mark.smoke
def test_payload_arena_identity_contract() -> None:
    """Same object = same handle; equal-but-distinct stays distinct;
    in-place mutation visible; replacement never disturbs aliases."""

    arena = PayloadArena()
    tensor = torch.zeros(3)
    equal_twin = torch.zeros(3)
    handle = arena.register(tensor)
    assert arena.register(tensor) == handle
    assert arena.register(equal_twin) != handle
    assert arena.value(handle) is tensor

    tensor.add_(1.0)  # in-place mutation stays visible through the handle
    assert torch.equal(arena.value(handle), torch.ones(3))

    replacement = arena.replace(handle, torch.full((3,), 5.0))
    assert replacement != handle
    assert arena.value(handle) is tensor  # original alias undisturbed


@pytest.mark.smoke
def test_overlay_and_transaction_rollback() -> None:
    """Overlay reads shadow the base; rollback restores atomically."""

    overlay = RowOverlay()
    overlay.write(0, "op.out", "patched")
    assert overlay.read(0, "op.out") == "patched"

    txn = Transaction({"main": overlay})
    overlay.write(0, "op.out", "mutated")
    overlay.write(1, "op.grad", "extra")
    txn.rollback()
    assert overlay.read(0, "op.out") == "patched"
    assert not overlay.has(1, "op.grad")


@pytest.mark.smoke
def test_core_write_routing_and_freeze() -> None:
    """Writes hit the base while building, the overlay after freeze."""

    core = TraceCore()
    table = core.table("op")
    row = table.new_row()
    core.write("op", row, "func_name", "relu")
    assert core.read("op", row, "func_name") == "relu"

    core.freeze()
    core.write("op", row, "func_name", "patched")
    assert core.read("op", row, "func_name") == "patched"
    assert table.get(row, "func_name") == "relu"  # base untouched


@pytest.mark.smoke
def test_core_fork_isolation_prototype() -> None:
    """COW fork: shared frozen base, isolated overlays both directions."""

    core = TraceCore()
    row = core.table("op").new_row()
    core.write("op", row, "func_name", "relu")
    core.freeze()
    core.write("op", row, "annotations", "parent-note")

    fork = core.fork()
    assert fork.read("op", row, "func_name") == "relu"
    assert fork.read("op", row, "annotations") == "parent-note"

    fork.write("op", row, "func_name", "fork-patch")
    assert core.read("op", row, "func_name") == "relu"
    core.write("op", row, "func_name", "parent-patch")
    assert fork.read("op", row, "func_name") == "fork-patch"
    assert fork.tables is core.tables  # base is shared, not copied


@pytest.mark.smoke
def test_core_collectable_after_drop() -> None:
    """A dropped core (with facades and payloads) is garbage-collected."""

    core = TraceCore()
    core.table("op").new_row()
    core.payloads.register(torch.zeros(2))
    core.set_facade_factory(lambda kind, r: object())
    core.facade("op", 0)
    ref = core.weak_self()
    del core
    gc.collect()
    assert ref() is None


@pytest.mark.smoke
def test_op_store_view_cow_isolation() -> None:
    """M11 store view: bidirectional write isolation + container copy-on-read."""

    from torchlens._trace_core.op_store import (
        _MISSING,
        OpRowStore,
        OpStoreLayout,
        OpStoreView,
    )

    layout = OpStoreLayout(("alpha", "beta", "items"))
    store = OpRowStore(layout)
    row = store.new_row()
    store.cell_set(row, 0, "base")
    store.cell_set(row, 2, {"k": [1, 2]})
    store.freeze()

    view = OpStoreView(store)
    view.cell_set(row, 0, "fork")
    assert store.cell_get(row, 0) == "base"
    store.cell_set(row, 0, "parent-after")
    assert view.cell_get(row, 0) == "fork"
    assert store.cell_get(row, 0) == "parent-after"

    forked_container = view.cell_get(row, 2)
    assert forked_container == {"k": [1, 2]}
    forked_container["k"].append(3)
    assert store.cell_get(row, 2) == {"k": [1, 2]}
    assert view.cell_get(row, 2)["k"] == [1, 2, 3]

    assert view.cell_del(row, 2)
    assert view.cell_get(row, 2) is _MISSING
    assert store.cell_get(row, 2) == {"k": [1, 2]}
    with pytest.raises(RuntimeError):
        view.new_row()


@pytest.mark.smoke
def test_op_store_view_translates_group_refs() -> None:
    """Group cells resolve through the fork's cloned tables, scrub-isolated."""

    from torchlens._trace_core.groups import GroupRef, MembershipGroups
    from torchlens._trace_core.op_store import OpRowStore, OpStoreLayout

    core = TraceCore()
    table = MembershipGroups(frozenset)
    core.groups["equivalent_ops"] = table
    group_id = table.add({"a", "b"})
    ref = GroupRef(table, group_id)

    layout = OpStoreLayout(("equivalent_ops",))
    store = OpRowStore(layout)
    row = store.new_row()
    store.cell_set(row, 0, ref)
    store.freeze()
    core.ops = store

    fork_core = core.fork()
    fork_ref = fork_core.ops.cell_get(row, 0)
    assert isinstance(fork_ref, GroupRef)
    assert fork_ref is not ref
    assert fork_ref.view() == frozenset({"a", "b"})
    assert fork_ref.view() is not ref.view()

    # Scrub on the parent never reaches the fork, and vice versa.
    table.replace(group_id, {"a"})
    assert fork_ref.view() == frozenset({"a", "b"})
    fork_core.groups["equivalent_ops"].replace(group_id, set())
    assert ref.view() == frozenset({"a"})


@pytest.mark.smoke
def test_core_transaction_rolls_back_stores_and_epochs() -> None:
    """One transaction restores core overlay, store overlays, and epochs."""

    from torchlens._trace_core.op_store import OpRowStore, OpStoreLayout

    core = TraceCore()
    layout = OpStoreLayout(("alpha",))
    store = OpRowStore(layout)
    row = store.new_row()
    store.cell_set(row, 0, "committed")
    store.freeze()
    core.ops = store
    kind_row = core.table("module").new_row()
    core.freeze()
    core.write("module", kind_row, "address", "committed-address")

    txn = core.transaction()
    store.cell_set(row, 0, "dirty")
    core.write("module", kind_row, "address", "dirty-address")
    core.backward_epochs.append(object())
    txn.rollback()

    assert store.cell_get(row, 0) == "committed"
    assert core.read("module", kind_row, "address") == "committed-address"
    assert core.backward_epochs == []


@pytest.mark.smoke
def test_facade_cache_weak_valued_with_strong_fallback() -> None:
    """The facade cache never pins unreferenced weak-able facades (M11 flip)."""

    import weakref

    class _WeakFacade:
        """Weak-referenceable facade stand-in."""

    core = TraceCore()
    row = core.table("op").new_row()
    core.set_facade_factory(lambda kind, r: _WeakFacade())
    first = core.facade("op", row)
    assert core.facade("op", row) is first
    facade_ref = weakref.ref(first)
    del first
    gc.collect()
    assert facade_ref() is None, "weak-valued cache must not pin a dropped facade"
    assert isinstance(core.facade("op", row), _WeakFacade)

    # Facade classes that refuse weak references (Op) stay strongly held.
    strong_core = TraceCore()
    strong_row = strong_core.table("op").new_row()
    strong_core.set_facade_factory(lambda kind, r: {"row": r})
    held = strong_core.facade("op", strong_row)
    assert strong_core.facade("op", strong_row) is held
    del held
    gc.collect()
    assert strong_core.facade("op", strong_row) == {"row": strong_row}


def test_record_dict_shadow_never_streams() -> None:
    """A ``__dict__`` shadow of a declared record field never reaches state.

    The cell descriptor is a data descriptor, so a raw ``__dict__`` write on
    a declared field name is unreachable through attribute access; streaming
    it into pickle/fork state let the dead entry silently replace the live
    cell value on restore (sol review finding 3).
    """

    import pickle

    import torchlens as tl
    from torchlens._trace_core.record_rows import record_state_items

    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU())
    trace = tl.trace(model, torch.randn(1, 3))
    param = next(iter(trace.params.values()))
    live_value = param.module_address

    param.__dict__["module_address"] = "__dict_shadow__"
    param.__dict__["user_note"] = "keep_me"

    assert param.module_address == live_value, "shadow must not affect the facade"
    occurrences = [v for k, v in record_state_items(param) if k == "module_address"]
    assert occurrences == [live_value], f"shadow leaked into state: {occurrences}"
    assert dict(record_state_items(param)).get("user_note") == "keep_me"

    restored = pickle.loads(pickle.dumps(param))
    assert restored.module_address == live_value
    assert restored.user_note == "keep_me"


@pytest.mark.smoke
def test_fork_isolation_sweep_is_index_driven_and_never_stale() -> None:
    """The eager fork sweep is SPARSE (index-driven) and the index never
    goes stale (F4; closure round 2, item 4).

    Two facts, each impossible on the eager whole-store implementation
    (which this test must FAIL against — the prior version passed there):

    * SPARSE: the sweep visits ONLY the base store's cached mutable-cell
      index. Poisoning the index (dropping one known-mutable cell) makes
      the next fork skip that cell's eager copy, so its copy-on-first-read
      backstop snapshots the parent's LATER in-place mutation — a sweep
      that ignored the index would have isolated the cell at fork time and
      kept the pre-mutation value. The index is also built once and reused
      by identity across forks.
    * NEVER STALE: a container written AFTER the index was built still
      isolates in the NEXT fork — sealed row-major stores register such
      writes in the index (``_SealedRowMajorOpRowStore.cell_set``),
      columnar stores route them through the overlay every fork snapshots.
    """

    from torchlens._trace_core.op_store import OpRowStore, OpStoreLayout, OpStoreView

    # --- Sealed row-major base (under the transpose threshold). ---
    layout = OpStoreLayout(("a", "b"))
    store = OpRowStore(layout)
    row = store.new_row()
    store.cell_set(row, 0, "atomic")
    store.cell_set(row, 1, {"x": 0})
    store.freeze()
    first = OpStoreView(store)
    first.isolate_mutable_cells()
    index = store._mutable_keys
    assert index is not None and row in index.get(1, set()), (
        "first sweep must build and cache the mutable-cell index"
    )

    # Never stale: a container written into an atomic cell after the index
    # was built registers there, so the next fork still isolates it.
    store.cell_set(row, 0, ["post-index"])
    assert row in index.get(0, set()), "post-seal container write must register in the cached index"
    second = OpStoreView(store)
    second.isolate_mutable_cells()
    assert store._mutable_keys is index, "index is cached, never rebuilt per fork"
    store.cell_get(row, 0).append("parent-mutation")
    assert second.cell_get(row, 0) == ["post-index"]

    # Sparse proof: the sweep TRUSTS the index. Drop the dict cell from it;
    # the next fork must skip the eager copy, and its lazy first read then
    # sees the parent's later in-place write. The cached sweep plan derives
    # from the index, so a hand-poisoned index drops it exactly like the
    # post-seal maintenance hooks do.
    index[1].discard(row)
    store._sweep_plan = None
    poisoned = OpStoreView(store)
    poisoned.isolate_mutable_cells()
    assert row * layout.n_fields + 1 not in poisoned._overlay, (
        "sweep visited a cell the index does not name — not index-driven"
    )
    store.cell_get(row, 1)["x"] = 99
    assert poisoned.cell_get(row, 1) == {"x": 99}, (
        "index-skipped cell was isolated eagerly — the sweep is not sparse"
    )
    index[1].add(row)

    # --- Columnar base: post-seal writes land in the overlay, which every
    # fork snapshots; unindexed frozen cells are likewise never visited. ---
    big = OpRowStore(OpStoreLayout(("a", "b")))
    for i in range(600):
        r = big.new_row()
        big.cell_set(r, 0, i)
        big.cell_set(r, 1, {"i": i})
    big.freeze()
    warm = OpStoreView(big)
    warm.isolate_mutable_cells()
    big_index = big._mutable_keys
    assert big_index is not None and list(big_index) == [1], (
        "packed numeric column must not enter the index"
    )
    big.cell_set(5, 0, {"late": 0})  # packed-column cell -> overlay write
    view = OpStoreView(big)
    view.isolate_mutable_cells()
    assert big._mutable_keys is big_index, "index is cached, never rebuilt per fork"
    big.cell_get(5, 0)["mut"] = 1
    assert view.cell_get(5, 0) == {"late": 0}

    # Sparse proof, columnar shape: drop one row from the index and the
    # next fork skips exactly that cell while its neighbors stay eager.
    remaining = list(big_index[1])
    remaining.remove(7)
    big_index[1] = tuple(remaining)
    big._sweep_plan = None
    poisoned_big = OpStoreView(big)
    poisoned_big.isolate_mutable_cells()
    n_fields = big.layout.n_fields
    assert 7 * n_fields + 1 not in poisoned_big._overlay, (
        "sweep visited a cell the index does not name — not index-driven"
    )
    assert 8 * n_fields + 1 in poisoned_big._overlay, (
        "indexed neighbor cells must still isolate eagerly at fork time"
    )
    big.cell_get(7, 1)["i"] = -1
    assert poisoned_big.cell_get(7, 1) == {"i": -1}, (
        "index-skipped cell was isolated eagerly — the sweep is not sparse"
    )


@pytest.mark.smoke
def test_fork_sweep_plan_survives_inplace_empty_to_nonempty_mutation() -> None:
    """A reused sweep plan must still snapshot an emptied-at-plan-build cell
    that the parent later filled IN PLACE (sol closure round 3 blocker).

    The cached plan classifies initially EMPTY dict/list/set cells as bare
    class allocations. In-place empty->non-empty mutation never goes through
    ``cell_set``, so no maintenance hook can invalidate the plan — the plan
    itself must guard each alloc entry against its observed value and fall
    back to a real copy when the container is no longer empty. The plan
    stays cached (no blanket invalidation: that would reintroduce the F4
    per-fork rebuild constant).
    """

    from torchlens._trace_core.op_store import OpRowStore, OpStoreLayout, OpStoreView

    # --- Sealed row-major base (under the transpose threshold). ---
    store = OpRowStore(OpStoreLayout(("d", "l", "s")))
    row = store.new_row()
    store.cell_set(row, 0, {})
    store.cell_set(row, 1, [])
    store.cell_set(row, 2, set())
    store.freeze()
    first = OpStoreView(store)
    first.isolate_mutable_cells()
    plan = store._sweep_plan
    assert plan is not None, "first fork must build and cache the sweep plan"

    # In-place empty->non-empty mutation of all three container classes:
    # no cell_set fires, so the cached plan MUST survive and stay correct.
    store.cell_get(row, 0)["present_before_second_fork"] = 1
    store.cell_get(row, 1).append(2)
    store.cell_get(row, 2).add(3)
    assert store._sweep_plan is plan, (
        "in-place mutation is invisible to the maintenance hooks; the fix "
        "must guard plan execution, not blanket-invalidate the plan"
    )

    second = OpStoreView(store)
    second.isolate_mutable_cells()
    assert second.cell_get(row, 0) == {"present_before_second_fork": 1}
    assert second.cell_get(row, 1) == [2]
    assert second.cell_get(row, 2) == {3}

    # Snapshot semantics both ways: later parent mutation stays invisible,
    # and the second fork's own writes never reach the parent.
    store.cell_get(row, 0)["late"] = 9
    store.cell_get(row, 1).append(9)
    store.cell_get(row, 2).add(9)
    assert second.cell_get(row, 0) == {"present_before_second_fork": 1}
    assert second.cell_get(row, 1) == [2]
    assert second.cell_get(row, 2) == {3}
    second.cell_get(row, 0)["fork_only"] = 0
    assert "fork_only" not in store.cell_get(row, 0)

    # --- Columnar base: cell BINDINGS are immutable post-seal, but the held
    # container object is not — the same in-place transition must isolate.
    big = OpRowStore(OpStoreLayout(("a", "b")))
    for i in range(600):
        r = big.new_row()
        big.cell_set(r, 0, f"op_{i}")
        big.cell_set(r, 1, {})
    big.freeze()
    assert big._columns is not None, "600 rows must have transposed to columns"
    warm = OpStoreView(big)
    warm.isolate_mutable_cells()
    big_plan = big._sweep_plan
    assert big_plan is not None
    big.cell_get(5, 1)["present_before_second_fork"] = 1
    assert big._sweep_plan is big_plan
    later = OpStoreView(big)
    later.isolate_mutable_cells()
    assert later.cell_get(5, 1) == {"present_before_second_fork": 1}
    big.cell_get(5, 1)["late"] = 9
    assert later.cell_get(5, 1) == {"present_before_second_fork": 1}


# Kept by shrink-r4 (per-id subset rule): sole coverage of _trace_core/op_store.py.
def test_sparse_isolation_fork_of_fork() -> None:
    """A fork-of-fork snapshots the parent view's container writes."""

    from torchlens._trace_core.op_store import OpRowStore, OpStoreLayout, OpStoreView

    store = OpRowStore(OpStoreLayout(("a", "b")))
    row = store.new_row()
    store.cell_set(row, 0, {"base": 0})
    store.cell_set(row, 1, "atomic")
    store.freeze()
    parent_view = OpStoreView(store)
    parent_view.isolate_mutable_cells()
    parent_view.cell_get(row, 0)["parent_write"] = 1
    child_view = OpStoreView(parent_view)
    child_view.isolate_mutable_cells()
    parent_view.cell_get(row, 0)["after_child"] = 2
    assert child_view.cell_get(row, 0) == {"base": 0, "parent_write": 1}


@pytest.mark.smoke
def test_rehydrate_rollback_after_relation_freeze_is_atomic() -> None:
    """A failure AFTER the relation freeze leaves detached topology intact.

    Rehydration used to adopt each detached store's ``_cells`` list BY
    IDENTITY; the relation freeze then mutated those lists in place
    (dataflow ``_CSR`` sentinel, ``GroupRef``/fact-block slot writes), so
    the rollback rebound the original stores to already-corrupted rows —
    sol's closure-round-2 injection showed ``parents`` flipping from
    ``('input_1',)`` to ``()`` after a failed rehydration (blocking item
    1). Adoption now snapshots the cells, making the documented
    all-or-nothing rollback semantically atomic.
    """

    import pickle
    from unittest import mock

    import torchlens as tl
    from torchlens._trace_core import relation_views
    from torchlens._trace_core.groups import GroupRef
    from torchlens._trace_core.op_store import _CSR, _FACT, DetachedOpStore

    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU())
    trace = tl.trace(model, torch.randn(1, 3))
    payload = pickle.dumps(trace)

    # Clean-load control: the public relation topology every op must keep.
    control = pickle.loads(payload)
    expected = {op.layer_label: (op.parents, op.children) for op in control._layer_list}

    real_freeze = relation_views.freeze_trace_relation_views

    def freeze_then_fail(target):
        # Mutate the adopted rows exactly like a real run, THEN fail — the
        # scenario the heterogeneous pre-scan tests never reach.
        real_freeze(target)
        raise RuntimeError("injected post-freeze rehydration failure")

    with mock.patch.object(
        relation_views, "freeze_trace_relation_views", side_effect=freeze_then_fail
    ):
        clone = pickle.loads(payload)  # load must survive (best-effort)

    assert clone.__dict__.get("_trace_core") is None, "rollback must stay coreless"
    for op in clone._layer_list:
        bound = object.__getattribute__(op, "_core")
        assert isinstance(bound, DetachedOpStore), (
            f"{op.layer_label} not rolled back to its detached store"
        )
        # Byte-level probe: no freeze artifact may survive in detached rows.
        for name, value in bound.items(0):
            assert value is not _CSR and value is not _FACT, (
                f"freeze sentinel leaked into detached cell {name}"
            )
            assert value.__class__ is not GroupRef, f"GroupRef leaked into detached cell {name}"
    for op in clone._layer_list:
        assert (op.parents, op.children) == expected[op.layer_label], (
            f"detached topology corrupted for {op.layer_label}"
        )
    assert clone.ops["linear_1_1"].parents == ("input_1",)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _data_classes/_trace_rehydrate.py, _postprocess/_executor.py.
def test_loaded_partial_capture_stays_staging() -> None:
    """A loaded partial/failed capture is never rehydrated or frozen.

    Partials are documented as keeping their staging surface; running the
    F9 relation freeze + seal on them at load re-bound their ops to a
    frozen shared store and converted relation cells to interned views and
    ``GroupRef`` group tables (closure review, blocking item 5 / fix item
    4). A loaded partial must match the pre-F9 coreless behavior exactly:
    no core, detached-backed unfrozen op rows, no relation-freeze
    artifacts. (The list->tuple container coercion of legacy load states
    in ``Op.__setstate__`` predates F9 and is out of this guard's scope.)
    """

    import pickle
    from unittest import mock

    import torchlens as tl
    import torchlens._postprocess.finalization as finalization
    from torchlens._trace_core.op_store import DetachedOpStore

    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU())

    # Fail LATE in postprocess (after layer logs exist) so the partial has
    # reachable ops — the case where an unguarded rehydrate would freeze.
    with mock.patch.object(finalization, "_build_module_logs", side_effect=RuntimeError("boom")):
        with pytest.raises(Exception) as exc_info:
            tl.trace(model, torch.randn(1, 3))
    partial = exc_info.value.partial_trace
    assert partial is not None
    source = partial.trace
    assert source.__dict__.get("_tracing_finished") is False
    assert len(source._layer_list) > 0, "late failure must leave reachable ops"

    clone = pickle.loads(pickle.dumps(source))
    assert clone.__dict__.get("_tracing_finished") is False
    assert clone.__dict__.get("_trace_core") is None, "partial captures must stay coreless on load"
    for staged in clone._layer_list:
        bound = object.__getattribute__(staged, "_core")
        assert isinstance(bound, DetachedOpStore), (
            f"partial op re-bound to {type(bound).__name__} on load"
        )
        assert not bound.frozen
        assert bound.dataflow_edges is None, "no relation-freeze artifacts"
    op = clone._layer_list[0]
    # Group cells stay raw (no GroupRef conversion) on a loaded partial.
    store = object.__getattribute__(op, "_core")
    fid = store.layout.fid_by_name.get("equivalent_ops")
    if fid is not None:
        cell = store.cell_get(0, fid)
        assert type(cell).__name__ != "GroupRef"
