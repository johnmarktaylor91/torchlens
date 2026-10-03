"""Unit suite for the M6 relation freeze (``_trace_core/relation_views.py``).

Covers the conversion mechanics directly on synthetic stores — edge
projection, differential CSR clearing vs explicit-view fallback, label
resolution failure modes — plus the end-to-end behavioral surface on real
captures: immutable views, CSR rematerialization identity, idempotence,
legacy load coercion, and removal scrubbing over immutable views.
"""

from __future__ import annotations

import pickle

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._trace_core.core import TraceCore
from torchlens._trace_core.op_store import _CSR, _MISSING, OpRowStore, OpStoreLayout
from torchlens._trace_core.relation_views import (
    DATAFLOW_FAMILY,
    freeze_op_relation_views,
    freeze_trace_relation_views,
    frozenset_view,
    intern_view,
    materialize_dataflow_view,
    tuple_view,
)

_SEED = 20260812


def _make_layout() -> OpStoreLayout:
    """Build a minimal layout carrying the relation fields under test."""

    return OpStoreLayout(
        (
            "layer_label",
            "parents",
            "children",
            "parent_arg_positions",
            "modules",
            "input_ancestors",
        )
    )


def _make_store(rows: list[dict[str, object]]) -> tuple[TraceCore, OpRowStore]:
    """Build a building-phase store from per-row field dicts."""

    layout = _make_layout()
    core = TraceCore()
    store = OpRowStore(layout)
    core.ops = store
    for fields in rows:
        cells: list[object] = [_MISSING] * layout.n_fields
        for name, value in fields.items():
            cells[layout.fid_by_name[name]] = value
        store.adopt_row(cells)
    return core, store


def _resolver(store: OpRowStore) -> dict[str, int]:
    """Map each row's ``layer_label`` cell to its row id."""

    fid = store.layout.fid_by_name["layer_label"]
    rows = store.rows_building()
    assert rows is not None
    return {cells[fid]: row for row, cells in enumerate(rows) if cells[fid] is not _MISSING}


class TestInternPool:
    """The view intern pool."""

    def test_equal_views_share_one_instance(self) -> None:
        pool: dict = {}
        first = tuple_view(["a", "b"], pool)
        second = tuple_view(("a", "b"), pool)
        assert first is second
        assert isinstance(first, tuple)

    def test_empty_collapses_to_shared_singleton(self) -> None:
        pool: dict = {}
        assert tuple_view([], pool) is tuple_view([], pool)
        assert frozenset_view(set(), pool) is frozenset_view(set(), pool)

    def test_unhashable_view_returned_unpooled(self) -> None:
        pool: dict = {}
        value = ([1, 2],)
        assert intern_view(value, pool) is value
        assert not pool


class TestFreezeOpRelationViews:
    """Direct conversion mechanics on synthetic stores."""

    def test_linear_chain_projects_and_clears(self) -> None:
        core, store = _make_store(
            [
                {"layer_label": "a", "parents": [], "children": ["b"]},
                {
                    "layer_label": "b",
                    "parents": ["a"],
                    "children": ["c"],
                    "parent_arg_positions": {"args": {0: "a"}, "kwargs": {}},
                },
                {
                    "layer_label": "c",
                    "parents": ["b"],
                    "children": [],
                    "parent_arg_positions": {"args": {0: "b"}, "kwargs": {}},
                },
            ]
        )
        stats = freeze_op_relation_views(core, store, _resolver(store).get)
        assert stats.edges == 2
        assert stats.unresolved_labels == 0
        assert stats.explicit_view_cells == 0
        rows = store.rows_building()
        assert rows is not None
        parents_fid = store.layout.fid_by_name["parents"]
        children_fid = store.layout.fid_by_name["children"]
        # Every populated dataflow cell cleared into the CSR.
        assert rows[1][parents_fid] is _CSR
        assert rows[1][children_fid] is _CSR
        assert materialize_dataflow_view(store, 1, "parents") == ("a",)
        assert materialize_dataflow_view(store, 1, "children") == ("c",)
        assert materialize_dataflow_view(store, 0, "children") == ("b",)
        assert materialize_dataflow_view(store, 2, "children") == ()

    def test_edge_occurrences_carry_arg_positions(self) -> None:
        core, store = _make_store(
            [
                {"layer_label": "x", "parents": [], "children": ["y"]},
                {
                    "layer_label": "y",
                    "parents": ["x"],
                    "children": [],
                    "parent_arg_positions": {
                        "args": {0: "x", 1: "x"},
                        "kwargs": {},
                    },
                },
            ]
        )
        freeze_op_relation_views(core, store, _resolver(store).get)
        edges = core.edge_table(DATAFLOW_FAMILY)
        # One occurrence per attributed argument position: x feeds y twice.
        occurrences = list(edges.in_edges(1))
        assert len(occurrences) == 2
        assert {edge.arg_position for edge in occurrences} == {
            ("args", 0),
            ("args", 1),
        }
        # The rematerialized public view still dedups to one label.
        assert materialize_dataflow_view(store, 1, "parents") == ("x",)

    def test_unresolved_label_keeps_explicit_view(self) -> None:
        core, store = _make_store(
            [
                {
                    "layer_label": "a",
                    "parents": ["ghost"],
                    "children": [],
                },
            ]
        )
        stats = freeze_op_relation_views(core, store, _resolver(store).get)
        assert stats.unresolved_labels == 1
        assert stats.explicit_view_cells >= 1
        rows = store.rows_building()
        assert rows is not None
        parents_fid = store.layout.fid_by_name["parents"]
        value = rows[0][parents_fid]
        assert value == ("ghost",)
        assert isinstance(value, tuple)

    def test_non_dataflow_families_convert_in_place(self) -> None:
        core, store = _make_store(
            [
                {
                    "layer_label": "a",
                    "parents": [],
                    "children": [],
                    "modules": ["m1:1", "m2:1"],
                    "input_ancestors": {"a"},
                },
            ]
        )
        freeze_op_relation_views(core, store, _resolver(store).get)
        rows = store.rows_building()
        assert rows is not None
        modules = rows[0][store.layout.fid_by_name["modules"]]
        ancestors = rows[0][store.layout.fid_by_name["input_ancestors"]]
        assert modules == ("m1:1", "m2:1") and isinstance(modules, tuple)
        assert ancestors == frozenset({"a"}) and isinstance(ancestors, frozenset)

    def test_children_order_mismatch_keeps_explicit_view(self) -> None:
        # Staging children in NON-chronological order cannot be reproduced by
        # the row-major edge emission; the differential check must keep the
        # staging order as an explicit view rather than silently reorder.
        core, store = _make_store(
            [
                {"layer_label": "a", "parents": [], "children": ["c", "b"]},
                {"layer_label": "b", "parents": ["a"], "children": []},
                {"layer_label": "c", "parents": ["a"], "children": []},
            ]
        )
        stats = freeze_op_relation_views(core, store, _resolver(store).get)
        rows = store.rows_building()
        assert rows is not None
        children_fid = store.layout.fid_by_name["children"]
        value = rows[0][children_fid]
        assert value == ("c", "b")
        assert isinstance(value, tuple)
        assert stats.explicit_view_cells >= 1


class _ChainCNN(nn.Module):
    """Deterministic conv chain for end-to-end assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.head = nn.Linear(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.conv(x))
        return self.head(y.mean(dim=(2, 3)))


def _capture() -> tl.Trace:
    torch.manual_seed(_SEED)
    return tl.trace(_ChainCNN(), torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4))


class TestFinishedTraceSurface:
    """End-to-end immutable-view surface on real captures."""

    def test_dataflow_round_trip_matches_capture(self) -> None:
        trace = _capture()
        op = trace.ops["relu_1_2"]
        assert op.parents == ("conv2d_1_1",)
        assert op.children == ("mean_1_3",)
        assert op.parents is op.parents

    def test_csr_backed_cells_rematerialize_lazily(self) -> None:
        trace = _capture()
        store = trace.__dict__["_trace_core"].ops
        assert store.dataflow_edges is not None
        assert len(store.dataflow_edges) > 0
        # Uninspected rows hold the CSR sentinel, not per-row containers.
        rows_or_none = store.rows_building()
        parents_fid = store.layout.fid_by_name["parents"]
        if rows_or_none is not None:
            assert any(cells[parents_fid] is _CSR for cells in rows_or_none)

    def test_freeze_is_idempotent(self) -> None:
        trace = _capture()
        assert freeze_trace_relation_views(trace) is None

    def test_removal_scrubs_immutable_views(self) -> None:
        trace = _capture()
        removed = trace.ops["relu_1_2"]
        surviving_parent = trace.ops["conv2d_1_1"]
        assert "relu_1_2" in surviving_parent.children
        trace._remove_log_entry(removed, remove_references=True)
        assert "relu_1_2" not in surviving_parent.children
        assert isinstance(surviving_parent.children, tuple)

    def test_pickle_round_trip_preserves_views(self) -> None:
        trace = _capture()
        clone = pickle.loads(pickle.dumps(trace))
        op = clone.ops["relu_1_2"]
        assert op.parents == ("conv2d_1_1",)
        assert isinstance(op.parents, tuple)
        assert isinstance(op.input_ancestors, frozenset)

    def test_legacy_list_state_coerces_to_views_on_load(self) -> None:
        trace = _capture()
        op = trace.ops["relu_1_2"]
        state = op.__getstate__()
        # Regress the state to the pre-M6 container types.
        state["parents"] = list(state["parents"])
        state["children"] = list(state["children"])
        state["input_ancestors"] = set(state["input_ancestors"])
        restored = object.__new__(type(op))
        restored.__setstate__(state)
        assert isinstance(restored.parents, tuple)
        assert isinstance(restored.children, tuple)
        assert isinstance(restored.input_ancestors, frozenset)

    def test_layer_aggregate_views(self) -> None:
        trace = _capture()
        layer = trace["relu_1_2"]
        assert isinstance(layer.parents, tuple)
        assert isinstance(layer.children, tuple)
        assert isinstance(layer.modules, tuple)

    def test_finished_assignment_normalizes_container_subclasses(self) -> None:
        # Review finding 5: an exact-type check let mutable list/set
        # SUBCLASSES bypass view normalization on finished ops.
        class MutableList(list):
            pass

        class MutableSet(set):
            pass

        trace = _capture()
        op = trace.ops["relu_1_2"]
        op.children = MutableList(["mean_1_3"])
        assert type(op.children) is tuple
        op.input_ancestors = MutableSet({"input_1_1"})
        assert type(op.input_ancestors) is frozenset
        op.equivalent_ops = MutableSet({"relu_1_2"})
        assert type(op.equivalent_ops) is frozenset

    def test_finished_assignment_refuses_non_container_values(self) -> None:
        import pytest

        trace = _capture()
        op = trace.ops["relu_1_2"]
        with pytest.raises(TypeError, match="finished relation field"):
            op.children = object()
        # None passes through (immutable, not a container).
        op.modules = None
        assert op.modules is None

    def test_layer_direct_assignment_normalizes_to_views(self) -> None:
        # Review finding 5: Layer shadow setters stored raw mutable
        # containers, so `layer.modules = [...]` re-exposed a mutable list
        # and `layer.equivalent_ops = {...}` a mutable set on finished
        # traces.
        trace = _capture()
        layer = trace["relu_1_2"]
        escape = ["__mutable_escape__"]
        layer.modules = escape
        assert type(layer.modules) is tuple
        escape.append("late_mutation")
        assert "late_mutation" not in layer.modules
        layer.equivalent_ops = {"relu_1_2"}
        assert type(layer.equivalent_ops) is frozenset
        layer.input_ancestors = {"input_1_1"}
        assert type(layer.input_ancestors) is frozenset

    def test_staging_phase_stays_mutable_for_detached_building(self) -> None:
        # The building-phase write path must stay raw: a fresh building store
        # accepts and returns the mutable staging containers unchanged.
        layout = _make_layout()
        store = OpRowStore(layout)
        cells: list[object] = [_MISSING] * layout.n_fields
        store.adopt_row(cells)
        staging = ["raw"]
        store.cell_set(0, layout.fid_by_name["parents"], staging)
        assert store.cell_get(0, layout.fid_by_name["parents"]) is staging


class TestFactBlocks:
    """The M7 shared-fact blocks (FunctionCall / ParamAlias group tables)."""

    @staticmethod
    def _fact_store(rows: list[dict[str, object]]) -> OpRowStore:
        """Build a building-phase store carrying the fact fields."""

        layout = OpStoreLayout(
            (
                "func_call_id",
                "code_context",
                "non_tensor_kwargs",
                "func_config",
                "arg_names",
                "param_shapes",
            )
        )
        store = OpRowStore(layout)
        for fields in rows:
            cells: list[object] = [_MISSING] * layout.n_fields
            for name, value in fields.items():
                cells[layout.fid_by_name[name]] = value
            store.adopt_row(cells)
        return store

    def test_call_siblings_share_one_canonical(self) -> None:
        from torchlens._trace_core.fact_blocks import convert_fact_cells
        from torchlens._trace_core.op_store import _FACT

        store = self._fact_store(
            [
                {"func_call_id": 7, "func_config": {"dim": 0}, "arg_names": ("x",)},
                {"func_call_id": 7, "func_config": {"dim": 0}, "arg_names": ("x",)},
                {"func_call_id": 8, "func_config": {"dim": 1}, "arg_names": ("x",)},
            ]
        )
        convert_fact_cells(store, {})
        rows = store.rows_building()
        assert rows is not None
        config_fid = store.layout.fid_by_name["func_config"]
        assert all(cells[config_fid] is _FACT for cells in rows)
        family = store.fact_blocks.families["call"]
        assert len(family) == 2
        # Equal arg_names canonicals intern to ONE tuple across call groups.
        names = family.columns["arg_names"]
        assert names[0] is names[1]

    def test_hydration_isolates_mutable_types_per_row(self) -> None:
        from torchlens._trace_core.fact_blocks import convert_fact_cells

        store = self._fact_store(
            [
                {"func_call_id": 1, "func_config": {"dim": 0}, "code_context": ["frame"]},
                {"func_call_id": 1, "func_config": {"dim": 0}, "code_context": ["frame"]},
            ]
        )
        convert_fact_cells(store, {})
        first = store.fact_blocks.hydrate(0, "func_config")
        second = store.fact_blocks.hydrate(1, "func_config")
        assert first == {"dim": 0} and second == {"dim": 0}
        assert first is not second
        assert isinstance(store.fact_blocks.hydrate(0, "code_context"), list)

    def test_unequal_sibling_keeps_explicit_cell(self) -> None:
        from torchlens._trace_core.fact_blocks import convert_fact_cells
        from torchlens._trace_core.op_store import _FACT

        divergent = {"dim": 999}
        store = self._fact_store(
            [
                {"func_call_id": 3, "func_config": {"dim": 0}},
                {"func_call_id": 3, "func_config": divergent},
            ]
        )
        convert_fact_cells(store, {})
        rows = store.rows_building()
        assert rows is not None
        config_fid = store.layout.fid_by_name["func_config"]
        assert rows[0][config_fid] is _FACT
        assert rows[1][config_fid] is divergent

    def test_param_family_groups_by_value(self) -> None:
        from torchlens._trace_core.fact_blocks import convert_fact_cells
        from torchlens._trace_core.op_store import _FACT

        store = self._fact_store(
            [
                {"func_call_id": 1, "param_shapes": [(4, 3), (4,)]},
                {"func_call_id": 2, "param_shapes": [(4, 3), (4,)]},
                {"func_call_id": 3, "param_shapes": [(2, 2)]},
            ]
        )
        convert_fact_cells(store, {})
        rows = store.rows_building()
        assert rows is not None
        fid = store.layout.fid_by_name["param_shapes"]
        assert all(cells[fid] is _FACT for cells in rows)
        family = store.fact_blocks.families["param"]
        assert len(family.columns["param_shapes"]) == 2
        assert store.fact_blocks.hydrate(0, "param_shapes") == [(4, 3), (4,)]

    def test_param_family_joins_are_scoped_by_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Independent param fact columns cannot overwrite each other's row joins."""

        from torchlens._trace_core.fact_blocks import (
            OP_PARAM_FACT_FIELDS,
            convert_fact_cells,
        )

        monkeypatch.setitem(OP_PARAM_FACT_FIELDS, "extra_param_facts", list)
        layout = OpStoreLayout(("param_shapes", "extra_param_facts"))
        store = OpRowStore(layout)
        for param_shapes, extra_facts in ((["shared"], ["first"]), (["shared"], ["second"])):
            cells: list[object] = [_MISSING] * layout.n_fields
            cells[layout.fid_by_name["param_shapes"]] = param_shapes
            cells[layout.fid_by_name["extra_param_facts"]] = extra_facts
            store.adopt_row(cells)

        convert_fact_cells(store, {})

        assert store.fact_blocks.hydrate(1, "param_shapes") == ["shared"]
        assert store.fact_blocks.hydrate(0, "extra_param_facts") == ["first"]
        assert store.fact_blocks.hydrate(1, "extra_param_facts") == ["second"]

    def test_finished_trace_fact_surface(self) -> None:
        from torchlens._trace_core.op_store import _FACT

        trace = _capture()
        store = trace.__dict__["_trace_core"].ops
        assert store.fact_blocks is not None
        op = trace.ops["relu_1_2"]
        fid = store.layout.fid_by_name["func_config"]
        assert store.cell_get(op._row, fid) is _FACT
        config = op.func_config
        assert isinstance(config, dict)
        # Identity-stable after the first read; per-row mutation isolated.
        assert op.func_config is config
        config["__probe__"] = 1
        sibling = trace.ops["conv2d_1_1"]
        assert "__probe__" not in sibling.func_config
        assert isinstance(op.code_context, list)
        assert isinstance(op.arg_names, tuple)

    def test_direct_write_overrides_shared_fact(self) -> None:
        trace = _capture()
        op = trace.ops["relu_1_2"]
        op.func_config = {"replaced": True}
        assert op.func_config == {"replaced": True}
        assert trace.ops["conv2d_1_1"].func_config != {"replaced": True}

    def test_fact_cell_delete_is_genuinely_absent(self) -> None:
        from torchlens._trace_core.fact_blocks import convert_fact_cells

        store = self._fact_store([{"func_call_id": 1, "func_config": {"dim": 0}}])
        convert_fact_cells(store, {})
        fid = store.layout.fid_by_name["func_config"]
        assert store.cell_del(0, fid) is True
        assert store.cell_get(0, fid) is _MISSING

    def test_pickle_round_trip_materializes_facts(self) -> None:
        trace = _capture()
        clone = pickle.loads(pickle.dumps(trace))
        op = clone.ops["relu_1_2"]
        assert isinstance(op.func_config, dict)
        assert isinstance(op.code_context, list)
        assert op.func_name == "relu"
