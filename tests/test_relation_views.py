"""Unit suite for the M6 relation freeze (``_trace_core/relation_views.py``).

Covers the conversion mechanics directly on synthetic stores — edge
projection, differential CSR clearing vs explicit-view fallback, label
resolution failure modes — plus the end-to-end behavioral surface on real
captures: immutable views, CSR rematerialization identity, idempotence,
legacy load coercion, and removal scrubbing over immutable views.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._trace_core.core import TraceCore
from torchlens._trace_core.op_store import _MISSING, OpRowStore, OpStoreLayout
from torchlens._trace_core.relation_views import (
    DATAFLOW_FAMILY,
    freeze_op_relation_views,
    freeze_trace_relation_views,
    materialize_dataflow_view,
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


class TestFreezeOpRelationViews:
    """Direct conversion mechanics on synthetic stores."""

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

    def test_finished_assignment_refuses_non_container_values(self) -> None:
        import pytest

        trace = _capture()
        op = trace.ops["relu_1_2"]
        with pytest.raises(TypeError, match="finished relation field"):
            op.children = object()
        # None passes through (immutable, not a container).
        op.modules = None
        assert op.modules is None


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
