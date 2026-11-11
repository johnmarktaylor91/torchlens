"""The write-time alias registry that replaced the freeze-seam alias census.

Container pooling must never replace a container cell that another cell also
reaches. The registry records, at write time, the op cells that may share a
container; the full census survives as a test oracle that the suite runs beside
every registry read (``tests/conftest.py``). These tests pin the registry's
writers, the oracle itself, and the oracle's coverage on real captures.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes import _op_alias_oracle, _op_container_census, op as op_mod
from torchlens._trace_core.alias_registry import register_shared_cells
from torchlens._trace_core.op_store import _MISSING, OpRowStore, OpStoreLayout, PooledCell

_LAYOUT = OpStoreLayout(("sanctioned", "other", "spare"))


def _store(rows: list[list[Any]]) -> OpRowStore:
    store = OpRowStore(_LAYOUT)
    for cells in rows:
        store.adopt_row(cells)
    return store


def test_suite_runs_the_census_oracle() -> None:
    """The completeness check is armed for the whole suite."""

    assert _op_container_census.ALIAS_ORACLE is _op_alias_oracle.assert_registry_matches_census, (
        "tests/conftest.py must arm the alias oracle"
    )


def test_cell_set_registers_a_shared_container() -> None:
    """A container written through ``cell_set`` into a non-candidate cell blocks pooling."""

    shared = ["x", "y"]
    store = _store([[_MISSING, _MISSING, _MISSING] for _ in range(4)])
    store.cell_set(0, 1, shared)
    for row in range(4):
        store.cell_set(row, 0, shared if row == 0 else ["x", "y"])
    op_mod._pool_container_cells([(store, (0,))], {})
    rows = store.rows_building()
    assert rows is not None
    assert rows[0][0] is shared and rows[0][1] is shared, "the alias must survive"
    assert {id(rows[r][0]) for r in (1, 2, 3)} == {id(rows[1][0])}
    assert rows[1][0].__class__ is PooledCell


def test_oracle_catches_an_unregistered_alias_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A writer that shares a container without registering it fails the oracle."""

    monkeypatch.setattr(
        _op_container_census,
        "ALIAS_ORACLE",
        _op_alias_oracle.assert_registry_matches_census,
    )
    shared: list[Any] = []
    # ``adopt_row`` is a bulk write that registers nothing: the non-candidate
    # cell of row 0 aliases the candidate cell of row 1 unseen by the registry.
    store = _store([[[], shared, _MISSING], [shared, _MISSING, _MISSING], [[], {}, _MISSING]])
    with pytest.raises(AssertionError, match="alias decisions differ"):
        op_mod._pool_container_cells([(store, (0,))], {})


def test_registry_read_matches_census_on_registered_writes() -> None:
    """With every sharing write registered, registry read and census agree."""

    shared = {"k": ["v"]}
    store = _store([[_MISSING, _MISSING, _MISSING] for _ in range(3)])
    store.cell_set(0, 1, shared)
    store.cell_set(1, 0, shared)
    store.cell_set(2, 0, {"k": ["v"]})
    stores = [(store, (0,))]
    result = _op_container_census._registry_container_cells(stores)
    _op_alias_oracle.assert_registry_matches_census(stores, result)
    assert result is not None
    assert result[0][id(shared)] == 2


def test_released_rows_and_overwrites_drop_stale_registrations() -> None:
    """A registered cell later overwritten or released no longer counts."""

    shared = ["a"]
    store = _store([[_MISSING, _MISSING, _MISSING] for _ in range(2)])
    store.cell_set(0, 1, shared)
    store.cell_set(1, 0, shared)
    store.cell_set(0, 1, ("plain",))
    result = _op_container_census._registry_container_cells([(store, (0,))])
    assert result is not None and result[0][id(shared)] == 1
    store.cell_set(0, 2, shared)
    store.release_row(0)
    result = _op_container_census._registry_container_cells([(store, (0,))])
    assert result is not None and result[0][id(shared)] == 1


def test_freeze_drops_the_registry() -> None:
    """The registry holds no reference past the freeze seam."""

    store = _store([[_MISSING, _MISSING, _MISSING]])
    store.cell_set(0, 1, ["x"])
    assert store._alias_sites
    store.freeze()
    assert not store._alias_sites
    store.cell_set(0, 1, ["y"])
    assert not store._alias_sites


def test_shallow_copy_registers_both_cells() -> None:
    """``register_shared_cells`` records the source and clone cells of a shared container."""

    class _Row:
        def __init__(self, store: OpRowStore, row: int) -> None:
            object.__setattr__(self, "_core", store)
            object.__setattr__(self, "_row", row)

    shared = ["s"]
    store = _store([[_MISSING, shared, ()], [_MISSING, shared, ()]])
    source, clone = _Row(store, 0), _Row(store, 1)
    assert register_shared_cells(source, clone) is clone
    assert store._alias_sites == {(0, 1): shared, (1, 1): shared}


def test_exec_ctx_fields_are_candidate_columns() -> None:
    """The function boundary's shared dicts land in columns the registry read walks in full.

    ``_function_boundary.open_boundary_call`` hands one ``rng_states`` and one
    ``autocast_state`` object to every op of a boundary call; they are covered
    without a write-time hook only while both columns stay pooling candidates.
    """

    assert {"func_rng_states", "func_autocast_state"} <= op_mod._POOLED_CONTAINER_FIELDS


class _ShareAutocast(torch.autograd.Function):
    """A custom Function: the boundary op and its input views share ``exec_ctx`` dicts."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return x * 2

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        return grad * 2


class _Mixed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.shape = [2, 8]
        self.extra = {"k": [1, 2]}

    def forward(self, x: torch.Tensor) -> Any:
        y = _ShareAutocast.apply(self.lin(x))
        z = torch.relu(y).reshape(self.shape)
        return z, y


@pytest.mark.parametrize("save", [None, "all"])
def test_oracle_holds_on_a_mixed_capture(save: str | None) -> None:
    """A capture with a Function boundary, module attributes and two outputs passes the oracle."""

    torch.manual_seed(0)
    model = _Mixed()
    x = torch.randn(4, 4)
    trace = tl.trace(model, x) if save is None else tl.trace(model, x, save=save)
    assert len(trace.ops) > 3
