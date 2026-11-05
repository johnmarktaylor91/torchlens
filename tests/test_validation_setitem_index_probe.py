"""In-domain index probe for integer and bool-mask index components of ``__setitem__``.

Forward validation proves each saved parent influences its op by perturbing
it. For ``dest[index] = value`` the default draw and the uniform value steps
can leave the write unchanged although the index edge is real: a permuted row
index writes the same cells with the same scalar, and the rows a step reaches
may already hold the written value. These tests pin that a real index edge is
proven by a write-changing in-domain move, that a dead index edge (no in-domain
position changes the write) still fails validation, and that ``__setitem__``
never enters the index-domain slot table the exemptions read.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.validation import _index_domain
from torchlens.validation._index_domain import (
    _INDEX_DOMAIN_ARG_SPECS,
    _INDEX_SINGLE_ENTRY_STRATEGY,
    _parent_is_index_domain_arg,
    _raise_retry_strategies,
    index_domain_single_entry_values,
    layer_has_index_domain_parent,
    layer_takes_single_entry_probe,
)
from torchlens.validation._setitem_index_probe import setitem_index_probe_values


class NeighbourRowWrite(nn.Module):
    """``state[i, 1] = -1`` at a traced row ``i`` (an asynchronous unit update)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x.clone()
        row = torch.arange(x.shape[0])[15]
        state[row, 1] = -1.0
        return state


class RowColumnWrite(nn.Module):
    """``dest[arange(2), cols] = 10`` with every row writing the same column."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dest = x.clone()
        dest[torch.arange(2), torch.full((2,), 10)] = 10.0
        return dest


def _neighbour_input(rows_holding_value: int) -> torch.Tensor:
    """Return a ``[20, 4]`` input whose first ``rows_holding_value`` rows hold -1 in column 1."""

    torch.manual_seed(0)
    x = torch.randn(20, 4)
    x[:, 1] = 1.0
    x[:rows_holding_value, 1] = -1.0
    return x


def _setitem_layer(destination: torch.Tensor, index: Any, value: Any, positions: dict) -> Any:
    return SimpleNamespace(
        func_name="__setitem__",
        saved_args=(destination, index, value),
        saved_kwargs={},
        parent_arg_positions={"args": positions, "kwargs": {}},
    )


@pytest.fixture
def probe_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[torch.Tensor, Any]]:
    """Record every ``__setitem__`` index probe: the saved index and the chosen move."""

    calls: list[tuple[torch.Tensor, Any]] = []
    probe = _index_domain.setitem_index_probe_values

    def spy(layer: Any, parent_label: str, parent_values: torch.Tensor) -> Any:
        moved = probe(layer, parent_label, parent_values)
        calls.append((parent_values.clone(), moved))
        return moved

    monkeypatch.setattr(_index_domain, "setitem_index_probe_values", spy)
    return calls


def test_write_into_rows_already_holding_the_value_validates(probe_calls) -> None:
    """Rows 0-16 already hold -1, so the draw (0-10) and the +-1 steps are no-ops.

    Only a move to row 17, 18 or 19 changes the write; the probe finds it and the
    replay proves the index edge.
    """

    assert tl.validate(NeighbourRowWrite(), _neighbour_input(17), scope="forward") is True
    assert probe_calls, "the probe never ran: the draw or a step proved the edge"
    saved, moved = probe_calls[-1]
    assert saved.tolist() == 15
    assert moved is not None and moved.tolist() == 17


def test_permuted_rows_writing_one_scalar_validate() -> None:
    """Seed 3 draws the row swap, which writes the same two cells; the probe moves one row."""

    torch.manual_seed(3)
    x = torch.randn(2, 20)
    assert tl.validate(RowColumnWrite(), x, scope="forward", seed=3) is True


def test_dead_setitem_index_edge_takes_the_held_value_proof(probe_calls) -> None:
    """Every row already holds -1: the probe finds no move, and only the held-value proof passes it.

    Every cell the row index can address (column 1 of every row) holds the written
    value, so the ``setitem_destination_holds_value`` proof exempts the edge. A
    destination where one addressable cell differs is still perturbation-tested:
    ``tests/test_validation_unread_value_proofs.py::test_setitem_held_value_planted_bug_fails``.
    """

    assert tl.validate(NeighbourRowWrite(), _neighbour_input(20), scope="forward") is True
    assert probe_calls and all(moved is None for _, moved in probe_calls)


def test_probe_moves_one_entry_to_a_position_that_changes_the_write() -> None:
    destination = torch.zeros(2, 20)
    rows, cols = torch.tensor([0, 1]), torch.tensor([10, 10])
    layer = _setitem_layer(destination, (rows, cols), 10, {0: "dest", (1, 0): "rows"})
    moved = setitem_index_probe_values(layer, "rows", rows)
    assert moved is not None and moved.tolist() == [1, 1]
    assert setitem_index_probe_values(layer, "dest", rows) is None


def test_probe_keeps_negative_indices_in_domain_and_bare_index_is_probed() -> None:
    destination = torch.zeros(5)
    destination[0] = 3.0
    index = torch.tensor([-1, -2])
    layer = _setitem_layer(destination, index, 3.0, {1: "index"})
    moved = setitem_index_probe_values(layer, "index", index)
    assert moved is not None and moved.tolist() == [0, -2]
    assert bool(((moved >= -5) & (moved < 5)).all())


_ROWS = torch.tensor([0, 1])
_MASK = torch.tensor([True, False, True])


@pytest.mark.parametrize(
    ("index", "positions", "label", "values"),
    [
        ((_ROWS, slice(None)), {(1, 0): "rows", 0: "rows"}, "rows", _ROWS),
        (_ROWS.to(torch.uint8), {1: "index"}, "index", _ROWS.to(torch.uint8)),
        (_ROWS.float(), {1: "index"}, "index", _ROWS.float()),
    ],
    ids=["parent-in-two-slots", "uint8-mask-dtype", "float-index"],
)
def test_probe_refuses_components_it_does_not_model(index, positions, label, values) -> None:
    layer = _setitem_layer(torch.zeros(3, 3), index, 1.0, positions)
    assert setitem_index_probe_values(layer, label, values) is None


def test_mask_flip_reaches_a_column_that_changes_the_write() -> None:
    """Row 2 holds -1 except in column 2; only flipping that mask entry changes the write."""

    state = torch.full((4, 3), -1.0)
    state[2, 2] = 1.0
    mask = torch.tensor([True, False, False])
    layer = _setitem_layer(state, (torch.tensor(2), mask), -1.0, {(1, 1): "mask"})
    flipped = setitem_index_probe_values(layer, "mask", mask)
    assert flipped is not None and flipped.tolist() == [True, False, True]
    dead = _setitem_layer(torch.full((4, 3), -1.0), (torch.tensor(2), mask), -1.0, {(1, 1): "mask"})
    assert setitem_index_probe_values(dead, "mask", mask) is None


def test_probe_finds_no_move_when_the_destination_already_holds_the_value() -> None:
    destination = torch.full((4, 3), 7.0)
    rows = torch.tensor([2])
    layer = _setitem_layer(destination, (rows, 1), 7.0, {(1, 0): "rows"})
    assert setitem_index_probe_values(layer, "rows", rows) is None


def test_setitem_never_enters_the_index_domain_slot_table() -> None:
    """The degenerate-domain and value-irrelevance exemptions read this table only."""

    rows = torch.tensor([0, 1])
    layer = _setitem_layer(torch.zeros(2, 20), (rows, rows), 1.0, {(1, 0): "rows"})
    assert "__setitem__" not in _INDEX_DOMAIN_ARG_SPECS
    assert _parent_is_index_domain_arg(layer, "rows") is False
    assert layer_has_index_domain_parent(layer) is False
    assert layer_takes_single_entry_probe(layer) is True
    assert _raise_retry_strategies(layer, ["rows"])[-1] == _INDEX_SINGLE_ENTRY_STRATEGY
    assert index_domain_single_entry_values(layer, "rows", rows) is not None


def test_ops_without_a_setitem_index_parent_keep_their_ladders() -> None:
    dest_only = _setitem_layer(torch.zeros(2, 2), (0, 1), 1.0, {0: "dest"})
    assert layer_takes_single_entry_probe(dest_only) is False
    assert _raise_retry_strategies(dest_only, ["dest"]) == ("step_up", "step_down")
    relu = SimpleNamespace(func_name="relu", parent_arg_positions={"args": {0: "x"}})
    assert layer_takes_single_entry_probe(relu) is False
