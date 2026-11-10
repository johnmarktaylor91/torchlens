"""The trace's op list and label dictionaries hold row keys and hand back records.

``torchlens/_data_classes/_op_row_containers.py`` replaces the plain list and
dictionaries of ``Op`` records with containers that store one integer key per
entry. These tests drive the containers with stand-in records and a stand-in
core, so they pin the container contract alone: list and dictionary behaviour,
the side-table layout, the move to a core's tables, copying and pickling.
"""

from __future__ import annotations

import copy
import gc
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes import _op_row_containers as containers
from torchlens._data_classes._state_adapter import plain_state
from torchlens._trace_core.view_tables import ViewTables

# The module's dotted name and its class names: this test file's own name ends
# with the module's bare name, and a pickle of a model defined here carries it.
_CONTAINER_MARKS = (
    b"_data_classes._op_row_containers",
    b"OpSequence",
    b"OpLabelMap",
    b"OpOrderedLabelMap",
    b"OpRows",
)
_PLAIN_TYPES = (list, dict, OrderedDict)
# A loaded file's label dictionaries are both plain dictionaries, as before the row keys.
_LOADED_TYPES = (list, dict, dict)


class _Record:
    """A stand-in op record: a store and a row, like ``Op``'s two slots."""

    __slots__ = ("_core", "_row", "name")

    def __init__(self, store: Any, row: int, name: str) -> None:
        """Bind the record to ``row`` of ``store``."""

        self._core = store
        self._row = row
        self.name = name


class _Core:
    """A stand-in trace core: an op store and the view tables, pinned as during a build."""

    def __init__(self) -> None:
        """Start with empty tables that hold their records strongly."""

        self.ops = object()
        self._views = ViewTables(self, pinned=True)


class _Trace:
    """A stand-in trace: an attribute dictionary the containers hang from."""


def _new_trace() -> Any:
    """Return a stand-in trace with its three empty containers."""

    trace: Any = _Trace()
    holders = containers.op_holders(trace)
    trace._layer_list, trace._layer_dict_main_keys, trace._layer_dict_all_keys = holders
    return trace


def _rows(core: _Core, start: int, count: int) -> list[_Record]:
    """Return ``count`` records on consecutive rows of ``core``'s store."""

    return [_Record(core.ops, row, f"r{row}") for row in range(start, start + count)]


def test_keys_are_store_rows_and_the_side_table_is_a_list_by_row() -> None:
    """A record on the core's store is keyed by its row; the cache slot is that row."""

    trace, core = _new_trace(), _Core()
    trace._trace_core = core
    records = _rows(core, 300, 4)
    for record in records:
        trace._layer_list.append(record)
        trace._layer_dict_all_keys[record.name] = record
    assert trace._layer_list._keys == [300, 301, 302, 303]
    assert trace._layer_list._keys[3] is records[3]._row  # the row integer itself: no new object
    assert core._views.ops[302] is records[2]
    assert core._views.ops[:300] == [None] * 300
    assert core._views.detached == {}
    assert list(trace._layer_list) == records
    assert trace._layer_dict_all_keys["r301"] is records[1]


def test_a_record_off_the_store_gets_the_next_negative_key() -> None:
    """A detached copy (row always 0 of its own store) is keyed below zero."""

    trace, core = _new_trace(), _Core()
    trace._trace_core = core
    records = _rows(core, 0, 3)
    for record in records:
        trace._layer_list.append(record)
    first, second = _Record(object(), 0, "copy1"), _Record(object(), 0, "copy2")
    trace._layer_list.insert(1, first)
    trace._layer_list.append(second)
    trace._layer_dict_all_keys["copy1"] = first
    assert trace._layer_list._keys == [0, -1, 1, 2, -2]
    assert core._views.detached == {-1: first, -2: second}
    assert trace._layer_dict_all_keys._keys == {"copy1": -1}  # one key per record, reused
    assert list(trace._layer_list) == [records[0], first, records[1], records[2], second]
    assert next(i for i, op in enumerate(trace._layer_list) if op is first) == 1


def test_records_held_before_the_core_exists_move_to_it_on_the_next_write() -> None:
    """An append that finds a new core moves every container's keys, itself included."""

    trace, core = _new_trace(), _Core()
    records = _rows(core, 0, 4)
    for record in records[:2]:
        trace._layer_list.append(record)
        trace._layer_dict_all_keys[record.name] = record
    assert trace._layer_list._keys == [-1, -2]
    trace._trace_core = core
    for record in records[2:]:
        trace._layer_list.append(record)
    assert trace._layer_list._keys == [0, 1, 2, 3]
    assert list(trace._layer_list) == records
    assert trace._layer_dict_all_keys._keys == {"r0": 0, "r1": 1}
    assert core._views.ops == records


def test_list_behaviour_matches_a_plain_list() -> None:
    """Indexing, slicing, comparison, reversal and membership answer like a list."""

    trace, core = _new_trace(), _Core()
    trace._trace_core = core
    records = _rows(core, 0, 5)
    trace._layer_list.extend(records)
    sequence = trace._layer_list
    assert len(sequence) == 5 and sequence[0] is records[0] and sequence[-1] is records[4]
    assert sequence[1:3] == records[1:3] and type(sequence[1:3]) is list
    assert sequence == records and records[2] in sequence
    assert list(reversed(sequence)) == records[::-1]
    assert sequence.index(records[3]) == 3
    assert sequence + [1] == [*records, 1] and type(sequence.copy()) is list
    del sequence[0]
    sequence[0] = records[4]
    assert list(sequence) == [records[4], records[2], records[3], records[4]]
    with pytest.raises(IndexError):
        sequence[10]
    with pytest.raises(TypeError):
        hash(sequence)


def test_dictionary_behaviour_matches_a_plain_dictionary() -> None:
    """Lookup, ``get``, views, deletion and the missing-key error answer like a dictionary."""

    trace, core = _new_trace(), _Core()
    trace._trace_core = core
    records = _rows(core, 0, 3)
    mapping = trace._layer_dict_all_keys
    for record in records:
        mapping[record.name] = record
    assert list(mapping) == ["r0", "r1", "r2"] and "r1" in mapping and "zz" not in mapping
    assert (
        mapping.get("zz") is None and mapping.get("zz", 7) == 7 and mapping.get("r2") is records[2]
    )
    assert list(mapping.values()) == records
    assert list(mapping.items()) == [(record.name, record) for record in records]
    assert dict(mapping) == {record.name: record for record in records}
    assert mapping == {record.name: record for record in records}
    with pytest.raises(KeyError) as missing:
        mapping["zz"]
    assert missing.value.args == ("zz",)
    del mapping["r0"]
    assert mapping.pop("r1") is records[1] and list(mapping) == ["r2"]


def test_pickle_state_and_pickles_are_plain_containers_of_the_same_type() -> None:
    """Projection restores ``list``, ``OrderedDict`` and (after a restart) ``dict``."""

    trace, core = _new_trace(), _Core()
    trace._trace_core = core
    containers.restart_op_holders(trace)
    records = _rows(core, 0, 2)
    for record in records:
        trace._layer_list.append(record)
        trace._layer_dict_main_keys[record.name] = record
        trace._layer_dict_all_keys[record.name] = record
    state = containers.project_op_containers(dict(trace.__dict__))
    assert type(state["_layer_list"]) is list and state["_layer_list"] == records
    assert type(state["_layer_dict_main_keys"]) is dict
    assert type(state["_layer_dict_all_keys"]) is OrderedDict
    assert state["_layer_dict_all_keys"]["r1"] is records[1]
    plain = pickle.loads(pickle.dumps(containers.OpSequence(containers.OpRows(), [1, 2])))
    assert type(plain) is list and plain == [1, 2]


def test_restored_plain_containers_are_adopted_after_the_core_is_rebuilt() -> None:
    """Adoption wraps the plain containers and keys them by the rebuilt core's rows."""

    trace: Any = _Trace()
    core = _Core()
    records = _rows(core, 0, 3)
    trace._layer_list = list(records)
    trace._layer_dict_main_keys = {"r0": records[0]}
    trace._layer_dict_all_keys = OrderedDict(r1=records[1])

    def rebuild(restored: Any) -> None:
        """Install the core, as rehydration does."""

        restored._trace_core = core

    containers.adopt_op_containers(trace, rebuild)
    assert trace._layer_list._keys == [0, 1, 2]
    assert core._views.ops == records
    assert trace._layer_dict_all_keys._ordered and not trace._layer_dict_main_keys._ordered
    assert trace._layer_dict_main_keys["r0"] is records[0]


def test_deepcopy_follows_the_memo_to_the_copied_trace_and_its_records() -> None:
    """A fork's field copy yields containers over the fork's records, then binds to its core."""

    trace, core = _new_trace(), _Core()
    trace._trace_core = core
    records = _rows(core, 0, 3)
    for record in records:
        trace._layer_list.append(record)
        trace._layer_dict_all_keys[record.name] = record
    fork: Any = _Trace()
    fork_core = _Core()
    shells = [_Record(fork_core.ops, record._row, "f" + record.name) for record in records]
    memo: dict[Any, Any] = {id(trace): fork}
    memo.update({id(record): shell for record, shell in zip(records, shells, strict=True)})
    fork._layer_list = copy.deepcopy(trace._layer_list, memo)
    fork._layer_dict_all_keys = copy.deepcopy(trace._layer_dict_all_keys, memo)
    assert list(fork._layer_list) == shells and list(trace._layer_list) == records
    assert fork._layer_list._owner is fork._layer_dict_all_keys._owner
    assert not fork_core._views.ops and not fork_core._views.detached
    containers.bind_op_containers(fork, fork_core)
    assert fork._trace_core is fork_core and fork._layer_list._keys == [0, 1, 2]
    assert fork_core._views.ops == shells
    assert fork._layer_dict_all_keys["r1"] is shells[1]
    assert core._views.ops == records


def test_the_list_still_reads_after_the_core_is_dropped() -> None:
    """A cleaned trace drops its core; the list keeps handing back its records."""

    trace, core = _new_trace(), _Core()
    trace._trace_core = core
    records = _rows(core, 0, 3)
    trace._layer_list.extend(records)
    del trace._trace_core, core
    gc.collect()
    assert list(trace._layer_list) == records


def test_restart_and_clear_give_a_new_list_object() -> None:
    """The labeling restart and the backend clear replace the list, not empty it in place."""

    trace, core = _new_trace(), _Core()
    trace._trace_core = core
    trace._layer_list.extend(_rows(core, 0, 2))
    before = trace._layer_list
    all_keys = trace._layer_dict_all_keys
    all_keys["r0"] = before[0]
    containers.restart_op_holders(trace)
    assert trace._layer_list is not before and len(trace._layer_list) == 0 and len(before) == 2
    assert trace._layer_dict_all_keys is all_keys and len(all_keys) == 1
    restarted = trace._layer_list
    containers.clear_op_holders(trace)
    assert trace._layer_list is not restarted and len(all_keys) == 0


class _Tiny(nn.Module):
    """A linear layer and a relu: the smallest model with several ops."""

    def __init__(self) -> None:
        """Build the one layer."""

        super().__init__()
        self.fc = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layer, then relu."""

        return torch.relu(self.fc(x))


def _names_containers(data: bytes) -> bool:
    """Return whether ``data`` names the row-key container module or one of its classes."""

    return any(mark in data for mark in _CONTAINER_MARKS)


def _container_types(state: Any) -> tuple[type, type, type]:
    """Return the types of the three op containers in one state dictionary."""

    return (
        type(state["_layer_list"]),
        type(state["_layer_dict_main_keys"]),
        type(state["_layer_dict_all_keys"]),
    )


def test_pickle_state_pickles_and_saved_files_keep_the_plain_container_types(
    tmp_path: Path,
) -> None:
    """Pickle state, a pickle round trip and a save and load hold list, dict and OrderedDict."""

    trace = tl.trace(_Tiny(), torch.ones(1, 3))
    assert type(trace._layer_list) is containers.OpSequence
    assert type(trace._layer_dict_main_keys) is containers.OpLabelMap
    assert type(trace._layer_dict_all_keys) is containers.OpOrderedLabelMap
    labels = [op.label for op in trace._layer_list]
    assert _container_types(trace.__getstate__()) == _PLAIN_TYPES
    assert _container_types(plain_state(trace)) == _PLAIN_TYPES

    payload = pickle.dumps(trace)
    assert not _names_containers(payload)
    restored = pickle.loads(payload)
    assert _container_types(restored.__getstate__()) == _PLAIN_TYPES
    assert [op.label for op in restored._layer_list] == labels

    bundle = tmp_path / "plain_containers.tlspec"
    trace.save(bundle)
    files = [bundle] if bundle.is_file() else [path for path in bundle.rglob("*") if path.is_file()]
    assert files and not any(_names_containers(path.read_bytes()) for path in files)
    loaded = tl.load(bundle)
    assert type(loaded._layer_dict_all_keys) is containers.OpLabelMap
    assert _container_types(loaded.__getstate__()) == _LOADED_TYPES
    assert _container_types(plain_state(loaded)) == _LOADED_TYPES
    assert [op.label for op in loaded._layer_list] == labels
