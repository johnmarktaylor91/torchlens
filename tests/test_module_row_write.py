"""``Module`` and ``ModuleCall`` are written as one row, in the order of their field tables.

The constructors hand ``write_row`` a list with one value per stored field.
A value in the wrong position would read back under another field's name, so
every field is checked by name against the value the constructor was given.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from torchlens._data_classes._module_facade_fields import (
    _MODULE_CALL_STORED_FIELDS,
    _MODULE_STORED_FIELDS,
)
from torchlens._data_classes.module import Module, ModuleCall, ModuleCallAccessor
from torchlens._data_classes.param import ParamAccessor
from torchlens._quantities import Duration
from torchlens._trace_core.record_rows import record_state_items, write_row


def _distinct_arguments(cls: type, fixed: dict[str, Any]) -> dict[str, Any]:
    """Return one distinct value per constructor argument of ``cls``."""

    names = [name for name in inspect.signature(cls.__init__).parameters if name != "self"]
    given = {name: [f"{cls.__name__}.{name}"] for name in names}
    given["_source_trace"] = None
    given.update(fixed)
    return given


def _stored(record: Any) -> dict[str, Any]:
    """Return a record's stored fields by name."""

    declared = type(record)._TL_LAYOUT.fid_by_name
    return {name: value for name, value in record_state_items(record) if name in declared}


def test_module_call_row_matches_its_field_table() -> None:
    given = _distinct_arguments(
        ModuleCall, {"forward_args": (1, 2, 3), "forward_kwargs": {"a": 1}, "forward_duration": 0.5}
    )
    call = ModuleCall(**given)
    stored = _stored(call)
    assert tuple(stored) == _MODULE_CALL_STORED_FIELDS
    derived = {
        "input_ops": list(given["input_layers"]),
        "num_forward_pos_args": 3,
        "num_forward_kwargs": 1,
        "num_forward_args_total": 4,
        "forward_args_summary": "",
        "forward_kwargs_summary": "",
        "forward_duration": Duration(0.5),
    }
    for name in _MODULE_CALL_STORED_FIELDS:
        expected = derived[name] if name in derived else given[name]
        assert stored[name] == expected, name
        if name not in derived:
            assert stored[name] is given[name], name
    assert call.forward_args_template is given["forward_args_template"]
    assert call.forward_kwargs_template is given["forward_kwargs_template"]
    assert list(call.__dict__)[:2] == ["_tl_core", "_tl_row"]


def test_module_call_defaults_fill_the_row() -> None:
    call = ModuleCall("enc", 1, "enc:1", ["relu_1:1"], ["input_1"], ["relu_1"])
    stored = _stored(call)
    assert tuple(stored) == _MODULE_CALL_STORED_FIELDS
    assert stored["all_addresses"] == ["enc"]
    assert (
        stored["output_ops"] == ["relu_1"] and stored["output_ops"] is not stored["output_layers"]
    )
    assert stored["output_paths"] == () and stored["forward_pre_hook_effects"] == ()
    assert stored["forward_arg_names"] == [] and stored["code_context"] == []
    assert stored["module_call_stack"] == [] and stored["call_children"] == []
    assert stored["num_forward_args_total"] == 0


def test_module_row_matches_its_field_table() -> None:
    first = ModuleCall("enc", 1, "enc:1", ["a:1"], ["in_a"], ["out_a"], output_ops=["out_a:1"])
    second = ModuleCall("enc", 2, "enc:2", ["a:2"], ["in_a", "in_b"], ["out_b"])
    params = ParamAccessor({})
    given = _distinct_arguments(Module, {"ops": {1: first, 2: second}, "params": params})
    module = Module(**given)
    stored = _stored(module)
    assert tuple(stored) == _MODULE_STORED_FIELDS
    assert isinstance(stored["ops"], ModuleCallAccessor)
    assert list(stored["ops"].values()) == [first, second]
    derived = {
        "input_ops": ["in_a", "in_b"],
        "input_layers": ["in_a", "in_b"],
        "output_ops": ["out_a:1", "out_b"],
        "output_layers": ["out_a", "out_b"],
        "_buffer_accessor": None,
    }
    for name in _MODULE_STORED_FIELDS:
        if name == "ops":
            continue
        if name in derived:
            assert stored[name] == derived[name], name
        else:
            assert stored[name] is given[name], name


def test_module_defaults_fill_the_row() -> None:
    module = Module("enc")
    stored = _stored(module)
    assert tuple(stored) == _MODULE_STORED_FIELDS
    assert stored["all_addresses"] == ["enc"]
    assert isinstance(stored["params"], ParamAccessor) and len(stored["params"]) == 0
    for name in ("address_children", "call_children", "call_labels", "layer_labels"):
        assert stored[name] == [], name
    assert stored["custom_attributes"] == {} and stored["custom_methods"] == []


def test_write_row_keeps_a_row_the_record_already_has() -> None:
    call = ModuleCall("enc", 1, "enc:1", [], [], [])
    store = call.__dict__["_tl_core"]
    cells = [f"value {index}" for index in range(len(_MODULE_CALL_STORED_FIELDS))]
    write_row(call, list(cells))
    assert call.__dict__["_tl_core"] is store
    assert list(_stored(call).values()) == cells


@pytest.mark.parametrize("cls", [Module, ModuleCall])
def test_row_builders_cover_every_stored_field(cls: type) -> None:
    fields = _MODULE_STORED_FIELDS if cls is Module else _MODULE_CALL_STORED_FIELDS
    assert cls._TL_LAYOUT.n_fields == len(fields)
