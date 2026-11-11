"""Empty container cells the capture leaves default-omitted read back as the writer's value.

The row-direct emit and the row ingest skip writing an empty, freshly built
container into the cells listed in ``_OMIT_EMPTY_LIST_CELLS``,
``_OMIT_EMPTY_DICT_CELLS`` and ``_OMIT_EMPTY_JOIN_FIELDS``; the freeze settles
the registered default instead. That is exact only when the default equals the
value the writer used to write, in type and value, and is a fresh object per
row. These tests pin both, per field.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.op import _OP_STORE_LAYOUT
from torchlens._ir.op_record_scatter import _write_function_ref_cells
from torchlens._postprocess._materialize import _empty_module_output_fields
from torchlens._postprocess._row_ingest import _OMIT_EMPTY_JOIN_FIELDS
from torchlens._trace_core.cell_states import STATE_DEFAULT, LazyCell
from torchlens.backends.torch._ops_row_cells import (
    _CALL_OMIT_EMPTY_CELLS,
    _OMIT_EMPTY_DICT_CELLS,
    _OMIT_EMPTY_LIST_CELLS,
    _plan_for,
)

_OMITTED = (*_OMIT_EMPTY_LIST_CELLS, *_OMIT_EMPTY_DICT_CELLS, *_OMIT_EMPTY_JOIN_FIELDS)


def _empty_function_ref() -> Any:
    """Return a function-call ref stand-in whose container facts are all empty."""

    return SimpleNamespace(
        func=None,
        func_id=0,
        func_call_id=0,
        func_name="relu",
        func_qualname="relu",
        code_context=(),
        func_duration=0,
        flops_forward=None,
        flops_backward=None,
        func_rng_states={},
        func_autocast_state={},
        arg_names=(),
        num_args_total=1,
        num_pos_args=1,
        num_kwargs=0,
        non_tensor_pos_args=(),
        non_tensor_kwargs={},
        func_non_tensor_args=(),
        is_inplace=False,
        func_config={},
    )


def _writer_values() -> dict[str, Any]:
    """Return the value each removed writer wrote for an op with nothing to record."""

    written: dict[str, Any] = {}
    _write_function_ref_cells(
        written, {name: name for name in _CALL_OMIT_EMPTY_CELLS}, _empty_function_ref()
    )
    values = {name: written[name] for name in ("non_tensor_kwargs", "func_config")}
    # ``_fill_core_graph`` builds ``internal_source_parents`` as a fresh list and
    # the join builds the three param cells as list comprehensions.
    values["internal_source_parents"] = []
    values["parent_params"] = []
    values["_param_barcodes"] = []
    values["param_shapes"] = []
    values["output_of_modules"] = _empty_module_output_fields()["output_of_modules"]
    return values


@pytest.mark.parametrize("name", sorted(set(_OMITTED)))
def test_registered_default_equals_writer_value(name: str) -> None:
    """The template's default for an omitted cell is a fresh copy of the writer's value."""

    template = _plan_for(_OP_STORE_LAYOUT).template
    cell = template[_OP_STORE_LAYOUT.fid_by_name[name]]
    assert cell.__class__ is LazyCell and cell.state == STATE_DEFAULT and cell.make
    expected = _writer_values()[name]
    first, second = cell.source(), cell.source()
    assert type(first) is type(expected)
    assert first == expected
    assert first is not second


class _ParamFree(nn.Module):
    """Ops with no parameters, no kwargs and no module nesting below the root."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Chain three elementwise calls."""

        return torch.tanh(torch.relu(x) + 1)


@pytest.mark.parametrize(
    "model", [_ParamFree(), nn.Sequential(nn.Linear(4, 4), nn.ReLU())], ids=["free", "linear"]
)
def test_settled_cells_are_fresh_per_row(model: nn.Module) -> None:
    """After the freeze every omitted cell of every op holds its own container."""

    torch.manual_seed(0)
    trace = tl.trace(model, torch.randn(2, 4))
    store = trace.__dict__["_trace_core"].ops
    fid_by_name = _OP_STORE_LAYOUT.fid_by_name
    expected = _writer_values()
    for name in sorted(set(_OMITTED)):
        fid = fid_by_name[name]
        empties = []
        for row in range(len(store)):
            value = store.cell_get(row, fid)
            assert type(value) is type(expected[name]), (name, row, type(value))
            if not value:
                empties.append(value)
        assert len({id(value) for value in empties}) == len(empties), name
