"""Sanctioned empty containers stay default-omitted at write (capture-core part IA).

The emit, the ingest join and the rename pass leave an empty sanctioned
(poolable) container cell default-omitted instead of writing a fresh empty
container; the freeze settles it onto the shared pooled empty cell, as the
container pooling would have done for a written one.

Two contracts are pinned here:

* each omitted field's registered default is the exact empty value (type and
  value) its writer used to write, so a settled cell reads as before;
* a writer that hands several cells one shared object keeps writing it: the
  sibling outputs of one call (an ordinary multi-output call, and a boundary
  op of a two-output foreign kernel) keep their shared ``func_rng_states``
  dict, so the alias census still keeps it out of the pool.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.op import _OP_STORE_LAYOUT, _POOLED_CONTAINER_FIELDS
from torchlens._ir.op_record import FACET_DEFAULTS
from torchlens._ir.op_record_scatter import _write_function_ref_cells
from torchlens._postprocess._materialize import _empty_module_input_fields
from torchlens._postprocess._row_ingest import _OMITTED_EMPTY_JOIN_CELLS
from torchlens._trace_core.cell_states import LazyCell, default_answer
from torchlens._trace_core.op_store import PooledCell
from torchlens.backends.torch import _ops_row_cells
from torchlens.backends.torch._function_boundary import FOREIGN_FUNCTION_FUNC_NAME
from torchlens.options import CaptureOptions


def _function_ref_cells() -> dict[str, Any]:
    """Return the cells the shared function-ref writer fills for a call with no RNG state."""

    ref = SimpleNamespace(
        func=None,
        func_id=None,
        func_call_id=0,
        func_name="f",
        func_qualname="f",
        code_context=(),
        func_duration=0,
        flops_forward=0,
        flops_backward=0,
        func_rng_states={},  # the wrappers' snapshot when RNG saving is off
        func_autocast_state={},
        arg_names=(),
        num_args_total=0,
        num_pos_args=0,
        num_kwargs=0,
        non_tensor_pos_args=(),
        non_tensor_kwargs={},
        func_non_tensor_args=(),
        is_inplace=False,
        func_config={},
    )
    cells: dict[str, Any] = {}
    _write_function_ref_cells(cells, _ops_row_cells._NAME_KEYS, ref)
    return cells


#: Each omitted field and the empty value its writer wrote before omission.
_WRITER_EMPTY_VALUES: dict[str, Callable[[], Any]] = {
    # emit: ``_fill_facets``
    "annotations": lambda: dict(FACET_DEFAULTS["annotations"]().annotations),
    "edits": lambda: [],
    # emit: ``_write_function_ref_cells``
    "var_names": lambda: _function_ref_cells()["var_names"],
    "func_rng_states": lambda: _function_ref_cells()["func_rng_states"],
    # ingest join (``_apply_join_cells``): a merged dict and a comprehension
    "out_versions_by_child": lambda: {**{}, **{}},
    "parent_param_ops": lambda: {},
    "module_entry_arg_keys": lambda: _empty_module_input_fields()["module_entry_arg_keys"],
    # ingest seed
    "_grad_records": lambda: [],
    # rename pass, empty arm
    "conditional_arm_children": lambda: {},
    "conditional_elif_children": lambda: {},
}


def test_every_sanctioned_empty_field_is_covered() -> None:
    """The table covers every omitted field, and every one is a sanctioned field."""

    omitted = {name for name, _ in _ops_row_cells._OMITTED_EMPTY_CELLS}
    omitted |= {name for name, _ in _OMITTED_EMPTY_JOIN_CELLS}
    omitted |= {"_grad_records", "module_entry_arg_keys"}
    omitted |= set(_ops_row_cells._EMPTY_DICT_CELLS)
    assert omitted == set(_WRITER_EMPTY_VALUES)
    assert omitted <= _POOLED_CONTAINER_FIELDS


@pytest.mark.parametrize("name", sorted(_WRITER_EMPTY_VALUES))
def test_default_equals_writer_value(name: str) -> None:
    """The registered default answers the writer's empty value: same type, same value."""

    fields = _ops_row_cells._default_fields(_OP_STORE_LAYOUT)
    marker = fields.cell(name)
    default = default_answer(marker)
    written = _WRITER_EMPTY_VALUES[name]()
    assert type(default) is type(written)
    assert default == written
    assert not written
    if isinstance(written, defaultdict):
        assert default.default_factory is written.default_factory
    # A mutable default is built per row, never shared.
    assert marker.make and default_answer(marker) is not default


def test_join_omission_classes_match_defaults() -> None:
    """The ingest join skips only values of the exact class its default answers."""

    fields = _ops_row_cells._default_fields(_OP_STORE_LAYOUT)
    for name, cls in _OMITTED_EMPTY_JOIN_CELLS:
        assert type(default_answer(fields.cell(name))) is cls


class _SplitModel(nn.Module):
    """A three-way split (one call, three outputs) between two single-output ops."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(6, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b, c = self.proj(x).split(2, dim=-1)
        return torch.relu(a) + b * c


@pytest.fixture(scope="module")
def split_trace() -> Iterator[Any]:
    """Yield one split-model trace (RNG saving off) and release it after the module."""

    torch.manual_seed(0)
    trace = tl.trace(_SplitModel(), torch.randn(2, 6))
    try:
        yield trace
    finally:
        trace.cleanup()


def _raw(op: Any, name: str) -> Any:
    """Return ``op``'s raw stored cell ``name`` (no default answered, nothing hydrated)."""

    store = object.__getattribute__(op, "_core")
    row = object.__getattribute__(op, "_row")
    return store.cell_get(row, store.layout.fid_by_name[name], True)


def _call_groups(trace: Any) -> dict[int, list[Any]]:
    """Return the trace's ops grouped by function call id."""

    groups: dict[int, list[Any]] = defaultdict(list)
    for op in trace._layer_list:
        groups[op.func_call_id].append(op)
    return groups


def test_single_output_empty_cells_settle_pooled(split_trace: Any) -> None:
    """A single-output op's omitted empty cells hold the shared pooled cell."""

    singles = [
        op
        for ops in _call_groups(split_trace).values()
        if len(ops) == 1
        for op in ops
        if not (op.is_input or op.is_output or op.in_multi_output)
    ]
    assert singles
    for name in ("func_rng_states", "var_names", "edits", "annotations", "out_versions_by_child"):
        cells = [_raw(op, name) for op in singles]
        assert all(cell.__class__ is PooledCell for cell in cells), (name, cells)
        assert len({id(cell) for cell in cells}) == 1, name
    for op in singles:
        assert op.func_rng_states == {} and type(op.func_rng_states) is dict
        assert op.var_names == [] and op.edits == [] and op.annotations == {}


def test_multi_output_call_keeps_its_shared_rng_dict(split_trace: Any) -> None:
    """The split's three outputs keep one shared ``func_rng_states`` dict, unpooled."""

    siblings = [ops for ops in _call_groups(split_trace).values() if len(ops) == 3]
    assert len(siblings) == 1, [op.label for ops in siblings for op in ops]
    cells = [_raw(op, "func_rng_states") for op in siblings[0]]
    assert all(cell.__class__ is dict and not cell for cell in cells), cells
    assert cells[0] is cells[1] is cells[2]
    assert all(cell.__class__ is not LazyCell for cell in cells)


def test_boundary_siblings_keep_their_shared_rng_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """A two-output boundary op's outputs share the boundary's RNG dict; it stays unpooled."""

    from tests.test_foreign_kernel_dependence import FlashAttnFunc, _KernelModel
    from torchlens._utils import _torch_compat as tc

    monkeypatch.setattr(tc, "HAS_FLASH_ATTENTION_4_INTERFACE", True)
    torch.manual_seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(
            _KernelModel(FlashAttnFunc), torch.randn(2, 4), capture=CaptureOptions(seed=0)
        )
    try:
        ops = [op for op in trace._layer_list if op.func_name == FOREIGN_FUNCTION_FUNC_NAME]
        assert len(ops) == 2 and len({op.func_call_id for op in ops}) == 1
        cells = [_raw(op, "func_rng_states") for op in ops]
        assert cells[0].__class__ is dict and not cells[0], cells
        assert cells[0] is cells[1]
    finally:
        trace.cleanup()
