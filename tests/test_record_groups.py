"""Op and Layer concern groups (fields.1): group reads and typed refusals of the flat spellings.

The cost group is the first applied group: ``op.cost.flops_forward`` reads the
same persisted column ``op.flops_forward`` used to, derived members compute on
read from the group, and every former flat spelling refuses typed
(``facade_redirect``) with a remedy naming the new spelling, on a live trace
and after a pickle round trip.
"""

from __future__ import annotations

# record-groups: keep-flat
import pickle
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import FacadeTeachingError, InvalidArgumentError
from torchlens._utils._record_group_table import (
    ACTIVE_GROUPS,
    LAYER_MEMBER_MOVES,
    LAYER_MOVED_MEMBERS,
    OP_MEMBER_MOVES,
    OP_MOVED_MEMBERS,
    record_member,
)

_OP_COST_ROWS = [row for row in OP_MEMBER_MOVES if row[1] == "cost"]
_LAYER_COST_ROWS = [row for row in LAYER_MEMBER_MOVES if row[1] == "cost"]
_OP_ACTIVE_ROWS = [row for row in OP_MEMBER_MOVES if row[1] in ACTIVE_GROUPS]
_LAYER_ACTIVE_ROWS = [row for row in LAYER_MEMBER_MOVES if row[1] in ACTIVE_GROUPS]


@pytest.fixture(scope="module")
def trace() -> Iterator[tl.Trace]:
    """A small two-layer trace with one repeated module call (a multi-pass layer)."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    captured = tl.trace(model, torch.randn(2, 3))
    yield captured
    captured.cleanup()


def test_cost_group_is_applied() -> None:
    """The cost rows are live in the refusal tables of both records."""

    assert {"cost", "control_flow", "call", "source", "autograd", "module_context"} <= ACTIVE_GROUPS
    assert len(_OP_COST_ROWS) == 16
    assert len(_LAYER_COST_ROWS) == 17
    assert {old for old, _, _ in _OP_COST_ROWS} <= set(OP_MOVED_MEMBERS)
    assert {old for old, _, _ in _LAYER_COST_ROWS} <= set(LAYER_MOVED_MEMBERS)


def test_group_classes_are_the_public_types(trace: tl.Trace) -> None:
    """``op.cost`` and ``layer.cost`` are ``tl.types.OpCost`` and ``tl.types.LayerCost``."""

    op = trace.ops["linear_1_1"]
    layer = trace.layers["linear_1"]
    assert type(op.cost) is tl.types.OpCost
    assert type(layer.cost) is tl.types.LayerCost
    assert type(op.control_flow) is tl.types.OpControlFlow
    assert type(layer.control_flow) is tl.types.LayerControlFlow
    assert type(op.call) is tl.types.OpCall
    assert type(layer.call) is tl.types.LayerCall
    assert type(op.source) is tl.types.OpSource
    assert type(layer.source) is tl.types.LayerSource
    assert type(op.autograd) is tl.types.OpAutograd
    assert type(layer.autograd) is tl.types.LayerAutograd
    assert type(op.module_context) is tl.types.OpModuleContext
    assert type(layer.module_context) is tl.types.LayerModuleContext
    assert "flops_forward" in dir(op.cost)
    assert "total_flops_forward" in dir(layer.cost)
    assert "OpCost" in repr(op.cost)


@pytest.mark.parametrize(("old", "group", "leaf"), _OP_ACTIVE_ROWS)
def test_op_flat_spelling_refuses_typed(trace: tl.Trace, old: str, group: str, leaf: str) -> None:
    """Each former flat Op spelling refuses typed, naming ``op.<group>.<leaf>``."""

    op = trace.ops["linear_1_1"]
    with pytest.raises(FacadeTeachingError) as excinfo:
        getattr(op, old)
    assert excinfo.value.fields["code"] == "facade_redirect"
    assert f"op.{group}.{leaf}" in str(excinfo.value)
    assert not hasattr(op, old)
    getattr(getattr(op, group), leaf)


@pytest.mark.parametrize(("old", "group", "leaf"), _LAYER_ACTIVE_ROWS)
def test_layer_flat_spelling_refuses_typed(
    trace: tl.Trace, old: str, group: str, leaf: str
) -> None:
    """Each former flat Layer spelling refuses typed, naming ``layer.<group>.<leaf>``."""

    layer = trace.layers["linear_1"]
    with pytest.raises(FacadeTeachingError) as excinfo:
        getattr(layer, old)
    assert excinfo.value.fields["code"] == "facade_redirect"
    assert f"layer.{group}.{leaf}" in str(excinfo.value)
    assert not hasattr(layer, old)
    getattr(getattr(layer, group), leaf)


def test_layer_flat_spelling_refuses_writes(trace: tl.Trace) -> None:
    """Writing or deleting a former flat Layer spelling refuses instead of shadowing it."""

    layer = trace.layers["linear_1"]
    with pytest.raises(FacadeTeachingError):
        layer.flops_forward = 1
    with pytest.raises(FacadeTeachingError):
        del layer.flops_forward
    with pytest.raises(FacadeTeachingError):
        layer.is_terminal_bool = True
    assert layer.cost.flops_forward == trace.ops["linear_1_1"].cost.flops_forward


def test_op_cost_reads_the_persisted_columns(trace: tl.Trace) -> None:
    """Stored cost members read the same column the flat spelling read."""

    op = trace.ops["linear_1_1"]
    assert int(op.cost.flops_forward) == 2 * 2 * 3 * 4 + 2 * 4
    assert op.cost.flops_forward == op._cost_flops_forward
    assert op.cost.duration == op._cost_duration
    assert int(op.cost.activation_memory) == 2 * 4 * 4
    assert int(op.cost.macs_forward) == 2 * 3 * 4
    assert op.cost.flops_total == (op.cost.flops_forward or 0) + (op.cost.flops_backward or 0)


def test_layer_cost_matches_its_ops(trace: tl.Trace) -> None:
    """Layer cost members agree with the first pass and the per-pass sums."""

    layer = trace.layers["linear_1"]
    ops = list(layer.ops.values())
    assert layer.cost.flops_forward == ops[0].cost.flops_forward
    assert layer.cost.macs_forward == ops[0].cost.macs_forward
    assert int(layer.cost.total_flops_forward) == sum(int(op.cost.flops_forward) for op in ops)
    assert (
        layer.cost.total_flops == layer.cost.total_flops_forward + layer.cost.total_flops_backward
    )
    assert layer.cost.total_macs is None
    assert layer.cost.duration == ops[0].cost.duration


def test_groups_survive_a_pickle_round_trip(trace: tl.Trace) -> None:
    """A reloaded trace serves the same group values and still refuses the flat spellings."""

    loaded = pickle.loads(pickle.dumps(trace))
    try:
        for label in ("linear_1_1", "relu_1_2"):
            assert loaded.ops[label].cost.flops_forward == trace.ops[label].cost.flops_forward
        layer = loaded.layers["linear_1"]
        assert layer.cost.flops_forward == trace.layers["linear_1"].cost.flops_forward
        assert layer.cost.activation_memory == trace.layers["linear_1"].cost.activation_memory
        with pytest.raises(FacadeTeachingError):
            _ = layer.flops_forward
        with pytest.raises(FacadeTeachingError):
            _ = loaded.ops["linear_1_1"].flops_forward
        with pytest.raises(FacadeTeachingError):
            _ = loaded.ops["linear_1_1"].conditional_depth
    finally:
        loaded.cleanup()


def test_record_member_follows_moves(trace: tl.Trace) -> None:
    """``record_member`` reads a former name through its group on an Op or Layer only."""

    op = trace.ops["linear_1_1"]
    layer = trace.layers["linear_1"]
    assert record_member(op, "func_duration") == op.cost.duration
    assert record_member(layer, "total_flops_total") == layer.cost.total_flops
    assert record_member(op, "label") == op.label

    class _Other:
        flops_forward = 7

    assert record_member(_Other(), "flops_forward") == 7
    assert record_member(object(), "flops_forward", None) is None


class _Branchy(nn.Module):
    """One data-dependent ``if`` so control-flow members carry values."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        if y.sum() > 0:
            return torch.relu(y)
        return -y


def test_control_flow_group_reads_the_branch() -> None:
    """The control-flow group serves the terminal bool and the arm members of a branch."""

    torch.manual_seed(0)
    captured = tl.trace(_Branchy(), torch.ones(1, 3))
    try:
        bools = [op for op in captured.ops if op.control_flow.is_terminal_bool]
        assert len(bools) == 1
        flow = bools[0].control_flow
        assert flow.is_scalar_bool is True
        assert flow.bool_value is bool(flow.bool_value)
        assert flow.is_terminal_conditional_bool is True
        assert flow.is_in_conditional_evaluation is True
        assert flow.depth == 1
        layer = captured.layers[bools[0].layer_label]
        assert layer.control_flow.terminal_bool_for == layer.__dict__["terminal_bool_for"]
        assert layer.control_flow.branch_stack == bools[0].control_flow.branch_stack
        body = [op for op in captured.ops if op.control_flow.is_in_conditional_body]
        assert body
        assert all(op.control_flow.is_in_conditional for op in body)
        with pytest.raises(FacadeTeachingError):
            _ = bools[0].is_terminal_bool
    finally:
        captured.cleanup()


def test_call_group_reads_the_call(trace: tl.Trace) -> None:
    """The call group serves the stored call facts and the Layer's single-pass members."""

    op = trace.ops["linear_1_1"]
    layer = trace.layers["linear_1"]
    assert op.call.num_pos_args == op._call_num_pos_args
    assert isinstance(op.call.call_id, int)
    assert op.call.config == op._call_config
    assert layer.call.num_pos_args == op.call.num_pos_args
    assert layer.call.rng_states == op.call.rng_states
    assert layer.call.call_id == op.call.call_id
    assert set(layer.call.parent_arg_positions) <= {"args", "kwargs"}
    with pytest.raises(FacadeTeachingError):
        _ = op.func_call_id
    with pytest.raises(FacadeTeachingError):
        _ = layer.saved_args


def test_source_and_autograd_groups_read_the_op(trace: tl.Trace) -> None:
    """The source and autograd groups serve the stored columns on Op and the first pass on Layer."""

    op = trace.ops["linear_1_1"]
    layer = trace.layers["linear_1"]
    assert op.source.frames is op._source_frames
    assert op.source.func_qualname == op._source_func_qualname
    assert layer.source.frames == op.source.frames
    assert layer.source.fx_path == op.source.fx_path
    assert op.autograd.grad_fn_class_name == "AddmmBackward0"
    assert layer.autograd.grad_fn_class_name == op.autograd.grad_fn_class_name
    assert op.autograd.num_autograd_tensors == op._autograd_num_autograd_tensors
    with pytest.raises(FacadeTeachingError):
        _ = op.code_context
    with pytest.raises(FacadeTeachingError):
        _ = layer.grad_fn_class_name


class _Nested(nn.Module):
    """A Linear inside a Sequential, so module-context members carry values."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Linear(3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.block(x))


def test_module_context_group_reads_the_nesting() -> None:
    """The module-context group serves the stored columns and derives addresses and depth."""

    torch.manual_seed(0)
    captured = tl.trace(_Nested(), torch.ones(1, 3))
    try:
        op = captured.ops["linear_1_1"]
        context = op.module_context
        assert context.is_atomic_module is True
        assert context.atomic_module_call == "block.0:1"
        assert context.atomic_module_address == "block.0"
        assert context.atomic_module is captured.modules["block.0"]
        assert context.call_depth == len(context.call_stack) == len(op.modules)
        assert context.is_module_output is True
        source = captured.ops["input_1"].module_context
        assert source.is_module_input is True
        assert "block" in source.input_to_modules
        layer = captured.layers["linear_1"]
        assert layer.module_context.call_depth == context.call_depth
        assert layer.module_context.is_module_input == context.is_module_input
        assert layer.module_context.output_of_modules == layer.__dict__["output_of_modules"]
        with pytest.raises(FacadeTeachingError):
            _ = op.module_call_depth
        with pytest.raises(FacadeTeachingError):
            _ = layer.output_of_modules
    finally:
        captured.cleanup()


class _Twice(nn.Module):
    """One module called twice, so its layer has two passes."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.fc(x))


def test_multi_pass_layer_group_refuses_per_pass_members() -> None:
    """A two-pass Layer refuses a per-pass group member as ambiguous, naming the Op spelling."""

    torch.manual_seed(0)
    captured = tl.trace(_Twice(), torch.ones(1, 3))
    try:
        layer = next(lay for lay in captured.layers if lay.num_passes > 1)
        assert layer.num_passes == 2
        with pytest.raises(InvalidArgumentError, match=r"\.call\.call_id"):
            _ = layer.call.call_id
        with pytest.raises(InvalidArgumentError, match=r"\.call\.rng_states"):
            _ = layer.call.rng_states
        with pytest.raises(InvalidArgumentError, match=r"\.source\.frames"):
            _ = layer.source.frames
    finally:
        captured.cleanup()
