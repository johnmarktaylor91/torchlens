"""Smoke coverage for glossary v5 locked field additions."""

from io import StringIO
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.data_classes import ParamAccessor
from torchlens.data_classes.layer import OpAccessor
from torchlens.data_classes.module import Module, ModuleCall
from torchlens.data_classes.op import Op


class _NestedModel(nn.Module):
    """Small nested model with parameters and module-call hierarchy."""

    def __init__(self) -> None:
        """Initialize child modules."""

        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
        self.head = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        """Run a nested forward pass."""

        return self.head(self.block(x)) * scale


def _make_trace() -> Any:
    """Return a trace with argument templates enabled."""

    model = _NestedModel()
    return tl.trace(model, torch.randn(1, 4), intervention_ready=True)


@pytest.mark.smoke
def test_trace_call_tree_and_num_modules() -> None:
    """Trace exposes module count and call-tree navigation."""

    trace = _make_trace()
    stream = StringIO()
    calls = list(trace.walk_calls())

    assert trace.num_modules == len(trace.modules)
    assert trace.num_module_calls == len(trace.module_calls)
    assert len(calls) == len(trace.module_calls)
    assert calls[0].call_label == "self:1"
    assert calls[0].max_descendant_depth >= 2

    trace.show_call_tree(file=stream)
    assert "self:1" in stream.getvalue()
    assert "├──" in stream.getvalue() or "└──" in stream.getvalue()


@pytest.mark.smoke
def test_op_input_side_properties() -> None:
    """Op input-side properties resolve parent Ops and saved activations."""

    trace = _make_trace()
    op = next(record for record in trace.ops if record.parents)
    parent = trace.ops[op.parents[0]]

    assert isinstance(op, Op)
    assert isinstance(op.input_ops, OpAccessor)
    assert op.num_inputs == len(op.parents)
    assert op.input_shapes[0] == parent.shape
    assert op.input_dtypes[0] == parent.dtype
    assert op.input_memory >= parent.activation_memory
    assert op.input_activations[0] is not None


@pytest.mark.smoke
def test_module_recursive_params_memory_and_call_tree() -> None:
    """Module exposes address-recursive params, memory, and call-tree navigation."""

    trace = _make_trace()
    root = trace.modules["self"]
    block = trace.modules["block"]
    stream = StringIO()
    block_descendants = list(block.walk_descendants())

    assert isinstance(root, Module)
    assert isinstance(root.recursive_params, ParamAccessor)
    assert root.recursive_param_addresses == [
        "block.0.weight",
        "block.0.bias",
        "head.weight",
        "head.bias",
    ]
    assert root.param_memory == trace.total_param_memory
    assert block.num_recursive_params == 2
    assert block.num_recursive_param_tensors == 2
    assert len(block_descendants) == block.num_descendant_calls
    assert [call.call_label for call in block_descendants] == ["block.0:1", "block.1:1"]
    assert block.num_descendant_calls >= 2
    assert block.max_descendant_depth >= 1
    block.show_call_tree(file=stream)
    assert "block:1" in stream.getvalue()
    assert block.forward_args_template is not None
    assert block.forward_kwargs_template is None
    assert block.backward_duration is None
    assert block.total_backward_duration is None
    assert isinstance(block.total_output_activation_memory, tl.Bytes)
    assert isinstance(block.total_internal_activation_memory, tl.Bytes)


@pytest.mark.smoke
def test_module_call_memory_templates_and_call_tree() -> None:
    """ModuleCall exposes templates, memory quadrants, params, and call-tree navigation."""

    trace = _make_trace()
    call = trace.module_calls["block:1"]
    stream = StringIO()
    descendants = list(call.walk_descendants())

    assert isinstance(call, ModuleCall)
    assert call.forward_args_template is not None
    assert call.forward_kwargs_template is None
    assert call.backward_duration is None
    assert call.output_activation_memory >= 0
    assert call.internal_activation_memory > 0
    assert call.output_gradient_memory == 0
    assert call.internal_gradient_memory == 0
    assert call.autograd_memory >= 0
    assert call.param_memory == trace.modules["block"].param_memory
    assert isinstance(call.param_memory, tl.Bytes)
    assert descendants[0] is call
    assert len(descendants) == call.num_descendant_calls + 1
    assert call.num_descendant_calls >= 2
    assert call.max_descendant_depth >= 1
    call.show_call_tree(show_call_index=False, file=stream)
    assert "block" in stream.getvalue()
    assert "block:1" not in stream.getvalue()
