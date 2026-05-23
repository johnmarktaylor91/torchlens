"""Smoke coverage for glossary v5 locked field additions."""

from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.data_classes import CallTreeNode, ParamAccessor
from torchlens.data_classes.layer_log import OpAccessor
from torchlens.data_classes.module_log import Module, ModuleCall
from torchlens.data_classes.op_log import Op


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
    """Trace exposes module count and full call tree."""

    trace = _make_trace()

    assert trace.num_modules == len(trace.modules)
    assert isinstance(trace.call_tree, CallTreeNode)
    assert trace.call_tree.call.call_label == "self:1"
    assert trace.call_tree.children


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
    assert op.input_memory >= parent.memory
    assert op.input_activations[0] is not None


@pytest.mark.smoke
def test_module_recursive_params_memory_and_call_tree() -> None:
    """Module exposes address-recursive params, memory, and call-tree metrics."""

    trace = _make_trace()
    root = trace.modules["self"]
    block = trace.modules["block"]

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
    assert isinstance(block.call_tree, CallTreeNode)
    assert block.num_descendant_calls >= 2
    assert block.max_descendant_depth >= 1
    assert block.forward_args_template is not None
    assert block.forward_kwargs_template is None
    assert block.backward_duration is None
    assert block.total_backward_duration is None
    assert isinstance(block.total_output_activation_memory_str, str)
    assert isinstance(block.total_internal_activation_memory_str, str)


@pytest.mark.smoke
def test_module_call_memory_templates_and_call_tree() -> None:
    """ModuleCall exposes templates, memory quadrants, params, and call-tree metrics."""

    trace = _make_trace()
    call = trace.module_calls["block:1"]

    assert isinstance(call, ModuleCall)
    assert call.forward_args_template is not None
    assert call.forward_kwargs_template is None
    assert call.backward_duration is None
    assert call.backward_duration_str is None
    assert call.output_activation_memory >= 0
    assert call.internal_activation_memory > 0
    assert call.output_gradient_memory == 0
    assert call.internal_gradient_memory == 0
    assert call.autograd_memory >= 0
    assert call.param_memory == trace.modules["block"].param_memory
    assert isinstance(call.param_memory_str, str)
    assert isinstance(call.call_tree, CallTreeNode)
    assert call.num_descendant_calls >= 2
    assert call.max_descendant_depth >= 1
