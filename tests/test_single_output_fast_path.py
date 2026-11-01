"""Regression tests for Torch backend per-output field handling (M7)."""

from __future__ import annotations

import copy

import pytest
import torch

import torchlens as tl


class _SingleOutputModel(torch.nn.Module):
    """Model with one ordinary tensor output from a wrapped torch call."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a single-output torch operation.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU output.
        """

        return torch.relu(x)


class _SplitOutputModel(torch.nn.Module):
    """Model with a multi-output wrapped torch call."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split an input tensor and consume both outputs.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Sum of both split outputs.
        """

        left, right = torch.split(x, 1, dim=0)
        return left + right


@pytest.fixture()
def split_trace() -> tl.Trace:
    """Trace of a two-output ``torch.split`` call."""

    return tl.trace(_SplitOutputModel(), torch.randn(2, 3))


def test_single_output_fast_path_traces_shared_fields() -> None:
    """Single-output torch calls keep exact per-output metadata."""

    trace = tl.trace(_SingleOutputModel(), torch.randn(2, 3))

    relu_layers = [layer for layer in trace._layer_list if layer.func_name == "relu"]
    assert len(relu_layers) == 1
    relu_layer = relu_layers[0]
    assert relu_layer.in_multi_output is False
    assert relu_layer.multi_output_index is None
    assert relu_layer.container_path == ()
    assert relu_layer.container_spec is None
    assert relu_layer.parent_arg_positions["args"]


def test_multi_output_keeps_sibling_field_isolation(split_trace: tl.Trace) -> None:
    """Multi-output torch calls still isolate per-output mutable metadata."""

    split_layers = [layer for layer in split_trace._layer_list if layer.func_name == "split"]
    assert len(split_layers) == 2

    first_positions = split_layers[0].parent_arg_positions
    sibling_positions = split_layers[1].parent_arg_positions
    sibling_positions_before = copy.deepcopy(sibling_positions)

    first_arg_key = next(iter(first_positions["args"]))
    first_positions["args"][first_arg_key] = "sentinel_parent"

    assert sibling_positions == sibling_positions_before


def test_multi_output_siblings_share_one_function_call_ref(split_trace: tl.Trace) -> None:
    """Sibling outputs of one wrapped call share ONE journal FunctionCallRef (M7)."""

    events = [
        event
        for event in split_trace._capture_events.op_events
        if event.function.func_name == "split"
    ]
    assert len(events) == 2
    first_ref, second_ref = events[0].function, events[1].function
    assert first_ref.func_call_id == second_ref.func_call_id
    # Same per-output facts -> the exact same frozen ref object; a sibling
    # whose FLOPs/is_inplace genuinely differ would get a replace() derivative
    # that still shares every container field by reference.
    if (
        first_ref.flops_forward == second_ref.flops_forward
        and first_ref.flops_backward == second_ref.flops_backward
        and first_ref.is_inplace == second_ref.is_inplace
    ):
        assert first_ref is second_ref
    else:
        assert first_ref.code_context is second_ref.code_context
        assert first_ref.non_tensor_pos_args is second_ref.non_tensor_pos_args
        assert first_ref.func_config is second_ref.func_config
