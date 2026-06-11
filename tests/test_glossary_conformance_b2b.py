"""Focused coverage for glossary-conformance phase B2b derived properties."""

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl


class _B2bModel(nn.Module):
    """Small model with trainable, frozen, buffer, and uncalled records."""

    def __init__(self) -> None:
        """Initialize modules used by the B2b conformance tests."""

        super().__init__()
        self.frozen = nn.Linear(4, 4)
        self.trainable = nn.Linear(4, 2)
        self.norm = nn.BatchNorm1d(4)
        self.unused = nn.Linear(4, 4)
        for param in self.frozen.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the logged forward pass."""

        return self.trainable(torch.relu(self.norm(self.frozen(x))))


def _make_trace() -> Any:
    """Create a trace with saved activations and gradients enabled."""

    model = _B2bModel()
    x = torch.randn(3, 4, requires_grad=True)
    return tl.trace(model, x, layers_to_save="all", save_grads="all")


@pytest.mark.smoke
def test_trace_derived_counts_match_accessors() -> None:
    """Trace count properties are derived from their documented accessors."""

    trace = _make_trace()

    assert trace.num_layers == len(trace.layers)
    assert trace.num_compute_layers == len(trace.compute_layers)
    assert trace.num_compute_ops == len(trace.compute_ops)
    assert trace.num_saved_grad_ops == len(trace.saved_grad_ops)
    assert trace.num_saved_grad_layers == len(trace.saved_grad_layers)
    assert trace.num_input_layers == len(trace.input_layers)
    assert trace.num_output_layers == len(trace.output_layers)
    assert trace.num_buffer_layers == len(trace.buffer_layers)
    assert trace.num_internal_source_ops == len(trace.internal_source_ops)
    assert trace.num_internal_sink_ops == len(trace.internal_sink_ops)
    assert trace.num_uncalled_modules == len(trace.uncalled_modules)
    assert trace.num_param_tensors_trainable == sum(
        1 for param in trace.params if param.is_trainable
    )
    assert trace.num_param_tensors_frozen == sum(
        1 for param in trace.params if not param.is_trainable
    )
    assert trace.has_trainable_params is True
    assert trace.has_frozen_params is True


@pytest.mark.smoke
def test_op_and_layer_convenience_properties() -> None:
    """Op and Layer convenience properties aggregate already captured data."""

    trace = _make_trace()
    op = next(record for record in trace.compute_ops if record.num_param_tensors)
    layer = trace.layers[op.layer_label]

    assert op.flops_total == (op.flops_forward or 0) + (op.flops_backward or 0)
    assert op.macs_total == op.flops_total // 2
    assert op.param_names == [param.name for param in op.params]
    assert op.param_dtypes == [param.dtype for param in op.params]
    assert op.num_param_tensors_trainable == sum(1 for param in op.params if param.is_trainable)
    assert op.num_param_tensors_frozen == sum(1 for param in op.params if not param.is_trainable)
    assert op.has_trainable_params == (op.num_params_trainable > 0)
    assert op.has_frozen_params == (op.num_params_frozen > 0)
    assert op.is_compute_op is True

    assert layer.num_ops == len(layer.ops)
    assert layer.total_activation_memory == sum(
        (child.activation_memory or 0) for child in layer.ops.values()
    )
    assert layer.total_gradient_memory == sum(
        (child.gradient_memory or 0) for child in layer.ops.values()
    )
    assert layer.flops_total == (layer.flops_forward or 0) + (layer.flops_backward or 0)
    assert layer.total_flops_total == layer.total_flops_forward + layer.total_flops_backward
    assert layer.macs_total == layer.flops_total // 2
    assert layer.total_macs_total == layer.total_flops_total // 2
    assert layer.param_names == [param.name for param in layer.params]
    assert layer.param_dtypes == [param.dtype for param in layer.params]
    assert layer.has_trainable_params == (layer.num_params_trainable > 0)
    assert layer.has_frozen_params == (layer.num_params_frozen > 0)
    assert layer.is_compute_layer is True
    assert layer.is_orphan == any(child.is_orphan for child in layer.ops.values())

    if op.fx_label is not None:
        assert layer.fx_label == op.fx_label
        assert layer.fx_qualpath == op.fx_qualpath
        assert layer.fx_call_index == op.fx_call_index
        assert trace[op.fx_label] is op


@pytest.mark.smoke
def test_record_trace_aliases_and_ordinal_indexes_round_trip() -> None:
    """Records expose the owning Trace through the universal trace alias."""

    trace = _make_trace()
    loss = trace[trace.output_layers[0]].out.sum()
    trace.log_backward(loss)

    op = trace.ops[0]
    layer = trace.layers[0]
    module = trace.modules[0]
    module_call = trace.module_calls[0]
    param = trace.params[0]
    buffer = trace.buffers[0]
    grad_fn = trace.grad_fns[0]
    grad_fn_call = trace.grad_fn_calls[0]

    assert op.trace is trace
    assert layer.trace is trace
    assert module.trace is trace
    assert module_call.trace is trace
    assert param.trace is trace
    assert buffer.trace is trace
    assert grad_fn.trace is trace
    assert grad_fn_call.trace is trace

    assert trace.params[param.ordinal_index] is param
    assert trace.grad_fns[grad_fn.ordinal_index] is grad_fn
    assert trace.grad_fn_calls[grad_fn_call.ordinal_index] is grad_fn_call
    assert trace.num_grad_fns_with_op == sum(1 for record in trace.grad_fns if record.has_op)
