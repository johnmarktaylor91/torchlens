"""Conformance tests for v7 memory quantity fields."""

import re

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.constants import (
    BUFFER_LOG_FIELD_ORDER,
    LAYER_LOG_FIELD_ORDER,
    LAYER_PASS_LOG_FIELD_ORDER,
    MODEL_LOG_FIELD_ORDER,
    MODULE_LOG_FIELD_ORDER,
    MODULE_PASS_LOG_FIELD_ORDER,
    PARAM_LOG_FIELD_ORDER,
)


class _MemoryModel(nn.Module):
    """Small model with parameters and a registered buffer."""

    def __init__(self) -> None:
        """Initialize the test modules and buffer."""

        super().__init__()
        self.register_buffer("offset", torch.ones(1, 4))
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass with buffer and parameter usage."""

        return self.linear(x + self.offset)


def _trace_memory_model() -> tl.Trace:
    """Return a trace with saved activations and gradients."""

    model = _MemoryModel()
    x = torch.randn(3, 4, requires_grad=True)
    return tl.trace(model, x, layers_to_save="all", gradients_to_save="all")


@pytest.mark.smoke
def test_bytes_formatting_and_numeric_behavior() -> None:
    """Bytes formats human-readably while preserving numeric operations."""

    value = tl.Bytes(1_234_567)

    assert isinstance(value, tl.Quantity)
    assert isinstance(value, int)
    assert str(value) == "1.2 MB"
    assert f"{value:raw}" == "1234567"
    assert f"{value:.2f MB}" == "1.18 MB"
    assert value + 1 == tl.Bytes(1_234_568)
    assert 1 + value == tl.Bytes(1_234_568)
    assert value - 567 == tl.Bytes(1_234_000)
    assert value * 2 == tl.Bytes(2_469_134)
    assert value / 2 == tl.Bytes(617_283)
    assert value / tl.Bytes(1_234_567) == 1.0
    assert value > 1_000_000


@pytest.mark.smoke
def test_memory_fields_are_bytes_and_format_directly() -> None:
    """Trace, Op, Layer, Buffer, Module, ModuleCall, and Param memory are Bytes."""

    trace = _trace_memory_model()
    op = next(record for record in trace.ops if record.activation_memory)
    layer = trace.layers[op.layer_label]
    module = trace.modules["linear"]
    call = module.ops[0]
    param = trace.params[0]
    buffer = trace.buffers[0]

    fields = [
        trace.total_activation_memory,
        trace.saved_activation_memory,
        op.activation_memory,
        op.input_memory,
        op.param_memory,
        layer.activation_memory,
        layer.total_activation_memory,
        module.param_memory,
        module.total_output_activation_memory,
        call.output_activation_memory,
        call.param_memory,
        param.param_memory,
        buffer.activation_memory,
    ]

    assert all(isinstance(field, tl.Bytes) for field in fields)
    assert re.fullmatch(r"-?\d+(?:\.\d+)? [KMGTPE]?B", str(op.activation_memory))
    assert op.activation_memory + 1 > op.activation_memory


@pytest.mark.smoke
def test_memory_str_and_bare_memory_fields_are_not_public() -> None:
    """Memory string companions and bare memory record fields are absent."""

    trace = _trace_memory_model()
    records = [
        trace,
        trace.ops[0],
        trace.layers[0],
        trace.modules[0],
        trace.module_calls[0],
        trace.params[0],
        trace.buffers[0],
    ]
    removed_fields = {
        "memory",
        "memory_str",
        "activation_memory_str",
        "gradient_memory_str",
        "autograd_memory_str",
        "param_memory_str",
        "output_activation_memory_str",
        "internal_activation_memory_str",
        "output_gradient_memory_str",
        "internal_gradient_memory_str",
        "total_activation_memory_str",
        "saved_activation_memory_str",
        "total_param_memory_str",
        "total_output_activation_memory_str",
        "total_internal_activation_memory_str",
        "total_autograd_memory_str",
    }

    for record in records:
        for field in removed_fields:
            assert not hasattr(record, field)

    field_orders = [
        LAYER_PASS_LOG_FIELD_ORDER,
        LAYER_LOG_FIELD_ORDER,
        PARAM_LOG_FIELD_ORDER,
        BUFFER_LOG_FIELD_ORDER,
        MODULE_PASS_LOG_FIELD_ORDER,
        MODULE_LOG_FIELD_ORDER,
        MODEL_LOG_FIELD_ORDER,
    ]
    for field_order in field_orders:
        assert "memory" not in field_order
        assert all(not field.endswith("_memory_str") for field in field_order)
        assert all(field != "memory_str" for field in field_order)
