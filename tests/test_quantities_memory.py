"""Conformance tests for v7 memory quantity fields."""

import re

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.constants import (
    BUFFER_LOG_FIELD_ORDER,
    GRAD_FN_LOG_FIELD_ORDER,
    GRAD_FN_PASS_LOG_FIELD_ORDER,
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
def test_duration_flops_and_macs_formatting_and_numeric_behavior() -> None:
    """Duration, Flops, and Macs format directly while staying numeric."""

    duration = tl.Duration(0.0012)
    flops = tl.Flops(1_234_000_000)
    macs = tl.Macs(3_400_000)

    assert isinstance(duration, float)
    assert isinstance(flops, int)
    assert isinstance(macs, int)
    assert str(duration) == "1.2 ms"
    assert str(tl.Duration(3.4)) == "3.4 s"
    assert f"{duration:raw}" == "0.0012"
    assert f"{duration:.2f ms}" == "1.20 ms"
    assert str(flops) == "1.23 GFLOPs"
    assert f"{flops:raw}" == "1234000000"
    assert f"{flops:.1f GFLOPs}" == "1.2 GFLOPs"
    assert str(macs) == "3.4 MMACs"
    assert f"{macs:.1f MMACs}" == "3.4 MMACs"
    assert duration + tl.Duration(0.001) == pytest.approx(tl.Duration(0.0022))
    assert flops + 1 == tl.Flops(1_234_000_001)
    assert macs / 2 == tl.Macs(1_700_000)


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
def test_duration_flops_and_macs_fields_are_quantity_types() -> None:
    """Runtime duration, FLOP, and MAC fields use quantity wrappers."""

    trace = _trace_memory_model()
    op = next(record for record in trace.ops if record.flops_forward)
    layer = trace.layers[op.layer_label]
    module = trace.modules["linear"]
    call = module.ops[0]

    duration_fields = [
        trace.setup_duration,
        trace.forward_duration,
        trace.cleanup_duration,
        trace.capture_duration,
        trace.func_calls_duration,
        op.func_duration,
        layer.func_duration,
        module.forward_duration,
        module.total_forward_duration,
        module.func_calls_duration,
        module.total_func_calls_duration,
        call.forward_duration,
        call.func_calls_duration,
    ]
    flops_fields = [
        op.flops_forward,
        op.flops_backward,
        op.flops_total,
        layer.flops_forward,
        layer.total_flops_forward,
        trace.total_flops_forward,
        trace.total_flops,
        module.flops_forward,
        module.flops,
        trace.flops_by_op_type()[op.layer_type]["forward"],
    ]
    macs_fields = [
        op.macs_forward,
        op.macs_total,
        layer.macs_forward,
        layer.total_macs_forward,
        trace.total_macs_forward,
        trace.total_macs,
        module.macs_forward,
        module.macs,
        trace.macs_by_op_type()[op.layer_type]["forward"],
    ]

    assert all(isinstance(field, tl.Duration) for field in duration_fields)
    assert all(isinstance(field, tl.Flops) for field in flops_fields if field is not None)
    assert all(isinstance(field, tl.Macs) for field in macs_fields if field is not None)


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


@pytest.mark.smoke
def test_no_str_suffixed_quantity_fields_remain_public() -> None:
    """Field orders and records no longer expose quantity string companions."""

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
    removed_suffixes = ("_memory_str", "_duration_str", "_flops_str", "_macs_str")
    removed_exact = {"memory_str", "duration_str", "flops_str", "macs_str"}
    field_orders = [
        LAYER_PASS_LOG_FIELD_ORDER,
        LAYER_LOG_FIELD_ORDER,
        PARAM_LOG_FIELD_ORDER,
        BUFFER_LOG_FIELD_ORDER,
        MODULE_PASS_LOG_FIELD_ORDER,
        MODULE_LOG_FIELD_ORDER,
        MODEL_LOG_FIELD_ORDER,
        GRAD_FN_PASS_LOG_FIELD_ORDER,
        GRAD_FN_LOG_FIELD_ORDER,
    ]

    for field_order in field_orders:
        assert all(not field.endswith(removed_suffixes) for field in field_order)
        assert all(field not in removed_exact for field in field_order)

    for record in records:
        for field in {
            "forward_duration_str",
            "backward_duration_str",
            "func_calls_duration_str",
            "total_forward_duration_str",
            "total_backward_duration_str",
            "total_func_calls_duration_str",
            "capture_duration_str",
        }:
            assert not hasattr(record, field)
