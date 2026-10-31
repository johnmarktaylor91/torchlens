"""Closed-form oracles for every door of the legacy count, byte and compute quantities.

Each expected value is derived by hand from the fixture's dimensions in the
comments, never read from TorchLens. Fixture A is one ``Linear(8, 16)`` on a
batch of 2; fixture B is ``Linear(4, 8)`` (frozen) -> ReLU -> ``Linear(8, 2)``
plus an unused ``Linear(3, 3)``, on a batch of 3. These are the I2 oracles the
census (``tests/oracles/data/legacy_quantities.tsv``) names for those doors.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl

FP32 = 4

# Fixture A: Linear(8, 16), batch 2.
A_PARAMS = 8 * 16 + 16  # weight + bias elements = 144
A_TENSORS = 2  # weight, bias
A_MACS = 2 * 8 * 16  # batch x in x out = 256
A_FLOPS = 2 * A_MACS + 2 * 16  # fma counted as 2, plus one bias add per output = 544
A_OUT_BYTES = 2 * 16 * FP32  # output activation = 128
A_IN_BYTES = 2 * 8 * FP32  # input activation = 64

# Fixture B: a = Linear(4, 8) frozen, b = Linear(8, 2), dead = Linear(3, 3) never called.
B_A, B_B, B_DEAD = 4 * 8 + 8, 8 * 2 + 2, 3 * 3 + 3  # 40, 18, 12 elements
B_MACS_A, B_MACS_B = 3 * 4 * 8, 3 * 8 * 2  # 96, 48
B_FLOPS_A, B_FLOPS_B = 2 * B_MACS_A + 3 * 8, 2 * B_MACS_B + 3 * 2  # 216, 102


class TwoLinear(nn.Module):
    """Fixture B."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 2)
        self.dead = nn.Linear(3, 3)
        for parameter in self.a.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


@pytest.fixture(scope="module")
def fixture_a():
    torch.manual_seed(0)
    trace = tl.trace(nn.Linear(8, 16).eval(), torch.randn(2, 8))
    yield trace, trace.summary()
    trace.cleanup()


@pytest.fixture(scope="module")
def fixture_b():
    torch.manual_seed(0)
    trace = tl.trace(TwoLinear().eval(), torch.randn(3, 4))
    yield trace, trace.summary()
    trace.cleanup()


def _only(accessor):
    items = list(accessor)
    assert len(items) == 1, items
    return items[0]


def test_parameter_counts_on_every_door(fixture_a, fixture_b) -> None:
    trace, report = fixture_a
    layer, op = _only(trace.layers.compute), _only(trace.compute_ops)
    module, call = _only(trace.modules), _only(trace.module_calls)
    for value in (trace.num_params, layer.num_params, op.num_params, module.num_params):
        assert int(value) == A_PARAMS
    assert int(report.total_params) == int(report.totals.params_total) == A_PARAMS
    by_name = {param.address: param for param in trace.params}
    assert int(by_name["weight"].num_params) == 8 * 16
    assert int(by_name["bias"].num_params) == 16
    assert len(trace.params) == A_TENSORS
    assert len(module.params) == A_TENSORS
    assert int(call.num_param_tensors) == A_TENSORS

    trace, report = fixture_b
    layers = {layer.layer_label: layer for layer in trace.layers.compute}
    ops = {op.label: op for op in trace.compute_ops}
    modules = {module.address: module for module in trace.modules}
    assert int(trace.num_params_frozen) == B_A
    assert int(trace.num_params_trainable) == B_B + B_DEAD
    assert int(report.frozen_params) == int(report.totals.params_frozen) == B_A
    assert int(report.trainable_params) == int(report.totals.params_trainable) == B_B + B_DEAD
    assert int(layers["linear_1_1"].num_params_frozen) == B_A
    assert int(layers["linear_1_1"].num_params_trainable) == 0
    assert int(layers["linear_2_3"].num_params_trainable) == B_B
    assert int(ops["linear_1_1:1"].num_params_frozen) == B_A
    assert int(ops["linear_2_3:1"].num_params_trainable) == B_B
    assert int(modules["a"].num_params_frozen) == B_A
    assert int(modules["b"].num_params_trainable) == B_B
    # Executed = consumed by a captured op: a and b; dead never ran.
    assert int(report.executed_params) == int(report.totals.params_executed) == B_A + B_B
    assert int(report.unexecuted_params) == int(report.totals.params_unexecuted) == B_DEAD


def test_parameter_bytes_on_every_door(fixture_a, fixture_b) -> None:
    trace, _ = fixture_a
    layer, op = _only(trace.layers.compute), _only(trace.compute_ops)
    module, call = _only(trace.modules), _only(trace.module_calls)
    for value in (
        op.param_memory,
        layer.total_param_memory,
        module.param_memory,
        call.param_memory,
        trace.total_param_memory,
    ):
        assert int(value) == A_PARAMS * FP32
    by_name = {param.address: param for param in trace.params}
    assert int(by_name["weight"].param_memory) == 8 * 16 * FP32
    assert int(by_name["bias"].param_memory) == 16 * FP32
    trace, _ = fixture_b
    # Declared parameters, the never-called module included.
    assert int(trace.total_param_memory) == (B_A + B_B + B_DEAD) * FP32


def test_macs_and_flops_on_every_door(fixture_a, fixture_b) -> None:
    trace, report = fixture_a
    layer, op, module = (
        _only(trace.layers.compute),
        _only(trace.compute_ops),
        _only(trace.modules),
    )
    for value in (
        op.macs_forward,
        layer.macs_forward,
        layer.total_macs_forward,
        module.total_macs_forward,
        report.total_macs_forward,
        report.totals.macs_forward,
        trace.cost.total_macs_forward,
    ):
        assert int(value) == A_MACS
    for value in (
        op.flops_forward,
        layer.flops_forward,
        layer.total_flops_forward,
        module.total_flops_forward,
        report.total_flops_forward,
        report.totals.flops_forward_fma2,
        trace.cost.total_flops_forward,
    ):
        assert int(value) == A_FLOPS
    trace, _ = fixture_b
    ops = {op.label: op for op in trace.compute_ops}
    assert int(ops["linear_1_1:1"].macs_forward) == B_MACS_A
    assert int(ops["linear_2_3:1"].macs_forward) == B_MACS_B
    assert int(ops["linear_1_1:1"].flops_forward) == B_FLOPS_A
    assert int(ops["linear_2_3:1"].flops_forward) == B_FLOPS_B
    # ReLU is not a multiply-accumulate, so the total is the two linears.
    assert int(trace.cost.total_macs_forward) == B_MACS_A + B_MACS_B


def test_op_pass_and_edge_counts_on_every_door(fixture_a, fixture_b) -> None:
    trace, _ = fixture_a
    layer, op = _only(trace.layers.compute), _only(trace.compute_ops)
    module, call = _only(trace.modules), _only(trace.module_calls)
    assert len(layer.ops) == 1 and int(layer.num_passes) == 1 and int(op.num_passes) == 1
    # A module call counts its boundary ops: input, linear, output.
    assert len(call.ops) == 3
    # input -> linear -> output.
    assert int(module.num_edges) == int(call.num_edges) == 2
    trace, _ = fixture_b
    modules = {module.address: module for module in trace.modules}
    # input -> a -> relu -> b -> output.
    assert int(trace.num_edges) == int(modules["self"].num_edges) == 4
    assert int(modules["a"].num_edges) == int(modules["b"].num_edges) == 2


def test_activation_bytes_on_every_compute_door(fixture_a) -> None:
    trace, _ = fixture_a
    layer, op = _only(trace.layers.compute), _only(trace.compute_ops)
    for value in (op.activation_memory, layer.activation_memory, layer.total_activation_memory):
        assert int(value) == A_OUT_BYTES
    # The trace total sums every retained activation: the input and the output.
    assert int(trace.total_activation_memory) == A_IN_BYTES + A_OUT_BYTES


class BufferHolder(nn.Module):
    """A float32 scalar buffer written in place and a float64 vector buffer read."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("scale", torch.full((4,), 2.0, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.count.add_(1.0)
        return x * self.scale


def test_buffer_activation_bytes_on_the_buffer_door() -> None:
    torch.manual_seed(0)
    trace = tl.trace(BufferHolder().eval(), torch.randn(2, 4))
    try:
        by_address = {buffer.address: buffer for buffer in trace.buffers}
        # The final observed value: a 0-d float32 (4 bytes) and 4 float64s (32 bytes).
        assert int(by_address["count"].activation_memory) == 1 * FP32
        assert int(by_address["scale"].activation_memory) == 4 * 8
    finally:
        trace.cleanup()


class MixedPrecision(nn.Module):
    """A float64 Linear(4, 4) and a float32 Linear(4, 4) registered under two names."""

    def __init__(self) -> None:
        super().__init__()
        self.wide = nn.Linear(4, 4).double()
        self.tied = nn.Linear(4, 4)
        self.tied2 = self.tied

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tied2(self.tied(self.wide(x.double()).float()))


def test_summary_parameter_bytes_count_unique_storage() -> None:
    torch.manual_seed(0)
    trace = tl.trace(MixedPrecision().eval(), torch.randn(2, 4))
    try:
        # wide: 20 float64 elements = 160 bytes; tied: 20 float32 elements = 80
        # bytes, counted once although two names hold it and it ran twice.
        assert int(trace.summary().totals.params_bytes) == 20 * 8 + 20 * FP32
    finally:
        trace.cleanup()
