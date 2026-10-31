"""Hand-counted oracles for the legacy graph-count quantities.

Every expected value below is counted by hand from the fixture's ``forward``
source in the comment beside it, never read from TorchLens. The graph model
the counts use: one Op per executed tensor operation and per boundary tensor
(each model input, each model output, each registered-buffer read and each
recorded buffer write); a Layer groups the passes of one repeated operation
(a module called twice is one Layer with two Ops); an edge is a distinct
(parent, child) pair. An internal source is a parentless node that is not a
model input (a factory such as ``torch.ones``, or a buffer read); an internal
sink is a node with no children that is not a model output (a discarded
result, or a buffer write that nothing reads back). These are the I2 oracles
the census (``tests/oracles/data/legacy_quantities.tsv``) names for those doors.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions


class Graph(nn.Module):
    """Two inputs, two outputs, a twice-called Linear, a buffer read, a factory, a dead end."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.register_buffer("scale", torch.full((4,), 2.0))
        self.idle = nn.Linear(2, 2)  # registered, never called

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(self.fc(x))  # linear pass 1, linear pass 2
        s = h * self.scale  # buffer read "scale", mul
        torch.sigmoid(s)  # discarded: the internal sink
        k = torch.ones(4)  # factory: an internal source
        return (s + k) + y, s.relu()  # add, add, relu; two outputs


# Ops of Graph, by hand: input x, input y, linear (2 passes), buffer read,
# mul, sigmoid, ones, add, add, relu, output 1, output 2.
G_INPUTS, G_OUTPUTS, G_BUFFER_READS = 2, 2, 1
G_COMPUTE_OPS = 2 + 6  # two linear passes + mul, sigmoid, ones, add, add, relu
G_COMPUTE_LAYERS = 1 + 6  # the linear passes share one Layer
G_OPS = G_INPUTS + G_OUTPUTS + G_BUFFER_READS + G_COMPUTE_OPS  # 13
G_LAYERS = G_INPUTS + G_OUTPUTS + G_BUFFER_READS + G_COMPUTE_LAYERS  # 12
# Layer-graph edges: x->linear, linear->linear (pass 1 feeds pass 2),
# linear->mul, scale->mul, mul->sigmoid, mul->add1, ones->add1, add1->add2,
# y->add2, mul->relu, add2->output1, relu->output2.
G_LAYER_EDGES = 12


class Counter(nn.Module):
    """An in-place buffer write and a float64 buffer read."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("scale", torch.full((4,), 2.0, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.count.add_(1.0)  # read "count", add_, write "count"
        return x * self.scale  # read "scale", mul


class LoopedBatchNorm(nn.Module):
    """A training BatchNorm1d and an in-place counter inside a three-step loop."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.fc = nn.Linear(4, 4)
        self.register_buffer("count", torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            self.count.add_(1.0)
            x = self.fc(self.bn(x)).relu()
        return x


class Grad(nn.Module):
    """Linear -> ReLU -> Linear, all differentiable."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


class Mixed(nn.Module):
    """A float64 Linear and one float32 Linear registered under two names and called twice."""

    def __init__(self) -> None:
        super().__init__()
        self.wide = nn.Linear(4, 4).double()
        self.tied = nn.Linear(4, 4)
        self.tied2 = self.tied

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tied2(self.tied(self.wide(x.double()).float()))


@pytest.fixture(scope="module")
def graph_trace():
    torch.manual_seed(0)
    # Hold the model: uncalled_modules reads the live source model.
    model = Graph().eval()
    trace = tl.trace(model, (torch.randn(2, 4), torch.randn(2, 4)))
    yield trace
    trace.cleanup()


@pytest.fixture(scope="module")
def counter_trace():
    torch.manual_seed(0)
    trace = tl.trace(Counter().eval(), torch.randn(2, 4))
    yield trace
    trace.cleanup()


@pytest.fixture(scope="module")
def batchnorm_trace():
    torch.manual_seed(0)
    trace = tl.trace(nn.Sequential(nn.BatchNorm1d(4)).train(), torch.randn(3, 4))
    yield trace
    trace.cleanup()


@pytest.fixture(scope="module")
def looped_trace():
    torch.manual_seed(0)
    trace = tl.trace(LoopedBatchNorm().train(), torch.randn(5, 4))
    yield trace
    trace.cleanup()


@pytest.fixture(scope="module")
def grad_trace():
    torch.manual_seed(0)
    trace = tl.trace(Grad().train(), torch.randn(3, 4), capture=CaptureOptions(save_grads=True))
    loss = trace[trace.output_layers[0]].out.sum()
    trace.backward(loss, retain_graph=True)
    trace.backward(loss)
    yield trace
    trace.cleanup()


@pytest.fixture(scope="module")
def mixed_trace():
    torch.manual_seed(0)
    model = Mixed().eval()
    trace = tl.trace(model, torch.randn(2, 4))
    yield trace
    trace.cleanup()


def test_op_and_layer_counts(graph_trace, mixed_trace) -> None:
    trace = graph_trace
    assert int(len(trace)) == G_OPS
    assert int(len(trace.layers)) == G_LAYERS
    assert int(len(trace.compute_ops)) == G_COMPUTE_OPS
    assert int(len(trace.layers.compute)) == G_COMPUTE_LAYERS
    assert int(trace.graph.num_layer_edges) == G_LAYER_EDGES
    # Mixed: input, double, wide linear, float, tied linear (2 passes), output.
    assert int(len(mixed_trace)) == 7
    assert int(len(mixed_trace.layers)) == 6
    assert int(len(mixed_trace.compute_ops)) == 5
    assert int(len(mixed_trace.layers.compute)) == 4


def test_boundary_counts(graph_trace) -> None:
    trace = graph_trace
    assert int(len(trace.input_ops)) == int(trace.num_input_layers) == G_INPUTS
    assert int(len(trace.output_ops)) == int(trace.num_output_layers) == G_OUTPUTS


def test_buffer_counts(graph_trace, counter_trace, batchnorm_trace, looped_trace) -> None:
    trace = graph_trace
    assert int(len(trace.buffer_read_ops)) == int(len(trace.buffer_read_ops)) == 1
    assert int(len(trace.buffer_write_ops)) == int(len(trace.buffer_write_ops)) == 0
    assert int(trace.graph.num_buffer_edges) == 1  # scale -> mul
    # Counter: reads of count and scale, one write of count.
    trace = counter_trace
    assert int(len(trace.buffer_read_ops)) == int(len(trace.buffer_read_ops)) == 2
    assert int(len(trace.buffer_write_ops)) == int(len(trace.buffer_write_ops)) == 1
    # count -> add_, add_ -> written count, scale -> mul.
    assert int(trace.graph.num_buffer_edges) == 3
    # Training BatchNorm1d reads running_mean, running_var and
    # num_batches_tracked, and writes all three (num_batches_tracked += 1 and
    # the two running-stat updates).
    trace = batchnorm_trace
    assert int(len(trace.buffer_read_ops)) == int(len(trace.buffer_read_ops)) == 3
    assert int(len(trace.buffer_write_ops)) == int(len(trace.buffer_write_ops)) == 3
    # nbt -> add_, add_ -> nbt', mean -> bn, var -> bn, bn -> mean', bn -> var'.
    assert int(trace.graph.num_buffer_edges) == 6
    # Looped: four buffers (count, nbt, mean, var) are each read once, from
    # their initial value; every step writes all four (12 writes), and later
    # steps consume the written versions directly.
    trace = looped_trace
    assert int(len(trace.buffer_read_ops)) == int(len(trace.buffer_read_ops)) == 4
    assert int(len(trace.buffer_write_ops)) == int(len(trace.buffer_write_ops)) == 12
    # Per step: count -> add_ -> count', nbt -> add_ -> nbt', mean -> bn,
    # var -> bn, bn -> mean', bn -> var': 8 buffer edges, 3 steps.
    assert int(trace.graph.num_buffer_edges) == 3 * 8


def test_internal_source_and_sink_counts(
    graph_trace, counter_trace, batchnorm_trace, looped_trace
) -> None:
    # Graph: sources are the buffer read and torch.ones; the sink is sigmoid.
    trace = graph_trace
    assert int(len(trace.internal_source_ops)) == int(len(trace.layers.internal_source)) == 2
    assert int(len(trace.internal_sink_ops)) == int(len(trace.layers.internal_sink)) == 1
    # Counter: the two buffer reads are sources; the written count, which has
    # the add_ as its parent, is a sink and not a source.
    trace = counter_trace
    assert int(len(trace.internal_source_ops)) == int(len(trace.layers.internal_source)) == 2
    assert int(len(trace.internal_sink_ops)) == int(len(trace.layers.internal_sink)) == 1
    # BatchNorm: the three reads are sources, the three writes are sinks.
    trace = batchnorm_trace
    assert int(len(trace.internal_source_ops)) == int(len(trace.layers.internal_source)) == 3
    assert int(len(trace.internal_sink_ops)) == int(len(trace.layers.internal_sink)) == 3
    # Looped: the four initial reads are sources; only each buffer's last
    # write is unread, so four sinks. Each buffer is one Layer whose passes
    # are its versions, so the source and sink Layers are the same four.
    trace = looped_trace
    assert int(len(trace.internal_source_ops)) == int(len(trace.layers.internal_source)) == 4
    assert int(len(trace.internal_sink_ops)) == int(len(trace.layers.internal_sink)) == 4


def test_parent_child_and_input_counts(graph_trace) -> None:
    trace = graph_trace
    layers = {layer.layer_label: layer for layer in trace.layers}
    by_type: dict[str, list] = {}
    for op in trace._layer_list:
        by_type.setdefault(op.type, []).append(op)
    (mul,) = by_type["mul"]
    (relu,) = by_type["relu"]
    first, second = by_type["linear"]
    add_inner, add_outer = by_type["add"]
    # mul consumes the second linear pass and the buffer read, and feeds
    # sigmoid, the inner add and relu.
    assert len(mul.parents) == int(mul.num_inputs) == 2
    assert len(mul.children) == 3
    assert len(first.parents) == len(first.children) == 1
    assert len(second.parents) == len(second.children) == 1
    assert len(add_inner.parents) == 2  # mul, ones
    assert len(add_outer.parents) == int(add_outer.num_inputs) == 2  # inner add, y
    assert len(relu.parents) == len(relu.children) == 1
    # The linear Layer: parents {x, itself}, children {itself, mul}.
    linear_layer = layers[first.layer_label]
    assert len(linear_layer.parents) == 2
    assert len(linear_layer.children) == 2
    assert len(layers[mul.layer_label].parents) == 2
    assert len(layers[mul.layer_label].children) == 3


def test_module_scope_edge_and_layer_counts(graph_trace) -> None:
    trace = graph_trace
    modules = {module.address: module for module in trace.modules}
    calls = {call.call_label: call for call in trace.module_calls}
    fc = modules["fc"]
    # fc holds both linear passes: one Layer; internal edge pass1 -> pass2,
    # input edge x -> pass1, output edge pass2 -> mul.
    assert len(fc.layer_labels) == 1
    assert (int(fc.num_internal_edges), int(fc.num_input_edges), int(fc.num_output_edges)) == (
        1,
        1,
        1,
    )
    # Each fc call holds one pass: no internal edge, one edge in, one out.
    for label in ("fc:1", "fc:2"):
        call = calls[label]
        assert (
            int(call.num_internal_edges),
            int(call.num_input_edges),
            int(call.num_output_edges),
        ) == (0, 1, 1)
    # The root holds every node, so every edge is internal.
    root, root_call = modules["self"], calls["self:1"]
    assert len(root.layer_labels) == G_LAYERS
    for scope in (root, root_call):
        assert (
            int(scope.num_internal_edges),
            int(scope.num_input_edges),
            int(scope.num_output_edges),
        ) == (12, 0, 0)


def test_module_call_and_param_use_counts(graph_trace, mixed_trace) -> None:
    trace = graph_trace
    # Calls: the root once, fc twice. idle is registered and never called.
    assert int(len(trace.module_calls)) == 3
    assert int(trace.num_uncalled_modules) == 1
    # The two linear passes use parameters; one Layer holds both.
    assert int(len(trace.ops_with_params)) == 2
    assert int(trace._num_layers_with_params) == 1
    # Mixed: root, wide, tied twice (tied2 is the same module).
    trace = mixed_trace
    assert int(len(trace.module_calls)) == 4
    assert int(trace.num_uncalled_modules) == 0
    assert int(len(trace.ops_with_params)) == 3  # wide, tied pass 1, tied pass 2
    assert int(trace._num_layers_with_params) == 2


def test_saved_activation_scope_counts(graph_trace, mixed_trace) -> None:
    # The default capture saves every Op, so every Layer, every module call
    # (each call's output Ops are saved) and every called module counts.
    trace = graph_trace
    assert len(trace.saved_layers) == G_LAYERS
    assert len(trace.saved_module_calls) == 3  # self:1, fc:1, fc:2
    assert int(len(trace.saved_modules)) == 2  # self, fc
    trace = mixed_trace
    assert len(trace.saved_layers) == 6
    assert len(trace.saved_module_calls) == 4  # self:1, wide:1, tied:1, tied:2
    assert int(len(trace.saved_modules)) == 3  # self, wide, tied


def test_saved_gradient_counts(grad_trace) -> None:
    trace = grad_trace
    # The input does not require grad; linear a, relu, linear b and the output
    # receive a gradient: four Ops, four Layers.
    assert int(len(trace.saved_grad_ops)) == 4
    assert int(len(trace.layers.saved_grad)) == 4
    # Every call's output Op received a gradient: self:1 (the output), a:1, b:1.
    assert int(len(trace.saved_grad_module_calls)) == 3
    assert int(len(trace.saved_grad_modules)) == 3
    # The backward nodes of the three captured differentiable ops (addmm,
    # relu, addmm) keep their gradients; two backward passes call each once.
    assert len(trace.autograd.saved_grad_fns) == 3
    assert len(trace.autograd.saved_grad_fn_calls) == 2 * 3


@pytest.mark.parametrize("axis", ["train_batchnorm", "in_place", "plain_cnn"])
def test_internal_source_flag_and_list_name_the_same_ops(axis: str) -> None:
    # The flag is set at materialization and the list at capture; an Op is an
    # internal source by one exactly when it is by the other. plain_cnn writes
    # no buffer, train_batchnorm and in_place do.
    from capture_oracle._models import build_model_case

    torch.manual_seed(0)
    model, model_input = build_model_case(axis)
    trace = tl.trace(model, model_input)
    try:
        listed = set(trace.internal_source_ops)
        flagged = {op.label for op in trace._layer_list if op.is_internal_source}
        assert listed == flagged, (listed, flagged)
        for op in trace._layer_list:
            if op.is_internal_source:
                assert not op.parents, op.label
    finally:
        trace.cleanup()
