"""Hand-derived oracles for the legacy distance, argument, byte and FLOP quantities.

Every expected value below is derived by hand from the fixture's ``forward``
source and dimensions in the comment beside it, or recounted from raw torch
state (``Tensor.grad``, the autograd graph reached through
``loss.grad_fn.next_functions`` and hooks on its nodes); none is read from
TorchLens. Conventions the closed forms use:

- Hop distances count Op-graph edges from the nearest and farthest model
  input (following children) and to the nearest and farthest model output
  (following parents); a node with no such path has none.
- Argument counts are the positional and keyword arguments of the torch call
  as written: ``nn.Linear`` calls ``F.linear(input, weight, bias)``.
- Bytes are element count times element size (float32: 4). A trace-level
  gradient total counts each gradient tensor once: the output Op is an
  identity copy of its parent, so it holds its parent's gradient.
- Forward FLOPs count a multiply-accumulate as 2 and one add per bias
  element; backward FLOPs are the documented multiplier estimate on forward
  FLOPs (linear 2x, elementwise 1x).

The root module encompasses every parameter of the model (its parameter
counts are the model's), by design. These are the I2 oracles the census
(``tests/oracles/data/legacy_quantities.tsv``) names for those doors.
"""

from __future__ import annotations

from collections import Counter as Tally

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions

from .test_legacy_graph_counts import Grad, Graph
from .test_legacy_quantity_doors import TwoLinear

FP32 = 4


class Block(nn.Module):
    """A Linear and a ReLU scaled by a keyword argument."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        return torch.relu(self.fc(x)) * scale  # linear, relu, mul


class Outer(nn.Module):
    """A Block called twice (once with a keyword), a head, a keyword reduction."""

    def __init__(self) -> None:
        super().__init__()
        self.block = Block()
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block(x, scale=2.0)
        h = self.block(h)
        return torch.sum(self.head(h), dim=1, keepdim=True)


class Loop(nn.Module):
    """One Linear applied three times."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.fc(x)
        return x


class GradBuffer(nn.Module):
    """A buffer that requires grad and one that does not."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("w", torch.ones(4, requires_grad=True))
        self.register_buffer("c", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.w + self.c


class PadTwice(nn.Module):
    """Two pads (no FLOP rule) and a relu (one FLOP per element)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(nn.functional.pad(nn.functional.pad(x, (1, 0)), (0, 1)))


def _raw_graph(loss: torch.Tensor) -> tuple[list[object], set[tuple[int, int]]]:
    """Every autograd node reachable from ``loss`` and every (node, next) edge."""

    nodes: dict[int, object] = {}
    edges: set[tuple[int, int]] = set()
    stack = [loss.grad_fn]
    while stack:
        node = stack.pop()
        if id(node) in nodes:
            continue
        nodes[id(node)] = node
        for next_node, _ in node.next_functions:
            if next_node is not None:
                edges.add((id(node), id(next_node)))
                stack.append(next_node)
    return list(nodes.values()), edges


def _counting_hook(tally: Tally, key: object):
    """A node post-hook that counts one execution under ``key``."""

    def hook(*_args: object) -> None:
        tally.update([key])

    return hook


def _backward_with_call_tally(trace, model_loss: torch.Tensor, passes: int) -> Tally:
    """Run ``passes`` backward passes, tallying each raw node's executions by hook."""

    tally: Tally = Tally()
    nodes, _ = _raw_graph(model_loss)
    for node in nodes:
        node.register_hook(_counting_hook(tally, id(node)))
    for index in range(passes):
        trace.backward(model_loss, retain_graph=index < passes - 1)
    return tally


def _grad_case(passes: int):
    torch.manual_seed(0)
    model = Grad().train()
    trace = tl.trace(model, torch.randn(3, 4), capture=CaptureOptions(save_grads=True))
    loss = trace[trace.output_layers[0]].out.sum()
    tally = _backward_with_call_tally(trace, loss, passes)
    return model, trace, loss, tally


@pytest.fixture(scope="module")
def graph_trace():
    torch.manual_seed(0)
    model = Graph().eval()
    trace = tl.trace(
        model,
        (torch.randn(2, 4), torch.randn(2, 4)),
        capture=CaptureOptions(compute_input_output_distances=True),
    )
    yield trace
    trace.cleanup()


@pytest.fixture(scope="module")
def two_linear_trace():
    torch.manual_seed(0)
    model = TwoLinear().eval()
    trace = tl.trace(model, torch.randn(3, 4))
    yield trace
    trace.cleanup()


@pytest.fixture(scope="module")
def grad_once():
    model, trace, loss, tally = _grad_case(passes=1)
    yield model, trace, loss, tally
    trace.cleanup()


@pytest.fixture(scope="module")
def grad_twice():
    model, trace, loss, tally = _grad_case(passes=2)
    yield model, trace, loss, tally
    trace.cleanup()


@pytest.fixture(scope="module")
def outer_case():
    torch.manual_seed(0)
    model = Outer().train()
    trace = tl.trace(model, torch.randn(3, 4), capture=CaptureOptions(save_grads=True))
    loss = trace[trace.output_layers[0]].out.sum()
    tally = _backward_with_call_tally(trace, loss, passes=1)
    yield model, trace, loss, tally
    trace.cleanup()


def _by_label(trace):
    ops = {op.label: op for op in trace.layer_list}
    layers = {layer.layer_label: layer for layer in trace.layers}
    modules = {module.address: module for module in trace.modules}
    calls = {call.call_label: call for call in trace.module_calls}
    return ops, layers, modules, calls


def test_param_tensor_counts_split_frozen_and_trainable(two_linear_trace) -> None:
    # TwoLinear: a = Linear(4, 8) frozen (weight, bias), b = Linear(8, 2) and
    # the never-called dead = Linear(3, 3) trainable: 2 frozen, 4 trainable.
    trace = two_linear_trace
    ops, layers, modules, calls = _by_label(trace)
    assert (int(trace.num_param_tensors_frozen), int(trace.num_param_tensors_trainable)) == (2, 4)
    expected = {"linear_1_1": (2, 0), "relu_1_2": (0, 0), "linear_2_3": (0, 2)}
    for label, pair in expected.items():
        for record in (layers[label], ops[f"{label}:1"]):
            assert (
                int(record.num_param_tensors_frozen),
                int(record.num_param_tensors_trainable),
            ) == pair, label
    # Modules own their address's tensors; the root encompasses all six.
    for address, call_label, pair in (
        ("a", "a:1", (2, 0)),
        ("b", "b:1", (0, 2)),
        ("self", "self:1", (2, 4)),
    ):
        for record in (modules[address], calls[call_label]):
            assert (
                int(record.num_param_tensors_frozen),
                int(record.num_param_tensors_trainable),
            ) == pair, address


# Graph (test_legacy_graph_counts): x -> linear -> linear -> mul(scale) ->
# {sigmoid (dead end), add(ones) -> add(y) -> output 1, relu -> output 2}.
# (min, max) hops from an input and to an output, None where no path exists.
GRAPH_DISTANCES = {
    "input_1:1": ((0, 0), (5, 6)),  # x: via relu (5) or via the two adds (6)
    "input_2:1": ((0, 0), (2, 2)),  # y -> add2 -> output 1
    "linear_1_1:1": ((1, 1), (4, 5)),
    "linear_1_1:2": ((2, 2), (3, 4)),
    "buffer_1:1": ((None, None), (3, 4)),  # scale is no input's descendant
    "mul_1_2:1": ((3, 3), (2, 3)),
    "sigmoid_1_3:1": ((4, 4), (None, None)),  # discarded: reaches no output
    "ones_1_4:1": ((None, None), (3, 3)),  # factory: ones -> add1 -> add2 -> output 1
    "add_1_5:1": ((4, 4), (2, 2)),
    "add_2_6:1": ((1, 5), (1, 1)),  # from y in one hop, from x in five
    "relu_1_7:1": ((4, 4), (1, 1)),
    "output_1:1": ((2, 6), (0, 0)),
    "output_2:1": ((5, 5), (0, 0)),
}


def test_input_and_output_hop_distances(graph_trace) -> None:
    ops, _, _, _ = _by_label(graph_trace)
    assert set(ops) == set(GRAPH_DISTANCES)
    for label, (from_input, to_output) in GRAPH_DISTANCES.items():
        op = ops[label]
        assert (op.min_distance_from_input, op.max_distance_from_input) == from_input, label
        assert (op.min_distance_to_output, op.max_distance_to_output) == to_output, label


def test_layer_op_count_and_parameter_uses(graph_trace) -> None:
    trace = graph_trace
    # The linear Layer runs twice; every other Layer once.
    assert int(trace.graph.max_ops_per_layer) == 2
    params = {param.address: param for param in trace.param_logs}
    for address in ("fc.weight", "fc.bias"):
        assert (int(params[address].num_uses_by_ops), int(params[address].num_uses_by_layers)) == (
            2,
            1,
        )
    for address in ("idle.weight", "idle.bias"):  # registered, never called
        assert (int(params[address].num_uses_by_ops), int(params[address].num_uses_by_layers)) == (
            0,
            0,
        )
    torch.manual_seed(0)
    loop = tl.trace(Loop().eval(), torch.randn(2, 4))
    try:
        assert int(loop.graph.max_ops_per_layer) == 3  # fc three times, one Layer
        params = {param.address: param for param in loop.param_logs}
        assert (
            int(params["fc.weight"].num_uses_by_ops),
            int(params["fc.weight"].num_uses_by_layers),
        ) == (3, 1)
    finally:
        loop.cleanup()


def test_op_and_layer_argument_counts(graph_trace, outer_case) -> None:
    ops, layers, _, _ = _by_label(graph_trace)
    # (positional, keyword) per call as written in Graph.forward.
    expected = {
        "linear_1_1": (3, 0),  # F.linear(input, weight, bias)
        "mul_1_2": (2, 0),  # h * self.scale
        "sigmoid_1_3": (1, 0),  # torch.sigmoid(s)
        "ones_1_4": (1, 0),  # torch.ones(4)
        "add_1_5": (2, 0),  # s + k
        "add_2_6": (2, 0),  # (...) + y
        "relu_1_7": (1, 0),  # s.relu()
    }
    for label, (positional, keyword) in expected.items():
        records = [layers[label]] + [op for op in ops.values() if op.layer_label == label]
        for record in records:
            assert (
                int(record.num_pos_args),
                int(record.num_kwargs),
                int(record.num_args_total),
            ) == (positional, keyword, positional + keyword), label
    _, trace, _, _ = outer_case
    ops, layers, _, _ = _by_label(trace)
    # torch.sum(h, dim=1, keepdim=True): one positional, two keywords. Grouping
    # rule rows 26-27 and section 1 (argument values are never evidence): both
    # block calls' ``* scale`` muls are one Layer, so the sum is the 5th Layer.
    for record in (layers["sum_1_5"], ops["sum_1_5:1"]):
        assert (int(record.num_pos_args), int(record.num_kwargs), int(record.num_args_total)) == (
            1,
            2,
            3,
        )


def test_module_call_forward_argument_counts(outer_case) -> None:
    _, trace, _, _ = outer_case
    calls = {call.call_label: call for call in trace.module_calls}
    # block(x, scale=2.0); block(h); fc(x) twice; head(h).
    expected = {
        "block:1": (1, 1),
        "block:2": (1, 0),
        "block.fc:1": (1, 0),
        "block.fc:2": (1, 0),
        "head:1": (1, 0),
    }
    assert set(calls) == set(expected) | {"self:1"}
    for label, (positional, keyword) in expected.items():
        call = calls[label]
        assert (
            int(call.num_forward_pos_args),
            int(call.num_forward_kwargs),
            int(call.num_forward_args_total),
        ) == (positional, keyword, positional + keyword), label


def test_root_call_forward_argument_counts(graph_trace, outer_case) -> None:
    # Outer.forward(x): one positional; Graph.forward(x, y): two.
    _, trace, _, _ = outer_case
    root = {call.call_label: call for call in trace.module_calls}["self:1"]
    assert (
        int(root.num_forward_pos_args),
        int(root.num_forward_kwargs),
        int(root.num_forward_args_total),
    ) == (1, 0, 1)
    root = {call.call_label: call for call in graph_trace.module_calls}["self:1"]
    assert (int(root.num_forward_pos_args), int(root.num_forward_args_total)) == (2, 2)


def test_activation_parameter_and_input_bytes(two_linear_trace, graph_trace, outer_case) -> None:
    trace = two_linear_trace
    _, _, modules, calls = _by_label(trace)
    # Batch 3: input (3, 4) 48 B, a (3, 8) 96 B, relu 96 B, b (3, 2) 24 B.
    # a:1 and b:1 return their only Op; the root returns the output Op (24 B)
    # and holds the input, a, relu and b inside (48 + 96 + 96 + 24).
    for label, pair in (("a:1", (96, 0)), ("b:1", (24, 0)), ("self:1", (24, 264))):
        call = calls[label]
        assert (int(call.output_activation_memory), int(call.internal_activation_memory)) == pair
    # Parameter bytes: a 40 elements, b 18, dead 12; the root's subtree holds all.
    for address, call_label, nbytes in (
        ("a", "a:1", 40 * FP32),
        ("b", "b:1", 18 * FP32),
        ("self", "self:1", (40 + 18 + 12) * FP32),
    ):
        assert int(modules[address].internal_param_memory) == nbytes, address
        assert int(calls[call_label].internal_param_memory) == nbytes, call_label
    # Outer: block holds no parameter itself, its fc 20 elements; head 10.
    _, trace, _, _ = outer_case
    _, _, modules, _ = _by_label(trace)
    expected_param = {
        "self": 30 * FP32,
        "block": 20 * FP32,
        "block.fc": 20 * FP32,
        "head": 10 * FP32,
    }
    for address, nbytes in expected_param.items():
        assert int(modules[address].internal_param_memory) == nbytes, address
    # Each block call returns its mul (3, 4) = 48 B and holds linear and relu
    # inside (48 + 48); two calls.
    assert int(modules["block"].total_output_activation_memory) == 2 * 48
    assert int(modules["block"].total_internal_activation_memory) == 2 * (48 + 48)
    # Op.input_memory: bytes of an Op's saved graph parents (Graph, batch 2).
    ops, _, _, _ = _by_label(graph_trace)
    expected_input = {
        "linear_1_1:1": 2 * 4 * FP32,  # x
        "linear_1_1:2": 2 * 4 * FP32,  # linear pass 1
        "mul_1_2:1": 2 * 4 * FP32 + 4 * FP32,  # linear pass 2 and the scale buffer
        "add_1_5:1": 2 * 4 * FP32 + 4 * FP32,  # mul and ones(4)
        "add_2_6:1": 2 * (2 * 4 * FP32),  # inner add and y
        "output_1:1": 2 * 4 * FP32,
        "input_1:1": 0,  # no parents
    }
    for label, nbytes in expected_input.items():
        assert int(ops[label].input_memory) == nbytes, label


def _param_grad_bytes(model: nn.Module) -> dict[str, int]:
    return {
        name: param.grad.numel() * param.grad.element_size()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def _assert_grad_case_bytes(model, trace, passes: int) -> None:
    ops, _, modules, calls = _by_label(trace)
    # Grad, batch 3: d/d(a out) (3, 8) 96 B, d/d(relu) 96 B, d/d(b out) (3, 2)
    # 24 B, shared by the output copy; the input needs no gradient.
    expected = {
        "input_1:1": 0,
        "linear_1_1:1": 96,
        "relu_1_2:1": 96,
        "linear_2_3:1": 24,
        "output_1:1": 24,
    }
    for label, nbytes in expected.items():
        assert int(ops[label].gradient_memory) == nbytes, label
    # save_grads keeps one gradient record per Op per backward pass
    # (``Op.grads``), so the trace holds every pass's gradients.
    assert int(trace.saved_gradient_memory) == passes * (96 + 96 + 24)
    assert int(trace.total_gradient_memory) == passes * (96 + 96 + 24)
    raw = _param_grad_bytes(model)  # recounted from Tensor.grad
    assert raw == {"a.weight": 128, "a.bias": 32, "b.weight": 64, "b.bias": 8}
    params = {param.address: param for param in trace.param_logs}
    for address, nbytes in raw.items():
        assert int(params[address].gradient_memory) == nbytes, address
    assert int(trace.total_param_gradient_memory) == sum(raw.values())
    for label, pair in (("a:1", (96, 0)), ("b:1", (24, 0)), ("self:1", (24, 96 + 96 + 24))):
        call = calls[label]
        assert (int(call.output_gradient_memory), int(call.internal_gradient_memory)) == pair
    root = modules["self"]
    assert (int(root.total_output_gradient_memory), int(root.total_internal_gradient_memory)) == (
        24,
        216,
    )


def test_gradient_bytes_after_one_backward_pass(grad_once) -> None:
    model, trace, _, _ = grad_once
    _assert_grad_case_bytes(model, trace, passes=1)


def test_gradient_bytes_after_two_backward_passes(grad_twice) -> None:
    # Op.gradient_memory is the latest pass's tensor size; the trace totals
    # count both passes' retained gradients. Parameter .grad accumulates in
    # place, so its size does not change.
    model, trace, _, _ = grad_twice
    _assert_grad_case_bytes(model, trace, passes=2)


def test_gradient_bytes_on_layers_modules_and_buffers(outer_case) -> None:
    _, trace, _, _ = outer_case
    ops, layers, modules, _ = _by_label(trace)
    # Outer, batch 3: linear, relu and mul outputs are (3, 4) = 48 B each, the
    # head (3, 2) = 24 B, the sum (3, 1) = 12 B (shared by the output copy).
    assert int(layers["linear_1_1"].total_gradient_memory) == 2 * 48  # two passes
    assert int(layers["relu_1_2"].total_gradient_memory) == 2 * 48
    # grouping rule rows 26-27: the two block calls' muls are one Layer (mul_1_3, 2 passes).
    assert int(layers["mul_1_3"].total_gradient_memory) == 2 * 48
    assert int(layers["sum_1_5"].total_gradient_memory) == 12
    assert int(trace.total_gradient_memory) == 6 * 48 + 24 + 12
    block = modules["block"]
    assert (int(block.total_output_gradient_memory), int(block.total_internal_gradient_memory)) == (
        2 * 48,
        2 * (48 + 48),
    )
    torch.manual_seed(0)
    buffered = tl.trace(
        GradBuffer(), torch.randn(2, 4, requires_grad=True), capture=CaptureOptions(save_grads=True)
    )
    try:
        buffered.backward(buffered[buffered.output_layers[0]].out.sum())
        by_address = {buffer.address: buffer for buffer in buffered.buffers}
        assert int(by_address["w"].gradient_memory) == 4 * FP32  # d/dw of x * w: (4,)
        assert int(by_address["c"].gradient_memory) == 0  # c needs no gradient
    finally:
        buffered.cleanup()


@pytest.mark.parametrize("case, passes", [("grad_once", 1), ("grad_twice", 2), ("outer_case", 1)])
def test_backward_graph_counts_match_a_raw_recount(case: str, passes: int, request) -> None:
    _, trace, loss, tally = request.getfixturevalue(case)
    nodes, edges = _raw_graph(loss)
    assert int(trace.num_grad_fns) == len(nodes)
    assert int(trace.autograd.num_backward_edges) == len(edges)
    # Every reachable node ran once per pass, by its hook.
    assert set(tally.values()) == {passes} and len(tally) == len(nodes)
    assert int(trace.num_grad_fn_calls) == sum(tally.values())
    # A node pairs with an Op when it is the grad_fn of an Op's own tensor:
    # Grad has addmm, relu, addmm; Outer has addmm, relu, mul per block call,
    # the head's addmm and the sum. The CloneBackward0 between the loss and
    # the model output belongs to the output Op's saved ``.out`` copy (the
    # loss is built from it), not to an Op's tensor: the output Op's own
    # tensor is the model output and shares its parent's node. The input
    # needs no gradient, so the input Op's tensor has no node.
    with_op = 3 if case != "outer_case" else 2 * 3 + 1 + 1
    assert int(trace.num_grad_fns_with_op) == with_op
    assert int(trace.num_grad_fns_without_op) == len(nodes) - with_op


def test_grad_fn_pairing_with_a_grad_requiring_input() -> None:
    # GradBuffer with an input that requires grad. The input Op's own tensor
    # is the capture's copy of x (a CloneBackward0 whose next node is x's
    # AccumulateGrad), so it pairs with the input Op; mul and add pair with
    # their Ops; the CloneBackward0 under the loss is the output Op's saved
    # ``.out`` copy and pairs with nothing; the buffer w and x each end in an
    # AccumulateGrad; the loss sum pairs with nothing.
    torch.manual_seed(0)
    trace = tl.trace(
        GradBuffer(), torch.randn(2, 4, requires_grad=True), capture=CaptureOptions(save_grads=True)
    )
    try:
        loss = trace[trace.output_layers[0]].out.sum()
        nodes, edges = _raw_graph(loss)
        assert sorted(type(node).__name__ for node in nodes) == [
            "AccumulateGrad",
            "AccumulateGrad",
            "AddBackward0",
            "CloneBackward0",
            "CloneBackward0",
            "MulBackward0",
            "SumBackward0",
        ]
        trace.backward(loss)
        assert int(trace.num_grad_fns) == len(nodes) == 7
        assert int(trace.autograd.num_backward_edges) == len(edges) == 6
        assert int(trace.num_grad_fns_with_op) == 3  # input copy, mul, add
        assert int(trace.num_grad_fns_without_op) == 4
    finally:
        trace.cleanup()


@pytest.mark.parametrize("backward_ready", [False, True])
def test_backward_flops_follow_the_backward_that_runs(backward_ready: bool) -> None:
    # TwoLinear: a = Linear(4, 8) frozen, relu, b = Linear(8, 2). Default
    # capture gives frozen parameters a grad during the forward, so a and
    # relu get autograd nodes; backward_ready=True keeps them frozen, so
    # only b has a node. Backward FLOPs are the multiplier estimate where a
    # node runs (a 2*216, relu 24, b 2*102) and zero where none does.
    torch.manual_seed(0)
    model = TwoLinear().eval()
    trace = tl.trace(
        model, torch.randn(3, 4), capture=CaptureOptions(backward_ready=backward_ready)
    )
    try:
        loss = trace[trace.output_layers[0]].out.sum()
        nodes, _ = _raw_graph(loss)
        ran: Tally = Tally()
        for node in nodes:
            node.register_hook(_counting_hook(ran, type(node).__name__))
        loss.backward()
        a_runs = not backward_ready
        assert ran["AddmmBackward0"] == (2 if a_runs else 1)
        assert ran["ReluBackward0"] == (1 if a_runs else 0)
        ops, _, _, _ = _by_label(trace)
        expected = {"linear_1_1:1": 432 if a_runs else 0, "relu_1_2:1": 24 if a_runs else 0}
        expected["linear_2_3:1"] = 204
        for label, flops in expected.items():
            assert int(ops[label].flops_backward) == flops, label
        assert int(trace.cost.total_flops_backward) == sum(expected.values())
    finally:
        trace.cleanup()


# Grad, batch 3: a = Linear(4, 8): 2*3*4*8 + 3*8 = 216; relu: 3*8 = 24;
# b = Linear(8, 2): 2*3*8*2 + 3*2 = 102. Backward: linear 2x, relu 1x.
GRAD_FLOPS = {"linear_1_1": (216, 432), "relu_1_2": (24, 24), "linear_2_3": (102, 204)}


def test_backward_and_total_flops(grad_once, outer_case) -> None:
    _, trace, _, _ = grad_once
    ops, layers, modules, _ = _by_label(trace)
    for label, (forward, backward) in GRAD_FLOPS.items():
        for record in (ops[f"{label}:1"], layers[label]):
            assert int(record.flops_backward) == backward, label
            assert int(record.flops_total) == forward + backward, label
        assert int(layers[label].total_flops_backward) == backward
        assert int(layers[label].total_flops_total) == forward + backward
    backward_sum = sum(backward for _, backward in GRAD_FLOPS.values())  # 660
    forward_sum = sum(forward for forward, _ in GRAD_FLOPS.values())  # 342
    for scope in (trace, modules["self"]):
        assert int(scope.total_flops_backward) == backward_sum
        assert int(scope.total_flops) == forward_sum + backward_sum
    assert (int(modules["a"].total_flops_backward), int(modules["a"].total_flops)) == (432, 648)
    # Outer, batch 3: fc = Linear(4, 4): 2*3*4*4 + 3*4 = 108; relu, mul: 12
    # each; per block call forward 132 and backward 2*108 + 12 + 12 = 240.
    # The block runs twice, so its totals double; fc's Layer has two passes.
    _, trace, _, _ = outer_case
    _, layers, modules, _ = _by_label(trace)
    assert int(layers["linear_1_1"].total_flops_backward) == 2 * 216
    assert (
        int(modules["block.fc"].total_flops_backward),
        int(modules["block.fc"].total_flops),
    ) == (
        2 * 216,
        2 * 324,
    )
    assert (int(modules["block"].total_flops_backward), int(modules["block"].total_flops)) == (
        2 * 240,
        2 * (132 + 240),
    )
    assert int(modules["block.fc"].total_flops_forward) == 2 * 108
    assert int(modules["block"].total_flops_forward) == 2 * 132
    # MACs: fc 3*4*4 = 48 per pass, head 3*4*2 = 24; relu, mul and sum have none.
    assert int(modules["block.fc"].total_macs_forward) == 2 * 48
    assert int(modules["block"].total_macs_forward) == 2 * 48
    assert int(modules["self"].total_macs_forward) == 2 * 48 + 24
    assert int(modules["self"].total_flops_backward) == int(trace.cost.total_flops_backward)


def test_arithmetic_intensity_by_the_read_once_write_once_model(two_linear_trace) -> None:
    ops, _, _, _ = _by_label(two_linear_trace)
    # flops / (input bytes + parameter bytes + output bytes), batch 3.
    # a: 216 / (48 + 40*4 + 96); relu: 24 / (96 + 96); b: 102 / (96 + 18*4 + 24).
    expected = {
        "linear_1_1:1": 216 / (48 + 160 + 96),
        "relu_1_2:1": 24 / (96 + 96),
        "linear_2_3:1": 102 / (96 + 72 + 24),
    }
    for label, value in expected.items():
        assert ops[label].arithmetic_intensity == value, label


def test_summary_counts_unknown_flop_ops() -> None:
    torch.manual_seed(0)
    trace = tl.trace(PadTwice(), torch.randn(2, 3))
    try:
        # Two pad calls have no FLOP rule; relu has one.
        assert int(trace.summary().unknown_flop_ops) == 2
    finally:
        trace.cleanup()
    torch.manual_seed(0)
    trace = tl.trace(Grad().eval(), torch.randn(3, 4))
    try:
        assert int(trace.summary().unknown_flop_ops) == 0  # linear, relu, linear
    finally:
        trace.cleanup()
