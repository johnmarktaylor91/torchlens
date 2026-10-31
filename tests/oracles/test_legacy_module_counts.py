"""Hand-counted oracles for the legacy module-tree, parameter-use and conditional quantities.

Every expected value is counted by hand from the fixture's ``forward`` source in
the comment beside it, or recounted from raw torch state
(``named_parameters(remove_duplicate=False)``), never read from TorchLens. The
call-tree model: the root call is ``self:1``; a call's descendants are the calls
made while it is on the stack; an Op's module-call depth is the number of
non-root calls on the stack when it ran. These are the I2 oracles the census
(``tests/oracles/data/legacy_quantities.tsv``) names for those doors.
"""

from __future__ import annotations

import gc
import pickle

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.oracles.test_legacy_graph_counts import (
    G_LAYERS,
    G_OPS,
    Counter,
    Graph,
    LoopedBatchNorm,
    Mixed,
)


class Block(nn.Module):
    """Linear -> ReLU -> Linear."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(4, 4)
        self.l2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(torch.relu(self.l1(x)))


class Nest(nn.Module):
    """Two Blocks, a grandchild called again from the root, a never-called child, one frozen weight."""

    def __init__(self) -> None:
        super().__init__()
        self.b1 = Block()
        self.b2 = Block()
        self.dead = nn.Linear(4, 4)  # registered, never called
        self.b1.l2.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.b2(self.b1(x))  # calls b1:1 (l1:1, l2:1), then b2:1 (l1:1, l2:1)
        return h + self.b1.l1(x)  # b1.l1:2, a direct root child; add at the root


class Cond(nn.Module):
    """Two nested ``if`` statements on tensor bools."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2  # mul_1_1, before both ifs
        if x.sum() > 0:  # sum_1_2, gt_1_3 (the bool conditional 1 consumes)
            y = y + 1  # add_1_4: input y from outside the arm (an entry edge)
            if y.mean() > 0:  # mean_1_5, gt_2_6 (the bool conditional 2 consumes)
                y = y * 3  # mul_2_7: input from outside conditional 2's arm
        return y.relu()  # relu_1_8, after both


# Calls of Nest by hand: self:1 > {b1:1 > {b1.l1:1, b1.l2:1}, b2:1 > {b2.l1:1, b2.l2:1}, b1.l1:2}.
NEST_DESCENDANTS = {
    "self:1": 7,
    "b1:1": 2,
    "b2:1": 2,
    "b1.l1:1": 0,
    "b1.l1:2": 0,
    "b1.l2:1": 0,
    "b2.l1:1": 0,
    "b2.l2:1": 0,
}
NEST_DEPTH = {"self:1": 2, "b1:1": 1, "b2:1": 1}  # leaves 0
NEST_MODULE_CALLS = {"self": 1, "b1": 1, "b2": 1, "b1.l1": 2, "b1.l2": 1, "b2.l1": 1, "b2.l2": 1}
# Op module-call depth: non-root calls on the stack (label -> depth).
NEST_OP_DEPTH = {
    "input_1:1": 0,
    "linear_1_1:1": 2,  # b1 > b1.l1
    "relu_1_2:1": 1,  # b1 (Block.forward's relu)
    "linear_2_3:1": 2,  # b1 > b1.l2
    "linear_3_4:1": 2,  # b2 > b2.l1
    "relu_2_5:1": 1,  # b2
    "linear_4_6:1": 2,  # b2 > b2.l2
    # grouping rule row 27 (fork 1, module identity): b1.l1 called again from the
    # root is the same Layer's second pass, so its second call is linear_1_1:2.
    "linear_1_1:2": 1,  # b1.l1:2 straight from the root
    "add_1_7:1": 0,
    "output_1:1": 0,
}
# Ops consuming each parameter: b1.l1 runs twice; dead never runs.
NEST_PARAM_CALLS = {
    "b1.l1.weight": 2,
    "b1.l1.bias": 2,
    "b1.l2.weight": 1,
    "b1.l2.bias": 1,
    "b2.l1.weight": 1,
    "b2.l1.bias": 1,
    "b2.l2.weight": 1,
    "b2.l2.bias": 1,
    "dead.weight": 0,
    "dead.bias": 0,
}
# Address-recursive parameter records (one per nn.Parameter): (all, trainable, frozen).
# Five Linears of 2 tensors each; only b1.l2.weight is frozen.
NEST_RECURSIVE = {
    "self": (10, 9, 1),  # dead's two count: they sit in the address subtree
    "b1": (4, 3, 1),
    "b2": (4, 4, 0),
    "b1.l1": (2, 2, 0),
    "b1.l2": (2, 1, 1),
    "b2.l1": (2, 2, 0),
    "b2.l2": (2, 2, 0),
}


@pytest.fixture(scope="module")
def nest_trace():
    torch.manual_seed(0)
    model = Nest()
    trace = tl.trace(model, torch.randn(2, 4))
    yield model, trace
    trace.cleanup()


def test_module_call_tree_counts(nest_trace) -> None:
    _model, trace = nest_trace
    for call in trace.module_calls:
        assert call.num_descendant_calls == NEST_DESCENDANTS[call.call_label], call.call_label
        assert call.max_descendant_depth == NEST_DEPTH.get(call.call_label, 0), call.call_label
    for module in trace.modules:
        calls = [label for label in NEST_DESCENDANTS if label.rsplit(":", 1)[0] == module.address]
        assert module.num_calls == NEST_MODULE_CALLS[module.address], module.address
        assert module.num_descendant_calls == sum(NEST_DESCENDANTS[c] for c in calls)
        assert module.max_descendant_depth == max(NEST_DEPTH.get(c, 0) for c in calls)
    for op in trace.ops:
        assert op.module_call_depth == NEST_OP_DEPTH[op.label], op.label
    # Layer depth reads its first pass (b1.l1's Layer spans b1.l1:1 and b1.l1:2).
    for layer in trace.layers:
        assert layer.module_call_depth == NEST_OP_DEPTH[f"{layer.layer_label}:1"], layer.layer_label


def test_param_call_counts(nest_trace) -> None:
    _model, trace = nest_trace
    assert {param.address: param.num_calls for param in trace.params} == NEST_PARAM_CALLS


def test_recursive_param_counts(nest_trace) -> None:
    _model, trace = nest_trace
    for module in trace.modules:
        total, trainable, frozen = NEST_RECURSIVE[module.address]
        assert module.num_recursive_params == total, module.address
        assert module.num_recursive_params_trainable == trainable, module.address
        assert module.num_recursive_params_frozen == frozen, module.address
        assert module.num_recursive_param_tensors == total, module.address
        assert module.num_recursive_param_tensors_trainable == trainable, module.address
        assert module.num_recursive_param_tensors_frozen == frozen, module.address


def test_params_by_path_recount(nest_trace) -> None:
    model, trace = nest_trace
    by_path = sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False))
    assert by_path == 5 * (4 * 4 + 4)  # five Linear(4, 4), no ties: 100
    assert trace.params.numel_by_path == by_path
    assert trace.summary().totals.params_per_path_total == by_path

    torch.manual_seed(0)
    mixed = Mixed()  # tied registered under two names
    tied = tl.trace(mixed, torch.randn(2, 4))
    try:
        by_path = sum(p.numel() for _, p in mixed.named_parameters(remove_duplicate=False))
        assert by_path == 20 + 2 * 20  # wide once, tied once per path: 60
        assert tied.params.numel_by_path == by_path
        assert tied.summary().totals.params_per_path_total == by_path
        assert tied.num_params == 40  # unique elements: the shared tensors once
        assert int(tied.total_param_memory) == 20 * 8 + 20 * 4  # float64 + float32, once
        tied_param = tied.params["tied.weight"]
        assert sorted(tied_param.all_addresses) == ["tied.weight", "tied2.weight"]
    finally:
        tied.cleanup()


def test_layer_and_op_counts_on_the_root_call() -> None:
    torch.manual_seed(0)
    model = Graph().eval()
    trace = tl.trace(model, (torch.randn(2, 4), torch.randn(2, 4)))
    try:
        root = trace.module_calls["self:1"]
        assert root.num_ops == G_OPS  # 13: the Linear ran twice
        assert root.num_layers == G_LAYERS  # 12: its two passes share one Layer
        assert len(trace.layers) == G_LAYERS
        assert len(trace.modules["self"].layer_labels) == G_LAYERS
        assert len(trace.modules["fc"].layer_labels) == 1  # one Layer, two passes
        for label in ("fc:1", "fc:2"):
            assert len(trace.module_calls[label].ops) == 1
            assert trace.module_calls[label].num_layers == 1
    finally:
        trace.cleanup()


@pytest.mark.parametrize(
    ("factory", "train", "expected"),
    [
        (Graph, False, 1),  # scale
        # Outside a loop each buffer version is its own Layer: count read,
        # count written version, scale read.
        (Counter, False, 3),
        # In the loop each buffer's read and three written versions are four
        # passes of one Layer: running_mean, running_var, num_batches_tracked, count.
        (LoopedBatchNorm, True, 4),
    ],
)
def test_buffer_layer_counts(factory, train, expected) -> None:
    torch.manual_seed(0)
    model = factory().train(train)
    inputs = (torch.randn(2, 4), torch.randn(2, 4)) if factory is Graph else torch.randn(3, 4)
    trace = tl.trace(model, inputs)
    try:
        assert trace.num_buffer_layers == expected
    finally:
        trace.cleanup()


def test_uncalled_module_count_survives_collection_and_is_unknown_after_pickle() -> None:
    torch.manual_seed(0)
    model = Graph().eval()  # idle is registered and never called
    trace = tl.trace(model, (torch.randn(2, 4), torch.randn(2, 4)))
    try:
        assert trace.num_uncalled_modules == 1
        del model
        gc.collect()
        assert trace.num_uncalled_modules == 1
        assert list(trace.uncalled_modules) == ["idle"]
        assert pickle.loads(pickle.dumps(trace)).num_uncalled_modules is None
    finally:
        trace.cleanup()
    torch.manual_seed(0)
    mixed = Mixed()  # tied2 is tied under a second name: called, not uncalled
    tied = tl.trace(mixed, torch.randn(2, 4))
    try:
        assert tied.num_uncalled_modules == 0
    finally:
        tied.cleanup()


# Cond by hand: two conditionals. Direct roles: gt_1_3 (evaluation of 1),
# add_1_4 (entry of 1's arm), gt_2_6 (evaluation of 2), mul_2_7 (entry of 2's arm).
COND_DEPTH = {"gt_1_3": 1, "add_1_4": 1, "gt_2_6": 1, "mul_2_7": 1}
# Nesting: the ops written inside arm 1 sit one deep, inside arm 2 two deep.
COND_BRANCH_DEPTH = {"add_1_4": 1, "mean_1_5": 1, "gt_2_6": 1, "mul_2_7": 2}


def test_conditional_counts() -> None:
    trace = tl.trace(Cond(), torch.ones(2, 3))
    try:
        assert len(trace.conditionals) == 2
        for op in trace.ops:
            assert op.conditional_depth == COND_DEPTH.get(op.layer_label, 0), op.label
            expected = COND_BRANCH_DEPTH.get(op.layer_label, 0)
            assert op.conditional_branch_depth == expected, op.label
        for layer in trace.layers:
            expected = COND_DEPTH.get(layer.layer_label, 0)
            assert layer.conditional_depth == expected, layer.layer_label
    finally:
        trace.cleanup()


class Boom(nn.Module):
    """Raises inside the forward, after one op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = x * 2
        raise RuntimeError("boom")


def _root_args_held(trace: object) -> object:
    workspace = getattr(trace, "__dict__", {}).get("_module_capture_ws")
    return getattr(workspace, "root_forward_args", None)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_root_forward_arguments_do_not_outlive_the_capture() -> None:
    torch.manual_seed(0)
    trace = tl.trace(Graph().eval(), (torch.randn(2, 4), torch.randn(2, 4)))
    try:
        assert _root_args_held(trace) is None
        root = trace.module_calls["self:1"]
        assert root.forward_args is None  # released like every call's (GC-11)
        assert root.num_forward_pos_args == 2  # the count survives the release
    finally:
        trace.cleanup()
    with pytest.raises(RuntimeError) as failure:
        tl.trace(Boom(), torch.randn(2, 4))
    partial = getattr(failure.value, "partial_log", None)
    assert _root_args_held(partial) is None
