"""Edge-count introspection coverage for Trace, Module, ModuleCall, Op, and Layer."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


def _trace(model: nn.Module, inputs: torch.Tensor, **kwargs: object) -> tl.Trace:
    """Capture a TorchLens trace for edge-count tests.

    Parameters
    ----------
    model:
        Module to trace.
    inputs:
        Tensor input for the model.
    **kwargs:
        Additional keyword arguments forwarded to ``torchlens.trace``.

    Returns
    -------
    tl.Trace
        Captured trace.
    """

    return tl.trace(model, inputs, save_raw_activations=False, **kwargs)


def _canonical_child_edges(trace: tl.Trace) -> set[tuple[str, str]]:
    """Return distinct graph edges by walking Op children."""

    return {
        (op.label, trace.ops[child_label].label) for op in trace.ops for child_label in op.children
    }


def _canonical_parent_edges(trace: tl.Trace) -> set[tuple[str, str]]:
    """Return distinct graph edges by walking Op parents."""

    return {
        (trace.ops[parent_label].label, op.label) for op in trace.ops for parent_label in op.parents
    }


def _assert_edge_invariants(trace: tl.Trace) -> None:
    """Assert child-walk and parent-walk edge counts match ``Trace.num_edges``."""

    child_edges = _canonical_child_edges(trace)
    parent_edges = _canonical_parent_edges(trace)
    assert child_edges == parent_edges
    assert sum(len({trace.ops[child].label for child in op.children}) for op in trace.ops) == len(
        child_edges
    )
    assert sum(len({trace.ops[parent].label for parent in op.parents}) for op in trace.ops) == len(
        parent_edges
    )
    assert len(child_edges) == trace.num_edges


class Chain(nn.Module):
    """Linear chain of Linear and ReLU ops."""

    def __init__(self, num_pairs: int) -> None:
        """Initialize the chain.

        Parameters
        ----------
        num_pairs:
            Number of Linear/ReLU pairs.
        """

        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(num_pairs):
            layers.extend([nn.Linear(4, 4), nn.ReLU()])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the linear chain."""

        return self.net(x)


class ComputeReuse(nn.Module):
    """Reuse a compute output across two downstream branches."""

    def __init__(self) -> None:
        """Initialize the reusable compute source."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a small residual-style fan-out/fan-in graph."""

        y = torch.relu(self.lin(x))
        left = y * 2
        right = y + 3
        return left + right


class NWayCombine(nn.Module):
    """Combine N independent branches with stack then sum."""

    def __init__(self, num_branches: int) -> None:
        """Initialize branch linears.

        Parameters
        ----------
        num_branches:
            Number of branches to combine.
        """

        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(num_branches)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all branches and combine their outputs."""

        return torch.stack([layer(x) for layer in self.layers], dim=0).sum(dim=0)


class RecurrentCell(nn.Module):
    """Small recurrent reassignment cell."""

    def __init__(self) -> None:
        """Initialize the recurrent linear."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the same cell three times."""

        h = x
        for _ in range(3):
            h = torch.relu(self.lin(h))
        return h


class NestedBlock(nn.Module):
    """Nested block used for Module and ModuleCall edge checks."""

    def __init__(self) -> None:
        """Initialize nested block layers."""

        super().__init__()
        self.lin1 = nn.Linear(4, 4)
        self.lin2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested block."""

        return self.lin2(torch.relu(self.lin1(x)))


class NestedModel(nn.Module):
    """Model with one nested block between two leaf linears."""

    def __init__(self) -> None:
        """Initialize the nested model."""

        super().__init__()
        self.pre = nn.Linear(4, 4)
        self.block = NestedBlock()
        self.post = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested model."""

        return self.post(self.block(self.pre(x)))


class BufferModel(nn.Module):
    """BatchNorm model that reads and writes buffers in train mode."""

    def __init__(self) -> None:
        """Initialize BatchNorm."""

        super().__init__()
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run BatchNorm."""

        return self.bn(x)


class BackwardModel(nn.Module):
    """Tiny model used for backward edge checks."""

    def __init__(self) -> None:
        """Initialize the projection."""

        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar loss-like output."""

        return self.lin(x).sum()


def test_linear_chain_edge_counts() -> None:
    """Linear chain has hand-counted sentinel and compute edges."""

    trace = _trace(Chain(num_pairs=2).eval(), torch.randn(1, 4))

    assert trace.num_edges == 5
    assert trace.num_compute_edges == 3
    assert trace.branching_factor == pytest.approx(1.25)
    assert trace.max_in_degree == 1
    assert trace.max_out_degree == 1
    assert trace.num_layer_edges == 5
    _assert_edge_invariants(trace)


def test_residual_reuse_counts_skip_edge() -> None:
    """Compute-output reuse contributes multiple outgoing edges."""

    trace = _trace(ComputeReuse().eval(), torch.randn(1, 4))

    assert trace.num_edges == 7
    assert trace.num_compute_edges == 5
    assert trace.max_in_degree == 2
    assert trace.max_out_degree == 2
    assert ("relu_1_2:1", "mul_1_3:1") in _canonical_child_edges(trace)
    assert ("relu_1_2:1", "add_1_4:1") in _canonical_child_edges(trace)
    _assert_edge_invariants(trace)


def test_n_way_combine_max_in_degree() -> None:
    """N-way stack records N parents on the combine op."""

    trace = _trace(NWayCombine(num_branches=3).eval(), torch.randn(1, 4))

    assert trace.num_edges == 8
    assert trace.num_compute_edges == 4
    assert trace.max_in_degree == 3
    assert trace.max_out_degree == 1
    _assert_edge_invariants(trace)


def test_recurrent_reassignment_layer_edges_collapse_pass_edges() -> None:
    """Rolling recurrent layers has fewer layer edges than per-pass Op edges."""

    trace = _trace(RecurrentCell().eval(), torch.randn(1, 4))

    assert trace.num_edges == 7
    assert trace.num_layer_edges == 4
    assert trace.num_layer_edges < trace.num_edges
    _assert_edge_invariants(trace)


def test_module_and_module_call_edge_footprints() -> None:
    """Nested Module and ModuleCall edge footprints are exact."""

    trace = _trace(NestedModel().eval(), torch.randn(1, 4))

    block_call = trace.module_calls["block:1"]
    assert block_call.num_internal_edges == 2
    assert block_call.num_input_edges == 1
    assert block_call.num_output_edges == 1
    assert block_call.num_edges == 4

    block = trace.modules["block"]
    assert block.num_internal_edges == 2
    assert block.num_input_edges == 1
    assert block.num_output_edges == 1
    assert block.num_edges == 4

    pre_call = trace.module_calls["pre:1"]
    assert pre_call.num_internal_edges == 0
    assert pre_call.num_input_edges == 1
    assert pre_call.num_output_edges == 1
    assert pre_call.num_edges == 2
    _assert_edge_invariants(trace)


def test_buffer_edges_are_counted() -> None:
    """BatchNorm train-mode buffer reads and writes contribute buffer edges."""

    trace = _trace(BufferModel().train(), torch.randn(3, 4))

    assert trace.num_buffer_edges == 6
    assert trace.num_buffer_edges > 0
    _assert_edge_invariants(trace)


def test_backward_edges_gate_on_backward_capture() -> None:
    """Backward edge count is None until backward data has been logged."""

    no_backward = _trace(BackwardModel(), torch.randn(2, 4, requires_grad=True))
    assert no_backward.num_backward_edges is None

    trace = _trace(
        BackwardModel(),
        torch.randn(2, 4, requires_grad=True),
        backward_ready=True,
        save_gradients=True,
    )
    assert trace.num_backward_edges is None
    loss = trace[trace.output_layers[0]].out
    trace.backward(loss, retain_graph=True)
    assert trace.num_backward_edges is not None
    assert trace.num_backward_edges > 0
    _assert_edge_invariants(trace)
