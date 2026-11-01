"""The ModuleCall Op and Layer checks catch every corruption of the call's Op list or counts.

The root ``self:1`` call lists pass-qualified Op labels like every other call,
and ``ModuleCall.num_layers`` counts the distinct Layers among them, so a call
that runs one Layer twice has fewer Layers than Ops. The function-root check
compares the root call with ``trace.op_labels`` (exact order) and its Layer
projection with ``trace.layer_labels``; the containment check compares
``num_layers`` with the call's own Op list. Each corruption
below must raise; each clean multi-pass trace must pass.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.oracles.test_legacy_graph_counts import Graph
from torchlens._data_classes.module import ModuleCall
from torchlens.errors import MetadataInvariantError
from torchlens.validation import check_metadata_invariants


class RootLoop(nn.Module):
    """A parameter-free loop at the root: repeated ops grouped into multi-pass Layers."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.tanh(x * 2)
        return x


class RootOnly(nn.Module):
    """Root-only compute: the shape a function-root backend produces."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x * 2) + 1


def _function_root(model: nn.Module) -> tl.Trace:
    """Trace ``model`` and route it through the backend-neutral function-root checks."""

    torch.manual_seed(0)
    trace = tl.trace(model, torch.randn(2, 4))
    trace._provenance_backend = "jax"
    trace._options_module_identity_mode = "function_root"
    trace._provenance_param_source = "none"
    return trace


@pytest.mark.parametrize("model_cls", [RootOnly, RootLoop])
def test_clean_function_root_traces_pass(model_cls) -> None:
    trace = _function_root(model_cls())
    try:
        if model_cls is RootLoop:
            assert len(trace.layer_labels) < len(trace.op_labels)  # passes share a Layer
        assert check_metadata_invariants(trace) is True
    finally:
        trace.cleanup()


@pytest.mark.parametrize("corruption", ["drop", "duplicate", "reorder"])
@pytest.mark.parametrize("model_cls", [RootOnly, RootLoop])
def test_corrupted_function_root_call_ops_raise(model_cls, corruption: str) -> None:
    trace = _function_root(model_cls())
    try:
        root_call = trace.modules["self"].ops.get(1)
        ops = list(root_call.ops)
        if corruption == "drop":
            root_call.ops = ops[:-1]
        elif corruption == "duplicate":
            root_call.ops = [*ops[:2], ops[1], *ops[2:]]
        else:
            root_call.ops = [ops[1], ops[0], *ops[2:]]
        with pytest.raises(MetadataInvariantError, match="self:1 ops must"):
            check_metadata_invariants(trace)
    finally:
        trace.cleanup()


@pytest.fixture
def graph_trace():
    torch.manual_seed(0)
    model = Graph().eval()  # the root runs fc twice: 12 Layers, 13 Ops
    trace = tl.trace(model, (torch.randn(2, 4), torch.randn(2, 4)))
    yield trace
    trace.cleanup()


def test_clean_multi_pass_root_passes_containment(graph_trace) -> None:
    root = graph_trace.module_calls["self:1"]
    assert root.num_layers < len(root.ops)
    assert check_metadata_invariants(graph_trace) is True


@pytest.mark.parametrize("count", ["num_layers"])
def test_off_by_one_call_counts_raise(graph_trace, monkeypatch, count: str) -> None:
    true_count = getattr(ModuleCall, count)
    monkeypatch.setattr(ModuleCall, count, property(lambda call: true_count.fget(call) + 1))
    with pytest.raises(MetadataInvariantError, match=f"{count}="):
        check_metadata_invariants(graph_trace)


def test_op_label_without_a_layer_raises(graph_trace) -> None:
    root = graph_trace.module_calls["self:1"]
    root.ops = [*root.ops, "not_an_op_9_9:1"]
    with pytest.raises(MetadataInvariantError):
        check_metadata_invariants(graph_trace)
