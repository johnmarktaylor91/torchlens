"""Perturbation at a TorchScript boundary op whose scripted body reads an argument's metadata only.

A scripted helper that reads only ``x.size(-1)`` (the DeBERTa ``scaled_size_sqrt`` shape)
returns the same value whatever ``x`` holds, so the boundary op is insensitive to that
parent by construction. The validator accepts that edge only through a graph proof that
every use of the argument is a metadata read, and records the proof as its own exemption
reason; a size read inside a nested scripted helper counts (the graph is inlined). Armed
negatives: a helper that reads one element, a helper mixing a size read and an element read,
an element read inside a nested scripted helper, an element read on a branch the call does
not take, and an argument reaching a node kind outside the enumerated list all keep the
perturbation check and still fail it; a parent that fills both a metadata slot and a value
slot gets no proof; plain eager capture is unaffected.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._script_boundary import SCRIPT_FUNCTION_FUNC_NAME
from torchlens.utils._torch_compat import HAS_SCRIPT_FUNCTION_CALL_HOOK
from torchlens.validation import last_validation_failure
from torchlens.validation._script_metadata_proof import (
    METADATA_READ_NODE_KINDS,
    SCRIPT_METADATA_ONLY_REASON,
    metadata_only_arguments,
    script_metadata_only_proof,
)
from torchlens.validation.diagnostics import CHECK_PERTURBATION

pytestmark = pytest.mark.skipif(
    not HAS_SCRIPT_FUNCTION_CALL_HOOK, reason="torch build without a ScriptFunction call hook"
)


def _plain_scaled_size_sqrt(query_layer: torch.Tensor, scale_factor: int) -> torch.Tensor:
    return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)


scaled_size_sqrt = torch.jit.script(_plain_scaled_size_sqrt)


@torch.jit.script
def metadata_reads(x: torch.Tensor) -> torch.Tensor:
    """Every enumerated metadata read of one argument, none of its elements."""

    rank = x.dim() + x.numel() * 0
    return torch.full([x.size(-1)], float(rank), dtype=x.dtype, device=x.device)


@torch.jit.script
def one_element_read(x: torch.Tensor) -> torch.Tensor:
    """Reads one element; finite perturbations happen not to move the output."""

    first = x.view(-1)[0]
    return (first != first).to(torch.float) + 8.0


@torch.jit.script
def size_and_element_read(x: torch.Tensor) -> torch.Tensor:
    """A size read plus an element read whose effect finite perturbations do not show."""

    first = x.view(-1)[0]
    size_term = torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float))
    return size_term + (first != first).to(torch.float)


@torch.jit.script
def nested_size_call(x: torch.Tensor) -> torch.Tensor:
    """Hands the argument to a scripted helper that reads only its size (proved inlined)."""

    return scaled_size_sqrt(x, 2)


@torch.jit.script
def nested_element_call(x: torch.Tensor) -> torch.Tensor:
    """Hands the argument to a scripted helper that reads one element."""

    return one_element_read(x)


@torch.jit.script
def untaken_branch_element_read(x: torch.Tensor) -> torch.Tensor:
    """Reads an element only on a branch that a 2-D argument never takes."""

    if x.dim() > 2:
        return x.view(-1)[0] * 0.0 + 8.0
    return torch.tensor(8.0)


@torch.jit.script
def is_cuda_read(x: torch.Tensor) -> torch.Tensor:
    """Reads ``x.is_cuda`` (``prim::is_cuda``), a metadata read deliberately not enumerated."""

    return torch.sqrt(torch.tensor(x.size(-1), dtype=torch.float)) + float(x.is_cuda)


@torch.jit.script
def scale_by_last_dim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Argument ``x`` is read for its size, ``y`` for its values."""

    return y * x.size(-1)


class _HelperModel(nn.Module):
    """Linear, then ``h / helper(h)``: the helper's only parent is ``h``."""

    def __init__(self, helper) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.helper = helper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        return h / self.helper(h)


class _ScaledModel(_HelperModel):
    """The DeBERTa shape: divide by ``sqrt(size(-1) * scale)``."""

    def __init__(self, helper=None) -> None:
        super().__init__(scaled_size_sqrt if helper is None else helper)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        return h / self.helper(h, 2)


class _TwoArgModel(nn.Module):
    """``scale_by_last_dim(a, b)``; ``same`` passes one tensor to both slots."""

    def __init__(self, same: bool) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.same = same

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        g = h if self.same else torch.tanh(h)
        return scale_by_last_dim(h, g)


def _trace_and_validate(model: nn.Module, x: torch.Tensor) -> tuple[tl.Trace, bool]:
    """Capture with every activation saved, then run forward validation on the trace."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(
            model, x, capture=tl.options.CaptureOptions(layers_to_save="all", save_arg_values=True)
        )
        result = trace.validate_forward_pass([model(x)], validate_metadata=False)
    return trace, result


def _boundary_op(trace: tl.Trace):
    """Return the trace's one ``jitscript`` op."""

    (op,) = [op for op in trace.layer_list if op.func_name == SCRIPT_FUNCTION_FUNC_NAME]
    return op


def _metadata_decisions(trace: tl.Trace) -> list[dict]:
    """Return the recorded metadata-only exemption decisions."""

    return [
        item
        for item in trace.validation_replay_status.decisions
        if item["reason"] == SCRIPT_METADATA_ONLY_REASON
    ]


def _assert_boundary_perturbation_failure(trace: tl.Trace, result: bool) -> None:
    """Assert validation failed perturbation at the boundary op with no metadata proof."""

    assert result is False
    failure = last_validation_failure()
    assert failure is not None and failure.check == CHECK_PERTURBATION
    assert failure.func_name == SCRIPT_FUNCTION_FUNC_NAME
    assert _metadata_decisions(trace) == []


def test_shape_only_helper_passes_with_a_recorded_proof() -> None:
    """The DeBERTa shape validates, and the exemption is recorded with its proof."""

    trace, result = _trace_and_validate(_ScaledModel(), torch.randn(3, 4))

    assert result is True
    (decision,) = _metadata_decisions(trace)
    assert decision["decision"] == "exempted"
    assert decision["phase"] == "perturbation"
    assert decision["op_label"] == _boundary_op(trace).label
    assert scaled_size_sqrt.qualified_name in decision["justification"]
    assert "query_layer only through aten::size" in decision["justification"]
    assert trace.validation_replay_status.exempted_reason_counts[SCRIPT_METADATA_ONLY_REASON] == 1


def test_graph_proof_names_exactly_the_metadata_only_arguments() -> None:
    """The proof covers every enumerated read and nothing that touches elements."""

    assert metadata_only_arguments(scaled_size_sqrt) == {"query_layer": frozenset({"aten::size"})}
    assert metadata_only_arguments(metadata_reads) == {"x": METADATA_READ_NODE_KINDS}
    assert metadata_only_arguments(scale_by_last_dim) == {"x": frozenset({"aten::size"})}
    assert metadata_only_arguments(nested_size_call) == {"x": frozenset({"aten::size"})}
    for function in (
        one_element_read,
        size_and_element_read,
        nested_element_call,
        untaken_branch_element_read,
        is_cuda_read,
    ):
        assert metadata_only_arguments(function) == {}


def test_every_enumerated_metadata_read_passes() -> None:
    """size, dim, numel, dtype and device reads together still give the proof."""

    trace, result = _trace_and_validate(_HelperModel(metadata_reads), torch.randn(3, 4))

    assert result is True
    assert len(_metadata_decisions(trace)) == 1


def test_one_element_read_still_fails_perturbation() -> None:
    """Armed negative: an element read gets no proof, so the insensitive edge fails."""

    trace, result = _trace_and_validate(_HelperModel(one_element_read), torch.randn(3, 4))
    _assert_boundary_perturbation_failure(trace, result)


def test_size_read_mixed_with_an_element_read_still_fails_perturbation() -> None:
    """Armed negative: one value use anywhere voids the proof for that argument."""

    trace, result = _trace_and_validate(_HelperModel(size_and_element_read), torch.randn(3, 4))
    _assert_boundary_perturbation_failure(trace, result)


def test_unknown_node_kind_keeps_the_perturbation_check() -> None:
    """Armed negative: ``x.is_cuda`` is metadata too, but ``prim::is_cuda`` is not enumerated."""

    trace, result = _trace_and_validate(_HelperModel(is_cuda_read), torch.randn(3, 4))
    _assert_boundary_perturbation_failure(trace, result)


def test_size_read_inside_a_nested_scripted_helper_passes() -> None:
    """The nested helper is inlined, so its size-only read gives the proof."""

    trace, result = _trace_and_validate(_HelperModel(nested_size_call), torch.randn(3, 4))

    assert result is True
    (decision,) = _metadata_decisions(trace)
    assert nested_size_call.qualified_name in decision["justification"]


def test_element_read_inside_a_nested_scripted_helper_still_fails_perturbation() -> None:
    """Armed negative: inlining exposes the nested helper's element read."""

    trace, result = _trace_and_validate(_HelperModel(nested_element_call), torch.randn(3, 4))
    _assert_boundary_perturbation_failure(trace, result)


def test_element_read_on_an_untaken_branch_still_fails_perturbation() -> None:
    """Armed negative: the proof is not path-sensitive; any value use voids it."""

    trace, result = _trace_and_validate(
        _HelperModel(untaken_branch_element_read), torch.randn(3, 4)
    )
    _assert_boundary_perturbation_failure(trace, result)


def test_parent_in_a_metadata_slot_and_a_value_slot_gets_no_proof() -> None:
    """A parent that also feeds a value-bearing argument stays perturbation-tested."""

    trace, result = _trace_and_validate(_TwoArgModel(same=True), torch.randn(3, 4))
    assert result is True
    op = _boundary_op(trace)
    (parent,) = op.parents
    assert script_metadata_only_proof(op, [parent]) is None
    assert _metadata_decisions(trace) == []


def test_distinct_parents_split_into_proven_and_tested() -> None:
    """With two parents only the size-read one is exempt; the value parent is tested."""

    trace, result = _trace_and_validate(_TwoArgModel(same=False), torch.randn(3, 4))
    assert result is True
    op = _boundary_op(trace)
    size_parent, value_parent = (op.parent_arg_positions["args"][pos] for pos in (0, 1))
    assert script_metadata_only_proof(op, [size_parent]) is not None
    assert script_metadata_only_proof(op, [value_parent]) is None
    (decision,) = _metadata_decisions(trace)
    assert "reads argument(s) x only" in decision["justification"]


def test_unreadable_graph_gives_no_proof() -> None:
    """An object without a readable graph or schema proves nothing."""

    class _NoGraph:
        @property
        def graph(self):
            raise RuntimeError("graph unavailable")

    assert metadata_only_arguments(_NoGraph()) == {}
    assert metadata_only_arguments(object()) == {}


def test_plain_eager_capture_is_unaffected() -> None:
    """The same helper as plain Python: op by op, no boundary, no metadata-only exemption."""

    model = _ScaledModel(helper=_plain_scaled_size_sqrt)
    trace, result = _trace_and_validate(model, torch.randn(3, 4))

    assert result is True
    assert not any(op.func_name == SCRIPT_FUNCTION_FUNC_NAME for op in trace.layer_list)
    assert _metadata_decisions(trace) == []
    for op in trace.layer_list:
        for parent in op.parents:
            assert script_metadata_only_proof(op, [parent]) is None
