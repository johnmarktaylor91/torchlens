"""Multi-exit module lookup collision tests."""

from __future__ import annotations

import torch

import torchlens as tl


class TupleExit(torch.nn.Module):
    """Module returning two output tensors from one module call."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two sibling exit tensors."""

        return x + 1, x * 2


class MultiExitModel(torch.nn.Module):
    """Wrapper whose child module has multiple exits."""

    def __init__(self) -> None:
        """Initialize the tuple-returning child."""

        super().__init__()
        self.split = TupleExit()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume both outputs from the tuple-returning child."""

        left, right = self.split(x)
        return left + right


def _multi_exit_trace() -> tl.Trace:
    """Return a trace with a single-call, multi-exit module."""

    return tl.trace(MultiExitModel(), torch.randn(2, 3))


def _split_exit_ops(trace: tl.Trace) -> list[tl.Op]:
    """Return Ops that exit the ``split`` module."""

    return [
        op
        for op in trace.ops
        if any(str(module_call).startswith("split:") for module_call in op.output_of_module_calls)
    ]


def test_multi_exit_ops_remain_reachable_by_qualified_keys() -> None:
    """Both sibling exits remain reachable through their Op labels."""

    trace = _multi_exit_trace()
    exit_ops = _split_exit_ops(trace)

    assert len(exit_ops) == 2
    for op in exit_ops:
        assert trace[op.label] is op
        assert trace.ops[op.label] is op


def test_raw_label_fetch_returns_exact_multi_exit_op() -> None:
    """Raw-label lookup is not shadowed by the module bare-alias collision."""

    trace = _multi_exit_trace()

    for op in _split_exit_ops(trace):
        assert trace[op.raw_label] is op
        assert trace[op._label_raw] is op
        assert trace.ops[op.raw_label] is op
        assert trace.ops[op._label_raw] is op


def test_raw_lookup_keys_exist_for_all_multi_exit_ops() -> None:
    """Raw lookup keys are registered for every colliding exit op."""

    trace = _multi_exit_trace()

    for op in _split_exit_ops(trace):
        assert op.raw_label in op.lookup_keys
        assert op._label_raw in op.lookup_keys
        assert op.raw_label in trace.layer_dict_all_keys
        assert op._label_raw in trace.layer_dict_all_keys
