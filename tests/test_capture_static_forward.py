"""A parameter-free loop with a ``staticmethod`` forward captures like any module.

``example_models.SimpleLoopNoParam`` computes ``log(sin(log(x + 2)))`` three
times. Its plain ``torch.rand`` input drives a log argument negative, so the
eager forward is all NaN and a validation on it fails on the degenerate
forward, not on the capture. On an input that keeps every log argument
positive the exhaustive capture and a cooked recording group the loop
identically and forward validation passes.
"""

from __future__ import annotations

import example_models
import torch

import torchlens as tl


def _in_domain_input() -> torch.Tensor:
    torch.manual_seed(0)
    return 20.25 + 0.1 * torch.rand(5, 5)


def _grouping(trace) -> list[tuple[str, int, int]]:
    return [(op.layer_label, op.pass_number, op.num_passes) for op in trace.ops]


def test_static_forward_loop_groups_the_same_both_ways() -> None:
    x = _in_domain_input()
    exhaustive = tl.trace(example_models.SimpleLoopNoParam().eval(), x)
    recording = tl.record(
        example_models.SimpleLoopNoParam().eval(), x, save=lambda ctx: ctx.kind == "op"
    )
    cooked = recording.to_trace()

    assert torch.isfinite(exhaustive["output_1"].out).all()
    assert _grouping(cooked) == _grouping(exhaustive)
    assert len(cooked.layers) == len(exhaustive.layers)
    log_ops = [op for op in exhaustive.ops if op.func_name == "log"]
    assert [op.pass_number for op in log_ops] == [1, 2, 3]
    assert {op.num_passes for op in log_ops} == {3}


def test_static_forward_loop_validates_forward_replay() -> None:
    assert tl.validate(example_models.SimpleLoopNoParam(), _in_domain_input(), scope="forward")


class _ClassmethodLoop(torch.nn.Module):
    """The same parameter-free loop with a ``classmethod`` forward."""

    @classmethod
    def forward(cls, x: torch.Tensor) -> torch.Tensor:
        x = x + 2
        for _ in range(3):
            x = torch.log(x)
            x = torch.sin(x)
        return x + 3


def test_static_forward_traces_on_the_plain_input_too() -> None:
    """A NaN forward is still a captured forward: the loop traces on ``torch.rand`` input."""

    torch.manual_seed(0)
    trace = tl.trace(example_models.SimpleLoopNoParam(), torch.rand(5, 5))
    assert len(trace.layers) == len(
        tl.trace(example_models.SimpleLoopNoParam(), _in_domain_input()).layers
    )
    assert [op.pass_number for op in trace.ops if op.func_name == "log"] == [1, 2, 3]


def test_classmethod_forward_loop_groups_the_same_both_ways() -> None:
    x = _in_domain_input()
    exhaustive = tl.trace(_ClassmethodLoop().eval(), x)
    cooked = tl.record(_ClassmethodLoop().eval(), x, save=lambda ctx: ctx.kind == "op").to_trace()

    assert _grouping(cooked) == _grouping(exhaustive)
    assert _grouping(exhaustive) == _grouping(
        tl.trace(example_models.SimpleLoopNoParam().eval(), x)
    )
    assert tl.validate(_ClassmethodLoop(), x, scope="forward")
