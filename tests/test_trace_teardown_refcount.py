"""A dropped Trace with recorded conditionals is freed by reference counting alone.

Each ``ConditionalArm`` used to hold its Trace strongly (Trace -> conditional accessor
-> Conditional -> arm -> Trace), so any forward that ran a Python ``if`` on a tensor
(every Hugging Face ``generate`` loop does) left its whole trace for the cyclic
collector's next full pass: about 106,000 objects for a GPT-2 generate of four tokens.
The arm now holds its Trace weakly, like every other record view, and a read through
an arm whose Trace was dropped refuses ``trace_reference_collected``.
"""

from __future__ import annotations

import copy
import gc
import pickle
import weakref
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.trace import ConditionalArm
from torchlens.errors import RecordBindingError


class _Branchy(nn.Module):
    """A small model whose forward records one if/else conditional."""

    def __init__(self) -> None:
        """Build one linear layer."""

        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the sign of the hidden sum."""

        h = self.fc(x)
        if h.sum() > 0:
            return torch.relu(h) * 2
        return torch.sigmoid(h) - 1


def _small_trace() -> Any:
    """Trace the branching model on a fixed input."""

    torch.manual_seed(0)
    return tl.trace(_Branchy().eval(), torch.randn(2, 4))


def _gpt2_trace() -> Any:
    """Trace a tiny random-init GPT-2 ``generate`` of two new tokens (no download)."""

    import transformers

    torch.manual_seed(0)
    config = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=64, n_positions=32)
    model = transformers.GPT2LMHeadModel(config).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    return tl.trace(
        model.generate,
        ids,
        input_kwargs={"max_new_tokens": 2, "do_sample": False, "pad_token_id": 0},
    )


def _freed_without_collector(make: Callable[[], Any]) -> bool:
    """Trace, drop the only reference with the collector off, and report whether it died."""

    trace = make()
    assert len(trace.graph.conditionals) > 0, "the case under test must record a conditional"
    ref = weakref.ref(trace)
    gc.collect()
    gc.disable()
    try:
        del trace
        return ref() is None
    finally:
        gc.enable()


@pytest.mark.parametrize("make", [_small_trace, _gpt2_trace], ids=["small", "gpt2"])
def test_dropped_trace_with_conditionals_dies_without_gc(make: Callable[[], Any]) -> None:
    """``del trace`` frees a trace that recorded conditionals, no ``gc.collect()`` needed."""

    assert _freed_without_collector(make), "the trace survived its last reference (held by a cycle)"


def test_retained_conditional_does_not_keep_its_trace() -> None:
    """A held ``Conditional`` neither pins its Trace nor reads silently after the Trace dies."""

    trace = _small_trace()
    conditional = trace.graph.conditionals[0]
    arm = conditional.arms[0]
    assert arm._trace is trace  # noqa: SLF001
    assert arm.evaluation_ops or arm.execution_ops
    ref = weakref.ref(trace)
    gc.disable()
    try:
        del trace
        assert ref() is None, "a retained Conditional kept its Trace alive"
    finally:
        gc.enable()
    assert conditional.num_arms >= 1
    assert repr(arm).startswith("ConditionalArm(")
    with pytest.raises(RecordBindingError) as excinfo:
        arm.evaluation_ops  # noqa: B018
    assert excinfo.value.code == "trace_reference_collected"
    with pytest.raises(RecordBindingError):
        arm.execution_ops  # noqa: B018


def test_unbound_arm_reads_empty() -> None:
    """An arm never bound to a Trace reads its role ops as empty, as before."""

    arm = ConditionalArm(kind="then")
    assert arm._trace is None  # noqa: SLF001
    assert arm.evaluation_ops == []
    assert arm.execution_ops == []


def test_copies_keep_arms_bound() -> None:
    """Pickle and deep copies rebind arms to the copied Trace; a copied arm stays a view."""

    trace = _small_trace()
    clone = pickle.loads(pickle.dumps(trace))
    deep = copy.deepcopy(trace)
    for copied in (clone, deep):
        for conditional in copied.graph.conditionals:
            for arm in conditional.arms:
                assert arm._trace is copied  # noqa: SLF001
    arm = trace.graph.conditionals[0].arms[0]
    assert copy.copy(arm)._trace is trace  # noqa: SLF001
    assert copy.deepcopy(arm)._trace is trace  # noqa: SLF001
    assert copy.deepcopy(arm) == arm
