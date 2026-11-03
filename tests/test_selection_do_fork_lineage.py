"""``do()`` accepts a selection resolved on the trace a fork descends from.

The README's select-then-intervene form resolves a field selection on the clean
trace and applies it to ``trace.fork()``. The fork shares the clean run's
captured index space, so ``do()`` rebinds such a selection through
``align_to`` instead of refusing ``selection_trace_mismatch``; a selection from
an unrelated capture still refuses.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import SelectionError


class _TwoConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3)
        self.c2 = nn.Conv2d(2, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c2(torch.relu(self.c1(x))))


@pytest.fixture(scope="module")
def clean():
    torch.manual_seed(0)
    trace = tl.trace(
        _TwoConv(),
        torch.randn(1, 1, 12, 12),
        capture=tl.options.CaptureOptions(replayable=True),
    )
    try:
        yield trace
    finally:
        trace.cleanup()


def _output(trace) -> torch.Tensor:
    return trace[trace.output_layers[0]].out


def test_fork_do_accepts_a_selection_resolved_on_the_clean_trace(clean) -> None:
    """``trace.fork().do(sel)`` with ``sel`` resolved on ``trace`` edits the fork."""

    selection = clean[tl.func("relu")]
    assert selection._trace is clean

    fork = clean.fork()
    returned = fork.do(selection, tl.zero_ablate())

    assert returned is fork
    assert torch.count_nonzero(_output(fork)) == 0
    assert torch.count_nonzero(_output(clean)) > 0  # the clean trace is untouched

    # Same result as resolving the query on the fork itself.
    twin = clean.fork()
    twin.do(twin[tl.func("relu")], tl.zero_ablate())
    assert torch.equal(_output(twin), _output(fork))


def test_grandchild_fork_accepts_the_root_selection(clean) -> None:
    """The rebind follows the whole fork lineage, not only the direct parent."""

    selection = clean[tl.func("relu")]
    grandchild = clean.fork().fork()
    grandchild.do(selection, tl.scale(0.0))
    assert torch.count_nonzero(_output(grandchild)) == 0


def test_rebound_selection_discloses_its_origin(clean) -> None:
    """The rebound selection is bound to the fork and says where it came from."""

    from torchlens._selection.do_plan import _resolve_do_target

    fork = clean.fork()
    resolved = _resolve_do_target(fork, clean[tl.func("relu")])
    assert resolved._trace is fork
    assert all("aligned from" in entry.provenance.source for entry in resolved)


def test_unrelated_trace_selection_still_refuses(clean) -> None:
    """A selection resolved on another capture is not a fork ancestor: typed refusal."""

    torch.manual_seed(1)
    other = tl.trace(
        _TwoConv(),
        torch.randn(1, 1, 12, 12),
        capture=tl.options.CaptureOptions(replayable=True),
    )
    try:
        foreign = other[tl.func("relu")]
        with pytest.raises(SelectionError) as info:
            clean.fork().do(foreign, tl.zero_ablate())
        assert info.value.fields["code"] == "selection_trace_mismatch"
    finally:
        other.cleanup()
