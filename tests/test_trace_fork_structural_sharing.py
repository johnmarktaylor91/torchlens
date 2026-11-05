"""Fork copy-on-write / structural-sharing invariants.

``Trace._fork_trace`` builds the fork as ONE structural copy of the parent graph
under a single shared ``copy.deepcopy`` memo instead of one independent copy per
field. These tests pin the two properties that make that safe:

1. ISOLATION -- no fork-owned object may alias a parent-owned object, so no
   mutation on the fork (or during ``Trace.run``) can reach the parent.
2. SELF-CONSISTENCY -- back-references inside the fork resolve to the FORK's own
   objects, not to orphan clones of the parent's.
"""

from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl


class _CondNet(nn.Module):
    """Small net with a buffer, a module tree, and a taken conditional branch."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
        self.head = nn.Linear(8, 4)
        self.register_buffer("offset", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder, take a data-independent branch, then the head."""

        hidden = self.encoder(x)
        if hidden.sum() > -1e30:
            hidden = hidden * 2
        else:  # pragma: no cover - the untaken arm exists to create an if-chain
            hidden = hidden * 3
        return self.head(hidden) + self.offset


@pytest.fixture(scope="module")
def cond_trace() -> Iterator[tl.Trace]:
    """Return a captured trace of ``_CondNet``."""

    torch.manual_seed(0)
    model = _CondNet()
    trace = tl.trace(model, torch.randn(2, 8))
    try:
        yield trace
    finally:
        trace.cleanup()


def test_fork_memo_rollback_discards_partial_entries() -> None:
    """A field copy that raises must not leave half-built objects in the memo."""

    from torchlens._data_classes._trace_fork import _ForkMemo, _memoized_deep_copy

    class _Boom:
        """Object whose deep copy always fails after the memo is seeded."""

        def __deepcopy__(self, memo: dict) -> "_Boom":
            """Seed a bogus entry, then fail."""

            memo[id(self)] = self
            memo[12345] = "half-built"
            raise RuntimeError("boom")

    memo = _ForkMemo()
    memo[999] = "kept"
    boom = _Boom()
    assert _memoized_deep_copy(boom, memo, on_failure=lambda value: value, fallback=None) is boom
    assert memo[999] == "kept"
    assert 12345 not in memo
    assert id(boom) not in memo
