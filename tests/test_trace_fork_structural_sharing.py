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


def _owned_object_ids(trace: tl.Trace) -> set[int]:
    """Return ids of the child objects a Trace owns outright."""

    owned = {id(op) for op in trace.layer_list}
    owned |= {id(layer) for layer in trace.layer_logs.values()}
    owned |= {id(module) for module in trace.modules}
    owned |= {id(call) for module in trace.modules for call in module.calls.values()}
    owned |= {id(conditional) for conditional in trace.conditionals}
    owned |= {id(arm) for conditional in trace.conditionals for arm in conditional.arms}
    return owned


def test_fork_shares_no_owned_object_with_parent(cond_trace: tl.Trace) -> None:
    """No fork-owned child object may be the parent's own object."""

    fork = cond_trace._fork_trace(name="isolation")
    assert not (_owned_object_ids(fork) & _owned_object_ids(cond_trace))


def test_fork_back_references_point_at_the_fork(cond_trace: tl.Trace) -> None:
    """Owner back-references inside the fork resolve to the fork itself."""

    fork = cond_trace._fork_trace(name="selfconsistent")
    assert all(op.source_trace is fork for op in fork.layer_list)
    assert all(layer.source_trace is fork for layer in fork.layer_logs.values())
    assert all(module._source_trace is fork for module in fork.modules)
    assert fork.conditionals, "fixture must produce at least one conditional"
    for conditional in fork.conditionals:
        for arm in conditional.arms:
            assert arm._trace is fork


def test_fork_lookup_containers_agree_on_identity(cond_trace: tl.Trace) -> None:
    """Every lookup container hands back the same fork Op object."""

    fork = cond_trace._fork_trace(name="identity")
    by_position = fork.layer_list
    assert set(map(id, fork.layer_dict_main_keys.values())) <= set(map(id, by_position))
    assert set(map(id, fork.layer_dict_all_keys.values())) <= set(map(id, by_position))
    for layer in fork.layer_logs.values():
        for op in layer.ops.values():
            assert any(op is candidate for candidate in by_position)


def test_fork_mutation_does_not_reach_parent(cond_trace: tl.Trace) -> None:
    """Mutating fork state leaves every parent-visible value untouched."""

    before = [dict(op.annotations) for op in cond_trace.layer_list]
    before_arms = [
        [arm.fired for arm in conditional.arms] for conditional in cond_trace.conditionals
    ]
    fork = cond_trace._fork_trace(name="mutation")
    for op in fork.layer_list:
        op.annotations["__probe__"] = "fork-only"
    for conditional in fork.conditionals:
        for arm in conditional.arms:
            arm.fired = not arm.fired
    assert [dict(op.annotations) for op in cond_trace.layer_list] == before
    assert [
        [arm.fired for arm in conditional.arms] for conditional in cond_trace.conditionals
    ] == before_arms


def test_fork_memo_rollback_discards_partial_entries() -> None:
    """A field copy that raises must not leave half-built objects in the memo."""

    from torchlens.data_classes._trace_intervention import _ForkMemo, _memoized_deep_copy

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
