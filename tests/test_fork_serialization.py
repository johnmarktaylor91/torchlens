"""Fork serialization round-trips (pickle, tl.save/tl.load, deepcopy).

``Trace.fork()`` builds COW shells over the parent's sealed stores; detached
fallback records (modules fork detached by design) must decode BOTH internal
M14 cell encodings (``PooledCell`` pooling and compacted singleton labels)
while copying, because ``DetachedOpStore`` has no pool/registry to decode them
later and streams cells verbatim into pickle state. A leaked ``PooledCell``
made every fork serialization fail: plain pickle refused typed at restore,
and ``tl.save`` SUCCEEDED while ``tl.load`` always refused (the safe
unpickler blocks the non-allowlisted internal type) -- a silent dead
artifact. These tests pin the round-trip for every spelling.
"""

import copy
import os
import pickle
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._trace_core.op_store import PooledCell
from torchlens._trace_core.record_rows import CORE_KEY


class _Block(nn.Module):
    """Linear + BatchNorm + ReLU block (buffers exercise kind tables)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.bn = nn.BatchNorm1d(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the block."""

        return torch.relu(self.bn(self.fc(x)))


class _Net(nn.Module):
    """Two blocks with one recurrent call plus a head (multi-pass layers)."""

    def __init__(self) -> None:
        super().__init__()
        self.b1 = _Block()
        self.b2 = _Block()
        self.head = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run b1 twice (recurrent), then b2 and the head."""

        x = self.b1(x)
        x = self.b1(x)
        x = self.b2(x)
        return self.head(x)


@pytest.fixture(scope="module")
def parent_trace() -> Iterator[tl.Trace]:
    """Return a captured trace of ``_Net`` with saved relu activations."""

    torch.manual_seed(0)
    model = _Net()
    trace = tl.trace(model, torch.randn(4, 8), save=tl.func("relu"))
    try:
        yield trace
    finally:
        trace.cleanup()


def _assert_fork_equivalent(restored: tl.Trace, fork: tl.Trace) -> None:
    """Assert the restored trace mirrors the fork's structure and payloads."""

    assert restored.trace_label == fork.trace_label
    assert len(restored.layer_list) == len(fork.layer_list)
    assert list(restored.layer_logs.keys()) == list(fork.layer_logs.keys())
    for source_op, restored_op in zip(fork.layer_list, restored.layer_list):
        assert restored_op.layer_label == source_op.layer_label
        assert restored_op.parents == source_op.parents
        assert restored_op.children == source_op.children
    saved = fork["relu_1_4"].ops[0].out
    restored_saved = restored["relu_1_4"].ops[0].out
    assert torch.equal(restored_saved, saved)


def test_fork_detached_records_carry_no_internal_cell_encodings(
    parent_trace: tl.Trace,
) -> None:
    """No fork record's backing store may hold a raw ``PooledCell``.

    The RED-capable core of the serialization bug: modules fork as detached
    duplicates, and the detached fill copied raw M14 pooled cells straight
    into stores that stream state verbatim.
    """

    fork = parent_trace.fork()
    records = [*fork.modules, *[call for m in fork.modules for call in m.calls.values()]]
    records += [*fork.params, *fork.buffers]
    leaked: list[str] = []
    for record in records:
        store = record.__dict__.get(CORE_KEY)
        if store is None or not hasattr(store, "_cells"):
            continue
        for name, value in store.items(0):
            if value.__class__ is PooledCell:
                leaked.append(f"{type(record).__name__}.{name}")
    assert not leaked, f"raw PooledCell leaked into detached fork records: {leaked}"


def test_fork_pickle_round_trip(parent_trace: tl.Trace) -> None:
    """A COW fork must survive plain ``pickle`` dumps/loads."""

    fork = parent_trace.fork()
    restored = pickle.loads(pickle.dumps(fork))
    _assert_fork_equivalent(restored, fork)


def test_fork_save_load_round_trip(parent_trace: tl.Trace, tmp_path) -> None:
    """``tl.save(fork)`` must produce an artifact ``tl.load`` accepts.

    Pre-fix this was the silent-dead-artifact case: save succeeded and load
    ALWAYS refused (blocked internal ``PooledCell`` type).
    """

    fork = parent_trace.fork()
    path = os.path.join(tmp_path, "fork.tlspec")
    tl.save(fork, path)
    restored = tl.load(path)
    _assert_fork_equivalent(restored, fork)


def test_fork_deepcopy(parent_trace: tl.Trace) -> None:
    """``copy.deepcopy`` of a fork must succeed and preserve structure."""

    fork = parent_trace.fork()
    duplicate = copy.deepcopy(fork)
    assert len(duplicate.layer_list) == len(fork.layer_list)


def test_fork_of_fork_pickle_round_trip(parent_trace: tl.Trace) -> None:
    """A fork chain (fork of a fork) must also pickle round-trip."""

    fork = parent_trace.fork()
    grandfork = fork.fork()
    restored = pickle.loads(pickle.dumps(grandfork))
    assert len(restored.layer_list) == len(grandfork.layer_list)


def test_pickle_and_deepcopy_after_receptive_field_access() -> None:
    """RF session caches must not leak mappingproxies into the pickle path.

    Hunt-6 incidental HIGH: after ANY receptive-field access the lazy
    influence-geometry caches (``_receptive_field_solution`` /
    ``_rf_directional_solutions``) held
    mappingproxy-bearing solution objects that ``__getstate__`` carried
    verbatim, so ``pickle.dumps(trace)`` and ``copy.deepcopy(trace)`` raised
    ``TypeError: cannot pickle 'mappingproxy' object`` while the SAME calls
    succeeded on a fresh trace (``tl.save`` was unaffected -- its scrub path
    drops the fields). Influence geometry is an advertised lazy surface, so
    inspect-RF-then-snapshot is a realistic user path.
    """

    model = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.ReLU())
    trace = tl.trace(model, torch.randn(1, 1, 8, 8))

    box = trace["relu_1_2"].receptive_field.at((2, 2))
    assert box is not None
    assert trace.__dict__.get("_receptive_field_solution") is not None

    payload = pickle.dumps(trace)  # crashed before the __getstate__ scrub
    restored = pickle.loads(payload)
    cloned = copy.deepcopy(trace)

    # The caches rebuild on demand: RF stays usable on both copies.
    for twin in (restored, cloned):
        assert twin.__dict__.get("_receptive_field_solution") is None
        rebuilt = twin["relu_1_2"].receptive_field.at((2, 2))
        assert rebuilt is not None
