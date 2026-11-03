"""Duplicate/empty mutable-container cell pooling (the M14 memory slice).

Covers the ``PooledCell`` contract end to end on a real capture: pooled
cells exist on the finished trace (the memory win is real), reads hydrate
fresh exact-type containers with stable identity and per-row isolation,
forks and pickles never observe the internal encoding, aliased containers
refuse to pool, and the empty/duplicate thresholds hold.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.op import (
    _container_pool_key,
)


class _Stack(nn.Module):
    def __init__(self, blocks: int = 6) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(blocks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


@pytest.fixture(scope="module")
def stack_trace() -> Iterator[tl.Trace]:
    """Yield one shared stack Trace and release it after the module."""

    torch.manual_seed(0)
    trace = tl.trace(_Stack(), torch.zeros(1, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def test_container_pool_key_injectivity() -> None:
    """Keys are class-tagged: bool/int/float never collapse, dict order and
    container classes distinguish, defaultdict factories key separately."""

    def key(value):
        return _container_pool_key(value, 0, [])

    assert key({"a": True}) != key({"a": 1})
    assert key([1.0]) != key([1])
    assert key({"a": 1, "b": 2}) != key({"b": 2, "a": 1})
    assert key([1, 2]) != key({1: 2})
    dd_list = defaultdict(list)
    dd_set = defaultdict(set)
    assert key(dd_list) != key(dd_set) != key({})
    assert key({"nested": [1, {2}]}) is not None
    cyclic: list = []
    cyclic.append(cyclic)
    assert key(cyclic) is None


def test_singleton_fork_isolation_and_detach(stack_trace: tl.Trace) -> None:
    """Fork decodes into its own overlay; detached records leave with a
    real list, never the encoding."""

    fork = stack_trace.fork()
    fork_list = fork.params[4].all_addresses
    assert type(fork_list) is list
    fork_list.append("fork_only")
    assert "fork_only" not in stack_trace.params[4].all_addresses

    from torchlens._trace_core.record_rows import detach_record

    param = stack_trace.params[5]
    expected = list(param.all_addresses)
    detach_record(param)
    detached_store = param.__dict__["_tl_core"]
    fid = detached_store.layout.fid_by_name["all_addresses"]
    assert type(detached_store.cell_get(0, fid)) is list
    assert param.all_addresses == expected
