"""The host allocation budget never exceeds the process's address-space limit.

The runnable preflight refuses a projected allocation that exceeds the host
budget. That budget is available memory plus free swap, and on a host with
about 2 TB available (a 224-CPU cluster node) a 1.6 TB allocation bomb fitted
inside it even when the process ran under an address-space limit
(``RLIMIT_AS``) far below the bomb, so the preflight admitted an allocation
that could never succeed and the refusal moved to the allocator. No single
allocation can exceed the address space the process may map, so the soft
limit, when one is set, bounds the budget on every host.
"""

from __future__ import annotations

import os
import resource

import pytest

from torchlens._runnable import state

linux_procfs = pytest.mark.skipif(
    not os.path.exists("/proc/self/status"),
    reason="sizes the limit from /proc/self/status VmSize (Linux only)",
)


def _vm_size_bytes() -> int:
    with open("/proc/self/status", encoding="ascii") as handle:
        return next(int(line.split()[1]) * 1024 for line in handle if line.startswith("VmSize"))


@linux_procfs
def test_host_budget_is_bounded_by_the_address_space_limit() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = _vm_size_bytes() + (1 << 30)
    if hard != resource.RLIM_INFINITY and hard < cap:
        cap = hard
    resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
    try:
        budget = state._host_memory_budget_bytes()
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    assert budget is not None
    assert budget <= cap


@pytest.mark.parametrize(
    ("available", "limit", "expected"),
    [
        (123 << 30, None, 123 << 30),
        (2 << 40, 8 << 30, 8 << 30),
        (4 << 30, 8 << 30, 4 << 30),
        (None, 8 << 30, 8 << 30),
        (None, None, None),
    ],
    ids=["no-limit", "limit-below-available", "available-below-limit", "unprobed", "neither"],
)
def test_host_budget_is_the_smaller_known_bound(
    monkeypatch: pytest.MonkeyPatch, available: int | None, limit: int | None, expected: int | None
) -> None:
    from torchlens._runnable import host_budget

    monkeypatch.setattr(host_budget, "_available_memory_bytes", lambda: available)
    monkeypatch.setattr(host_budget, "_address_space_limit_bytes", lambda: limit)
    assert state._host_memory_budget_bytes() == expected
