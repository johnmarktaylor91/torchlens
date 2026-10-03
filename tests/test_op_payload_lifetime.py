"""Retained-Op payload lifetime (last-owner tensor-payload eviction).

Pre-fix, ONE retained ``Op`` facade -- a two-word ``(_core, _row)`` view over
the shared sealed row store -- kept EVERY captured activation alive after its
Trace died and was garbage-collected, because the store holds all payload
cells and the facade holds the store strongly. ``Op`` deliberately refuses
weak references (aliases-v1 row 1b) and detached facades need a strong
``_core``, so the fix is ownership-based instead: every owning ``TraceCore``
(the capture's own core plus one per fork core viewing the base) registers a
``weakref.finalize`` on the sealed store, and when the LAST owner dies the
store evicts top-level tensor cells (payloads read ``None``, the existing
"no payload retained" spelling) while all metadata stays readable.

Measurement-bug guard: every check reads through ``weakref.ref`` snapshots
taken before deletion, and no local variable may keep a strong path to the
trace, its core, or a payload tensor at assert time (a prior probe
over-counted retention 20x through exactly such a stray local).
"""

import gc
import weakref
from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import PayloadUnavailableError


class _Net(nn.Module):
    """Two-layer relu net (several saved payloads)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both layers with relus."""

        return torch.relu(self.fc2(torch.relu(self.fc1(x))))


def _payload_refs(trace: tl.Trace) -> dict[str, Callable[[], torch.Tensor | None]]:
    """Return weakrefs to every saved payload tensor, keeping no strong refs."""

    refs: dict[str, Callable[[], torch.Tensor | None]] = {}
    for op in trace.ops.values():
        try:
            payload = op.out
        except PayloadUnavailableError:
            continue
        if isinstance(payload, torch.Tensor):
            refs[op.layer_label] = weakref.ref(payload)
        del payload
    del op
    return refs


def _capture() -> tl.Trace:
    """Capture a trace with saved relu activations."""

    torch.manual_seed(0)
    return tl.trace(_Net(), torch.randn(4, 16), save=tl.func("relu"))


def test_retained_op_does_not_pin_all_payloads() -> None:
    """The RED-capable core case: one live Op must not pin every payload."""

    trace = _capture()
    refs = _payload_refs(trace)
    assert len(refs) >= 2
    op = trace["relu_1_2"].ops[0]
    label = op.layer_label
    del trace
    gc.collect()
    gc.collect()
    alive = sorted(label for label, ref in refs.items() if ref() is not None)
    assert not alive, f"payloads pinned by one retained Op after trace GC: {alive}"
    # Metadata on the retained facade stays readable; the evicted payload
    # reads as the existing "no payload retained" spelling.
    assert op.layer_label == label
    assert op.out is None


@pytest.mark.heavy
def test_live_fork_keeps_shared_base_payloads() -> None:
    """A live fork co-owns the sealed base: parent death must not evict."""

    trace = _capture()
    fork = trace.fork()
    refs = _payload_refs(trace)
    del trace
    gc.collect()
    gc.collect()
    dead = sorted(label for label, ref in refs.items() if ref() is None)
    assert not dead, f"parent GC evicted payloads a live fork still reads: {dead}"
    assert isinstance(fork["relu_1_2"].ops[0].out, torch.Tensor)
    op = fork["relu_1_2"].ops[0]
    del fork
    gc.collect()
    gc.collect()
    alive = sorted(label for label, ref in refs.items() if ref() is not None)
    assert not alive, f"payloads pinned after parent AND fork GC: {alive}"
    assert op.out is None


@pytest.mark.heavy
def test_fork_chain_owners_all_counted() -> None:
    """Every core in a fork chain is an owner; the last death evicts."""

    trace = _capture()
    fork = trace.fork()
    grandfork = fork.fork()
    refs = _payload_refs(trace)
    op = grandfork["relu_1_2"].ops[0]
    del trace, fork
    gc.collect()
    gc.collect()
    assert all(ref() is not None for ref in refs.values())
    assert isinstance(op.out, torch.Tensor)
    del grandfork, op
    gc.collect()
    gc.collect()
    assert all(ref() is None for ref in refs.values())


def test_loaded_trace_fork_death_does_not_evict_parent_payloads(tmp_path) -> None:
    """R37-1: a loaded trace's core must register as a payload owner.

    A loaded (rehydrated) trace's core seals its op store but historically never
    called ``adopt_payload_owner``, so its owner count stayed 0. Forking it (the
    fork DOES adopt: 0 -> 1) and then dropping the fork drove the count back to 0
    and evicted the SEALED base -- silently NULLing the LIVE loaded parent's
    saved activations. The fork's death must leave the loaded parent unharmed.
    """

    art = tmp_path / "payloads.tlspec"
    tl.save(_capture(), art)
    loaded = tl.load(art)
    refs = _payload_refs(loaded)
    assert len(refs) >= 2, "loaded trace should carry rehydrated tensor payloads"
    fork = loaded.fork()
    del fork
    gc.collect()
    gc.collect()
    dead = sorted(label for label, ref in refs.items() if ref() is None)
    assert not dead, f"fork GC evicted a LIVE loaded parent's payloads: {dead}"
    assert isinstance(loaded["relu_1_2"].ops[0].out, torch.Tensor)


@pytest.mark.heavy
def test_no_handles_no_leak() -> None:
    """Guard: with no retained handles at all, everything collects."""

    trace = _capture()
    refs = _payload_refs(trace)
    trace_ref = weakref.ref(trace)
    del trace
    gc.collect()
    gc.collect()
    assert trace_ref() is None
    assert all(ref() is None for ref in refs.values())
