"""The frame-reachable inventory walk must not skip a new object at a freed address.

The deep inventory walk remembers what it visited for the whole RNG-monitor window.
When it remembered only addresses, an object it visited could be freed inside the
window and a new object could take the address; the walk then skipped the new object,
never digested a generator behind it, and a draw from that generator left the window
VERIFIED. These tests build that reuse and check the verdict, and check that what the
walk remembers never outlives the window.

NumPy>=2 draw methods emit no profile event, so the draws below are witnessed only by
the inventory digest. Every helper is a real ``.py`` frame, so it seeds the walk the
way user code does.
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest

from torchlens._utils.rng import host_nondeterminism_monitor, inventory_types as rng_inventory_types

pytestmark = pytest.mark.skipif(
    not rng_inventory_types._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST,
    reason="NumPy build emits c_call for RNG draw methods (digest belt inactive)",
)

_REUSE_ATTEMPTS = 2000
_CHANNEL = "frame_reachable_generator"


class _Plain:
    """A plain holder: weakly referenceable, with an instance ``__dict__``."""


def _look(holder: Any) -> None:
    """Enter a frame whose only local is ``holder`` (the walk visits it)."""

    del holder


def _draw(holder: Any) -> None:
    """Draw from a generator two edges below a local (outside the one-edge digest)."""

    _ = holder.inner.gen.random()


def _rebuild_at(address: int, build: Callable[[], Any]) -> tuple[Any, bool]:
    """Build objects until one sits at ``address``; return the last and whether it did.

    Misses stay alive until the search ends so the allocator moves on. Bounded.
    """

    misses = []
    for _ in range(_REUSE_ATTEMPTS):
        candidate = build()
        if id(candidate) == address:
            return candidate, True
        misses.append(candidate)
    return misses.pop(), False


def _hold_generator(holder: Any, seed: int) -> np.random.Generator:
    """Put a fresh generator two edges below ``holder`` and return it."""

    inner = _Plain()
    inner.gen = np.random.default_rng(seed)  # type: ignore[attr-defined]
    if isinstance(holder, dict):
        holder["inner"] = inner
    else:
        holder.inner = inner
    return inner.gen  # type: ignore[attr-defined]


_KINDS: dict[str, tuple[Callable[[], Any], Callable[[], Any]]] = {
    # (first visited object, a fresh object of the same shape)
    "plain_object": (_Plain, _Plain),
    "dict": (lambda: {"leaf": 1}, dict),
}


@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_walk_digests_a_new_holder_at_a_freed_address(kind: str) -> None:
    """A holder built at a visited, freed address is walked and its draw is marked."""

    make_first, make_fresh = _KINDS[kind]
    monitor = host_nondeterminism_monitor(None)
    code = _look.__code__
    first = make_first()
    monitor._deep_inventory_frame_reachable((first,), code)
    address = id(first)
    del first
    holder, reused = _rebuild_at(address, make_fresh)
    if kind == "plain_object":
        # The walk keeps a weakly referenceable object only weakly, so the
        # address is free to reuse; the identity check must catch it.
        assert reused, "the allocator never repeated the freed address"
    gen = _hold_generator(holder, seed=11)
    monitor._deep_inventory_frame_reachable((holder,), code)
    assert any(item is gen for item, _ in monitor._deep_generator_states)
    gen.random()
    monitor._compare_generator_inventories()
    assert _CHANNEL in monitor.result.channels


def test_window_marks_a_draw_behind_a_holder_at_a_reused_address() -> None:
    """Inside one real window: visit, free, rebuild at the address, draw, check verdict."""

    with host_nondeterminism_monitor(None) as result:
        first = _Plain()
        _look(first)
        address = id(first)
        del first
        holder, reused = _rebuild_at(address, _Plain)
        _hold_generator(holder, seed=12)
        _draw(holder)
    assert reused, "the allocator never repeated the freed address"
    assert _CHANNEL in result.channels


def test_window_without_a_draw_stays_unmarked() -> None:
    """Control: the same reuse with no draw marks nothing."""

    with host_nondeterminism_monitor(None) as result:
        first = _Plain()
        _look(first)
        address = id(first)
        del first
        holder, _ = _rebuild_at(address, _Plain)
        _hold_generator(holder, seed=13)
        _look(holder)
    assert _CHANNEL not in result.channels


@contextmanager
def _collector_off() -> Iterator[None]:
    """Disable the cyclic collector so only reference counting can free objects."""

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


class _WindowError(Exception):
    """Raised inside the window to end it by an exception."""


def _visit_and_drop(raise_inside: bool, refs: list[weakref.ref[Any]]) -> None:
    """Visit a plain object and one inside a list within a window; drop both."""

    with host_nondeterminism_monitor(None):
        direct = _Plain()
        boxed = [_Plain()]
        _look(direct)
        _look(boxed)
        refs.extend((weakref.ref(direct), weakref.ref(boxed[0])))
        del direct, boxed
        if raise_inside:
            raise _WindowError


@pytest.mark.parametrize("end", ["normal", "exception"])
def test_visited_objects_are_freed_by_reference_counting_after_the_window(end: str) -> None:
    """Nothing the walk visited outlives the window, without the collector's help."""

    refs: list[weakref.ref[Any]] = []
    with _collector_off():
        if end == "normal":
            _visit_and_drop(False, refs)
        else:
            with pytest.raises(_WindowError):
                _visit_and_drop(True, refs)
        assert len(refs) == 2
        assert [ref() for ref in refs] == [None, None]


def test_budget_exhaustion_releases_the_memo_and_keeps_its_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At the node budget the walk flags INCOMPLETE as before and lets go of its memo."""

    monkeypatch.setattr(rng_inventory_types, "_DEEP_INVENTORY_NODE_CAP", 3)
    monitor = host_nondeterminism_monitor(None)
    holder = [[_Plain()] for _ in range(4)]
    with _collector_off():
        # The walk pops the last inner list first and holds it strongly.
        visited = weakref.ref(holder[-1][0])
        monitor._deep_inventory_frame_reachable((holder,), _look.__code__)
        del holder
        assert visited() is None
    assert monitor._deep_inventory_exhausted
    assert "deep_inventory_budget_exhausted" in monitor.result.uncertain_detail
    assert len(monitor._deep_walk_seen_ids) == 0
