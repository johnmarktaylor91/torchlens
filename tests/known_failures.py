"""Ledger of tracked Weekly slow-tier real-model validation failures.

JMT's ruling (round 2, lane-L8-ci-fix / lane-L17-integrate, 2026-10-02/03): the
Weekly slow tier must not stay red as a durable state. A failure here is
either fixed (its entry is removed in the same change) or tracked here as a
strict ``xfail`` -- never silently skipped, and never loosened at the
``validation/`` layer itself (AGENTS.md "Validation Integrity (LOCKED
PRINCIPLE)" still applies in full: nothing here touches a tolerance, a check,
or an invariant).

This is the ONE place these tests are marked ``xfail``. ``tests/conftest.py``
reads :data:`KNOWN_FAILURES` during collection and applies
``pytest.mark.xfail(strict=True, reason=...)`` to each listed node id
directly -- no test file decorates ``xfail`` itself
(``tests/test_known_failures_ledger.py`` enforces both halves: every entry
matches a real collected node id, and no ledger-named file declares an
``xfail`` of its own). ``strict=True`` means an unexpected PASS fails the
run, so fixing one of these forces removing its row here in the same change
-- the ledger can never silently drift from reality in the fixed direction
either. ``pyproject.toml`` already sets ``xfail_strict = true`` suite-wide;
the per-entry ``strict=True`` below is kept explicit so this file's own
intent does not depend on that ini default.

Each entry's ``reason`` is the failure class actually observed on a real run
of the Weekly environment (torch 2.7.1+cpu / torchvision 0.22.1+cpu), not a
guess from memory or from an older round's notes -- see
tests/test_known_failures_ledger.py for how that is checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


@dataclass(frozen=True)
class KnownFailure:
    """One tracked, strict-xfail real-model test.

    Attributes
    ----------
    nodeid:
        Exact pytest node id, e.g.
        ``"tests/test_real_world_models.py::test_timm_beit_base_patch16_224"``.
    reason:
        The failure class observed on a real Weekly-environment run.
    tracking:
        Where the round-2 follow-up for this failure is recorded.
    """

    nodeid: str
    reason: str
    tracking: str


#: Populated from a real run of the Weekly slow tier (one test per process,
#: torch 2.7.1+cpu / torchvision 0.22.1+cpu); see tests/AGENTS.md "Testing
#: Tiers" for how to reproduce that environment.
KNOWN_FAILURES: tuple[KnownFailure, ...] = ()


def duplicate_nodeids() -> list[str]:
    """Return node ids that appear more than once in :data:`KNOWN_FAILURES`."""

    seen: set[str] = set()
    duplicates: list[str] = []
    for entry in KNOWN_FAILURES:
        if entry.nodeid in seen:
            duplicates.append(entry.nodeid)
        seen.add(entry.nodeid)
    return duplicates


def stale_entries(collected_nodeids: set[str]) -> list[KnownFailure]:
    """Return ledger entries whose node id was not actually collected.

    Pure helper (red-capable without a real pytest session): a stale entry
    means the test was renamed, removed, or never existed under this id --
    the ledger must never carry a row pytest cannot resolve.

    Parameters
    ----------
    collected_nodeids:
        Node ids pytest actually collected for the files this ledger names.

    Returns
    -------
    list[KnownFailure]
        Entries with no matching collected node id.
    """

    return [entry for entry in KNOWN_FAILURES if entry.nodeid not in collected_nodeids]


def ledger_by_nodeid() -> dict[str, KnownFailure]:
    """Return the ledger indexed by node id (duplicates keep the last entry)."""

    return {entry.nodeid: entry for entry in KNOWN_FAILURES}


def apply_xfail_marks(items: list[pytest.Item]) -> None:
    """Mark every collected item named in :data:`KNOWN_FAILURES` ``xfail``.

    Called from ``tests/conftest.py::pytest_collection_modifyitems`` so this
    stays the single place that turns a ledger row into a live pytest marker.

    Parameters
    ----------
    items:
        Collected pytest items for this session.
    """

    import pytest as _pytest

    ledger = ledger_by_nodeid()
    if not ledger:
        return
    for item in items:
        entry = ledger.get(item.nodeid)
        if entry is not None:
            item.add_marker(_pytest.mark.xfail(reason=entry.reason, strict=True))
