"""Tier markers: the smoke/heavy/slow/serial/rare partition, checked at collection.

pytest markers are additive, so a test carrying ``smoke`` together with
``heavy`` or ``slow`` still runs under ``-m smoke``. Such a combination is a
partition mistake; this plugin checks every item the session selects (after
``-m`` deselection, as the former lint test did) and fails the session before
the first test runs. It also fails any session that collects a smoke item
from a file outside the smoke file list (``scripts/tier_durations.py files
--tier smoke``), the files every smoke run collects. The marker names themselves are
declared in ``pyproject.toml`` (``[tool.pytest.ini_options] markers``), in
lockstep with ``tests/composition_expectations/marker_algebra.py``.

Registered by ``tests/conftest.py`` through ``pytest_plugins``.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from scripts.tier_durations import on_smoke_file_list

#: Markers that may never combine with ``smoke`` on one resolved item.
#: ``heavy``/``slow``: additive markers keep the item in ``-m smoke`` despite
#: the heavier tier. ``rare`` only runs on request, so a smoke+rare item would
#: sit in the commit gate with no duration budget. ``serial`` claims load
#: sensitivity, which contradicts running inside the parallel commit gate.
#: ``big_memory`` names a slow-tier real-model test the Weekly runner excludes
#: for memory; a smoke test is never memory-heavy enough to need that.
SMOKE_INCOMPATIBLE_MARKERS = ("heavy", "slow", "serial", "rare", "big_memory")

_SMOKE_ONLY_MARKEXPR = re.compile(r"^\s*smoke\s*(?:$|and\s)")

#: Node ids of smoke items collected from files outside the smoke file list.
_OFF_LIST_SMOKE_ITEMS = pytest.StashKey[list[str]]()


def smoke_only_session(config: pytest.Config) -> bool:
    """Return whether ``-m`` selects the smoke tier (``smoke`` or ``smoke and ...``)."""

    return bool(_SMOKE_ONLY_MARKEXPR.match(config.option.markexpr or ""))


def tier_combo_violations(
    marker_names: set[str], callspec_marker_names: set[str], nodeid: str
) -> list[str]:
    """Return tier-combination violations for one resolved item's markers.

    Policy:
    - ``smoke`` may not combine with any of ``SMOKE_INCOMPATIBLE_MARKERS``.
    - ``heavy`` + ``slow`` is a partition contradiction (a test cannot be both
      5-20s and >20s) unless ``slow`` arrived as a per-parametrize-cell
      refinement of a heavy family (``pytest.param(..., marks=slow)``), the
      sanctioned shape for "this one cell measures beyond heavy's ceiling".
    - ``big_memory`` without ``slow`` is a scheduling contradiction: the Weekly
      job only ever excludes ``big_memory`` from its ``slow and not rare``
      selection, so such a test would never be excluded from anywhere.

    Parameters
    ----------
    marker_names:
        All marker names on the resolved item.
    callspec_marker_names:
        Marker names contributed by the item's parametrize callspec.
    nodeid:
        Item node id used in violation strings.

    Returns
    -------
    list[str]
        Human-readable violation strings (empty when compliant).
    """

    violations = []
    if "smoke" in marker_names:
        for incompatible in SMOKE_INCOMPATIBLE_MARKERS:
            if incompatible in marker_names:
                violations.append(f"{nodeid} [smoke + {incompatible}]")
    if (
        "heavy" in marker_names
        and "slow" in marker_names
        and "slow" not in callspec_marker_names
        and "heavy" not in callspec_marker_names
    ):
        violations.append(f"{nodeid} [heavy + slow, not a per-cell refinement]")
    if "big_memory" in marker_names and "slow" not in marker_names:
        violations.append(f"{nodeid} [big_memory without slow]")
    return violations


def item_combo_violations(item: pytest.Item) -> list[str]:
    """Apply the tier-combination policy to one collected pytest item."""

    marker_names = {marker.name for marker in item.iter_markers()}
    callspec = getattr(item, "callspec", None)
    callspec_marker_names = {marker.name for marker in getattr(callspec, "marks", [])}
    return tier_combo_violations(marker_names, callspec_marker_names, item.nodeid)


def pytest_itemcollected(item: pytest.Item) -> None:
    """Resolve ``smoke_cells`` marks, then record the item's partition violations.

    ``@pytest.mark.smoke_cells("test_x[a]", "test_x[b]")`` on a function, class
    or module (``pytestmark``) puts exactly the named items (matched on
    ``item.name``) in the smoke tier, so a parametrized family keeps one or two
    representative cells while the rest run in the backstops. Applied at item
    collection, before any ``-m`` deselection.

    A smoke item whose file is off the smoke file list is recorded here, before
    deselection, so every session that collects the whole tree (the not-slow
    shards, the backstops) judges every smoke item, selected or not.
    """

    for mark in item.iter_markers("smoke_cells"):
        if item.name in mark.args:
            item.add_marker(pytest.mark.smoke)
            break
    if item.get_closest_marker("smoke") is not None and not _on_smoke_file_list(str(item.path)):
        item.config.stash.setdefault(_OFF_LIST_SMOKE_ITEMS, []).append(item.nodeid)


@functools.cache
def _on_smoke_file_list(path: str) -> bool:
    return on_smoke_file_list(Path(path))


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Fail the session on off-list smoke items or a contradictory tier combination.

    Off-list smoke items were recorded at item collection, so they are judged
    whether or not this session selects them. Tier combinations run after the
    mark plugin's deselection, so a session judges the items it will run
    (targeted runs included); items the session deselects are not judged, as
    the former lint test's ``session.items`` census did.
    """

    fail_on_off_list_smoke_items(config.stash.get(_OFF_LIST_SMOKE_ITEMS, []))
    fail_on_tier_violations(items)


def fail_on_off_list_smoke_items(nodeids: list[str]) -> None:
    """Raise ``pytest.UsageError`` naming smoke items the smoke file list would skip.

    Every smoke run collects only the smoke file list, so such an item would
    silently drop out of the commit gate and CI's smoke steps.
    """

    if nodeids:
        raise pytest.UsageError(
            "smoke tests live outside the smoke file list (`python scripts/tier_durations.py "
            "files --tier smoke`: test_*.py files under tests/ whose source spells "
            "`mark.smoke` or `mark.smoke_cells`), so every smoke run would skip them. Spell "
            "the mark through `pytest.mark.smoke` in the test file itself:\n  "
            + "\n  ".join(sorted(nodeids))
        )


def fail_on_tier_violations(items: Iterable[Any]) -> None:
    """Raise ``pytest.UsageError`` listing every contradictory combination among ``items``."""

    violations = [violation for item in items for violation in item_combo_violations(item)]
    if violations:
        raise pytest.UsageError(
            "tests carry contradictory tier-marker combinations (markers are additive; "
            "drop the smoke mark or split the cells):\n  " + "\n  ".join(violations)
        )


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    """Warn when a local smoke run differs from CI's enforcing smoke environment."""

    del exitstatus
    if os.environ.get("CI") or config.option.collectonly or not smoke_only_session(config):
        return
    gaps = ci_smoke_parity_gaps()
    if gaps:
        terminalreporter.write_line(
            "torchlens CI parity: this smoke run differs from CI's enforcing smoke row ("
            + "; ".join(gaps)
            + "), so its answer can differ from CI's; "
            "`python scripts/smoke_ci_parity.py` runs the gate in that row's environment",
            yellow=True,
        )


def ci_smoke_parity_gaps() -> list[str]:
    """Return how this interpreter differs from CI's enforcing smoke row.

    CI installs the ``tabular`` and ``viz`` extras on every row, and its
    enforcing row runs the environment the env-fingerprinted goldens were
    recorded under; off that environment, outside CI, those golden families
    fail closed by design (``tests/_oracle_env.py``).
    """

    from _oracle_env import env_fingerprint

    gaps = [
        f"{module} not installed (`{extra}` extra)"
        for module, extra in (("pandas", "tabular"), ("matplotlib", "viz"))
        if importlib.util.find_spec(module) is None
    ]
    marker = Path(__file__).parent / "godobject_oracle" / "goldens" / "ENV"
    canonical = marker.read_text().strip()
    if env_fingerprint() != canonical:
        gaps.append(f"environment {env_fingerprint()}, goldens recorded on {canonical}")
    return gaps
