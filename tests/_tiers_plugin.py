"""Tier markers: the smoke/heavy/slow/serial/rare partition, checked at collection.

pytest markers are additive, so a test carrying ``smoke`` together with
``heavy`` or ``slow`` still runs under ``-m smoke``. Such a combination is a
partition mistake; this plugin checks every item the session selects (after
``-m`` deselection, as the former lint test did) and fails the session before
the first test runs. It also fails any session that collects a smoke item
from a file outside the smoke file list (``scripts/tier_durations.py files
--tier smoke``), and a smoke session (``-m smoke``) skips every test module
off that list before import, so smoke runs collect only the listed files.
The marker names themselves are declared in ``pyproject.toml`` (``[tool.pytest.ini_options] markers``), in
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

from scripts.tier_durations import TESTS_DIR, on_smoke_file_list

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


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Skip, before import, a test module off the smoke file list in a smoke session.

    Under ``-m smoke`` (or ``-m "smoke and ..."``) a ``test_*.py`` module under
    ``tests/`` that is not on the smoke file list cannot hold a smoke item, so
    collecting ``tests/`` imports only the listed modules. Passing the listed
    files as arguments instead selects the same tests but costs more: pytest
    re-walks the parent directory for every file argument (measured 2026-11-02,
    459 file arguments: about 130 s to collect against about 65 s for the whole
    tree). A module named on the command line is always collected, and
    ``TORCHLENS_SMOKE_PREFILTER=0`` turns the skip off (a whole-tree count).
    """

    if (
        os.environ.get("TORCHLENS_SMOKE_PREFILTER", "1") == "0"
        or collection_path.suffix != ".py"
        or not collection_path.name.startswith("test_")
        or not smoke_only_session(config)
        or not collection_path.resolve().is_relative_to(TESTS_DIR.resolve())
    ):
        return None
    return None if _on_smoke_file_list(str(collection_path)) else True


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--tier-shard INDEX/COUNT`` (the not-slow tier's file shards)."""

    parser.addoption(
        "--tier-shard",
        metavar="INDEX/COUNT",
        default=None,
        help=(
            "run only the test files of shard INDEX (0-based) of COUNT, balanced by the "
            "tracked per-file durations (python scripts/tier_durations.py shards)"
        ),
    )


def parse_tier_shard(spec: str) -> tuple[int, int]:
    """Return ``(index, count)`` from ``INDEX/COUNT``; raise ``pytest.UsageError`` otherwise."""

    match = re.fullmatch(r"(\d+)/(\d+)", spec.strip())
    if match is None or not 0 <= int(match.group(1)) < int(match.group(2)):
        raise pytest.UsageError(f"--tier-shard wants INDEX/COUNT with 0 <= INDEX < COUNT: {spec!r}")
    return int(match.group(1)), int(match.group(2))


def tier_shard_items(
    items: list[pytest.Item], index: int, count: int, root: Path
) -> tuple[list[pytest.Item], list[pytest.Item]]:
    """Split ``items`` into this shard's and the rest, by whole test file.

    The files on disk are balanced exactly as ``tier_durations.py shards`` prints
    them, so the shards are disjoint and together cover every file; an item
    from a file outside that list (none today) runs in shard 0, never nowhere.
    """

    from scripts.tier_durations import balanced_shards, read_rows, test_files_on_disk

    on_disk = test_files_on_disk()
    mine = set(balanced_shards(read_rows(), count, on_disk)[index])
    known = set(on_disk)
    kept: list[pytest.Item] = []
    dropped: list[pytest.Item] = []
    for item in items:
        file = Path(str(item.path)).resolve().relative_to(root).as_posix()
        in_shard = file in mine or (file not in known and index == 0)
        (kept if in_shard else dropped).append(item)
    return kept, dropped


def apply_tier_shard(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep only this ``--tier-shard``'s test files when the option is given."""

    spec = config.getoption("--tier-shard", default=None)
    if spec is None:
        return
    index, count = parse_tier_shard(spec)
    kept, dropped = tier_shard_items(items, index, count, Path(str(config.rootpath)).resolve())
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = kept


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply ``--tier-shard``, then fail on off-list smoke items or a contradictory tier combination.

    Off-list smoke items were recorded at item collection, so they are judged
    whether or not this session selects them. Tier combinations run after the
    mark plugin's deselection, so a session judges the items it will run
    (targeted runs included); items the session deselects are not judged, as
    the former lint test's ``session.items`` census did. The shard filter keeps
    the invocation rooted at ``tests/``, so the conftest gates that arm only on
    an unfiltered ``tests/`` run stay armed in every shard.
    """

    apply_tier_shard(config, items)
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
