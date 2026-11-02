"""Tier markers: the smoke/heavy/slow/serial/rare partition, checked at collection.

pytest markers are additive, so a test carrying ``smoke`` together with
``heavy`` or ``slow`` still runs under ``-m smoke``. Such a combination is a
partition mistake; this plugin checks every item the session selects (after
``-m`` deselection, as the former lint test did) and fails the session before
the first test runs. The marker names themselves are
declared in ``pyproject.toml`` (``[tool.pytest.ini_options] markers``), in
lockstep with ``tests/composition_expectations/marker_algebra.py``.

Registered by ``tests/conftest.py`` through ``pytest_plugins``.
"""

from __future__ import annotations

import importlib.util
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

#: Markers that may never combine with ``smoke`` on one resolved item.
#: ``heavy``/``slow``: additive markers keep the item in ``-m smoke`` despite
#: the heavier tier. ``rare`` only runs on request, so a smoke+rare item would
#: sit in the commit gate with no duration budget. ``serial`` claims load
#: sensitivity, which contradicts running inside the parallel commit gate.
#: ``big_memory`` names a slow-tier real-model test the Weekly runner excludes
#: for memory; a smoke test is never memory-heavy enough to need that.
SMOKE_INCOMPATIBLE_MARKERS = ("heavy", "slow", "serial", "rare", "big_memory")

_SMOKE_ONLY_MARKEXPR = re.compile(r"^\s*smoke\s*(?:$|and\s)")


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
    """

    for mark in item.iter_markers("smoke_cells"):
        if item.name in mark.args:
            item.add_marker(pytest.mark.smoke)
            return


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
    """Apply ``--tier-shard``, then fail on any SELECTED contradictory tier combination.

    Runs after the mark plugin's deselection, so a session judges the items it
    will run (targeted runs included); items the session deselects are not
    judged, as the former lint test's ``session.items`` census did. The shard
    filter keeps the invocation rooted at ``tests/``, so the conftest gates that
    arm only on an unfiltered ``tests/`` run stay armed in every shard.
    """

    apply_tier_shard(config, items)
    fail_on_tier_violations(items)


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
