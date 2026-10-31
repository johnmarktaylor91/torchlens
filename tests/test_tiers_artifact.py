"""The published tier collection artifact is internally consistent and the docs quote it.

``tests/tiers/collection.json`` is written by ``tools/tiers/collection.py`` from
collect-only runs in the pinned smoke environment. These tests run no collection: they
check the artifact against itself, the smoke size ceiling, and the tier numbers that
``test-tiers.json``, ``AGENTS.md`` and ``tests/AGENTS.md`` quote. After regenerating the
artifact, re-true those numbers; ``python tools/tiers/collection.py --check`` compares the
artifact with a fresh collect output.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "checks"))
from docs_lockstep_names import TIER_CLAIM_RE  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = REPO_ROOT / "tests" / "tiers" / "collection.json"
DURATIONS_PATH = REPO_ROOT / "tests" / "tiers" / "durations.csv"
TIER_ORDER = ("smoke", "mid", "not_slow", "all", "unmarked")
NESTED = ("smoke", "mid", "not_slow", "all")


def _load_generator() -> ModuleType:
    path = REPO_ROOT / "tools" / "tiers" / "collection.py"
    spec = importlib.util.spec_from_file_location("_tiers_collection", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    return _load_generator()


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT_PATH.read_text())


def _number(text: str) -> int:
    return int(text.replace(",", ""))


def _smoke_ceiling() -> int:
    source = (REPO_ROOT / "tests" / "conftest.py").read_text()
    match = re.search(r"^SMOKE_TIER_SIZE_CEILING\b[^=]*=\s*([\d_]+)", source, re.MULTILINE)
    assert match, "SMOKE_TIER_SIZE_CEILING not found in tests/conftest.py"
    return int(match.group(1).replace("_", ""))


def test_artifact_lists_every_tier_with_the_generator_command(artifact, generator) -> None:
    assert tuple(artifact["tiers"]) == TIER_ORDER == tuple(generator.TIERS)
    for tier, entry in artifact["tiers"].items():
        assert entry["collect_command"] == generator.collect_command(tier)
        assert isinstance(entry["id_digest"], int) and entry["id_digest"] > 0


def test_tiers_nest_by_count_and_file(artifact) -> None:
    tiers = artifact["tiers"]
    for smaller, larger in zip(NESTED, NESTED[1:]):
        assert 0 < tiers[smaller]["count"] <= tiers[larger]["count"], (smaller, larger)
        assert 0 < tiers[smaller]["files"] <= tiers[larger]["files"], (smaller, larger)
    # smoke and unmarked are disjoint and both sit inside the mid backstop.
    assert tiers["smoke"]["count"] + tiers["unmarked"]["count"] <= tiers["mid"]["count"]


def test_wall_estimates_account_for_every_test(artifact) -> None:
    workers = artifact["estimate_workers"]
    for tier, entry in artifact["tiers"].items():
        estimate = entry["wall_estimate"]
        accounted = estimate["tests_with_duration"] + estimate["tests_without_duration"]
        assert accounted == entry["count"], tier
        assert estimate[f"n{workers}_seconds"] == pytest.approx(
            estimate["serial_seconds"] / workers, abs=0.1
        ), tier


def test_smoke_count_is_within_the_size_ceiling(artifact) -> None:
    assert artifact["tiers"]["smoke"]["count"] <= _smoke_ceiling()


def test_durations_summary_is_sorted_and_covers_the_estimates(artifact) -> None:
    with DURATIONS_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    files = [row["file"] for row in rows]
    assert files == sorted(files) and len(set(files)) == len(files)
    assert all(int(row["tests"]) > 0 and float(row["seconds"]) >= 0 for row in rows)
    total_tests = sum(int(row["tests"]) for row in rows)
    assert total_tests >= artifact["tiers"]["all"]["wall_estimate"]["tests_with_duration"]


def test_generator_round_trips_a_collect_output(generator) -> None:
    text = "tests/a.py::t1\ntests/b.py::t2[x]\n\n2/9 tests collected (7 deselected) in 1.0s\n"
    collected = generator.parse_collect_output(text)
    assert collected.ids == ("tests/a.py::t1", "tests/b.py::t2[x]")
    entry = generator.tier_entry("smoke", collected, {"tests/a.py::t1": 2.0}, {})
    fake = {"tiers": {"smoke": entry}}
    assert generator.check_artifact(fake, {"smoke": collected}) == []
    shrunk = generator.parse_collect_output("tests/a.py::t1\n1 test collected in 1.0s\n")
    assert len(generator.check_artifact(fake, {"smoke": shrunk})) == 2
    with pytest.raises(ValueError, match="errors"):
        generator.parse_collect_output("tests/a.py::t1\n1/2 tests collected, 1 error in 1s\n")


def test_test_tiers_json_quotes_the_artifact_and_maps_real_paths(artifact) -> None:
    config = json.loads((REPO_ROOT / "test-tiers.json").read_text())
    notes = " ".join(config["notes"])
    match = re.search(r"Smoke is ([\d,]+) coverage-chosen tests", notes)
    assert match, "test-tiers.json notes must quote the smoke count"
    assert _number(match.group(1)) == artifact["tiers"]["smoke"]["count"]
    for source, targets in config["inner"]["map"].items():
        for path in (source, *targets):
            assert (REPO_ROOT / path).exists(), f"test-tiers.json inner.map names missing {path}"
    for arg in config["worker_lane"]["args"]:
        if arg.startswith("tests/"):
            assert (REPO_ROOT / arg.split("::", 1)[0]).exists(), arg


def test_agents_docs_quote_the_artifact(artifact) -> None:
    tiers = artifact["tiers"]
    root = (REPO_ROOT / "AGENTS.md").read_text()
    claim = TIER_CLAIM_RE.search(root)
    assert claim, "AGENTS.md Testing Tiers lost its '(S/T collect-only, measured DATE)' claim"
    assert (_number(claim.group("smoke")), _number(claim.group("total"))) == (
        tiers["smoke"]["count"],
        tiers["all"]["count"],
    )
    tests_doc = (REPO_ROOT / "tests" / "AGENTS.md").read_text()
    match = re.search(r"The tier holds ([\d,]+) tests", tests_doc)
    assert match, "tests/AGENTS.md 'The smoke tier' must quote the smoke count"
    assert _number(match.group(1)) == tiers["smoke"]["count"]
    row = re.search(r"\(([\d,]+)/([\d,]+), (\d+)%, measured", tests_doc)
    assert row, "tests/AGENTS.md lost its unmarked census row"
    unmarked, total = _number(row.group(1)), _number(row.group(2))
    assert (unmarked, total) == (tiers["unmarked"]["count"], tiers["all"]["count"])
    assert int(row.group(3)) == round(100 * unmarked / total)
