"""Tests for the four tier-aligned crawler campaigns."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from menagerie.crawler.cli import _snapshot_driver_config
from menagerie.crawler.driver_contracts import DriverConfig
from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.intake import load_intake_snapshot
from menagerie.crawler.partitioner import (
    CAMPAIGN_SPECS,
    REVIEW_COHORT_SIZE,
    assert_campaign_partition,
    campaign_binding_for_snapshot,
    load_campaign_bindings,
    partition_roster,
)

_REPO_ROOT = Path(__file__).parents[3]
_ROSTER_PATH = _REPO_ROOT / "menagerie" / "data" / "crawl_roster.jsonl"
_RECORDS_ROOT = _REPO_ROOT / "menagerie" / "crawler" / "records"
_EXPECTED_COUNTS = {
    "c1-mech": 6_968,
    "c2-disco": 14_798,
    "c3-classics": 5_669,
    "c4-native": 1_047,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a test JSONL fixture as object rows.

    Parameters
    ----------
    path:
        JSONL path.

    Returns
    -------
    list[dict[str, Any]]
        Parsed rows.
    """

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _natural_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    """Return natural keys for roster or intake rows.

    Parameters
    ----------
    rows:
        Roster-shaped rows.

    Returns
    -------
    set[tuple[str, str, str]]
        Name, zoo, and variant keys.
    """

    return {
        (str(row["name"]), str(row["zoo"]), str(row.get("variant", "")))
        for row in rows
    }


def test_real_roster_campaigns_are_disjoint_and_complete() -> None:
    """Every real roster row lands in exactly one family-co-located campaign."""

    roster = _read_jsonl(_ROSTER_PATH)
    emitted = {
        spec.campaign_id: _read_jsonl(
            _RECORDS_ROOT / "partitions" / f"{spec.campaign_id}.jsonl"
        )
        for spec in CAMPAIGN_SPECS
    }

    assert len(roster) == 28_482
    assert {campaign_id: len(rows) for campaign_id, rows in emitted.items()} == _EXPECTED_COUNTS
    assert_campaign_partition(roster, emitted)
    expected = partition_roster(roster)
    assert all(emitted[campaign_id] == list(rows) for campaign_id, rows in expected.items())

    key_sets = [_natural_keys(rows) for rows in emitted.values()]
    assert sum(len(keys) for keys in key_sets) == len(set().union(*key_sets)) == len(roster)
    assert set().union(*key_sets) == _natural_keys(roster)

    family_owners: dict[str, set[str]] = defaultdict(set)
    for campaign_id, rows in emitted.items():
        for row in rows:
            family = row.get("family")
            if isinstance(family, str) and family:
                family_owners[family].add(campaign_id)
    assert all(len(owners) == 1 for owners in family_owners.values())


def test_c1_review_prefix_is_zoo_proportional() -> None:
    """The C1 canary prefix follows the full zoo mix within Hamilton rounding."""

    c1_rows = _read_jsonl(_RECORDS_ROOT / "partitions" / "c1-mech.jsonl")
    all_counts = Counter(str(row["zoo"]) for row in c1_rows)
    prefix_counts = Counter(str(row["zoo"]) for row in c1_rows[:REVIEW_COHORT_SIZE])

    assert len(c1_rows[:REVIEW_COHORT_SIZE]) == REVIEW_COHORT_SIZE
    for zoo, count in all_counts.items():
        exact_share = REVIEW_COHORT_SIZE * count / len(c1_rows)
        assert abs(prefix_counts[zoo] - exact_share) < 1


def test_campaign_manifest_binds_models_partitions_and_intakes() -> None:
    """The manifest binds each campaign to exact models and verified snapshots."""

    manifest_path = _RECORDS_ROOT / "campaigns.json"
    bindings = load_campaign_bindings(manifest_path)
    expected_models = {
        "c1-mech": ("claude-sonnet", "gpt-5.6-terra"),
        "c2-disco": ("claude-sonnet", "gpt-5.6-terra"),
        "c3-classics": ("claude-opus-5", "gpt-5.6-sol"),
        "c4-native": ("claude-sonnet", "gpt-5.6-terra"),
    }

    assert DriverConfig().author_model == "claude-sonnet"
    assert DriverConfig().checker_model == "gpt-5.6-terra"
    assert {
        binding.spec.campaign_id: (
            binding.spec.author_model,
            binding.spec.checker_model,
        )
        for binding in bindings
    } == expected_models

    for binding in bindings:
        partition_path = _RECORDS_ROOT / binding.partition_path
        partition_rows = _read_jsonl(partition_path)
        snapshot = load_intake_snapshot(_RECORDS_ROOT / binding.intake_path)
        intake_rows = [item.to_dict() for item in snapshot.items]

        assert binding.row_count == _EXPECTED_COUNTS[binding.spec.campaign_id]
        assert hash_bytes(partition_path.read_bytes()) == binding.partition_sha256
        assert len(snapshot.items) == binding.row_count
        assert _natural_keys(partition_rows) == _natural_keys(intake_rows)
        assert campaign_binding_for_snapshot(manifest_path, snapshot) == binding
        config = _snapshot_driver_config(_REPO_ROOT, snapshot)
        assert config.author_model == binding.spec.author_model
        assert config.checker_model == binding.spec.checker_model
