"""Tests for strict read-only reduction across the four crawler campaigns."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, Optional, Sequence

import pytest

from menagerie.crawler.authority import build_authority_context
from menagerie.crawler.campaign_merge import (
    CampaignMergeError,
    CampaignSource,
    merge_campaigns,
)
from menagerie.crawler.identity import canonical_json_bytes, payload_hash
from menagerie.crawler.intake import load_intake_snapshot
from menagerie.crawler.models import JsonObject
from menagerie.crawler.partitioner import CampaignBinding, emit_campaign_partitions
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import CanonicalReducer, default_ledger_paths
from menagerie.crawler.tests.conftest import (
    bind_terminal_attempts,
    make_failed_attempt,
    make_model,
)

_REPO_ROOT = Path(__file__).parents[3]
_ROSTER = _REPO_ROOT / "menagerie" / "data" / "crawl_roster.jsonl"


@dataclass(frozen=True)
class _MergeFixture:
    """Physical four-campaign merge fixture."""

    manifest: Path
    sources: tuple[CampaignSource, ...]
    bindings: tuple[CampaignBinding, ...]
    stable_ids: Mapping[str, str]
    output_root: Path


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write deterministic test JSONL.

    Parameters
    ----------
    path:
        Destination.
    rows:
        JSON object rows.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _roster_rows() -> list[JsonObject]:
    """Return one real roster row for each campaign classifier.

    Returns
    -------
    list[dict[str, Any]]
        Four production-shaped roster rows.
    """

    rows = [json.loads(line) for line in _ROSTER.read_text(encoding="utf-8").splitlines() if line]
    zoos = (
        "torchvision",
        "discovered-pytorch",
        "unregistered-classics-pytorch",
        "discovered-tensorflow",
    )
    return [next(row for row in rows if row["zoo"] == zoo) for zoo in zoos]


def _build_fixture(tmp_path: Path) -> _MergeFixture:
    """Build four isolated campaigns through the production reducer.

    Parameters
    ----------
    tmp_path:
        Temporary test root.
    Returns
    -------
    _MergeFixture
        Complete clean merge fixture.
    """

    primary_records = tmp_path / "primary" / "menagerie" / "crawler" / "records"
    roster_path = tmp_path / "primary" / "menagerie" / "data" / "crawl_roster.jsonl"
    _write_jsonl(roster_path, _roster_rows())
    bindings = emit_campaign_partitions(roster_path, primary_records)
    stable_ids: dict[str, str] = {}
    sources: list[CampaignSource] = []
    for binding in bindings:
        campaign_id = binding.spec.campaign_id
        clone = tmp_path / campaign_id
        records_root = clone / "menagerie" / "crawler" / "records"
        shutil.copytree(primary_records / "intake", records_root / "intake")
        snapshot = load_intake_snapshot(records_root / binding.intake_path)
        stable_id = snapshot.items[0].stable_id
        context = build_authority_context(
            active_intake_snapshot_id=snapshot.snapshot_id,
            active_intake_snapshot_sha256=snapshot.snapshot_sha256,
            intake_rows=(item.to_dict() for item in snapshot.items),
            author_model=binding.spec.author_model,
            author_version="current",
            checker_model=binding.spec.checker_model,
            checker_version="current",
        )
        stable_ids[campaign_id] = stable_id
        attempt_id = f"attempt-{stable_id}"
        attempt = make_failed_attempt(stable_id, attempt_id=attempt_id)
        model = bind_terminal_attempts(
            make_model(stable_id, status_code="failed:source", attempt_id=attempt_id),
            [attempt],
        )
        model["provenance"].update(
            {
                "author_model": binding.spec.author_model,
                "author_version": "current",
                "checker_model": binding.spec.checker_model,
                "checker_version": "current",
            }
        )
        with CanonicalReducer(default_ledger_paths(records_root), context) as reducer:
            reducer.append_attempt(attempt)
            reducer.append_model(reducer.prepare_model(model))
        source = CampaignSource(campaign_id, records_root, clone / ".crawl-local")
        sources.append(source)

    hot_path = sources[0].runtime_root / "instrumentation" / "ledger-hot-path.jsonl"
    _write_jsonl(
        hot_path,
        [
            {
                "metric": "final-authority-per-model-seconds-vs-terminal-count",
                "terminal_count": 10,
                "elapsed_seconds": 0.1,
                "checkpoint_cache_enabled": False,
                "checkpoint_cache_hit": False,
            },
            {
                "metric": "final-authority-per-model-seconds-vs-terminal-count",
                "terminal_count": 20,
                "elapsed_seconds": 0.2,
                "checkpoint_cache_enabled": True,
                "checkpoint_cache_hit": True,
            },
        ],
    )
    return _MergeFixture(
        manifest=primary_records / "campaigns.json",
        sources=tuple(sources),
        bindings=bindings,
        stable_ids=stable_ids,
        output_root=tmp_path / "primary" / "menagerie" / "crawler" / "views" / "merged",
    )


def _source_bytes(sources: tuple[CampaignSource, ...]) -> Mapping[str, bytes]:
    """Snapshot every campaign records byte for read-only assertions.

    Parameters
    ----------
    sources:
        Campaign inputs.

    Returns
    -------
    Mapping[str, bytes]
        Qualified relative paths to exact bytes.
    """

    return {
        f"{source.campaign_id}:{path.relative_to(source.records_root)}": path.read_bytes()
        for source in sources
        for path in source.records_root.rglob("*")
        if path.is_file()
    }


def _rewrite_model(
    source: CampaignSource,
    transform: Callable[[JsonObject], Optional[JsonObject]],
) -> None:
    """Rewrite one synthetic model fixture before the merge begins.

    Parameters
    ----------
    source:
        Campaign whose test input changes.
    transform:
        Callable mutating or replacing the sole row.
    """

    path = source.records_root / "models" / "current-shard.jsonl"
    records = scan_jsonl(path, validate=False)
    transformed = transform(deepcopy(records[0]))
    if transformed is not None:
        transformed["record_revision"] = payload_hash(transformed)
    _write_jsonl(path, [] if transformed is None else [transformed])


def test_clean_four_way_merge_is_read_only_and_reports_slope(tmp_path: Path) -> None:
    """A clean merge proves coverage, preserves sources, and reports risk R5."""

    fixture = _build_fixture(tmp_path)
    before = _source_bytes(fixture.sources)

    result = merge_campaigns(fixture.manifest, fixture.sources, fixture.output_root)

    assert len(result.current_records) == 4
    assert result.report["proof"] == {
        "frozen_partition_rechecked": True,
        "actual_processed_pairwise_disjoint": True,
        "actual_processed_union_equals_roster": True,
        "author_model_identity_matches_manifest": True,
        "campaign_ledgers_read_only": True,
        "final_checkpoint_validation": "passed",
    }
    assert result.report["total"]["status_counts"] == {"failed:source": 4}
    assert result.report["total"]["quality_rate"] == 0.0
    assert result.report["total"]["reject_rate"] == 1.0
    assert result.report["throughput"]["risk_r5"] == {
        "alert": True,
        "positive_slope_campaigns": ["c1-mech"],
        "interpretation": (
            "positive slope means per-model final-authority time grows with terminal count"
        ),
    }
    assert (fixture.output_root / "merge-report.json").is_file()
    assert _source_bytes(fixture.sources) == before


def test_duplicate_stable_id_across_campaigns_fails_loudly(
    tmp_path: Path,
) -> None:
    """A stable ID processed in two campaigns names the duplicate and refuses."""

    fixture = _build_fixture(tmp_path)
    duplicate = fixture.stable_ids["c1-mech"]

    def duplicate_id(record: JsonObject) -> JsonObject:
        """Move the C2 row onto C1's stable identity."""

        record["stable_id"] = duplicate
        return record

    _rewrite_model(fixture.sources[1], duplicate_id)
    with pytest.raises(
        CampaignMergeError,
        match=rf"duplicate_stable_ids=\['{duplicate}'\]",
    ):
        merge_campaigns(fixture.manifest, fixture.sources, fixture.output_root)
    assert not fixture.output_root.exists()


def test_missing_stable_id_across_campaigns_fails_loudly(
    tmp_path: Path,
) -> None:
    """An unprocessed roster ID names the omission and refuses."""

    fixture = _build_fixture(tmp_path)
    missing = fixture.stable_ids["c4-native"]
    _rewrite_model(fixture.sources[3], lambda _record: None)

    with pytest.raises(
        CampaignMergeError,
        match=rf"missing_stable_ids=\['{missing}'\]",
    ):
        merge_campaigns(fixture.manifest, fixture.sources, fixture.output_root)
    assert not fixture.output_root.exists()


def test_author_model_mismatch_fails_loudly(
    tmp_path: Path,
) -> None:
    """A campaign model label differing from its frozen binding refuses."""

    fixture = _build_fixture(tmp_path)
    stable_id = fixture.stable_ids["c3-classics"]

    def wrong_author(record: JsonObject) -> JsonObject:
        """Replace Opus with the wrong author tier."""

        record["provenance"]["author_model"] = "claude-sonnet"
        return record

    _rewrite_model(fixture.sources[2], wrong_author)
    with pytest.raises(
        CampaignMergeError,
        match=rf"{stable_id}:author_model='claude-sonnet',expected='claude-opus-5'",
    ):
        merge_campaigns(fixture.manifest, fixture.sources, fixture.output_root)
    assert not fixture.output_root.exists()
