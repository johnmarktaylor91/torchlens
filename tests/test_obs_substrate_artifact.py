"""History artifact tests: atomic append, recovery, ring policies (D19/D20)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from torchlens._observability import (
    Histogram,
    ObservationRecord,
    RunningStats,
    RunRecord,
    SiteRecord,
    StepBlockRecord,
)
from torchlens._observability._artifact import CommittedBlock, HistoryWriter, RamRing
from torchlens.errors import HistoryArtifactError
from torchlens.trackers import HistoryReader


def _run(run_id: str = "run-a", segment_id: str = "seg-a") -> RunRecord:
    return RunRecord(run_id=run_id, segment_id=segment_id)


def _site(site_id: str = "module:enc.0") -> SiteRecord:
    return SiteRecord(site_id=site_id, kind="module", display_label="enc.0", numel=64)


def _block(step: int, *, with_sketch: bool = False, value: float = 1.0) -> CommittedBlock:
    spine = RunningStats()
    spine.update(torch.tensor([value, -value, 0.0]))
    sketch = None
    if with_sketch:
        histogram = Histogram()
        histogram.update(torch.tensor([value, -value, 0.0]))
        sketch = histogram.compute()
    observation = ObservationRecord(
        global_step=step,
        site_id="module:enc.0",
        stream="activation",
        phase="forward",
        presence="observed",
        spine=spine.compute(),
        sketch=sketch,
    )
    record = StepBlockRecord(segment_id="seg-a", global_step=step, provenance="explicit")
    return CommittedBlock(block=record, observations=(observation,), step_lo=step, step_hi=step)


class TestRoundTrip:
    """Write -> read equality, incl. sketch payloads and nullable cells."""

    def test_full_round_trip(self, tmp_path: Path) -> None:
        writer = HistoryWriter(tmp_path / "hist", _run())
        writer.add_site(_site())
        for step in range(3):
            writer.append_block(_block(step, with_sketch=step == 1))
        reader = HistoryReader(tmp_path / "hist")
        assert reader.run.run_id == "run-a"
        assert reader.num_chunks == 3
        assert [b.global_step for b in reader.step_blocks()] == [0, 1, 2]
        rows = reader.observations(step=1)
        assert len(rows) == 1
        row = rows[0]
        assert row.spine is not None and row.spine.numel == 3
        assert row.sketch is not None and row.sketch.specials["zero"] == 1
        no_sketch = reader.observations(step=0)[0]
        assert no_sketch.sketch is None

    def test_readable_without_model_and_indexed_by_site_stream(self, tmp_path: Path) -> None:
        writer = HistoryWriter(tmp_path / "hist", _run())
        writer.add_site(_site())
        writer.append_block(_block(0))
        reader = HistoryReader(tmp_path / "hist")
        assert reader.observations(site_id="module:enc.0", stream="activation")
        assert reader.observations(site_id="module:enc.0", stream="param") == []
        assert reader.sites["module:enc.0"].display_label == "enc.0"

    def test_uncataloged_site_refuses(self, tmp_path: Path) -> None:
        writer = HistoryWriter(tmp_path / "hist", _run())
        with pytest.raises(HistoryArtifactError) as excinfo:
            writer.append_block(_block(0))
        assert excinfo.value.fields["code"] == "history_artifact_invalid"

    def test_existing_manifest_refuses_forged_identity(self, tmp_path: Path) -> None:
        HistoryWriter(tmp_path / "hist", _run())
        with pytest.raises(HistoryArtifactError) as excinfo:
            HistoryWriter(tmp_path / "hist", _run(run_id="other"))
        assert excinfo.value.fields["code"] == "history_artifact_invalid"


class TestCrashRecovery:
    """A killed writer leaves the artifact readable to the last committed chunk."""

    def test_committed_chunk_corruption_refuses_typed(self, tmp_path: Path) -> None:
        art = tmp_path / "hist"
        writer = HistoryWriter(art, _run())
        writer.add_site(_site())
        writer.append_block(_block(0))
        chunk = art / "chunk-000000.npz"
        payload = bytearray(chunk.read_bytes())
        payload[40] ^= 0xFF
        chunk.write_bytes(bytes(payload))
        with pytest.raises(HistoryArtifactError) as excinfo:
            HistoryReader(art).observations()
        assert excinfo.value.fields["code"] == "history_artifact_corrupt"

    def test_missing_committed_chunk_refuses_typed(self, tmp_path: Path) -> None:
        art = tmp_path / "hist"
        writer = HistoryWriter(art, _run())
        writer.add_site(_site())
        writer.append_block(_block(0))
        os.remove(art / "chunk-000000.npz")
        with pytest.raises(HistoryArtifactError) as excinfo:
            HistoryReader(art).observations()
        assert excinfo.value.fields["code"] == "history_artifact_corrupt"

    def test_newer_schema_refuses(self, tmp_path: Path) -> None:
        art = tmp_path / "hist"
        HistoryWriter(art, _run())
        manifest = json.loads((art / "manifest.json").read_text())
        manifest["run"]["schema_version"] = 99
        (art / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(HistoryArtifactError) as excinfo:
            HistoryReader(art)
        assert excinfo.value.fields["code"] == "history_artifact_invalid"


class TestRamRing:
    """The three explicit RAM-only policies; disk eviction is a non-event."""

    def test_drop_oldest_counts(self) -> None:
        ring = RamRing(capacity=2, policy="drop_oldest")
        for step in range(5):
            ring.will_admit()
            ring.admit(_block(step))
        assert len(ring) == 2
        assert ring.dropped_blocks == 3
        assert [b.block.global_step for b in ring.blocks] == [3, 4]

    def test_coarsen_preserves_counts_with_spans(self) -> None:
        ring = RamRing(capacity=2, policy="coarsen")
        for step in range(4):
            ring.will_admit()
            ring.admit(_block(step))
        assert ring.coarsen_merges == 2
        first = ring.blocks[0]
        assert first.coarsened
        assert first.step_lo == 0 and first.step_hi >= 1
        merged_obs = first.observations[0]
        assert merged_obs.spine is not None
        # Exact count preservation across the pairwise merges.
        total = sum(
            b.observations[0].spine.numel  # type: ignore[union-attr]
            for b in ring.blocks
        )
        assert total == 4 * 3

    def test_disk_backed_eviction_is_plain_lru_non_event(self) -> None:
        ring = RamRing(capacity=2, policy="refuse", disk_backed=True)
        for step in range(5):
            ring.will_admit()  # never raises with disk on
            ring.admit(_block(step))
        assert len(ring) == 2
        assert ring.dropped_blocks == 0

    def test_policy_vocabulary(self) -> None:
        with pytest.raises(HistoryArtifactError):
            RamRing(policy="best_effort")
        with pytest.raises(HistoryArtifactError):
            RamRing(capacity=1)
