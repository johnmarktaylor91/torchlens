"""History artifact tests: atomic append, recovery, ring policies (D19/D20)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from torchlens.observability import (
    CommittedBlock,
    Histogram,
    HistoryArtifactError,
    HistoryReader,
    HistoryWriter,
    ObservationRecord,
    RamRing,
    RunRecord,
    SiteRecord,
    Spine,
    StepBlockRecord,
    coarsen_pair,
)

pytestmark = pytest.mark.smoke


def _run(run_id: str = "run-a", segment_id: str = "seg-a") -> RunRecord:
    return RunRecord(run_id=run_id, segment_id=segment_id)


def _site(site_id: str = "module:enc.0") -> SiteRecord:
    return SiteRecord(site_id=site_id, kind="module", display_label="enc.0", numel=64)


def _block(step: int, *, with_sketch: bool = False, value: float = 1.0) -> CommittedBlock:
    spine = Spine()
    spine.update(torch.tensor([value, -value, 0.0]))
    sketch = None
    if with_sketch:
        histogram = Histogram()
        histogram.update(torch.tensor([value, -value, 0.0]))
        sketch = histogram.result()
    observation = ObservationRecord(
        global_step=step,
        site_id="module:enc.0",
        stream="activation",
        phase="forward",
        presence="observed",
        spine=spine.result(),
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
        assert reader.n_chunks == 3
        assert [b.global_step for b in reader.step_blocks()] == [0, 1, 2]
        rows = reader.observations(step=1)
        assert len(rows) == 1
        row = rows[0]
        assert row.spine is not None and row.spine.count_total == 3
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

    def test_partial_tmp_chunk_is_ignored(self, tmp_path: Path) -> None:
        art = tmp_path / "hist"
        writer = HistoryWriter(art, _run())
        writer.add_site(_site())
        writer.append_block(_block(0))
        writer.append_block(_block(1))
        # Simulate a kill mid-write: an fsynced-but-unrenamed tmp chunk and
        # no index row for it.
        (art / "chunk-000002.npz.tmp").write_bytes(b"partial garbage")
        reader = HistoryReader(art)
        assert reader.n_chunks == 2
        assert len(reader.observations()) == 2

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

    def test_no_pickle_anywhere(self, tmp_path: Path) -> None:
        art = tmp_path / "hist"
        writer = HistoryWriter(art, _run())
        writer.add_site(_site())
        writer.append_block(_block(0))
        # np.load with allow_pickle=False is the reader's own path; verify
        # the manifest and index are plain JSON.
        json.loads((art / "manifest.json").read_text())
        json.loads((art / "index.json").read_text())
        json.loads((art / "catalog.json").read_text())

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

    def test_refuse_is_default_and_fires_before_the_block(self) -> None:
        ring = RamRing(capacity=2)
        ring.admit(_block(0))
        ring.admit(_block(1))
        with pytest.raises(HistoryArtifactError) as excinfo:
            ring.will_admit()
        assert excinfo.value.fields["code"] == "history_ram_budget_exceeded"
        assert "drop_oldest" in excinfo.value.fields["remedy"]  # remedies named
        assert len(ring) == 2  # nothing dropped or degraded

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
            b.observations[0].spine.count_total  # type: ignore[union-attr]
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

    def test_coarsen_pair_is_count_preserving_and_labeled(self) -> None:
        merged = coarsen_pair(_block(0, value=1.0), _block(1, value=2.0))
        assert merged.coarsened
        assert (merged.step_lo, merged.step_hi) == (0, 1)
        spine = merged.observations[0].spine
        assert spine is not None and spine.count_total == 6
