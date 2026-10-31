"""Explorer sign-off 4: no raw-tensor snapshot tier; a nullable ``trace_ref`` instead.

``StepBlockRecord.trace_ref`` points a step at a retained ``Trace`` (or is
``None``). It persists through the history artifact, history schema v1
artifacts written before the field existed load unchanged with
``trace_ref=None``, and a malformed pointer refuses typed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from torchlens.observability import (
    HISTORY_SCHEMA_VERSION,
    CommittedBlock,
    HistoryReader,
    HistoryWriter,
    ObservationRecord,
    RunningStats,
    RunRecord,
    SiteRecord,
    StepBlockRecord,
)
from torchlens.observability._errors import HistorySchemaError
from torchlens.observability._op_tier import OpTierCollector


def _committed(step: int, trace_ref: str | None) -> CommittedBlock:
    spine = RunningStats()
    spine.update(torch.tensor([1.0, -2.0, 0.0]))
    observation = ObservationRecord(
        global_step=step,
        site_id="module:enc.0",
        stream="activation",
        phase="forward",
        presence="observed",
        spine=spine.result(),
    )
    record = StepBlockRecord(
        segment_id="seg-a", global_step=step, provenance="explicit", trace_ref=trace_ref
    )
    return CommittedBlock(block=record, observations=(observation,), step_lo=step, step_hi=step)


def _write(path: Path) -> None:
    writer = HistoryWriter(path, RunRecord(run_id="run-a", segment_id="seg-a"))
    writer.add_site(SiteRecord(site_id="module:enc.0", kind="module", display_label="enc.0"))
    writer.append_block(_committed(0, None))
    writer.append_block(_committed(1, "traces/step-1.tlspec"))


def test_trace_ref_round_trips_and_defaults_to_none(tmp_path: Path) -> None:
    _write(tmp_path)
    blocks = HistoryReader(tmp_path).step_blocks()
    assert [block.trace_ref for block in blocks] == [None, "traces/step-1.tlspec"]
    assert HistoryReader(tmp_path).run.schema_version == HISTORY_SCHEMA_VERSION == 2


def test_schema_v1_artifact_without_trace_ref_loads_unchanged(tmp_path: Path) -> None:
    """An artifact written before the field existed: no key, schema_version 1."""

    _write(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["run"]["schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest))
    index_path = tmp_path / "index.json"
    index = json.loads(index_path.read_text())
    for chunk in index["chunks"]:
        del chunk["block"]["trace_ref"]
    index_path.write_text(json.dumps(index))
    reader = HistoryReader(tmp_path)
    assert reader.run.schema_version == 1
    assert [block.trace_ref for block in reader.step_blocks()] == [None, None]
    assert len(list(reader.observations())) == 2


@pytest.mark.parametrize("bad", ["", 7])
def test_malformed_trace_ref_refuses_typed(bad: object) -> None:
    with pytest.raises(HistorySchemaError) as info:
        StepBlockRecord(segment_id="s", global_step=0, provenance="explicit", trace_ref=bad)  # type: ignore[arg-type]
    assert info.value.fields["code"] == "history_trace_ref_invalid"


def test_op_tier_settles_trace_ref_and_unknown_optimizer_status() -> None:
    """The op tier's settle door: the caller's truth wins over StepTruth's default."""

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    collector = OpTierCollector(model)
    collector.discover(torch.randn(2, 4))
    with collector.step(0):
        collector.observe_step(torch.randn(2, 4))
        collector.settle_open_block(optimizer_status="unknown", trace_ref="kept/step-0")
    with collector.step(1):
        collector.observe_step(torch.randn(2, 4))
    first, second = (committed.block for committed in collector.ring.blocks)
    assert (first.optimizer_status, first.trace_ref) == ("unknown", "kept/step-0")
    assert (second.optimizer_status, second.trace_ref) == ("applied", None)
