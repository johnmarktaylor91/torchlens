"""Explorer sign-off 4: no raw-tensor snapshot tier; a nullable ``trace_ref`` instead.

``StepBlockRecord.trace_ref`` points a step at a retained ``Trace`` (or is
``None``). It persists through the history artifact, history schema v1
artifacts written before the field existed load unchanged with
``trace_ref=None``, and a malformed pointer refuses typed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torchlens._observability import (
    ObservationRecord,
    RunningStats,
    RunRecord,
    SiteRecord,
    StepBlockRecord,
)
from torchlens._observability._artifact import CommittedBlock, HistoryWriter
from torchlens._observability._errors import HistorySchemaError


def _committed(step: int, trace_ref: str | None) -> CommittedBlock:
    spine = RunningStats()
    spine.update(torch.tensor([1.0, -2.0, 0.0]))
    observation = ObservationRecord(
        global_step=step,
        site_id="module:enc.0",
        stream="activation",
        phase="forward",
        presence="observed",
        spine=spine.compute(),
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


@pytest.mark.parametrize("bad", ["", 7])
def test_malformed_trace_ref_refuses_typed(bad: object) -> None:
    with pytest.raises(HistorySchemaError) as info:
        StepBlockRecord(segment_id="s", global_step=0, provenance="explicit", trace_ref=bad)  # type: ignore[arg-type]
    assert info.value.fields["code"] == "history_trace_ref_invalid"
