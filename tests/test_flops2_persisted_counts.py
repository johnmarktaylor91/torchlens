"""The counted executed backward survives save and load (BackwardPass.executed_flops_record).

A live pass contributes its census attribution as a closed record when the
trace is saved; a loaded trace reads the counted answer and its attribution
from it. Artifacts written before the field (v9, and v10 before it existed)
load the record as ``None`` and read unknown. A malformed record refuses at
load with ``artifact_executed_flops_record_invalid``, never coerced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import torchlens as tl
from tests.test_flops1_backward_dispatch import ZOO_BUILDERS, _traced_backward
from tests.test_flops2_backward_attribution import SEGMENT_FORWARD, _checkpointed_trace
from torchlens.data_classes.backward_pass import BackwardPass
from torchlens.report import backward_status
from torchlens.report._backward_attribution import executed_flops_record_problem


def _save_load(trace: Any, tmp_path: Path, name: str) -> Any:
    path = tmp_path / f"{name}.tlspec"
    trace.save(path)
    return tl.load(path)


def _row_facts(status: Any) -> list[tuple[Any, ...]]:
    return [
        (row.pass_index, row.grad_fn_label, row.call_index, row.op_label, row.flops)
        for attribution in status.attributions
        for row in attribution.rows
    ]


@pytest.mark.parametrize("case", ["partial_requires_grad", "eager_attention_h2"])
def test_loaded_trace_reads_the_saved_count_and_attribution(case: str, tmp_path: Path) -> None:
    """Round trip: the loaded count, rows and bucket equal the live ones."""

    trace = _traced_backward(*ZOO_BUILDERS[case]())
    try:
        live = backward_status(trace)
        loaded = _save_load(trace, tmp_path, case)
        status = backward_status(loaded)
        assert status.executed_flops == live.executed_flops is not None
        assert status.executed_evidence == "formula_exact"
        assert status.executed_flops_per_pass == live.executed_flops_per_pass
        assert status.uncounted_ops == live.uncounted_ops
        assert status.unattributed_flops == live.unattributed_flops
        assert _row_facts(status) == _row_facts(live)
        (attribution,) = status.attributions
        assert attribution.source == "saved record" and attribution.other_trace_ops is None
        (record,) = loaded.backward_passes
        assert record.dispatched_ops is None
        assert executed_flops_record_problem(record.executed_flops_record) is None
    finally:
        trace.cleanup()


def test_loaded_checkpointed_trace_keeps_its_recompute_label(tmp_path: Path) -> None:
    """The recompute share and its label survive the round trip."""

    trace = _checkpointed_trace("reentrant")
    try:
        loaded = _save_load(trace, tmp_path, "checkpointed")
        status = backward_status(loaded)
        assert status.recompute_flops == SEGMENT_FORWARD
        assert [a.recompute for a in status.attributions] == ["labeled"]
    finally:
        trace.cleanup()


@pytest.mark.parametrize("version", [9, 10])
def test_artifacts_saved_without_the_field_load_none_and_read_unknown(
    version: int, tmp_path: Path
) -> None:
    """A v9 or pre-field v10 pass state default-fills None; the status reads unknown."""

    record = BackwardPass(pass_index=1, trigger="log_backward", implicit=False)
    state = dict(record.__getstate__())
    state.pop("executed_flops_record", None)
    state["tlspec_version"] = version
    restored = BackwardPass.__new__(BackwardPass)
    restored.__setstate__(state)
    assert restored.executed_flops_record is None
    assert restored.flops_attribution is None

    trace = _traced_backward(*ZOO_BUILDERS["frozen_backbone"]())
    try:
        loaded = _save_load(trace, tmp_path, "pre_field")
        (loaded_pass,) = loaded.backward_passes
        loaded_pass.executed_flops_record = None  # what an artifact saved before the field holds
        status = backward_status(loaded)
        assert status.executed_flops is None and status.executed_evidence == "unknown"
        assert "no dispatch record" in status.reason
    finally:
        trace.cleanup()


_GOOD = {
    "convention": "fma2",
    "executed_flops": 30,
    "uncounted_ops": [["sum", 1]],
    "rows": [["mm_back_1_1", 1, "mm_1_1", 30, 0]],
    "unattributed": [["aten.ones_like.default", 0]],
}


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"extra": 1}, "record_keys"),
        ({"executed_flops": 30.0}, "count_type"),
        ({"executed_flops": True}, "count_type"),
        ({"rows": [["mm_back_1_1", 1, "mm_1_1", "30", 0]]}, "rows_entry"),
        ({"rows": [["mm_back_1_1", 0, "mm_1_1", 30, 0]]}, "rows_entry"),
        ({"rows": [[7, 1, "mm_1_1", 30, 0]]}, "rows_entry"),
        ({"uncounted_ops": [["sum", 1.5]]}, "uncounted_ops_entry"),
        ({"unattributed": "aten.ones_like"}, "unattributed_type"),
        ({"convention": "x" * 201}, "convention_type"),
        ({"executed_flops": 31}, "rows_incoherent"),
    ],
)
def test_malformed_records_are_named(change: dict[str, Any], reason: str) -> None:
    """Every malformed shape names its reason; the sound record passes."""

    assert executed_flops_record_problem(_GOOD) is None
    problem = executed_flops_record_problem({**_GOOD, **change})
    assert problem is not None and problem[1] == reason


def test_malformed_record_refuses_at_load(tmp_path: Path) -> None:
    """The load boundary refuses a hand-edited record with its typed code."""

    from torchlens._io import TorchLensIOError
    from torchlens._io.forgery_validation import validate_persisted_forgery_surfaces

    trace = _traced_backward(*ZOO_BUILDERS["frozen_backbone"]())
    try:
        loaded = _save_load(trace, tmp_path, "tampered")
        (loaded_pass,) = loaded.backward_passes
        validate_persisted_forgery_surfaces(loaded)
        loaded_pass.executed_flops_record = {**_GOOD, "executed_flops": -1}
        with pytest.raises(TorchLensIOError) as refusal:
            validate_persisted_forgery_surfaces(loaded)
        assert refusal.value.fields["code"] == "artifact_executed_flops_record_invalid"
    finally:
        trace.cleanup()
