"""History schema tests: vocabularies, payload coherence, merges (D4-D8)."""

from __future__ import annotations

import pytest
import torch

from torchlens._observability import (
    ObservationRecord,
    RunningStats,
    RunRecord,
    SiteRecord,
    StepBlockRecord,
    merge_observations,
)
from torchlens.errors import HistorySchemaError


def _observed(step: int = 1, *, stream: str = "activation", **kwargs) -> ObservationRecord:
    spine = RunningStats()
    spine.update(torch.tensor([1.0, -2.0, 0.0]))
    defaults: dict = {
        "global_step": step,
        "site_id": "module:block.0",
        "stream": stream,
        "phase": "forward",
        "presence": "observed",
        "spine": spine.compute(),
    }
    defaults.update(kwargs)
    return ObservationRecord(**defaults)


class TestVocabularies:
    """Closed vocabularies refuse invented tokens with the teaching code."""

    @pytest.mark.parametrize(
        "field_name, value",
        [("stream", "logits"), ("phase", "sideways"), ("presence", "maybe")],
    )
    def test_invented_tokens_refuse(self, field_name: str, value: str) -> None:
        with pytest.raises(HistorySchemaError) as excinfo:
            _observed(**{field_name: value})
        assert excinfo.value.fields["code"] == "history_vocab_invalid"
        assert excinfo.value.fields["remedy"]

    def test_step_block_vocab(self) -> None:
        with pytest.raises(HistorySchemaError):
            StepBlockRecord(segment_id="s", global_step=0, provenance="psychic")
        block = StepBlockRecord(segment_id="s", global_step=0, provenance="implicit")
        assert block.optimizer_status == "unknown"

    def test_run_record_validation(self) -> None:
        with pytest.raises(HistorySchemaError):
            RunRecord(run_id="", segment_id="s")
        with pytest.raises(HistorySchemaError):
            RunRecord(run_id="r", segment_id="s", cadences={"activation": 0})
        with pytest.raises(HistorySchemaError):
            RunRecord(run_id="r", segment_id="s", cadences={"nonsense": 1})


class TestPayloadCoherence:
    """Missing is never zero (D6); gradient scale never masquerades (D7)."""

    def test_observed_requires_spine(self) -> None:
        with pytest.raises(HistorySchemaError) as excinfo:
            _observed(spine=None)
        assert excinfo.value.fields["code"] == "history_vocab_invalid"

    def test_non_observed_may_not_carry_payloads(self) -> None:
        with pytest.raises(HistorySchemaError):
            _observed(presence="capture_failed")
        # And the legal spelling works: no payload, a reason.
        record = ObservationRecord(
            global_step=1,
            site_id="module:x",
            stream="activation",
            phase="forward",
            presence="capture_failed",
            reason="reducer raised",
        )
        assert record.spine is None

    def test_gradient_streams_must_stamp_scale(self) -> None:
        with pytest.raises(HistorySchemaError) as excinfo:
            _observed(stream="param_grad", phase="pre_step")
        assert "grad_scale" in str(excinfo.value)
        stamped = _observed(stream="param_grad", phase="pre_step", grad_scale="unknown")
        assert stamped.grad_scale == "unknown"


class TestMerges:
    """Rank/segment merge semantics: exact on integers, typed on conflicts."""

    def test_identity_mismatch_refuses(self) -> None:
        with pytest.raises(HistorySchemaError) as excinfo:
            merge_observations(_observed(1), _observed(2))
        assert excinfo.value.fields["code"] == "history_merge_incompatible"

    def test_conflicting_grad_scale_refuses(self) -> None:
        a = _observed(stream="param_grad", phase="pre_step", grad_scale="scaled")
        b = _observed(stream="param_grad", phase="pre_step", grad_scale="unscaled")
        with pytest.raises(HistorySchemaError) as excinfo:
            merge_observations(a, b)
        assert excinfo.value.fields["code"] == "history_merge_incompatible"

    def test_observed_beats_non_observed_without_invention(self) -> None:
        gap = ObservationRecord(
            global_step=1,
            site_id="module:block.0",
            stream="activation",
            phase="forward",
            presence="capture_failed",
            reason="rank 1 reducer raised",
        )
        merged = merge_observations(_observed(), gap)
        assert merged.presence == "observed"

    def test_conflicting_non_observed_refuses(self) -> None:
        base = {
            "global_step": 1,
            "site_id": "module:block.0",
            "stream": "activation",
            "phase": "forward",
        }
        a = ObservationRecord(presence="capture_failed", reason="x", **base)
        b = ObservationRecord(presence="budget_dropped", reason="y", **base)
        with pytest.raises(HistorySchemaError):
            merge_observations(a, b)

    def test_mixed_sketch_presence_refuses(self) -> None:
        from torchlens._observability import Histogram

        h = Histogram()
        h.update(torch.tensor([1.0]))
        with_sketch = _observed(sketch=h.result())
        without = _observed()
        with pytest.raises(HistorySchemaError) as excinfo:
            merge_observations(with_sketch, without)
        assert excinfo.value.fields["code"] == "history_merge_incompatible"


class TestStepOrder:
    """Duplicate/decreasing steps refuse without a declared new segment."""

    def test_drift_ends_a_series_with_disclosure(self) -> None:
        site = SiteRecord(site_id="module:x", kind="module", display_label="x")
        ended = site.ended(7, "output shape changed at step 7")
        assert ended.ended_step == 7
        assert ended.ended_reason
        assert site.ended_step is None  # original is immutable
