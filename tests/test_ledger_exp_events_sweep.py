"""F03 ledger memo items 1 + 2: canonical EVENT audit rows and sweep baseline.

Item 1's F03 slice: the tlspec-v9 ``kind="EVENT"`` audit-row WRITER (the v9
admission declared the contract; this lane lands the writer) with its
seq/prev-digest hash chain, persisting through save/load validation. Item 2:
``tl.sweep(include_baseline=)`` mints one PRISTINE baseline member (the
measured defect: sweep bundles could not be compared at all), the legacy
default keeps its cardinality and DISCLOSES the missing baseline at
construction.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TorchLensWarning
from torchlens.intervention.audit import (
    append_event_audit_row,
    event_audit_rows,
    record_intervention_event,
)
from torchlens.intervention.errors import BaselineUndeterminedError

pytestmark = pytest.mark.smoke


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


def _ready_trace() -> tl.Trace:
    torch.manual_seed(3)
    return tl.trace(
        _Tiny(), torch.randn(2, 3), capture=tl.options.CaptureOptions(intervention_ready=True)
    )


# ---------------------------------------------------------------------------
# item 1: the EVENT audit-row writer + hash chain
# ---------------------------------------------------------------------------


def _write_event(trace: tl.Trace, *, door: str = "vary", status: str = "fired") -> dict:
    return record_intervention_event(
        trace,
        lane="live_hook",
        door=door,
        edit_names=("zero_ablate",),
        selection_repr="func('relu')",
        status=status,  # type: ignore[arg-type]
        fire_count=1 if status == "fired" else 0,
        site_keys=("relu_1_2",),
        rules=({"rule_id": "r1", "where": "func('relu')", "action": "zero_ablate()"},),
        append_audit_row=False,
        audit_event_row=True,
    )


def test_event_rows_chain_in_canonical_audit() -> None:
    fork = _ready_trace().fork()
    _write_event(fork)
    _write_event(fork, door="site_sweep", status="no_fire")
    events = event_audit_rows(fork)
    assert len(events) == 2
    first, second = events
    assert first["kind"] == "EVENT"
    assert first["schema"] == "intervention_event_v2"
    assert first["seq"] == 1 and first["prev_event_digest"] is None
    assert second["seq"] == 2
    assert second["prev_event_digest"] == first["event_digest"]
    # extra/door-specific keys never enter the closed canonical row.
    assert "staged_only" not in first


def test_event_rows_survive_save_load_validation(tmp_path) -> None:
    fork = _ready_trace().fork()
    _write_event(fork)
    _write_event(fork, status="error")
    path = str(tmp_path / "f03_events.tlspec")
    tl.save(fork, path)
    loaded = tl.load(path)
    events = event_audit_rows(loaded)
    assert len(events) == 2
    assert events[1]["prev_event_digest"] == events[0]["event_digest"]


def test_event_row_projection_strips_extra_disclosures() -> None:
    fork = _ready_trace().fork()
    envelope = record_intervention_event(
        fork,
        lane="live_hook",
        door="vary",
        edit_names=("scale",),
        selection_repr="func('relu')",
        status="fired",
        fire_count=1,
        extra={"staged_only": True},
        append_audit_row=False,
        audit_event_row=True,
    )
    assert envelope["staged_only"] is True  # state_history keeps the disclosure
    (event,) = event_audit_rows(fork)
    assert "staged_only" not in event


def test_append_event_audit_row_is_idempotent_per_envelope_call() -> None:
    fork = _ready_trace().fork()
    envelope = _write_event(fork)
    # A second explicit append chains rather than duplicating in place.
    row = append_event_audit_row(fork, envelope)
    assert row["seq"] == 2


# ---------------------------------------------------------------------------
# item 2: sweep pristine baseline
# ---------------------------------------------------------------------------


def test_sweep_default_keeps_cardinality_and_discloses() -> None:
    torch.manual_seed(0)
    model = _Tiny()
    x = torch.randn(2, 3)
    with pytest.warns(TorchLensWarning, match="NO pristine baseline") as caught:
        bundle = tl.sweep(model, x, at="relu", values=[0.0, 1.0])
    assert caught[0].message.fields["code"] == "sweep_baseline_absent"
    assert bundle.names == ["sweep_0", "sweep_1"]
    assert bundle.baseline_name is None
    with pytest.raises(BaselineUndeterminedError):
        bundle.most_changed()


def test_sweep_include_baseline_mints_pristine_first() -> None:
    torch.manual_seed(0)
    model = _Tiny()
    x = torch.randn(2, 3)
    bundle = tl.sweep(model, x, at="relu", values=[0.0, 1.0], include_baseline=True)
    assert bundle.names == ["baseline", "sweep_0", "sweep_1"]
    assert bundle.baseline_name == "baseline"
    # The pristine member is un-intervened; swept members are intervened.
    assert bundle["baseline"].intervention_audit == []
    ranked = bundle.most_changed()
    assert ranked, "most_changed must run on a baselined sweep"
    # Lineage: anchors + chronology row.
    assert bundle._member_construction["sweep_0"]["origin"] == "swept"
    assert bundle._member_construction["baseline"]["origin"] == "constructed"
    operation = bundle._operations[-1]
    assert operation.kind == "sweep"
    assert operation.params["include_baseline"] is True
    assert operation.params["values"] == ["0.0", "1.0"]


def test_sweep_baseline_name_collision_refuses_typed() -> None:
    torch.manual_seed(0)
    model = _Tiny()
    x = torch.randn(2, 3)
    with pytest.raises(Exception) as excinfo:
        tl.sweep(
            model,
            x,
            at="relu",
            values=[0.0],
            names=["baseline"],
            include_baseline=True,
        )
    assert excinfo.value.fields["code"] == "sweep_baseline_name_collision"  # type: ignore[attr-defined]


def test_swept_members_are_distinguishable_by_value() -> None:
    """The measured amnesia row: differing swept values differ in the record."""

    torch.manual_seed(0)
    model = _Tiny()
    x = torch.randn(2, 3)
    with pytest.warns(TorchLensWarning):
        bundle = tl.sweep(model, x, at="relu", values=[0.25, 0.75])

    def _envelope(member: tl.Trace) -> dict:
        rows = [
            row
            for row in member.state_history
            if isinstance(row, dict) and row.get("op") == "intervention_event"
        ]
        assert rows, "capture door wrote no envelope"
        return rows[-1]

    low = _envelope(bundle["sweep_0"])
    high = _envelope(bundle["sweep_1"])
    assert low["edit_names"] != high["edit_names"]
    assert "0.25" in str(low["edit_names"])
    assert "0.75" in str(high["edit_names"])


def test_sweep_over_specs_include_baseline() -> None:
    torch.manual_seed(0)
    model = _Tiny()
    x = torch.randn(2, 3)
    specs = [
        tl.when(tl.func("relu"), tl.scale(0.5)),
        tl.when(tl.func("relu"), tl.scale(2.0)),
    ]
    bundle = tl.sweep(model, x, values=specs, include_baseline=True)
    assert bundle.names[0] == "baseline"
    assert bundle.baseline_name == "baseline"
    operation = bundle._operations[-1]
    assert operation.kind == "sweep"
    assert len(operation.params["spec_digests"]) == 2
