"""C03 row gate: the fire-evidence rule and the misfire test.

Leverage B6 / ledger D1a / edits D12, one substrate:

- FIRE EVIDENCE: ``intervention_audit`` is NEVER empty after an attempt --
  every door (do/replay, do/set_only, capture ``intervene=``) writes one
  transaction envelope, including no-fire and error outcomes.
- MISFIRE: a rule that never fired is DATA (``no_fire`` status /
  ``zero_fire_rule_ids``), never silence.
- DISTINGUISHABILITY (the ledger memo's D1d oracle): two transactions
  differing in ANY edit parameter (``noise(std=0.1)`` vs ``noise(std=0.9)``)
  produce different canonical audit rows -- they were byte-identical before
  the ONE builder existed.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _ready_trace(model: nn.Module) -> tl.Trace:
    torch.manual_seed(0)
    return tl.trace(
        model,
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )


def _event_rows(trace: tl.Trace) -> list[dict]:
    return [
        row
        for row in trace.state_history
        if isinstance(row, dict) and row.get("op") == "intervention_event"
    ]


# ---------------------------------------------------------------------------
# Fire evidence: audit never empty after an attempt
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_do_replay_attempt_leaves_audit_evidence() -> None:
    model = _TinyModel()
    fork = _ready_trace(model).fork()
    assert fork.intervention_audit == []
    fork.do(tl.when(tl.func("relu"), tl.scale(0.0)))
    assert fork.intervention_audit, "audit empty after a fired do() attempt"
    events = _event_rows(fork)
    assert events and events[-1]["status"] == "fired"
    assert events[-1]["fire_count"] >= 1
    assert events[-1]["lane"] == "replay"
    # site-key-first target refs: the fired site resolves to its site_key_v1
    assert any(key.startswith("s1|") for key in events[-1]["site_keys"])


@pytest.mark.smoke
def test_do_set_only_attempt_leaves_audit_evidence() -> None:
    model = _TinyModel()
    fork = _ready_trace(model).fork()
    fork.do(
        "relu_1_2",
        torch.zeros(2, 4),
        intervention=tl.options.InterventionOptions(engine="set_only"),
    )
    assert fork.intervention_audit, "audit empty after a set_only attempt"
    events = _event_rows(fork)
    assert events[-1]["lane"] == "set_only"
    assert events[-1].get("staged_only") is True


@pytest.mark.smoke
def test_capture_door_attempt_leaves_audit_evidence() -> None:
    model = _TinyModel()
    log = tl.trace(model, torch.randn(2, 4), intervene=tl.when(tl.func("relu"), tl.scale(0.0)))
    assert log.intervention_audit, "audit empty after an intervened capture"
    events = _event_rows(log)
    assert events[-1]["lane"] == "capture"
    assert events[-1]["status"] == "fired"
    assert events[-1]["fire_count"] >= 1


@pytest.mark.smoke
def test_failed_do_attempt_leaves_error_evidence() -> None:
    """A refused attempt is still an attempt: the envelope records the error."""

    model = _TinyModel()
    fork = _ready_trace(model).fork()
    from torchlens._errors import InvalidArgumentError

    bad = tl.when(lambda ctx: True, tl.scale(0.0))  # value-dependent: refused on replay
    with pytest.raises(InvalidArgumentError):
        fork.do(bad)
    events = _event_rows(fork)
    assert events and events[-1]["status"] == "error"
    assert events[-1]["error"]


# ---------------------------------------------------------------------------
# Misfire: zero fires are data, never silence
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_zero_match_capture_records_no_fire() -> None:
    model = _TinyModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log = tl.trace(
            model, torch.randn(2, 4), intervene=tl.when(tl.func("conv2d"), tl.scale(0.0))
        )
    events = _event_rows(log)
    assert events, "no envelope after a zero-match intervened capture"
    assert events[-1]["status"] == "no_fire"
    assert events[-1]["fire_count"] == 0
    assert log.intervention_audit, "canonical audit empty after a no-fire attempt"


@pytest.mark.smoke
def test_two_edit_misfire_names_the_unfired_rule() -> None:
    """The two-edit misfire test: one rule fires, the other's silence is named.

    On the capture lane a rule whose WHERE never matches is a recorded
    misfire (``zero_fire_rule_ids``), never silence.
    """

    model = _TinyModel()
    firing = tl.when(tl.func("relu"), tl.scale(0.0))
    misfiring = tl.when(tl.func("softmax"), tl.noise(std=0.5))  # no softmax in the model
    spec = firing.merge(misfiring)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log = tl.trace(model, torch.randn(2, 4), intervene=spec)
    events = _event_rows(log)
    assert events, "no envelope after a two-edit intervened capture"
    row = events[-1]
    assert row["fire_count"] >= 1
    assert misfiring.rules[0].rule_id in row["zero_fire_rule_ids"]
    assert firing.rules[0].rule_id not in row["zero_fire_rule_ids"]


@pytest.mark.smoke
def test_replay_door_zero_site_rule_refuses_with_error_evidence() -> None:
    """On the recorded-graph door a zero-site rule REFUSES typed (never a
    silent no-op), and the refused attempt still leaves error evidence."""

    model = _TinyModel()
    fork = _ready_trace(model).fork()
    spec = tl.when(tl.func("relu"), tl.scale(0.0)).merge(
        tl.when(tl.func("softmax"), tl.noise(std=0.5))
    )
    from torchlens.intervention.errors import SiteResolutionError

    with pytest.raises(SiteResolutionError):
        fork.do(spec)
    events = _event_rows(fork)
    assert events and events[-1]["status"] == "error"


# ---------------------------------------------------------------------------
# Distinguishability (D1d): any differing edit parameter changes the record
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_noise_std_distinguishes_audit_rows() -> None:
    model = _TinyModel()
    log = _ready_trace(model)
    torch.manual_seed(1)
    low = log.fork()
    low.do(tl.when(tl.func("relu"), tl.noise(std=0.1)))
    torch.manual_seed(1)
    high = log.fork()
    high.do(tl.when(tl.func("relu"), tl.noise(std=0.9)))
    low_row = low.intervention_audit[-1]
    high_row = high.intervention_audit[-1]
    assert low_row != high_row, "audit rows byte-identical across differing noise std"
    assert low_row["resolve_digest"] != high_row["resolve_digest"]
    low_event = _event_rows(low)[-1]
    high_event = _event_rows(high)[-1]
    assert low_event["rules"][0]["action"] != high_event["rules"][0]["action"]
    assert "0.1" in low_event["rules"][0]["action"]
    assert "0.9" in high_event["rules"][0]["action"]


@pytest.mark.smoke
def test_sweep_members_carry_distinguishable_values() -> None:
    """Swept members' envelopes differ by the typed swept value (no closures)."""

    from torchlens.intervention.sweep import sweep

    model = _TinyModel()
    bundle = sweep(model, torch.randn(2, 4), at="relu", values=[0.0, 1.5])
    digests = []
    for member in bundle.members.values():
        rows = _event_rows(member)
        assert rows, "swept member missing its capture envelope"
        digests.append(rows[-1]["event_digest"])
        assert "sweep_replace" in rows[-1]["edit_names"][0]
    assert digests[0] != digests[1]


# ---------------------------------------------------------------------------
# The envelope chain and the ONE builder's uniform stamps
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_event_chain_is_lineage_ordered() -> None:
    model = _TinyModel()
    fork = _ready_trace(model).fork()
    fork.do(tl.when(tl.func("relu"), tl.scale(0.5)))
    fork.do(tl.when(tl.func("relu"), tl.scale(0.25)))
    rows = _event_rows(fork)
    assert len(rows) == 2
    lineage_a, ordinal_a = rows[0]["event_id"].split(":")
    lineage_b, ordinal_b = rows[1]["event_id"].split(":")
    assert lineage_a == lineage_b
    assert (int(ordinal_a), int(ordinal_b)) == (1, 2)
    assert rows[1]["parent_event_id"] == rows[0]["event_id"]


@pytest.mark.smoke
def test_fire_records_carry_uniform_timestamps() -> None:
    """The ONE builder stamps every FireRecord (replay lane included)."""

    model = _TinyModel()
    fork = _ready_trace(model).fork()
    fork.do(tl.when(tl.func("relu"), tl.scale(0.0)))
    records = [
        record
        for label in fork.op_labels
        for record in (getattr(fork.ops[label], "interventions", None) or ())
    ]
    assert records
    assert all(record.timestamp is not None for record in records)


@pytest.mark.smoke
def test_envelope_rows_survive_save_load() -> None:
    import os
    import tempfile

    model = _TinyModel()
    fork = _ready_trace(model).fork()
    fork.do(tl.when(tl.func("relu"), tl.scale(0.0)))
    path = os.path.join(tempfile.mkdtemp(), "c03_evidence.tlspec")
    tl.save(fork, path)
    loaded = tl.load(path)
    events = _event_rows(loaded)
    assert events and events[-1]["status"] == "fired"
    assert loaded.intervention_audit
