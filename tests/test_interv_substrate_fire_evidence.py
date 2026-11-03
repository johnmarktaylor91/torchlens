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
        capture=tl.options.CaptureOptions(replayable=True),
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


# ---------------------------------------------------------------------------
# Misfire: zero fires are data, never silence
# ---------------------------------------------------------------------------


def test_replay_door_zero_site_rule_refuses_with_error_evidence() -> None:
    """On the recorded-graph door a zero-site rule REFUSES typed (never a
    silent no-op), and the refused attempt still leaves error evidence."""

    model = _TinyModel()
    fork = _ready_trace(model).fork()
    spec = tl.when(tl.func("relu"), tl.scale(0.0)).merge(
        tl.when(tl.func("softmax"), tl.add_noise(std=0.5))
    )
    from torchlens.intervention.errors import SiteResolutionError

    with pytest.raises(SiteResolutionError):
        fork.do(spec)
    events = _event_rows(fork)
    assert events and events[-1]["status"] == "error"


# ---------------------------------------------------------------------------
# Distinguishability (D1d): any differing edit parameter changes the record
# ---------------------------------------------------------------------------


def test_noise_std_distinguishes_audit_rows() -> None:
    model = _TinyModel()
    log = _ready_trace(model)
    torch.manual_seed(1)
    low = log.fork()
    low.do(tl.when(tl.func("relu"), tl.add_noise(std=0.1)))
    torch.manual_seed(1)
    high = log.fork()
    high.do(tl.when(tl.func("relu"), tl.add_noise(std=0.9)))
    low_row = low.intervention_audit[-1]
    high_row = high.intervention_audit[-1]
    assert low_row != high_row, "audit rows byte-identical across differing noise std"
    assert low_row["resolve_digest"] != high_row["resolve_digest"]
    low_event = _event_rows(low)[-1]
    high_event = _event_rows(high)[-1]
    assert low_event["rules"][0]["action"] != high_event["rules"][0]["action"]
    assert "0.1" in low_event["rules"][0]["action"]
    assert "0.9" in high_event["rules"][0]["action"]


# ---------------------------------------------------------------------------
# The envelope chain and the ONE builder's uniform stamps
# ---------------------------------------------------------------------------


def test_envelope_rows_survive_save_load() -> None:
    import os
    import tempfile

    model = _TinyModel()
    fork = _ready_trace(model).fork()
    fork.do(tl.when(tl.func("relu"), tl.scale(0.0)))
    path = os.path.join(tempfile.mkdtemp(), "c03_evidence.tlspec")
    fork.save(path)
    loaded = tl.load(path)
    events = _event_rows(loaded)
    assert events and events[-1]["status"] == "fired"
    assert loaded.intervention_audit


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
