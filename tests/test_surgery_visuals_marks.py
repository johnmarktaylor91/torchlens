"""Surgery visuals unit gates (lane F43): marks, wording, census, diff.

Every claim the renderer draws must cite a C03 audit entry: fire records,
transaction envelopes, region fact rows, injected-op records. These tests
pin the fact-vs-heuristic line-style law, the per-lane wording (one lane's
text is false in the other lane and never borrowed), the never-worded-as-
execution-removal rule, the census travel, and the site-key-first diff on
hand-built substrates. Real-model gates live in
``test_surgery_visuals_realism.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens._visualization.surgery_visuals import (
    surgery_census,
    surgery_facts,
    surgery_stage,
)
from torchlens.viz import intervention_diff

_CAPTURE = tl.options.CaptureOptions(replayable=True, save_args=True)

#: The execution-removal verbs no surgery wording may ever contain (the F38
#: lint's vocabulary; unrepresentable here by construction).
_BANNED_VERBS = ("skipped", "deleted", "removed")

_REPLAY_WORDING = "exit values were substituted; the interior was not replayed"
_LIVE_WORDING = "the original op ran; edited values replaced its output after execution"


class _Chain(nn.Module):
    """fc1 -> relu -> fc2 -> tanh -> fc3: the linear-chain substrate."""

    def __init__(self) -> None:
        """Build the three linears."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain."""

        return self.fc3(torch.tanh(self.fc2(torch.relu(self.fc1(x)))))


class _ChainExtra(nn.Module):
    """The chain plus one sigmoid: the structural-drift diff substrate."""

    def __init__(self) -> None:
        """Build the three linears."""

        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the chain with an extra sigmoid."""

        return self.fc3(torch.sigmoid(torch.tanh(self.fc2(torch.relu(self.fc1(x))))))


class _SharedBlock(nn.Module):
    """Calls the SAME submodule twice: the site-key-cohort substrate."""

    def __init__(self) -> None:
        """Build one shared linear."""

        super().__init__()
        self.block = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Two passes of the shared block with a relu between."""

        return self.block(torch.relu(self.block(x)))


@pytest.fixture()
def chain_capture():
    """An intervention-ready chain capture plus the model/input."""

    torch.manual_seed(0)
    model = _Chain().eval()
    x = torch.randn(3, 4)
    return model, x, tl.trace(model, x, capture=_CAPTURE)


def _assert_no_banned_verbs(text: str) -> None:
    """No surgery wording may claim execution removal."""

    lowered = text.lower()
    for verb in _BANNED_VERBS:
        assert verb not in lowered, verb


# ---------------------------------------------------------------------------
# the mark family: fact vs heuristic, closed vocabularies
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# per-lane wording: one lane's text is false in the other lane
# ---------------------------------------------------------------------------


def test_zero_fire_rule_is_disclosed(chain_capture) -> None:
    """A declared rule that never fired is census DATA, never silence."""

    model, x, _log = chain_capture
    with pytest.warns(UserWarning, match="matched zero sites"):
        log = tl.trace(
            model,
            x,
            intervene=tl.when(tl.func("sigmoid"), tl.zero_ablate()),
            capture=tl.options.CaptureOptions(replayable=True),
        )
    text = surgery_census(log).to_text()
    assert "declared and never fired" in text
    assert "a misfire is data, not silence" in text


@pytest.mark.smoke
def test_region_do_marks_carry_recorded_effect(chain_capture) -> None:
    """Region members/exits mark FACT with the ENGINE-recorded effect."""

    _model, _x, log = chain_capture
    fork = log.fork()
    fork.do(fork.between(["linear_2_3"], ["tanh_1_4"]).as_region(), tl.scale(0.5))
    facts = surgery_facts(fork)
    kinds = {mark.kind for mark in facts.marks}
    assert "region_member" in kinds
    assert "region_exit" in kinds
    assert all(mark.basis == "fact" for mark in facts.marks)
    text = surgery_census(fork).to_text()
    # region rows carry execution_effect verbatim -> source "recorded".
    assert "(recorded)" in text
    assert _REPLAY_WORDING in text


def test_injection_hosts_mark_and_census_count() -> None:
    """F01 stage-1 injected ops mark their host and count in the census."""

    torch.manual_seed(0)
    model = _Chain().eval()
    x = torch.randn(3, 4)
    splice = nn.Linear(4, 4)
    log = tl.trace(
        model,
        x,
        save=tl.func("relu"),
        intervene=tl.when(tl.func("relu"), tl.splice_module(splice)),
        capture=tl.options.CaptureOptions(replayable=True, log_injections=True),
    )
    facts = surgery_facts(log)
    host_marks = [mark for mark in facts.marks if mark.kind == "injection_host"]
    assert len(host_marks) == 1
    assert host_marks[0].basis == "fact"
    fire_rows = [mark.row for mark in facts.marks if mark.kind == "fire"]
    assert any("spliced: splice_module" in row for row in fire_rows)
    assert "injected ops recorded: 1" in surgery_census(log).to_text()


# ---------------------------------------------------------------------------
# the lens row: registration, headline refusal, census travel
# ---------------------------------------------------------------------------


def test_surgery_lens_refuses_without_audit_evidence(chain_capture) -> None:
    """Missing headline evidence refuses typed, naming the capture remedy."""

    _model, _x, log = chain_capture
    with pytest.raises(InvalidArgumentError) as excinfo:
        surgery_stage(log)
    assert excinfo.value.fields["code"] == "surgery_evidence_missing"
    assert "fork.do" in excinfo.value.fields["remedy"]


# ---------------------------------------------------------------------------
# the site-key-first surgery diff
# ---------------------------------------------------------------------------


def test_diff_absence_is_disclosed_never_ghosted(chain_capture) -> None:
    """A one-side op is a full-strength disclosure, never execution removal."""

    _model, x, log = chain_capture
    torch.manual_seed(0)
    other = _ChainExtra().eval()
    subject = tl.trace(other, x)
    diff = intervention_diff(subject, log)
    only_rows = [row for row in diff.rows if row.kind == "only_in_subject"]
    assert len(only_rows) == 1
    assert only_rows[0].subject_label == "sigmoid_1_5:1"
    assert only_rows[0].notes == ("not recorded in the reference capture",)
    text = diff.to_text()
    _assert_no_banned_verbs(text)
    assert "ghost" not in text.lower()


def test_diff_refuses_self_and_empty_operands(chain_capture) -> None:
    """Self-diff and an op-free operand refuse surgery_diff_incomparable."""

    _model, _x, log = chain_capture
    with pytest.raises(InvalidArgumentError) as excinfo:
        intervention_diff(log, log)
    assert excinfo.value.fields["code"] == "surgery_diff_incomparable"

    class _Husk:
        """A trace-shaped husk recording no ops (the empty-operand guard)."""

        op_labels: tuple[str, ...] = ()
        ops: dict = {}
        state_history: tuple = ()
        intervention_audit: list = []

    with pytest.raises(InvalidArgumentError) as excinfo:
        intervention_diff(log, _Husk())  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "surgery_diff_incomparable"
