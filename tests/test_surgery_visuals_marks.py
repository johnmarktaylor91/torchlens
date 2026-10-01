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
from torchlens._vocab.node_spec import NodeSpec
from torchlens.visualization import surgery_diff
from torchlens.visualization.surgery_visuals import (
    CREDIT_ROWS,
    MARK_BASES,
    MARK_KINDS,
    make_surgery_mark_spec_fn,
    render_surgery,
    splice_box_lines,
    surgery_census,
    surgery_facts,
    surgery_stage,
)

_CAPTURE = tl.options.CaptureOptions(intervention_ready=True, save_arg_values=True)

#: The execution-removal verbs no surgery wording may ever contain (the F38
#: lint's vocabulary; unrepresentable here by construction).
_BANNED_VERBS = ("skipped", "deleted", "removed")

_REPLAY_WORDING = "exit values were substituted; the interior was not replayed"
_LIVE_WORDING = "the original op ran; edited values replaced its output after execution"
_STAGED_WORDING = "staged only; nothing has executed"


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


def _edited_fork(log):
    """A fork with one recorded replay edit at the relu."""

    fork = log.fork()
    fork.do(tl.units("relu_1_2", [(0, 0)]).resolve(fork), tl.zero_ablate())
    return fork


def _assert_no_banned_verbs(text: str) -> None:
    """No surgery wording may claim execution removal."""

    lowered = text.lower()
    for verb in _BANNED_VERBS:
        assert verb not in lowered, verb


# ---------------------------------------------------------------------------
# the mark family: fact vs heuristic, closed vocabularies
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_fire_mark_is_fact_and_solid(chain_capture) -> None:
    """A recorded FireRecord yields a FACT mark: solid border, penwidth 3."""

    _model, _x, log = chain_capture
    fork = _edited_fork(log)
    facts = surgery_facts(fork)
    fire_marks = [mark for mark in facts.marks if mark.kind == "fire"]
    assert len(fire_marks) == 1
    assert fire_marks[0].basis == "fact"
    assert fire_marks[0].call_label == "relu_1_2:1"
    assert fire_marks[0].site_key == fork.ops["relu_1_2:1"].site_key
    spec_fn = make_surgery_mark_spec_fn(fork, facts)
    spec = spec_fn(fork.ops["relu_1_2:1"], NodeSpec(["relu_1_2"]))
    assert spec is not None
    assert spec.penwidth == 3.0
    assert "dashed" not in (spec.style or "")
    assert any("zero_ablate" in line for line in spec.lines)


@pytest.mark.smoke
def test_cohort_mark_is_heuristic_and_dashed() -> None:
    """A targeted site's un-fired sibling pass gets a DASHED heuristic mark."""

    torch.manual_seed(0)
    model = _SharedBlock().eval()
    x = torch.randn(3, 4)
    log = tl.trace(model, x, capture=_CAPTURE)
    fork = log.fork()
    fork.do(tl.units("linear_1_1:1", [(0, 0)]).resolve(fork), tl.zero_ablate())
    facts = surgery_facts(fork)
    by_label = {mark.call_label: mark for mark in facts.marks}
    assert by_label["linear_1_1:1"].basis == "fact"
    assert by_label["linear_1_1:2"].basis == "heuristic"
    assert by_label["linear_1_1:2"].kind == "declared_target"
    assert "no fire recorded at this instance" in by_label["linear_1_1:2"].row
    spec_fn = make_surgery_mark_spec_fn(fork, facts)
    heuristic_spec = spec_fn(fork.ops["linear_1_1:2"], NodeSpec(["linear_1_1:2"]))
    assert heuristic_spec is not None
    assert "dashed" in (heuristic_spec.style or "")
    assert heuristic_spec.penwidth == 2.25
    fact_spec = spec_fn(fork.ops["linear_1_1:1"], NodeSpec(["linear_1_1:1"]))
    assert fact_spec is not None
    assert "dashed" not in (fact_spec.style or "")


@pytest.mark.smoke
def test_mark_vocabularies_are_closed(chain_capture) -> None:
    """Every derived mark uses the closed kind and basis vocabularies."""

    _model, _x, log = chain_capture
    fork = _edited_fork(log)
    for mark in surgery_facts(fork).marks:
        assert mark.kind in MARK_KINDS
        assert mark.basis in MARK_BASES


@pytest.mark.smoke
def test_staged_set_only_attributes_no_marks(chain_capture) -> None:
    """A staged envelope with no recorded site refs mints NO marks.

    No ref, no mark, no fabrication -- the transaction still appears in the
    census with the staged lane's own (third) truthful wording.
    """

    _model, _x, log = chain_capture
    fork = log.fork()
    fork.do(
        tl.units("relu_1_2", [(0, 0)]).resolve(fork),
        tl.zero_ablate(),
        intervention=tl.options.InterventionOptions(engine="set_only"),
    )
    facts = surgery_facts(fork)
    assert facts.marks == ()
    assert facts.transactions[-1].lane == "set_only"
    box = "\n".join(splice_box_lines(facts))
    assert _STAGED_WORDING in box
    assert _REPLAY_WORDING not in box
    assert _LIVE_WORDING not in box


# ---------------------------------------------------------------------------
# per-lane wording: one lane's text is false in the other lane
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_replay_lane_wording_never_borrows_live_text(chain_capture) -> None:
    """The replay fork's box carries ONLY the replay lane's wording."""

    _model, _x, log = chain_capture
    fork = _edited_fork(log)
    text = surgery_census(fork).to_text()
    assert _REPLAY_WORDING in text
    assert _LIVE_WORDING not in text
    _assert_no_banned_verbs(text)


@pytest.mark.smoke
def test_live_lane_wording_never_borrows_replay_text() -> None:
    """The live-intervened capture's box carries ONLY the live wording."""

    torch.manual_seed(0)
    model = _Chain().eval()
    x = torch.randn(3, 4)
    log = tl.trace(
        model,
        x,
        intervene=tl.when(tl.func("relu"), tl.zero_ablate()),
        capture=tl.options.CaptureOptions(intervention_ready=True),
    )
    text = surgery_census(log).to_text()
    assert _LIVE_WORDING in text
    assert _REPLAY_WORDING not in text
    _assert_no_banned_verbs(text)


@pytest.mark.smoke
def test_zero_fire_rule_is_disclosed(chain_capture) -> None:
    """A declared rule that never fired is census DATA, never silence."""

    model, x, _log = chain_capture
    with pytest.warns(UserWarning, match="matched zero sites"):
        log = tl.trace(
            model,
            x,
            intervene=tl.when(tl.func("sigmoid"), tl.zero_ablate()),
            capture=tl.options.CaptureOptions(intervention_ready=True),
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


@pytest.mark.smoke
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
        capture=tl.options.CaptureOptions(intervention_ready=True, log_injections=True),
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


@pytest.mark.smoke
def test_surgery_lens_registered_over_c05_registry() -> None:
    """The surgery row rides C05's lens registry, never a new view kind."""

    from torchlens.visualization.theme_registry import get_lens

    lens = get_lens("surgery")
    assert lens.headline_evidence == "intervention_audit"
    assert "vis_mode" not in lens.members  # never forks the view
    for line in lens.disclosure:
        _assert_no_banned_verbs(line)


@pytest.mark.smoke
def test_surgery_lens_refuses_without_audit_evidence(chain_capture) -> None:
    """Missing headline evidence refuses typed, naming the capture remedy."""

    _model, _x, log = chain_capture
    with pytest.raises(InvalidArgumentError) as excinfo:
        surgery_stage(log)
    assert excinfo.value.fields["code"] == "surgery_evidence_missing"
    assert "fork.do" in excinfo.value.fields["remedy"]


@pytest.mark.heavy
def test_render_surgery_census_travels_with_figure(chain_capture, tmp_path) -> None:
    """The census renders INTO the figure and the sidecar travels beside it."""

    _model, _x, log = chain_capture
    fork = _edited_fork(log)
    outpath = str(tmp_path / "surgery_fig")
    render_surgery(fork, vis_outpath=outpath, vis_fileformat="svg", vis_save_only=True)
    figure = tmp_path / "surgery_fig.svg"
    sidecar = tmp_path / "surgery_fig.census.txt"
    assert figure.exists()
    assert sidecar.exists()
    svg = figure.read_text()
    assert "surgery census" in svg
    assert "pyvene" in svg  # CREDIT rows render into the figure caption
    _assert_no_banned_verbs(svg)
    census_text = sidecar.read_text()
    for credit in CREDIT_ROWS:
        assert credit in census_text
    assert _REPLAY_WORDING in census_text
    _assert_no_banned_verbs(census_text)


# ---------------------------------------------------------------------------
# the site-key-first surgery diff
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_diff_joins_site_key_first_and_marks_travel(chain_capture) -> None:
    """Site-key joins are FACT rows; subject-side marks ride the rows."""

    _model, _x, log = chain_capture
    fork = _edited_fork(log)
    diff = surgery_diff(fork, log)
    matched = [row for row in diff.rows if row.kind == "matched"]
    assert matched and all(row.join == "site_key" and row.basis == "fact" for row in matched)
    edited = [row for row in matched if row.subject_label == "relu_1_2:1"]
    assert any("zero_ablate" in note for note in edited[0].notes)
    assert diff.counts()["only_in_subject"] == 0


@pytest.mark.smoke
def test_diff_positional_cohort_join_is_heuristic() -> None:
    """Equal cohorts under one shared site key pair positionally: HEURISTIC."""

    torch.manual_seed(0)
    model = _SharedBlock().eval()
    x = torch.randn(3, 4)
    subject = tl.trace(model, x, capture=_CAPTURE)
    reference = tl.trace(model, x, capture=_CAPTURE)
    fork = subject.fork()
    fork.do(tl.units("linear_1_1:1", [(0, 0)]).resolve(fork), tl.zero_ablate())
    diff = surgery_diff(fork, reference)
    joins = {row.subject_label: row for row in diff.rows if row.kind == "matched"}
    assert joins["linear_1_1:1"].join == "site_key_positional"
    assert joins["linear_1_1:1"].basis == "heuristic"
    assert joins["relu_1_2:1"].join == "site_key"
    assert joins["relu_1_2:1"].basis == "fact"


@pytest.mark.smoke
def test_diff_absence_is_disclosed_never_ghosted(chain_capture) -> None:
    """A one-side op is a full-strength disclosure, never execution removal."""

    _model, x, log = chain_capture
    torch.manual_seed(0)
    other = _ChainExtra().eval()
    subject = tl.trace(other, x)
    diff = surgery_diff(subject, log)
    only_rows = [row for row in diff.rows if row.kind == "only_in_subject"]
    assert len(only_rows) == 1
    assert only_rows[0].subject_label == "sigmoid_1_5:1"
    assert only_rows[0].notes == ("not recorded in the reference capture",)
    text = diff.to_text()
    _assert_no_banned_verbs(text)
    assert "ghost" not in text.lower()


@pytest.mark.smoke
def test_diff_refuses_self_and_empty_operands(chain_capture) -> None:
    """Self-diff and an op-free operand refuse surgery_diff_incomparable."""

    _model, _x, log = chain_capture
    with pytest.raises(InvalidArgumentError) as excinfo:
        surgery_diff(log, log)
    assert excinfo.value.fields["code"] == "surgery_diff_incomparable"

    class _Husk:
        """A trace-shaped husk recording no ops (the empty-operand guard)."""

        op_labels: tuple[str, ...] = ()
        ops: dict = {}
        state_history: tuple = ()
        intervention_audit: list = []

    with pytest.raises(InvalidArgumentError) as excinfo:
        surgery_diff(log, _Husk())  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "surgery_diff_incomparable"


@pytest.mark.heavy
def test_diff_draw_line_styles_and_census_sidecar(tmp_path) -> None:
    """The diff figure draws solid fact / dashed heuristic joins + census."""

    torch.manual_seed(0)
    model = _SharedBlock().eval()
    x = torch.randn(3, 4)
    subject = tl.trace(model, x, capture=_CAPTURE)
    reference = tl.trace(model, x, capture=_CAPTURE)
    fork = subject.fork()
    fork.do(tl.units("linear_1_1:1", [(0, 0)]).resolve(fork), tl.zero_ablate())
    diff = surgery_diff(fork, reference)
    rendered = diff.draw(str(tmp_path / "diff_fig"), vis_fileformat="svg", vis_save_only=True)
    figure_text = (tmp_path / "diff_fig.svg").read_text()
    assert rendered.endswith("diff_fig.svg")
    assert "stroke-dasharray" in figure_text  # the heuristic join renders dashed
    assert "surgery diff" in figure_text
    _assert_no_banned_verbs(figure_text)
    sidecar = tmp_path / "diff_fig.census.txt"
    assert sidecar.exists()
    census_text = sidecar.read_text()
    assert "site_key_positional" in census_text
    for credit in CREDIT_ROWS:
        assert credit in census_text
