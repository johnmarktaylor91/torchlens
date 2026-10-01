"""FIXD03-F12 regressions: the four D03-returned defects in the F12 lens
territory (browser-packet reserved prefix; evidence in the D03 themes packet).

- D03-R1: the transformer subject probe on a trace with a REUSED module
  refuses ``lens_attention_subject_absent`` (factual absence of SUBJECT),
  never the accessor's ``module_call_ambiguous``.
- D03-R2: Stage-0 label boxes are sized from VISIBLE text -- raw HTML-table
  markup length manufactured phantom geometry penetrations.
- D03-R6: sentinel (known-bad) packets carry a scoreable channel-semantics
  probe, so the control can FAIL; a probe-less sentinel refuses typed.
- D03-R7: the scorer accepts tied extrema, alternative label spellings,
  order-insensitive label sets, and elaborated boolean answers.
"""

from __future__ import annotations

import shutil
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization import lenses
from torchlens.visualization.lenses import audit
from torchlens.visualization.lenses.audit import stage0
from torchlens.visualization.lenses.audit.answer_key import AnswerKey, KeyQuestion

pytestmark = pytest.mark.smoke  # measured <0.5s per test (W051-GATE, AUD-CODE 0.1)

DOT_AVAILABLE = shutil.which("dot") is not None


@pytest.fixture(scope="module")
def reused_module_log() -> Any:
    """An attention-free trace whose one submodule is called twice."""

    class Reuse(nn.Module):
        """Tied linear applied twice: every-corpus reused-module shape."""

        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 4)

        def forward(self, x: Any) -> Any:
            return self.shared(torch.relu(self.shared(x)))

    log = tl.trace(Reuse(), torch.randn(1, 4))
    yield log
    log.cleanup()


@pytest.fixture(scope="module")
def keyed_speed() -> Any:
    """A speed-lens resolution + generated key on the two-linear toy.

    The toy's input/output ops both record 0.0 duration -- the exact
    deterministic tied-minimum the D03 judge round hit on resnet18.
    """

    class Toy(nn.Module):
        """Two-linear toy."""

        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.fc2 = nn.Linear(8, 4)

        def forward(self, x: Any) -> Any:
            return self.fc2(torch.relu(self.fc1(x)))

    log = tl.trace(Toy(), torch.randn(2, 8))
    resolution = lenses.resolve_lens(log, "speed")
    key = audit.generate_answer_key(log, member_name="toy", resolution=resolution)
    yield log, resolution, key
    log.cleanup()


# ---------------------------------------------------------------------------
# D03-R1: refusal taxonomy on reused-module traces.
# ---------------------------------------------------------------------------


def test_transformer_refuses_subject_absent_on_reused_module_trace(
    reused_module_log: Any,
) -> None:
    """Factual absence of SUBJECT, never the per-module accessor refusal."""

    with pytest.raises(Exception) as excinfo:
        lenses.resolve_lens(reused_module_log, "transformer")
    assert excinfo.value.fields["code"] == "lens_attention_subject_absent"
    assert "overview" in str(excinfo.value)


def test_transformer_resolves_on_reused_module_attention_trace() -> None:
    """A reused module never hides REAL attention structure from the probe."""

    class TinyAttention(nn.Module):
        """q/k/v projections into scaled_dot_product_attention."""

        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.k_proj = nn.Linear(8, 8)
            self.v_proj = nn.Linear(8, 8)

        def forward(self, x: Any) -> Any:
            query, key, value = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            attended = torch.nn.functional.scaled_dot_product_attention(
                query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)
            )
            return attended.squeeze(1)

    class ReuseAttention(nn.Module):
        """Attention downstream of a twice-called shared linear."""

        def __init__(self) -> None:
            super().__init__()
            self.attn = TinyAttention()
            self.shared = nn.Linear(8, 8)

        def forward(self, x: Any) -> Any:
            return self.attn(self.shared(self.shared(x)))

    log = tl.trace(ReuseAttention(), torch.randn(2, 8))
    try:
        assert lenses.resolve_lens(log, "transformer").draw_kwargs
    finally:
        log.cleanup()


# ---------------------------------------------------------------------------
# D03-R2: label boxes measure visible text, never raw markup.
# ---------------------------------------------------------------------------

_HTML_TABLE_LABEL = (
    '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="1">'
    '<TR><TD ALIGN="CENTER">In 2-6</TD></TR></TABLE>'
)


def test_label_boxes_measure_visible_text_not_markup() -> None:
    """An HTML-table label and its visible text get the SAME box width."""

    layout = {
        "edges": [
            {"label": _HTML_TABLE_LABEL, "lp": "100,100"},
            {"label": "In 2-6", "lp": "300,300"},
        ]
    }
    boxes = stage0._label_boxes(layout)
    assert len(boxes) == 2
    (text_a, box_a), (text_b, box_b) = boxes
    assert text_a == "In 2-6"
    width_from_markup = box_a[2] - box_a[0]
    width_from_text = box_b[2] - box_b[0]
    assert width_from_markup == pytest.approx(width_from_text)
    # ~6 visible chars at the 8pt default, never 100s of pt of markup.
    assert width_from_markup < 40.0


def test_label_box_with_no_visible_text_has_no_box() -> None:
    """Pure markup with no text content contributes no phantom box."""

    layout = {"edges": [{"label": "<TABLE><TR><TD></TD></TR></TABLE>", "lp": "10,10"}]}
    assert stage0._label_boxes(layout) == []


@pytest.mark.skipif(not DOT_AVAILABLE, reason="graphviz dot binary unavailable")
def test_html_table_edge_label_does_not_trip_geometry() -> None:
    """The shipped In/Out multiplicity family passes the geometry gate."""

    dot_source = f"digraph {{ a -> b [label=<{_HTML_TABLE_LABEL}>]; }}"
    findings = stage0.geometry_findings(dot_source)
    label_finding = next(
        finding for finding in findings if finding.check == "geometry_label_penetration"
    )
    assert label_finding.passed, label_finding.detail


# ---------------------------------------------------------------------------
# D03-R6: the sentinel channel-semantics probe.
# ---------------------------------------------------------------------------


def test_sentinel_key_mints_channel_semantics_probe(keyed_speed: Any) -> None:
    """The known-bad control can FAIL: a legend-reading judge misses the
    probe and confirms the ABSENCE class; the no-meaning option is single."""

    log, resolution, _ = keyed_speed
    sentinel_key = audit.generate_answer_key(
        log, member_name="toy", resolution=resolution, sentinel=True
    )
    probes = [q for q in sentinel_key.questions if q.kind == "channel_semantics"]
    assert len(probes) == 1
    probe = probes[0]
    assert probe.honesty_class == "ABSENCE"
    packet = audit.build_packet(sentinel_key, ["a.png"], seed=3, sentinel=True)
    assert packet.judges == 5
    entry = next(q for q in packet.questions if q.get("kind") == "channel_semantics")
    assert entry["phase"] == "forced_choice"
    options_lower = [option.strip().lower() for option in entry["options"]]
    assert options_lower.count("no meaning") == 1
    misled = audit.score_responses(sentinel_key, {probe.id: probe.distractors[0]})
    assert "ABSENCE" in misled.honesty_hits
    honest = audit.score_responses(sentinel_key, {probe.id: "no meaning"})
    assert honest.honesty_hits == ()
    assert honest.accuracy == 1.0


def test_sentinel_packet_without_probe_refuses_typed(keyed_speed: Any) -> None:
    """A sentinel packet whose control cannot fail is refused, teaching
    the sentinel key spelling."""

    _, _, key = keyed_speed
    with pytest.raises(Exception) as excinfo:
        audit.build_packet(key, ["a.png"], seed=3, sentinel=True)
    assert excinfo.value.fields["code"] == "battery_sentinel_probe_missing"
    assert "sentinel=True" in str(excinfo.value)


def test_non_sentinel_key_carries_no_channel_semantics_probe(keyed_speed: Any) -> None:
    """The probe is sentinel-only in v1; ordinary keys are unchanged."""

    _, _, key = keyed_speed
    assert not any(q.kind == "channel_semantics" for q in key.questions)


# ---------------------------------------------------------------------------
# D03-R7: scorer/key format defects fire only on WRONG readings.
# ---------------------------------------------------------------------------


def test_tied_extrema_accept_any_tied_label(keyed_speed: Any) -> None:
    """The generated key names every tied extremum; any one scores correct."""

    log, resolution, key = keyed_speed
    member = resolution.source.member
    values = {
        op.layer_label: float(getattr(op, member))
        for op in log.ops
        if getattr(op, member, None) is not None
    }
    minimum = min(values.values())
    tied = tuple(sorted(label for label, value in values.items() if value == minimum))
    question = next(q for q in key.questions if q.id == "extrema-min")
    assert question.accepted == tied
    assert len(tied) >= 2, "fixture must exercise a real tie (input/output at 0.0)"
    for response in tied:
        assert audit.score_responses(key, {"extrema-min": response}).accuracy == 1.0
    # Naming the complete tie as a collection also scores.
    assert audit.score_responses(key, {"extrema-min": list(tied)}).accuracy == 1.0
    # A non-tied label is still wrong.
    wrong = next(label for label in values if label not in tied)
    assert audit.score_responses(key, {"extrema-min": wrong}).accuracy == 0.0
    # Distractors on the max probe never name a tied-maximum label.
    max_question = next(q for q in key.questions if q.id == "extrema-max")
    assert not set(max_question.distractors) & set(max_question.accepted)


def test_status_key_dedupes_alt_spellings_and_matches_either() -> None:
    """One op appears under ONE spelling; judges may answer either."""

    class Blowup(nn.Module):
        """Chain whose middle op mints a NaN deterministically."""

        def forward(self, x: Any) -> Any:
            scaled = x * 0.5
            poisoned = scaled / torch.zeros_like(scaled)  # +/-Inf
            return poisoned - poisoned  # NaN

    log = tl.trace(Blowup(), torch.ones(2, 4))
    try:
        resolution = lenses.resolve_lens(log, "debug")
        key = audit.generate_answer_key(log, member_name="blowup", resolution=resolution)
        question = next(q for q in key.questions if q.id == "status-1")
        answer = list(question.answer)
        assert answer, "fixture must produce at least one NaN-carrying op"
        # No label rides beside its own ':N'-qualified alias.
        for label in answer:
            stem, sep, tail = label.rpartition(":")
            if sep and tail.isdigit():
                assert stem not in answer
        # The bare and pass-qualified spellings both score.
        assert audit.score_responses(key, {"status-1": answer}).accuracy == 1.0
        qualified = [f"{label}:1" if ":" not in label else label for label in answer]
        assert audit.score_responses(key, {"status-1": qualified}).accuracy == 1.0
        # Order-insensitive: a reversed list is the same set.
        assert audit.score_responses(key, {"status-1": list(reversed(answer))}).accuracy == 1.0
    finally:
        log.cleanup()


def test_elaborated_boolean_answer_scores_correct() -> None:
    """'No (false). ...' is a correct refusal, never an ORDINAL-as-RATIO hit."""

    question = KeyQuestion(
        id="ordinal-1",
        kind="ranking_pair",
        question="Is A necessarily MANY TIMES larger than B?",
        answer=False,
        honesty_class="ORDINAL-as-RATIO",
    )
    key = AnswerKey(member="m", lens="speed", questions=(question,))
    elaborated = (
        "No (false). The legend describes a rank mapping (ordinal, not "
        "ratio), which cannot by itself support a 'many times larger' claim."
    )
    score = audit.score_responses(key, {"ordinal-1": elaborated})
    assert score.accuracy == 1.0
    assert score.honesty_hits == ()
    hit = audit.score_responses(key, {"ordinal-1": "Yes, many times larger."})
    assert hit.accuracy == 0.0
    assert hit.honesty_hits == ("ORDINAL-as-RATIO",)
    # A response with no boolean token never matches.
    assert audit.score_responses(key, {"ordinal-1": "the scale is a rank"}).accuracy == 0.0
