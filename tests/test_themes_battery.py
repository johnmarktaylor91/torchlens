"""F12 pins: RenderIR answer keys and the naive-battery harness.

Keys are GENERATED (hand keys prohibited); packets are seed-deterministic
with the memo's fixed question-phase order; honesty classes are
zero-tolerance in scoring; threshold freezing follows the anchor-midpoint
procedure and refuses on overlapping bounds.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.visualization import lenses
from torchlens.visualization.lenses import audit


@pytest.fixture(scope="module")
def keyed() -> Any:
    """A speed-lens resolution + generated key on a small toy."""

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


def test_key_is_generated_from_machine_record(keyed: Any) -> None:
    """Extrema answers equal the argmax/argmin of the resolved member."""

    log, resolution, key = keyed
    values = {
        op.layer_label: float(getattr(op, resolution.source.member))
        for op in log.ops
        if getattr(op, resolution.source.member, None) is not None
    }
    by_id = {question.id: question for question in key.questions}
    assert by_id["extrema-max"].answer == max(values, key=values.__getitem__)
    assert by_id["extrema-min"].answer == min(values, key=values.__getitem__)
    assert by_id["path-1"].answer is True
    assert by_id["ordinal-1"].honesty_class == "ORDINAL-as-RATIO"
    assert key.to_dict()["lens"] == "speed"


@pytest.mark.smoke
def test_packets_are_seed_deterministic(keyed: Any) -> None:
    """Same seed, same order; filenames masked; phases honour the memo."""

    log, resolution, key = keyed
    first = audit.build_packet(key, ["a.png"], seed=11)
    second = audit.build_packet(key, ["a.png"], seed=11)
    assert [q["id"] for q in first.questions] == [q["id"] for q in second.questions]
    assert first.masked_image_names == ("image_00.png",)
    phases = [q["phase"] for q in first.questions]
    assert phases[0] == "free_response"
    assert phases[-1] == "transcription"
    assert first.judges == 3
    # A sentinel packet needs the sentinel-generated key (its known-bad
    # control must be able to FAIL -- D03-R6).
    sentinel_key = audit.generate_answer_key(
        log, member_name="toy", resolution=resolution, sentinel=True
    )
    assert audit.build_packet(sentinel_key, ["a.png"], seed=11, sentinel=True).judges == 5


@pytest.mark.smoke
def test_scoring_flags_honesty_hits_zero_tolerance(keyed: Any) -> None:
    """A wrong honesty-probe answer is a confirmed class hit."""

    _, _, key = keyed
    perfect = audit.score_responses(key, {q.id: q.answer for q in key.questions})
    assert perfect.accuracy == 1.0
    assert perfect.honesty_hits == ()
    wrong = audit.score_responses(key, {"ordinal-1": True})
    assert wrong.honesty_hits == ("ORDINAL-as-RATIO",)


def test_honesty_classes_are_the_nine() -> None:
    """The closed zero-tolerance vocabulary."""

    assert len(audit.HONESTY_CLASSES) == 9
    assert "CHANNEL-COLLISION" in audit.HONESTY_CLASSES
    assert "NOT-CHECKED-as-FINITE" in audit.HONESTY_CLASSES


@pytest.mark.smoke
def test_threshold_freezing_midpoint_rounds_up_to_five() -> None:
    """The anchor-midpoint procedure, rounded UP to five points."""

    good = [92.0, 94.0, 96.0, 95.0, 93.0, 94.0, 95.0, 96.0, 92.0, 93.0]
    bad = [40.0, 45.0, 42.0, 38.0, 44.0, 41.0, 43.0, 39.0, 42.0, 40.0]
    frozen = audit.freeze_threshold(good, bad)
    assert frozen % 5 == 0
    assert 45.0 < frozen < 95.0


def test_overlapping_bounds_invalidate_the_metric() -> None:
    """Overlap refuses typed: the metric, never the candidate, is at fault."""

    with pytest.raises(Exception) as excinfo:
        audit.freeze_threshold([60.0, 70.0, 50.0, 80.0], [55.0, 65.0, 45.0, 75.0])
    assert excinfo.value.fields["code"] == "battery_metric_invalid"


def test_insufficient_anchors_refuse() -> None:
    """Fewer than two judgements per side cannot freeze."""

    with pytest.raises(Exception) as excinfo:
        audit.freeze_threshold([90.0], [40.0, 41.0])
    assert excinfo.value.fields["code"] == "battery_anchor_insufficient"


@pytest.mark.smoke
def test_debug_key_guards_not_checked_as_finite() -> None:
    """A partial-coverage debug render generates the NOT-CHECKED probe."""

    member = next(m for m in audit.CORPUS if m.name == "nonfinite_chain")
    model, x = member.build()
    log = tl.trace(model, x, save=tl.func("relu"))
    try:
        resolution = lenses.resolve_lens(log, "debug")
        key = audit.generate_answer_key(log, member_name="nonfinite_chain", resolution=resolution)
        probes = [q for q in key.questions if q.honesty_class == "NOT-CHECKED-as-FINITE"]
        assert probes and probes[0].answer == "not checked"
        hit = audit.score_responses(key, {probes[0].id: "verified finite"})
        assert hit.honesty_hits == ("NOT-CHECKED-as-FINITE",)
    finally:
        log.cleanup()
