"""A-BUNDLE-OUTCOME proofs: Bundle declares outcome authority (foldB D9 + D8).

Bundle is the THIRD instance of the "two answers for one product"
outcome-authority disease (PartialTrace: ``_OUTCOME_DELEGATE_FIELD``;
Recording: ``_OUTCOME_SELF_AUTHORITY``): ``outcome_for(bundle)`` returned
``None`` while every member was COMPLETE, so ``tl.save(bundle)`` false-refused
EVERY bundle (N1 status 'unknown' + a RuntimeWarning blaming the user for
hand-building an object ``tl.bundle()`` built) while ``bundle.save()`` worked.

This file pins the retirement of that disagreement:

- Bundle declares ``_OUTCOME_SELF_AUTHORITY`` over a worst-of-members fold
  (COMPLETE never blessed above the weakest member; derived disclosure,
  never a settlement).
- Both public save doors agree on every settled bundle; the false
  hand-built-object warning is gone.
- Per-member gating stays INTACT (the probe-REFUTED-hypothesis pin: with one
  member FAILED, BOTH doors refuse via the MEMBER's N1) and the container
  fold never gates a read on a member it does not describe (D8).
"""

from __future__ import annotations

import warnings

import pytest
import torch
from fixtures.capture_outcome_models import ThreeStageModel, halt_on_relu

import torchlens as tl
from torchlens.bundle import Bundle, _fold_member_outcomes
from torchlens.capture.outcome import (
    CaptureOutcome,
    CaptureOutcomeError,
    CaptureStatus,
    outcome_for,
    require_capture_capability,
)
from torchlens.data_classes.trace import Trace
from torchlens.errors import InvalidArgumentError

pytestmark = pytest.mark.smoke


# Least -> most severe; the fold picks the maximum over members.
SEVERITY_ORDER = [
    CaptureStatus.COMPLETE,
    CaptureStatus.UNATTESTED,
    CaptureStatus.HALTED,
    CaptureStatus.ABORTED_NONFINITE,
    CaptureStatus.FAILED,
    CaptureStatus.UNKNOWN,
]


def _outcome(status: CaptureStatus) -> CaptureOutcome:
    return CaptureOutcome(status=status)


class ExplodingModel(torch.nn.Module):
    """Tiny model whose forward raises after one real op."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.linear(x)
        raise RuntimeError("user forward boom")


def _failed_trace() -> Trace:
    """Return a real FAILED-settled trace via the partial-recovery door."""

    try:
        with torch.no_grad():
            tl.trace(ExplodingModel(), torch.ones(1, 3))
    except RuntimeError as exc:
        return tl.partial.from_failed_capture(exc).trace
    raise AssertionError("capture unexpectedly succeeded")


@pytest.fixture(scope="module")
def complete_pair():
    """Bundle of two real COMPLETE captures, built by tl.bundle()."""

    with torch.no_grad():
        a = tl.trace(ThreeStageModel(), torch.ones(1, 3))
        b = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    try:
        yield tl.bundle({"a": a, "b": b})
    finally:
        a.cleanup()
        b.cleanup()


# ---------------------------------------------------------------------------
# The worst-of-members fold (unit legs over the declared severity order)
# ---------------------------------------------------------------------------


def test_fold_all_complete_is_complete() -> None:
    folded = _fold_member_outcomes(
        [("a", _outcome(CaptureStatus.COMPLETE)), ("b", _outcome(CaptureStatus.COMPLETE))]
    )
    assert folded.status is CaptureStatus.COMPLETE
    assert folded.derived is True
    assert "worst-of-members" in (folded.settlement_note or "")


@pytest.mark.parametrize("status", [s for s in SEVERITY_ORDER if s is not CaptureStatus.COMPLETE])
def test_fold_complete_never_blessed_above_weakest_member(status: CaptureStatus) -> None:
    """Any non-COMPLETE member forces the fold off COMPLETE."""

    folded = _fold_member_outcomes(
        [("good", _outcome(CaptureStatus.COMPLETE)), ("weak", _outcome(status))]
    )
    assert folded.status is status
    assert folded.status is not CaptureStatus.COMPLETE


@pytest.mark.parametrize(
    ("weaker", "stronger"),
    [(SEVERITY_ORDER[i], SEVERITY_ORDER[i + 1]) for i in range(len(SEVERITY_ORDER) - 1)],
)
def test_fold_total_severity_order(weaker: CaptureStatus, stronger: CaptureStatus) -> None:
    """Each adjacent severity pair folds to the more severe status."""

    folded = _fold_member_outcomes([("w", _outcome(weaker)), ("s", _outcome(stronger))])
    assert folded.status is stronger


def test_fold_unsettled_member_is_unknown_fail_closed() -> None:
    """A member with no settled outcome drives the fold to UNKNOWN."""

    folded = _fold_member_outcomes([("a", _outcome(CaptureStatus.COMPLETE)), ("b", None)])
    assert folded.status is CaptureStatus.UNKNOWN
    assert "no settled outcome" in (folded.settlement_note or "")


def test_fold_empty_domain_is_unknown_fail_closed() -> None:
    """Zero members (unreachable via the public constructor) never bless COMPLETE."""

    folded = _fold_member_outcomes([])
    assert folded.status is CaptureStatus.UNKNOWN
    assert folded.derived is True


def test_fold_note_names_driving_member() -> None:
    folded = _fold_member_outcomes(
        [
            ("good", _outcome(CaptureStatus.COMPLETE)),
            ("bad", _outcome(CaptureStatus.FAILED)),
        ]
    )
    note = folded.settlement_note or ""
    assert "'bad'" in note
    assert "failed" in note


# ---------------------------------------------------------------------------
# Declared authority: outcome_for(bundle) answers with the fold
# ---------------------------------------------------------------------------


def test_bundle_declares_self_authority() -> None:
    assert Bundle._OUTCOME_SELF_AUTHORITY is True


def test_outcome_for_bundle_returns_fold_not_none(complete_pair: Bundle) -> None:
    """The retired disagreement: outcome_for(bundle) was None on COMPLETE members."""

    settled = outcome_for(complete_pair)
    assert settled is not None
    assert settled.status is CaptureStatus.COMPLETE
    assert settled.derived is True
    assert complete_pair.outcome.status is CaptureStatus.COMPLETE


def test_capability_gate_on_bundle_no_false_hand_built_warning(complete_pair: Bundle) -> None:
    """The container gate reads the fold: no UNKNOWN, no hand-built-object warning."""

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        settled = require_capture_capability(complete_pair, "save_analysis")
    assert settled.status is CaptureStatus.COMPLETE


# ---------------------------------------------------------------------------
# Both public save doors agree on every settled bundle
# ---------------------------------------------------------------------------


def test_both_doors_save_a_complete_bundle(tmp_path, complete_pair: Bundle) -> None:
    """tl.save(bundle) stops false-refusing; both doors round-trip identically."""

    door_tl = tmp_path / "door_tl.tlspec"
    door_method = tmp_path / "door_method.tlspec"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.save(complete_pair, door_tl)
    hand_built = [w for w in caught if "no settled capture outcome" in str(w.message)]
    assert not hand_built, "the false hand-built-object warning must be gone"
    complete_pair.save(door_method)

    loaded_tl = tl.load(door_tl)
    loaded_method = tl.load(door_method)
    assert loaded_tl.names == loaded_method.names == ["a", "b"]
    for loaded in (loaded_tl, loaded_method):
        settled = outcome_for(loaded)
        assert settled is not None
        assert settled.status is CaptureStatus.COMPLETE


def test_both_doors_save_a_halted_bundle(tmp_path) -> None:
    """HALTED is savable (N1 allows it): both doors agree and disclose the fold."""

    with torch.no_grad():
        complete = tl.trace(ThreeStageModel(), torch.ones(1, 3))
        halted = tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=halt_on_relu)
    mixed = tl.bundle({"complete": complete, "halted": halted})
    assert mixed.outcome.status is CaptureStatus.HALTED

    tl.save(mixed, tmp_path / "tl_door.tlspec")
    mixed.save(tmp_path / "method_door.tlspec")
    loaded = tl.load(tmp_path / "tl_door.tlspec")
    settled = outcome_for(loaded)
    assert settled is not None
    assert settled.status is CaptureStatus.HALTED


def test_refuted_hypothesis_pin_failed_member_refuses_both_doors(tmp_path) -> None:
    """The probe-REFUTED N1-hole pin: one FAILED member -> BOTH doors refuse.

    Per-member gating is INTACT and must never regress: the refusal is the
    MEMBER's own N1 gate, fired from inside the container write.
    """

    with torch.no_grad():
        good = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    bad = _failed_trace()
    mixed = tl.bundle({"good": good, "bad": bad})
    assert mixed.outcome.status is CaptureStatus.FAILED

    with pytest.raises(CaptureOutcomeError) as tl_door:
        tl.save(mixed, tmp_path / "tl_door.tlspec")
    assert tl_door.value.fields["code"] == "N1"
    with pytest.raises(CaptureOutcomeError) as method_door:
        mixed.save(tmp_path / "method_door.tlspec")
    assert method_door.value.fields["code"] == "N1"


def test_container_fold_never_gates_a_member_read(tmp_path) -> None:
    """D8 corollary: the container-level fold is a disclosure beside results;
    the capability gate applies at the MEMBER whose facts a read cites."""

    with torch.no_grad():
        good = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    bad = _failed_trace()
    mixed = tl.bundle({"good": good, "bad": bad})

    # Container fold is FAILED, but the COMPLETE member's facts stay readable
    # and savable through its own per-member gate.
    assert mixed.outcome.status is CaptureStatus.FAILED
    good_member = mixed["good"]
    assert require_capture_capability(good_member, "validation_entry").status is (
        CaptureStatus.COMPLETE
    )
    tl.save(good_member, tmp_path / "good_member.tlspec")
    # The FAILED member's own gate keeps refusing (per-member, not container).
    with pytest.raises(CaptureOutcomeError):
        require_capture_capability(mixed["bad"], "validation_entry")


# ---------------------------------------------------------------------------
# Door-surface agreement: refusals match on the container surface
# ---------------------------------------------------------------------------


def test_runnable_level_refuses_identically_at_both_doors(tmp_path, complete_pair: Bundle) -> None:
    with pytest.raises(InvalidArgumentError) as tl_door:
        tl.save(complete_pair, tmp_path / "r1.tlspec", level="runnable")
    assert tl_door.value.fields["code"] == "artifact_save_level_unsupported"
    with pytest.raises(InvalidArgumentError) as method_door:
        complete_pair.save(tmp_path / "r2.tlspec", level="runnable")
    assert method_door.value.fields["code"] == "artifact_save_level_unsupported"


def test_trace_only_save_options_refuse_typed(tmp_path, complete_pair: Bundle) -> None:
    """tl.save(bundle) honors exactly the container door's surface; per-trace
    payload options refuse typed instead of being silently dropped."""

    for kwargs in ({"include_weights": True}, {"include_outs": False}, {"strict": False}):
        with pytest.raises(InvalidArgumentError) as excinfo:
            tl.save(complete_pair, tmp_path / "opt.tlspec", **kwargs)
        assert excinfo.value.fields["code"] == "bundle_save_option_unsupported"
    # Explicitly passing a value equal to the trace-door default is honored.
    tl.save(complete_pair, tmp_path / "default_ok.tlspec", strict=True, overwrite=True)
    assert tl.load(tmp_path / "default_ok.tlspec").names == ["a", "b"]
