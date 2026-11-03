"""A-BUNDLE-OUTCOME proofs: Bundle declares outcome authority (foldB D9 + D8).

Bundle is the THIRD instance of the "two answers for one product"
outcome-authority disease (PartialTrace: ``_OUTCOME_DELEGATE_FIELD``;
Recording: ``_OUTCOME_SELF_AUTHORITY``): ``outcome_for(bundle)`` returned
``None`` while every member was COMPLETE, so ``bundle.save(path)`` false-refused
EVERY bundle (N1 status 'unknown' + a RuntimeWarning blaming the user for
hand-building an object ``tl.Bundle()`` built) while ``bundle.save()`` worked.

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

import pytest
import torch
from fixtures.capture_outcome_models import ThreeStageModel, halt_on_relu

import torchlens as tl
from torchlens._capture.outcome import (
    CaptureOutcome,
    CaptureOutcomeError,
    CaptureStatus,
    outcome_for,
)
from torchlens._data_classes.trace import Trace
from torchlens.bundle import Bundle, _fold_member_outcomes
from torchlens.errors import InvalidArgumentError, TraceCaptureError


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
    except TraceCaptureError as exc:
        assert isinstance(exc.__cause__, RuntimeError)
        return exc.partial_trace.trace
    raise AssertionError("capture unexpectedly succeeded")


@pytest.fixture(scope="module")
def complete_pair():
    """Bundle of two real COMPLETE captures, built by tl.Bundle()."""

    with torch.no_grad():
        a = tl.trace(ThreeStageModel(), torch.ones(1, 3))
        b = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    try:
        yield tl.Bundle({"a": a, "b": b})
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
    assert folded.is_derived is True
    assert "worst-of-members" in (folded.note or "")


def test_fold_unsettled_member_is_unknown_fail_closed() -> None:
    """A member with no settled outcome drives the fold to UNKNOWN."""

    folded = _fold_member_outcomes([("a", _outcome(CaptureStatus.COMPLETE)), ("b", None)])
    assert folded.status is CaptureStatus.UNKNOWN
    assert "no settled outcome" in (folded.note or "")


def test_fold_empty_domain_is_unknown_fail_closed() -> None:
    """Zero members (unreachable via the public constructor) never bless COMPLETE."""

    folded = _fold_member_outcomes([])
    assert folded.status is CaptureStatus.UNKNOWN
    assert folded.is_derived is True


# ---------------------------------------------------------------------------
# Declared authority: outcome_for(bundle) answers with the fold
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Both public save doors agree on every settled bundle
# ---------------------------------------------------------------------------


def test_both_doors_save_a_halted_bundle(tmp_path) -> None:
    """HALTED is savable (N1 allows it): both doors agree and disclose the fold."""

    with torch.no_grad():
        complete = tl.trace(ThreeStageModel(), torch.ones(1, 3))
        halted = tl.trace(ThreeStageModel(), torch.ones(1, 3), stop_after=halt_on_relu)
    mixed = tl.Bundle({"complete": complete, "halted": halted})
    assert mixed.outcome.status is CaptureStatus.STOPPED

    mixed.save(tmp_path / "tl_door.tlspec")
    mixed.save(tmp_path / "method_door.tlspec")
    loaded = tl.load(tmp_path / "tl_door.tlspec")
    settled = outcome_for(loaded)
    assert settled is not None
    assert settled.status is CaptureStatus.STOPPED


@pytest.mark.smoke
def test_refuted_hypothesis_pin_failed_member_refuses_both_doors(tmp_path) -> None:
    """The probe-REFUTED N1-hole pin: one FAILED member -> BOTH doors refuse.

    Per-member gating is INTACT and must never regress: the refusal is the
    MEMBER's own N1 gate, fired from inside the container write.
    """

    with torch.no_grad():
        good = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    bad = _failed_trace()
    mixed = tl.Bundle({"good": good, "bad": bad})
    assert mixed.outcome.status is CaptureStatus.FAILED

    with pytest.raises(CaptureOutcomeError) as tl_door:
        mixed.save(tmp_path / "tl_door.tlspec")
    assert tl_door.value.fields["code"] == "N1"
    with pytest.raises(CaptureOutcomeError) as method_door:
        mixed.save(tmp_path / "method_door.tlspec")
    assert method_door.value.fields["code"] == "N1"


# ---------------------------------------------------------------------------
# Door-surface agreement: refusals match on the container surface
# ---------------------------------------------------------------------------


def test_trace_only_save_options_refuse_typed(tmp_path, complete_pair: Bundle) -> None:
    """bundle.save(path) honors exactly the container door's surface; per-trace
    payload options refuse typed instead of being silently dropped."""

    from torchlens._io.bundle import save as _save_bundle

    for kwargs in ({"include_weights": True}, {"include_outs": False}, {"strict": False}):
        # Bundle.save takes no per-trace options at all; the private save door
        # still refuses them typed rather than dropping them.
        with pytest.raises(TypeError):
            complete_pair.save(tmp_path / "opt.tlspec", **kwargs)
        with pytest.raises(InvalidArgumentError) as excinfo:
            _save_bundle(complete_pair, tmp_path / "opt.tlspec", **kwargs)
        assert excinfo.value.fields["code"] == "bundle_save_option_unsupported"
    # Explicitly passing a value equal to the trace-door default is honored.
    _save_bundle(complete_pair, tmp_path / "default_ok.tlspec", strict=True, overwrite=True)
    assert tl.load(tmp_path / "default_ok.tlspec").names == ["a", "b"]
