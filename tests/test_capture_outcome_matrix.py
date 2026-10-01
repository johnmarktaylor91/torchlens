"""P5 outcome-matrix cells and lockstep proofs.

The remaining design-matrix cells not covered by the P0-P3 suites:
raise_on_nan's honest raw-tensor-only scope, the no-frontier halt failure
class, cooked-halted gate arming (N4/N5 on ``Recording.to_trace()``
products), the OOM double-fault, the never-fired halt predicate, and the
source-lockstep proof that every gated entry actually consults the one
chokepoint.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import torch
from fixtures.capture_outcome_models import ThreeStageModel, halt_on_relu
from torch import nn

import torchlens as tl
from torchlens.capture.outcome import (
    CaptureOutcomeError,
    CapturePhase,
    CaptureStatus,
)

pytestmark = pytest.mark.smoke

TORCHLENS_DIR = pathlib.Path(tl.__file__).resolve().parent


class NaNProducingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        return torch.relu(y / torch.zeros_like(y))


# ---------------------------------------------------------------------------
# raise_on_nan's honest scope: RAW tensors only
# ---------------------------------------------------------------------------


def test_transform_introduced_nan_stays_complete() -> None:
    """A NaN INTRODUCED by activation_transform never trips the raw tripwire."""

    def poison(tensor: torch.Tensor) -> torch.Tensor:
        return tensor * float("nan")

    trace = tl.trace(
        ThreeStageModel(),
        torch.ones(1, 3),
        capture=tl.options.CaptureOptions(raise_on_nan=True),
        save=tl.options.SaveOptions(activation_transform=poison),
    )
    assert trace.outcome.status is CaptureStatus.COMPLETE


def test_raw_nan_aborts_even_when_transform_sanitizes() -> None:
    """A raw-tensor NaN aborts even if the transform would have removed it."""

    def sanitize(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(tensor)

    from torchlens.errors import CaptureError

    with pytest.raises(CaptureError) as exc_info:
        tl.trace(
            NaNProducingModel(),
            torch.ones(1, 3),
            capture=tl.options.CaptureOptions(raise_on_nan=True),
            save=tl.options.SaveOptions(activation_transform=sanitize),
        )
    outcome = tl.partial.from_failed_capture(exc_info.value).trace.outcome
    assert outcome.status is CaptureStatus.ABORTED_NONFINITE


# ---------------------------------------------------------------------------
# No-frontier halt: FAILED/FINALIZE, never a fabricated frontier
# ---------------------------------------------------------------------------


def test_no_frontier_halt_settles_failed_finalize(monkeypatch) -> None:
    """A halt with no recoverable tensor frontier fails typed at FINALIZE."""

    import torchlens.backends.torch.ops as torch_ops
    from torchlens import _state
    from torchlens.fastlog._halt import HaltSignal

    original = torch_ops.evaluate_halt_stop

    def _strip_frontier(trace, ctx, options, frontier_output=None):
        return original(trace, ctx, options, frontier_output=None)

    monkeypatch.setattr(torch_ops, "evaluate_halt_stop", _strip_frontier)

    captured: list = []

    def _halt_and_orphan(ctx: object) -> bool:
        if ctx.kind == "op" and ctx.func_name == "relu":
            active = _state._active_trace
            captured.append(active)
            active._raw_graph_ws.raw_layer_dict.clear()
            return True
        return False

    with pytest.raises(RuntimeError, match="could not identify a tensor frontier") as exc_info:
        tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=_halt_and_orphan)
    assert isinstance(exc_info.value.__cause__, HaltSignal)
    outcome = captured[0].outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.FINALIZE
    assert "halted finalization failed" in (outcome.settlement_note or "")


def test_never_fired_halt_predicate_stays_complete() -> None:
    """A halt predicate that never fires yields an ordinary COMPLETE capture."""

    trace = tl.trace(
        ThreeStageModel(),
        torch.ones(1, 3),
        halt=lambda ctx: ctx.kind == "op" and ctx.func_name == "definitely_absent",
    )
    assert trace.halted is False
    assert trace.outcome.status is CaptureStatus.COMPLETE


# ---------------------------------------------------------------------------
# Cooked-halted products hit N4/N5 like any halted trace
# ---------------------------------------------------------------------------


def test_cooked_halted_trace_refuses_runnable_save_and_live_replay(tmp_path) -> None:
    recording = tl.record(
        ThreeStageModel(),
        torch.ones(1, 3),
        save=lambda ctx: ctx.kind == "op",
        halt=halt_on_relu,
    )
    cooked = recording.to_trace()
    assert cooked.outcome.status is CaptureStatus.HALTED

    from torchlens.errors import RunnablePreflightError

    with pytest.raises(RunnablePreflightError):
        tl.save(cooked, tmp_path / "cooked_halted.tlspec", level="runnable", overwrite=True)
    with pytest.raises(CaptureOutcomeError) as exc_info:
        cooked.push()
    assert exc_info.value.fields["code"] == "N5"


# ---------------------------------------------------------------------------
# OOM double-fault: MemoryError still settles and propagates
# ---------------------------------------------------------------------------


def test_memory_error_in_postprocess_settles_failed(monkeypatch) -> None:
    import torchlens.postprocess as pp

    captured: list = []

    def _boom(trace: object, *args: object, **kwargs: object) -> None:
        captured.append(trace)
        raise MemoryError("planted OOM")

    monkeypatch.setattr(pp, "_add_output_layers", _boom)
    with pytest.raises(MemoryError):
        tl.trace(ThreeStageModel(), torch.ones(1, 3))
    outcome = captured[0].outcome
    assert outcome is not None
    assert outcome.status is CaptureStatus.FAILED
    assert outcome.phase is CapturePhase.POSTPROCESS
    assert outcome.error_type == "MemoryError"


# ---------------------------------------------------------------------------
# Chokepoint lockstep: every gated entry consults the ONE authority
# ---------------------------------------------------------------------------


def test_gated_entries_consult_the_chokepoint() -> None:
    """Source lockstep: the N-gate entries call require_capture_capability.

    A refactor that drops a gate call from an entry (or adds a competing
    inline status check) fails here; the chokepoint is the only sanctioned
    consumer of the capability table.
    """

    validation_source = (TORCHLENS_DIR / "data_classes" / "_trace_validation.py").read_text(
        encoding="utf-8"
    )
    gate_calls = re.findall(r'require_capture_capability\(self, "([a-z_]+)"\)', validation_source)
    # save_new_outs, push, push_from, run (entry pre-gate + loaded-sparse +
    # analysis-load pre-gate + unified-live + legacy-live), log_backward,
    # recording_backward, validate_forward_pass, check_metadata_invariants.
    assert sorted(gate_calls) == sorted(
        [
            "live_replay",  # save_new_outs
            "live_replay",  # push
            "live_replay",  # push_from
            "live_replay",  # run(): entry pre-gate (R06: FAILED/ABORTED/UNKNOWN
            # refuse with their N-code before the husk/_runnable read can mask it)
            "loaded_sparse_run",  # run(): loaded sparse provider
            "live_replay",  # run(): HALTED analysis-load pre-gate (N5 before the
            # generic analysis refusal's dead-end remedy)
            "live_replay",  # run(): unified live provider
            "live_replay",  # run(): legacy rerun surface
            "backward",  # log_backward
            "backward",  # recording_backward
            "validation_entry",  # validate_forward_pass
            "validation_entry",  # check_metadata_invariants
        ]
    )
    bundle_source = (TORCHLENS_DIR / "_io" / "bundle.py").read_text(encoding="utf-8")
    assert 'require_capture_capability(trace, "save_analysis")' in bundle_source
    assert 'require_capture_capability(trace, "save_runnable")' in bundle_source
    # No inline second authority: nothing outside the authority module reads
    # the capability table directly.
    for path in TORCHLENS_DIR.rglob("*.py"):
        if path.name == "outcome.py" and path.parent.name == "capture":
            continue
        text = path.read_text(encoding="utf-8")
        assert "CAPTURE_OUTCOME_CAPABILITIES[" not in text, str(path)
