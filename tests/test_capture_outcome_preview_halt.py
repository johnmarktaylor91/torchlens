"""B1-01: a halted preview-backend capture never stamps COMPLETE.

Every preview backend (tf / mlx / tinygrad / paddle / jax) settles through the
ONE ``stamp_backend_finalized`` chokepoint as the last act of its capture
entry. Paddle ships ``halt=`` and MLX carries ``_mlx_halt_selector``; both
catch their ``HaltSignal``, write the structural halt fields, and fall through
the same tail to that stamp. Before the fix the stamp was unconditionally
COMPLETE, so a halted preview product carried a wrongly-blessed settled
status: on save+load the coherence matrix saw ``complete`` against structural
``halted=True``, degraded to UNKNOWN, and the artifact then refused N1 re-save
and N2 validation entry.

The framework-free arms below exercise the real chokepoint on a real halted
Trace (the fix site is backend-neutral by construction). The
``importorskip`` arms prove it end-to-end where the preview frameworks exist,
and the source-lockstep arms keep the fall-through placement from silently
regressing in a venv that cannot run them.
"""

from __future__ import annotations

import pathlib
import re

import pytest
import torch
from fixtures.capture_outcome_models import ThreeStageModel, halt_on_relu

import torchlens as tl
from torchlens.capture.outcome import CaptureStatus, outcome_for, stamp_backend_finalized

TORCHLENS_DIR = pathlib.Path(tl.__file__).resolve().parent


def _halted_trace() -> tl.Trace:
    """Return a real halted trace carrying the structural halt fields."""

    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=halt_on_relu)
    assert trace.halted is True
    return trace


# ---------------------------------------------------------------------------
# The chokepoint itself (framework-free runtime proof)
# ---------------------------------------------------------------------------


def test_preview_stamp_settles_halted_when_trace_is_halted() -> None:
    """The preview stamp mirrors settle_halted on a halted trace."""

    trace = _halted_trace()
    outcome = stamp_backend_finalized(trace)
    assert outcome.status is CaptureStatus.HALTED
    # An attested stamp, not a structural derivation.
    assert outcome.derived is False
    assert outcome.partial is True
    assert outcome.reason == trace.halt_reason
    assert outcome.boundary_label == trace.halt_frontier
    assert outcome.frontier_labels == tuple(trace.output_layers)
    assert outcome.n_ops_committed is not None
    # The stamp is the trace's settled outcome, in both homes.
    assert outcome_for(trace) is outcome
    assert trace.outcome is outcome


def test_preview_stamp_still_settles_complete_when_not_halted() -> None:
    """Backends with no halt path keep the byte-identical COMPLETE arm."""

    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    assert bool(getattr(trace, "halted", False)) is False
    outcome = stamp_backend_finalized(trace)
    assert outcome.status is CaptureStatus.COMPLETE
    assert outcome.derived is False
    assert outcome.partial is False
    assert outcome.reason is None
    assert outcome.frontier_labels is None


def test_preview_halted_stamp_is_coherent_and_survives_save_load(tmp_path) -> None:
    """The degradation chain is closed: no UNKNOWN, no N1/N2 refusal.

    This is the payoff. A COMPLETE stamp on a halted product is incoherent
    against the structural fields, so load degrades it to UNKNOWN and the
    artifact loses re-save (N1) and validation entry (N2). A HALTED stamp
    round-trips as HALTED.
    """

    trace = _halted_trace()
    stamp_backend_finalized(trace)
    path = tmp_path / "preview_halted.tlspec"
    tl.save(trace, str(path))
    loaded = tl.load(str(path))
    assert loaded.halted is True
    assert loaded.outcome is not None
    assert loaded.outcome.status is CaptureStatus.HALTED
    # N1: a non-degraded halted artifact re-saves.
    tl.save(loaded, str(tmp_path / "preview_halted_again.tlspec"))


# ---------------------------------------------------------------------------
# Source lockstep: the halt arms fall through to the ONE stamp
# ---------------------------------------------------------------------------

_HALT_CAPABLE_PREVIEW_BACKENDS = ("mlx", "paddle")


@pytest.mark.parametrize("backend", _HALT_CAPABLE_PREVIEW_BACKENDS)
def test_halt_capable_preview_backend_falls_through_to_the_single_stamp(backend: str) -> None:
    """The halt arm sets the structural fields BEFORE the one stamp.

    The fix is chokepoint-side, so what must not regress on the backend side
    is exactly this: (1) the halt arm writes ``trace.halted`` (the fact the
    stamp reads), and (2) there is exactly ONE stamp call, reached by
    fall-through, with no separate early COMPLETE stamp on the halt arm.
    """

    source = (TORCHLENS_DIR / "backends" / backend / "backend.py").read_text(encoding="utf-8")
    assert "except HaltSignal" in source, backend
    assert re.search(r"trace\.halted\s*=\s*True", source), backend
    stamp_lines = [
        index
        for index, line in enumerate(source.splitlines())
        if "stamp_backend_finalized(trace)" in line
    ]
    assert len(stamp_lines) == 1, f"{backend}: expected one stamp call, found {len(stamp_lines)}"
    halt_line = next(
        index
        for index, line in enumerate(source.splitlines())
        if re.search(r"trace\.halted\s*=\s*True", line)
    )
    assert halt_line < stamp_lines[0], backend


def test_preview_stamp_reads_the_structural_halt_fields() -> None:
    """Source lockstep on the chokepoint's halt awareness.

    A refactor that drops the ``halted`` read re-opens B1-01 for every
    preview backend at once, including ones that gain ``halt=`` later.
    """

    source = (TORCHLENS_DIR / "capture" / "outcome.py").read_text(encoding="utf-8")
    body = source.split("def stamp_backend_finalized(", 1)[1]
    body = body.split("\ndef ", 1)[0]
    assert 'getattr(trace, "halted"' in body
    assert "CaptureStatus.HALTED" in body


# ---------------------------------------------------------------------------
# End-to-end runtime arms (where the preview frameworks exist)
# ---------------------------------------------------------------------------


def test_mlx_halted_capture_settles_halted() -> None:
    pytest.importorskip("mlx.core")
    import mlx.core as mx
    import mlx.nn as mlx_nn

    class Tiny(mlx_nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = mlx_nn.Linear(3, 3)

        def __call__(self, x):  # noqa: D102
            return mlx_nn.relu(self.fc(x))

    trace = tl.trace(
        Tiny(),
        mx.ones((1, 3)),
        backend="mlx",
        halt=tl.func("relu"),
    )
    assert trace.halted is True
    assert trace.outcome is not None
    assert trace.outcome.status is CaptureStatus.HALTED


def test_paddle_halted_capture_settles_halted() -> None:
    pytest.importorskip("paddle")
    import paddle
    import paddle.nn as paddle_nn

    class Tiny(paddle_nn.Layer):
        def __init__(self) -> None:
            super().__init__()
            self.fc = paddle_nn.Linear(3, 3)

        def forward(self, x):  # noqa: D102
            return paddle_nn.functional.relu(self.fc(x))

    trace = tl.trace(
        Tiny(),
        paddle.ones([1, 3]),
        backend="paddle",
        halt=tl.func("relu"),
    )
    assert trace.halted is True
    assert trace.outcome is not None
    assert trace.outcome.status is CaptureStatus.HALTED


# ---------------------------------------------------------------------------
# R06 round 3: a SWALLOWED preview halt never settles COMPLETE
# ---------------------------------------------------------------------------


def test_preview_stamp_refuses_swallowed_halt_and_settles_failed() -> None:
    """A latched stop request with no structural halt is a swallowed signal.

    The halt-capable preview raise sites latch a ``StopRequest`` before
    raising ``HaltSignal`` (torch ``evaluate_halt`` parity). If user code eats
    the signal in a broad ``except``, the capture reaches the stamp un-halted;
    before the fix that took the COMPLETE arm -- a wrongly-blessed settled
    status for a capture whose halt never actually stopped the forward.
    """

    from torchlens.capture.outcome import StopRequest
    from torchlens.errors import StopSignalSwallowedError

    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3))
    assert bool(getattr(trace, "halted", False)) is False
    trace.__dict__["_stop_requested"] = StopRequest(
        kind="halt",
        reason="relu_1_2",
        boundary_label="relu_1_2",
    )
    with pytest.raises(StopSignalSwallowedError, match="swallowed the control signal"):
        stamp_backend_finalized(trace)
    # The capture settled FAILED, never COMPLETE, and the latch was consumed.
    settled = outcome_for(trace)
    assert settled is not None
    assert settled.status is CaptureStatus.FAILED
    assert "_stop_requested" not in trace.__dict__


def test_preview_stamp_consumes_latch_on_the_normal_halted_arm() -> None:
    """A normally-halted capture (latch + structural halt) stamps HALTED."""

    from torchlens.capture.outcome import StopRequest

    trace = _halted_trace()
    trace.__dict__["_stop_requested"] = StopRequest(
        kind="halt",
        reason=trace.halt_reason,
        boundary_label=trace.halt_frontier,
    )
    outcome = stamp_backend_finalized(trace)
    assert outcome.status is CaptureStatus.HALTED
    assert "_stop_requested" not in trace.__dict__


@pytest.mark.parametrize("backend", _HALT_CAPABLE_PREVIEW_BACKENDS)
def test_halt_capable_preview_backend_latches_stop_request_before_raising(backend: str) -> None:
    """Source lockstep: every preview HaltSignal raise latches the request first.

    The latch write must precede EVERY ``raise HaltSignal`` in the backend so
    a swallowed signal is detectable at the stamp; a raise without a latch
    reopens the swallowed-halt COMPLETE blessing for that site.
    """

    source = (TORCHLENS_DIR / "backends" / backend / "backend.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    raise_lines = [index for index, line in enumerate(lines) if "raise HaltSignal(" in line]
    assert raise_lines, backend
    latch_lines = [index for index, line in enumerate(lines) if '"_stop_requested"' in line]
    for raise_line in raise_lines:
        preceding = [index for index in latch_lines if 0 <= raise_line - index <= 12]
        assert preceding, (
            f"{backend}: raise HaltSignal at line {raise_line + 1} has no "
            "StopRequest latch write within the preceding 12 lines"
        )


def test_preview_stamp_source_checks_the_stop_latch() -> None:
    """Source lockstep: the one preview stamp consumes and checks the latch."""

    source = (TORCHLENS_DIR / "capture" / "outcome.py").read_text(encoding="utf-8")
    body = source.split("def stamp_backend_finalized(", 1)[1]
    body = body.split("\ndef ", 1)[0]
    assert '"_stop_requested"' in body
    assert "StopSignalSwallowedError" in body
    assert "settle_failed" in body
