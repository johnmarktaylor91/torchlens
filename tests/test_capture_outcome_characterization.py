"""P0 characterization pins for the early-stopping unification.

These tests pin TODAY's terminal-path behavior before the outcome/settlement
machinery lands (design of record: early-stopping unification v4). Each pin is
either a permanent invariant (exception identity, chaining, structural facts)
or an explicitly-labelled CURRENT-BEHAVIOR characterization that a later phase
(F5, N4, N5) is expected to flip; flips must update the pin in the same commit
and cite the phase.
"""

from __future__ import annotations

import pathlib
import pickle
import re

import pytest
import torch
from fixtures.capture_outcome_models import ThreeStageModel, halt_on_relu
from torch import nn

import torchlens as tl
import torchlens.postprocess as pp
from torchlens.fastlog import Recorder
from torchlens.fastlog._halt import HaltSignal

TORCHLENS_DIR = pathlib.Path(tl.__file__).resolve().parent


class FailingForwardModel(nn.Module):
    """Model whose forward raises after one traced op."""

    def __init__(self, exc: BaseException) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self._exc = exc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.linear(x)
        raise self._exc


class BatchNormModel(nn.Module):
    """BN model for the F3b halted buffer-reconciliation characterization."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(3)
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(self.bn(x)))


def _halt_on_linear(ctx: object) -> bool:
    return ctx.kind == "op" and ctx.func_name == "linear"


# ---------------------------------------------------------------------------
# Writer-set lockstep: `_tracing_finished = True` writer inventory
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_tracing_finished_writer_inventory_is_exact() -> None:
    """The `_tracing_finished = True` writer set is a reviewed inventory.

    The settlement design treats the flag as a STRUCTURAL finished-ness
    marker, never a settlement point. Every trace-level write must sit inside
    a region whose only success exit reaches an authority stamp. A new writer
    appearing here is a contract change: extend the inventory AND route the
    new region through the settlement authority.
    """

    # The sanctioned writers: postprocess/finalization.py::_set_tracing_finished
    # (torch step 17 + the no-layers early return), the shared preview finalize
    # (both finish_before_module_logs arms), and JAX's own finalize path.
    writer_files: dict[str, int] = {}
    for path in TORCHLENS_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        # Trace-level writers only: `trace._tracing_finished = True` or
        # `self._tracing_finished = True` (per-op writers set it on tensors).
        hits = re.findall(
            r"^\s*(?:trace|self)\._tracing_finished\s*=\s*True\s*$",
            text,
            flags=re.MULTILINE,
        )
        if hits:
            writer_files[str(path.relative_to(TORCHLENS_DIR))] = len(hits)
    assert writer_files == {
        "postprocess/finalization.py": 1,
        "backends/_finalize.py": 2,
        "backends/jax/backend.py": 1,
    }, (
        "The `_tracing_finished = True` trace-level writer inventory changed: "
        f"{writer_files}. Flag writes are structural, never terminal; a new "
        "writer must route its region through the settlement authority and be "
        "added to this inventory in the same reviewed change."
    )


# ---------------------------------------------------------------------------
# Planted postprocess failures (exhaustive torch capture)
# ---------------------------------------------------------------------------


def test_planted_step1_failure_survives_with_partial_log(monkeypatch) -> None:
    """A pre-seam postprocess failure propagates and attaches partial_log."""

    def _boom(*args: object, **kwargs: object) -> None:
        raise ValueError("planted step1")

    monkeypatch.setattr(pp, "_add_output_layers", _boom)
    with pytest.raises(ValueError, match="planted step1") as exc_info:
        tl.trace(ThreeStageModel(), torch.ones(1, 3))
    assert hasattr(exc_info.value, "partial_log")
    partial = tl.partial.from_failed_capture(exc_info.value)
    assert partial.trace.__dict__.get("_tracing_finished") is False


def test_planted_freeze_failure_propagates_original(monkeypatch) -> None:
    """FLIPPED (F5): a post-seam freeze failure propagates the ORIGINAL error.

    The P0 characterization pinned the double-fault: the freeze runs after
    the step-17 transient-state pops, and the unguarded
    ``cleanup_failed_forward_session`` workspace read masked the planted
    error with ``AttributeError: _raw_graph_ws`` (original only as
    ``__context__``). F5 guards the cleanup so the original exception
    survives, and settlement still stamps FAILED/POSTPROCESS.
    """

    def _boom(*args: object, **kwargs: object) -> None:
        raise ValueError("planted freeze")

    monkeypatch.setattr(pp, "_freeze_relation_views", _boom)
    with pytest.raises(ValueError, match="planted freeze"):
        tl.trace(ThreeStageModel(), torch.ones(1, 3))


def test_planted_step20_failure_propagates_original(monkeypatch) -> None:
    """FLIPPED (F5): a step-20 failure propagates the original error."""

    from torchlens.data_classes.trace import Trace

    def _boom(self: object, **kwargs: object) -> None:
        raise ValueError("planted step20")

    monkeypatch.setattr(Trace, "release_param_refs", _boom)
    with pytest.raises(ValueError, match="planted step20"):
        tl.trace(ThreeStageModel(), torch.ones(1, 3))


# ---------------------------------------------------------------------------
# Halted-secondary failures
# ---------------------------------------------------------------------------


def test_halted_postprocess_failure_propagates_with_halt_context(monkeypatch) -> None:
    """A halted-postprocess failure keeps the original error and HaltSignal chain."""

    def _boom(*args: object, **kwargs: object) -> None:
        raise ValueError("planted halted-pp")

    monkeypatch.setattr(pp, "_add_output_layers", _boom)
    with pytest.raises(ValueError, match="planted halted-pp") as exc_info:
        tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=halt_on_relu)
    assert isinstance(exc_info.value.__context__, HaltSignal)


@pytest.mark.smoke
def test_halted_finalize_failure_propagates_with_halt_context(monkeypatch) -> None:
    """A halted-FINALIZE failure (cleanup) propagates with the halt chained.

    This is the FAILED/FINALIZE class: the failure happens inside
    ``_finalize_halted_trace`` before the halted postprocess starts.
    """

    from torchlens.backends.torch.backend import TorchBackend

    def _boom(self: object, session: object, prepared: object) -> None:
        raise ValueError("planted halted-cleanup")

    monkeypatch.setattr(TorchBackend, "cleanup_model_session", _boom)
    with pytest.raises(ValueError, match="planted halted-cleanup") as exc_info:
        tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=halt_on_relu)
    assert isinstance(exc_info.value.__context__, HaltSignal)


# ---------------------------------------------------------------------------
# Halted product capabilities (N4/N5 adjudication baselines)
# ---------------------------------------------------------------------------


def test_halted_trace_basic_shape() -> None:
    """A halted capture finishes postprocess: finished truthy, halted truthy."""

    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=halt_on_relu)
    assert trace.halted is True
    assert trace._tracing_finished is True
    assert trace.halt_frontier == trace.halt_reason


def test_halted_trace_has_no_transient_attribution_leak() -> None:
    """FLIPPED (P2): the halted finalizer pops the transient attribution list.

    The P0 characterization pinned the leak (an unpopped
    ``_output_attribution_input_tensors`` refused every halted ``tl.save``
    via PORTABLE_STATE_SPEC); the settlement phase fixed it so the halted
    finalizer mirrors the completed paths. The positive save round-trip
    lives in test_capture_settlement.py.
    """

    trace = tl.trace(ThreeStageModel(), torch.ones(1, 3), halt=halt_on_relu)
    assert "_output_attribution_input_tensors" not in trace.__dict__


def test_halted_trace_refresh_refuses_typed() -> None:
    """FLIPPED (N5): halted refresh refuses with the typed live-provider gate.

    The P0 characterization pinned the misleading "computational graph
    changed" crash; N5 converts it into a typed HALTED refusal naming the
    re-arm follow-on.
    """

    from torchlens.capture.outcome import CaptureOutcomeError

    model = ThreeStageModel()
    x = torch.ones(1, 3)
    trace = tl.trace(model, x, halt=halt_on_relu)
    with pytest.raises(CaptureOutcomeError) as exc_info:
        trace.save_new_outs(model, x)
    assert exc_info.value.fields["code"] == "N5"


def test_halted_trace_fast_run_refuses_typed() -> None:
    """FLIPPED (N5): fast=True no longer silently accepts a halted trace.

    The P0 characterization pinned the false-blessing hole: the guarded
    static-loop path re-executed the NATIVE forward (the full graph) against
    a prefix trace and reported success. N5 refuses HALTED on every live
    provider.
    """

    from torchlens.capture.outcome import CaptureOutcomeError

    model = ThreeStageModel()
    x = torch.ones(1, 3)
    trace = tl.trace(model, x, halt=halt_on_relu)
    with pytest.raises(CaptureOutcomeError) as exc_info:
        trace.run(inputs=x, fast=True)
    assert exc_info.value.fields["code"] == "N5"


@pytest.mark.smoke
def test_halted_trace_runnable_save_refuses_typed() -> None:
    """FLIPPED (N4): halted runnable save is a typed runnable refusal.

    The P0 characterization proved no legacy workflow successfully saved a
    halted runnable (preflight already failed), so N4 is a crash->typed
    conversion, surfacing through the runnable error vocabulary.
    """

    from torchlens.errors import RunnablePreflightError
    from torchlens.runnable import RunnableErrorCode

    model = ThreeStageModel()
    x = torch.ones(1, 3)
    trace = tl.trace(model, x, halt=halt_on_relu)
    with pytest.raises(RunnablePreflightError) as exc_info:
        tl.save(trace, "/tmp/tl_p0_halted_runnable.tlspec", level="runnable", overwrite=True)
    assert exc_info.value.fields.get("code") == RunnableErrorCode.HALTED_CAPTURE_NOT_RUNNABLE.value


# ---------------------------------------------------------------------------
# F3b: halted BN buffer-reconciliation characterization
# ---------------------------------------------------------------------------


def test_halted_bn_buffer_topology_matches_complete_prefix() -> None:
    """F3b characterization: halted BN capture matches the complete capture.

    The halted path skips ``finalize_forward_session`` (and with it
    ``reconcile_buffer_writes``), yet the observable buffer topology and the
    buffer-write event count are identical to the complete capture through
    the halt frontier. Byte-identity here adjudicates F3b to DISCLOSE (no
    behavior fix): flipping this pin requires a real divergence.
    """

    model_full, model_halt = BatchNormModel(), BatchNormModel()
    model_halt.load_state_dict(model_full.state_dict())
    x = torch.randn(4, 3)
    trace_full = tl.trace(model_full, x)
    trace_halt = tl.trace(model_halt, x, halt=_halt_on_linear)

    assert list(trace_full.buffer_layers) == list(trace_halt.buffer_layers)
    full_labels = [entry.layer_label for entry in trace_full.layer_list]
    halt_labels = [entry.layer_label for entry in trace_halt.layer_list]
    # The halted graph is the complete graph's prefix plus its own output node.
    assert halt_labels[:-1] == [
        label for label in full_labels if label not in ("relu_1_4", "output_1")
    ]
    events_full = trace_full.__dict__.get("_capture_events")
    events_halt = trace_halt.__dict__.get("_capture_events")
    assert len(events_full.buffer_write_events) == len(events_halt.buffer_write_events)
    assert torch.allclose(model_full.bn.running_mean, model_halt.bn.running_mean)


# ---------------------------------------------------------------------------
# Interrupt pins (path 6 and path 13)
# ---------------------------------------------------------------------------


def test_keyboard_interrupt_propagates_without_partial_banner() -> None:
    """BaseException escapes propagate bare: no partial_log, no banner."""

    with pytest.raises(KeyboardInterrupt) as exc_info:
        tl.trace(FailingForwardModel(KeyboardInterrupt()), torch.ones(1, 3))
    assert not hasattr(exc_info.value, "partial_log")


@pytest.mark.smoke
def test_recorder_keyboard_interrupt_leaves_no_product() -> None:
    """Recorder BaseException: propagates; `.recording` raises typed."""

    from torchlens.fastlog.exceptions import RecorderStateError

    recorder = Recorder(FailingForwardModel(KeyboardInterrupt()), save=lambda ctx: True)
    with pytest.raises(KeyboardInterrupt), recorder as active:
        active.log(torch.ones(1, 3))
    with pytest.raises(RecorderStateError):
        _ = recorder.recording


# ---------------------------------------------------------------------------
# Recorder mixed multi-pass pin
# ---------------------------------------------------------------------------


def test_recorder_mixed_multipass_halt_pins() -> None:
    """One completed pass plus one halted pass: per-pass halt bookkeeping."""

    class TwoOpModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(x))

    def _halt_second_pass(ctx: object) -> bool:
        return ctx.kind == "op" and ctx.func_name == "relu" and ctx.pass_index >= 2

    x = torch.ones(1, 3)
    with Recorder(
        TwoOpModel(), save=lambda ctx: ctx.kind == "op", halt=_halt_second_pass
    ) as recorder:
        first_output = recorder.log(x)
        second_output = recorder.log(x)
    recording = recorder.recording
    assert first_output is not None
    assert second_output is None
    assert recording.status == "halted"
    assert recording.halted is True
    assert set(recording.halts_by_pass) == {2}


# ---------------------------------------------------------------------------
# Rehydration alignment (bb0e69aa): falsy-finished loads stay coreless
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_failed_partial_pickle_loads_coreless() -> None:
    """A pickled falsy-finished partial loads coreless (staging surface)."""

    with pytest.raises(RuntimeError, match="user forward boom"), torch.no_grad():
        tl.trace(
            FailingForwardModel(RuntimeError("user forward boom")),
            torch.ones(1, 3),
        )
    # from_failed_capture needs the live exception object.
    try:
        with torch.no_grad():
            tl.trace(
                FailingForwardModel(RuntimeError("user forward boom")),
                torch.ones(1, 3),
            )
    except RuntimeError as exc:
        partial = tl.partial.from_failed_capture(exc)
    source = partial.trace
    assert source.__dict__.get("_tracing_finished") is False
    clone = pickle.loads(pickle.dumps(source))
    assert clone.__dict__.get("_tracing_finished") is False
    assert clone.__dict__.get("_trace_core") is None
