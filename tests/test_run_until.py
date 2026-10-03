"""L4 until= battery: truncation term, positive-claim cap, prefix projection.

Pinned per the design-review-converged L4 design memo (sec 2-3) and S2 ratification
sec 2: truncation is a run-result term, never a capture outcome; a truncated
run never settles a positive claim; the prefix projection never loosens; the
two post-return feedback values are explicitly neutralized on a live-until
fork (the r5 fork-proof rule).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import RunCapabilityUnavailableError
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    NumericAttestationStatus,
    PathFaithfulness,
    RunnableErrorCode,
    TensorSlotRole,
)

pytestmark = pytest.mark.smoke


class _Chain(nn.Module):
    """Bare-tensor-output chain (the adversarial shape for the fork proof)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        return self.fc2(h).sigmoid()


class _EvalBnChain(nn.Module):
    """Eval-mode BatchNorm inside the executed prefix (D18 composition)."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn(x))
        return self.fc(h).sigmoid()


def _type_label(log, layer_type):
    return next(layer.layer_label for layer in log.layer_list if layer.layer_type == layer_type)


def _live_until_relu(model_cls=_Chain):
    model = model_cls()
    if isinstance(model, _EvalBnChain):
        model.eval()
    x = torch.randn(2, 4)
    log = tl.trace(model, x)
    result = log.run(inputs=torch.randn(2, 4), until=_type_label(log, "relu"))
    return model, log, result


def _loaded_runnable(tmp_path, model=None):
    model = model or _Chain()
    x = torch.randn(2, 4)
    log = tl.trace(
        model,
        x,
        capture=CaptureOptions(intervention_ready=True, capture_container_structure=True),
    )
    path = tmp_path / "until.tlspec"
    tl.save(log, str(path), level="runnable", include_weights=True)
    return tl.load(str(path)), x


def test_live_truncated_fork_output_proof_fails_closed():
    """r5 fork-proof pin: neither inherited nor frontier proof survives the latch.

    Bare-tensor-output model truncated mid-forward (the adversarial shape: the
    inherited source stamp IS positive there). A build that merely skips the
    feedback copy leaves the inherited positive Mapping in place and FAILS the
    first assertion.
    """

    from torchlens._runnable_execution import _fresh_bare_tensor_root

    _model, log, result = _live_until_relu()
    assert log._runnable.output_losslessness is not None  # the source stamp IS positive
    assert result.trace._runnable.output_losslessness is None
    assert _fresh_bare_tensor_root(result.trace) is False


def test_live_truncation_does_not_assert_replay_arg_completeness():
    """The witness-family completeness flag is SET FALSE, never merely skipped."""

    model = _Chain()
    x = torch.randn(2, 4)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(save_arg_values=True))
    result = log.run(inputs=torch.randn(2, 4), until=_type_label(log, "relu"))
    assert result.trace._replay_arg_version_data_complete is False


def test_live_truncated_output_is_none_and_disclosed():
    """RunResult.output is None under live truncation; the record discloses."""

    _model, _log, result = _live_until_relu()
    report = result.report
    assert result.output is None
    assert report.truncated is True
    assert report.truncation is not None
    assert report.truncation.regime == "live_stop_after"
    assert report.stopped_at is not None and report.stopped_at.startswith("relu")
    assert report.truncation.skipped_count > 0
    # The flagship extraction chain: executed-prefix values read off the trace.
    assert result.trace[_type_label(result.trace, "relu")].out is not None


def test_live_truncation_leaves_source_outcome_complete():
    """The internal HALTED settle is invisible on both source and result."""

    from torchlens.capture.outcome import CaptureStatus

    model, log, result = _live_until_relu()
    assert log.outcome.status is CaptureStatus.COMPLETE
    # The result fork's outcome is exactly what a FULL run's fork derives
    # (forked from the COMPLETE source) -- the internal HALTED settle never
    # reaches it.
    assert result.trace.outcome.status is not CaptureStatus.HALTED
    full = log.run(inputs=torch.randn(2, 4))
    assert result.trace.outcome.status is full.trace.outcome.status


def test_live_truncation_leaves_no_halted_trace_registered():
    """No HALTED trace minted by a truncated live run is enumerable; the result IS.

    The oracle is scoped to the logs THIS run minted, against a strong-ref
    snapshot of the registry (strong refs pin identities, so a prior log
    cannot die mid-window and donate its address to a fresh one). The
    registry is process-global and a HALTED capture is a legitimate
    registered product elsewhere -- ``tl.trace(halt=...)`` returns one, and
    its dropped cyclic graph (settle evidence carries the halt traceback,
    which welds the capture frames into the trace's cycle) stays weakly
    enumerable until a full gen-2 collection, which CPython's long-lived
    25% rule makes rare mid-session. The former whole-registry sweep
    therefore failed whenever any halt-firing test ran earlier in the same
    randomized session (train T05; reproduced at chunk seed 36 with the
    polluter being test_zero_match_disclosures's own returned halted log).
    Scoping to fresh logs keeps full sensitivity to the contract under
    test: an internal throwaway left registered by the truncated run is a
    fresh HALTED entry and still fails.
    """

    from torchlens import io as tlio
    from torchlens.capture.outcome import CaptureStatus

    prior = tlio.list_logs()
    _model, _log, result = _live_until_relu()
    fresh = [entry for entry in tlio.list_logs() if not any(entry is p for p in prior)]
    halted_fresh = [
        entry
        for entry in fresh
        if getattr(entry, "outcome", None) is not None
        and entry.outcome.status is CaptureStatus.HALTED
    ]
    assert not halted_fresh, (
        "the truncated live run left a HALTED trace enumerable through "
        "tl.io.list_logs(); the internal halt settle must stay invisible"
    )
    assert any(entry is result.trace for entry in fresh)


def test_skipped_site_carries_no_stale_payload():
    """3.3.3 sanitation: skipped sites never read as fresh capture-time values."""

    _model, _log, result = _live_until_relu()
    assert result.trace[_type_label(result.trace, "sigmoid")].out is None
    assert result.trace["output_1"].out is None


def test_truncated_run_never_settles_verified(tmp_path):
    """2.4(e) cap tamper-pin: maximally clean truncated runs settle the regime pair.

    Both regimes: the call RETURNS (the sparse attestation-failure raise site is
    unreachable under truncation because the archive is never opened).
    """

    _model, _log, live_result = _live_until_relu()
    assert live_result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert live_result.report.numeric_attestation is NumericAttestationStatus.NOT_PRESENT
    assert live_result.report.truncation is not None

    loaded, x = _loaded_runnable(tmp_path)
    loaded_result = loaded.run(inputs=x, until=_type_label(loaded, "relu"))
    assert loaded_result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert loaded_result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert loaded_result.report.truncation is not None
    # The counterfactual full run on the same pair settles clean (meta-pair (b)(ii)).
    full = loaded.run(inputs=x)
    assert full.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert full.report.truncated is False


def test_seeded_prefix_bug_still_fails_under_until(tmp_path: Path) -> None:
    """2.4(c): a contradiction INSIDE the executed prefix fails like the full run."""

    from torchlens.errors import PathDivergenceError

    loaded, x = _loaded_runnable(tmp_path)
    descriptor = loaded.runnable_descriptor
    assert descriptor is not None
    # Plant the contradiction INSIDE the guard machinery: retain the real
    # runtime input but forge the descriptor's recorded model-input shape.
    tampered_slots = tuple(
        replace(slot, shape=(99, 99)) if slot.role is TensorSlotRole.MODEL_INPUT else slot
        for slot in descriptor.tensor_slots
    )
    loaded._runnable.descriptor = replace(descriptor, tensor_slots=tampered_slots)

    with pytest.raises(PathDivergenceError) as full_exc:
        loaded.run(inputs=x)
    with pytest.raises(PathDivergenceError) as until_exc:
        loaded.run(inputs=x, until=_type_label(loaded, "relu"))
    assert full_exc.value.fields["code"] == RunnableErrorCode.INPUT_SHAPE_MISMATCH.value
    assert until_exc.value.fields["code"] == full_exc.value.fields["code"]


def test_prefix_projection_mismatch_refuses_like_full():
    """A seeded graph change INSIDE the prefix refuses with the pinned term."""

    class _FlipInPrefix(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 4)
            self.flip = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.fc1(x)
            h = torch.tanh(h) if self.flip else torch.relu(h)
            return self.fc2(h).sigmoid()

    model = _FlipInPrefix()
    log = tl.trace(model, torch.randn(2, 4))
    until_label = _type_label(log, "relu")
    model.flip = True
    with pytest.raises(ValueError, match="computational graph changed"):
        log.run(inputs=torch.randn(2, 4), until=until_label)


def test_prefix_projection_never_loosens():
    """A graph change BEYOND the stop is not consulted -- and the run is still capped."""

    class _FlipAfterStop(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.fc2 = nn.Linear(4, 4)
            self.flip = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.fc1(x))
            tail = self.fc2(h)
            return torch.tanh(tail) if self.flip else tail.sigmoid()

    model = _FlipAfterStop()
    log = tl.trace(model, torch.randn(2, 4))
    until_label = _type_label(log, "relu")
    model.flip = True
    result = log.run(inputs=torch.randn(2, 4), until=until_label)
    # The post-stop flip never executed, so the prefix matched; omission buys
    # no verdict -- the run is still capped, never VERIFIED.
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.truncated is True


def test_until_x_eval_bn_prefix_projection():
    """D18 composition: eval-BN sinks inside the prefix compare at full strength."""

    _model, _log, result = _live_until_relu(_EvalBnChain)
    assert result.report.truncated is True
    assert result.trace[_type_label(result.trace, "relu")].out is not None


def test_until_x_eval_bn_prefix_tamper_refuses_typed():
    """An in-prefix buffer-sink evidence tamper refuses typed under until=."""

    from torchlens.errors import BufferSinkRoutingError

    model = _EvalBnChain()
    model.eval()
    log = tl.trace(model, torch.randn(2, 4))
    for label in log.internal_sink_ops:
        layer = log.layer_dict_all_keys[label]
        if layer.layer_type == "buffer":
            layer._internal_set("buffer_value_changed", None)
    with pytest.raises(BufferSinkRoutingError):
        log.run(inputs=torch.randn(2, 4), until=_type_label(log, "relu"))


def test_truncated_result_refuses_all_saves(tmp_path):
    """3.3.1 floor: every save of a truncated result refuses at the poison gate."""

    _model, _log, result = _live_until_relu()
    for level in ("analysis", "runnable"):
        with pytest.raises(Exception) as exc_info:  # noqa: PT011 - poison-gate class
            tl.save(result.trace, str(tmp_path / f"t_{level}.tlspec"), level=level)
        assert "poison" in str(exc_info.value).lower() or "truncat" in str(exc_info.value).lower()


def test_truncated_result_refuses_rerun():
    """3.3.2: re-running a truncated result refuses typed at the run door."""

    from torchlens.runnable import RunnableErrorCode

    _model, _log, result = _live_until_relu()
    with pytest.raises(RunCapabilityUnavailableError) as exc_info:
        result.trace.run(inputs=torch.randn(2, 4))
    assert exc_info.value.fields["code"] == RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value
    assert exc_info.value.fields["detection_stage"] == "truncated_result_rerun"


def test_until_requires_full_inputs(tmp_path):
    """Require-all inputs: the closure never shrinks the input contract."""

    class _TwoInput(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            h = torch.relu(self.fc(x))
            return h + y

    model = _TwoInput()
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    log = tl.trace(
        model,
        [x, y],
        capture=CaptureOptions(intervention_ready=True, capture_container_structure=True),
    )
    path = tmp_path / "two.tlspec"
    tl.save(log, str(path), level="runnable", include_weights=True)
    loaded = tl.load(str(path))
    # until= the relu: y's consumer (the add) is skipped, but the full recorded
    # input tree remains REQUIRED -- a partial-inputs surface would let omission
    # relax a boundary check.
    relu_label = _type_label(loaded, "relu")
    with pytest.raises((ValueError, RuntimeError)):
        loaded.run(inputs=[x], until=relu_label)
    result = loaded.run(inputs=[x, y], until=relu_label)
    assert result.report.truncated is True


def test_until_fast_refuses_typed():
    """Composition matrix: until= x fast=True refuses typed."""

    model = _Chain()
    log = tl.trace(model, torch.randn(2, 4))
    with pytest.raises(ValueError) as exc_info:
        log.run(inputs=torch.randn(2, 4), fast=True, until=_type_label(log, "relu"))
    assert exc_info.value.fields["code"] == "run_fast_until_unsupported"


def test_until_predicate_form_refuses_pre_s4():
    """Predicate/selector until= refuses typed until the S4 contract merge."""

    model = _Chain()
    log = tl.trace(model, torch.randn(2, 4))
    with pytest.raises(RunCapabilityUnavailableError) as exc_info:
        log.run(inputs=torch.randn(2, 4), until=lambda ctx: True)
    assert exc_info.value.fields["detection_stage"] == "predicate_surface_pending"


def test_until_junk_form_refuses_typed():
    """Non-string static forms refuse typed with the form code."""

    model = _Chain()
    log = tl.trace(model, torch.randn(2, 4))
    with pytest.raises(ValueError) as exc_info:
        log.run(inputs=torch.randn(2, 4), until=3)
    assert exc_info.value.fields["code"] == "run_until_form_invalid"


def test_until_substring_token_refuses_typed():
    """A substring token resolvable by ``trace[...]`` still refuses as an until= site.

    ``until=`` accepts exact layer labels and module addresses only; a
    substring like ``"relu"`` passes the flexible ``__getitem__`` lookup but
    is not a resolvable site, so the resolver's own typed refusal (not the
    fuzzy lookup error) must fire.
    """

    model = _Chain()
    log = tl.trace(model, torch.randn(2, 4))
    assert log["relu"] is not None  # the flexible lookup itself resolves
    with pytest.raises(ValueError) as exc_info:
        log.run(inputs=torch.randn(2, 4), until=["relu"])
    assert exc_info.value.fields["code"] == "run_until_form_invalid"
    assert "did not resolve to layers" in str(exc_info.value)


def test_run_save_retention_reselects():
    """Run-time save= re-selects retention only; verification is untouched."""

    model = _Chain()
    log = tl.trace(model, torch.randn(2, 4))
    result = log.run(inputs=torch.randn(2, 4), save=_type_label(log, "relu"))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.trace[_type_label(result.trace, "relu")].out is not None
    assert result.trace[_type_label(result.trace, "sigmoid")].out is None
    # Output boundary payloads are retained like the capture-time machinery.
    assert result.trace["output_1"].out is not None


def test_run_save_outside_until_window_refuses_typed():
    """2.5 row: a save= target outside the executed closure refuses typed."""

    model = _Chain()
    log = tl.trace(model, torch.randn(2, 4))
    with pytest.raises(ValueError) as exc_info:
        log.run(
            inputs=torch.randn(2, 4),
            until=_type_label(log, "relu"),
            save=_type_label(log, "sigmoid"),
        )
    assert exc_info.value.fields["code"] == "run_until_form_invalid"
