"""L4 until= battery: truncation term, positive-claim cap, prefix projection.

Pinned per the design-review-converged L4 design memo (sec 2-3) and S2 ratification
sec 2: truncation is a run-result term, never a capture outcome; a truncated
run never settles a positive claim; the prefix projection never loosens; the
two post-return feedback values are explicitly neutralized on a live-until
fork (the r5 fork-proof rule).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._record_type import record_type
from torchlens._runnable import (
    NumericAttestationStatus,
    PathFaithfulness,
    RunnableErrorCode,
)
from torchlens.errors import RunCapabilityUnavailableError
from torchlens.options import CaptureOptions


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
    return next(layer.layer_label for layer in log._layer_list if layer.type == layer_type)


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
        capture=CaptureOptions(replayable=True, capture_container_structure=True),
    )
    path = tmp_path / "until.tlspec"
    log.save(str(path), level="runnable", include_weights=True)
    return tl.load(str(path)), x


# Kept: sole branch coverage of _user_funcs.py, io/__init__.py.
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
    from torchlens._capture.outcome import CaptureStatus

    prior = tlio.list_traces()
    _model, _log, result = _live_until_relu()
    fresh = [entry for entry in tlio.list_traces() if not any(entry is p for p in prior)]
    halted_fresh = [
        entry
        for entry in fresh
        if getattr(entry, "outcome", None) is not None
        and entry.outcome.status is CaptureStatus.STOPPED
    ]
    assert not halted_fresh, (
        "the truncated live run left a HALTED trace enumerable through "
        "tl.io.list_traces(); the internal halt settle must stay invisible"
    )
    assert any(entry is result.trace for entry in fresh)


# Kept: smoke tier id (frozen).
@pytest.mark.smoke
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


# Kept: sole branch coverage of _capture/projectors.py.
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


# Kept: sole branch coverage of _capture/projectors.py.
def test_until_x_eval_bn_prefix_tamper_refuses_typed():
    """An in-prefix buffer-sink evidence tamper refuses typed under until=."""

    from torchlens.errors import BufferSinkRoutingError

    model = _EvalBnChain()
    model.eval()
    log = tl.trace(model, torch.randn(2, 4))
    for label in log.internal_sink_ops:
        layer = log._layer_dict_all_keys[label]
        if record_type(layer) == "buffer":
            layer._internal_set("buffer_value_changed", None)
    with pytest.raises(BufferSinkRoutingError):
        log.run(inputs=torch.randn(2, 4), until=_type_label(log, "relu"))


# Kept: sole branch coverage of _runnable/execution.py, intervention/_trace_run.py.
def test_truncated_result_refuses_rerun():
    """3.3.2: re-running a truncated result refuses typed at the run door."""

    _model, _log, result = _live_until_relu()
    with pytest.raises(RunCapabilityUnavailableError) as exc_info:
        result.trace.run(inputs=torch.randn(2, 4))
    assert exc_info.value.fields["code"] == RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value
    assert exc_info.value.fields["detection_stage"] == "truncated_result_rerun"


# Kept: sole branch coverage of _runnable/input_metadata.py.
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
        capture=CaptureOptions(replayable=True, capture_container_structure=True),
    )
    path = tmp_path / "two.tlspec"
    log.save(str(path), level="runnable", include_weights=True)
    loaded = tl.load(str(path))
    # until= the relu: y's consumer (the add) is skipped, but the full recorded
    # input tree remains REQUIRED -- a partial-inputs surface would let omission
    # relax a boundary check.
    relu_label = _type_label(loaded, "relu")
    with pytest.raises((ValueError, RuntimeError)):
        loaded.run(inputs=[x], until=relu_label)
    result = loaded.run(inputs=[x, y], until=relu_label)
    assert result.report.truncated is True


# Kept: sole branch coverage of _fast_live_until.py, _fast_run.py, backends/torch/wrappers.py and more.
def test_until_fast_matrix_live_runs_loaded_refuses(tmp_path: Path):
    """Composition matrix (U26.3): until= x fast=True stops a live run; a loaded one refuses.

    The live fast tier cuts its native forward after the saved site; a loaded
    artifact's fast tier compiles the full recorded path and still refuses typed.
    """

    model = _Chain()
    log = tl.trace(model, torch.randn(2, 4))
    relu_label = _type_label(log, "relu")
    result = log.run(inputs=torch.randn(2, 4), fast=True, until=relu_label)
    assert result.report.truncated is True
    assert result.report.stopped_at == relu_label
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.output is None

    loaded, _x = _loaded_runnable(tmp_path)
    with pytest.raises(ValueError) as exc_info:
        loaded.run(inputs=torch.randn(2, 4), fast=True, until=_type_label(loaded, "relu"))
    assert exc_info.value.fields["code"] == "run_fast_until_unsupported"


# Kept: sole branch coverage of _runnable/transaction.py.
def test_until_predicate_form_refuses_pre_s4():
    """Predicate/selector until= refuses typed until the S4 contract merge."""

    model = _Chain()
    log = tl.trace(model, torch.randn(2, 4))
    with pytest.raises(RunCapabilityUnavailableError) as exc_info:
        log.run(inputs=torch.randn(2, 4), until=lambda ctx: True)
    assert exc_info.value.fields["detection_stage"] == "predicate_surface_pending"


# Kept: sole branch coverage of _runnable/transaction.py.
def test_until_junk_form_refuses_typed():
    """Non-string static forms refuse typed with the form code."""

    model = _Chain()
    log = tl.trace(model, torch.randn(2, 4))
    with pytest.raises(ValueError) as exc_info:
        log.run(inputs=torch.randn(2, 4), until=3)
    assert exc_info.value.fields["code"] == "run_until_form_invalid"


# Kept: sole branch coverage of _data_classes/_trace_accessors.py, _data_classes/_trace_views.py, _data_classes/interface.py and more.
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


# Kept: sole branch coverage of _runnable/transaction.py, intervention/_trace_run.py.
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


# Kept: sole branch coverage of _runnable/transaction.py, intervention/_trace_run.py.
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
