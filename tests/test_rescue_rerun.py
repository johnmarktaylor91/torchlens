"""Rescue-rerun mechanism tests (stage-2 safety net).

The outcome corpus (``test_detached_reference_capture_outcomes.py``) pins
per-escape-class outcomes; THIS module pins the driver mechanics: the net is
never armed on a clean primary capture, ineligible captures skip the rescue,
RNG is restored to capture entry so stochastic re-runs replay the primary
draw, and the opt-in escape-detector diagnostic also triggers the rescue.
"""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any

import pytest
import torch
from _stale_holders import OpaqueCallable
from torch import nn

import torchlens as tl
from torchlens._utils._tensor_label import is_decorated_function
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
from torchlens.options import CaptureOptions


@pytest.fixture()
def raw_cos() -> Any:
    """A pristine pre-wrap ``torch.cos`` in an unrebindable holder, rewrapping afterwards."""
    unwrap_torch()
    raw = torch.cos
    assert not is_decorated_function(raw)
    try:
        yield OpaqueCallable(raw)
    finally:
        wrap_torch()


def _stale_closure_model(raw: Callable[..., Any]) -> nn.Module:
    def invoke(v: torch.Tensor) -> torch.Tensor:
        return raw(v)

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(invoke(torch.sigmoid(v)))

    return Model()


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_streaming_capture_skips_rescue(raw_cos: Any, tmp_path: Any) -> None:
    """A disk-streamed capture is not re-runnable: escape DISCLOSED, no rescue.

    grind-r4 b6-opus R16-1: this path formerly returned clean-capture fields
    (verified=None / reason=None / rescue_rerun=None) with the escaped op
    silently missing -- bit-indistinguishable from a genuinely clean capture.
    Ineligibility must skip only the re-run, never the disclosure.
    """

    wrap_torch()
    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = tl.trace(
            _stale_closure_model(raw_cos),
            torch.tensor([0.25, 0.5]),
            storage=tl.to_disk(str(tmp_path / "run.tlspec")),
        )
    assert "cos" not in [op.func_name for op in trace.ops]
    assert trace.provenance.verified is False
    assert trace.provenance.verification_reason == "escape_rescue_unrecovered"
    assert trace._rescue_rerun is not None
    assert trace._rescue_rerun["recovered"] is False
    assert trace._rescue_rerun["skipped_reason"] == "rescue_ineligible"
    assert trace._rescue_rerun["forward_runs"] == 1


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_ineligible_capture_with_escape_signal_settles_disclosure() -> None:
    """Driver-level R16-1 pin: eligible=False + live signal -> marked trace,
    exactly ONE forward run, and the ineligibility warning."""

    from torchlens.backends.torch.rescue import capture_with_rescue

    runs: list[int] = []

    def run_capture() -> Any:
        runs.append(1)
        return _stub_trace(["relu"], signal=True)

    with pytest.warns(UserWarning, match="skipped the rescue"):
        trace = capture_with_rescue(run_capture, eligible=False)
    assert len(runs) == 1
    assert trace._provenance_verified is False
    assert trace._provenance_verification_reason == "escape_rescue_unrecovered"
    assert trace._rescue_rerun["skipped_reason"] == "rescue_ineligible"


def _stub_trace(
    op_names: list[str],
    *,
    signal: bool = False,
    modules: list[Any] | None = None,
    **extra: Any,
) -> Any:
    """Build a minimal trace stub for direct driver tests.

    ``ops`` and ``layer_list`` share one row list: the driver reads every row
    through ``layer_list`` (sumfam F1), and tests append rows through ``ops``.
    """

    rows = [types.SimpleNamespace(func_name=name) for name in op_names]
    fields: dict[str, Any] = {
        "escape_diagnostics": [],
        "_had_unattributed_tensor_args": signal,
        "ops": rows,
        "_layer_list": rows,
        "modules": modules or [],
        "_provenance_verified": None,
        "_provenance_verification_reason": None,
    }
    fields.update(extra)
    return types.SimpleNamespace(**fields)


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_defusion_is_not_counted_as_recovery() -> None:
    """R16-1: a rescued op multiset that LOSES ops is mode perturbation.

    The one-sided ``Counter.__sub__`` oracle read eval-MHA de-fusion (one
    fused op replaced by many small ops) as pure gains and swapped the
    canonical fused primary for the mode-perturbed rescue. The two-sided
    oracle keeps the mode-free primary and discloses both deltas.
    """

    from torchlens.backends.torch.rescue import capture_with_rescue

    runs: list[str] = []
    primary = _stub_trace(["mha_fused", "relu"], signal=True)
    rescued = _stub_trace(["matmul", "softmax", "matmul", "relu"])

    def run_capture() -> Any:
        runs.append("run")
        return primary if len(runs) == 1 else rescued

    result = capture_with_rescue(run_capture)
    assert result is primary
    assert result._provenance_verification_reason == "escape_rescue_unrecovered"
    info = result._rescue_rerun
    assert info["recovered"] is False
    assert "mha_fused" in info["lost_ops"]
    assert "matmul" in info["recovered_ops"]


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_buffer_writing_primary_refuses_the_rescue_rerun() -> None:
    """R16-2: a primary that WROTE buffers would double-apply them; refuse."""

    from torchlens.backends.torch.rescue import capture_with_rescue

    runs: list[str] = []
    primary = _stub_trace(["relu"], signal=True)
    primary.ops.append(
        types.SimpleNamespace(func_name="none", buffer_write_kind="inplace", label_raw="buffer_bn1")
    )

    def run_capture() -> Any:
        runs.append("run")
        return primary

    with pytest.warns(UserWarning, match="double-apply"):
        result = capture_with_rescue(run_capture)
    assert len(runs) == 1, "the forward must run exactly once for a buffer-writing primary"
    assert result is primary
    info = result._rescue_rerun
    assert info["skipped_reason"] == "buffer_writes_double_forward"
    assert info["forward_runs"] == 1
    assert result._provenance_verification_reason == "escape_rescue_unrecovered"


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/rescue.py, campaign mutant kills.
def test_eval_mode_batchnorm_capture_still_rescues_end_to_end(raw_cos: Any) -> None:
    """Eval-mode BN journals fused no-op writes; the rescue must still run."""

    wrap_torch()

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn = nn.BatchNorm1d(2)

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(raw_cos(self.bn(v)))

    model = Model()
    model.eval()
    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = tl.trace(model, torch.randn(4, 2))
    assert model.bn.num_batches_tracked.item() == 0, "eval-mode BN never writes its counter"
    info = trace._rescue_rerun
    assert info is not None
    assert info["recovered"] is True
    assert info["forward_runs"] == 2
    assert "cos" in [op.func_name for op in trace.ops]
    assert trace.provenance.verification_reason == "mode_rescue_rerun"


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/rescue.py.
def test_attribution_failed_rescue_fails_closed_without_write_record() -> None:
    """An unprovable buffer-write record refuses the re-run, never guesses."""

    from torchlens._errors import OutputAttributionError
    from torchlens.backends.torch.rescue import capture_with_rescue

    runs: list[str] = []

    def run_capture() -> Any:
        runs.append("run")
        raise OutputAttributionError("output could not be attributed")

    with (
        pytest.warns(UserWarning, match="buffer-write state unprovable"),
        pytest.raises(OutputAttributionError),
    ):
        capture_with_rescue(run_capture)
    assert len(runs) == 1


# Kept by shrink-r4 (per-id subset rule): sole coverage of _runnable/state_live.py, campaign mutant kills.
def test_train_bn_attribution_failure_never_double_mutates(raw_cos: Any) -> None:
    """End-to-end: OAE + train-mode BN raises with the counter at ONE."""

    from torchlens._errors import OutputAttributionError

    wrap_torch()

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn = nn.BatchNorm1d(2)

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            # Output produced solely by the stale reference: unattributable.
            return raw_cos(self.bn(v))

    model = Model()
    model.train()
    with (
        pytest.warns(UserWarning, match="double-apply"),
        pytest.raises(OutputAttributionError) as exc_info,
    ):
        tl.trace(model, torch.randn(4, 2), capture=CaptureOptions(carry_state=True))
    assert model.bn.num_batches_tracked.item() == 1, "the forward must not run twice"
    partial = getattr(exc_info.value, "partial_trace", None)
    partial_trace = getattr(partial, "trace", None)
    assert partial_trace is not None
    info = partial_trace._rescue_rerun
    assert info is not None
    assert info["skipped_reason"] == "buffer_writes_double_forward"
    assert info["forward_runs"] == 1


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_refused_attribution_rescue_flushes_failure_advisory(raw_cos: Any) -> None:
    """When the re-run is refused and the error propagates, the advisory does too."""

    import warnings as warnings_module

    from torchlens._errors import OutputAttributionError

    wrap_torch()

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn = nn.BatchNorm1d(2)

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return raw_cos(self.bn(v))

    model = Model()
    model.train()
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        with pytest.raises(OutputAttributionError):
            tl.trace(model, torch.randn(4, 2))
    advisories = [w for w in caught if "capture attempt failed" in str(w.message)]
    assert len(advisories) == 1, [str(w.message) for w in caught]
    assert issubclass(advisories[0].category, RuntimeWarning)


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/rescue.py.
def test_recovered_rescue_never_clobbers_dynamo_verdict() -> None:
    """R16-3: dynamo_region_not_logged outranks mode_rescue_rerun in _mark."""

    from torchlens.backends.torch.rescue import capture_with_rescue

    runs: list[str] = []
    primary = _stub_trace(["relu"], signal=True)
    rescued = _stub_trace(
        ["relu", "cos"],
        _raw_dynamo_region_detected=True,
        _provenance_verification_reason="dynamo_region_not_logged",
    )

    def run_capture() -> Any:
        runs.append("run")
        return primary if len(runs) == 1 else rescued

    result = capture_with_rescue(run_capture)
    assert result is rescued
    assert result._provenance_verification_reason == "dynamo_region_not_logged"
    assert result._rescue_rerun is not None, "the rescue attempt must stay disclosed"


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/escape_detection.py, backends/torch/rescue.py, campaign mutant kills.
def test_escape_detector_diagnostic_triggers_rescue(raw_cos: Any) -> None:
    """The opt-in shadow detector's diagnostics are a rescue trigger too."""

    from torchlens._errors import CaptureGapWarning

    try:
        wrap_torch(escape_detector="shadow")
        with pytest.warns(CaptureGapWarning):
            trace = tl.trace(_stale_closure_model(raw_cos), torch.tensor([0.25, 0.5]))
        info = trace._rescue_rerun
        assert info is not None and info["recovered"] is True
        # REVIEWED FLIP (b3-fable R02-2 / b6-fable R16-3): the rescued run's
        # own shadow-detector report is a MORE specific verdict than the
        # generic mode_rescue_rerun stamp and is no longer clobbered by it;
        # the rerun stays disclosed through ``rescue_rerun`` above.
        assert trace.provenance.verification_reason == "callable_escape_shadow_report"
        assert info["primary_escape_diagnostics"]
        assert "cos" in [op.func_name for op in trace.ops]
    finally:
        unwrap_torch()
        wrap_torch()


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/rescue.py, campaign mutant kills.
def test_rescue_never_double_applies_in_forward_parameter_writes(raw_cos: Any) -> None:
    """A forward that writes declared PARAMETER state must not be applied twice.

    R02-1 regression: the journal-based refusal indexes registered-BUFFER
    writes only, so a stale-ref escape on a param-mutating forward re-ran the
    forward, doubled the write, and returned the second, differently
    parameterized capture with no warning. The state snapshot now detects the
    rescue's write, restores the duplicate application, and keeps the primary
    with the escape disclosed.
    """

    class ParamWriter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            self.lin.weight.data.mul_(2.0)
            return torch.relu(raw_cos(self.lin(v)))

    model = ParamWriter()
    model.eval()
    baseline = model.lin.weight.detach().clone()

    with pytest.warns(UserWarning, match="wrote model state"):
        trace = tl.trace(model, torch.randn(3, 4))

    # Exactly ONE forward's worth of mutation survives on the user's model.
    assert torch.equal(model.lin.weight.detach(), baseline * 2.0)
    # The rescue was attempted, detected as state-writing, undone, refused.
    assert trace._rescue_rerun is not None
    assert trace._rescue_rerun["recovered"] is False
    assert trace._rescue_rerun["skipped_reason"] == "state_writes_double_forward_undone"
    assert trace._rescue_rerun["forward_runs"] == 2
    assert trace.provenance.verified is False


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_journal_visible_buffer_writes_still_refuse_before_the_rerun(raw_cos: Any) -> None:
    """Pin: a value-changing registered-buffer write still refuses PRE-rescue.

    The cheap journal guard runs first and skips the second forward entirely
    (forward_runs == 1); the snapshot transaction is the backstop for the
    classes the journal cannot see, never a replacement for this fast path.
    """

    class BufferWriter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.register_buffer("count", torch.zeros(1))

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            self.count.detach().numpy()[0] += 1.0
            return torch.relu(raw_cos(self.lin(v)))

    model = BufferWriter()
    model.eval()

    with pytest.warns(UserWarning, match="wrote module buffer state"):
        trace = tl.trace(model, torch.randn(3, 4), capture=CaptureOptions(carry_state=True))

    assert float(model.count) == 1.0
    assert trace._rescue_rerun is not None
    assert trace._rescue_rerun["skipped_reason"] == "buffer_writes_double_forward"
    assert trace._rescue_rerun["forward_runs"] == 1


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/_module_boundary_adoption.py, campaign mutant kills.
def test_ownership_snapshot_pins_members_against_id_reuse() -> None:
    """Round-3 b1/b3/b4 merged: the pre-forward ownership snapshot PINS its members.

    A bare ``set[int]`` of recyclable ids had no liveness pinning: a model
    that dropped a snapshotted cache tensor mid-forward (``self.cache = new``)
    freed the object, and a stale-pre-wrap escape product allocated later
    could reuse the exact id -- classified "model-owned known source", so the
    module-entry adoption disclosure was silently suppressed (the laundering
    3c721316 closed, reopened through the exemption added one commit later).
    Pinning every snapshot member for the session makes id reuse impossible;
    the workspace drop releases the pins at session end.
    """

    import gc
    import weakref

    from torchlens.backends.torch._module_boundary_adoption import _collect_model_owned_tensor_ids

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.cache = torch.randn(4)

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return self.lin(v)

    model = Model()
    snapshot = _collect_model_owned_tensor_ids(model)
    cache_id = id(model.cache)
    assert cache_id in snapshot, "pre-forward cache tensor missing from the snapshot"
    dropped = weakref.ref(model.cache)
    model.cache = None  # the mid-forward drop shape, minus the forward
    gc.collect()
    assert dropped() is not None, (
        "the ownership snapshot did not pin its members: the dropped cache "
        "tensor was freed, so its id is recyclable by a mid-forward "
        "stale-pre-wrap escape product (adoption disclosure laundering)"
    )


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_recovered_rescue_preserves_specific_verification_reasons() -> None:
    """A recovered rescue must not demote a specific verdict to the generic stamp.

    Only the dynamo reason was protected; an armed detector/witness verdict on
    the rescued run (owner_thread_tripwire_changed, callable_escape_shadow_report,
    dispatch_witness_unaccounted_ops, ...) was clobbered into mode_rescue_rerun.
    """

    from torchlens.backends.torch.rescue import capture_with_rescue

    primary = _stub_trace(["none", "linear"], signal=True)
    rescued = _stub_trace(
        ["cos", "linear"],
        _provenance_verification_reason="owner_thread_tripwire_changed",
    )
    traces = iter([primary, rescued])

    result = capture_with_rescue(lambda: next(traces))

    assert result is rescued
    assert result._provenance_verification_reason == "owner_thread_tripwire_changed"
    assert result._rescue_rerun["recovered"] is True


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/rescue.py, campaign mutant kills.
def test_warning_recorder_leaves_a_user_installed_handler_in_place() -> None:
    """The rescue driver's showwarning swaps restore identity-checked.

    A user or callback that installs its own ``warnings.showwarning`` during
    the recorded window must not be silently reverted at window exit.
    """

    import warnings as warnings_module

    from torchlens.backends.torch.rescue import _record_emitted_warnings

    def user_handler(*args: Any, **kwargs: Any) -> None:
        return None

    before = warnings_module.showwarning
    try:
        with _record_emitted_warnings(set()):
            warnings_module.showwarning = user_handler
        assert warnings_module.showwarning is user_handler
    finally:
        warnings_module.showwarning = before


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/rescue.py, campaign mutant kills.
def test_failed_rescue_rerun_still_restores_state_writes() -> None:
    """R63: a rescue re-run that WRITES declared state and then RAISES must
    still restore the snapshot -- the except arm formerly skipped it, leaving
    the write double-applied on the user's model."""

    from torchlens.backends.torch.rescue import capture_with_rescue

    model = nn.Linear(4, 4)
    baseline = model.weight.detach().clone()
    calls = {"n": 0}

    def run_capture() -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return _stub_trace(["relu"], signal=True)
        with torch.no_grad():
            model.weight.add_(1.0)  # the partial forward's state write
        raise RuntimeError("rescue died mid-forward")

    trace = capture_with_rescue(run_capture, model=model)
    assert calls["n"] == 2
    assert torch.equal(model.weight, baseline), (
        "the failed rescue re-run's state write was not restored "
        "(double-applied mutation left on the model)"
    )
    assert trace._provenance_verification_reason == "escape_rescue_unrecovered"
    assert "rescue died mid-forward" in trace._rescue_rerun["rescue_error"]


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/rescue.py.
def test_interrupted_rescue_rerun_still_restores_state_writes() -> None:
    """R63 interrupt arm: KeyboardInterrupt mid-re-run propagates, but the
    snapshot restore still runs on the way out."""

    from torchlens.backends.torch.rescue import capture_with_rescue

    model = nn.Linear(4, 4)
    baseline = model.weight.detach().clone()
    calls = {"n": 0}

    def run_capture() -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            return _stub_trace(["relu"], signal=True)
        with torch.no_grad():
            model.weight.add_(1.0)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        capture_with_rescue(run_capture, model=model)
    assert torch.equal(model.weight, baseline), (
        "the interrupted rescue re-run's state write was not restored"
    )


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_successful_rescue_rerun_warns_about_the_double_forward() -> None:
    """R67: the success path was the ONLY silent rescue outcome.

    Fail-before: a recovered rescue returned capture_verified=False with zero
    warnings while the user's forward had executed twice.
    """

    from torchlens._errors import CaptureGapWarning
    from torchlens.backends.torch.rescue import capture_with_rescue

    runs: list[str] = []
    primary = _stub_trace(["relu"], signal=True)
    rescued = _stub_trace(["relu", "cos"])

    def run_capture() -> Any:
        runs.append("run")
        return primary if len(runs) == 1 else rescued

    with pytest.warns(CaptureGapWarning, match="executed TWICE"):
        result = capture_with_rescue(run_capture)
    assert result is rescued


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_restore_changed_state_is_nan_aware() -> None:
    """r8 R16: a state slot legitimately holding NaN is not "changed".

    ``torch.equal`` answers False for bitwise-identical NaNs, so a NaN-bearing
    buffer read as rescue-mutated on every re-run -- a false double-mutation
    report plus a pointless restore copy (red-capable: pre-fix the changed
    tuple names the buffer).
    """

    import torch
    from torch import nn

    from torchlens.backends.torch.rescue import _restore_changed_state

    class _NanBuffered(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("stat", torch.tensor([1.0, float("nan"), 3.0]))

    model = _NanBuffered()
    snapshot = {"buffer:stat": model.stat.detach().clone()}
    assert _restore_changed_state(model, snapshot) == ()

    model.stat[0] = 2.0  # a REAL change must still be caught and restored
    changed = _restore_changed_state(model, snapshot)
    assert changed == ("buffer:stat",)
    assert model.stat[0].item() == 1.0
