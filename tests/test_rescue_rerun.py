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
from _stale_holders import CountedRoot, OpaqueCallable
from torch import nn

import torchlens as tl
from torchlens.backends.torch._tl import is_decorated_function
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch


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


def test_clean_capture_is_never_rescued() -> None:
    """No escape signal -> exactly one mode-free run, no disclosure."""

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(v))

    trace = tl.trace(Model(), torch.randn(3, 4))
    assert trace.rescue_rerun is None
    assert trace.capture_verification_reason != "mode_rescue_rerun"


def test_rescue_restores_rng_to_capture_entry() -> None:
    """The rescue run sees EXACTLY the capture-entry RNG state.

    Driven directly against the driver: a capture stub draws from torch RNG
    on every invocation and reports an escape signal on the first. The
    driver must restore RNG before the re-run, so both invocations observe
    the identical draw (the rescue replays the primary's randomness).
    """

    from torchlens.backends.torch.rescue import capture_with_rescue

    draws: list[float] = []

    def run_capture() -> Any:
        draws.append(torch.rand(1).item())
        return types.SimpleNamespace(
            escape_diagnostics=[],
            _had_unattributed_tensor_args=len(draws) == 1,
            ops=[],
        )

    capture_with_rescue(run_capture)
    assert len(draws) == 2
    assert draws[0] == draws[1]


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
    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "escape_rescue_unrecovered"
    assert trace.rescue_rerun is not None
    assert trace.rescue_rerun["recovered"] is False
    assert trace.rescue_rerun["skipped_reason"] == "rescue_ineligible"
    assert trace.rescue_rerun["forward_runs"] == 1


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
    assert trace.capture_verified is False
    assert trace.capture_verification_reason == "escape_rescue_unrecovered"
    assert trace.rescue_rerun["skipped_reason"] == "rescue_ineligible"


def test_ineligible_clean_capture_stays_clean() -> None:
    """eligible=False with NO signal must not stamp any rescue field."""

    from torchlens.backends.torch.rescue import capture_with_rescue

    def run_capture() -> Any:
        return _stub_trace(["relu"], signal=False)

    trace = capture_with_rescue(run_capture, eligible=False)
    assert trace.capture_verification_reason is None
    assert not hasattr(trace, "rescue_rerun") or trace.rescue_rerun is None


def _stub_trace(
    op_names: list[str],
    *,
    signal: bool = False,
    modules: list[Any] | None = None,
    **extra: Any,
) -> Any:
    """Build a minimal trace stub for direct driver tests."""

    fields: dict[str, Any] = {
        "escape_diagnostics": [],
        "_had_unattributed_tensor_args": signal,
        "ops": [types.SimpleNamespace(func_name=name) for name in op_names],
        "modules": modules or [],
        "capture_verification_reason": None,
    }
    fields.update(extra)
    return types.SimpleNamespace(**fields)


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
    assert result.capture_verification_reason == "escape_rescue_unrecovered"
    info = result.rescue_rerun
    assert info["recovered"] is False
    assert "mha_fused" in info["lost_ops"]
    assert "matmul" in info["recovered_ops"]


def test_strict_superset_rescue_still_recovers() -> None:
    """R16-1 control: gains with zero losses stay a genuine recovery."""

    from torchlens.backends.torch.rescue import capture_with_rescue

    runs: list[str] = []
    primary = _stub_trace(["relu"], signal=True)
    rescued = _stub_trace(["relu", "cos"])

    def run_capture() -> Any:
        runs.append("run")
        return primary if len(runs) == 1 else rescued

    result = capture_with_rescue(run_capture)
    assert result is rescued
    assert result.capture_verification_reason == "mode_rescue_rerun"
    assert result.rescue_rerun["recovered_ops"] == ("cos",)


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
    info = result.rescue_rerun
    assert info["skipped_reason"] == "buffer_writes_double_forward"
    assert info["forward_runs"] == 1
    assert result.capture_verification_reason == "escape_rescue_unrecovered"


def test_train_mode_batchnorm_capture_skips_rescue_end_to_end(raw_cos: Any) -> None:
    """R16-2 end-to-end: a train-mode BN model's counter increments ONCE."""

    wrap_torch()

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn = nn.BatchNorm1d(2)

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(raw_cos(self.bn(v)))

    model = Model()
    model.train()
    with pytest.warns(UserWarning):
        trace = tl.trace(model, torch.randn(4, 2))
    assert model.bn.num_batches_tracked.item() == 1, "the forward must not run twice"
    info = trace.rescue_rerun
    assert info is not None
    assert info["skipped_reason"] == "buffer_writes_double_forward"
    assert "cos" not in [op.func_name for op in trace.ops]


def test_provably_unchanged_buffer_journal_does_not_refuse_rescue() -> None:
    """Journal PRESENCE alone is not a write: value_changed=False rescues.

    Fused norm mutators journal unconditionally, so every EVAL-mode BN/IN/GN
    capture carries ``buffer_write_kind`` records with
    ``buffer_value_changed=False`` (bytes provably unchanged). Refusing on
    presence alone permanently disabled the rescue for the most common
    capture class; the refusal must key on an ACTUAL write.
    """

    from torchlens.backends.torch.rescue import capture_with_rescue

    runs: list[str] = []
    primary = _stub_trace(["relu"], signal=True)
    primary.ops.append(
        types.SimpleNamespace(
            func_name="none",
            buffer_write_kind="fused",
            buffer_value_changed=False,
            label_raw="buffer_3",
        )
    )
    rescued = _stub_trace(["relu", "cos"])
    rescued.ops.append(
        types.SimpleNamespace(
            func_name="none",
            buffer_write_kind="fused",
            buffer_value_changed=False,
            label_raw="buffer_3",
        )
    )

    def run_capture() -> Any:
        runs.append("run")
        return primary if len(runs) == 1 else rescued

    result = capture_with_rescue(run_capture)
    assert len(runs) == 2, "a provably-unchanged buffer journal must not refuse the re-run"
    assert result is rescued
    assert result.capture_verification_reason == "mode_rescue_rerun"


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
    info = trace.rescue_rerun
    assert info is not None
    assert info["recovered"] is True
    assert info["forward_runs"] == 2
    assert "cos" in [op.func_name for op in trace.ops]
    assert trace.capture_verification_reason == "mode_rescue_rerun"


def test_attribution_failed_rescue_refused_for_buffer_writing_primary() -> None:
    """R16-2 covers the attribution-failed trigger too: no double forward.

    The refusal used to run only in the completed-primary branch, so an
    output-attribution failure on a train-mode BN model re-ran the forward
    and double-mutated the buffers (num_batches_tracked == 2) undisclosed.
    """

    from torchlens._errors import OutputAttributionError
    from torchlens.backends.torch.rescue import capture_with_rescue

    runs: list[str] = []

    def run_capture() -> Any:
        runs.append("run")
        exc = OutputAttributionError("output could not be attributed")
        exc._torchlens_actual_buffer_writes = ("bn.num_batches_tracked",)
        raise exc

    with pytest.warns(UserWarning, match="double-apply"), pytest.raises(OutputAttributionError):
        capture_with_rescue(run_capture)
    assert len(runs) == 1, "the forward must run exactly once for a buffer-writing primary"


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
        tl.trace(model, torch.randn(4, 2))
    assert model.bn.num_batches_tracked.item() == 1, "the forward must not run twice"
    partial = getattr(exc_info.value, "partial_log", None)
    partial_trace = getattr(partial, "trace", None)
    assert partial_trace is not None
    info = partial_trace.rescue_rerun
    assert info is not None
    assert info["skipped_reason"] == "buffer_writes_double_forward"
    assert info["forward_runs"] == 1


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


def test_successful_attribution_rescue_suppresses_failure_advisory(raw_cos: Any) -> None:
    """A rescued attribution failure emits NO capture-attempt-failed advisory.

    The advisory tells the user diagnostics ride the exception
    (``exc.partial_log``) — untruthful when the rescue swallowed the failure
    and returned a trace. It is deferred while a rescue is possible and
    dropped on success; re-raising paths flush it (pinned by
    ``test_capture_failure_reporting``).
    """

    import warnings as warnings_module

    wrap_torch()

    class Model(nn.Module):
        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return raw_cos(torch.sigmoid(v))

    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        trace = tl.trace(Model(), torch.tensor([0.25, 0.5]))
    info = trace.rescue_rerun
    assert info is not None
    assert info["trigger"] == "output_attribution_failed"
    assert info["recovered"] is True
    advisories = [w for w in caught if "capture attempt failed" in str(w.message)]
    assert not advisories, [str(w.message) for w in advisories]


def test_stateless_primary_still_rescues() -> None:
    """R16-2 control: no buffer writes -> the re-run proceeds."""

    from torchlens.backends.torch.rescue import capture_with_rescue

    runs: list[str] = []
    primary = _stub_trace(["relu"], signal=True)
    rescued = _stub_trace(["relu", "cos"])

    def run_capture() -> Any:
        runs.append("run")
        return primary if len(runs) == 1 else rescued

    result = capture_with_rescue(run_capture)
    assert len(runs) == 2
    assert result is rescued


def test_recovered_rescue_never_clobbers_dynamo_verdict() -> None:
    """R16-3: dynamo_region_not_logged outranks mode_rescue_rerun in _mark."""

    from torchlens.backends.torch.rescue import capture_with_rescue

    runs: list[str] = []
    primary = _stub_trace(["relu"], signal=True)
    rescued = _stub_trace(
        ["relu", "cos"],
        _raw_dynamo_region_detected=True,
        capture_verification_reason="dynamo_region_not_logged",
    )

    def run_capture() -> Any:
        runs.append("run")
        return primary if len(runs) == 1 else rescued

    result = capture_with_rescue(run_capture)
    assert result is rescued
    assert result.capture_verification_reason == "dynamo_region_not_logged"
    assert result.rescue_rerun is not None, "the rescue attempt must stay disclosed"


def test_escape_detector_diagnostic_triggers_rescue(raw_cos: Any) -> None:
    """The opt-in shadow detector's diagnostics are a rescue trigger too."""

    from torchlens._errors import TorchLensCaptureGapWarning

    try:
        wrap_torch(escape_detector="shadow")
        with pytest.warns(TorchLensCaptureGapWarning):
            trace = tl.trace(_stale_closure_model(raw_cos), torch.tensor([0.25, 0.5]))
        info = trace.rescue_rerun
        assert info is not None and info["recovered"] is True
        # REVIEWED FLIP (b3-fable R02-2 / b6-fable R16-3): the rescued run's
        # own shadow-detector report is a MORE specific verdict than the
        # generic mode_rescue_rerun stamp and is no longer clobbered by it;
        # the rerun stays disclosed through ``rescue_rerun`` above.
        assert trace.capture_verification_reason == "callable_escape_shadow_report"
        assert info["primary_escape_diagnostics"]
        assert "cos" in [op.func_name for op in trace.ops]
    finally:
        unwrap_torch()
        wrap_torch()


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
    assert trace.rescue_rerun is not None
    assert trace.rescue_rerun["recovered"] is False
    assert trace.rescue_rerun["skipped_reason"] == "state_writes_double_forward_undone"
    assert trace.rescue_rerun["forward_runs"] == 2
    assert trace.capture_verified is False


def test_rescue_never_double_applies_journal_invisible_param_writes(raw_cos: Any) -> None:
    """A write the buffer-write journal cannot see must not double-apply.

    R02-2 regression: the success-path guard read an EMPTY buffer-write
    journal as proof of no writes, but the journal is structurally blind to
    PARAMETER storage (its index skips ``nn.Parameter``), so a host write
    into a param during the forward left no record and the re-run
    double-applied it. The state snapshot audits the re-run by bytes instead
    of trusting journal emptiness.
    """

    class HiddenParamWriter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            # Journal-invisible write: raw host write into PARAMETER storage.
            self.lin.weight.detach().numpy()[0, 0] += 1.0
            return torch.relu(raw_cos(self.lin(v)))

    model = HiddenParamWriter()
    model.eval()
    baseline = model.lin.weight.detach().clone()

    with pytest.warns(UserWarning, match="wrote model state"):
        trace = tl.trace(model, torch.randn(3, 4))

    expected = baseline.clone()
    expected[0, 0] += 1.0
    assert torch.equal(model.lin.weight.detach(), expected)
    assert trace.rescue_rerun is not None
    assert trace.rescue_rerun["skipped_reason"] == "state_writes_double_forward_undone"


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
        trace = tl.trace(model, torch.randn(3, 4))

    assert float(model.count) == 1.0
    assert trace.rescue_rerun is not None
    assert trace.rescue_rerun["skipped_reason"] == "buffer_writes_double_forward"
    assert trace.rescue_rerun["forward_runs"] == 1


def test_minted_source_nodes_do_not_refuse_a_perfect_rescue() -> None:
    """R16-1 follow-up: bookkeeping ``none`` nodes are outside the oracle.

    When the primary minted an ``internalsource`` orphan (func_name ``none``)
    for the escaped op's output and the rescue captured the real op instead,
    the rescued trace read one ``none`` short -- ``lost_ops=('none',)`` -- and
    the two-sided oracle refused a PERFECT rescue, leaving the user the broken
    primary. Functionless source nodes must not count as losses.
    """

    from torchlens.backends.torch.rescue import capture_with_rescue

    primary = _stub_trace(["none", "linear", "relu"], signal=True)
    rescued = _stub_trace(["cos", "linear", "relu"])
    traces = iter([primary, rescued])

    result = capture_with_rescue(lambda: next(traces))

    assert result is rescued
    assert result.capture_verification_reason == "mode_rescue_rerun"
    assert result.rescue_rerun["recovered"] is True
    assert result.rescue_rerun["recovered_ops"] == ("cos",)
    assert result.rescue_rerun["lost_ops"] == ()


def test_module_consumed_stale_ref_is_disclosed_and_rescued(raw_cos: Any) -> None:
    """R16: module-entry adoption must not LAUNDER a stale-ref escape.

    A stale pre-wrap reference whose output is first consumed by a MODULE
    (``self.lin(stale_fn(x))`` -- the overwhelmingly common shape) used to be
    adopted as a clean ``internalsource`` node: op absent, zero warnings,
    ``rescue_rerun`` None -- indistinguishable from a clean capture, while the
    identical escape consumed by a wrapped FUNCTION warned and rescued.
    Disclosure must not be consumption-order-dependent.
    """

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return self.lin(raw_cos(v))

    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = tl.trace(Model(), torch.randn(3, 4))

    assert "cos" in [op.func_name for op in trace.ops]
    assert trace.rescue_rerun is not None
    assert trace.rescue_rerun["recovered"] is True
    assert trace.capture_verification_reason == "mode_rescue_rerun"


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

    from torchlens.backends.torch.model_prep import _collect_model_owned_tensor_ids

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


def test_intervened_capture_never_reruns_user_callables(raw_cos: Any) -> None:
    """A rescue re-run would invoke user intervention callables a SECOND time.

    Side-effecting user callables (counters, file writes, externally-held
    state) double-applied invisibly on the rescue path; interventions now
    refuse the re-run fail-closed, exactly like streaming/halt captures.
    """

    calls = {"intervene": 0}

    def counting_transform(value: torch.Tensor, *, hook: Any) -> torch.Tensor:
        calls["intervene"] += 1
        return value * 0.5

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return torch.relu(raw_cos(self.lin(v)))

    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = tl.trace(
            Model(),
            torch.randn(3, 4),
            intervene=tl.when(tl.func("linear"), counting_transform),
        )

    assert calls["intervene"] == 1
    # R16-1: the refused re-run is DISCLOSED, never silent clean fields.
    assert trace.rescue_rerun["skipped_reason"] == "rescue_ineligible"
    assert trace.rescue_rerun["forward_runs"] == 1
    assert trace.capture_verification_reason == "escape_rescue_unrecovered"


def test_transform_callables_never_double_fire_on_rescue(raw_cos: Any) -> None:
    """b6-opus-R16-1 reopen: the transform channels must gate rescue eligibility.

    The eligibility gate refused ``intervene=``/hooks for double-invocation
    side effects but let ``activation_transform``/``grad_transform``/
    ``output_transform`` through -- and they DID run twice on a recovered
    rescue (measured: activation 2x per saved op, output once per forward
    x2). A transform that appends to a list, writes a file, or accumulates
    statistics silently double-applied. All in-capture user-callable channels
    now refuse the re-run fail-closed, exactly like ``intervene=``.
    """

    calls = {"act": 0, "out": 0}

    def counting_act(value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        calls["act"] += 1
        return value

    def counting_out(value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        calls["out"] += 1
        return value

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return self.lin(raw_cos(v))

    wrap_torch()

    # Control: identical capture shape with no escape fixes the single-run
    # activation-transform count.
    class Control(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, v: torch.Tensor) -> torch.Tensor:
            return self.lin(torch.cos(v))

    tl.trace(
        Control(),
        torch.randn(3, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all", output_transform=counting_out),
        save=tl.options.SaveOptions(activation_transform=counting_act),
    )
    control_act, control_out = calls["act"], calls["out"]
    assert control_out == 1
    calls["act"] = calls["out"] = 0

    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = tl.trace(
            Model(),
            torch.randn(3, 4),
            grad_transform=lambda value, **kwargs: value,
            capture=tl.options.CaptureOptions(layers_to_save="all", output_transform=counting_out),
            save=tl.options.SaveOptions(activation_transform=counting_act),
        )

    assert trace.rescue_rerun["skipped_reason"] == "rescue_ineligible", (
        "a capture with in-capture transform callables must refuse the rescue "
        "re-run (side effects double-apply) and disclose the refusal"
    )
    assert trace.rescue_rerun["forward_runs"] == 1
    assert calls["out"] == 1, f"output_transform fired {calls['out']}x (expected once)"
    # The escaped cos op is invisible to the primary, so the escape capture
    # saves at most the control's op count; strictly more means a second run.
    assert calls["act"] <= control_act, (
        f"activation_transform fired {calls['act']}x vs {control_act}x on the "
        "no-escape control: the rescue re-ran the user callable"
    )


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
        capture_verification_reason="owner_thread_tripwire_changed",
    )
    traces = iter([primary, rescued])

    result = capture_with_rescue(lambda: next(traces))

    assert result is rescued
    assert result.capture_verification_reason == "owner_thread_tripwire_changed"
    assert result.rescue_rerun["recovered"] is True


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
    assert trace.capture_verification_reason == "escape_rescue_unrecovered"
    assert "rescue died mid-forward" in trace.rescue_rerun["rescue_error"]


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


def test_successful_rescue_rerun_warns_about_the_double_forward() -> None:
    """R67: the success path was the ONLY silent rescue outcome.

    Fail-before: a recovered rescue returned capture_verified=False with zero
    warnings while the user's forward had executed twice.
    """

    from torchlens._errors import TorchLensCaptureGapWarning
    from torchlens.backends.torch.rescue import capture_with_rescue

    runs: list[str] = []
    primary = _stub_trace(["relu"], signal=True)
    rescued = _stub_trace(["relu", "cos"])

    def run_capture() -> Any:
        runs.append("run")
        return primary if len(runs) == 1 else rescued

    with pytest.warns(TorchLensCaptureGapWarning, match="executed TWICE"):
        result = capture_with_rescue(run_capture)
    assert result is rescued


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


def test_nan_state_does_not_false_flag_the_state_audit() -> None:
    """R16: IEEE ``torch.equal`` returns False for NaN==NaN, so a model
    legitimately holding a NaN parameter/buffer was falsely accused of a
    double-applied state write by the untouched-model audit -- the rescue
    trace was discarded and the user warned about writes that never happened.
    The audit must be NaN-safe (byte-exact, never tolerant)."""

    from torchlens.backends.torch.rescue import (
        _restore_changed_state,
        _snapshot_declared_state,
    )

    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    model.register_buffer("nan_buffer", torch.tensor([float("nan"), 1.0]))

    snapshot = _snapshot_declared_state(model)
    assert snapshot is not None
    changed = _restore_changed_state(model, snapshot)
    assert changed == (), (
        "nothing wrote model state between snapshot and audit, yet the audit "
        f"reported changes: {changed!r} (NaN-blind torch.equal compare)"
    )


def test_nan_holding_model_still_flags_a_real_state_write() -> None:
    """Positive control for the NaN-safe audit: a genuine write on a model
    that also holds NaN state is still detected and restored byte-exactly."""

    from torchlens.backends.torch.rescue import (
        _restore_changed_state,
        _snapshot_declared_state,
    )

    model = nn.Linear(2, 2)
    model.register_buffer("nan_buffer", torch.tensor([float("nan"), 1.0]))

    snapshot = _snapshot_declared_state(model)
    assert snapshot is not None
    with torch.no_grad():
        model.nan_buffer[1] = 7.0
        model.weight.add_(1.0)
    changed = _restore_changed_state(model, snapshot)
    assert set(changed) == {"buffer:nan_buffer", "param:weight"}
    assert torch.equal(model.weight, snapshot["param:weight"]), "the flagged write was not restored"
    assert model.nan_buffer[1].item() == 1.0


def test_closure_held_original_is_rebound_without_rescue() -> None:
    """The stale-closure case with the bare original, as the rescue tests first held it.

    Capture preparation rebinds the closure cell to the wrapper, so ``cos``
    is captured in one forward with no rescue, and the cell holds the
    original again afterwards.
    """

    unwrap_torch()
    raw = torch.cos
    wrap_torch()
    model = CountedRoot(_stale_closure_model(raw))
    trace = tl.trace(model, torch.tensor([0.25, 0.5]))

    assert model.calls == [1]
    assert trace.rescue_rerun is None
    assert [op.func_name for op in trace.ops].count("cos") == 1
