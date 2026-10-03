"""Direct liveness killers for ``_check_postprocess_contract`` (b9-sol R74-1).

The postprocess contract checker was a surviving mutant: a return-None disarm
stayed green across the mutation arming suite because the driver only ran
validation files, and the planted enforcement tests in
``tests/test_postprocess_dag.py`` need a real armed capture. These tests call
the checker DIRECTLY with synthetic violating inputs, so neutering its body is
killed in every ordinary pytest run — no audit env vars, no capture, no driver.

The checker is assert-based by design (see the executor header note), so these
tests carry ``requires_assertions`` and skip under ``python -O``.
"""

from __future__ import annotations

import pytest

from torchlens._trace_core.op_store import StepAuditResult
from torchlens.postprocess import _check_postprocess_contract

pytestmark = pytest.mark.requires_assertions


def _audit(
    *,
    written_columns: set[str] | None = None,
    released_rows: int = 0,
    read_columns: set[str] | None = None,
    clone_read_columns: set[str] | None = None,
    effective_write_columns: set[str] | None = None,
) -> StepAuditResult:
    """Build a synthetic closed-window audit result.

    Parameters
    ----------
    written_columns, released_rows, read_columns, clone_read_columns, effective_write_columns:
        Field overrides; everything defaults to the empty observation.

    Returns
    -------
    StepAuditResult
        Synthetic audit observation for one step window.
    """

    return StepAuditResult(
        written_columns=written_columns or set(),
        released_rows=released_rows,
        read_columns=read_columns or set(),
        clone_read_columns=clone_read_columns or set(),
        effective_write_columns=effective_write_columns or set(),
    )


def test_unknown_step_contract_is_rejected() -> None:
    """A step id with no registered contract raises, never silently passes."""

    with pytest.raises(AssertionError, match="Unknown postprocess step contract"):
        _check_postprocess_contract(object(), "not-a-real-step", None)


def test_undeclared_write_is_rejected() -> None:
    """An observed write outside the step's declared write set raises.

    Step 2 declares exactly two write columns; a synthetic foreign column in
    the window's observations must trip the undeclared-write assertion.
    """

    audit = _audit(written_columns={"tl_totally_undeclared_column"})
    with pytest.raises(AssertionError, match="wrote undeclared op-store columns"):
        _check_postprocess_contract(object(), "2", audit)


def test_unsanctioned_row_release_is_rejected() -> None:
    """Whole-row releases on a step without a 'deletes' sanction raise."""

    audit = _audit(released_rows=3)
    with pytest.raises(AssertionError, match="without a 'deletes' row_effects sanction"):
        _check_postprocess_contract(object(), "2", audit)


def test_undeclared_read_is_rejected_in_enforce_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the read audit in enforce mode, an undeclared read raises."""

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_READ_AUDIT", "enforce")
    audit = _audit(read_columns={"tl_totally_undeclared_column"})
    with pytest.raises(AssertionError, match="read undeclared op-store columns"):
        _check_postprocess_contract(object(), "2", audit)


def test_unsanctioned_clone_read_is_rejected_in_enforce_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Row-clone reads on a step without a 'creates' sanction raise."""

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_READ_AUDIT", "enforce")
    audit = _audit(clone_read_columns={"out"})
    with pytest.raises(AssertionError, match="row-clone reads"):
        _check_postprocess_contract(object(), "2", audit)


def test_step_postcondition_runs_on_none_audit() -> None:
    """Postconditions fire even without a window (step-0 prologue path).

    A stub trace with empty ``output_layers`` must trip step 1's
    output-registration postcondition when the audit result is ``None``.
    """

    class _Stub:
        output_layers: list[str] = []

    with pytest.raises(AssertionError, match="must register output layers"):
        _check_postprocess_contract(_Stub(), "1", None)


def test_open_window_past_step_20_is_rejected() -> None:
    """`_assert_no_open_window` trips on a registered open audit window.

    Direct liveness killer (b9 round 5: the whole-function neuter survived
    the arming suite — nothing exercised the raising branch directly). A
    fake core whose op store id sits in ``_AUDIT_COLLECTORS`` must raise;
    an unregistered store must pass.
    """

    from types import SimpleNamespace

    from torchlens._trace_core.op_store import _AUDIT_COLLECTORS
    from torchlens.postprocess import _assert_no_open_window

    store = object()
    fake_trace = SimpleNamespace()
    fake_trace.__dict__["_trace_core"] = SimpleNamespace(ops=store)

    _assert_no_open_window(fake_trace)  # unregistered: must not raise
    _AUDIT_COLLECTORS[id(store)] = set()
    try:
        with pytest.raises(AssertionError, match="audit window open past step 20"):
            _assert_no_open_window(fake_trace)
    finally:
        _AUDIT_COLLECTORS.pop(id(store), None)


def test_step0_prologue_seam_checks_the_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_assert_postprocess_contract` closes the window AND checks the step.

    Direct liveness killer (b9 round 5 survivor): with assertions armed, the
    step-0 prologue seam must route into the contract checker — an unknown
    step id observed through the seam has to raise exactly like a direct
    checker call, so a whole-function neuter cannot stay green.
    """

    from types import SimpleNamespace

    import torchlens.postprocess as postprocess

    monkeypatch.setattr(postprocess, "_postprocess_assertions_enabled", lambda: True)
    fake_trace = SimpleNamespace()
    fake_trace.__dict__["_trace_core"] = None
    with pytest.raises(AssertionError, match="Unknown postprocess step contract"):
        postprocess._assert_postprocess_contract(fake_trace, "not-a-real-step")


def test_step_13_clears_cuda_cache_exactly_once_when_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Step 13 empties the CUDA cache exactly once on an armed capture.

    r7 R74 (sol HIGH): the ``executor#_run_step_13`` return-None mutant
    survived the real bounded campaign with zero killers — no suite test
    armed both gates (CUDA availability AND capture-touched-CUDA), so
    neutering the step's body was invisible. This drives a REAL capture
    with both predicates patched true and a recording ``empty_cache``, so
    both the body-removed and predicate-false mutants die here.
    """

    import torch
    from torch import nn

    import torchlens as tl
    import torchlens.postprocess as pp
    from torchlens.utils import tensor_utils

    calls: list[bool] = []
    monkeypatch.setattr(pp, "_is_cuda_available", lambda: True)
    monkeypatch.setattr(tensor_utils, "capture_touched_cuda", lambda _trace: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(True))

    trace = tl.trace(nn.Sequential(nn.Linear(3, 3), nn.ReLU()), torch.randn(2, 3))
    try:
        assert len(calls) == 1, (
            f"armed step 13 must clear the CUDA cache exactly once, saw {len(calls)}"
        )
    finally:
        trace.cleanup()


@pytest.mark.parametrize(
    ("cuda_available", "touched_cuda"),
    [(False, True), (True, False)],
    ids=["cuda-unavailable", "cpu-only-capture"],
)
def test_step_13_never_flushes_the_allocator_ungated(
    monkeypatch: pytest.MonkeyPatch, cuda_available: bool, touched_cuda: bool
) -> None:
    """Step 13 stays silent unless BOTH gates agree (pins the R36-3 gating).

    The CPU negative case: a CPU-only trace inside a GPU training loop must
    never flush the caller's allocator, and an unavailable CUDA runtime must
    never be poked at all.
    """

    import torch
    from torch import nn

    import torchlens as tl
    import torchlens.postprocess as pp
    from torchlens.utils import tensor_utils

    calls: list[bool] = []
    monkeypatch.setattr(pp, "_is_cuda_available", lambda: cuda_available)
    monkeypatch.setattr(tensor_utils, "capture_touched_cuda", lambda _trace: touched_cuda)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(True))

    trace = tl.trace(nn.Sequential(nn.Linear(3, 3), nn.ReLU()), torch.randn(2, 3))
    try:
        assert not calls, "step 13 flushed the allocator with a gate false"
    finally:
        trace.cleanup()


def test_step_20_releases_every_live_param_ref() -> None:
    """Postprocess ends with no live parameter reference in any ParamLog.

    r7 R74 (opus): the ``executor#_run_step_20`` return-None mutant's only
    killer was an incidental failure-propagation test — nothing PURPOSEFULLY
    asserted the step's contract. Step 20 runs ``release_param_refs``, so a
    finished capture must hold ``_param_ref is None`` (released) on every
    ParamLog; keeping live refs pins the model's parameters against GC.
    """

    import torch
    from torch import nn

    import torchlens as tl

    trace = tl.trace(nn.Sequential(nn.Linear(3, 3), nn.ReLU()), torch.randn(2, 3))
    try:
        param_logs = list(trace.param_logs.values())
        assert param_logs, "expected captured parameters"
        offenders = [
            log.param_address
            for log in param_logs
            if log._param_ref is not None or not log._param_ref_released
        ]
        assert not offenders, f"finished capture still holds live parameter references: {offenders}"
    finally:
        trace.cleanup()
