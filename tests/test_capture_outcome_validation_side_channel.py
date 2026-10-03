"""B1-04 (SF-39): validate-then-save must not hard-refuse.

``validate_forward_pass`` / ``validate_saved_outs`` are public methods on a
user-held Trace, and validation ENTRY unconditionally writes the failure side
channel: ``reset_validation_failure`` sets ``_last_validation_failure = None``
on every run (``validation/core.py``), and non-failing diagnostics accumulate
in ``_validation_diagnostics``. Neither name had a ``PORTABLE_STATE_SPEC`` /
``FIELD_POLICY`` row, so the portable scrub's undeclared-field audit rejected
them and a plain validate-then-save sequence died with

    TorchLensIOError: Trace._last_validation_failure is missing from
    PORTABLE_STATE_SPEC

Both are session-time diagnostics carried on the trace as a side channel --
the ``_fast_run_session`` class -- so they are declared ``FieldPolicy.DROP``.
Like ``_fast_run_session`` they stay in plain pickle (in-process round-trips
legitimately carry session state); the DROP row governs the portable artifact.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.field_policy import FieldPolicy
from torchlens.validation.diagnostics import (
    TRACE_DIAGNOSTICS_ATTR,
    TRACE_FAILURE_ATTR,
    ValidationDiagnostic,
    ValidationFailure,
    record_validation_diagnostic,
    record_validation_failure,
)

_SIDE_CHANNEL_ATTRS = (TRACE_FAILURE_ATTR, TRACE_DIAGNOSTICS_ATTR)


class _Small(nn.Module):
    """Two-op model with a real replayable graph."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return torch.relu(self.fc(x))


def _traced() -> tuple[tl.Trace, torch.Tensor]:
    model = _Small().eval()
    x = torch.ones(1, 3)
    ground_truth = model(x)
    return tl.trace(model, x), ground_truth


def test_both_side_channel_attrs_are_declared_drop() -> None:
    """The declaration is the fix; the names are session-time, not portable."""

    policy = tl.Trace.FIELD_POLICY
    for name in _SIDE_CHANNEL_ATTRS:
        assert name in policy, name
        assert policy[name].portable_policy is FieldPolicy.DROP, name


def test_validate_then_save_round_trips(tmp_path) -> None:
    """The reported reproduction: validate, then save, on one user-held trace."""

    trace, ground_truth = _traced()
    status = trace.validate_forward_pass([ground_truth])
    assert status is not None
    # Validation entry writes the side channel unconditionally, even on a pass.
    assert TRACE_FAILURE_ATTR in trace.__dict__
    path = tmp_path / "validated.tlspec"
    tl.save(trace, str(path))
    loaded = tl.load(str(path))
    # DROP scrubs the VALUE to None (the key stays, like every other DROP
    # row); no session-time diagnostic reaches the portable artifact.
    for name in _SIDE_CHANNEL_ATTRS:
        assert loaded.__dict__.get(name) is None, name


def test_save_after_a_recorded_failure_payload_round_trips(tmp_path) -> None:
    """The failing branch, not just the ``None`` reset.

    A recorded ``ValidationFailure`` / ``ValidationDiagnostic`` carries a
    structured ``extra`` payload; the scrub must drop the whole record rather
    than try to make it portable.
    """

    trace, _ = _traced()
    record_validation_failure(
        trace,
        ValidationFailure(check="probe_check", message="probe message", extra={"op": "relu_1_1"}),
    )
    record_validation_diagnostic(
        trace,
        ValidationDiagnostic(check="probe_diag", message="probe diag", extra={"n": 1}),
    )
    assert isinstance(trace.__dict__[TRACE_FAILURE_ATTR], ValidationFailure)
    assert trace.__dict__[TRACE_DIAGNOSTICS_ATTR]
    path = tmp_path / "failed_validation.tlspec"
    tl.save(trace, str(path))
    loaded = tl.load(str(path))
    # The whole record is gone -- not a partially-portable ValidationFailure.
    for name in _SIDE_CHANNEL_ATTRS:
        assert loaded.__dict__.get(name) is None, name
    assert not any(
        isinstance(value, (ValidationFailure, ValidationDiagnostic))
        for value in vars(loaded).values()
    )


def test_repeated_validate_save_cycles_stay_saveable(tmp_path) -> None:
    """Validation is re-entrant against saving; no one-shot escape."""

    trace, ground_truth = _traced()
    for index in range(3):
        trace.validate_forward_pass([ground_truth])
        tl.save(trace, str(tmp_path / f"cycle_{index}.tlspec"))


def test_side_channel_survives_plain_pickle_like_fast_run_session() -> None:
    """DROP governs the portable artifact, not in-process pickle.

    Pinned deliberately: the sibling session-time DROP fields
    (``_fast_run_session``, ``save_budget``, ``measure_python_peak_memory``)
    behave the same way, and quietly diverging here would make the DROP class
    mean two different things.
    """

    import pickle

    trace, _ = _traced()
    record_validation_failure(trace, ValidationFailure(check="probe_check"))
    restored = pickle.loads(pickle.dumps(trace))
    assert TRACE_FAILURE_ATTR in restored.__dict__


@pytest.mark.smoke
def test_validate_fail_emits_one_summarizing_warning() -> None:
    """R67: a validate() FAIL points at the structured diagnosis.

    Fail-before: a planted ground-truth mismatch returned a bare silent False
    while last_validation_failure() held the full diagnosis unreferenced.
    """

    import warnings

    import torch
    from torch import nn

    import torchlens as tl
    import torchlens.user_funcs as user_funcs_module
    from torchlens.errors import TorchLensWarning

    real_validate = user_funcs_module.validate_forward_pass
    user_funcs_module.validate_forward_pass = lambda *args, **kwargs: False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            passed = tl.validate(nn.Linear(3, 3), torch.ones(2, 3), scope="forward")
    finally:
        user_funcs_module.validate_forward_pass = real_validate
    assert passed is False
    matching = [
        w
        for w in caught
        if issubclass(w.category, TorchLensWarning) and "tl.validate FAILED" in str(w.message)
    ]
    assert len(matching) == 1
    assert "last_validation_failure" in str(matching[0].message)
