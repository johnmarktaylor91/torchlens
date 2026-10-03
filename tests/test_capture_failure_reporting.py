"""Failure-arm reporting: routed warning, partial_log note, empty_cache gate.

Pins the B8-35/B8-44/B8-45/R16-4b fixes in ``backends/torch/backend.py``:

- The historical unconditional stdout banner ("Feature extraction failed;
  returning model and environment to normal") was factually false on
  rescue-recovered captures and supported return_partial flows, corrupted
  machine-readable stdout, and was unfilterable. It is now ONE accurate
  routed ``RuntimeWarning`` naming the failure and the recovery surface.
- The SUCCESS path of partial-log attachment adds an ``add_note`` telling the
  user ``exc.partial_log`` exists (previously only the two attachment-FAILURE
  arms carried notes).
- ``cleanup_forward_memory`` clears the CUDA allocator cache only when the
  capture actually touched CUDA (the capture-touched-CUDA predicate), never
  unconditionally.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl


class _BoomError(RuntimeError):
    pass


class _FailingModel(nn.Module):
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        raise _BoomError("forward exploded")


def test_failed_capture_emits_routed_warning_not_stdout_banner(capsys):
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(_BoomError):
            tl.trace(_FailingModel(), torch.randn(1, 2))
    out = capsys.readouterr().out
    assert "Feature extraction failed" not in out, "stdout banner must be gone"
    routed = [
        record
        for record in records
        if issubclass(record.category, RuntimeWarning)
        and "capture attempt failed" in str(record.message)
    ]
    assert len(routed) == 1, [str(record.message) for record in records]
    assert "_BoomError" in str(routed[0].message)
    assert "partial_log" in str(routed[0].message)
    # The advisory rides a dedicated RuntimeWarning subclass so the rescue
    # driver can defer it while a rescue re-run may still swallow the failure;
    # user RuntimeWarning filters keep matching.
    from torchlens.backends.torch.rescue import CaptureAttemptFailedWarning

    assert routed[0].category is CaptureAttemptFailedWarning


@pytest.mark.skipif(
    not hasattr(BaseException, "add_note"),
    reason="BaseException.add_note requires Python 3.11+",
)
def test_failed_capture_success_path_notes_partial_log():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(_BoomError) as exc_info:
            tl.trace(_FailingModel(), torch.randn(1, 2))
    exc = exc_info.value
    assert getattr(exc, "partial_log", None) is not None
    notes = getattr(exc, "__notes__", [])
    assert any("partial_log" in note for note in notes), notes


def test_failed_capture_advisory_never_replaces_user_exception_under_error_filter():
    """b3-opus-R07-1: a warnings-as-error filter raises the terminal advisory
    AT THE WARN SITE, replacing the user's real forward exception (and killing
    the ``exc.partial_log`` recovery the advisory's own text advertises). The
    user's exception must propagate; the advisory degrades to a note."""
    from torchlens.backends.torch.rescue import CaptureAttemptFailedWarning

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.filterwarnings("error", category=CaptureAttemptFailedWarning)
        with pytest.raises(_BoomError) as exc_info:
            tl.trace(_FailingModel(), torch.randn(1, 2))
    exc = exc_info.value
    assert getattr(exc, "partial_log", None) is not None
    if hasattr(BaseException, "add_note"):
        notes = getattr(exc, "__notes__", [])
        assert any("advisory" in note for note in notes), notes


def test_partial_construction_failure_warning_never_replaces_user_exception(monkeypatch):
    """Sibling sweep of b3-opus-R07-1: the construction-failure RuntimeWarning
    inside the partial-attachment arm has the same as-error filter hazard."""
    from torchlens.partial import PartialTrace

    def _boom_from_trace(*args, **kwargs):
        raise ValueError("construction exploded")

    monkeypatch.setattr(PartialTrace, "from_trace", classmethod(_boom_from_trace))
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.filterwarnings("error", category=RuntimeWarning)
        with pytest.raises(_BoomError):
            tl.trace(_FailingModel(), torch.randn(1, 2))


def test_cleanup_forward_memory_gated_on_capture_touched_cuda(monkeypatch):
    from torchlens.backends.torch import backend as backend_module
    from torchlens.backends.torch.backend import TorchBackend

    calls: list[str] = []
    monkeypatch.setattr(backend_module, "_is_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("emptied"))

    cpu_trace = tl.trace(nn.Linear(2, 2), torch.randn(1, 2))
    TorchBackend().cleanup_forward_memory(cpu_trace)
    assert calls == [], "a CPU-only capture must never clear the CUDA allocator cache"


def test_cleanup_double_fault_still_raises_the_primary_error(monkeypatch):
    """grind-r6 b1 R06 (sol MED, probe): the primary user error propagates.

    When ``backend.cleanup_failed_forward_session`` itself raised while
    handling an ordinary forward failure, the SECONDARY cleanup exception
    escaped instead of ``raise e``: the settled CaptureOutcome named the
    primary while the caller caught an unrelated error with no partial_log.
    The secondary now rides the primary as a note (or warning on 3.10),
    mirroring the interrupt arm.
    """

    from torchlens.backends.torch.backend import TorchBackend

    class _FailingModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(x)
            raise ValueError("primary-user-error")

    real_cleanup = TorchBackend.cleanup_failed_forward_session

    def raising_cleanup(self, session, model_tuple, exc):
        real_cleanup(self, session, model_tuple, exc)
        raise KeyError("secondary-cleanup-error")

    monkeypatch.setattr(TorchBackend, "cleanup_failed_forward_session", raising_cleanup)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="primary-user-error") as excinfo:
            tl.trace(_FailingModel(), torch.ones(2))
    exc = excinfo.value
    notes = getattr(exc, "__notes__", [])
    noted = any("secondary-cleanup-error" in note for note in notes)
    assert noted or not hasattr(exc, "add_note"), (
        f"cleanup double-fault not disclosed on the primary: notes={notes}"
    )
    assert getattr(exc, "partial_log", None) is not None
