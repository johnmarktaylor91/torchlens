"""Regression tests for nested trace refusal."""

import pytest
import torch
from torch import nn

import torchlens as tl


class _NestedTraceModel(nn.Module):
    """Module that attempts to start a nested TorchLens trace."""

    def __init__(self) -> None:
        """Initialize the inner module used by the nested trace."""

        super().__init__()
        self.inner = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attempt a nested trace from inside the active forward pass."""

        tl.trace(self.inner, x)
        return x


def test_nested_trace_raises_typed_error_with_active_model_name() -> None:
    """Nested tracing should raise the public typed re-entrancy error."""

    with pytest.raises(tl.errors.ReentrantTraceError, match="NestedTraceModel") as exc_info:
        tl.trace(_NestedTraceModel(), torch.ones(1, 2))

    assert exc_info.value.fields["code"] == "reentrant_trace"
    assert exc_info.value.fields["remedy"]
    assert exc_info.value.fields["active_model"] is not None
    assert "Remedy:" in str(exc_info.value)


def test_reentrant_trace_error_is_exported() -> None:
    """The re-entrancy exception is public at its torchlens.errors home."""

    assert tl.errors.ReentrantTraceError.__name__ == "ReentrantTraceError"
    assert "ReentrantTraceError" in tl.errors.__all__
    assert "ReentrantTraceError" not in tl.__all__


def test_reentrant_trace_error_joins_the_taxonomy_and_keeps_runtime_error() -> None:
    """The refusal is a taxonomy member without breaking historical handlers.

    It must be catchable as ``tl.errors.CaptureError`` (typed taxonomy) AND as
    ``RuntimeError`` (its historical builtin lineage), and it must resolve
    through the registered ``torchlens.errors`` surface.
    """

    from torchlens import errors

    assert issubclass(tl.errors.ReentrantTraceError, errors.CaptureError)
    assert issubclass(tl.errors.ReentrantTraceError, RuntimeError)
    assert errors.ReentrantTraceError is tl.errors.ReentrantTraceError

    with pytest.raises(RuntimeError):
        tl.trace(_NestedTraceModel(), torch.ones(1, 2))
    with pytest.raises(errors.CaptureError):
        tl.trace(_NestedTraceModel(), torch.ones(1, 2))


def test_nested_trace_in_pre_admission_reserved_window_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """grind-r6 b7 R55 (sol HIGH): the pre-admission window is not a hole.

    User code running between the outer capture's reservation claim and
    ``active_logging`` (model prep, compiled-capture prep, input setup) used
    to be able to start a nested public ``tl.trace()`` that COMPLETED both
    captures (sol probe ``OUTER_OK``) -- the same-thread reservation
    passthrough carried no owner token. The nested call must refuse typed
    while the outer capture still completes cleanly.
    """

    import torchlens.capture.trace as capture_trace

    real_prep = capture_trace.prepare_compiled_capture
    probe: dict[str, object] = {}

    def hostile_prep(model: nn.Module) -> object:
        """Attempt a nested public capture from inside the reserved window."""

        if "fired" not in probe:
            probe["fired"] = True
            try:
                tl.trace(nn.Linear(2, 2), torch.ones(1, 2))
                probe["nested"] = "OUTER_OK"
            except tl.errors.ReentrantTraceError:
                probe["nested"] = "refused"
        return real_prep(model)

    monkeypatch.setattr(capture_trace, "prepare_compiled_capture", hostile_prep)
    outer = tl.trace(nn.ReLU(), torch.ones(1, 2))
    assert probe.get("nested") == "refused", f"nested capture outcome: {probe.get('nested')}"
    assert any(op.func_name == "relu" for op in outer.compute_ops)

    # The refused nested entry must not have poisoned admission: a fresh
    # capture afterwards works.
    monkeypatch.setattr(capture_trace, "prepare_compiled_capture", real_prep)
    recovered = tl.trace(nn.ReLU(), torch.ones(1, 2))
    assert any(op.func_name == "relu" for op in recovered.compute_ops)
