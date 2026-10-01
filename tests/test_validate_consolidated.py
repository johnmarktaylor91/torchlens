"""Phase 5a tests for consolidated validation."""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest
import torch

import torchlens as tl
import torchlens.validation.consolidated as consolidated_validation
from torchlens.validation.consolidated import InterventionValidationReport


class TinyModel(torch.nn.Module):
    """Small deterministic model for validation tests."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear layer.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return self.linear(x)


@pytest.fixture
def model_and_input() -> tuple[TinyModel, torch.Tensor]:
    """Return a deterministic model/input pair.

    Returns
    -------
    tuple[TinyModel, torch.Tensor]
        Model and input tensor.
    """

    torch.manual_seed(0)
    model = TinyModel()
    torch.manual_seed(1)
    return model, torch.randn(1, 3)


def test_validate_forward_and_saved_scopes(model_and_input: tuple[TinyModel, torch.Tensor]) -> None:
    """Consolidated forward and saved scopes should return booleans."""

    model, x = model_and_input
    assert isinstance(tl.validate(model, x, scope="forward", random_seed=42), bool)
    assert isinstance(tl.validate(model, x, scope="saved", random_seed=42), bool)


def test_validate_backward_scope(model_and_input: tuple[TinyModel, torch.Tensor]) -> None:
    """Consolidated backward scope should call the backward validator."""

    model, x = model_and_input

    def loss_fn(output: torch.Tensor) -> torch.Tensor:
        """Return a scalar loss.

        Parameters
        ----------
        output:
            Model output.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """

        return output.sum()

    assert tl.validate(model, x, scope="backward", loss_fn=loss_fn)


def test_validate_backward_scope_detects_corrupted_captured_grad(
    model_and_input: tuple[TinyModel, torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consolidated default must inspect captured gradient payloads."""

    model, x = model_and_input
    original_log_backward = tl.Trace.log_backward

    def corrupt_captured_grad(trace: tl.Trace, *args: object, **kwargs: object) -> object:
        """Corrupt one captured module-output grad after backward logging."""

        result = original_log_backward(trace, *args, **kwargs)
        for op in trace.layer_list:
            grad = getattr(op, "grad", None)
            if not isinstance(grad, torch.Tensor) or not getattr(op, "modules", None):
                continue
            op.grads.for_pass(1).grad = torch.full_like(grad, 12345.0)
            break
        return result

    monkeypatch.setattr(tl.Trace, "log_backward", corrupt_captured_grad)

    assert not tl.validate(model, x, scope="backward", random_seed=42)


def test_scope_backward_with_metadata(model_and_input: tuple[TinyModel, torch.Tensor]) -> None:
    """Backward scope forwards default validate_metadata=True."""

    model, x = model_and_input
    with mock.patch(
        "torchlens.validation.consolidated.validate_backward_pass",
        wraps=consolidated_validation.validate_backward_pass,
    ) as validator:
        assert tl.validate(model, x, scope="backward")
    assert validator.call_args is not None
    assert validator.call_args.kwargs["validate_metadata"] is True


def test_scope_backward_with_metadata_forwards_false(
    model_and_input: tuple[TinyModel, torch.Tensor],
) -> None:
    """Backward scope accepts and forwards validate_metadata=False."""

    model, x = model_and_input
    with mock.patch(
        "torchlens.validation.consolidated.validate_backward_pass",
        wraps=consolidated_validation.validate_backward_pass,
    ) as validator:
        assert tl.validate(model, x, scope="backward", validate_metadata=False)
    assert validator.call_args is not None
    assert validator.call_args.kwargs["validate_metadata"] is False


def test_validate_scope_backward_random_seed(
    model_and_input: tuple[TinyModel, torch.Tensor],
) -> None:
    """Backward scope accepts and forwards random_seed."""

    model, x = model_and_input
    with mock.patch(
        "torchlens.validation.consolidated.validate_backward_pass",
        wraps=consolidated_validation.validate_backward_pass,
    ) as validator:
        assert tl.validate(model, x, scope="backward", random_seed=42)
    assert validator.call_args is not None
    assert validator.call_args.kwargs["random_seed"] == 42
    assert tl.validate(model, x, scope="backward", random_seed=42)


def test_validate_intervention_scope_returns_report(
    model_and_input: tuple[TinyModel, torch.Tensor],
) -> None:
    """Intervention scope returns a report with explicit unevaluated axes."""

    model, x = model_and_input
    report = tl.validate(model, x, scope="intervention", random_seed=42)
    assert isinstance(report, InterventionValidationReport)
    assert bool(report) is report.passed
    assert set(report.details) == {
        "invariance",
        "specificity",
        "completeness",
        "consistency",
        "locality",
    }
    assert report.invariance is True
    assert report.specificity is False
    assert report.completeness is False
    assert report.consistency is False
    assert report.locality is False
    assert bool(report) is False


def test_scope_keyword_visibility(model_and_input: tuple[TinyModel, torch.Tensor]) -> None:
    """Backward-only keywords should fail outside backward scope."""

    model, x = model_and_input
    with pytest.raises(TypeError, match="loss_fn only valid for scope='backward'"):
        tl.validate(model, x, scope="forward", loss_fn=lambda output: output.sum())


def test_legacy_validator_positionals(model_and_input: tuple[TinyModel, torch.Tensor]) -> None:
    """Deprecated wrappers should preserve legacy positional argument binding."""

    model, x = model_and_input

    def loss_fn(output: Any) -> torch.Tensor:
        """Return a scalar loss.

        Parameters
        ----------
        output:
            Model output.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """

        return output.sum()

    forward_result = tl.validation.validate_forward_pass(model, x, None, 42)
    saved_result = tl.validation.validate_forward_pass(model, x, None, 42)
    backward_result = tl.validation.validate_backward_pass(model, x, None, loss_fn)

    assert isinstance(forward_result, bool)
    assert isinstance(saved_result, bool)
    assert isinstance(backward_result, bool)


def test_validate_scope_end_trims_the_host_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """r7 b6-opus R33: validate must trim AFTER its comparison/report phase.

    The internal ``second_trace.cleanup()`` trims run BEFORE validate's
    highest-water phase reallocates, so ~180-225 MB of freed glibc arena
    stayed resident per call (a manual ``malloc_trim`` recovered it). The
    scope now trims once more on exit; this pins the call count -- the two
    internal cleanup trims plus the scope-end trim (red-capable: pre-fix
    exactly the two internal trims fire).
    """

    from torch import nn

    from torchlens.data_classes import cleanup as cleanup_mod

    calls = {"n": 0}
    real_trim = cleanup_mod._trim_host_allocator

    def counting_trim() -> None:
        calls["n"] += 1
        real_trim()

    monkeypatch.setattr(cleanup_mod, "_trim_host_allocator", counting_trim)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    assert tl.validate(model, torch.randn(2, 4), scope="forward", random_seed=0) is True
    assert calls["n"] >= 3, f"only {calls['n']} allocator trims ran; the scope-end trim is missing"


def test_validate_peak_instrumentation_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R33: the peak-memory probes are an env opt-in, not an always-on cost.

    The RSS/CUDA peak observations never feed the verdict, and the
    ``torch.cuda.is_available()`` probe can trigger driver init on some
    setups -- so a default validate call pays nothing and reports ``None``
    (mirroring the capture-side ``measure_python_peak_memory`` opt-in),
    while ``TORCHLENS_VALIDATE_PEAK_MEMORY=1`` populates the observations.
    """

    from torch import nn

    from torchlens.validation.diagnostics import last_validation_peak_memory

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())

    monkeypatch.delenv("TORCHLENS_VALIDATE_PEAK_MEMORY", raising=False)
    assert tl.validate(model, torch.randn(2, 4), scope="forward", random_seed=0) is True
    assert last_validation_peak_memory() is None, (
        "default validate populated peak observations without the opt-in"
    )

    monkeypatch.setenv("TORCHLENS_VALIDATE_PEAK_MEMORY", "1")
    assert tl.validate(model, torch.randn(2, 4), scope="forward", random_seed=0) is True
    peaks = last_validation_peak_memory()
    assert peaks is not None and "host_rss_peak_delta_bytes" in peaks
    assert peaks["host_rss_peak_delta_bytes"] >= 0
