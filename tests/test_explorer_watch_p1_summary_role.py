"""P1: the non-differentiable summary-transform role (explorer memo item 1).

The refusal surface the memo names -- the trace-side train-mode validator,
its fastlog sibling, the streaming validator, and the grad path -- with the
NARROW declared-role carve-out: a ``summary(fn)`` transform reduces a
detached view after the save point and is exempt from the differentiability
tripwire; an UNDECLARED detached transform keeps refusing with
``transform_not_differentiable`` (the tripwire stays armed).
"""

from __future__ import annotations

import pytest
import torch

import torchlens as tl
from torchlens._ir.summary_role import (
    ensure_summary_output_owns_storage,
    summary,
)
from torchlens._training_validation import TrainingModeConfigError
from torchlens.recording import CaptureSpec


def _histo64(t: torch.Tensor) -> torch.Tensor:
    return torch.histc(t.float(), bins=8).to(torch.int64)


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())


class TestSummaryWrapper:
    """The role marker itself."""

    def test_non_callable_refuses(self) -> None:
        with pytest.raises(TypeError, match="callable reducer"):
            summary(42)  # type: ignore[arg-type]


class TestTraceTrainModeCarveOut:
    """Trace path: backward_ready + summary role passes; undeclared refuses."""

    def test_summary_int64_reducer_passes_backward_ready(self) -> None:
        model = _model()
        x = torch.randn(2, 4, requires_grad=True)
        log = tl.trace(
            model,
            x,
            capture=tl.options.CaptureOptions(
                backward_ready=True,
                activation_transform=summary(_histo64),
                save_raw_activations=False,
            ),
        )
        op = log["linear_1_1"]
        assert op.out is None
        assert op.payload.transformed_out.dtype == torch.int64
        assert int(op.payload.transformed_out.sum()) == 8  # every element binned

    def test_undeclared_detached_transform_still_refuses(self) -> None:
        model = _model()
        x = torch.randn(2, 4, requires_grad=True)
        with pytest.raises(TrainingModeConfigError) as excinfo:
            tl.trace(
                model,
                x,
                capture=tl.options.CaptureOptions(
                    backward_ready=True, activation_transform=_histo64
                ),
            )
        assert excinfo.value.fields["code"] == "transform_not_differentiable"
        # The refusal teaches the new remedy at the point of failure.
        assert "summary(" in str(excinfo.value)

    def test_summary_grad_transform_passes(self) -> None:
        model = _model()
        x = torch.randn(2, 4, requires_grad=True)
        log = tl.trace(
            model,
            x,
            capture=tl.options.CaptureOptions(
                backward_ready=True,
                save_grads=True,
                grad_transform=summary(_histo64),
                save_raw_grads=False,
            ),
        )
        log.backward(log[log.output_layers[0]].out.sum())
        op = log["linear_1_1"]
        assert op.payload.transformed_grad is not None
        assert op.payload.transformed_grad.dtype == torch.int64


class TestFastlogCarveOut:
    """Fastlog path: keep_grad + summary role passes; undeclared refuses."""

    def test_summary_reducer_lands_counts_at_every_site(self) -> None:
        model = _model()
        x = torch.randn(2, 4, requires_grad=True)
        rec = tl.record(
            model,
            x,
            recording=tl.options.RecordingOptions(
                default_op=CaptureSpec(save_out=True, save_grad=True)
            ),
            capture=tl.options.CaptureOptions(
                activation_transform=summary(_histo64), save_raw_activations=False
            ),
        )
        op_records = [r for r in rec.records if r.ctx.kind == "op"]
        assert op_records, "expected op records"
        for record in op_records:
            assert record.ram_payload is None
            payload = record.transformed_ram_payload
            assert payload.dtype == torch.int64
            numel = 1
            for dim in record.ctx.shape:
                numel *= dim
            assert int(payload.sum()) == numel, "totals exact at every site"

    def test_undeclared_detached_transform_still_refuses(self) -> None:
        model = _model()
        x = torch.randn(2, 4, requires_grad=True)
        with pytest.raises(TrainingModeConfigError) as excinfo:
            tl.record(
                model,
                x,
                recording=tl.options.RecordingOptions(
                    default_op=CaptureSpec(save_out=True, save_grad=True)
                ),
                capture=tl.options.CaptureOptions(activation_transform=_histo64),
            )
        assert excinfo.value.fields["code"] == "transform_not_differentiable"
        assert "summary(" in str(excinfo.value)

    def test_input_records_reduced_too(self) -> None:
        # Explorer P3 pin (memo V7): source events get the transform.
        model = _model()
        x = torch.randn(2, 4)
        rec = tl.record(
            model,
            x,
            recording=tl.options.RecordingOptions(default_op=True, include_source_events=True),
            capture=tl.options.CaptureOptions(
                activation_transform=summary(_histo64), save_raw_activations=False
            ),
        )
        inputs = [r for r in rec.records if r.ctx.kind == "input"]
        assert inputs, "expected an input source record"
        for record in inputs:
            assert record.transformed_ram_payload is not None
            assert record.transformed_ram_payload.dtype == torch.int64


class TestStreamingTripwireUnchanged:
    """The streaming serialization validator is NOT relaxed by the role."""

    def test_summary_non_tensor_output_refuses_under_streaming(self, tmp_path) -> None:
        from torchlens._io import ArtifactError

        model = _model()
        x = torch.randn(2, 4)
        with pytest.raises(ArtifactError):
            tl.trace(
                model,
                x,
                storage=tl.to_disk(str(tmp_path / "run.tlspec")),
                capture=tl.options.CaptureOptions(
                    activation_transform=summary(lambda t: float(t.sum())),
                    save_raw_activations=False,
                ),
            )


class TestAliasGuardUnit:
    """ensure_summary_output_owns_storage: fail-safe aliasing behavior."""

    def test_non_tensor_passes_through(self) -> None:
        live = torch.randn(8)
        assert ensure_summary_output_owns_storage(3.5, live) == 3.5

    def test_unreadable_storage_still_fails_safe(self) -> None:
        sentinel = object()

        class _Unreadable:
            def untyped_storage(self) -> None:
                raise RuntimeError("storage unreadable")

            def clone(self) -> object:
                return sentinel

        assert ensure_summary_output_owns_storage(_Unreadable(), torch.randn(8)) is sentinel

    def test_recursion_error_propagates(self) -> None:
        """A recursion inside the check is capture re-entry, never "unprovable aliasing"."""

        class _Recursing:
            def untyped_storage(self) -> None:
                raise RecursionError("maximum recursion depth exceeded")

            def clone(self) -> None:
                raise AssertionError("a recursion must surface, not clone")

        with pytest.raises(RecursionError):
            ensure_summary_output_owns_storage(_Recursing(), torch.randn(8))
