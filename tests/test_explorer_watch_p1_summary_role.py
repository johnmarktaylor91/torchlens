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
from torchlens._training_validation import TrainingModeConfigError
from torchlens.fastlog import CaptureSpec
from torchlens.ir.summary_role import (
    SummaryTransform,
    ensure_summary_output_owns_storage,
    is_summary_transform,
    summary,
    transform_role,
)


def _histo64(t: torch.Tensor) -> torch.Tensor:
    return torch.histc(t.float(), bins=8).to(torch.int64)


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())


class TestSummaryWrapper:
    """The role marker itself."""

    def test_declares_role_and_detaches(self) -> None:
        seen: list[bool] = []

        def probe(t: torch.Tensor) -> torch.Tensor:
            seen.append(t.requires_grad)
            return t.sum().unsqueeze(0)

        wrapper = summary(probe)
        assert is_summary_transform(wrapper)
        assert transform_role(wrapper) == "summary"
        assert transform_role(probe) == "standard"
        assert transform_role(None) == "standard"
        live = torch.randn(3, requires_grad=True)
        wrapper(live)
        assert seen == [False], "summary reducer must see a detached view"

    def test_repr_and_name(self) -> None:
        assert "summary" not in repr(summary(_histo64)) or "SummaryTransform" in repr(
            summary(_histo64)
        )
        assert summary(_histo64).name == "_histo64"
        assert summary(_histo64, name="counts").name == "counts"

    def test_non_callable_refuses(self) -> None:
        with pytest.raises(TypeError, match="callable reducer"):
            summary(42)  # type: ignore[arg-type]

    def test_duck_typed_foreign_declaration(self) -> None:
        class Foreign:
            _tl_transform_role = "summary"

            def __call__(self, t: torch.Tensor) -> torch.Tensor:
                return t.detach().sum().unsqueeze(0)

        assert is_summary_transform(Foreign())

        # Unknown role tokens read as standard: misdeclaration keeps the tripwire.
        class Misdeclared:
            _tl_transform_role = "something_else"

        assert not is_summary_transform(Misdeclared())

    def test_subclass_marker_class(self) -> None:
        assert isinstance(summary(_histo64), SummaryTransform)


class TestTraceTrainModeCarveOut:
    """Trace path: backward_ready + summary role passes; undeclared refuses."""

    def test_summary_int64_reducer_passes_backward_ready(self) -> None:
        model = _model()
        x = torch.randn(2, 4, requires_grad=True)
        log = tl.trace(
            model,
            x,
            save=tl.options.SaveOptions(
                activation_transform=summary(_histo64),
                save_raw_activations=False,
            ),
            capture=tl.options.CaptureOptions(backward_ready=True),
        )
        op = log["linear_1_1"]
        assert op.out is None
        assert op.transformed_out.dtype == torch.int64
        assert int(op.transformed_out.sum()) == 8  # every element binned

    def test_user_graph_survives_summary_capture(self) -> None:
        model = _model()
        x = torch.randn(2, 4, requires_grad=True)
        log = tl.trace(
            model,
            x,
            save=tl.options.SaveOptions(
                activation_transform=summary(_histo64),
                save_raw_activations=False,
            ),
            capture=tl.options.CaptureOptions(backward_ready=True),
        )
        del log
        out = model(x)
        out.sum().backward()
        assert x.grad is not None

    def test_undeclared_detached_transform_still_refuses(self) -> None:
        model = _model()
        x = torch.randn(2, 4, requires_grad=True)
        with pytest.raises(TrainingModeConfigError) as excinfo:
            tl.trace(
                model,
                x,
                save=tl.options.SaveOptions(activation_transform=_histo64),
                capture=tl.options.CaptureOptions(backward_ready=True),
            )
        assert excinfo.value.fields["code"] == "transform_not_differentiable"
        # The refusal teaches the new remedy at the point of failure.
        assert "summary(" in str(excinfo.value)

    @pytest.mark.smoke
    def test_summary_grad_transform_passes(self) -> None:
        model = _model()
        x = torch.randn(2, 4, requires_grad=True)
        log = tl.trace(
            model,
            x,
            grad_transform=summary(_histo64),
            save=tl.options.SaveOptions(save_raw_gradients=False),
            capture=tl.options.CaptureOptions(backward_ready=True, save_grads=True),
        )
        log.log_backward(log[log.output_layers[0]].out.sum())
        op = log["linear_1_1"]
        assert op.transformed_grad is not None
        assert op.transformed_grad.dtype == torch.int64


class TestFastlogCarveOut:
    """Fastlog path: keep_grad + summary role passes; undeclared refuses."""

    def test_summary_reducer_lands_counts_at_every_site(self) -> None:
        model = _model()
        x = torch.randn(2, 4, requires_grad=True)
        rec = tl.record(
            model,
            x,
            default_op=CaptureSpec(save_out=True, keep_grad=True),
            activation_transform=summary(_histo64),
            save_raw_activations=False,
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
                default_op=CaptureSpec(save_out=True, keep_grad=True),
                activation_transform=_histo64,
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
            default_op=True,
            include_source_events=True,
            activation_transform=summary(_histo64),
            save_raw_activations=False,
        )
        inputs = [r for r in rec.records if r.ctx.kind == "input"]
        assert inputs, "expected an input source record"
        for record in inputs:
            assert record.transformed_ram_payload is not None
            assert record.transformed_ram_payload.dtype == torch.int64


class TestStreamingTripwireUnchanged:
    """The streaming serialization validator is NOT relaxed by the role."""

    @pytest.mark.smoke
    def test_summary_non_tensor_output_refuses_under_streaming(self, tmp_path) -> None:
        from torchlens._io import TorchLensIOError

        model = _model()
        x = torch.randn(2, 4)
        with pytest.raises(TorchLensIOError):
            tl.trace(
                model,
                x,
                save=tl.options.SaveOptions(
                    activation_transform=summary(lambda t: float(t.sum())),
                    save_raw_activations=False,
                ),
                storage=tl.to_disk(str(tmp_path / "run.tlspec")),
            )


class TestAliasGuardUnit:
    """ensure_summary_output_owns_storage: fail-safe aliasing behavior."""

    def test_view_output_cloned(self) -> None:
        live = torch.randn(8)
        view = live.detach()[:4]
        owned = ensure_summary_output_owns_storage(view, live)
        assert owned.untyped_storage().data_ptr() != live.untyped_storage().data_ptr()
        assert torch.equal(owned, view)

    def test_owning_output_passes_through(self) -> None:
        live = torch.randn(8)
        reduced = live.detach().sum().unsqueeze(0)
        assert ensure_summary_output_owns_storage(reduced, live) is reduced

    def test_non_tensor_passes_through(self) -> None:
        live = torch.randn(8)
        assert ensure_summary_output_owns_storage(3.5, live) == 3.5
