"""Observe-kit items 5, 6, 10: backward NaN bisector, shared shape, AMP disclosure.

Healing is measured fact: a sqrt-backward birth erased by a downstream
relu-backward means a finite loss gradient does NOT prove a clean backward,
so the bisector carries the full fire-order ledger with the earliest provable
birth headlined -- and the gradient-flow audit stops blaming the innocent
CARRIER (a per-op grad is w.r.t. the op's OUTPUT, so a non-finite gradient on
X implicates X's CONSUMER's backward).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.debug import (
    FirstBadThing,
    bisect_nan_backward,
    gradient_flow_audit_rows,
)

pytestmark = pytest.mark.smoke


class _NanBirthBackward(torch.autograd.Function):
    """Identity forward whose backward emits NaN (a controlled birth site)."""

    @staticmethod
    def forward(ctx: object, x: torch.Tensor) -> torch.Tensor:
        """Pass the input through unchanged."""

        return x.clone()

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> torch.Tensor:
        """Emit a NaN gradient regardless of the arriving gradient."""

        return grad_output * float("nan")


class _SqrtHealModel(nn.Module):
    """sqrt(relu(x)): the sqrt backward births NaN at x==0, relu backward heals it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compose relu into sqrt so the zero positions blow up in backward."""

        return torch.sqrt(torch.relu(x)).sum()


def _backward_trace(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    """Capture one trace with saved gradients and log one backward pass."""

    captured = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(backward_ready=True, save_grads="all"),
        save_mode="reference",
    )
    output = captured.output_ops[0].out
    captured.log_backward(output)
    return captured


def test_clean_backward_gets_complete_clean_verdict() -> None:
    """A finite backward with full grad coverage reports clean_complete."""

    model = nn.Sequential(nn.Linear(3, 3), nn.Tanh(), nn.Linear(3, 1))

    class _Summed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Reduce to a scalar loss for a plain backward."""

            return self.body(x).sum()

    captured = _backward_trace(_Summed(), torch.randn(2, 3, requires_grad=True))
    try:
        result = bisect_nan_backward(captured)
        assert not result.found
        assert result.birth is None
        assert result.verdict_scope in ("clean_complete", "clean_among_checked")
        assert result.transitions, "the ledger is never empty for a walked pass"
    finally:
        captured.cleanup()


def test_backward_birth_is_localized_with_forward_join() -> None:
    """The sqrt-at-zero birth is localized with its paired forward op + line."""

    x = torch.tensor([[0.0, 1.0, 4.0]], requires_grad=True)
    captured = _backward_trace(_SqrtHealModel(), x)
    try:
        result = bisect_nan_backward(captured)
        assert result.found
        assert result.birth is not None
        assert result.birth.verdict == "birth"
        assert result.birth.arriving == "none"
        assert result.birth.emitted == "inf"
        assert result.birth.op_label is not None and result.birth.op_label.startswith("sqrt")
        assert result.birth.source_line is not None
        assert "born in the backward" in result.message
        shared = result.first_bad_thing
        assert isinstance(shared, FirstBadThing)
        assert shared.tool == "bisect_nan_backward"
        assert shared.backward_pass == 1
        assert shared.label == result.birth.op_label
    finally:
        captured.cleanup()


def test_unpaired_birth_is_never_overclaimed() -> None:
    """A birth inside an unsaved (unpaired) node stays first-among-checked.

    A custom autograd Function has no paired traced op, so ``save_grads``
    never saved its payloads: the tool must report the non-finite evidence
    WITHOUT fabricating a birth claim it cannot prove.
    """

    class _WithBadBackward(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Route through the NaN-emitting custom Function."""

            return _NanBirthBackward.apply(self.linear(x)).sum()

    captured = _backward_trace(_WithBadBackward(), torch.randn(2, 3, requires_grad=True))
    try:
        result = bisect_nan_backward(captured)
        assert result.found
        assert result.birth is None
        assert result.verdict_scope == "found_first_among_checked"
        assert result.uncertainty_zone, "the unchecked custom node must be named"
        assert any(t.verdict == "propagated" for t in result.transitions)
        assert "no birth is provable" in result.message
    finally:
        captured.cleanup()


def test_healed_birth_reported_despite_finite_final_gradient() -> None:
    """sqrt-birth erased by relu-backward: healed is measured, not decoration."""

    x = torch.tensor([[0.0, 1.0, 4.0]], requires_grad=True)
    captured = _backward_trace(_SqrtHealModel(), x)
    try:
        result = bisect_nan_backward(captured)
        assert result.found, "the sqrt backward births a NaN at the zero position"
        assert result.healed, "the relu backward heals the NaN and must be recorded"
        healed_transitions = [t for t in result.transitions if t.verdict == "healed"]
        assert healed_transitions
        assert "finite loss gradient does not prove a clean backward" in result.message
    finally:
        captured.cleanup()


def test_multi_backward_requires_bwd_selector() -> None:
    """Two captured passes refuse an unselected walk; passes never collapse."""

    from torchlens._errors import InvalidArgumentError

    class _Summed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Scalar loss for repeated backwards."""

            return self.linear(x).sum()

    captured = tl.trace(
        _Summed(),
        torch.randn(2, 3, requires_grad=True),
        capture=tl.options.CaptureOptions(backward_ready=True, save_grads="all"),
        save_mode="reference",
    )
    try:
        output = captured.output_ops[0].out
        captured.log_backward(output, retain_graph=True)
        captured.log_backward(output)
        with pytest.raises(InvalidArgumentError) as excinfo:
            bisect_nan_backward(captured)
        assert excinfo.value.fields["code"] == "bwd_selector_required"
        with pytest.raises(InvalidArgumentError) as unknown_info:
            bisect_nan_backward(captured, bwd=7)
        assert unknown_info.value.fields["code"] == "bwd_selector_unknown"
        first = bisect_nan_backward(captured, bwd=1)
        second = bisect_nan_backward(captured, bwd=2)
        assert first.backward_pass == 1
        assert second.backward_pass == 2
    finally:
        captured.cleanup()


def test_backward_capture_required_refusal_teaches() -> None:
    """A forward-only trace refuses with the re-trace remedy."""

    from torchlens._errors import InvalidArgumentError

    captured = tl.trace(nn.Linear(3, 3), torch.randn(2, 3))
    try:
        with pytest.raises(InvalidArgumentError) as excinfo:
            bisect_nan_backward(captured)
        assert excinfo.value.fields["code"] == "backward_capture_required"
        assert "save_grads" in str(excinfo.value)
    finally:
        captured.cleanup()


def test_gradient_flow_audit_relabels_carrier_and_points_at_bisector() -> None:
    """The audit's non-finite rows are carrier observations, not birth claims."""

    class _WithBadBackward(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Route through the NaN-emitting custom Function."""

            return _NanBirthBackward.apply(self.linear(x)).sum()

    captured = _backward_trace(_WithBadBackward(), torch.randn(2, 3, requires_grad=True))
    try:
        rows, _attrs = gradient_flow_audit_rows(captured)
        nonfinite_rows = [
            row
            for row in rows
            if row["grad_norm"] is not None and row["grad_norm"] != row["grad_norm"]
        ]
        carrier_rows = [row for row in rows if "carrier observation" in str(row["reason"])]
        assert carrier_rows, "non-finite landings must be labeled carrier observations"
        assert all("bisect_nan_backward" in str(row["reason"]) for row in carrier_rows)
        del nonfinite_rows
    finally:
        captured.cleanup()


def test_audit_trace_runs_bisector_automatically_when_unambiguous() -> None:
    """audit_trace adds the backward-birth finding for one captured backward."""

    class _WithBadBackward(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Route through the NaN-emitting custom Function."""

            return _NanBirthBackward.apply(self.linear(x)).sum()

    captured = _backward_trace(_WithBadBackward(), torch.randn(2, 3, requires_grad=True))
    try:
        audit = tl.debug.audit_trace(captured)
        assert "bisect_nan_backward" in audit.checks_run
        assert any(finding.check == "bisect_nan_backward" for finding in audit.findings)
    finally:
        captured.cleanup()


def test_amp_hint_fires_only_for_pure_inf_findings_without_scale() -> None:
    """Inf-only findings without grad_scale carry the shared AMP disclosure."""

    class _InfBirthBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx: object, x: torch.Tensor) -> torch.Tensor:
            """Pass the input through unchanged."""

            return x.clone()

        @staticmethod
        def backward(ctx: object, grad_output: torch.Tensor) -> torch.Tensor:
            """Emit an inf gradient (the fp16 GradScaler overflow shape)."""

            return grad_output * float("inf")

    class _WithInfBackward(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Route through the inf-emitting custom Function."""

            return _InfBirthBackward.apply(self.linear(x)).sum()

    captured = _backward_trace(_WithInfBackward(), torch.randn(2, 3, requires_grad=True))
    try:
        undisclosed = bisect_nan_backward(captured)
        assert undisclosed.amp_hint is not None
        assert "GradScaler" in undisclosed.amp_hint
        disclosed = bisect_nan_backward(captured, grad_scale=65536.0)
        assert disclosed.amp_hint is None
        assert disclosed.grad_scale == 65536.0
    finally:
        captured.cleanup()
