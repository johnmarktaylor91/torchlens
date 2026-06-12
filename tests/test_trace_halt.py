"""Regression tests for public ``trace(halt=...)`` partial captures."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.fastlog import RecordContext
from torchlens.options import CaptureOptions


class _ThreeStageModel(nn.Module):
    """Tiny model with an operation after the halt target."""

    def __init__(self) -> None:
        """Initialize child modules."""

        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run three observable stages."""

        x = self.linear(x)
        x = self.relu(x)
        return torch.sigmoid(x)


def _halt_on_relu(ctx: RecordContext) -> bool:
    """Return True for the relu operation event."""

    return ctx.kind == "op" and ctx.func_name == "relu"


def _halt_on_linear_exit(ctx: RecordContext) -> bool:
    """Return True for the linear module-exit event."""

    return ctx.kind == "module_exit" and ctx.address == "linear"


def test_trace_halt_op_returns_partial_trace_after_save_decision() -> None:
    """trace(halt=...) returns a finalized partial graph through the matching op."""

    trace = tl.trace(_ThreeStageModel(), torch.ones(1, 3), halt=_halt_on_relu)

    assert trace.halted is True
    assert trace.halt_frontier == trace.halt_reason
    assert trace.halt_reason is not None
    assert "relu" in trace.halt_reason
    assert any(
        layer.func_name == "relu" and layer.has_saved_activation for layer in trace.layer_list
    )
    assert not any(layer.func_name == "sigmoid" for layer in trace.layer_list)


def test_trace_halt_module_exit_stops_after_module_output() -> None:
    """Module halt predicates fire at exit after the module output is captured."""

    trace = tl.trace(
        _ThreeStageModel(),
        torch.ones(1, 3),
        halt=_halt_on_linear_exit,
    )

    assert trace.halted is True
    assert trace.halt_reason == "linear:exit:1"
    assert any(layer.func_name == "linear" for layer in trace.layer_list)
    assert not any(layer.func_name == "relu" for layer in trace.layer_list)


def test_trace_halt_rejects_selective_two_pass_layers_to_save() -> None:
    """halt= cannot be combined with the legacy two-pass selective save path."""

    with pytest.raises(ValueError, match="halt=predicate.*selective two-pass"):
        tl.trace(
            _ThreeStageModel(),
            torch.ones(1, 3),
            layers_to_save=["linear"],
            halt=_halt_on_relu,
        )


def test_halted_trace_validation_skips_full_model_output_only() -> None:
    """Halted validation still checks replay and metadata through the frontier."""

    trace = tl.trace(
        _ThreeStageModel(),
        torch.ones(1, 3),
        halt=_halt_on_relu,
        capture=CaptureOptions(save_arg_values=True),
    )

    assert trace.validate_forward_pass([torch.full((1, 3), 123.0)]) is True
