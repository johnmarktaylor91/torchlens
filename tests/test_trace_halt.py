"""Regression tests for public ``trace(halt=...)`` partial captures."""

from __future__ import annotations

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


def test_trace_halt_supports_selective_layers_to_save() -> None:
    """halt= composes with predicate-backed selective ``layers_to_save``."""

    trace = tl.trace(
        _ThreeStageModel(),
        torch.ones(1, 3),
        halt=_halt_on_relu,
        capture=tl.options.CaptureOptions(layers_to_save=["linear"]),
    )

    assert trace.halted is True
    assert any(
        layer.func_name == "linear" and layer.has_saved_activation for layer in trace.layer_list
    )
    assert any(layer.func_name == "relu" for layer in trace.layer_list)
    assert not any(layer.func_name == "sigmoid" for layer in trace.layer_list)


def test_halted_trace_validation_skips_full_model_output_only() -> None:
    """Halted validation still checks replay and metadata through the frontier."""

    # Hold the model: validation's saved-arg parameter matching reads live
    # params through the source-model weakref, and a temporary model survives
    # only until the next cyclic gc pass (a latent order-dependent flake --
    # reproduced at base 01be3c89 with a forced gc.collect()).
    model = _ThreeStageModel()
    trace = tl.trace(
        model,
        torch.ones(1, 3),
        halt=_halt_on_relu,
        capture=CaptureOptions(save_args=True),
    )

    assert trace.validate_forward_pass([torch.full((1, 3), 123.0)]) is True


class _WeightsBeforeHaltModel(nn.Module):
    """A constant table computed before the halt op and consumed only after it."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale the activation by a table that was built ahead of it."""

        weights = torch.ones(3) * 2
        return torch.relu(x) * weights


def test_trace_halt_keeps_values_awaiting_consumers() -> None:
    """A halted capture keeps every op that ran, consumers pending or not.

    The weights table is input-disconnected and, at the halt, childless (its
    consumer never ran), so the orphan flood from inputs and the halt frontier
    pruned it as dead computation: the halted trace started at the relu while
    the complete trace started at the ones op, and the "honest prefix" promise
    of a HALTED outcome broke on the first op.
    """

    model = _WeightsBeforeHaltModel()
    x = torch.ones(1, 3)
    plain = tl.trace(model, x)
    halted = tl.trace(model, x, halt=_halt_on_relu)
    assert halted.halted is True
    order = lambda trace: [  # noqa: E731
        op.func_name for op in sorted(trace.ops, key=lambda op: op.raw_index)
    ]
    plain_names, halted_names = order(plain), order(halted)
    assert halted_names == ["ones", "__mul__", "relu"]
    assert plain_names[: len(halted_names)] == halted_names
    assert len(halted_names) < len(plain_names)
