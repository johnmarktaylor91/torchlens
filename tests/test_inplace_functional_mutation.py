"""In-place functionals (``inplace=True``) are captured as mutating calls."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import torchlens as tl
from torchlens._io.runnable import build_sparse_run_descriptor
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

_MUTATING: dict[str, tuple[Callable[[torch.Tensor], torch.Tensor], str]] = {
    "leaky_relu": (lambda h: F.leaky_relu(h, 0.1, inplace=True), "leaky_relu"),
    "hardtanh": (lambda h: F.hardtanh(h, inplace=True), "hardtanh"),
    "relu6": (lambda h: F.relu6(h, inplace=True), "relu6"),
    "relu6_module": (nn.ReLU6(inplace=True), "hardtanh"),
    "elu": (lambda h: F.elu(h, inplace=True), "elu"),
    "hardswish_module_positional": (nn.Hardswish(inplace=True), "hardswish"),
}
_NOT_MUTATING: dict[str, tuple[Callable[[torch.Tensor], torch.Tensor], str]] = {
    "dropout_eval": (lambda h: F.dropout(h, 0.5, training=False, inplace=True), "dropout_"),
    "dropout_module_eval": (nn.Dropout(0.5, inplace=True), "dropout_"),
    "contiguous": (lambda h: h.contiguous(), "contiguous"),
    "leaky_relu_out_of_place": (lambda h: F.leaky_relu(h, 0.1), "leaky_relu"),
}


class _ReusesReceiver(nn.Module):
    """Apply ``fn`` to a hidden activation, then read that activation again."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Store the activation under test next to an affine layer."""

        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.fn = fn

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Reading ``hidden`` after ``fn`` exposes whether ``fn`` mutated it."""

        hidden = self.linear(value)
        activated = self.fn(hidden)
        return activated * 2.0 + hidden


def _op_flags(trace: Any, func_name: str) -> list[bool]:
    """Return ``is_inplace`` for every op with ``func_name``."""

    ops = trace.ops.values() if hasattr(trace.ops, "values") else trace.ops
    return [bool(op.is_inplace) for op in ops if op.func_name == func_name]


def _runnable_trace(model: nn.Module, inputs: torch.Tensor) -> Any:
    """Capture a runnable-ready trace."""

    return tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )


@pytest.mark.smoke
@pytest.mark.parametrize("case", sorted(_MUTATING))
def test_inplace_functional_is_recorded_as_mutating(case: str) -> None:
    """Default and runnable-ready captures both flag the call as in-place."""

    fn, func_name = _MUTATING[case]
    torch.manual_seed(0)
    model = _ReusesReceiver(fn).eval()
    inputs = torch.randn(2, 4)
    assert _op_flags(tl.trace(model, inputs), func_name) == [True]
    trace = _runnable_trace(model, inputs)
    assert _op_flags(trace, func_name) == [True]
    descriptor = build_sparse_run_descriptor(trace)
    names = {entry.registry_id: entry.key.qualname for entry in descriptor.callable_registry}
    mutating = [
        call.is_inplace for call in descriptor.calls if names[call.registry_id] == func_name
    ]
    assert mutating == [True]


@pytest.mark.smoke
@pytest.mark.parametrize("case", sorted(_NOT_MUTATING))
def test_same_object_return_without_a_write_is_not_mutating(case: str) -> None:
    """Eval-mode ``dropout_`` returns its input untouched; runnable capture says so."""

    fn, func_name = _NOT_MUTATING[case]
    torch.manual_seed(0)
    trace = _runnable_trace(_ReusesReceiver(fn).eval(), torch.randn(2, 4))
    assert _op_flags(trace, func_name) == [False]


@pytest.mark.parametrize("case", sorted(_MUTATING) + sorted(_NOT_MUTATING))
def test_loaded_run_of_inplace_functional_is_verified(case: str, tmp_path: Path) -> None:
    """Save, load and run: the version-divergence check passes and outputs match."""

    fn, _func_name = {**_MUTATING, **_NOT_MUTATING}[case]
    torch.manual_seed(0)
    model = _ReusesReceiver(fn).eval()
    inputs = torch.randn(2, 4)
    path = tmp_path / "model.tlspec"
    tl.save(_runnable_trace(model, inputs), path, level="runnable", include_weights=True)
    result = tl.load(path).run(inputs=inputs, seed=0)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    torch.testing.assert_close(result.output, model(inputs))


@pytest.mark.smoke
def test_identity_return_after_a_sibling_view_write_is_not_mutating() -> None:
    """A stale version baseline never turns an identity return into a mutation."""

    class _SiblingViews(nn.Module):
        """Write through one row view, then return another row unchanged."""

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            hidden = self.linear(value)
            first, second = hidden[0], hidden[1]
            first.add_(1.0)
            return second.contiguous() + first

    trace = _runnable_trace(_SiblingViews().eval(), torch.randn(2, 4))
    assert _op_flags(trace, "add_") == [True]
    assert _op_flags(trace, "contiguous") == [False]
