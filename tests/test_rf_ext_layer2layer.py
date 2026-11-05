"""Layer-to-layer per-unit receptive-field extension tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from support.rf_isolation import preserved_rf_registry
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _rules
from torchlens.receptive_field._errors import NoInfluencePathError
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, _RuleResult


@pytest.fixture(autouse=True)
def isolated_rule_registry() -> Iterator[None]:
    """Restore the process-global receptive-field rule registry after each test."""

    with preserved_rf_registry(bump_epoch=True):
        yield


def _register_residual_rules() -> None:
    """Register the exact convolution and residual-merge rules used by these goldens."""

    @_rules.register_rf_rule("conv1d")
    def convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Emit the captured one-dimensional convolution window."""

        return context.window(
            kernel=(int(context.cfg("kernel_size")),),
            stride=context.cfg("stride", 1),
            padding=context.cfg("padding", 0),
            dilation=context.cfg("dilation", 1),
        )

    @_rules.register_rf_rule("add")
    def addition(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Emit an exact elementwise residual merge."""

        return context.passthrough()


class _ResidualStage(nn.Module):
    """One residual one-dimensional stage with three same-padded convolutions."""

    def __init__(self) -> None:
        """Create the stage modules."""

        super().__init__()
        self.early = nn.Conv1d(1, 1, 3, padding=1, bias=False)
        self.middle = nn.Conv1d(1, 1, 3, padding=1, bias=False)
        self.post = nn.Conv1d(1, 1, 3, padding=1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply a two-branch residual stage."""

        early = self.early(inputs)
        return self.post(self.middle(early) + early)


def _ops(trace: object) -> tuple[object, object, object]:
    """Return the early, middle, and post convolution operations in execution order."""

    convolutions = [op for op in trace._layer_list if op.func_name == "conv1d"]  # type: ignore[attr-defined]
    assert len(convolutions) == 3
    return tuple(convolutions)  # type: ignore[return-value]


def test_receptive_source_requires_an_ancestor_path() -> None:
    """Reject a source that lies downstream of the receptive-view owner."""

    _register_residual_rules()
    trace = tl.trace(_ResidualStage(), torch.ones(1, 1, 11))
    _, middle, post = _ops(trace)

    with pytest.raises(NoInfluencePathError, match="directed A -> B path"):
        middle.receptive_field.at((5,), source=post)
