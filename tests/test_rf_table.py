"""Tests for the trace-level receptive-field table."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction

import pandas as pd
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _rules
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, _RuleResult
from torchlens.receptive_field._types import (
    GridLayout,
    ReceptiveFieldAlignment,
    ReceptiveFieldStatus,
)


@pytest.fixture(autouse=True)
def isolated_rule_registry() -> Iterator[None]:
    """Restore the process-global RF rule registry after every test."""

    saved_rules = dict(_rules._RF_RULES)
    saved_epoch = _rules._RF_RULES_EPOCH
    yield
    _rules._RF_RULES.clear()
    _rules._RF_RULES.update(saved_rules)
    _rules._RF_RULES_EPOCH = saved_epoch


def _register_standard_rules() -> None:
    """Register the compact rule set required by the table fixtures."""

    @_rules.register_rf_rule("conv2d")
    def convolution(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Emit captured two-dimensional convolution recurrence parameters."""

        kernel = context.cfg("kernel_size")
        assert isinstance(kernel, tuple)
        return context.window(
            kernel=kernel,
            stride=context.cfg("stride", (1, 1)),
            padding=context.cfg("padding", (0, 0)),
            dilation=context.cfg("dilation", (1, 1)),
        )

    @_rules.register_rf_rule("add", "mul", "relu")
    def elementwise(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Preserve geometry through elementwise operations and merges."""

        return context.passthrough()

    @_rules.register_rf_rule("mean")
    def mean_rule(context: ReceptiveFieldRuleContext) -> _RuleResult:
        """Mark captured reduction dimensions as whole-input dependencies."""

        return context.full(axes=context.arg("dim", (-2, -1)))


class _Residual(nn.Module):
    """Small ResNet-like stack with an aligned skip merge."""

    def __init__(self) -> None:
        """Initialize the spatial and pointwise branches."""

        super().__init__()
        self.wide = nn.Conv2d(2, 2, 3, padding=1)
        self.point = nn.Conv2d(2, 2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Merge the two aligned residual branches."""

        return self.wide(inputs) + self.point(inputs)


class _GlobalBranch(nn.Module):
    """Global-pooling branch that broadcasts back to the input grid."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Multiply values by their spatial mean."""

        return inputs * inputs.mean(dim=(-2, -1), keepdim=True)


def _trace_residual() -> object:
    """Capture the residual fixture after registering its RF rules.

    Returns
    -------
    object
        Completed TorchLens trace.
    """

    _register_standard_rules()
    return tl.trace(_Residual(), torch.randn(1, 2, 16, 16))


@pytest.mark.parametrize("level", ["op", "layer", "call", "module"])
def test_receptive_field_table_has_one_row_per_entity_output_and_input(level: str) -> None:
    """Enumerate every represented boundary output once for the sole model input."""

    trace = _trace_residual()
    frame = trace.receptive_fields(level=level).to_pandas()  # type: ignore[union-attr]

    if level == "op":
        expected = len(trace.layer_list)  # type: ignore[union-attr]
    elif level == "layer":
        expected = sum(len(layer.op_labels) for layer in trace.layer_logs.values())  # type: ignore[union-attr]
    elif level == "call":
        expected = sum(len(call.output_ops) for call in trace.module_calls.values())  # type: ignore[union-attr]
    else:
        expected = sum(len(module.output_ops) for module in trace.modules.values())  # type: ignore[union-attr]

    assert len(frame) == expected
    assert set(frame["kind"]) == {level}


def test_receptive_field_table_columns_have_semantic_types() -> None:
    """Expose typed geometric values rather than formatted string approximations."""

    trace = _trace_residual()
    frame = trace.receptive_fields().to_pandas()  # type: ignore[union-attr]
    row = frame.loc[frame["rule"] == "conv2d"].iloc[0]

    assert {
        "size",
        "jump",
        "center0",
        "status",
        "input_role",
        "batch_coupled",
        "exact",
        "layout",
    }.issubset(frame.columns)
    assert isinstance(row["size"], tuple)
    assert isinstance(row["jump"], tuple)
    assert isinstance(row["center0"], tuple)
    assert all(isinstance(value, Fraction) for value in row["jump"])
    assert isinstance(row["status"], ReceptiveFieldStatus)
    assert isinstance(row["alignment"], ReceptiveFieldAlignment)
    assert isinstance(row["layout"], GridLayout)
    assert frame["batch_coupled"].dtype == bool
    assert frame["exact"].dtype == bool


def test_receptive_field_table_reports_per_entity_whole_input_status() -> None:
    """Keep a global-pool-derived row's status instead of collapsing table certainty."""

    _register_standard_rules()
    trace = tl.trace(_GlobalBranch(), torch.randn(2, 3, 8, 8))
    frame = trace.receptive_fields().to_pandas()

    assert ReceptiveFieldStatus.WHOLE_INPUT in set(frame["status"])


def test_receptive_field_table_filters_by_input_op_handle_only() -> None:
    """Accept the graph-native input handle and reject ambiguous string spellings."""

    trace = _trace_residual()
    input_op = next(op for op in trace.layer_list if op.is_input)  # type: ignore[union-attr]
    filtered = trace.receptive_fields(input=input_op).to_pandas()  # type: ignore[union-attr]

    assert not filtered.empty
    assert set(filtered["input_op"]) == {input_op.label}
    with pytest.raises(TypeError, match="Op handle"):
        trace.receptive_fields(input=input_op.layer_label)  # type: ignore[union-attr]


def test_receptive_field_profile_to_pandas_returns_a_copy() -> None:
    """Round-trip the frozen profile dataframe without leaking its internal frame."""

    trace = _trace_residual()
    profile = trace.receptive_fields()  # type: ignore[union-attr]
    first = profile.to_pandas()
    first.loc[:, "name"] = "changed"

    assert isinstance(first, pd.DataFrame)
    assert not (profile.to_pandas()["name"] == "changed").all()


def test_residual_merge_row_is_exact_not_chain_approximated() -> None:
    """Preserve whole-DAG skip-merge exactness in the layer-level table."""

    trace = _trace_residual()
    frame = trace.receptive_fields(level="layer").to_pandas()  # type: ignore[union-attr]
    add_row = frame.loc[frame["rule"] == "add"].iloc[0]

    assert add_row["size"] == (3, 3)
    assert add_row["status"] is ReceptiveFieldStatus.EXACT
    assert bool(add_row["exact"]) is True
