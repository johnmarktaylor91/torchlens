"""Tests for the trace-level receptive-field table."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from support.rf_isolation import preserved_rf_registry
from torch import nn

import torchlens as tl
from torchlens.receptive_field import _rules
from torchlens.receptive_field._rules import ReceptiveFieldRuleContext, _RuleResult
from torchlens.receptive_field._types import (
    ReceptiveFieldStatus,
)

pd = pytest.importorskip("pandas")


@pytest.fixture(autouse=True)
def isolated_rule_registry() -> Iterator[None]:
    """Restore the process-global RF rule registry after every test."""

    with preserved_rf_registry(bump_epoch=True):
        yield


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
        expected = len(trace._layer_list)  # type: ignore[union-attr]
    elif level == "layer":
        expected = sum(len(layer.op_labels) for layer in trace._layer_logs.values())  # type: ignore[union-attr]
    elif level == "call":
        expected = sum(len(call.output_ops) for call in trace.module_calls.values())  # type: ignore[union-attr]
    else:
        expected = sum(len(module.output_ops) for module in trace.modules.values())  # type: ignore[union-attr]

    assert len(frame) == expected
    assert set(frame["kind"]) == {level}


def test_receptive_field_table_filters_by_input_op_handle_only() -> None:
    """Accept the graph-native input handle and reject ambiguous string spellings."""

    trace = _trace_residual()
    input_op = next(op for op in trace._layer_list if op.is_input)  # type: ignore[union-attr]
    filtered = trace.receptive_fields(input_op=input_op).to_pandas()  # type: ignore[union-attr]

    assert not filtered.empty
    assert set(filtered["input_op"]) == {input_op.label}
    with pytest.raises(TypeError, match="Op handle"):
        trace.receptive_fields(input_op=input_op.layer_label)  # type: ignore[union-attr]


def test_residual_merge_row_is_exact_not_chain_approximated() -> None:
    """Preserve whole-DAG skip-merge exactness in the layer-level table."""

    trace = _trace_residual()
    frame = trace.receptive_fields(level="layer").to_pandas()  # type: ignore[union-attr]
    add_row = frame.loc[frame["rule"] == "add"].iloc[0]

    assert add_row["size"] == (3, 3)
    assert add_row["status"] is ReceptiveFieldStatus.EXACT
    assert bool(add_row["exact"]) is True


def test_profile_repr_shows_level_row_count_and_first_rows() -> None:
    """``repr(profile)`` is a bounded preview: level, row count and the first rows."""

    from torchlens.receptive_field._types import ReceptiveFieldProfile

    rows = [
        {"name": f"conv_{index}", "input_role": "input_1", "status": "exact", "size": (3, 3)}
        for index in range(8)
    ]
    profile = ReceptiveFieldProfile(frame=pd.DataFrame(rows), level="layer")

    text = repr(profile)
    lines = text.splitlines()

    assert lines[0] == "ReceptiveFieldProfile(level='layer', rows=8)"
    assert "conv_0" in text and "conv_4" in text
    assert "conv_5" not in text
    assert "3 more rows" in text
    assert ".to_pandas()" in text
    assert len(lines) <= 9


def test_profile_repr_of_a_captured_trace_and_an_empty_profile() -> None:
    """A real profile previews its rows; an empty one is the header line alone."""

    from torchlens.receptive_field._types import ReceptiveFieldProfile

    trace = _trace_residual()
    profile = trace.receptive_fields(level="layer")  # type: ignore[union-attr]
    frame = profile.to_pandas()

    text = repr(profile)

    assert text.splitlines()[0] == f"ReceptiveFieldProfile(level='layer', rows={len(frame)})"
    assert str(frame.loc[0, "name"]) in text
    assert "frame=" not in text
    empty = ReceptiveFieldProfile(frame=pd.DataFrame(columns=["name", "status"]), level="op")
    assert repr(empty) == "ReceptiveFieldProfile(level='op', rows=0)"


def test_profile_repr_html_is_the_frame_table() -> None:
    """Notebooks render the profile as its frame's HTML table under a one-line header."""

    trace = _trace_residual()
    profile = trace.receptive_fields(level="layer")  # type: ignore[union-attr]

    rendered = profile._repr_html_()

    assert rendered is not None
    assert rendered.startswith("<p>ReceptiveFieldProfile(level=&#x27;layer&#x27;, rows=")
    assert "<table" in rendered
    assert str(profile.to_pandas().loc[0, "name"]) in rendered
