"""F08 config grammar: one compat table, typed conflicts, nothing silent.

Summary memo 3.10: unknown names teach with a nearest match; contradictory
axes raise ``summary_option_conflict``; legacy spellings keep their
historical byte-stable rendering through the ONE compatibility table; the
fma1 convention is honored or refused typed, never accepted-and-ignored.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import ArgumentConflictError, InvalidArgumentError, ShapeInferenceError

pytestmark = pytest.mark.smoke


class _Toy(nn.Module):
    """Two-Linear toy with an orphan relu."""

    def __init__(self) -> None:
        """Two Linears."""

        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """fc1 -> relu -> fc2."""

        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture(scope="module")
def toy_trace():
    """One finished toy trace for the whole module."""

    trace = tl.trace(_Toy(), torch.randn(2, 8))
    try:
        yield trace
    finally:
        trace.cleanup()


def test_unknown_option_teaches_with_nearest_match(toy_trace) -> None:
    """An unknown option names itself and the valid grammar."""

    with pytest.raises(TypeError, match="colums|unexpected keyword"):
        toy_trace.summary(colums=["name"])


def test_unknown_view_refuses_typed(toy_trace) -> None:
    """view= is a closed vocabulary with a teaching refusal."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        toy_trace.summary(view="overveiw")
    assert excinfo.value.fields["code"] == "summary_option_invalid"
    assert "overview" in str(excinfo.value)  # the did-you-mean


def test_unknown_level_keeps_the_historical_code(toy_trace) -> None:
    """level= refusals keep the pinned summary_level_invalid code."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        toy_trace.summary(level="bogus")
    assert excinfo.value.fields["code"] == "summary_level_invalid"


def test_mixed_grammars_conflict_typed(toy_trace) -> None:
    """Legacy spellings + rebuilt axes in one call contradict (ONE table)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        toy_trace.summary(level="memory", view="compute")
    assert excinfo.value.fields["code"] == "summary_option_conflict"
    with pytest.raises(InvalidArgumentError) as excinfo:
        toy_trace.summary(show_ops=True, style="unicode")
    assert excinfo.value.fields["code"] == "summary_option_conflict"


def test_op_level_conflicts_with_depth(toy_trace) -> None:
    """level='op' + depth= contradict; the refusal teaches the fix."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        toy_trace.summary(level="op", depth=2)
    assert excinfo.value.fields["code"] == "summary_option_conflict"


def test_buffer_rows_refuse_until_qualified_names(toy_trace) -> None:
    """buffers='rows' is gated on qualified-name plumbing (memo 3.10)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        toy_trace.summary(buffers="rows")
    assert excinfo.value.fields["code"] == "summary_option_invalid"
    assert "qualified" in str(excinfo.value)


def test_legacy_spellings_stay_byte_stable(toy_trace) -> None:
    """The compat table's legacy route reproduces the historical text."""

    from torchlens.visualization._summary_internal import render_model_summary

    assert str(toy_trace.summary(level="overview")) == render_model_summary(toy_trace)
    # preset-only historically CRASHED against the default level; the compat
    # route resolves it to its own preset (a strict fix, not a text change).
    assert str(toy_trace.summary(preset="graph")) == render_model_summary(
        toy_trace, level="graph", preset="graph"
    )


def test_columns_bundle_exact_and_deltas(toy_trace) -> None:
    """columns= accepts a bundle name, an exact list, or +/- deltas."""

    bundle = toy_trace.summary(columns="compute")
    assert "macs" in bundle
    exact = toy_trace.summary(columns=["name", "params"])
    assert "fwd flops" not in exact
    delta = toy_trace.summary(columns=["+macs", "-params_pct"])
    assert "macs" in delta
    with pytest.raises(InvalidArgumentError) as excinfo:
        toy_trace.summary(columns=["name", "watts"])
    assert excinfo.value.fields["code"] == "summary_option_invalid"


def test_filter_discloses_coverage_and_preserves_totals(toy_trace) -> None:
    """Filters are presentation-only: coverage named, totals whole-model."""

    filtered = toy_trace.summary(filter="fc1")
    assert "showing" in filtered
    assert "visible coverage" in filtered
    full = toy_trace.summary()
    assert filtered.total_params == full.total_params
    assert filtered.total_flops_forward == full.total_flops_forward


def test_fma1_renders_from_the_two_term_record(toy_trace) -> None:
    """flop_convention='fma1' recounts totals; fma2 stays the default."""

    fma2 = toy_trace.summary()
    fma1 = toy_trace.summary(flop_convention="fma1")
    assert "(fma=2)" in fma2
    assert "(fma=1)" in fma1
    assert fma1.total_flops_forward == fma2.total_flops_forward  # raw data unchanged


def test_one_call_only_kwargs_refuse_on_trace_summary(toy_trace) -> None:
    """input_size/execution_mode/grad_mode are one-call-only spellings."""

    for kwargs in ({"input_size": (2, 8)}, {"execution_mode": "eval"}, {"grad_mode": "off"}):
        with pytest.raises(InvalidArgumentError) as excinfo:
            toy_trace.summary(**kwargs)
        assert excinfo.value.fields["code"] == "summary_one_call_only"
        assert "tl.summary" in str(excinfo.value)


def test_input_args_xor_input_size() -> None:
    """The one-call door refuses mixed input spellings (ladder rung law, F17)."""

    with pytest.raises(ArgumentConflictError) as excinfo:
        tl.summary(_Toy(), torch.randn(2, 8), input_size=(2, 8))
    assert excinfo.value.fields["code"] == "input_rung_conflict"


def test_synthetic_input_refuses_decoded_output_view() -> None:
    """A label table computed from noise refuses typed (memo 3.7)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.summary(_Toy(), input_size=(2, 8), level="output")
    assert excinfo.value.fields["code"] == "summary_synthetic_output_refused"
    assert "synthetic" in str(excinfo.value)


def test_zero_input_inference_failure_teaches_the_real_input_spelling() -> None:
    """An uninferrable model refuses typed and names both remedies."""

    class _Uninferrable(nn.Module):
        """Rejects every probe input the inference ladder can synthesize."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Refuse all inputs so shape inference cannot succeed."""

            raise RuntimeError("no synthetic input satisfies this forward")

    with pytest.raises(ShapeInferenceError) as excinfo:
        tl.summary(_Uninferrable())
    assert excinfo.value.fields["code"] == "input_inference_failed"
    assert "input_size=" in str(excinfo.value.fields["remedy"])


def test_input_size_synthesizes_and_discloses() -> None:
    """input_size= builds a seeded synthetic input and says so."""

    report = tl.summary(_Toy(), input_size=(2, 8))
    assert "synthetic input" in str(report)
    assert "input_size=(2, 8)" in str(report)
    assert report.total_params == (8 * 16 + 16) + (16 * 8 + 8)


def test_one_call_and_trace_summary_agree(toy_trace) -> None:
    """Composition row 1: identical report given identical capture + args."""

    one_call = tl.summary(_Toy(), torch.randn(2, 8))
    trace_side = toy_trace.summary()
    assert one_call.total_params == trace_side.total_params
    assert one_call.total_flops_forward == trace_side.total_flops_forward
    assert [row.row_id for row in one_call._rebuilt.view.rows] == [
        row.row_id for row in trace_side._rebuilt.view.rows
    ]


def test_max_rows_none_means_unbounded(toy_trace) -> None:
    """max_rows=None disables the budget instead of crashing."""

    report = toy_trace.summary(max_rows=None)
    assert "view:" in report
