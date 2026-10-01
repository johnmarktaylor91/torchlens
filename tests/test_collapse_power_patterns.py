"""F11 pattern-folding v1 gates (collapse memo item 12 / D11 / K5).

Typed AST parsing ({k} sugar, acyclic name-inlining, typed syntax
refusals), the honesty core (external-edge and junction refusals, counted
disclosures, near-miss suggestions, no-match warning), K5 chip grammar in
the DOT source, the v1 combination refusal, and the reserved text-grammar
lint (memo D10: each marker means exactly one thing graph-wide).
"""

from __future__ import annotations

import warnings as warnings_module

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.visualization.collapse_patterns import (
    IDIOMATIC_PATTERNS,
    match_patterns,
    parse_patterns,
    resolve_pattern_request,
)
from torchlens.visualization.collapse_plan import RenderContext

pytestmark = pytest.mark.smoke


class _Cbr(nn.Module):
    """conv2d > batch_norm > relu block."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply conv, norm, relu."""

        return torch.relu(self.bn(self.conv(x)))


class _CbrNet(nn.Module):
    """Three sequential CBR blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_Cbr() for _ in range(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply every block."""

        for block in self.blocks:
            x = block(x)
        return x


class _ResidualNet(nn.Module):
    """A residual skip crosses the pattern interior: honesty must refuse."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """conv -> bn -> relu with a skip from the conv OUTPUT past bn."""

        y = self.conv(x)
        z = torch.relu(self.bn(y))
        return z + y


@pytest.fixture(scope="module")
def cbr_trace():  # noqa: ANN201 - generator fixture
    """Shared three-block CBR trace."""

    trace = tl.trace(_CbrNet().eval(), torch.randn(1, 3, 8, 8))
    try:
        yield trace
    finally:
        trace.cleanup()


def test_parse_expands_repetition_and_inlines_names() -> None:
    """{k} sugar expands flat; earlier names inline acyclically."""

    specs = parse_patterns({"CBR": "conv2d > batch_norm > relu", "Deep": "CBR{2} > conv2d"})
    assert [atom.token for atom in specs[0].atoms] == ["conv2d", "batch_norm", "relu"]
    assert [atom.token for atom in specs[1].atoms] == [
        "conv2d",
        "batch_norm",
        "relu",
        "conv2d",
        "batch_norm",
        "relu",
        "conv2d",
    ]


def test_parse_refusals_are_typed() -> None:
    """Syntax refusals carry stable codes and remedies."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        parse_patterns({"": "relu"})
    assert excinfo.value.fields["code"] == "pattern_name_invalid"
    with pytest.raises(InvalidArgumentError) as excinfo:
        parse_patterns({"P": " > "})
    assert excinfo.value.fields["code"] == "pattern_syntax_invalid"
    with pytest.raises(InvalidArgumentError) as excinfo:
        parse_patterns({"P": "relu{0}"})
    assert excinfo.value.fields["code"] == "pattern_syntax_invalid"
    with pytest.raises(InvalidArgumentError) as excinfo:
        parse_patterns({"P": "re lu"})
    assert excinfo.value.fields["code"] == "pattern_syntax_invalid"
    with pytest.raises(InvalidArgumentError) as excinfo:
        resolve_pattern_request(42)
    assert excinfo.value.fields["code"] == "pattern_request_invalid"


def test_matches_fold_and_disclose(cbr_trace) -> None:
    """Three CBR sites fold; the report counts them; chips carry K5 grammar."""

    specs = parse_patterns({"ConvBnRelu": "conv2d > batch_norm > relu"})
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore")
        segments, report = match_patterns(cbr_trace, RenderContext(), specs)
    assert len(report.folded["ConvBnRelu"]) == 3
    assert len(segments) == 3
    for descriptor in segments.values():
        assert descriptor.label.startswith("PATTERN 'ConvBnRelu' -- 3 ops")
        assert descriptor.num_ops == 3
        # K5 grammar: never "(xN)", never "+N more" on a pattern chip.
        assert "(x" not in descriptor.label
        assert "more" not in descriptor.label


def test_residual_interior_edge_refuses_and_is_counted() -> None:
    """A skip reading the interior refuses THAT instance, counted."""

    trace = tl.trace(_ResidualNet().eval(), torch.randn(1, 3, 8, 8))
    specs = parse_patterns({"ConvBnRelu": "conv2d > batch_norm > relu"})
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        _segments, report = match_patterns(trace, RenderContext(), specs)
    assert not report.folded.get("ConvBnRelu")
    reasons = report.refused.get("ConvBnRelu", {})
    assert any("external edge" in reason for reason in reasons), reasons
    codes = {getattr(w.message, "fields", {}).get("code") for w in caught}
    assert "pattern_fold_refusals" in codes


def test_unknown_token_near_miss_and_no_match_warn(cbr_trace) -> None:
    """Unknown atoms suggest near misses; no-match patterns warn once."""

    specs = parse_patterns({"Typo": "conv2d > batch_nrom"})
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        _segments, report = match_patterns(cbr_trace, RenderContext(), specs)
    assert "batch_nrom" in report.unknown_tokens
    assert "batch_norm" in report.unknown_tokens["batch_nrom"]
    codes = {getattr(w.message, "fields", {}).get("code") for w in caught}
    assert {"pattern_token_unknown", "pattern_no_match"} <= codes


def test_draw_pattern_only_view_and_combination_refusal(cbr_trace, tmp_path) -> None:
    """The pattern-only view renders chips; auto+patterns refuses typed."""

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore")
        dot = cbr_trace.draw(
            collapse="none",
            fold_patterns={"ConvBnRelu": "conv2d > batch_norm > relu"},
            vis_save_only=True,
            vis_fileformat="dot",
            order_siblings=False,
        )
    assert dot.count("PATTERN 'ConvBnRelu'") == 3
    with pytest.raises(InvalidArgumentError) as excinfo:
        cbr_trace.draw(
            collapse="auto",
            fold_patterns={"ConvBnRelu": "conv2d > batch_norm > relu"},
            vis_save_only=True,
            vis_fileformat="dot",
        )
    assert excinfo.value.fields["code"] == "pattern_collapse_combination_unsupported"


def test_patterns_default_off(cbr_trace) -> None:
    """No fold_patterns argument means zero chips (memo D11: default off)."""

    dot = cbr_trace.draw(
        collapse="none", vis_save_only=True, vis_fileformat="dot", order_siblings=False
    )
    assert "PATTERN" not in dot


def test_idiomatic_preset_parses_and_matches(cbr_trace) -> None:
    """The curated preset is available (off by default) and matches CBR."""

    assert "ConvBnRelu" in IDIOMATIC_PATTERNS
    specs = resolve_pattern_request("idiomatic")
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore")
        _segments, report = match_patterns(cbr_trace, RenderContext(), specs)
    assert report.folded.get("ConvBnRelu")


def test_reserved_text_grammar_lint(cbr_trace) -> None:
    """Reserved markers mean exactly one thing graph-wide (memo D10).

    In one pattern-only DOT source: "PATTERN '<name>' --" appears only on
    chip labels, and the recurrence/fold markers "(xN)" and "+N more" never
    appear on a chip (they belong to K1 recurrence and K3 folds
    respectively).
    """

    with warnings_module.catch_warnings():
        warnings_module.simplefilter("ignore")
        dot = cbr_trace.draw(
            collapse="none",
            fold_patterns={"ConvBnRelu": "conv2d > batch_norm > relu"},
            vis_save_only=True,
            vis_fileformat="dot",
            order_siblings=False,
        )
    pattern_lines = [line for line in dot.splitlines() if "PATTERN" in line]
    assert len(pattern_lines) == 3
    for line in pattern_lines:
        assert "(x" not in line
        assert "more" not in line


def test_rank_layout_refuses_pattern_chips(cbr_trace) -> None:
    """Chips fail closed on the rank backend (pre-existing __segment__ gap)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        cbr_trace.draw(
            collapse="none",
            fold_patterns={"ConvBnRelu": "conv2d > batch_norm > relu"},
            vis_node_placement="rank",
            vis_save_only=True,
            vis_fileformat="dot",
        )
    assert excinfo.value.fields["code"] == "pattern_rank_layout_unsupported"
