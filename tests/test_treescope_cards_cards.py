"""F16 B2/B3: the four cards, the copy-root ladder, motifs, and the grid.

Test-discipline law (treescope memo s12 rule 4): card tests assert
CONTENT, never non-emptiness -- the historical ``_repr_html_`` returned a
94-byte plain repr without IPython that passed any non-empty check.
"""

from __future__ import annotations

import re

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TraceCaptureError
from torchlens.report._notebook._access import CopyRoot, resolve_copy_root
from torchlens.report._notebook._axis_labels import (
    axis_labels,
    decode_token_labels,
    sdpa_attention_hint,
)
from torchlens.report._notebook._grid import array_grid_html


class _Recurrent(nn.Module):
    """Weight-reused loop: multi-pass layers for the pass-row contract."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Three shared-weight steps then a head."""
        for _ in range(3):
            x = torch.relu(self.fc(x))
        return self.head(x)


@pytest.fixture(scope="module")
def recurrent_log() -> object:
    """One captured recurrent trace shared across the module."""

    log = tl.trace(_Recurrent(), torch.randn(2, 4))
    try:
        yield log
    finally:
        log.cleanup()


def test_single_pass_layer_degenerates_to_op_card(recurrent_log) -> None:
    """ONE code path: a single-pass layer renders the Op card."""

    html = recurrent_log._layer_logs["linear_2_3"]._repr_html_()
    assert "TorchLens Op" in html
    assert "card unavailable" not in html


def test_op_card_unsaved_value_discloses(recurrent_log) -> None:
    """save= exclusion renders an honest reason, never a pretended value."""

    log = tl.trace(_Recurrent(), torch.randn(2, 4), save=tl.func("linear"))
    html = log["relu_1_2:1"]._repr_html_()
    assert "value not saved" in html
    assert '<table class="tl-grid-table"' not in html


def test_partial_trace_card_renders_under_inference_mode() -> None:
    """The failure card renders inside ``torch.inference_mode`` too.

    The failure card matters most when a forward has failed, and an
    inference-mode forward is where the banner itself once raised. The
    frontier names committed records by label; it never inlines a
    record's tensor dump.
    """

    class DiesInInference(nn.Module):
        """Fixture model that fails after two committed ops."""

        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Raise after the linear and relu ops are committed."""
            torch.relu(self.a(x))
            raise RuntimeError("inference boom")

    partial = None
    with torch.inference_mode():
        try:
            tl.trace(DiesInInference(), torch.randn(2, 4))
        except TraceCaptureError as error:
            assert isinstance(error.__cause__, RuntimeError)
            partial = error.partial_trace
        assert partial is not None
        html = partial._repr_html_()
    assert "card unavailable" not in html and "Traceback" not in html
    assert "FAILED CAPTURE" in html
    assert "inference boom" in html
    assert "committed prefix" in html
    frontier = re.search(r"frontier \(last committed\): ([^<]*)<", html)
    assert frontier is not None
    assert "relu" in frontier.group(1)
    assert "Tensor contents" not in html


def test_render_side_effect_free(recurrent_log) -> None:
    """Pinned regression: rendering must not perturb the trace."""

    before = (
        len(recurrent_log.ops),
        tuple(recurrent_log.layer_labels),
        str(recurrent_log.outcome.status),
    )
    recurrent_log._repr_html_()
    recurrent_log["relu_1_2:2"]._repr_html_()
    recurrent_log._layer_logs["relu_1_2"]._repr_html_()
    after = (
        len(recurrent_log.ops),
        tuple(recurrent_log.layer_labels),
        str(recurrent_log.outcome.status),
    )
    assert before == after


# ---------------------------------------------------------------------------
# Copy-root ladder (memo section 4 + F-G repair)


def test_copy_root_ladder_rungs() -> None:
    """Explicit wins; treescope path second; no root -> disabled with reason."""

    target = object()
    explicit = resolve_copy_root(target, explicit_root="log", treescope_path=".x")
    assert (explicit.expression, explicit.source) == ("log", "explicit")
    nested = resolve_copy_root(target, treescope_path=".x")
    assert (nested.expression, nested.source) == (".x", "treescope_path")
    disabled = resolve_copy_root(target, allow_namespace_scan=False)
    assert disabled.expression is None and disabled.source == "disabled"
    assert disabled.reason and "key" in disabled.reason


def test_copy_expression_composition() -> None:
    """Key expressions compose ROOT[key]; keyless roots copy the literal."""

    assert CopyRoot("log", "explicit").key_expression("relu_1_2") == "log['relu_1_2']"
    assert CopyRoot(None, "disabled", "why").key_expression("relu_1_2") == "'relu_1_2'"


# ---------------------------------------------------------------------------
# Motifs, bounds, grid, axis labels, SDPA hint


def test_grid_motifs_truncation_and_empty() -> None:
    """Grid: motif glyphs render, truncation discloses, EMPTY is explicit."""

    tensor = torch.randn(4, 6)
    tensor[0, 0] = float("nan")
    tensor[1, 2] = float("inf")
    render = array_grid_html(tensor)
    assert "tl-motif-nan" in render.html and "tl-motif-posinf" in render.html
    big = array_grid_html(torch.randn(300, 300))
    assert big.truncated and "tl-motif-masked" in big.html
    assert "truncated to" in big.disclosure
    empty = array_grid_html(torch.empty(0, 4))
    assert "EMPTY" in empty.html and empty.cells_total == 0


def test_grid_facet_gaps_outer_2x_inner() -> None:
    """Static facet convention: outer group gaps render 2x the inner gap."""

    render = array_grid_html(torch.randn(2, 3, 8, 8))
    assert "border-top:4px" in render.html  # outer boundary (level 2)
    assert "border-top:2px" in render.html  # inner boundary (level 1)


def test_axis_labels_positional_fallback_disclosed() -> None:
    """No proven roles -> positional badges, provenance says so."""

    labels = axis_labels((2, 3), None)
    assert [label.badge for label in labels] == ["axis 0:2", "axis 1:3"]
    assert all(label.provenance == "positional" for label in labels)
    roled = axis_labels((2, 3), ("batch", None))
    assert roled[0].badge == "batch:2" and roled[0].provenance == "role"
    assert roled[1].provenance == "positional"
    # Wrong-length roles are ignored entirely -- labels never guess.
    assert all(label.provenance == "positional" for label in axis_labels((2, 3), ("batch",)))


def test_token_labels_threshold_and_degrade() -> None:
    """Token decode is opt-in above the threshold and never raises."""

    lookup = {0: "<bos>", 1: "hello"}.__getitem__
    assert decode_token_labels([0, 1], lookup) == ("<bos>", "hello")
    assert decode_token_labels([0, 7], lookup) == ("<bos>", "7")  # KeyError degrades
    assert decode_token_labels(list(range(100)), lookup) is None  # over threshold
    assert decode_token_labels([0], None) is None


def test_sdpa_hint_fires_only_without_softmax() -> None:
    """The SDPA product hint: sdpa present + softmax absent."""

    class FusedAttention(nn.Module):
        """Minimal model whose capture contains only fused attention."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """One fused scaled-dot-product-attention call."""
            return torch.nn.functional.scaled_dot_product_attention(x, x, x)

    fused_log = tl.trace(FusedAttention(), torch.randn(2, 3, 4, 8))
    hint = sdpa_attention_hint(fused_log)
    assert hint is not None and "attn_implementation='eager'" in hint
    assert "attention weights were never materialized" in fused_log._repr_html_()

    class EagerAttention(nn.Module):
        """Attention with an explicit softmax -- the hint must NOT fire."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Materialized attention pattern via softmax."""
            scores = torch.softmax(x @ x.transpose(-1, -2), dim=-1)
            return scores @ x

    eager_log = tl.trace(EagerAttention(), torch.randn(2, 4, 8))
    assert sdpa_attention_hint(eager_log) is None


def test_namespace_scan_excludes_history_names() -> None:
    """F-G repair: the identity scan never picks `_`/history slots."""

    ipython = pytest.importorskip("IPython")
    shell = ipython.core.interactiveshell.InteractiveShell.instance()
    target = object()
    shell.user_ns.update(
        {"_": target, "_3": target, "_i7": target, "_ih": [target], "mylog": target, "zz": target}
    )
    try:
        resolved = resolve_copy_root(target)
        # shortest-then-lexicographic among the SURVIVING names: "zz".
        assert resolved.source == "namespace" and resolved.expression == "zz"
    finally:
        for name in ("_", "_3", "_i7", "mylog", "zz"):
            shell.user_ns.pop(name, None)
