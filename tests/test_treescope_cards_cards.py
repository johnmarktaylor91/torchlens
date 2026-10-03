"""F16 B2/B3: the four cards, the copy-root ladder, motifs, and the grid.

Test-discipline law (treescope memo s12 rule 4): card tests assert
CONTENT, never non-emptiness -- the historical ``_repr_html_`` returned a
94-byte plain repr without IPython that passed any non-empty check.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.notebook._access import CopyRoot, resolve_copy_root
from torchlens.notebook._axis_labels import axis_labels, decode_token_labels, sdpa_attention_hint
from torchlens.notebook._grid import array_grid_html
from torchlens.notebook._motifs import classify_value, resolve_signed_bounds
from torchlens.notebook.cardtree import Card, CardText, render_card_html


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


@pytest.mark.smoke
def test_trace_card_content_and_proof_row(recurrent_log) -> None:
    """The Trace card carries identity, counts, keys, and the PROOF row."""

    html = recurrent_log._repr_html_()
    assert "TorchLens Trace" in html
    assert "capture proof" in html
    assert "capture outcome:" in html
    assert "save policy:" in html
    assert "lookup keys" in html
    assert "data-lookup-key" in html
    assert "card unavailable" not in html


@pytest.mark.smoke
def test_layer_card_pass_rows_are_the_shape(recurrent_log) -> None:
    """A reused layer renders N pass rows via ops[k]; no pooled stats."""

    html = recurrent_log.layer_logs["relu_1_2"]._repr_html_()
    assert "3 passes" in html
    for pass_index in (1, 2, 3):
        assert f"pass {pass_index}/3: relu_1_2:{pass_index}" in html
    assert "card unavailable" not in html


def test_single_pass_layer_degenerates_to_op_card(recurrent_log) -> None:
    """ONE code path: a single-pass layer renders the Op card."""

    html = recurrent_log.layer_logs["linear_2_3"]._repr_html_()
    assert "TorchLens Op" in html
    assert "card unavailable" not in html


@pytest.mark.smoke
def test_op_card_zones(recurrent_log) -> None:
    """Op card: identity, grid, wrappers untouched, neighbors, proof."""

    html = recurrent_log["relu_1_2:2"]._repr_html_()
    assert "TorchLens Op: relu_1_2:2" in html
    assert "(pass 2/3)" in html
    assert "tl-grid" in html  # budgeted array view
    assert "flops fwd" in html and "time:" in html  # str(wrapper), untouched
    assert "capture proof" in html


def test_op_card_unsaved_value_discloses(recurrent_log) -> None:
    """save= exclusion renders an honest reason, never a pretended value."""

    log = tl.trace(_Recurrent(), torch.randn(2, 4), save=tl.func("linear"))
    html = log["relu_1_2:1"]._repr_html_()
    assert "value not saved" in html
    assert '<table class="tl-grid-table"' not in html


def test_partial_trace_card_failure_first() -> None:
    """PartialTrace card: banner, phase, escaped exception, prefix."""

    class Dies(nn.Module):
        """Fixture model that fails mid-forward."""

        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Raise after one committed op, with hostile markup in the text."""
            torch.relu(self.a(x))
            raise RuntimeError("<script>alert(1)</script> boom")

    with pytest.raises(RuntimeError):
        tl.trace(Dies(), torch.randn(2, 4))
    # Recover the partial through the documented door.
    try:
        tl.trace(Dies(), torch.randn(2, 4))
    except RuntimeError as error:
        partial = tl.partial.from_failed_capture(error)
    assert partial is not None
    html = partial._repr_html_()
    assert "FAILED CAPTURE" in html
    assert "committed prefix" in html
    # Hostile exception text is escaped at the leaf, never live markup.
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


@pytest.mark.smoke
def test_inference_mode_capture_renders_without_raise() -> None:
    """Composition row X9: inference-mode captures never paint a traceback."""

    with torch.inference_mode():
        log = tl.trace(nn.Sequential(nn.Linear(4, 4)), torch.randn(2, 4))
    html = log._repr_html_()
    assert "TorchLens Trace" in html
    assert "Traceback" not in html and "card unavailable" not in html


def test_render_side_effect_free(recurrent_log) -> None:
    """Pinned regression: rendering must not perturb the trace."""

    before = (
        recurrent_log.num_ops,
        tuple(recurrent_log.layer_labels),
        str(recurrent_log.outcome.status),
    )
    recurrent_log._repr_html_()
    recurrent_log["relu_1_2:2"]._repr_html_()
    recurrent_log.layer_logs["relu_1_2"]._repr_html_()
    after = (
        recurrent_log.num_ops,
        tuple(recurrent_log.layer_labels),
        str(recurrent_log.outcome.status),
    )
    assert before == after


def test_hostile_labels_escape_at_the_leaf() -> None:
    """Escape-once law: text is DATA; markup never survives the leaf."""

    fragment = render_card_html(
        Card(title="<script>alert(1)</script>", children=(CardText("<b>x</b> & y"),))
    )
    assert "<script>alert" not in fragment
    assert "&lt;script&gt;" in fragment and "&lt;b&gt;x&lt;/b&gt; &amp; y" in fragment


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


def test_copy_expression_composition() -> None:
    """Key expressions compose ROOT[key]; keyless roots copy the literal."""

    assert CopyRoot("log", "explicit").key_expression("relu_1_2") == "log['relu_1_2']"
    assert CopyRoot(None, "disabled", "why").key_expression("relu_1_2") == "'relu_1_2'"


# ---------------------------------------------------------------------------
# Motifs, bounds, grid, axis labels, SDPA hint


def test_six_state_classification_and_bounds() -> None:
    """NaN/Inf/masked/out-of-range/finite classify; all-positive discloses."""

    bounds = resolve_signed_bounds(-5.0, 2.0, 0.0, 0.5)
    assert bounds.mode == "diverging" and bounds.trimmed
    assert bounds.vmin == -bounds.vmax
    assert "3-sigma trim" in bounds.disclosure
    assert classify_value(float("nan"), True, bounds) == "nan"
    assert classify_value(float("inf"), True, bounds) == "posinf"
    assert classify_value(float("-inf"), True, bounds) == "neginf"
    assert classify_value(0.5, False, bounds) == "masked"
    assert classify_value(4.9, True, bounds) == "out_of_range"
    assert classify_value(0.1, True, bounds) == "finite"
    positive = resolve_signed_bounds(0.0, 3.0, 1.0, 0.5)
    assert positive.mode == "sequential" and "all-positive" in positive.disclosure


@pytest.mark.smoke
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


@pytest.mark.smoke
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


@pytest.mark.smoke
def test_token_labels_threshold_and_degrade() -> None:
    """Token decode is opt-in above the threshold and never raises."""

    lookup = {0: "<bos>", 1: "hello"}.__getitem__
    assert decode_token_labels([0, 1], lookup) == ("<bos>", "hello")
    assert decode_token_labels([0, 7], lookup) == ("<bos>", "7")  # KeyError degrades
    assert decode_token_labels(list(range(100)), lookup) is None  # over threshold
    assert decode_token_labels([0], None) is None


@pytest.mark.smoke
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
