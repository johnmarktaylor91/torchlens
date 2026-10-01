"""CardTree IR + safe renderer tests (lane C05; treescope memo B1).

Pins the presentation-only IR contracts: typed-leaf escaping, the
never-raise boundary, budgeted collections with disclosed truncation,
versioned scoped CSS on themes tokens, native ``<details>`` folding,
clipboard keys with click-to-select degradation, stable ``data-lookup-key``
attributes -- and the migration of today's Trace/PartialTrace HTML through
it with the IPython generation gate REMOVED (card tests assert CONTENT).
"""

from __future__ import annotations

import builtins

import pytest
import torch

import torchlens as tl
from torchlens.notebook.cardtree import (
    CARD_CSS_VERSION,
    Card,
    CardCollection,
    CardKey,
    CardSection,
    CardText,
    card_css,
    render_card_html,
    safe_card_html,
)

pytestmark = pytest.mark.smoke


def test_typed_leaf_escaping_happens_exactly_once() -> None:
    html = render_card_html(
        Card(
            title="<script>alert(1)</script>",
            children=(CardText("a < b & c > d"),),
        ),
        include_css=False,
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "a &lt; b &amp; c &gt; d" in html


def test_never_raise_boundary_degrades_to_one_line() -> None:
    def exploding_build() -> Card:
        raise ValueError("user data: <boom>")

    html = safe_card_html(exploding_build)
    assert "card unavailable: ValueError" in html
    # Exception text is user data: escaped, never markup.
    assert "<boom>" not in html
    assert "&lt;boom&gt;" in html


def test_budgeted_collection_discloses_truncation() -> None:
    collection = CardCollection(
        children=tuple(CardText(f"row {index}") for index in range(30)), budget=5
    )
    html = render_card_html(Card(title="t", children=(collection,)), include_css=False)
    assert "row 4" in html
    assert "row 5" not in html
    assert "... 25 more (30 total)" in html


def test_css_is_versioned_and_scoped_on_theme_tokens() -> None:
    css = card_css("torchlens")
    assert f".{CARD_CSS_VERSION}" in css
    # No unscoped selectors: every rule mentions the versioned root.
    for rule in css.removeprefix("<style>").removesuffix("</style>").split("}"):
        if rule.strip():
            assert CARD_CSS_VERSION in rule
    dark = card_css("dark")
    assert dark != css  # tokens come from the themes presets


def test_details_folding_is_native_html() -> None:
    html = render_card_html(
        Card(
            title="t",
            children=(
                CardSection(title="closed", children=(CardText("hidden"),), folded=True),
                CardSection(title="opened", children=(CardText("shown"),), folded=False),
            ),
        ),
        include_css=False,
    )
    assert "<details><summary>closed</summary>" in html
    assert "<details open><summary>opened</summary>" in html


def test_card_key_carries_stable_lookup_attribute_and_degradation() -> None:
    html = render_card_html(Card(title="t", children=(CardKey("log['relu_1_2']"),)))
    assert 'data-lookup-key="log[&#x27;relu_1_2&#x27;]"' in html
    assert "navigator.clipboard" in html
    # JS-off degradation: one click selects the whole expression via CSS.
    assert "user-select:all" in card_css()


def test_trace_repr_html_has_no_ipython_gate(monkeypatch) -> None:
    # The silent IPython gate returned a 94-byte plain repr without IPython,
    # which also let naive card tests pass against the fallback. Content must
    # render identically with IPython unimportable.
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    log = tl.trace(model, torch.randn(2, 4))

    real_import = builtins.__import__

    def no_ipython(name, *args, **kwargs):
        if name == "IPython" or name.startswith("IPython."):
            raise ImportError("IPython disabled for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ipython)
    html = log._repr_html_()
    assert CARD_CSS_VERSION in html
    assert "TorchLens Trace" in html
    assert "Layers:" in html
    assert "lookup keys" in html
    assert "data-lookup-key" in html


def test_trace_repr_html_stays_bounded() -> None:
    model = torch.nn.Sequential(*(torch.nn.Linear(4, 4) for _ in range(30)))
    log = tl.trace(model, torch.randn(2, 4))
    html = log._repr_html_()
    # The key index is budgeted: truncation is disclosed, size stays O(50KB).
    assert "more (" in html
    assert len(html) < 50_000


def test_partial_trace_repr_html_is_failure_first() -> None:
    class Exploding(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            torch.relu(x)
            raise RuntimeError("mid-forward failure <with markup>")

    with pytest.raises(RuntimeError) as excinfo:
        tl.trace(Exploding(), torch.randn(2, 4))
    partial = tl.partial.from_failed_capture(excinfo.value)
    html = partial._repr_html_()
    assert "FAILED CAPTURE" in html
    assert "PartialTrace" in html
    assert "&lt;with markup&gt;" in html
    assert "<with markup>" not in html
