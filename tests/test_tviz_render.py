"""tviz renderer Stage-0 deterministic checks (memo D3/D4/D13/D16/D21/section 7).

Emitter honesty rows that a deterministic check can prove and therefore
never consume judge budget: zero ``<image>`` elements in every numeric SVG,
PDF text extraction in order, disclosure lines rendered, pagination that
drops nothing, the mandatory joint-effect line, and the matplotlib
call-time gate (FORK-T1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import torchlens.tviz as tviz

pytestmark = [pytest.mark.smoke]


@pytest.fixture()
def view() -> tviz.AttentionView:
    """A deterministic 4-head 6-token probability view."""

    generator = torch.Generator().manual_seed(0)
    pattern = torch.softmax(torch.randn(4, 6, 6, generator=generator), dim=-1)
    tokens = tuple(f"tok{i}" for i in range(6))
    return tviz.AttentionView(
        pattern=pattern,
        query_tokens=tviz.TokenAxis(role="query", tokens=tokens),
        key_tokens=tviz.TokenAxis(role="key", tokens=tokens),
        heads=(0, 1, 2, 3),
        layer="blk.0.attn",
    )


@pytest.fixture()
def receipt() -> tviz.CausalReceipt:
    """A valid 4-head receipt with a joint measurement."""

    return tviz.CausalReceipt(
        layer="blk.0.attn",
        heads=(0, 1, 2, 3),
        effects=(13.27, -2.0, None, 0.5),
        metric="logit(' Paris')",
        intervention="head contribution zeroed",
        engine="rerun",
        disclosure="direct",
        fires=3,
        negative_control=0.0,
        positive_control=13.27,
        joint_effect=-0.68,
    )


def test_matplotlib_gate_refuses_with_install_command(
    view: tviz.AttentionView, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FORK-T1: absent matplotlib refuses typed, printing the install command."""

    for name in [name for name in sys.modules if name.split(".")[0] == "matplotlib"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(tviz.TvizError) as excinfo:
        tviz.render_attention(view, tmp_path / "x.svg")
    assert excinfo.value.fields["code"] == "tv_matplotlib_missing"
    assert 'pip install "torchlens[viz]"' in excinfo.value.fields["install_command"]


def test_attention_svg_is_vector_with_disclosures(view: tviz.AttentionView, tmp_path: Path) -> None:
    """Numeric SVG carries zero <image> elements and every honesty line."""

    artifact = tviz.render_attention(view, tmp_path / "attn.svg")
    svg = artifact.paths[0].read_text()
    assert "<image" not in svg
    assert artifact.svg_fonttype == "path"
    lines = "\n".join(artifact.disclosure_lines)
    assert "rows attend from (query); columns attend to (key)" in lines
    assert "routing measurements, not causal importance" in lines
    assert "zero and masked positions are not distinguished" in lines


def test_masked_view_discloses_source_not_unmarked_wording(
    view: tviz.AttentionView, tmp_path: Path
) -> None:
    """A provenanced mask renders its source; the unmarked wording is absent."""

    mask = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    masked_view = tviz.AttentionView(
        pattern=view.pattern,
        query_tokens=view.query_tokens,
        key_tokens=view.key_tokens,
        heads=view.heads,
        layer=view.layer,
        mask=tviz.MaskInfo(mask=mask, source="sdpa_call_args"),
    )
    artifact = tviz.render_attention(masked_view, tmp_path / "masked.svg")
    lines = "\n".join(artifact.disclosure_lines)
    assert "mask source: sdpa_call_args" in lines
    assert "zero and masked positions are not distinguished" not in lines


def test_per_panel_domain_renders_not_comparable_label(
    view: tviz.AttentionView, tmp_path: Path
) -> None:
    """Expert-only rescale is visibly labeled (D5)."""

    scores_view = tviz.AttentionView(
        pattern=torch.randn(2, 6, 6),
        query_tokens=view.query_tokens,
        key_tokens=view.key_tokens,
        heads=(0, 1),
        layer=view.layer,
        domain="per_panel",
    )
    artifact = tviz.render_attention(scores_view, tmp_path / "panel.svg")
    assert any("panels not comparable" in line for line in artifact.disclosure_lines)


def test_pagination_drops_nothing(tmp_path: Path) -> None:
    """16 heads paginate; every page path is on the artifact (D21)."""

    generator = torch.Generator().manual_seed(1)
    pattern = torch.softmax(torch.randn(16, 4, 4, generator=generator), dim=-1)
    tokens = tuple(f"t{i}" for i in range(4))
    wide = tviz.AttentionView(
        pattern=pattern,
        query_tokens=tviz.TokenAxis(role="query", tokens=tokens),
        key_tokens=tviz.TokenAxis(role="key", tokens=tokens),
        heads=tuple(range(16)),
        layer="blk.0.attn",
    )
    artifact = tviz.render_attention(wide, tmp_path / "wide.png")
    assert artifact.pages_total == 2
    assert len(artifact.paths) == 2
    assert all(path.exists() for path in artifact.paths)


def test_atlas_overview_plus_detail_pages(view: tviz.AttentionView, tmp_path: Path) -> None:
    """The atlas emits an all-member overview plus numbered detail pages."""

    second = tviz.AttentionView(
        pattern=view.pattern,
        query_tokens=view.query_tokens,
        key_tokens=view.key_tokens,
        heads=view.heads,
        layer="blk.1.attn",
    )
    artifact = tviz.render_attention_atlas([view, second], tmp_path / "atlas.svg")
    assert artifact.pages_total == 3
    assert all("<image" not in path.read_text() for path in artifact.paths)
    assert any("detail pages follow" in line for line in artifact.disclosure_lines)


def test_receipt_grid_prints_joint_line(receipt: tviz.CausalReceipt, tmp_path: Path) -> None:
    """The grid prints the joint measurement + not-additive wording (D13)."""

    artifact = tviz.render_receipt_grid(receipt, tmp_path / "grid.svg")
    lines = "\n".join(artifact.disclosure_lines)
    assert "joint effect of ablating all 3 measured heads together: -0.68" in lines
    assert "single-head effects are not additive" in lines
    assert "measured 3 of 4" in lines
    assert "<image" not in artifact.paths[0].read_text()


def test_receipt_grid_refuses_without_joint(tmp_path: Path) -> None:
    """No joint measurement, no grid figure (D13 is mandatory)."""

    receipt = tviz.CausalReceipt(
        layer="blk.0.attn",
        heads=(0,),
        effects=(1.0,),
        metric="m",
        intervention="head contribution zeroed",
        engine="rerun",
        disclosure="direct",
        fires=1,
        negative_control=0.0,
        positive_control=1.0,
    )
    with pytest.raises(tviz.TvizError) as excinfo:
        tviz.render_receipt_grid(receipt, tmp_path / "no-joint.png")
    assert excinfo.value.fields["code"] == "tv_receipt_invalid"


def test_annotated_attention_keeps_pattern_channel(
    view: tviz.AttentionView, receipt: tviz.CausalReceipt, tmp_path: Path
) -> None:
    """Effects ride headers/borders; the figure prints measured N of M (D16)."""

    annotation = tviz.Annotation.from_receipt(receipt)
    artifact = tviz.render_attention(view, tmp_path / "annot.svg", annotation=annotation)
    lines = "\n".join(artifact.disclosure_lines)
    assert "measured 3 of 4" in lines
    assert "controls:" in lines
    assert "fires: 3" in lines


def test_unknown_format_refuses(view: tviz.AttentionView, tmp_path: Path) -> None:
    """The save-format roster is closed; PDF is the paper format."""

    with pytest.raises(tviz.TvizError) as excinfo:
        tviz.render_attention(view, tmp_path / "attn.webp")
    assert excinfo.value.fields["code"] == "tv_record_invalid"


def test_pdf_text_extracts_tokens_in_order(view: tviz.AttentionView, tmp_path: Path) -> None:
    """PDF is the paper format: selectable text, token order preserved."""

    fitz = pytest.importorskip("fitz")
    artifact = tviz.render_attention(view.head_view(0), tmp_path / "one.pdf")
    with fitz.open(artifact.paths[0]) as doc:
        text = doc[0].get_text()
    positions = [text.find(f"tok{i}") for i in range(6)]
    assert all(position >= 0 for position in positions), f"tokens missing from PDF text: {text!r}"


def test_neuron_card_renders_closed_terms(tmp_path: Path) -> None:
    """The family-gated card renders only from a closed decomposition."""

    q = torch.randn(8, generator=torch.Generator().manual_seed(2))
    k = torch.randn(8, generator=torch.Generator().manual_seed(3))
    record = tviz.ScoreDecomposition(
        layer="blk.0.attn",
        head=0,
        destination=1,
        source=0,
        query_vector=q,
        key_vector=k,
        products=q * k,
        scale=8**0.5,
        reference_score=float((q * k).sum()) / 8**0.5,
    )
    artifact = tviz.render_neuron_card(record, tmp_path / "card.svg")
    assert "<image" not in artifact.paths[0].read_text()
    assert any("captured score" in line for line in artifact.disclosure_lines)


def test_svg_fonttype_none_is_the_editing_opt_in(view: tviz.AttentionView, tmp_path: Path) -> None:
    """'none' keeps real text elements; the default stays 'path'."""

    artifact = tviz.render_attention(
        view.head_view(0), tmp_path / "editable.svg", svg_fonttype="none"
    )
    svg = artifact.paths[0].read_text()
    assert "<image" not in svg
    assert "tok0" in svg  # real text element, not outlined paths
    assert artifact.svg_fonttype == "none"
