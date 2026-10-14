"""Layers-by-positions logit-lens heatmap over ``LogitLensResult``.

Cell values are checked against an independent softmax of the same
logits; the footer must carry the projection and validation wording, and
every out-of-contract request refuses instead of cropping or guessing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import torchlens.tviz as tviz
from torchlens.semantic.logit_lens import LogitLensEntry, LogitLensResult
from torchlens.tviz._lens_heatmap import MAX_HEATMAP_POSITIONS, _clip, _heatmap_cells

VOCAB = 13


class _Tokenizer:
    """Minimal tokenizer: id -> word."""

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [f"w{token_id}" if token_id != 3 else "averyverylongtoken" for token_id in ids]


def _result(*, n_layers: int = 3, seq: int = 5, validated: bool = True) -> LogitLensResult:
    """Deterministic synthetic lens with a sharpening answer across layers."""

    generator = torch.Generator().manual_seed(0)
    entries = []
    for layer in range(n_layers):
        logits = torch.randn(2, seq, VOCAB, generator=generator)
        logits[..., 3] += 2.0 * layer
        entries.append(
            LogitLensEntry(
                address=f"h.{layer}", layer_index=layer, facet="resid_post", logits=logits
            )
        )
    return LogitLensResult(
        entries=tuple(entries),
        facet="resid_post",
        lens_source="model_head",
        validated=validated,
        final_logits=None,
    )


def test_cells_match_an_independent_softmax() -> None:
    """Top-1 ids, top-1 probabilities and target probabilities are exact."""

    result = _result()
    columns = [0, 2, 4]
    values, top_ids = _heatmap_cells(result, columns=columns, batch_index=1, target_token_id=None)
    target, _ = _heatmap_cells(result, columns=columns, batch_index=1, target_token_id=7)
    for row, entry in enumerate(result.entries):
        probs = torch.nn.functional.softmax(entry.logits[1, columns].double(), dim=-1)
        assert torch.allclose(torch.from_numpy(values[row]), probs.max(-1).values, atol=1e-6)
        assert top_ids[row].tolist() == probs.argmax(-1).tolist()
        assert torch.allclose(torch.from_numpy(target[row]), probs[:, 7], atol=1e-6)


def test_render_writes_vector_output_with_the_footer(tmp_path: Path) -> None:
    """SVG output is vector-only and the footer states what each row is."""

    artifact = tviz.render_logit_lens_heatmap(
        _result(validated=False),
        tmp_path / "lens.svg",
        tokens=["The", "cat", "sat", "on", "the"],
        tokenizer=_Tokenizer(),
    )
    text = artifact.paths[0].read_text()
    assert "<image" not in text
    lines = "\n".join(artifact.disclosure_lines)
    assert "UNVALIDATED" in lines
    assert "3 rows projected through final norm/head" in lines
    assert "none is the model's own output" in lines
    assert "full-vocabulary denominator" in lines
    # The long token is clipped in its cell and the clip is disclosed.
    assert "token text clipped to 7 characters" in lines


def test_whitespace_tokens_stay_visible_and_long_tokens_clip() -> None:
    """A newline or space prediction never renders as an empty cell."""

    assert _clip("\n") == ("'\\n'", False)
    assert _clip(" ") == ("' '", False)
    assert _clip("averyverylongtoken") == ("averyv~", True)
    assert _clip("Paris") == ("Paris", False)


def test_target_mode_names_the_target_and_pdf_renders(tmp_path: Path) -> None:
    """``target_prob`` colors by the tracked token and says which one."""

    artifact = tviz.render_logit_lens_heatmap(
        _result(),
        tmp_path / "lens.pdf",
        value="target_prob",
        target_token_id=3,
        tokenizer=_Tokenizer(),
        positions=[-2, -1],
    )
    assert artifact.paths[0].stat().st_size > 0
    assert any("target token 'averyverylongtoken'" in line for line in artifact.disclosure_lines)
    assert any("numerically validated" in line for line in artifact.disclosure_lines)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"value": "rank"},
        {"value": "target_prob"},
        {"target_token_id": 2},
        {"value": "target_prob", "target_token_id": VOCAB},
        {"tokens": ["a", "b"]},
        {"positions": [5]},
        {"positions": []},
    ],
)
def test_out_of_contract_requests_refuse(tmp_path: Path, kwargs: dict) -> None:
    """Bad requests refuse typed before anything is drawn."""

    with pytest.raises(tviz.TvizError) as caught:
        tviz.render_logit_lens_heatmap(_result(), tmp_path / "x.png", **kwargs)
    assert caught.value.fields["code"] == "tv_record_invalid"
    assert not (tmp_path / "x.png").exists()


def test_wide_lens_refuses_rather_than_cropping(tmp_path: Path) -> None:
    """Above the column cap the caller must name positions; nothing is cut."""

    wide = _result(n_layers=1, seq=MAX_HEATMAP_POSITIONS + 1)
    with pytest.raises(tviz.TvizError, match="never crops silently"):
        tviz.render_logit_lens_heatmap(wide, tmp_path / "x.png")
    artifact = tviz.render_logit_lens_heatmap(wide, tmp_path / "x.png", positions=range(10))
    assert artifact.paths[0].exists()


def test_positionless_lens_refuses(tmp_path: Path) -> None:
    """Classifier-style ``[batch, classes]`` lenses have no position axis."""

    entry = LogitLensEntry(
        address="h.0", layer_index=0, facet="resid_post", logits=torch.randn(1, 5)
    )
    result = LogitLensResult(
        entries=(entry,),
        facet="resid_post",
        lens_source="classifier_head",
        validated=True,
        final_logits=None,
    )
    with pytest.raises(tviz.TvizError, match="needs \\[batch, positions, vocab\\]"):
        tviz.render_logit_lens_heatmap(result, tmp_path / "x.png")
