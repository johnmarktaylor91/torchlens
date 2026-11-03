"""tviz strips, prediction pictures, and metric oracles (memo items 8-10).

The zero-dependency emitters (bare-install duty), the paper strip, the
lens-fed prediction pictures with their provenance wording, and the
loss/entropy alignment oracle (composition row 9: top-k + loss + entropy
from ONE logits source, position alignment proven -- off-by-one is the
classic defect).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import torchlens
import torchlens.errors
import torchlens.transformer as tviz
from torchlens._semantic.logit_lens import (
    PROVENANCE_NATIVE,
    PROVENANCE_PROJECTED,
    LogitLensPrediction,
    LogitLensPredictions,
)
from torchlens.transformer._strip import token_strip_svg as _token_strip_svg


def _scores() -> tviz.TokenScores:
    """A two-row strip with signed scores and an N/A cell."""

    return tviz.TokenScores(
        tokens=("The", "cat", "sat"),
        rows=(
            tviz.TokenScoreRow(label="ig", scores=(0.4, -0.9, None)),
            tviz.TokenScoreRow(label="factor 0", scores=(0.1, 0.0, 0.7)),
        ),
        footer_lines=("target: logits[0, 42]",),
        provenance="test rows",
    )


class TestZeroDependencyEmitters:
    """Bare-install duty: escaped, self-contained, no network, no <image>."""

    def test_html_is_escaped_and_carries_footers(self) -> None:
        """Tokens are escaped; footer disclosures always render."""

        record = tviz.TokenScores(
            tokens=("<script>", "b"),
            rows=(tviz.TokenScoreRow(label="x", scores=(1.0, None)),),
            footer_lines=("disclosure line",),
        )
        html = tviz.token_strip_html(record)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "disclosure line" in html
        assert 'title="n/a"' in html

    def test_svg_has_no_image_and_no_network(self) -> None:
        """The SVG emitter is self-contained vector output."""

        svg = _token_strip_svg(_scores())
        assert "<image" not in svg
        assert "http" not in svg.replace("http://www.w3.org/2000/svg", "")
        assert "factor 0" in svg


def _lens_row(address: str, provenance: str, *, k: int = 3, seed: int = 0) -> LogitLensPrediction:
    """Build a deterministic synthetic lens row."""

    generator = torch.Generator().manual_seed(seed)
    return LogitLensPrediction(
        address=address,
        layer_index=None if provenance == PROVENANCE_NATIVE else 0,
        facet=None if provenance == PROVENANCE_NATIVE else "resid_post",
        provenance=provenance,
        positions=(0, 1, 2),
        top_ids=torch.randint(0, 50, (1, 3, k), generator=generator),
        top_logits=torch.randn(1, 3, k, generator=generator),
        top_probs=torch.rand(1, 3, k, generator=generator) * 0.3,
        logsumexp=torch.randn(1, 3, generator=generator),
        token_ids=(7,),
        token_logits=torch.randn(1, 3, 1, generator=generator),
        token_probs=torch.rand(1, 3, 1, generator=generator),
        token_ranks=torch.randint(1, 50, (1, 3, 1), generator=generator),
    )


def _predictions() -> LogitLensPredictions:
    """Two projected rows plus the native row."""

    return LogitLensPredictions(
        rows=(
            _lens_row("h.0", PROVENANCE_PROJECTED, seed=1),
            _lens_row("h.1", PROVENANCE_PROJECTED, seed=2),
            _lens_row("lm_head", PROVENANCE_NATIVE, seed=3),
        ),
        k=3,
        facet="resid_post",
        lens_source="model_head",
        is_validated=True,
    )


class TestPredictionPictures:
    """Ribbon / trajectory / table with load-bearing provenance wording."""

    def test_untracked_target_refuses(self) -> None:
        """A target the extractor did not retain refuses with the remedy."""

        with pytest.raises(torchlens.errors.RenderError) as excinfo:
            tviz.prediction_trajectory(_predictions(), target_token_id=999)
        assert excinfo.value.fields["code"] == "tv_record_invalid"

    def test_answer_trajectory_requires_target(self, tmp_path: Path) -> None:
        """No tracked token, no answer panel."""

        trajectory = tviz.prediction_trajectory(_predictions())
        with pytest.raises(torchlens.errors.RenderError) as excinfo:
            tviz.render_answer_trajectory(trajectory, tmp_path / "answer.png")
        assert excinfo.value.fields["code"] == "tv_record_invalid"

    def test_table_serves_only_the_native_row(self, tmp_path: Path) -> None:
        """PROJECTION-AS-PREDICTION: the table refuses without a native row."""

        table = tviz.prediction_table(_predictions())
        assert table.provenance == PROVENANCE_NATIVE
        html = tviz.prediction_table_html(table)
        assert "top 1" in html
        projected_only = LogitLensPredictions(
            rows=(_lens_row("h.0", PROVENANCE_PROJECTED),),
            k=3,
            facet="resid_post",
            lens_source="model_head",
            is_validated=True,
        )
        with pytest.raises(torchlens.errors.RenderError) as excinfo:
            tviz.prediction_table(projected_only)
        assert excinfo.value.fields["code"] == "tv_record_invalid"

    def test_table_pagination_whole_positions(self, tmp_path: Path) -> None:
        """Table pages keep whole positions and list every page."""

        table = tviz.prediction_table(_predictions())
        artifact = torchlens.viz.prediction_table(table, tmp_path / "table.pdf")
        assert artifact.num_pages == 1
        assert artifact.paths[0].exists()


class TestMetricOracles:
    """Loss/entropy against independent PyTorch math (composition row 9)."""

    def test_metric_strip_renders(self, tmp_path: Path) -> None:
        """The metric strip rides the token-strip renderer with conventions."""

        generator = torch.Generator().manual_seed(9)
        logits = torch.randn(4, 30, generator=generator)
        ids = torch.randint(0, 30, (4,), generator=generator)
        loss, _ = tviz.token_metrics(logits, ids, ("a", "b", "c", "d"))
        artifact = torchlens.viz.metric_strip(loss, tmp_path / "loss.svg")
        assert any("first token N/A" in line for line in artifact.disclosure_lines)
