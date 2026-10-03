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
import torch.nn.functional as F

import torchlens.tviz as tviz
from torchlens.semantic.logit_lens import (
    PROVENANCE_NATIVE,
    PROVENANCE_PROJECTED,
    LogitLensPrediction,
    LogitLensPredictions,
)


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

    @pytest.mark.smoke
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

    @pytest.mark.smoke
    def test_svg_has_no_image_and_no_network(self) -> None:
        """The SVG emitter is self-contained vector output."""

        svg = tviz.token_strip_svg(_scores())
        assert "<image" not in svg
        assert "http" not in svg.replace("http://www.w3.org/2000/svg", "")
        assert "factor 0" in svg

    def test_multi_row_shares_one_token_axis(self) -> None:
        """The NMF multi-row form: every row over the same tokens."""

        svg = tviz.token_strip_svg(_scores())
        assert svg.count("cat") == 2  # one per row


def test_paper_strip_renders_all_formats(tmp_path: Path) -> None:
    """The matplotlib paper path emits vector SVG and PDF."""

    record = _scores()
    for fmt in ("svg", "pdf", "png"):
        artifact = tviz.render_token_strip(record, tmp_path / f"strip.{fmt}")
        assert artifact.paths[0].exists()
    assert "<image" not in (tmp_path / "strip.svg").read_text()


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
        validated=True,
    )


class TestPredictionPictures:
    """Ribbon / trajectory / table with load-bearing provenance wording."""

    @pytest.mark.smoke
    def test_trajectory_carries_per_row_provenance(self) -> None:
        """Projected rows never claim native output."""

        trajectory = tviz.prediction_trajectory(_predictions(), target_token_id=7)
        assert trajectory.provenance == (
            PROVENANCE_PROJECTED,
            PROVENANCE_PROJECTED,
            PROVENANCE_NATIVE,
        )
        assert trajectory.target_ranks is not None
        assert all(rank >= 1 for rank in trajectory.target_ranks)  # one-based

    def test_untracked_target_refuses(self) -> None:
        """A target the extractor did not retain refuses with the remedy."""

        with pytest.raises(tviz.TvizError) as excinfo:
            tviz.prediction_trajectory(_predictions(), target_token_id=999)
        assert excinfo.value.fields["code"] == "tv_record_invalid"

    @pytest.mark.smoke
    def test_ribbon_marks_native_rows(self, tmp_path: Path) -> None:
        """The ribbon renders with the lens-validation disclosure."""

        trajectory = tviz.prediction_trajectory(_predictions())
        artifact = tviz.render_prediction_ribbon(trajectory, tmp_path / "ribbon.svg")
        lines = "\n".join(artifact.disclosure_lines)
        assert "full-vocabulary denominator" in lines
        assert "1 native-output rows" in lines
        assert "<image" not in artifact.paths[0].read_text()

    def test_answer_trajectory_requires_target(self, tmp_path: Path) -> None:
        """No tracked token, no answer panel."""

        trajectory = tviz.prediction_trajectory(_predictions())
        with pytest.raises(tviz.TvizError) as excinfo:
            tviz.render_answer_trajectory(trajectory, tmp_path / "answer.png")
        assert excinfo.value.fields["code"] == "tv_record_invalid"

    @pytest.mark.smoke
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
            validated=True,
        )
        with pytest.raises(tviz.TvizError) as excinfo:
            tviz.prediction_table(projected_only)
        assert excinfo.value.fields["code"] == "tv_record_invalid"

    @pytest.mark.smoke
    def test_table_pagination_whole_positions(self, tmp_path: Path) -> None:
        """Table pages keep whole positions and list every page."""

        table = tviz.prediction_table(_predictions())
        artifact = tviz.render_prediction_table(table, tmp_path / "table.pdf")
        assert artifact.pages_total == 1
        assert artifact.paths[0].exists()


class TestMetricOracles:
    """Loss/entropy against independent PyTorch math (composition row 9)."""

    def test_loss_matches_unreduced_shifted_cross_entropy(self) -> None:
        """loss[i] == CE(logits[i-1], token[i]); first cell N/A."""

        generator = torch.Generator().manual_seed(4)
        logits = torch.randn(7, 50, generator=generator)
        ids = torch.randint(0, 50, (7,), generator=generator)
        loss, entropy = tviz.token_metrics(logits, ids, tuple(f"t{i}" for i in range(7)))
        assert loss.values[0] is None
        oracle = F.cross_entropy(logits[:-1], ids[1:], reduction="none")
        for value, expected in zip(loss.values[1:], oracle.tolist(), strict=True):
            assert value == pytest.approx(expected, abs=1e-5)

    def test_entropy_matches_independent_logsumexp(self) -> None:
        """entropy[i] == logsumexp(z) - sum(softmax(z) * z) at position i."""

        generator = torch.Generator().manual_seed(5)
        logits = torch.randn(5, 40, generator=generator)
        ids = torch.randint(0, 40, (5,), generator=generator)
        _, entropy = tviz.token_metrics(logits, ids, tuple(f"t{i}" for i in range(5)))
        probs = torch.softmax(logits, dim=-1)
        oracle = -(probs * torch.log(probs)).sum(dim=-1)
        for value, expected in zip(entropy.values, oracle.tolist(), strict=True):
            assert value == pytest.approx(expected, abs=1e-4)

    @pytest.mark.smoke
    def test_one_source_fingerprint_is_the_alignment_proof(self) -> None:
        """Both strips carry the SAME logits fingerprint."""

        generator = torch.Generator().manual_seed(6)
        logits = torch.randn(4, 30, generator=generator)
        ids = torch.randint(0, 30, (4,), generator=generator)
        loss, entropy = tviz.token_metrics(logits, ids, ("a", "b", "c", "d"))
        assert loss.source_fingerprint == entropy.source_fingerprint

    def test_planted_off_by_one_is_caught(self) -> None:
        """Shifting the ids by one breaks the loss oracle (the classic defect)."""

        generator = torch.Generator().manual_seed(7)
        logits = torch.randn(6, 30, generator=generator)
        ids = torch.randint(0, 30, (6,), generator=generator)
        loss, _ = tviz.token_metrics(logits, ids, tuple(f"t{i}" for i in range(6)))
        unshifted_oracle = F.cross_entropy(logits[1:], ids[1:], reduction="none")
        mismatched = sum(
            1
            for value, wrong in zip(loss.values[1:], unshifted_oracle.tolist(), strict=True)
            if abs(value - wrong) > 1e-4
        )
        assert mismatched > 0

    def test_padding_cells_are_na_on_both_strips(self) -> None:
        """Padded positions carry None, never zero."""

        generator = torch.Generator().manual_seed(8)
        logits = torch.randn(4, 30, generator=generator)
        ids = torch.randint(0, 30, (4,), generator=generator)
        mask = torch.tensor([1, 1, 0, 0])
        loss, entropy = tviz.token_metrics(
            logits, ids, ("a", "b", "<pad>", "<pad>"), padding_mask=mask
        )
        assert loss.values[2] is None and loss.values[3] is None
        assert entropy.values[2] is None and entropy.values[3] is None

    @pytest.mark.smoke
    def test_metric_strip_renders(self, tmp_path: Path) -> None:
        """The metric strip rides the token-strip renderer with conventions."""

        generator = torch.Generator().manual_seed(9)
        logits = torch.randn(4, 30, generator=generator)
        ids = torch.randint(0, 30, (4,), generator=generator)
        loss, _ = tviz.token_metrics(logits, ids, ("a", "b", "c", "d"))
        artifact = tviz.render_metric_strip(loss, tmp_path / "loss.svg")
        assert any("first token N/A" in line for line in artifact.disclosure_lines)
