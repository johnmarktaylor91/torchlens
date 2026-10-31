"""tviz typed-record contract rows (memo D2/D5/D6/D12/D13/D16/D19).

Construction validation, honesty metadata, the closed annotation-kind
vocabulary, receipt attachment validation, and the sum-to-score invariant --
every ``tv_*`` refusal class is provoked here or in the render/bridge suites.
"""

from __future__ import annotations

import pytest
import torch

import torchlens.tviz as tviz


def _view(n_heads: int = 2, n_dst: int = 4, n_src: int = 4, **kwargs) -> tviz.AttentionView:
    """Build a small valid probability-domain view."""

    pattern = torch.softmax(torch.randn(n_heads, n_dst, n_src), dim=-1)
    defaults: dict = {
        "pattern": pattern,
        "query_tokens": tviz.TokenAxis(role="query", tokens=tuple(f"q{i}" for i in range(n_dst))),
        "key_tokens": tviz.TokenAxis(role="key", tokens=tuple(f"k{i}" for i in range(n_src))),
        "heads": tuple(range(n_heads)),
        "layer": "blk.0.attn",
    }
    defaults.update(kwargs)
    return tviz.AttentionView(**defaults)


def _receipt(**kwargs) -> tviz.CausalReceipt:
    """Build a valid receipt with both controls passing."""

    defaults: dict = {
        "layer": "blk.0.attn",
        "heads": (0, 1),
        "effects": (1.5, -0.5),
        "metric": "logit",
        "intervention": "head contribution zeroed",
        "engine": "rerun",
        "disclosure": "direct",
        "fires": 2,
        "negative_control": 0.0,
        "positive_control": 4.0,
        "joint_effect": 0.25,
    }
    defaults.update(kwargs)
    return tviz.CausalReceipt(**defaults)


def _code(excinfo: pytest.ExceptionInfo) -> str:
    """Return the stable refusal code."""

    return excinfo.value.fields["code"]


class TestAttentionView:
    """AttentionView construction and honesty metadata."""

    def test_valid_view_constructs(self) -> None:
        """A softmax pattern with aligned axes constructs."""

        view = _view()
        assert view.fingerprint.startswith("sha256:")
        assert view.provenance_wording == "captured"

    def test_probability_domain_violation_refuses(self) -> None:
        """Values outside [0, 1] cannot claim the probability domain."""

        with pytest.raises(tviz.TvizError) as excinfo:
            _view(pattern=torch.randn(2, 4, 4) * 10)
        assert _code(excinfo) == "tv_record_invalid"

    def test_per_panel_domain_admits_scores(self) -> None:
        """Score-valued matrices ride the labeled per_panel domain."""

        view = _view(pattern=torch.randn(2, 4, 4), domain="per_panel")
        assert view.domain == "per_panel"

    def test_nonfinite_pattern_refuses(self) -> None:
        """A NaN pattern is a capture defect, not a rendering choice."""

        pattern = torch.softmax(torch.randn(2, 4, 4), dim=-1)
        pattern[0, 0, 0] = float("nan")
        with pytest.raises(tviz.TvizError) as excinfo:
            _view(pattern=pattern)
        assert _code(excinfo) == "tv_record_invalid"

    def test_axis_misalignment_refuses(self) -> None:
        """Token axes must match the pattern dims exactly."""

        with pytest.raises(tviz.TvizError) as excinfo:
            _view(query_tokens=tviz.TokenAxis(role="query", tokens=("just-one",)))
        assert _code(excinfo) == "tv_record_invalid"

    def test_query_key_roles_enforced(self) -> None:
        """Separate query/key axis records even for self-attention."""

        with pytest.raises(tviz.TvizError) as excinfo:
            _view(key_tokens=tviz.TokenAxis(role="query", tokens=tuple("abcd")))
        assert _code(excinfo) == "tv_record_invalid"

    def test_rectangular_cross_attention(self) -> None:
        """T5-style rectangular patterns construct with distinct axes."""

        view = _view(
            pattern=torch.softmax(torch.randn(2, 3, 5), dim=-1),
            query_tokens=tviz.TokenAxis(role="query", tokens=tuple(f"d{i}" for i in range(3))),
            key_tokens=tviz.TokenAxis(role="key", tokens=tuple(f"e{i}" for i in range(5))),
        )
        assert view.pattern.shape == (2, 3, 5)

    def test_provenance_vocab_closed(self) -> None:
        """user_supplied never earns a reconstruction badge; unknowns refuse."""

        view = _view(provenance="user_supplied")
        assert view.provenance_wording == "user supplied; unvalidated"
        with pytest.raises(tviz.TvizError) as excinfo:
            _view(provenance="totally_verified")
        assert _code(excinfo) == "tv_record_invalid"


class TestCrop:
    """crop_to: cropping never renormalizes (D5)."""

    @pytest.mark.smoke
    def test_crop_reports_omitted_mass_and_never_renormalizes(self) -> None:
        """Cropped values are the original values; omitted mass is exact."""

        view = _view(n_dst=6, n_src=6)
        cropped = view.crop_to(slice(0, 6), slice(0, 3))
        assert torch.equal(cropped.pattern, view.pattern[:, :, 0:3])
        assert cropped.crop is not None
        assert cropped.crop.omitted_keys == 3
        expected = float((view.pattern[:, :, 3:].sum(dim=-1)).max())
        assert cropped.crop.omitted_mass_max == pytest.approx(expected, abs=1e-6)
        assert "NOT renormalized" in cropped.crop.disclosure()

    def test_empty_crop_refuses(self) -> None:
        """An empty crop is a caller error."""

        with pytest.raises(tviz.TvizError) as excinfo:
            _view().crop_to(slice(0, 0), slice(0, 2))
        assert _code(excinfo) == "tv_record_invalid"


class TestMaskInfo:
    """Mask provenance: the closed three-source hierarchy (D6)."""

    def test_zeros_inference_has_no_source_token(self) -> None:
        """The banned zeros method cannot be spelled."""

        with pytest.raises(tviz.TvizError) as excinfo:
            tviz.MaskInfo(mask=torch.zeros(4, 4, dtype=torch.bool), source="inferred_from_zeros")
        assert _code(excinfo) == "tv_record_invalid"

    def test_valid_sources(self) -> None:
        """All three hierarchy sources construct."""

        for source in ("sdpa_call_args", "eager_additive_mask", "user_metadata"):
            info = tviz.MaskInfo(mask=torch.zeros(2, 2, dtype=torch.bool), source=source)
            assert info.source == source


class TestGqa:
    """GQA disclosure arithmetic (D15)."""

    @pytest.mark.smoke
    def test_group_header_wording(self) -> None:
        """Shared-heads wording names the group and its query-head span."""

        gqa = tviz.GqaInfo(n_query_heads=14, n_kv_heads=2)
        assert gqa.group_of(0) == 0
        assert gqa.group_of(7) == 1
        header = gqa.header(3)
        assert "kv group 1 of 2" in header
        assert "query heads 0-6" in header

    def test_uneven_grouping_refuses(self) -> None:
        """Non-dividing grouping is a config error."""

        with pytest.raises(tviz.TvizError):
            tviz.GqaInfo(n_query_heads=14, n_kv_heads=4)


class TestAnnotationKinds:
    """The closed four-kind annotation vocabulary (D16)."""

    def test_kinds_are_closed(self) -> None:
        """An unknown kind refuses with the closed-set message."""

        with pytest.raises(tviz.TvizError) as excinfo:
            tviz.Annotation(kind="head_importance", heads=(0,), values=(1.0,), source="s")
        assert _code(excinfo) == "tv_annotation_invalid"

    def test_intervention_effect_requires_receipt(self) -> None:
        """Only a valid CausalReceipt can mint intervention_effect."""

        with pytest.raises(tviz.TvizError) as excinfo:
            tviz.Annotation(
                kind="intervention_effect", heads=(0,), values=(1.0,), source="painted number"
            )
        assert _code(excinfo) == "tv_annotation_invalid"

    def test_descriptive_kinds_construct_without_receipt(self) -> None:
        """Non-causal kinds carry numbers without receipts."""

        for kind in ("descriptive_head_score", "additive_logit_contribution", "screening_estimate"):
            annotation = tviz.Annotation(kind=kind, heads=(0, 1), values=(0.5, None), source="s")
            assert annotation.values[1] is None  # unmeasured stays blank

    @pytest.mark.smoke
    def test_from_receipt_mints_intervention_effect(self) -> None:
        """The receipt door mints the causal kind with its evidence."""

        annotation = tviz.Annotation.from_receipt(_receipt())
        assert annotation.kind == "intervention_effect"
        assert annotation.receipt is not None


class TestCausalReceipt:
    """Receipt attachment validation (D12)."""

    def test_zero_fires_refuses(self) -> None:
        """Zero fires is a typed error, never a zero effect."""

        with pytest.raises(tviz.TvizError) as excinfo:
            _receipt(fires=0)
        assert _code(excinfo) == "tv_receipt_invalid"

    def test_moved_negative_control_refuses(self) -> None:
        """A self-patch that moves the metric fails the machinery."""

        with pytest.raises(tviz.TvizError) as excinfo:
            _receipt(negative_control=0.5)
        assert _code(excinfo) == "tv_receipt_invalid"

    def test_unmoved_positive_control_refuses(self) -> None:
        """The positive control is the check that catches silent no-fire."""

        with pytest.raises(tviz.TvizError) as excinfo:
            _receipt(positive_control=0.0)
        assert _code(excinfo) == "tv_receipt_invalid"

    def test_nonfinite_effect_refuses(self) -> None:
        """A non-finite effect is a failed measurement."""

        with pytest.raises(tviz.TvizError) as excinfo:
            _receipt(effects=(float("inf"), 1.0))
        assert _code(excinfo) == "tv_receipt_invalid"

    def test_measured_zero_is_kept(self) -> None:
        """A genuine measured zero is a value, not a refusal."""

        receipt = _receipt(effects=(0.0, None))
        assert receipt.measured_n == 1
        assert receipt.cell_sum == 0.0


class TestScoreDecomposition:
    """The hard sum-to-score invariant (D19)."""

    def test_closed_terms_construct(self) -> None:
        """Terms reproducing the reference score construct."""

        q = torch.randn(8)
        k = torch.randn(8)
        scale = 8**0.5
        reference = float((q * k).sum()) / scale
        record = tviz.ScoreDecomposition(
            layer="blk.0.attn",
            head=0,
            destination=1,
            source=0,
            query_vector=q,
            key_vector=k,
            products=q * k,
            scale=scale,
            reference_score=reference,
        )
        assert record.total == pytest.approx(reference, abs=1e-5)

    def test_unclosed_terms_refuse(self) -> None:
        """A decomposition that misses the score REFUSES (never misleads)."""

        q = torch.randn(8)
        k = torch.randn(8)
        with pytest.raises(tviz.TvizError) as excinfo:
            tviz.ScoreDecomposition(
                layer="blk.0.attn",
                head=0,
                destination=1,
                source=0,
                query_vector=q,
                key_vector=k,
                products=q * k,
                scale=8**0.5,
                reference_score=float((q * k).sum()) / 8**0.5 + 5.0,
            )
        assert _code(excinfo) == "tv_decomposition_unclosed"


class TestTokenScores:
    """Strip records: alignment, N/A cells, the attribution door (D28)."""

    def test_row_alignment_enforced(self) -> None:
        """Every row aligns to the shared token axis."""

        with pytest.raises(tviz.TvizError) as excinfo:
            tviz.TokenScores(
                tokens=("a", "b"),
                rows=(tviz.TokenScoreRow(label="x", scores=(1.0,)),),
            )
        assert _code(excinfo) == "tv_record_invalid"

    def test_from_attribution_payload(self) -> None:
        """The attribution payload converts with footers intact."""

        class Payload:
            display_tokens = ["a", "b"]
            scores = [0.5, -0.5]
            footer_lines = ["target: logits[0, 1]"]
            score_domain = "zero_centered_diverging"

        record = tviz.TokenScores.from_attribution(Payload())
        assert record.footer_lines == ("target: logits[0, 1]",)
        assert record.rows[0].scores == (0.5, -0.5)


class TestEpisodeCoordinates:
    """Episode coordinates land in the records from day one (D22)."""

    def test_view_carries_episode_coordinates(self) -> None:
        """Records address episode steps without a wave-2 renderer."""

        episode = tviz.EpisodeCoordinates(step=3, role="decode", completion="complete")
        view = _view(episode=episode)
        assert view.episode is not None
        assert view.episode.step == 3

    @pytest.mark.smoke
    def test_role_vocabulary_closed(self) -> None:
        """Unknown roles refuse."""

        with pytest.raises(tviz.TvizError):
            tviz.EpisodeCoordinates(step=0, role="speculate")


class _Modules(dict):
    """Duck-typed stand-in for the trace modules accessor.

    The real accessor iterates Module RECORDS (not addresses) while
    supporting ``in`` / ``[]`` by address key; tests mimic that contract.
    """

    def __iter__(self):  # type: ignore[override]
        return iter(self.values())


class TestExtractionRefusals:
    """Trace-facing refusals on duck-typed traces (no capture needed)."""

    @pytest.mark.smoke
    def test_missing_pattern_facet_refuses(self) -> None:
        """A pattern-less trace refuses tv_facet_missing with the remedy."""

        from types import SimpleNamespace

        trace = SimpleNamespace(modules=_Modules(m=SimpleNamespace(address="m", facets=None)))
        with pytest.raises(tviz.TvizError) as excinfo:
            tviz.attention_view(trace, "m")
        assert _code(excinfo) == "tv_facet_missing"
        with pytest.raises(tviz.TvizError) as excinfo:
            tviz.attention_views(trace)
        assert _code(excinfo) == "tv_facet_missing"

    @pytest.mark.smoke
    def test_missing_payload_refuses(self) -> None:
        """A facet without a captured tensor payload refuses tv_payload_missing."""

        from types import SimpleNamespace

        module = SimpleNamespace(
            address="m", facets={"q": SimpleNamespace(value="payload was not captured")}
        )
        trace = SimpleNamespace(modules=_Modules(m=module))
        with pytest.raises(tviz.TvizError) as excinfo:
            tviz.score_decomposition(trace, "m", head=0, destination=0, source=0)
        assert _code(excinfo) == "tv_payload_missing"
