"""tvscope B2/B3: authority resolver, audit oracle, diagnostics, FP suite.

Oracle assertions (tvscope memo section 5): the audit names mismatched
fields; strict mode refuses with DISTINCT typed codes; permissive mode
stamps ``mismatch``, never ``verified``; the demoted tier-4 fallback never
yields verified; the tensor arm never returns match. The false-positive
suite is a first-class test class: every legitimate pipeline must produce
ZERO error-grade findings.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torchvision")

import torchlens.preprocessing as pp  # noqa: E402
from torchlens.data_classes.trace import ResolvedPreprocessing  # noqa: E402


@pytest.fixture(scope="module")
def resnet50_v2_resolution() -> pp.Resolution:
    """The torchvision ResNet50 V2 authority (the resize=232 case)."""

    from torchvision.models import ResNet50_Weights

    return pp.resolve(ResNet50_Weights.IMAGENET1K_V2)


@pytest.fixture(scope="module")
def detection_resolution() -> pp.Resolution:
    """The declares-nothing detection authority (honest-unknown family)."""

    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

    return pp.resolve(FasterRCNN_ResNet50_FPN_Weights.COCO_V1)


class TestResolver:
    """B2: adapters normalize authorities; refusals teach."""

    @pytest.mark.smoke
    def test_torchvision_v2_declares_resize_232(self, resnet50_v2_resolution) -> None:
        """The V2 preset's declared resize is 232 -- the default-guess-is-wrong case."""

        declared = resnet50_v2_resolution.declared
        assert declared is not None
        assert declared.resize_size == 232
        assert declared.crop_size == 224
        assert declared.mean == pytest.approx((0.485, 0.456, 0.406))
        assert resnet50_v2_resolution.status == pp.STATUS_AUTHORITATIVE
        assert resnet50_v2_resolution.transform is not None

    @pytest.mark.smoke
    def test_detection_authority_declares_nothing(self, detection_resolution) -> None:
        """A real shipped family declares NO fields; resolution stays honest."""

        assert detection_resolution.declared is None
        assert detection_resolution.record.config.get("declares") == "nothing"
        assert detection_resolution.transform is not None

    @pytest.mark.smoke
    def test_explicit_mapping_resolves_with_aliases(self) -> None:
        """Explicit declarations accept canonical + alias keys, disclose the rest."""

        resolution = pp.resolve({"image_mean": [0.5, 0.5, 0.5], "std": 0.5, "mystery_knob": 3})
        assert resolution.record.source == "explicit_declaration"
        assert resolution.status == pp.STATUS_AUTHORITATIVE
        assert resolution.declared is not None
        assert resolution.declared.mean == pytest.approx((0.5, 0.5, 0.5))
        assert resolution.declared.extras["unrecognized_keys"] == ["mystery_knob"]

    @pytest.mark.heavy  # merged-tree timm resolution outgrew the smoke budget (T82b)
    def test_timm_config_mapping_resolves(self) -> None:
        """A timm-shaped data config normalizes (crop_pct -> resize)."""

        resolution = pp.resolve(
            {
                "input_size": (3, 224, 224),
                "mean": (0.5, 0.5, 0.5),
                "std": (0.5, 0.5, 0.5),
                "crop_pct": 0.875,
                "interpolation": "bicubic",
            }
        )
        assert resolution.record.source == "timm"
        assert resolution.declared is not None
        assert resolution.declared.crop_size == 224
        assert resolution.declared.resize_size == 256
        assert resolution.declared.interpolation == "bicubic"

    @pytest.mark.smoke
    def test_compose_pipeline_parses(self) -> None:
        """A torchvision/open_clip-style Compose parses into declared fields."""

        from torchvision import transforms as T

        pipeline = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.48, 0.46, 0.41], std=[0.27, 0.26, 0.28]),
            ]
        )
        resolution = pp.resolve(pipeline)
        assert resolution.record.source == "compose_pipeline"
        assert resolution.declared is not None
        assert resolution.declared.resize_size == 256
        assert resolution.declared.crop_size == 224
        assert resolution.declared.value_range == (0.0, 1.0)

    @pytest.mark.smoke
    def test_unrecognized_authority_refuses_typed(self) -> None:
        """A non-authority object refuses with the teaching code."""

        from torchlens._errors import InvalidArgumentError

        with pytest.raises(InvalidArgumentError) as excinfo:
            pp.resolve(42)
        assert excinfo.value.fields["code"] == "preprocessing_authority_unrecognized"

    @pytest.mark.smoke
    def test_opaque_callable_authority_is_unknown_with_transform(self) -> None:
        """An opaque callable declares nothing; the callable is preserved."""

        resolution = pp.resolve(lambda x: x)
        assert resolution.status == pp.STATUS_UNKNOWN
        assert resolution.declared is None
        assert resolution.transform is not None

    @pytest.mark.smoke
    def test_no_metadata_model_resolves_unknown(self) -> None:
        """A plain nn.Module resolves unknown -- never a TorchLens recipe."""

        resolution = pp.resolve(model=torch.nn.Linear(4, 4))
        assert resolution.status == pp.STATUS_UNKNOWN
        assert resolution.record.source == "unknown"
        assert resolution.transform is None

    @pytest.mark.smoke
    def test_registry_extension_wins_over_builtins(self) -> None:
        """register_authority_adapter probes before the builtin mapping adapter."""

        marker = {"mean": [0.1, 0.2, 0.3], "custom_kind": True}

        def probe(candidate) -> bool:
            """Match only the marker mapping."""

            return isinstance(candidate, dict) and candidate.get("custom_kind") is True

        def adapt(candidate) -> pp.Resolution:
            """Resolve the marker with a distinctive source."""

            resolution = pp.unknown_resolution("custom")
            resolution.record.source = "custom_adapter"
            return resolution

        pp.register_authority_adapter("custom", probe, adapt)
        try:
            assert pp.resolve(marker).record.source == "custom_adapter"
        finally:
            from torchlens.preprocessing import _authorities

            _authorities._REGISTERED_ADAPTERS.clear()


class TestAuditOracle:
    """B3 oracle assertions: configuration comparison is the verdict."""

    @pytest.mark.smoke
    def test_audit_names_mismatched_fields(self, resnet50_v2_resolution) -> None:
        """Wrong constants are named per field, before any extraction."""

        report = pp.audit(resnet50_v2_resolution, {"mean": [0.5] * 3, "std": [0.5] * 3})
        assert report.verdict == "mismatch"
        assert set(report.mismatched_fields) == {"mean", "std"}
        mean_row = next(f for f in report.findings if f.field == "mean")
        assert mean_row.evidence_class == "mismatch"
        assert "128-image" in mean_row.consequence  # the stimulus set is named

    @pytest.mark.smoke
    def test_strict_mismatch_and_unknown_have_distinct_codes(
        self, resnet50_v2_resolution, detection_resolution
    ) -> None:
        """Strict mode refuses BOTH failure kinds, distinctly typed."""

        with pytest.raises(pp.PreprocessingAuditError) as mismatch_info:
            pp.audit(
                resnet50_v2_resolution,
                {"mean": [0.5] * 3, "std": [0.5] * 3},
                strict=True,
            )
        assert mismatch_info.value.fields["code"] == "preprocessing_audit_mismatch"
        with pytest.raises(pp.PreprocessingAuditError) as unknown_info:
            pp.audit(detection_resolution, None, strict=True)
        assert unknown_info.value.fields["code"] == "preprocessing_audit_unknown"

    @pytest.mark.smoke
    def test_permissive_stamps_mismatch_never_raises(self, resnet50_v2_resolution) -> None:
        """Default mode reports; it never raises and never says verified."""

        report = pp.audit(resnet50_v2_resolution, {"mean": [0.5] * 3, "std": [0.5] * 3})
        assert report.verdict == "mismatch"

    @pytest.mark.smoke
    def test_fallback_record_never_yields_verified(self) -> None:
        """The demoted ImageNet default can never anchor a verified verdict."""

        fallback = ResolvedPreprocessing(
            source="imagenet_default",
            identifier="ImageNet-default-resize256-crop224",
            verified=False,
            config={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            description="fallback",
        )
        assert fallback.status == pp.STATUS_UNVERIFIED_FALLBACK
        resolution = pp.Resolution(record=fallback, declared=None, transform=None)
        report = pp.audit(resolution, None)
        assert report.verdict == "unknown"
        assert "authority_status_unverified_fallback" in report.unknown_reasons

    @pytest.mark.smoke
    def test_opaque_applied_side_never_matches(self, resnet50_v2_resolution) -> None:
        """An opaque callable's fields are unknown -- never match (memo D4)."""

        report = pp.audit(resnet50_v2_resolution, lambda x: x)
        assert report.verdict == "unknown"
        assert all(f.verdict == "unknown" for f in report.findings)
        assert "opaque_transform" in report.unknown_reasons

    @pytest.mark.smoke
    def test_self_audit_of_authoritative_resolution_verifies(self, resnet50_v2_resolution) -> None:
        """Applying the authority's own transform is verified-by-construction."""

        report = pp.audit(resnet50_v2_resolution, resnet50_v2_resolution)
        assert report.verdict == "verified"

    @pytest.mark.smoke
    def test_report_round_trips_to_json(self, resnet50_v2_resolution) -> None:
        """The report is JSON-portable (the deferred auto-tier plumbing)."""

        import json

        report = pp.audit(resnet50_v2_resolution, {"mean": [0.5] * 3})
        payload = json.loads(json.dumps(report.to_json()))
        assert payload["schema"] == "tl_preprocessing_audit_v1"
        assert payload["verdict"] == report.verdict
        assert len(payload["findings"]) == len(pp.COMPARABLE_FIELDS)

    @pytest.mark.smoke
    def test_status_property_serves_legacy_records(self) -> None:
        """The derived status is read-time: legacy-shaped records serve it."""

        legacy = ResolvedPreprocessing(
            source="hf_auto_image_processor",
            identifier="some/model",
            verified=True,
            config={},
            description="legacy record without resolution_method",
        )
        assert legacy.status == pp.STATUS_AUTHORITATIVE


class TestDiagnosticsNeverMatch:
    """The tensor arm may contradict or fail-to-contradict, never verify."""

    @pytest.mark.smoke
    def test_outcome_vocabulary_has_no_match_member(self) -> None:
        """The type itself forbids a match-shaped outcome (memo D4)."""

        import torchlens.preprocessing._diagnostics as diag

        vocabulary = {
            diag.OUTCOME_CONTRADICTION,
            diag.OUTCOME_NO_CONTRADICTION,
            diag.OUTCOME_NOT_APPLICABLE,
        }
        assert "match" not in vocabulary
        assert "verified" not in vocabulary
        assert not hasattr(pp.InputDiagnostics(findings=()), "verified")

    @pytest.mark.smoke
    def test_nonfinite_contradicts(self) -> None:
        """NaNs are an intrinsic contradiction."""

        result = pp.diagnose(torch.full((2, 3, 4, 4), float("nan")))
        assert any(f.check == "nonfinite" and f.outcome == "contradiction" for f in result.findings)

    @pytest.mark.smoke
    def test_uint8_under_float_declaration_contradicts(self) -> None:
        """An integer batch cannot have passed a declared float normalization."""

        declared = pp.DeclaredPreprocessing(mean=(0.5,) * 3, std=(0.5,) * 3)
        result = pp.diagnose(
            torch.randint(0, 255, (2, 3, 4, 4), dtype=torch.uint8), declared=declared
        )
        assert any(
            f.check == "dtype_regime" and f.outcome == "contradiction" for f in result.findings
        )

    @pytest.mark.smoke
    def test_unknown_check_refuses_typed(self) -> None:
        """The check vocabulary is closed and teaches the valid set."""

        from torchlens._errors import InvalidArgumentError

        with pytest.raises(InvalidArgumentError) as excinfo:
            pp.diagnose(torch.randn(2, 3, 4, 4), checks=("telepathy",))
        assert excinfo.value.fields["code"] == "preprocessing_diagnostic_unknown_check"

    @pytest.mark.smoke
    def test_confinement_is_opt_in_and_prints_blind_spots(self) -> None:
        """The measured-weak arm never runs by default; both blind spots print."""

        batch = torch.rand(2, 3, 4, 4)
        default = pp.diagnose(batch)
        assert all(f.check != "confinement" for f in default.findings)
        declared = pp.DeclaredPreprocessing(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), value_range=(0.0, 1.0)
        )
        opted = pp.diagnose((batch - 0.5) / 0.5, declared=declared, checks=("confinement",))
        finding = opted.findings[0]
        assert "blind spots" in finding.detail


class TestFalsePositiveSuite:
    """First-class FP class: legitimate pipelines produce ZERO error-grade findings.

    The template for any future automatic tier (memo section 7): a check
    that fires on any row here is disqualified from automatic status.
    """

    @staticmethod
    def _correct_normalized_batch(dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """A correctly normalized ImageNet-style batch in the given dtype."""

        torch.manual_seed(0)
        raw = torch.rand(2, 3, 8, 8)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return ((raw - mean) / std).to(dtype)

    @staticmethod
    def _declared_imagenet() -> pp.DeclaredPreprocessing:
        """The matching ImageNet declaration."""

        return pp.DeclaredPreprocessing(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            value_range=(0.0, 1.0),
        )

    def _assert_zero_error_grade(self, batch: torch.Tensor, declared) -> None:
        """No contradiction from the default checks + the confinement arm."""

        result = pp.diagnose(
            batch,
            declared=declared,
            checks=("nonfinite", "dtype_regime", "layout", "confinement"),
        )
        contradictions = list(result.contradictions)
        assert contradictions == [], [f.detail for f in contradictions]

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.smoke
    def test_correct_pipeline_across_dtypes(self, dtype) -> None:
        """A plain reduced-precision cast is a legitimate pipeline."""

        self._assert_zero_error_grade(
            self._correct_normalized_batch(dtype), self._declared_imagenet()
        )

    @pytest.mark.smoke
    def test_correct_plus_post_normalization_noise(self) -> None:
        """Routine robustness noise is a legitimate pipeline."""

        torch.manual_seed(1)
        batch = self._correct_normalized_batch() + 0.05 * torch.randn(2, 3, 8, 8)
        result = pp.diagnose(batch, declared=self._declared_imagenet())
        assert list(result.contradictions) == []

    @pytest.mark.smoke
    def test_per_image_standardized_stimuli(self) -> None:
        """Per-image standardization (routine luminance control) never alarms
        on the default checks."""

        torch.manual_seed(2)
        batch = torch.randn(2, 3, 8, 8)
        batch = (batch - batch.mean(dim=(1, 2, 3), keepdim=True)) / batch.std(
            dim=(1, 2, 3), keepdim=True
        )
        result = pp.diagnose(batch, declared=self._declared_imagenet())
        assert list(result.contradictions) == []

    @pytest.mark.smoke
    def test_detection_checkpoint_correct_unit_range_batch(self, detection_resolution) -> None:
        """The declares-nothing family: [0,1] input, verdict unknown, no alarm."""

        report = pp.audit(detection_resolution, None)
        assert report.verdict == "unknown"
        result = pp.diagnose(torch.rand(2, 3, 8, 8), declared=detection_resolution.declared)
        assert list(result.contradictions) == []

    @pytest.mark.smoke
    def test_timm_half_half_batch_not_flagged(self) -> None:
        """A correct 0.5/0.5 batch must not be flagged against its own config."""

        declared = pp.DeclaredPreprocessing(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), value_range=(0.0, 1.0)
        )
        batch = (torch.rand(2, 3, 8, 8) - 0.5) / 0.5
        result = pp.diagnose(
            batch, declared=declared, checks=("nonfinite", "dtype_regime", "layout", "confinement")
        )
        assert list(result.contradictions) == []

    @pytest.mark.smoke
    def test_no_metadata_model_and_opaque_lambda_stay_unknown(self) -> None:
        """Unknown is the honest verdict; never an error-grade claim."""

        unknown_authority = pp.resolve(model=torch.nn.Linear(2, 2))
        report = pp.audit(unknown_authority, lambda x: x)
        assert report.verdict == "unknown"
