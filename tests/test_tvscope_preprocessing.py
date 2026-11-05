"""tvscope B2/B3: authority resolver, audit oracle, diagnostics, FP suite.

Oracle assertions (tvscope memo section 5): the audit names mismatched
fields; strict mode refuses with DISTINCT typed codes; permissive mode
stamps ``mismatch``, never ``verified``; the demoted tier-4 fallback never
yields verified; the tensor arm never returns match. The false-positive
suite is a first-class test class: every legitimate pipeline must produce
ZERO error-grade findings.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytest.importorskip("torchvision")

import torchlens._data_classes._preprocessing_provenance as provenance  # noqa: E402
import torchlens.errors
import torchlens.preprocessing as pp  # noqa: E402


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

    def test_explicit_mapping_resolves_with_aliases(self) -> None:
        """Explicit declarations accept canonical + alias keys, disclose the rest."""

        resolution = pp.resolve({"image_mean": [0.5, 0.5, 0.5], "std": 0.5, "mystery_knob": 3})
        assert resolution.preprocessing.source == "explicit_declaration"
        assert resolution.status == provenance.STATUS_AUTHORITATIVE
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
        assert resolution.preprocessing.source == "timm"
        assert resolution.declared is not None
        assert resolution.declared.crop_size == 224
        assert resolution.declared.resize_size == 256
        assert resolution.declared.interpolation == "bicubic"

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
        assert resolution.preprocessing.source == "compose_pipeline"
        assert resolution.declared is not None
        assert resolution.declared.resize_size == 256
        assert resolution.declared.crop_size == 224
        assert resolution.declared.value_range == (0.0, 1.0)

    def test_unrecognized_authority_refuses_typed(self) -> None:
        """A non-authority object refuses with the teaching code."""

        from torchlens._errors import InvalidArgumentError

        with pytest.raises(InvalidArgumentError) as excinfo:
            pp.resolve(42)
        assert excinfo.value.fields["code"] == "preprocessing_authority_unrecognized"

    def test_registry_extension_wins_over_builtins(self) -> None:
        """register_authority_adapter probes before the builtin mapping adapter."""

        marker = {"mean": [0.1, 0.2, 0.3], "custom_kind": True}

        def probe(candidate) -> bool:
            """Match only the marker mapping."""

            return isinstance(candidate, dict) and candidate.get("custom_kind") is True

        def adapt(candidate) -> pp.Resolution:
            """Resolve the marker with a distinctive source."""

            resolution = pp.unknown_resolution("custom")
            resolution.preprocessing.source = "custom_adapter"
            return resolution

        pp.register_authority_adapter("custom", probe, adapt)
        try:
            assert pp.resolve(marker).preprocessing.source == "custom_adapter"
        finally:
            from torchlens.preprocessing import _authorities

            _authorities._REGISTERED_ADAPTERS.clear()


class TestAntialiasCoercion:
    """torchvision's legacy "warn" antialias sentinel must declare, not vanish.

    torchvision < ~0.17's classification presets default ``antialias`` to
    the literal string ``"warn"``, not a bool. Collapsing that to ``None``
    ("undeclared") made a preset's self-audit register as ``unknown`` (one
    undeclared field forces the whole verdict to ``unknown``) even when both
    sides are the exact same declared sentinel.
    """

    def test_unrecognized_value_is_undeclared(self) -> None:
        from torchlens.preprocessing._authorities import _coerce_antialias

        assert _coerce_antialias(None) is None
        assert _coerce_antialias("something_else") is None

    def test_self_audit_of_a_warn_sentinel_preset_verifies(self) -> None:
        """A synthetic preset mirroring old torchvision's "warn" default."""

        preset = SimpleNamespace(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            resize_size=[256],
            crop_size=[224],
            interpolation="bilinear",
            antialias="warn",
        )

        class _FakeWeights:
            meta: dict[str, Any] = {}
            url = "https://example.invalid/fake-weights"

            def transforms(self) -> Any:
                return preset

        resolution = pp.resolve(_FakeWeights())
        report = pp.compare(resolution, resolution)
        assert report.verdict == "verified"


class TestAuditOracle:
    """B3 oracle assertions: configuration comparison is the verdict."""

    def test_strict_mismatch_and_unknown_have_distinct_codes(
        self, resnet50_v2_resolution, detection_resolution
    ) -> None:
        """Strict mode refuses BOTH failure kinds, distinctly typed."""

        with pytest.raises(torchlens.errors.PreprocessingAuditError) as mismatch_info:
            pp.compare(
                resnet50_v2_resolution,
                {"mean": [0.5] * 3, "std": [0.5] * 3},
                strict=True,
            )
        assert mismatch_info.value.fields["code"] == "preprocessing_audit_mismatch"
        with pytest.raises(torchlens.errors.PreprocessingAuditError) as unknown_info:
            pp.compare(detection_resolution, None, strict=True)
        assert unknown_info.value.fields["code"] == "preprocessing_audit_unknown"

    def test_opaque_applied_side_never_matches(self, resnet50_v2_resolution) -> None:
        """An opaque callable's fields are unknown -- never match (memo D4)."""

        report = pp.compare(resnet50_v2_resolution, lambda x: x)
        assert report.verdict == "unknown"
        assert all(f.verdict == "unknown" for f in report.findings)
        assert "opaque_transform" in report.unknown_reasons


class TestDiagnosticsNeverMatch:
    """The tensor arm may contradict or fail-to-contradict, never verify."""

    def test_nonfinite_contradicts(self) -> None:
        """NaNs are an intrinsic contradiction."""

        result = pp.diagnose(torch.full((2, 3, 4, 4), float("nan")))
        assert any(
            f.check_name == "nonfinite" and f.outcome == "contradiction" for f in result.findings
        )

    def test_uint8_under_float_declaration_contradicts(self) -> None:
        """An integer batch cannot have passed a declared float normalization."""

        declared = pp.DeclaredPreprocessing(mean=(0.5,) * 3, std=(0.5,) * 3)
        result = pp.diagnose(
            torch.randint(0, 255, (2, 3, 4, 4), dtype=torch.uint8), declared=declared
        )
        assert any(
            f.check_name == "dtype_regime" and f.outcome == "contradiction" for f in result.findings
        )

    def test_unknown_check_refuses_typed(self) -> None:
        """The check vocabulary is closed and teaches the valid set."""

        from torchlens._errors import InvalidArgumentError

        with pytest.raises(InvalidArgumentError) as excinfo:
            pp.diagnose(torch.randn(2, 3, 4, 4), checks=("telepathy",))
        assert excinfo.value.fields["code"] == "preprocessing_diagnostic_unknown_check"


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

    def test_detection_checkpoint_correct_unit_range_batch(self, detection_resolution) -> None:
        """The declares-nothing family: [0,1] input, verdict unknown, no alarm."""

        report = pp.compare(detection_resolution, None)
        assert report.verdict == "unknown"
        result = pp.diagnose(torch.rand(2, 3, 8, 8), declared=detection_resolution.declared)
        assert list(result.contradictions) == []

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

    def test_no_metadata_model_and_opaque_lambda_stay_unknown(self) -> None:
        """Unknown is the honest verdict; never an error-grade claim."""

        unknown_authority = pp.resolve(model=torch.nn.Linear(2, 2))
        report = pp.compare(unknown_authority, lambda x: x)
        assert report.verdict == "unknown"
