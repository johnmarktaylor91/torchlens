"""tvscope B4/B5 + composition rows 2-3: the manifest block + the input path.

Every new artifact carries the schema-versioned ``input_preprocessing``
block (allowed to say unknown); legacy/pre-block artifacts read as unknown;
resume never grafts new provenance; the input-transform identity joins the
resume signature so a differing input path refuses through the ONE D16
door; opaque input callables apply-and-disclose and refuse resume
continuation.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

pytest.importorskip("torchvision")

import torchlens as tl  # noqa: E402
import torchlens.preprocessing as pp  # noqa: E402
from torchlens.dataset_extraction import (  # noqa: E402
    INPUT_PREPROCESSING_SCHEMA,
    DatasetExtractionResumeError,
    input_preprocessing_of,
)

pytestmark = [pytest.mark.smoke]


class _SmallCNN(nn.Module):
    """Small CNN standing in for the extraction engine's model side."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Conv + relu + pool."""

        return self.pool(torch.relu(self.conv(x)))


@pytest.fixture()
def model() -> nn.Module:
    """Seeded eval-mode CNN."""

    torch.manual_seed(0)
    return _SmallCNN().eval()


@pytest.fixture()
def resolution() -> pp.Resolution:
    """A FULL-FIELD explicit authority WITH a runnable transform.

    Every comparable field is declared: partial metadata never becomes
    match (memo D4), so the verified-by-construction golden path needs a
    fully declarative authority.
    """

    base = pp.resolve(
        {
            "resize_size": 8,
            "crop_size": 8,
            "interpolation": "bilinear",
            "antialias": True,
            "channel_order": "rgb",
            "value_range": (0.0, 1.0),
            "mean": [0.5] * 3,
            "std": [0.5] * 3,
        }
    )
    return pp.Resolution(
        record=base.record,
        declared=base.declared,
        transform=lambda batch: (batch - 0.5) / 0.5,
    )


def _stimuli(n: int = 5) -> torch.Tensor:
    """Seeded stimulus tensor."""

    torch.manual_seed(1)
    return torch.rand(n, 3, 8, 8)


def test_resolution_backed_run_stamps_verified_block(model, resolution, tmp_path) -> None:
    """The golden path: applied + recorded, verdict verified-by-construction."""

    out_dir = tmp_path / "run"
    tl.extract_dataset(
        model,
        _stimuli(),
        ["conv"],
        batch_size=2,
        output_dir=out_dir,
        input_transform=resolution,
        stimulus_ids=[f"s{i}" for i in range(5)],
        progress=False,
    )
    manifest = json.loads((out_dir / "manifest.json").read_text())
    block = manifest["input_preprocessing"]
    assert block["schema"] == INPUT_PREPROCESSING_SCHEMA
    assert block["verdict"] == "verified"
    assert block["applied_by"] == "extract_dataset"
    assert block["authority"]["source"] == "explicit_declaration"
    assert block["versions"]["torchlens"]
    signature_value = manifest["signature"]["transform_pipeline"]
    assert set(signature_value) == {"output", "input"}
    assert signature_value["input"]["kind"] == "resolved"
    loaded = tl.load_extraction(out_dir)
    assert loaded.input_preprocessing["verdict"] == "verified"


def test_undeclared_run_carries_honest_unknown_block(model, tmp_path) -> None:
    """No input path declared -> the block still exists and says unknown."""

    out_dir = tmp_path / "plain"
    tl.extract_dataset(
        model, _stimuli(), ["conv"], batch_size=2, output_dir=out_dir, progress=False
    )
    manifest = json.loads((out_dir / "manifest.json").read_text())
    block = manifest["input_preprocessing"]
    assert block["verdict"] == "unknown"
    assert "input_preprocessing_undeclared" in block["unknown_reasons"]
    # the historical signature value shape is preserved byte-for-byte
    assert not isinstance(manifest["signature"]["transform_pipeline"], dict) or (
        "input" not in manifest["signature"]["transform_pipeline"]
    )


def test_legacy_manifest_reads_as_unknown() -> None:
    """Pre-block artifacts are never read as asserted-clean (memo D10)."""

    block = input_preprocessing_of({"schema": "tl_extract_manifest_v2"})
    assert block["verdict"] == "unknown"
    assert block["legacy"] is True
    assert "legacy_artifact_predates_input_preprocessing_block" in block["unknown_reasons"]


def test_resume_with_differing_input_path_refuses_typed(model, resolution, tmp_path) -> None:
    """The input identity rides the ONE signature door (D16)."""

    out_dir = tmp_path / "sig"
    kwargs = {"batch_size": 2, "output_dir": out_dir, "progress": False}
    tl.extract_dataset(model, _stimuli(), ["conv"], input_transform=resolution, **kwargs)
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        tl.extract_dataset(model, _stimuli(), ["conv"], resume=True, **kwargs)
    assert excinfo.value.fields["code"] == "extraction_resume_signature_mismatch"
    assert "transform_pipeline" in excinfo.value.fields["mismatched_fields"]


def test_resume_never_grafts_new_provenance(model, resolution, tmp_path) -> None:
    """Composition row 3: a compatible resume keeps the artifact's own block."""

    out_dir = tmp_path / "graft"
    kwargs = {"batch_size": 2, "output_dir": out_dir, "progress": False}
    paths = tl.extract_dataset(model, _stimuli(), ["conv"], input_transform=resolution, **kwargs)
    before = json.loads((out_dir / "manifest.json").read_text())["input_preprocessing"]
    again = tl.extract_dataset(
        model, _stimuli(), ["conv"], input_transform=resolution, resume=True, **kwargs
    )
    after = json.loads((out_dir / "manifest.json").read_text())["input_preprocessing"]
    assert after == before
    assert [p.name for p in again] == [p.name for p in paths]


def test_opaque_input_transform_applies_and_refuses_continuation(model, tmp_path) -> None:
    """Bare callables run + disclose opaque; interrupted resume refuses."""

    out_dir = tmp_path / "opaque"
    tl.extract_dataset(
        model,
        _stimuli(4),
        ["conv"],
        batch_size=2,
        output_dir=out_dir,
        input_transform=lambda batch: batch * 2.0,
        progress=False,
    )
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["input_preprocessing"]["verdict"] == "unknown"
    assert "opaque_input_transform" in manifest["input_preprocessing"]["unknown_reasons"]
    assert manifest["signature"]["transform_pipeline"]["input"]["kind"] == "opaque"
    # simulate an interruption: drop the terminal status, then ask to continue
    manifest["status"] = "in_progress"
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli(4),
            ["conv"],
            batch_size=2,
            output_dir=out_dir,
            input_transform=lambda batch: batch * 2.0,
            resume=True,
            progress=False,
        )
    assert excinfo.value.fields["code"] == "extraction_resume_opaque_transform"
    assert excinfo.value.fields["input_opaque"] is True


def test_input_transform_values_actually_apply(model, resolution) -> None:
    """The input path does real work in memory mode (B5 is not a stamp-only)."""

    stimuli = _stimuli(4)
    transformed = tl.extract_dataset(
        model, stimuli, ["conv"], batch_size=2, input_transform=resolution, progress=False
    )
    manual = tl.extract_dataset(
        model, (stimuli - 0.5) / 0.5, ["conv"], batch_size=2, progress=False
    )
    assert torch.allclose(transformed["conv"], manual["conv"])


def test_input_provenance_in_memory_is_a_false_affordance(model, resolution) -> None:
    """Provenance-only stamps need a manifest to land in."""

    from torchlens._errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.extract_dataset(model, _stimuli(), ["conv"], input_provenance=resolution, progress=False)
    assert excinfo.value.fields["code"] == "extraction_input_provenance_in_memory_unsupported"


def test_provenance_only_stamp_records_applied_not_audited(model, resolution, tmp_path) -> None:
    """Caller-preprocessed stimuli get the authority stamped, verdict unknown."""

    out_dir = tmp_path / "stamp"
    tl.extract_dataset(
        model,
        _stimuli(),
        ["conv"],
        batch_size=2,
        output_dir=out_dir,
        input_provenance=resolution,
        progress=False,
    )
    block = json.loads((out_dir / "manifest.json").read_text())["input_preprocessing"]
    assert block["applied_by"] == "caller"
    assert block["verdict"] == "unknown"
    assert "applied_not_audited" in block["unknown_reasons"]
    assert block["authority"]["source"] == "explicit_declaration"


def test_audit_provenance_stamp_carries_its_verdict(model, resolution, tmp_path) -> None:
    """A PreprocessingAudit provenance stamp rides with its own verdict."""

    report = pp.audit(resolution, resolution)
    out_dir = tmp_path / "auditstamp"
    tl.extract_dataset(
        model,
        _stimuli(),
        ["conv"],
        batch_size=2,
        output_dir=out_dir,
        input_provenance=report,
        progress=False,
    )
    block = json.loads((out_dir / "manifest.json").read_text())["input_preprocessing"]
    assert block["verdict"] == report.verdict == "verified"
    assert block["audit"]["schema"] == "tl_preprocessing_audit_v1"


def test_declaration_only_resolution_as_transform_refuses(model, tmp_path) -> None:
    """A resolution with no callable cannot be the input transform."""

    from torchlens._errors import InvalidArgumentError

    declaration_only = pp.resolve({"mean": [0.5] * 3})
    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli(),
            ["conv"],
            output_dir=tmp_path / "x",
            input_transform=declaration_only,
            progress=False,
        )
    assert excinfo.value.fields["code"] == "extraction_input_transform_missing_callable"


def test_row_count_contract_guards_the_input_path(model) -> None:
    """A row-eating input transform refuses instead of mislabeling rows."""

    from torchlens._errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.extract_dataset(
            model,
            _stimuli(4),
            ["conv"],
            batch_size=4,
            input_transform=lambda batch: batch[:1],
            progress=False,
        )
    assert excinfo.value.fields["code"] == "extraction_input_transform_row_mismatch"


def test_pil_iterable_with_item_level_authority_transform(model, tmp_path) -> None:
    """The tv-shaped path: raw PIL images + the authority's own transform."""

    pytest.importorskip("PIL")
    import numpy as np
    from PIL import Image
    from torchvision.models import ResNet18_Weights

    resolution = pp.resolve(ResNet18_Weights.IMAGENET1K_V1)
    torch.manual_seed(2)
    images = [
        Image.fromarray(np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)) for _ in range(2)
    ]
    from torchvision.models import resnet18

    net = resnet18(weights=None).eval()
    out_dir = tmp_path / "pil"
    tl.extract_dataset(
        net,
        images,
        ["avgpool"],
        batch_size=2,
        output_dir=out_dir,
        input_transform=resolution,
        stimulus_ids=["a", "b"],
        progress=False,
    )
    loaded = tl.load_extraction(out_dir)
    assert loaded.activations["avgpool"].shape[0] == 2
    assert loaded.input_preprocessing["verdict"] == "verified"
