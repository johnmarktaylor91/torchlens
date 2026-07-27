"""Family-template and metadata demarcation tests for crawler Slice D."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from menagerie.crawler.family_templates import (
    FamilyTemplateError,
    instantiate_size_variant,
    mechanical_variant_parameter_input_line,
    validate_size_variant,
)
from menagerie.crawler.metadata import MetadataValidationError, validate_external_metadata
from menagerie.crawler.tests.conftest import make_model


def test_size_variant_changes_only_measured_parameter_input_line() -> None:
    """Variant prose remains byte-identical to the accepted representative."""

    representative = make_model(accepted=True)
    variant = instantiate_size_variant(
        representative,
        representative_model_id="m_example",
        variant_parameter_input_line="11700000 parameters; input [1, 3, 224, 224]",
    )
    for field in ("tagline", "description", "key_contribution", "voice_version"):
        assert variant[field].encode() == representative["website"][field].encode()
    assert variant["variant_parameter_input_line"].startswith("11700000")


def test_size_variant_with_changed_prose_requires_revet() -> None:
    """Any non-measurement prose change is refused."""

    representative = make_model(accepted=True)
    variant = instantiate_size_variant(
        representative,
        representative_model_id="m_example",
        variant_parameter_input_line="12000000 parameters; input [1, 3, 224, 224]",
    )
    variant["description"] += " Changed."
    with pytest.raises(FamilyTemplateError, match="requires a new accuracy vet"):
        validate_size_variant(representative, variant, "m_example")


def test_variant_line_is_mechanical_and_closed_to_prose() -> None:
    """Only exact integer count and input-contract shape facts can enter the line."""

    model = make_model(accepted=True)
    line = mechanical_variant_parameter_input_line(2, model["input_contract"])
    assert line == "2 parameters; input [1, 3, 8, 8]"
    with pytest.raises(FamilyTemplateError, match="must match"):
        instantiate_size_variant(
            model,
            representative_model_id="m_example",
            variant_parameter_input_line=f"{line}; best compact model",
        )


def test_representative_revision_supersession_stales_variant() -> None:
    """A template hash binds the exact accepted representative revision."""

    representative = make_model(accepted=True)
    variant_website = instantiate_size_variant(
        representative,
        representative_model_id="m_example",
        variant_parameter_input_line="2 parameters; input [1, 3, 8, 8]",
    )
    superseding = deepcopy(representative)
    superseding["record_revision"] = "sha256:" + "b" * 64
    with pytest.raises(FamilyTemplateError, match="stale or mismatched"):
        validate_size_variant(superseding, variant_website, "m_example")


def test_size_variant_must_measure_a_different_constructed_size() -> None:
    """A cloned representative recipe cannot masquerade as a genuine size variant."""

    representative = make_model(accepted=True)
    variant = deepcopy(representative)
    variant["website"] = instantiate_size_variant(
        representative,
        representative_model_id="m_example",
        variant_parameter_input_line="2 parameters; input [1, 3, 8, 8]",
    )
    with pytest.raises(FamilyTemplateError, match="must differ"):
        validate_size_variant(
            representative,
            variant,
            "m_example",
            parameter_count_total=2,
            input_contract=variant["input_contract"],
        )


def test_variant_parity_is_schema_derived_for_authored_mode_leaves() -> None:
    """Variants inherit authored mode expectations while reducer mode results may differ."""

    representative = make_model(accepted=True)
    variant = deepcopy(representative)
    variant["website"] = instantiate_size_variant(
        representative,
        representative_model_id="m_example",
        variant_parameter_input_line="2 parameters; input [1, 3, 8, 8]",
    )
    variant["modes"]["train_eval_divergence"] = "structural"
    variant["modes"]["divergence_evidence"] = "reducer-derived signature difference"
    validate_size_variant(representative, variant, "m_example")

    changed_expectation = deepcopy(variant)
    changed_expectation["modes"]["meaningful_modes"] = ["train"]
    with pytest.raises(FamilyTemplateError, match="modes.meaningful_modes"):
        validate_size_variant(representative, changed_expectation, "m_example")

    changed_external_claim = deepcopy(variant)
    changed_external_claim["external_metadata"]["modes"]["train_eval_divergence"] = "structural"
    with pytest.raises(FamilyTemplateError, match="external_metadata.modes"):
        validate_size_variant(representative, changed_external_claim, "m_example")


def _external_metadata() -> dict[str, Any]:
    """Return complete mandatory external metadata.

    Returns
    -------
    dict[str, Any]
        Accepted fixture metadata.
    """

    return deepcopy(make_model(accepted=True)["external_metadata"])


@pytest.mark.parametrize(
    "field",
    [
        "modality",
        "architecture_class",
        "domain",
        "task",
        "paradigm",
        "lineage",
        "tags",
        "keywords",
        "venue",
        "year",
        "country",
        "authors",
        "institution",
        "citation",
        "license",
        "key_contribution",
        "description",
    ],
)
def test_missing_mandatory_external_field_blocks(field: str) -> None:
    """Every source-read checklist field is mandatory.

    Parameters
    ----------
    field:
        Mandatory field removed for this case.
    """

    metadata = _external_metadata()
    del metadata[field]
    with pytest.raises(MetadataValidationError, match="missing mandatory"):
        validate_external_metadata(metadata)


def test_missing_torchlens_derivable_fields_does_not_block() -> None:
    """Structural observations remain optional at the metadata gate."""

    report = validate_external_metadata(_external_metadata())
    assert not report.derivable_fields_present
