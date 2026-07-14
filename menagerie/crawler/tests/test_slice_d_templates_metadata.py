"""Family-template and metadata demarcation tests for crawler Slice D."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from menagerie.crawler.family_templates import (
    FamilyTemplateError,
    instantiate_size_variant,
    validate_size_variant,
)
from menagerie.crawler.metadata import MetadataValidationError, validate_external_metadata
from menagerie.crawler.tests.conftest import make_model


def test_size_variant_changes_only_measured_parameter_input_line() -> None:
    """Variant prose remains byte-identical to the accepted representative."""

    representative = make_model(accepted=True)["website"]
    variant = instantiate_size_variant(
        representative,
        representative_model_id="m_example",
        variant_parameter_input_line="11.7M parameters; input [1, 3, 224, 224].",
    )
    for field in ("tagline", "description", "key_contribution", "voice_version"):
        assert variant[field].encode() == representative[field].encode()
    assert variant["variant_parameter_input_line"].startswith("11.7M")


def test_size_variant_with_changed_prose_requires_revet() -> None:
    """Any non-measurement prose change is refused."""

    representative = make_model(accepted=True)["website"]
    variant = instantiate_size_variant(
        representative,
        representative_model_id="m_example",
        variant_parameter_input_line="12M parameters; input [1, 3, 224, 224].",
    )
    variant["description"] += " Changed."
    with pytest.raises(FamilyTemplateError, match="requires a new accuracy vet"):
        validate_size_variant(representative, variant, "m_example")


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
