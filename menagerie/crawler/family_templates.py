"""Deterministic family prose reuse for measured size variants."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from menagerie.crawler.identity import stable_hash
from menagerie.crawler.models import JsonObject

VETTED_TEXT_FIELDS = ("tagline", "description", "key_contribution", "voice_version")


class FamilyTemplateError(ValueError):
    """Raised when a size variant changes vetted family prose."""


def instantiate_size_variant(
    representative_website: Mapping[str, Any],
    *,
    representative_model_id: str,
    variant_parameter_input_line: str,
) -> JsonObject:
    """Instantiate family text with only a new measured size/input line.

    Parameters
    ----------
    representative_website:
        Accurate-gated representative website block.
    representative_model_id:
        Stable ID of the accepted family representative.
    variant_parameter_input_line:
        Mechanically measured parameter-count/input-shape line.

    Returns
    -------
    dict[str, Any]
        Size-variant website block reusing all vetted prose byte-for-byte.

    Raises
    ------
    FamilyTemplateError
        If the representative or measured line is incomplete.
    """

    _validate_representative(representative_website)
    if not representative_model_id.strip() or not variant_parameter_input_line.strip():
        raise FamilyTemplateError("representative ID and measured variant line must be non-empty")
    family_text = {field: representative_website[field] for field in VETTED_TEXT_FIELDS}
    template_hash = stable_hash(
        {
            "family_grounding_id": representative_website["family_grounding_id"],
            "vetted_text": family_text,
            "template_source_model_id": representative_model_id,
        }
    )
    result = deepcopy(dict(representative_website))
    result.update(
        {
            "kind": "size-variant-template",
            "template_source_model_id": representative_model_id,
            "variant_parameter_input_line": variant_parameter_input_line,
            "template_hash": template_hash,
        }
    )
    validate_size_variant(representative_website, result, representative_model_id)
    return result


def validate_size_variant(
    representative_website: Mapping[str, Any],
    variant_website: Mapping[str, Any],
    representative_model_id: str,
) -> None:
    """Prove that a variant changed no vetted text outside the measured line.

    Parameters
    ----------
    representative_website:
        Accepted representative website block.
    variant_website:
        Proposed deterministic variant block.
    representative_model_id:
        Stable ID of the representative.

    Raises
    ------
    FamilyTemplateError
        If family prose, grounding, identity, or template hash differs.
    """

    _validate_representative(representative_website)
    for field in VETTED_TEXT_FIELDS:
        representative_bytes = str(representative_website.get(field)).encode("utf-8")
        variant_bytes = str(variant_website.get(field)).encode("utf-8")
        if variant_bytes != representative_bytes:
            raise FamilyTemplateError(
                f"variant differs in vetted field {field!r}; it requires a new accuracy vet"
            )
    if variant_website.get("family_grounding_id") != representative_website.get(
        "family_grounding_id"
    ):
        raise FamilyTemplateError("variant changed family grounding")
    if variant_website.get("kind") != "size-variant-template":
        raise FamilyTemplateError("variant must declare size-variant-template")
    if variant_website.get("template_source_model_id") != representative_model_id:
        raise FamilyTemplateError("variant does not reference its accepted representative")
    line = variant_website.get("variant_parameter_input_line")
    if not isinstance(line, str) or not line.strip():
        raise FamilyTemplateError("variant requires a measured parameter/input line")
    expected_hash = stable_hash(
        {
            "family_grounding_id": representative_website["family_grounding_id"],
            "vetted_text": {field: representative_website[field] for field in VETTED_TEXT_FIELDS},
            "template_source_model_id": representative_model_id,
        }
    )
    if variant_website.get("template_hash") != expected_hash:
        raise FamilyTemplateError("variant template hash is stale or mismatched")


def _validate_representative(website: Mapping[str, Any]) -> None:
    """Validate required accepted representative text.

    Parameters
    ----------
    website:
        Representative website block.

    Raises
    ------
    FamilyTemplateError
        If required text/grounding is absent or this is not a representative.
    """

    if website.get("kind") != "family-representative":
        raise FamilyTemplateError("template source must be a family representative")
    required = (*VETTED_TEXT_FIELDS, "family_grounding_id")
    for field in required:
        value = website.get(field)
        if not isinstance(value, str) or not value.strip():
            raise FamilyTemplateError(f"representative {field} must be non-empty")
