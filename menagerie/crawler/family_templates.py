"""Deterministic family metadata reuse for measured size variants."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping, Optional

from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.models import JsonObject

VETTED_TEXT_FIELDS = ("tagline", "description", "key_contribution", "voice_version")
INHERITED_METADATA_FIELDS = (
    "taxonomy",
    "external_metadata",
    "people_and_origin",
    "dates",
    "citation",
    "licenses",
    "source_resolution",
    "evidence",
)
VARIANT_PARAMETER_INPUT_PATTERN = re.compile(
    r"^(?:0|[1-9][0-9]*) parameters; input \[(?:0|[1-9][0-9]*)"
    r"(?:, (?:0|[1-9][0-9]*))*\]$"
)


class FamilyTemplateError(ValueError):
    """Raised when a size variant changes inherited family metadata."""


def mechanical_variant_parameter_input_line(
    parameter_count_total: Any,
    input_contract: Mapping[str, Any],
) -> str:
    """Build the only variant-specific website line from observed mechanical facts.

    Parameters
    ----------
    parameter_count_total:
        Parameter count reported by the isolated worker after construction.
    input_contract:
        Accepted input contract for the executed recipe.

    Returns
    -------
    str
        Strict ``N parameters; input [D, ...]`` line.

    Raises
    ------
    FamilyTemplateError
        If the count or single tensor input shape is not mechanically usable.
    """

    if (
        isinstance(parameter_count_total, bool)
        or not isinstance(parameter_count_total, int)
        or parameter_count_total < 0
    ):
        raise FamilyTemplateError("variant parameter count must be a non-negative integer")
    tensor_inputs: list[Mapping[str, Any]] = []
    for collection in ("args", "kwargs"):
        values = input_contract.get(collection)
        if not isinstance(values, list):
            raise FamilyTemplateError("variant input contract must contain args and kwargs arrays")
        tensor_inputs.extend(
            value
            for value in values
            if isinstance(value, Mapping) and value.get("kind") == "tensor"
        )
    if len(tensor_inputs) != 1:
        raise FamilyTemplateError("variant template requires exactly one tensor input")
    shape = tensor_inputs[0].get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape
        )
    ):
        raise FamilyTemplateError(
            "variant input shape must contain non-negative integer dimensions"
        )
    line = f"{parameter_count_total} parameters; input [{', '.join(map(str, shape))}]"
    _validate_variant_line(line)
    return line


def instantiate_size_variant(
    representative: Mapping[str, Any],
    *,
    representative_model_id: str,
    variant_parameter_input_line: str,
) -> JsonObject:
    """Instantiate family text with only a new measured size/input line.

    ``representative`` should be the complete accepted canonical record in production.
    Website-only mappings remain accepted for compatibility with the original helper,
    but only complete records bind template currency to a record revision and all
    inherited metadata roots.

    Parameters
    ----------
    representative:
        Accurate-gated representative record or its website block.
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

    representative_website = _representative_website(representative)
    _validate_representative(representative_website)
    if not representative_model_id.strip():
        raise FamilyTemplateError("representative ID must be non-empty")
    _validate_variant_line(variant_parameter_input_line)
    template_hash = stable_hash(_template_identity_payload(representative, representative_model_id))
    result = deepcopy(dict(representative_website))
    result.update(
        {
            "kind": "size-variant-template",
            "template_source_model_id": representative_model_id,
            "variant_parameter_input_line": variant_parameter_input_line,
            "template_hash": template_hash,
        }
    )
    validate_size_variant(representative, result, representative_model_id)
    return result


def validate_size_variant(
    representative: Mapping[str, Any],
    variant: Mapping[str, Any],
    representative_model_id: str,
    *,
    parameter_count_total: Optional[int] = None,
    input_contract: Optional[Mapping[str, Any]] = None,
) -> None:
    """Prove that a variant changed no inherited metadata outside the measured line.

    Parameters
    ----------
    representative:
        Accepted representative record or website block.
    variant:
        Proposed variant record or website block.
    representative_model_id:
        Stable ID of the representative.
    parameter_count_total, input_contract:
        Optional real execution facts used to prove the measured line mechanically.

    Raises
    ------
    FamilyTemplateError
        If family metadata, grounding, revision currency, or measured text differs.
    """

    representative_website = _representative_website(representative)
    variant_website = _variant_website(variant)
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
    if not isinstance(line, str):
        raise FamilyTemplateError("variant requires a measured parameter/input line")
    _validate_variant_line(line)
    if (parameter_count_total is None) != (input_contract is None):
        raise FamilyTemplateError("mechanical line validation requires count and input contract")
    if parameter_count_total is not None and input_contract is not None:
        expected_line = mechanical_variant_parameter_input_line(
            parameter_count_total, input_contract
        )
        if line != expected_line:
            raise FamilyTemplateError("variant parameter/input line does not match execution facts")
        if "website" in representative:
            representative_count = representative.get("observed", {}).get("parameter_count_total")
            if representative_count == parameter_count_total:
                raise FamilyTemplateError(
                    "size variant parameter count must differ from its representative"
                )
    expected_hash = stable_hash(_template_identity_payload(representative, representative_model_id))
    if variant_website.get("template_hash") != expected_hash:
        raise FamilyTemplateError("variant template hash is stale or mismatched")
    if "website" in representative and "website" in variant:
        _validate_inherited_metadata(representative, variant)


def _template_identity_payload(
    representative: Mapping[str, Any], representative_model_id: str
) -> JsonObject:
    """Return the exact representative bytes bound by a variant template hash.

    Parameters
    ----------
    representative:
        Accepted representative record or website block.
    representative_model_id:
        Stable representative ID.

    Returns
    -------
    dict[str, Any]
        Canonical template identity payload.
    """

    website = _representative_website(representative)
    payload: JsonObject = {
        "family_grounding_id": website["family_grounding_id"],
        "vetted_text": {field: website[field] for field in VETTED_TEXT_FIELDS},
        "template_source_model_id": representative_model_id,
    }
    if "website" in representative:
        revision = representative.get("record_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise FamilyTemplateError("representative record revision must be non-empty")
        payload["template_source_revision"] = revision
        payload["inherited_metadata"] = {
            field: representative.get(field) for field in INHERITED_METADATA_FIELDS
        }
    return payload


def _validate_inherited_metadata(
    representative: Mapping[str, Any], variant: Mapping[str, Any]
) -> None:
    """Compare every inherited family-level canonical root byte-for-byte.

    Parameters
    ----------
    representative, variant:
        Complete canonical family records.

    Raises
    ------
    FamilyTemplateError
        If either record is incomplete or any inherited root differs.
    """

    if "website" not in representative or "website" not in variant:
        raise FamilyTemplateError("family template validation requires two complete records")
    for field in INHERITED_METADATA_FIELDS:
        if canonical_json_bytes(representative.get(field)) != canonical_json_bytes(
            variant.get(field)
        ):
            raise FamilyTemplateError(f"variant differs in inherited metadata field {field!r}")


def _representative_website(representative: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract a representative website block from a record or direct mapping.

    Parameters
    ----------
    representative:
        Complete record or website mapping.

    Returns
    -------
    Mapping[str, Any]
        Representative website block.
    """

    website = representative.get("website") if "website" in representative else representative
    if not isinstance(website, Mapping):
        raise FamilyTemplateError("representative website block is missing")
    return website


def _variant_website(variant: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract a variant website block from a record or direct mapping.

    Parameters
    ----------
    variant:
        Complete record or website mapping.

    Returns
    -------
    Mapping[str, Any]
        Variant website block.
    """

    website = variant.get("website") if "website" in variant else variant
    if not isinstance(website, Mapping):
        raise FamilyTemplateError("variant website block is missing")
    return website


def _validate_variant_line(line: str) -> None:
    """Require the closed non-prose parameter/input line grammar.

    Parameters
    ----------
    line:
        Candidate website line.

    Raises
    ------
    FamilyTemplateError
        If the line is not in the exact closed grammar.
    """

    if VARIANT_PARAMETER_INPUT_PATTERN.fullmatch(line) is None:
        raise FamilyTemplateError(
            "variant line must match 'N parameters; input [D, ...]' with integer facts"
        )


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
