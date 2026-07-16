"""Deterministic family metadata reuse for measured size variants."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping, Optional

from menagerie.crawler.authority import DependencyState, FamilyAuthority
from menagerie.crawler.constants import MODEL_SCHEMA_VERSION_V3
from menagerie.crawler.identity import canonical_json_bytes, stable_hash
from menagerie.crawler.metadata import (
    authored_model_leaves,
    proposal_fact_block,
    validate_authored_facts_for_write,
)
from menagerie.crawler.models import JsonObject

VETTED_TEXT_FIELDS = ("tagline", "description", "key_contribution", "voice_version")
VARIANT_MECHANICAL_AUTHOR_PATHS = frozenset(
    {
        "website.kind",
        "website.template_hash",
        "website.template_source_model_id",
        "website.variant_parameter_input_line",
        "implementation.library_recipe.kwargs",
        "implementation.library_recipe.symbol",
        "implementation.recipe_revision",
    }
)
_LEGACY_V2_VARIANT_PATHS = frozenset({"identity.canonical_name"})
VARIANT_PARAMETER_INPUT_PATTERN = re.compile(
    r"^(?:0|[1-9][0-9]*) parameters; input \[(?:0|[1-9][0-9]*)"
    r"(?:, (?:0|[1-9][0-9]*))*\]$"
)
VARIANT_SELECTOR_KEYS = ("variant", "model_name", "arch", "architecture", "size")


class FamilyTemplateError(ValueError):
    """Raised when a size variant changes inherited family metadata."""


def build_size_variant_derivation(
    representative: Mapping[str, Any],
    *,
    representative_model_id: str,
    variant_token: str,
) -> JsonObject:
    """Build the closed reducer-verifiable recipe specialization proof.

    Parameters
    ----------
    representative:
        Exact accepted representative model revision.
    representative_model_id:
        Trusted stable ID of the representative.
    variant_token:
        Exact trusted intake variant token.

    Returns
    -------
    dict[str, Any]
        Exact source revision/recipe and the sole permitted specialization delta.

    Raises
    ------
    FamilyTemplateError
        If the representative cannot be specialized by the closed rule.
    """

    revision = representative.get("record_revision")
    implementation = representative.get("implementation")
    if not isinstance(revision, str) or not revision.strip():
        raise FamilyTemplateError("family representative has no exact record revision")
    if representative.get("stable_id") != representative_model_id:
        raise FamilyTemplateError("family representative stable ID is mismatched")
    if not isinstance(variant_token, str) or not variant_token.strip():
        raise FamilyTemplateError("trusted intake variant token must be non-empty")
    if (
        not isinstance(implementation, Mapping)
        or implementation.get("recipe_type") != "declarative-library"
    ):
        raise FamilyTemplateError("family representative has no declarative library recipe")
    recipe = implementation.get("library_recipe")
    if not isinstance(recipe, Mapping):
        raise FamilyTemplateError("family representative library recipe is incomplete")
    kwargs = recipe.get("kwargs")
    if not isinstance(kwargs, Mapping):
        raise FamilyTemplateError("family representative recipe kwargs are incomplete")
    selector_keys = tuple(key for key in VARIANT_SELECTOR_KEYS if key in kwargs)
    if len(selector_keys) == 1:
        selector_key = selector_keys[0]
        selector_rule: JsonObject = {"kind": "kwarg", "key": selector_key}
        recipe_path = ["library_recipe", "kwargs", selector_key]
        previous_value = deepcopy(kwargs[selector_key])
    elif selector_keys:
        raise FamilyTemplateError("family recipe has ambiguous variant selector kwargs")
    elif variant_token.isidentifier():
        selector_rule = {"kind": "symbol", "key": None}
        recipe_path = ["library_recipe", "symbol"]
        previous_value = deepcopy(recipe.get("symbol"))
    else:
        raise FamilyTemplateError(
            "family recipe has no closed variant selector; full author fallback is required"
        )
    return {
        "variant_token": variant_token,
        "template_source_model_id": representative_model_id,
        "template_source_revision": revision,
        "representative_recipe_revision": implementation.get("recipe_revision"),
        "representative_library_recipe": deepcopy(dict(recipe)),
        "selector_rule": selector_rule,
        "allowed_recipe_delta": {
            "path": recipe_path,
            "previous_value": previous_value,
            "new_value": variant_token,
        },
        "allowed_input_delta": "unchanged",
    }


def specialize_size_variant_recipe(
    representative: Mapping[str, Any],
    *,
    representative_model_id: str,
    variant_token: str,
) -> tuple[JsonObject, JsonObject, JsonObject]:
    """Derive the exact implementation and input contract for one size variant.

    Parameters
    ----------
    representative:
        Exact accepted representative model revision.
    representative_model_id:
        Trusted stable ID of the representative.
    variant_token:
        Exact trusted intake token selecting the sibling recipe.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
        Specialized implementation, unchanged input contract, and derivation proof.

    Raises
    ------
    FamilyTemplateError
        If the representative recipe or input contract is incomplete.
    """

    derivation = build_size_variant_derivation(
        representative,
        representative_model_id=representative_model_id,
        variant_token=variant_token,
    )
    implementation = representative.get("implementation")
    input_contract = representative.get("input_contract")
    if not isinstance(implementation, Mapping) or not isinstance(input_contract, Mapping):
        raise FamilyTemplateError("family representative recipe/input facts are incomplete")
    specialized = deepcopy(dict(implementation))
    recipe = specialized.get("library_recipe")
    if not isinstance(recipe, dict):
        raise FamilyTemplateError("family representative library recipe is incomplete")
    selector = derivation["selector_rule"]
    if selector["kind"] == "kwarg":
        kwargs = recipe.get("kwargs")
        if not isinstance(kwargs, dict):
            raise FamilyTemplateError("family representative recipe kwargs are incomplete")
        kwargs[str(selector["key"])] = variant_token
    else:
        recipe["symbol"] = variant_token
    return specialized, deepcopy(dict(input_contract)), derivation


def validate_size_variant_derivation(
    representative: Mapping[str, Any],
    variant: Mapping[str, Any],
    representative_model_id: str,
    *,
    trusted_variant_token: str,
) -> None:
    """Verify that a variant recipe is the sole mechanical representative delta.

    Parameters
    ----------
    representative, variant:
        Exact representative and proposed canonical variant records.
    representative_model_id:
        Trusted representative stable ID.
    trusted_variant_token:
        Exact variant token from trusted intake.

    Raises
    ------
    FamilyTemplateError
        If the proof, recipe, input, or trusted family binding differs.
    """

    expected_implementation, expected_input, expected_derivation = specialize_size_variant_recipe(
        representative,
        representative_model_id=representative_model_id,
        variant_token=trusted_variant_token,
    )
    if canonical_json_bytes(variant.get("family_variant_derivation")) != canonical_json_bytes(
        expected_derivation
    ):
        raise FamilyTemplateError("variant derivation payload is stale or mismatched")
    identity = variant.get("identity")
    if not isinstance(identity, Mapping) or identity.get("variant") != trusted_variant_token:
        raise FamilyTemplateError("variant identity does not match trusted intake token")
    variant_implementation = variant.get("implementation")
    if not isinstance(variant_implementation, Mapping):
        raise FamilyTemplateError("variant implementation is incomplete")
    expected_implementation["recipe_revision"] = variant_implementation.get("recipe_revision")
    if canonical_json_bytes(variant_implementation) != canonical_json_bytes(
        expected_implementation
    ):
        raise FamilyTemplateError(
            "variant implementation is not the mechanical representative specialization"
        )
    if canonical_json_bytes(variant.get("input_contract")) != canonical_json_bytes(expected_input):
        raise FamilyTemplateError("variant input contract changed outside the closed derivation")


def family_representative_is_usable(
    representative: Mapping[str, Any] | None, representative_model_id: str
) -> bool:
    """Return whether a representative can currently authorize family variants.

    Parameters
    ----------
    representative:
        Candidate current representative record.
    representative_model_id:
        Stable ID named by the dependent variant.

    Returns
    -------
    bool
        True only for accepted, executed, current family authority.
    """

    if not isinstance(representative, Mapping):
        return False
    identity = representative.get("identity")
    website = representative.get("website")
    fidelity = representative.get("fidelity", {})
    return bool(
        representative.get("stable_id") == representative_model_id
        and representative.get("authored_metadata_state") == "accepted"
        and representative.get("status", {}).get("kind") == "runs"
        and representative.get("accuracy_gate", {}).get("current") is True
        and representative.get("accuracy_gate", {}).get("verdict") == "accurate"
        and representative.get("execution", {}).get("current") is True
        and (not fidelity.get("required") or fidelity.get("current") is True)
        and isinstance(identity, Mapping)
        and identity.get("variant_scope") == "family"
        and identity.get("family_representative_id") == representative_model_id
        and isinstance(website, Mapping)
        and website.get("kind") == "family-representative"
    )


def family_variant_currency_error(
    variant: Mapping[str, Any], current: Mapping[str, Mapping[str, Any]]
) -> Optional[str]:
    """Return why a structural family variant is stale, if it is stale.

    Parameters
    ----------
    variant:
        Candidate current variant record.
    current:
        Independently selected current records keyed by stable ID.

    Returns
    -------
    str | None
        Stable-family reason, or ``None`` for a non-variant/current variant.
    """

    if variant.get("schema_version") == MODEL_SCHEMA_VERSION_V3:
        authority_block = variant.get("family_authority")
        if not isinstance(authority_block, Mapping):
            return "missing trusted family authority"
        if authority_block.get("binding_state") == "ordinary":
            return None
        if authority_block.get("binding_state") != "variant":
            return "invalid trusted family binding state"
        representative_id = authority_block.get("representative_stable_id")
    else:
        website = variant.get("website")
        if not isinstance(website, Mapping) or website.get("kind") != "size-variant-template":
            return None
        representative_id = website.get("template_source_model_id")
    if not isinstance(representative_id, str) or not representative_id:
        return "missing representative binding"
    representative = current.get(representative_id)
    if not isinstance(representative, Mapping) or not family_representative_is_usable(
        representative, representative_id
    ):
        return "representative is not current accepted usable authority"
    derivation = variant.get("family_variant_derivation")
    if not isinstance(derivation, Mapping):
        return "missing derivation payload"
    try:
        if variant.get("schema_version") == MODEL_SCHEMA_VERSION_V3:
            authority = _family_authority_from_model(variant)
            validate_family_variant_authority(representative, variant, authority)
        validate_size_variant(
            representative,
            variant,
            representative_id,
            parameter_count_total=variant.get("observed", {}).get("parameter_count_total"),
            input_contract=variant.get("input_contract", {}),
        )
        validate_size_variant_derivation(
            representative,
            variant,
            representative_id,
            trusted_variant_token=str(derivation.get("variant_token", "")),
        )
    except FamilyTemplateError as exc:
        return str(exc)
    return None


def validate_family_variant_authority(
    representative: Mapping[str, Any],
    variant: Mapping[str, Any],
    authority: FamilyAuthority,
    *,
    representative_gate_item: Mapping[str, Any] | None = None,
) -> None:
    """Validate a v3 variant against trusted family authority and gate parity.

    Parameters
    ----------
    representative, variant:
        Exact representative and derived variant model revisions.
    authority:
        Frozen trusted-intake family binding supplied by the authority kernel.
    representative_gate_item:
        Optional exact accepted representative checker item. When supplied, the
        ordinary authored write gate is rerun over representative facts and the
        variant must bind the same representative vet identity.

    Raises
    ------
    FamilyTemplateError
        If trusted binding, representative revision, author-gated parity, or gate
        parity differs.
    """

    representative_id = authority.representative_stable_id
    if not isinstance(representative_id, str) or representative_id in {
        DependencyState.NOT_APPLICABLE.value,
        DependencyState.PENDING_UNTRUSTED.value,
    }:
        raise FamilyTemplateError("variant family authority lacks a representative stable ID")
    if authority.stable_id != variant.get("stable_id"):
        raise FamilyTemplateError("family authority stable ID does not match the variant")
    if representative.get("stable_id") != representative_id:
        raise FamilyTemplateError("family authority representative stable ID is mismatched")
    if authority.representative_revision != representative.get("record_revision"):
        raise FamilyTemplateError("family authority representative revision is stale")
    if authority.template_source_revision != representative.get("record_revision"):
        raise FamilyTemplateError("family authority template source revision is stale")
    derivation = variant.get("family_variant_derivation")
    if not isinstance(derivation, Mapping):
        raise FamilyTemplateError("variant lacks its reducer-derived family derivation")
    if derivation.get("variant_token") != authority.variant_token:
        raise FamilyTemplateError("family authority variant token is mismatched")
    if derivation.get("template_source_revision") != authority.template_source_revision:
        raise FamilyTemplateError("family derivation does not bind the authority template revision")
    _validate_inherited_metadata(representative, variant)
    if representative_gate_item is None:
        return
    if representative_gate_item.get("stable_id") != representative_id:
        raise FamilyTemplateError("variant gate parity did not use the representative checker item")
    representative_accuracy = representative.get("accuracy_gate")
    variant_accuracy = variant.get("accuracy_gate")
    if not isinstance(representative_accuracy, Mapping) or not isinstance(
        variant_accuracy, Mapping
    ):
        raise FamilyTemplateError("family gate parity requires complete accuracy-gate blocks")
    expected_vet = representative_accuracy.get("vet_identity")
    if representative_gate_item.get("vet_identity") != expected_vet:
        raise FamilyTemplateError("representative checker item vet identity is stale")
    if variant_accuracy.get("vet_identity") != expected_vet:
        raise FamilyTemplateError("variant does not inherit the representative vet identity")
    try:
        validate_authored_facts_for_write(
            proposal_fact_block(representative), representative_gate_item
        )
    except ValueError as exc:
        raise FamilyTemplateError(f"representative authored write gate failed: {exc}") from exc


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
        payload["inherited_author_gated_leaves"] = _inherited_author_gated_leaves(representative)
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
    representative_leaves = _inherited_author_gated_leaves(representative)
    variant_leaves = _inherited_author_gated_leaves(variant)
    if canonical_json_bytes(representative_leaves) == canonical_json_bytes(variant_leaves):
        return
    changed = sorted(
        path
        for path in set(representative_leaves) | set(variant_leaves)
        if canonical_json_bytes(representative_leaves.get(path))
        != canonical_json_bytes(variant_leaves.get(path))
    )
    raise FamilyTemplateError(f"variant differs in inherited author-gated leaves: {changed}")


def _inherited_author_gated_leaves(model: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return schema-derived family leaves after ruled mechanical deltas."""

    try:
        leaves = authored_model_leaves(model)
    except ValueError as exc:
        raise FamilyTemplateError(f"cannot derive authored family ownership: {exc}") from exc
    excluded = VARIANT_MECHANICAL_AUTHOR_PATHS
    if model.get("schema_version") != MODEL_SCHEMA_VERSION_V3:
        # The unwired v2 hub still derives a display name from trusted intake. V2 is
        # legacy-untrusted under the frozen contract; the v3 authority path below
        # intentionally has no such exception.
        excluded = excluded | _LEGACY_V2_VARIANT_PATHS
    return {path: value for path, value in leaves.items() if path not in excluded}


def _family_authority_from_model(model: Mapping[str, Any]) -> FamilyAuthority:
    """Parse the frozen family authority value already carried by a v3 model."""

    value = model.get("family_authority")
    if not isinstance(value, Mapping):
        raise FamilyTemplateError("v3 variant is missing family authority")
    required = (
        "representative_stable_id",
        "representative_revision",
        "representative_gate_id",
        "representative_proposal_id",
        "variant_token",
        "template_source_revision",
        "derivation_rule_identity",
    )
    if any(not isinstance(value.get(field), str) for field in required):
        raise FamilyTemplateError("v3 variant family authority is incomplete")
    return FamilyAuthority(
        stable_id=str(model.get("stable_id")),
        representative_stable_id=str(value["representative_stable_id"]),
        representative_revision=str(value["representative_revision"]),
        representative_gate_id=str(value["representative_gate_id"]),
        representative_proposal_id=str(value["representative_proposal_id"]),
        variant_token=str(value["variant_token"]),
        template_source_revision=str(value["template_source_revision"]),
        derivation_rule_identity=str(value["derivation_rule_identity"]),
    )


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
