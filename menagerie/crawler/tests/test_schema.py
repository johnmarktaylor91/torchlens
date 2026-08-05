"""Executable crawler schema contract tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from menagerie.crawler.author_dispatch import (
    AuthorResultMalformedError,
    validate_author_result_mapping,
)
from menagerie.crawler.constants import FAILURE_REASON_CODES, TERMINAL_STATUS_CODES, FailureStage
from menagerie.crawler.constants import AUTHOR_PROPOSAL_SCHEMA_VERSION_V3
from menagerie.crawler.schema import (
    OWNERSHIP_ANNOTATED_SCHEMA_VERSIONS,
    REQUIRED_FIELD_PROJECTION_SPECS,
    SCHEMA_FILES,
    PayloadValidationError,
    RequiredFieldProjection,
    SchemaOwnershipError,
    SchemaOwner,
    author_gated_schema_paths,
    load_schema,
    owned_schema_leaves,
    owned_schema_leaves_from_schema,
    required_field_projection_spec,
    validate_payload,
    validate_required_field_projection_specs,
)
from menagerie.crawler.tests.conftest import (
    make_attempt,
    make_author_proposal,
    make_gate,
    make_model,
    make_operational_event,
    make_shutdown_interruption_event,
)

_RUNG8_ARCHIVE = Path(
    "/Users/jmt/.claude/research/torchlens/crawler-launch-sprint/"
    "rung-archive/rung8-19a72e4a-complete"
)


def _require_rung8_archive() -> Path:
    """Return the local frozen rung-8 archive, or skip when absent.

    Returns
    -------
    Path
        Existing rung-8 archive root.
    """

    if not _RUNG8_ARCHIVE.exists():
        pytest.skip("frozen rung-8 archive is not present on this host")
    return _RUNG8_ARCHIVE


def _load_rung8_json(*parts: str) -> dict[str, Any]:
    """Load one JSON object from the frozen rung-8 archive.

    Parameters
    ----------
    parts:
        Path components relative to the archive root.

    Returns
    -------
    dict[str, Any]
        Decoded JSON object.
    """

    path = _require_rung8_archive().joinpath(*parts)
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "payload",
    [
        make_model(accepted=True),
        make_model(accepted=False),
        make_attempt(),
        make_gate(),
        make_gate(["m_example"], gate_kind="fidelity", fidelity_identity="sha256:" + "a" * 64),
        make_author_proposal(),
        make_operational_event(),
        make_shutdown_interruption_event(),
    ],
)
def test_representative_records_validate(payload: dict[str, Any]) -> None:
    """Every representative full schema record validates.

    Parameters
    ----------
    payload:
        Full schema payload under test.
    """

    validate_payload(payload)


def test_sandbox_unavailability_uses_planned_policy_stage() -> None:
    """Sandbox absence adds a versioned reason without expanding public failure stages."""

    assert "sandbox-unavailable" not in {stage.value for stage in FailureStage}
    assert "failed:sandbox-unavailable" not in TERMINAL_STATUS_CODES
    assert "sandbox-unavailable-v1" in FAILURE_REASON_CODES["policy"]


def test_unknown_fields_are_rejected(valid_model: dict[str, Any]) -> None:
    """Unknown fields fail at both root and nested typed objects.

    Parameters
    ----------
    valid_model:
        Valid accepted model fixture.
    """

    root_unknown = deepcopy(valid_model)
    root_unknown["surprise"] = True
    nested_unknown = deepcopy(valid_model)
    nested_unknown["external_metadata"]["surprise"] = True
    with pytest.raises(PayloadValidationError):
        validate_payload(root_unknown)
    with pytest.raises(PayloadValidationError):
        validate_payload(nested_unknown)


def test_validate_payload_reports_all_independent_schema_violations(
    valid_model: dict[str, Any],
) -> None:
    """Schema diagnostics enumerate sibling violations in one refusal."""

    malformed = deepcopy(valid_model)
    del malformed["external_metadata"]["architecture_class"]
    del malformed["external_metadata"]["domain"]

    with pytest.raises(PayloadValidationError) as caught:
        validate_payload(malformed)

    diagnostic = str(caught.value)
    assert "validation failed with" in diagnostic
    assert "'architecture_class' is a required property" in diagnostic
    assert "'domain' is a required property" in diagnostic


def test_union_validation_error_names_undeclared_property() -> None:
    """A union rejection identifies the closest branch and unexpected property."""

    payload = {
        "schema_version": "menagerie.crawler.source-discovery.v1",
        "stable_id": "m-diagnostic",
        "work_id": "work-m-diagnostic",
        "arm": "FOUND",
        "payload": {
            "arm": "FOUND",
            "sources": [
                {
                    "source_id": "impl-main",
                    "kind": "forge-file",
                    "repo": "github.com/example/model",
                    "path": "model.py",
                    "ref": "main",
                    "requested_role": "implementation",
                    "basis": "Observed implementation entry point.",
                    "undeclared_observation": "This property has no contract home.",
                }
            ],
        },
    }

    with pytest.raises(PayloadValidationError) as caught:
        validate_payload(payload)

    diagnostic = str(caught.value)
    assert "payload.sources[0]" in diagnostic
    assert "oneOf[0]" in diagnostic
    assert "additionalProperties" in diagnostic
    assert "undeclared_observation" in diagnostic


def test_union_validation_error_names_wrong_enum_and_allowed_values() -> None:
    """A union rejection identifies an enum field and its closed allowed values."""

    payload = {
        "schema_version": "menagerie.crawler.source-discovery.v1",
        "stable_id": "m-diagnostic",
        "work_id": "work-m-diagnostic",
        "arm": "FOUND",
        "payload": {
            "arm": "FOUND",
            "sources": [
                {
                    "source_id": "impl-main",
                    "kind": "forge-file",
                    "repo": "github.com/example/model",
                    "path": "model.py",
                    "ref": "main",
                    "requested_role": "configuration",
                    "basis": "Observed implementation entry point.",
                }
            ],
        },
    }

    with pytest.raises(PayloadValidationError) as caught:
        validate_payload(payload)

    diagnostic = str(caught.value)
    assert "payload.sources[0].requested_role" in diagnostic
    assert "enum" in diagnostic
    assert "configuration" in diagnostic
    assert "implementation" in diagnostic
    assert "documentation" in diagnostic


def test_rung8_m9577_schema_replay_reports_latent_source_errors() -> None:
    """The frozen m9577 raw attempt surfaces repeated latent schema failures."""

    raw = _load_rung8_json(
        "work",
        "m9577",
        "author",
        "attempts",
        "attempt-002-d1b1978e6e87499b80bce5cac13c181c",
        "result.json",
    )
    proposal = raw["payload"]["proposal"]
    proposal["schema_version"] = AUTHOR_PROPOSAL_SCHEMA_VERSION_V3

    with pytest.raises(PayloadValidationError) as caught:
        validate_payload(proposal, AUTHOR_PROPOSAL_SCHEMA_VERSION_V3)

    diagnostic = str(caught.value)
    assert "validation failed with" in diagnostic
    assert "source_resolution.sources[0].mirror_digest" in diagnostic
    assert "source_resolution.sources[1].mirror_digest" in diagnostic
    assert "source_resolution.sources[10].mirror_digest" in diagnostic
    assert "implementation.builder_symbol" in diagnostic


@pytest.mark.parametrize(
    ("stable_id", "expected_claims"),
    [
        (
            "m9304",
            (
                "external_metadata.architecture_class",
                "external_metadata.authors",
                "taxonomy.novel_ops",
                "website",
            ),
        ),
        (
            "m9617",
            (
                "external_metadata.domain",
                "external_metadata.run_framework",
                "taxonomy.family",
                "taxonomy.tasks",
            ),
        ),
    ],
)
def test_rung8_author_result_replay_reports_all_proposal_categories(
    stable_id: str, expected_claims: tuple[str, ...]
) -> None:
    """Frozen author-result validation preserves proposal-level multi-error output.

    Parameters
    ----------
    stable_id:
        Archived model identifier.
    expected_claims:
        Representative claim categories that must appear in the same refusal.
    """

    archive = _require_rung8_archive()
    author_dir = archive / "work" / stable_id / "author"
    request = _load_rung8_json("work", stable_id, "author", "request.json")
    result = _load_rung8_json("work", stable_id, "author", "result.json")

    with pytest.raises(AuthorResultMalformedError) as caught:
        validate_author_result_mapping(result, request, cas_root=author_dir / "source-cas")

    diagnostic = str(caught.value)
    assert "ungrounded claim categories" in diagnostic
    for claim in expected_claims:
        assert claim in diagnostic


@pytest.mark.parametrize("value", [None, "adapter.py"])
def test_v3_input_contract_code_path_presence_is_rejected(value: object) -> None:
    """Null and string forms of the deleted v3 executable-path leaf both reject.

    Parameters
    ----------
    value:
        Legacy input-contract value whose presence must fail closed.
    """

    model = make_model(accepted=True)
    model["input_contract"]["code_path"] = value
    proposal = make_author_proposal()
    proposal["proposed_facts"]["input_contract"]["code_path"] = value
    with pytest.raises(PayloadValidationError):
        validate_payload(model)
    with pytest.raises(PayloadValidationError):
        validate_payload(proposal)


def test_missing_mandatory_fields_are_rejected(valid_model: dict[str, Any]) -> None:
    """Missing mandatory fields fail strict validation.

    Parameters
    ----------
    valid_model:
        Valid accepted model fixture.
    """

    malformed = deepcopy(valid_model)
    del malformed["external_metadata"]["architecture_class"]
    with pytest.raises(PayloadValidationError):
        validate_payload(malformed)


def test_authored_blocks_are_atomic(valid_model: dict[str, Any]) -> None:
    """Accepted metadata cannot be null and pending metadata cannot be populated.

    Parameters
    ----------
    valid_model:
        Valid accepted model fixture.
    """

    accepted_with_null = deepcopy(valid_model)
    accepted_with_null["website"] = None
    pending_with_text = deepcopy(valid_model)
    pending_with_text["authored_metadata_state"] = "pending"
    with pytest.raises(PayloadValidationError):
        validate_payload(accepted_with_null)
    with pytest.raises(PayloadValidationError):
        validate_payload(pending_with_text)


def test_metadata_gate_final_tail_size_is_enforced() -> None:
    """Metadata gates permit a final one-item tail but never an empty result."""

    validate_payload(make_gate(["m_tail"]))
    undersized = make_gate([])
    with pytest.raises(PayloadValidationError):
        validate_payload(undersized)


def test_forward_attempt_requires_mode() -> None:
    """A meaningful forward attempt cannot omit its runtime mode."""

    malformed = make_attempt(mode=None)
    with pytest.raises(PayloadValidationError):
        validate_payload(malformed)


def test_schema_properties_have_nonempty_descriptions() -> None:
    """Every declared schema property carries a non-empty self-documenting description.

    Returns
    -------
    None
        The assertion validates all crawler schemas.
    """

    def assert_descriptions(node: object, path: str = "$") -> None:
        """Recursively assert descriptions for every JSON Schema property.

        Parameters
        ----------
        node:
            JSON-compatible schema node.
        path:
            Human-readable node location for assertion failures.

        Returns
        -------
        None
            Raises when a property is undocumented.
        """

        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                for name, property_schema in properties.items():
                    assert isinstance(property_schema, dict)
                    assert property_schema.get("description", "").strip(), f"{path}.{name}"
                    assert_descriptions(property_schema, f"{path}.{name}")
            for key, value in node.items():
                if key not in {"properties", "description"}:
                    assert_descriptions(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                assert_descriptions(value, f"{path}[{index}]")

    schema_root = Path(__file__).parents[1] / "schemas"
    for schema_path in schema_root.glob("*.json"):
        assert_descriptions(json.loads(schema_path.read_text()), schema_path.name)


def test_all_bundled_schema_contracts_load() -> None:
    """Every registered v2/v3 and event schema is a valid Draft 2020-12 schema."""

    for schema_version in SCHEMA_FILES:
        assert load_schema(schema_version)["properties"]["schema_version"]


def test_v3_schema_leaf_ownership_is_complete_and_exhaustive() -> None:
    """Every normalized v3 leaf has exactly one schema-declared authority owner."""

    for schema_version in OWNERSHIP_ANNOTATED_SCHEMA_VERSIONS:
        owned = owned_schema_leaves(schema_version)
        assert owned
        assert len({leaf.path for leaf in owned}) == len(owned)
        assert author_gated_schema_paths(schema_version).issubset({leaf.path for leaf in owned})


def test_required_field_projection_specs_are_owner_explicit_and_schema_aligned() -> None:
    """Python-owned field projections retain every owner key and schema parity."""

    assert set(REQUIRED_FIELD_PROJECTION_SPECS) == set(RequiredFieldProjection)
    for projection in RequiredFieldProjection:
        spec = required_field_projection_spec(projection)
        assert set(spec.fields_by_owner) == set(SchemaOwner)
        assert set(spec.field_order) == {
            field.name for owner in SchemaOwner for field in spec.fields_for(owner)
        }
        assert spec.names_for(SchemaOwner.WORKER_OBSERVED) == tuple(
            field.name for field in spec.fields_for(SchemaOwner.WORKER_OBSERVED)
        )
        assert spec.names_for(SchemaOwner.PARENT_OBSERVED) == tuple(
            field.name for field in spec.fields_for(SchemaOwner.PARENT_OBSERVED)
        )
    validate_required_field_projection_specs()


def test_schema_leaf_ownership_rejects_a_new_unclassified_leaf() -> None:
    """A newly allowed nested leaf cannot ship without an explicit owner annotation."""

    schema = deepcopy(load_schema("menagerie.crawler.author-result.v3"))
    payload_schema = schema["$defs"]["blocked_payload"]
    payload_schema["properties"]["unclassified_canary"] = {
        "type": "string",
        "description": "Synthetic CI canary that deliberately has no authority owner.",
    }
    with pytest.raises(SchemaOwnershipError, match="missing"):
        owned_schema_leaves_from_schema(schema)


def test_split_skip_reasons_require_vague_text_and_sufficiency_gap(
    valid_model: dict[str, Any],
) -> None:
    """The three skip terminals retain the ruled evidence distinctions.

    Parameters
    ----------
    valid_model:
        Complete accepted model fixture.

    Returns
    -------
    None
        The assertion validates all three terminal skip paths.
    """

    skip_codes = {
        "skipped:insufficient-description",
        "skipped:no-description",
        "skipped:not-a-real-NN",
    }
    assert skip_codes.issubset(TERMINAL_STATUS_CODES)

    for code in skip_codes - {"skipped:insufficient-description"}:
        record = deepcopy(valid_model)
        record["status"]["kind"] = "skipped"
        record["status"]["code"] = code
        validate_payload(record)

    insufficient = deepcopy(valid_model)
    insufficient["status"]["kind"] = "skipped"
    insufficient["status"]["code"] = "skipped:insufficient-description"
    insufficient["source_resolution"]["rung"] = "R5_SKIP"
    insufficient["source_resolution"]["sufficiency_gap"] = (
        "concept described but no layer configs/dims/connectivity"
    )
    insufficient["evidence"]["excerpts"][0]["text"] = "A novel neural architecture for vision."
    insufficient["evidence"]["excerpts"][0]["disposition"] = "insufficient-for-faithful-reimpl"
    validate_payload(insufficient)

    missing_gap = deepcopy(insufficient)
    missing_gap["source_resolution"]["sufficiency_gap"] = None
    with pytest.raises(PayloadValidationError):
        validate_payload(missing_gap)

    missing_vague_excerpt = deepcopy(insufficient)
    missing_vague_excerpt["evidence"]["excerpts"][0]["disposition"] = "supporting"
    with pytest.raises(PayloadValidationError):
        validate_payload(missing_vague_excerpt)
